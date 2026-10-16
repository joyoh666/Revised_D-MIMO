"""Run a complete single-stream MRT transmit/receive experiment.

This entry point deliberately implements beamforming gain rather than spatial
multiplexing.  One common PDSCH stream ``s`` is generated, antenna ``i``
transmits ``w[i] * s``, and the UE equalizes the coherently combined stream
using the full-band reference symbol at the beginning of every slot.

Example for the currently configured two TX antennas::

    python -m runtime.mrt_transceiver \
        --channel 0.8+0.1j 0.3-0.4j \
        --frames 20

The run saves the transmitted symbols, equalized symbols, hard decisions,
summary metrics, and a constellation comparison under ``--output-dir``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import math
from multiprocessing import get_context
from pathlib import Path
from queue import Empty
import secrets
from typing import Sequence

import numpy as np

from config.radio_config import (
    BANDWIDTH,
    CLOCK_SOURCE,
    OTW_FORMAT,
    REFERENCE_SEQUENCE_SEED,
    RX_ANTENNA,
    RX_CHANNELS,
    RX_SUBDEV_SPEC,
    RX_USRP_DEVICE_ARGS,
    RX_TRAILING_TIME,
    Rx_gain,
    TIME_SOURCE,
    TX_ANTENNA,
    TX_CHANNELS,
    TX_SUBDEV_SPEC,
    TX_USRP_DEVICE_ARGS,
    Tx_gain,
    WAIT_TIME,
)
from runtime.mrt_transmitter import (
    apply_mrt_precoding,
    calculate_mrt_weights,
)
from runtime.receiver import USRPReceiver
from runtime.transmitter import USRPTransmitter
from USRP.build_frame import build_frame
from USRP.helper_functions import (
    generate_known_reference_sequence,
    get_system_params,
)
from USRP.modulate import modulate
from USRP.ofdm import ofdm_demodulate
from USRP.postprocessing import (
    estimate_and_correct_ffo,
    estimate_and_correct_rfo,
    synchronize_frame_timing,
)


DEFAULT_OUTPUT_DIR = "mrt_link_results"


@dataclass(frozen=True)
class MRTFrameBatch:
    """TX waveforms and the common PDSCH symbols needed at the UE."""

    waveforms: np.ndarray
    pdsch_symbols: np.ndarray
    pdsch_indices: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ss_td_with_cp: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class RecoveredFrame:
    """Demodulation output for one received radio frame."""

    frame_index: int
    raw_symbols: np.ndarray
    equalized_symbols: np.ndarray
    effective_channel: np.ndarray
    sync_index: int
    ffo_hz: np.ndarray
    rfo_hz: np.ndarray


def _common_pdsch_indices(
    tx_pdsch_indices: tuple[np.ndarray, ...],
    source_tx_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove the TX axis from ``build_frame``'s PDSCH index tuple."""
    if len(tx_pdsch_indices) != 5:
        raise ValueError("PDSCH indices must contain five axes")
    source_positions = tx_pdsch_indices[0] == source_tx_index
    indices = tuple(
        np.asarray(axis[source_positions], dtype=np.int64)
        for axis in tx_pdsch_indices[1:]
    )
    if not indices[0].size:
        raise ValueError("the source stream contains no PDSCH symbols")
    return indices


def build_single_stream_mrt_batch(
    params,
    known_ref_seq,
    channel_coefficients: Sequence[complex] | np.ndarray,
    *,
    num_frames: int,
    frame_seed: int,
) -> MRTFrameBatch:
    """Build common-PDSCH frames and apply one MRT weight per antenna.

    ``build_frame`` creates independent data for each TX by default.  Only the
    synchronization TX stream is selected here; that one stream is then
    copied through the MRT vector.  Consequently, every antenna carries the
    same PDSCH symbols before its beamforming coefficient is applied.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be greater than zero")
    if frame_seed < 0:
        raise ValueError("frame_seed must be non-negative")

    num_tx = int(params["num_tx_ant"])
    frame_length = int(params["frame_length"])
    source_tx_index = int(params["sync_tx_idx"])
    weights = calculate_mrt_weights(
        channel_coefficients,
        num_tx_antennas=num_tx,
    )
    rng = np.random.default_rng(frame_seed)
    waveforms = np.empty(
        (num_frames, num_tx, frame_length),
        dtype=np.complex64,
    )
    pdsch_symbols = None
    common_indices = None
    ss_td_with_cp = None

    for frame_index in range(num_frames):
        frame = build_frame(params, known_ref_seq, rng=rng)
        source_waveform = np.asarray(
            frame["waveform"][source_tx_index],
            dtype=np.complex64,
        )
        waveforms[frame_index] = apply_mrt_precoding(
            source_waveform,
            channel_coefficients,
            num_tx_antennas=num_tx,
        )

        frame_indices = _common_pdsch_indices(
            frame["pdsch_idx"],
            source_tx_index,
        )
        if common_indices is None:
            common_indices = frame_indices
            num_pdsch = int(frame_indices[0].size)
            pdsch_symbols = np.empty(
                (num_frames, num_pdsch),
                dtype=np.complex64,
            )
            ss_td_with_cp = np.asarray(
                frame["ss_td_with_cp"],
                dtype=np.complex64,
            )
        elif any(
            not np.array_equal(current, expected)
            for current, expected in zip(
                frame_indices,
                common_indices,
                strict=True,
            )
        ):
            raise ValueError("PDSCH positions changed between frames")

        source_grid = np.asarray(
            frame["resource_maps_tx"][source_tx_index],
            dtype=np.complex64,
        )
        pdsch_symbols[frame_index] = source_grid[common_indices]

    return MRTFrameBatch(
        waveforms=waveforms,
        pdsch_symbols=pdsch_symbols,
        pdsch_indices=common_indices,
        ss_td_with_cp=ss_td_with_cp,
        weights=weights,
    )


def synchronize_and_demodulate(
    received_samples: np.ndarray,
    params,
    ss_td_with_cp: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray | int]]:
    """Synchronize a capture and return its active-subcarrier resource grid."""
    samples = np.asarray(received_samples, dtype=np.complex64)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    if samples.ndim != 2:
        raise ValueError("received_samples must have shape (RX, sample)")
    if samples.shape[1] < int(params["frame_length"]):
        raise ValueError("received_samples is shorter than one radio frame")

    samples = samples.copy()
    samples -= np.mean(samples, axis=1, keepdims=True)
    frame, sync_index, timing_peaks = synchronize_frame_timing(
        samples,
        ss_td_with_cp,
        params,
    )
    frame, ffo_hz = estimate_and_correct_ffo(frame, params)
    frame, rfo_hz, rfo_hz_per_slot = estimate_and_correct_rfo(frame, params)
    resource_grid = ofdm_demodulate(frame, params)
    diagnostics = {
        "sync_index": sync_index,
        "timing_correlation_peak": timing_peaks,
        "ffo_hz": ffo_hz,
        "rfo_hz": rfo_hz,
        "rfo_hz_per_slot": rfo_hz_per_slot,
    }
    return resource_grid, diagnostics


def equalize_common_pdsch(
    received_grid: np.ndarray,
    params,
    known_ref_seq: np.ndarray,
    pdsch_indices: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    minimum_channel_magnitude: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the combined channel and ZF-equalize the common PDSCH.

    Symbol zero of every slot is a known full-band reference.  Because the
    complete source waveform is MRT-precoded, this reference experiences the
    same combined channel as the PDSCH stream.
    """
    grid = np.asarray(received_grid, dtype=np.complex64)
    expected_shape = (
        int(params["num_rx_ant"]),
        int(params["num_subframe_per_frame"]),
        int(params["num_slot_per_subframe"]),
        int(params["num_symbols_per_slot"]),
        int(params["N"]),
    )
    if grid.shape != expected_shape:
        raise ValueError(
            f"received_grid has shape {grid.shape}, expected {expected_shape}"
        )
    reference = np.asarray(known_ref_seq, dtype=np.complex64)
    if reference.shape != (int(params["N"]),):
        raise ValueError("known_ref_seq has the wrong number of subcarriers")
    if minimum_channel_magnitude <= 0:
        raise ValueError("minimum_channel_magnitude must be positive")

    effective_channel = grid[:, :, :, 0, :] / reference[None, None, None, :]
    sf_index, slot_index, symbol_index, subcarrier_index = pdsch_indices
    raw_symbols = grid[
        :, sf_index, slot_index, symbol_index, subcarrier_index
    ]
    pdsch_channel = effective_channel[
        :, sf_index, slot_index, subcarrier_index
    ]
    equalized = np.full_like(raw_symbols, np.nan + 1j * np.nan)
    valid = np.abs(pdsch_channel) >= minimum_channel_magnitude
    np.divide(
        raw_symbols,
        pdsch_channel,
        out=equalized,
        where=valid,
    )
    return raw_symbols, equalized, effective_channel


def recover_frame(
    received_samples: np.ndarray,
    params,
    known_ref_seq: np.ndarray,
    frame_batch: MRTFrameBatch,
    frame_index: int,
) -> RecoveredFrame:
    """Synchronize, demodulate, and equalize one frame."""
    grid, diagnostics = synchronize_and_demodulate(
        received_samples,
        params,
        frame_batch.ss_td_with_cp,
    )
    raw, equalized, effective_channel = equalize_common_pdsch(
        grid,
        params,
        known_ref_seq,
        frame_batch.pdsch_indices,
    )
    return RecoveredFrame(
        frame_index=frame_index,
        raw_symbols=raw,
        equalized_symbols=equalized,
        effective_channel=effective_channel,
        sync_index=int(diagnostics["sync_index"]),
        ffo_hz=np.asarray(diagnostics["ffo_hz"], dtype=np.float32),
        rfo_hz=np.asarray(diagnostics["rfo_hz"], dtype=np.float32),
    )


def _constellation_points(modulation_order: int) -> np.ndarray:
    bits_per_symbol = int(np.log2(modulation_order))
    symbol_indices = np.arange(modulation_order, dtype=np.uint64)
    shifts = np.arange(bits_per_symbol - 1, -1, -1, dtype=np.uint64)
    bits = ((symbol_indices[:, None] >> shifts) & 1).astype(np.uint8)
    return np.asarray(
        modulate(bits.reshape(-1), modulation_order),
        dtype=np.complex64,
    )


def hard_decision(
    symbols: np.ndarray,
    modulation_order: int,
) -> np.ndarray:
    """Return nearest-neighbor constellation decisions."""
    values = np.asarray(symbols, dtype=np.complex64)
    points = _constellation_points(modulation_order)
    flat = values.reshape(-1)
    decisions = np.full(flat.shape, np.nan + 1j * np.nan, dtype=np.complex64)
    valid = np.isfinite(flat.real) & np.isfinite(flat.imag)
    distances = np.abs(flat[valid, None] - points[None, :])
    decisions[valid] = points[np.argmin(distances, axis=1)]
    return decisions.reshape(values.shape)


def calculate_link_metrics(
    transmitted_symbols: np.ndarray,
    equalized_symbols: np.ndarray,
    channel_coefficients: np.ndarray,
    modulation_order: int,
) -> dict[str, float | int]:
    """Calculate EVM, EVM-derived SNR, SER, and predicted MRT gain."""
    transmitted = np.asarray(transmitted_symbols, dtype=np.complex64).reshape(-1)
    equalized = np.asarray(equalized_symbols, dtype=np.complex64).reshape(-1)
    if transmitted.shape != equalized.shape:
        raise ValueError("transmitted and equalized symbol counts must match")
    valid = np.isfinite(equalized.real) & np.isfinite(equalized.imag)
    if not np.any(valid):
        raise ValueError("no finite equalized symbols are available")

    transmitted = transmitted[valid]
    equalized = equalized[valid]
    error_power = float(np.mean(np.abs(equalized - transmitted) ** 2))
    reference_power = float(np.mean(np.abs(transmitted) ** 2))
    evm_ratio = math.sqrt(error_power / max(reference_power, 1e-20))
    decisions = hard_decision(equalized, modulation_order)
    symbol_error_rate = float(np.mean(decisions != transmitted))

    channel = np.asarray(channel_coefficients, dtype=np.complex64)
    mrt_amplitude = float(np.linalg.norm(channel))
    best_single_amplitude = float(np.max(np.abs(channel)))
    predicted_gain_db = 20.0 * math.log10(
        mrt_amplitude / max(best_single_amplitude, 1e-20)
    )
    return {
        "num_compared_symbols": int(transmitted.size),
        "evm_rms_percent": 100.0 * evm_ratio,
        "evm_snr_db": -20.0 * math.log10(max(evm_ratio, 1e-20)),
        "symbol_error_rate": symbol_error_rate,
        "predicted_array_gain_vs_best_single_tx_db": predicted_gain_db,
    }


def plot_constellation_comparison(
    transmitted_symbols: np.ndarray,
    equalized_symbols: np.ndarray,
    output_path: Path,
    *,
    modulation_order: int,
    max_symbols: int = 10000,
) -> Path:
    """Save side-by-side transmitted and recovered constellation plots."""
    if max_symbols <= 0:
        raise ValueError("max_symbols must be greater than zero")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    transmitted = np.asarray(transmitted_symbols).reshape(-1)
    equalized = np.asarray(equalized_symbols).reshape(-1)
    valid = np.isfinite(equalized.real) & np.isfinite(equalized.imag)
    transmitted = transmitted[valid]
    equalized = equalized[valid]
    if transmitted.size > max_symbols:
        selected = np.linspace(
            0,
            transmitted.size - 1,
            max_symbols,
            dtype=np.int64,
        )
        transmitted = transmitted[selected]
        equalized = equalized[selected]

    ideal = _constellation_points(modulation_order)
    all_values = np.concatenate((ideal, transmitted, equalized))
    axis_limit = max(1.0, float(np.max(np.abs(all_values)))) * 1.2
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    axes[0].scatter(
        transmitted.real,
        transmitted.imag,
        s=8,
        alpha=0.35,
        label="Transmitted PDSCH",
    )
    axes[1].scatter(
        equalized.real,
        equalized.imag,
        s=8,
        alpha=0.25,
        label="Equalized PDSCH",
    )
    for axis, title in zip(
        axes,
        ("Original common stream", "Recovered stream after MRT combining"),
        strict=True,
    ):
        axis.scatter(
            ideal.real,
            ideal.imag,
            marker="x",
            s=80,
            linewidths=2,
            color="black",
            label="Ideal constellation",
        )
        axis.set_title(title)
        axis.set_xlabel("In-phase")
        axis.set_ylabel("Quadrature")
        axis.set_xlim(-axis_limit, axis_limit)
        axis.set_ylim(-axis_limit, axis_limit)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def _transmit_worker(
    *,
    frames,
    stop_event,
    params,
    sync_barrier,
    receiver_ready_event,
    start_time_queue,
) -> None:
    """Initialize the RU devices and transmit the precomputed MRT batch."""
    try:
        from USRP.usrp_utils import initialize_tx_usrp

        usrp = initialize_tx_usrp(
            device_args=TX_USRP_DEVICE_ARGS,
            tx_subdev_spec=TX_SUBDEV_SPEC,
            tx_antenna=TX_ANTENNA,
            tx_channels=TX_CHANNELS,
            clock_source=CLOCK_SOURCE,
            time_source=TIME_SOURCE,
            sync_barrier=sync_barrier,
        )
        while not stop_event.is_set():
            if receiver_ready_event.wait(timeout=0.2):
                break
        else:
            return

        transmitter = USRPTransmitter(
            usrp=usrp,
            frames=frames,
            carrier_frequency=params["carrier_frequency"],
            sampling_rate=params["sampling_rate"],
            gain=Tx_gain,
            channels=TX_CHANNELS,
            transmission_delay=WAIT_TIME,
            stop_event=stop_event,
            otw_format=OTW_FORMAT,
            start_time_queue=start_time_queue,
        )
        transmitter.run()
    except KeyboardInterrupt:
        stop_event.set()
    except Exception:
        stop_event.set()
        raise


def _receive_worker(
    *,
    frames_queue,
    stop_event,
    params,
    sync_barrier,
    receiver_ready_event,
    start_time_queue,
) -> None:
    """Initialize the UE device and receive frames through USRPReceiver."""
    try:
        import uhd
        from USRP.usrp_utils import initialize_rx_usrp

        usrp = initialize_rx_usrp(
            device_args=RX_USRP_DEVICE_ARGS,
            rx_subdev_spec=RX_SUBDEV_SPEC,
            rx_antenna=RX_ANTENNA,
            rx_channels=RX_CHANNELS,
            clock_source=CLOCK_SOURCE,
            time_source=TIME_SOURCE,
            sync_barrier=sync_barrier,
        )
        receiver = USRPReceiver(
            usrp=usrp,
            frames_queue=frames_queue,
            num_samples=params["frame_length"],
            carrier_frequency=params["carrier_frequency"],
            sampling_rate=params["sampling_rate"],
            gain=Rx_gain,
            channels=RX_CHANNELS,
            start_time=None,
            otw_format=OTW_FORMAT,
            stop_event=stop_event,
            uhd_module=uhd,
        )
        receiver_ready_event.set()

        while not stop_event.is_set():
            try:
                receiver.start_time = start_time_queue.get(timeout=0.2)
                break
            except Empty:
                continue
        else:
            return

        receiver.run()
    except KeyboardInterrupt:
        stop_event.set()
    except Exception:
        stop_event.set()
        raise
    finally:
        cancel_join = getattr(frames_queue, "cancel_join_thread", None)
        if cancel_join is not None:
            cancel_join()


def _stop_processes(processes, stop_event) -> None:
    stop_event.set()
    for process in processes:
        process.join(timeout=5)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join()


def _collect_recovered_frames(
    frames_queue,
    stop_event,
    receiver_process,
    transmitter_process,
    params,
    known_ref_seq,
    frame_batch: MRTFrameBatch,
) -> list[RecoveredFrame]:
    """Consume continuous RX frames; the next frame supplies timing tail."""
    target_frames = int(frame_batch.waveforms.shape[0])
    trailing_samples = max(
        1,
        int(float(params["sampling_rate"]) * RX_TRAILING_TIME),
    )
    pending_frame = None
    recovered = []

    while not stop_event.is_set() and len(recovered) < target_frames:
        try:
            current_frame = frames_queue.get(timeout=0.5)
        except Empty:
            failed = [
                process
                for process in (receiver_process, transmitter_process)
                if process.exitcode not in (None, 0)
            ]
            if failed:
                names = ", ".join(process.name for process in failed)
                raise RuntimeError(f"radio worker failed: {names}")
            continue

        if pending_frame is None:
            pending_frame = current_frame
            continue
        if current_frame.frame_id != pending_frame.frame_id + 1:
            print(
                f"Dropped RX frame after {pending_frame.frame_id}; "
                "restarting at the newest frame"
            )
            pending_frame = current_frame
            continue

        tx_frame_index = int(pending_frame.frame_id)
        if tx_frame_index >= target_frames:
            break
        capture = np.concatenate(
            (
                np.asarray(pending_frame.samples, dtype=np.complex64),
                np.asarray(current_frame.samples, dtype=np.complex64)[
                    :, :trailing_samples
                ],
            ),
            axis=1,
        )
        recovered.append(
            recover_frame(
                capture,
                params,
                known_ref_seq,
                frame_batch,
                tx_frame_index,
            )
        )
        pending_frame = current_frame

    stop_event.set()
    return recovered


def _save_results(
    output_dir: Path,
    params,
    channel: np.ndarray,
    frame_batch: MRTFrameBatch,
    recovered_frames: list[RecoveredFrame],
    *,
    frame_seed: int,
    max_plot_symbols: int,
) -> dict[str, object]:
    if not recovered_frames:
        raise RuntimeError("no radio frames were recovered")

    frame_indices = np.asarray(
        [frame.frame_index for frame in recovered_frames],
        dtype=np.int64,
    )
    transmitted = frame_batch.pdsch_symbols[frame_indices]
    raw = np.stack([frame.raw_symbols for frame in recovered_frames], axis=0)
    equalized = np.stack(
        [frame.equalized_symbols for frame in recovered_frames],
        axis=0,
    )
    effective_channel = np.stack(
        [frame.effective_channel for frame in recovered_frames],
        axis=0,
    )
    decisions = hard_decision(equalized, int(params["modulation_order"]))

    # The current online path targets one UE RX. Retain the RX axis in the NPZ
    # while evaluating and plotting the first configured UE antenna.
    first_rx_equalized = equalized[:, 0, :]
    metrics = calculate_link_metrics(
        transmitted,
        first_rx_equalized,
        channel,
        int(params["modulation_order"]),
    )
    run_name = datetime.now().strftime("mrt_link_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{run_name}.npz"
    json_path = output_dir / f"{run_name}.json"
    plot_path = output_dir / f"{run_name}_constellation.png"

    np.savez_compressed(
        npz_path,
        predicted_channel=channel,
        mrt_weights=frame_batch.weights,
        rx_frame_indices=frame_indices,
        tx_pdsch_symbols=transmitted,
        rx_pdsch_symbols_raw=raw,
        rx_pdsch_symbols_equalized=equalized,
        rx_pdsch_symbols_hard=decisions,
        effective_channel=effective_channel,
        sync_indices=np.asarray(
            [frame.sync_index for frame in recovered_frames],
            dtype=np.int64,
        ),
        ffo_hz=np.stack([frame.ffo_hz for frame in recovered_frames]),
        rfo_hz=np.stack([frame.rfo_hz for frame in recovered_frames]),
    )
    plot_constellation_comparison(
        transmitted,
        first_rx_equalized,
        plot_path,
        modulation_order=int(params["modulation_order"]),
        max_symbols=max_plot_symbols,
    )
    summary = {
        "bandwidth_mhz": float(params["bandwidth_mhz"]),
        "num_tx_antennas": int(params["num_tx_ant"]),
        "num_rx_antennas": int(params["num_rx_ant"]),
        "num_requested_frames": int(frame_batch.waveforms.shape[0]),
        "num_recovered_frames": len(recovered_frames),
        "frame_seed": int(frame_seed),
        "predicted_channel": [
            [float(value.real), float(value.imag)] for value in channel
        ],
        "mrt_weights": [
            [float(value.real), float(value.imag)]
            for value in frame_batch.weights
        ],
        **metrics,
    }
    json_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return {
        "npz_path": npz_path,
        "json_path": json_path,
        "plot_path": plot_path,
        "metrics": metrics,
    }


def run_mrt_transceiver(
    channel_coefficients: Sequence[complex] | np.ndarray,
    *,
    num_frames: int = 20,
    frame_seed: int | None = None,
    bandwidth_mhz: float = BANDWIDTH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    max_plot_symbols: int = 10000,
) -> dict[str, object]:
    """Execute the complete RU-to-UE MRT experiment and save its results."""
    params = get_system_params(bandwidth_mhz)
    if len(TX_CHANNELS) != int(params["num_tx_ant"]):
        raise ValueError("TX_CHANNELS does not match params['num_tx_ant']")
    if len(RX_CHANNELS) != int(params["num_rx_ant"]):
        raise ValueError("RX_CHANNELS does not match params['num_rx_ant']")
    resolved_seed = secrets.randbits(63) if frame_seed is None else frame_seed
    if resolved_seed < 0:
        raise ValueError("frame_seed must be non-negative")
    channel = np.asarray(channel_coefficients, dtype=np.complex64).reshape(-1)
    reference_seed = REFERENCE_SEQUENCE_SEED + int(10 * bandwidth_mhz)
    known_ref_seq = generate_known_reference_sequence(
        params["N"],
        reference_seed,
    )
    frame_batch = build_single_stream_mrt_batch(
        params,
        known_ref_seq,
        channel,
        num_frames=num_frames,
        frame_seed=resolved_seed,
    )

    context = get_context("spawn")
    frames_queue = context.Queue(maxsize=8)
    start_time_queue = context.Queue(maxsize=1)
    stop_event = context.Event()
    receiver_ready_event = context.Event()
    sync_barrier = context.Barrier(2)
    receiver = context.Process(
        name="mrt-ue-receiver",
        target=_receive_worker,
        kwargs={
            "frames_queue": frames_queue,
            "stop_event": stop_event,
            "params": params,
            "sync_barrier": sync_barrier,
            "receiver_ready_event": receiver_ready_event,
            "start_time_queue": start_time_queue,
        },
    )
    transmitter = context.Process(
        name="mrt-ru-transmitter",
        target=_transmit_worker,
        kwargs={
            "frames": frame_batch.waveforms,
            "stop_event": stop_event,
            "params": params,
            "sync_barrier": sync_barrier,
            "receiver_ready_event": receiver_ready_event,
            "start_time_queue": start_time_queue,
        },
    )
    processes = (receiver, transmitter)
    started = []
    try:
        for process in processes:
            process.start()
            started.append(process)
        recovered = _collect_recovered_frames(
            frames_queue,
            stop_event,
            receiver,
            transmitter,
            params,
            known_ref_seq,
            frame_batch,
        )
    except KeyboardInterrupt:
        print("Stopping MRT transceiver...")
        raise
    finally:
        _stop_processes(started, stop_event)
        for process_queue in (frames_queue, start_time_queue):
            process_queue.close()
            process_queue.join_thread()

    failed = [
        process
        for process in processes
        if process.exitcode not in (0, None)
    ]
    if failed:
        names = ", ".join(process.name for process in failed)
        raise RuntimeError(f"radio worker failed: {names}")
    return _save_results(
        Path(output_dir),
        params,
        channel,
        frame_batch,
        recovered,
        frame_seed=resolved_seed,
        max_plot_symbols=max_plot_symbols,
    )


def _parse_complex(value: str) -> complex:
    try:
        return complex(value.strip().replace("i", "j"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid complex coefficient {value!r}"
        ) from exc


def _load_channel_file(path: Path, key: str) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.complex64)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive:
                raise KeyError(f"{key!r} was not found in {path}")
            return np.asarray(archive[key], dtype=np.complex64)
    raise ValueError("--channel-file must be a .npy or .npz file")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transmit one common PDSCH stream with MRT, receive it at the "
            "UE, equalize it, and save constellation results."
        )
    )
    channel_group = parser.add_mutually_exclusive_group(required=True)
    channel_group.add_argument(
        "--channel",
        type=_parse_complex,
        nargs="+",
        help="Predicted complex channel coefficient for each TX antenna.",
    )
    channel_group.add_argument(
        "--channel-file",
        type=Path,
        help="NPY or NPZ file containing the predicted channel vector.",
    )
    parser.add_argument(
        "--channel-key",
        default="predicted_channel",
        help="NPZ key used by --channel-file.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=20,
        help="Number of 5 ms frames to transmit and recover.",
    )
    parser.add_argument(
        "--frame-seed",
        type=int,
        default=None,
        help="Random PDSCH seed; a fresh seed is used when omitted.",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=BANDWIDTH,
        help="Configured RF bandwidth in MHz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Directory for NPZ, JSON, and constellation PNG files.",
    )
    parser.add_argument(
        "--max-plot-symbols",
        type=int,
        default=10000,
        help="Maximum number of symbols drawn in each constellation panel.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate the common-stream MRT frames without USRPs.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    channel = (
        np.asarray(args.channel, dtype=np.complex64)
        if args.channel is not None
        else _load_channel_file(args.channel_file, args.channel_key).reshape(-1)
    )
    seed = secrets.randbits(63) if args.frame_seed is None else args.frame_seed
    if args.dry_run:
        params = get_system_params(args.bandwidth)
        known_ref_seq = generate_known_reference_sequence(
            params["N"],
            REFERENCE_SEQUENCE_SEED + int(10 * args.bandwidth),
        )
        frame_batch = build_single_stream_mrt_batch(
            params,
            known_ref_seq,
            channel,
            num_frames=args.frames,
            frame_seed=seed,
        )
        print(f"MRT weights: {frame_batch.weights}")
        print(
            "Dry run complete: common PDSCH symbols "
            f"{frame_batch.pdsch_symbols.shape}, TX waveforms "
            f"{frame_batch.waveforms.shape}"
        )
        return

    artifacts = run_mrt_transceiver(
        channel,
        num_frames=args.frames,
        frame_seed=seed,
        bandwidth_mhz=args.bandwidth,
        output_dir=args.output_dir,
        max_plot_symbols=args.max_plot_symbols,
    )
    print("MRT transceiver run complete")
    for name in ("npz_path", "json_path", "plot_path"):
        print(f"  {name}: {artifacts[name]}")
    for name, value in artifacts["metrics"].items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
