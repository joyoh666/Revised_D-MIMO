"""Transmit a single OFDM stream with MRT precoding.

The channel predictor is expected to provide one complex downlink channel
coefficient per configured TX antenna.  For a channel vector ``h``, this
module uses the unit-norm MRT vector ``conj(h) / ||h||`` and transmits the
same baseband stream from every antenna after applying its corresponding
weight.

Example (two configured TX antennas)::

    python -m runtime.mrt_transmitter \
        --channel 0.8+0.1j 0.3-0.4j \
        --frames 100

Use ``--dry-run`` to validate the channel and construct the frames without
opening a USRP device.
"""

from __future__ import annotations

import argparse
import secrets
from pathlib import Path
from threading import Event
from typing import Sequence

import numpy as np

from config.radio_config import (
    BANDWIDTH,
    CLOCK_SOURCE,
    OTW_FORMAT,
    REFERENCE_SEQUENCE_SEED,
    TIME_SOURCE,
    TX_ANTENNA,
    TX_CHANNELS,
    TX_SUBDEV_SPEC,
    TX_USRP_DEVICE_ARGS,
    Tx_gain,
    WAIT_TIME,
)
from phy.precoder import MRTPrecoder
from runtime.transmitter import USRPTransmitter
from USRP.build_frame import build_frame
from USRP.helper_functions import (
    generate_known_reference_sequence,
    get_system_params,
)


def _validate_channel_coefficients(
    channel_coefficients: Sequence[complex] | np.ndarray,
    num_tx_antennas: int,
) -> np.ndarray:
    """Return a finite, nonzero channel vector with one value per TX."""
    channel = np.asarray(channel_coefficients, dtype=np.complex64)
    if channel.shape != (num_tx_antennas,):
        raise ValueError(
            "channel_coefficients must have shape "
            f"{(num_tx_antennas,)}, got {channel.shape}"
        )
    if not np.all(np.isfinite(channel.real)) or not np.all(
        np.isfinite(channel.imag)
    ):
        raise ValueError("channel_coefficients must contain finite values")
    if np.linalg.norm(channel) < 1e-12:
        raise ValueError("channel_coefficients must not be an all-zero vector")
    return channel


def calculate_mrt_weights(
    channel_coefficients: Sequence[complex] | np.ndarray,
    *,
    num_tx_antennas: int | None = None,
) -> np.ndarray:
    """Calculate ``conj(h) / ||h||`` for a single-user MISO channel."""
    channel = np.asarray(channel_coefficients, dtype=np.complex64)
    if num_tx_antennas is None:
        if channel.ndim != 1:
            raise ValueError("channel_coefficients must be a one-dimensional vector")
        num_tx_antennas = int(channel.size)
    channel = _validate_channel_coefficients(channel, num_tx_antennas)
    weights = MRTPrecoder().precode(channel)
    return np.asarray(weights, dtype=np.complex64)


def apply_mrt_precoding(
    baseband_signal: Sequence[complex] | np.ndarray,
    channel_coefficients: Sequence[complex] | np.ndarray,
    *,
    num_tx_antennas: int | None = None,
) -> np.ndarray:
    """Map one baseband stream to ``(num_tx_antennas, num_samples)``.

    MRT weights have unit Euclidean norm, so precoding preserves the total
    sample power summed across all TX antennas.
    """
    signal = np.asarray(baseband_signal, dtype=np.complex64)
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("baseband_signal must be a non-empty 1-D array")
    if not np.all(np.isfinite(signal.real)) or not np.all(
        np.isfinite(signal.imag)
    ):
        raise ValueError("baseband_signal must contain finite values")

    weights = calculate_mrt_weights(
        channel_coefficients,
        num_tx_antennas=num_tx_antennas,
    )
    return np.ascontiguousarray(
        weights[:, np.newaxis] * signal[np.newaxis, :],
        dtype=np.complex64,
    )


def build_mrt_frames(
    params,
    known_ref_seq,
    channel_coefficients: Sequence[complex] | np.ndarray,
    *,
    num_frames: int,
    frame_seed: int,
) -> np.ndarray:
    """Build finite random-PDSCH frames carrying one MRT-precoded stream.

    ``build_frame`` normally creates an independent PDSCH stream for every
    antenna.  MRT requires a common stream, so the configured synchronization
    stream is selected as the source waveform and copied to every antenna
    through the MRT weights.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be greater than zero")
    if frame_seed < 0:
        raise ValueError("frame_seed must be non-negative")

    num_tx_antennas = int(params["num_tx_ant"])
    channel = _validate_channel_coefficients(
        channel_coefficients,
        num_tx_antennas,
    )
    frame_length = int(params["frame_length"])
    source_tx_index = int(params["sync_tx_idx"])
    if not 0 <= source_tx_index < num_tx_antennas:
        raise ValueError("params['sync_tx_idx'] is outside the TX antenna range")

    frames = np.empty(
        (num_frames, num_tx_antennas, frame_length),
        dtype=np.complex64,
    )
    rng = np.random.default_rng(frame_seed)
    for frame_index in range(num_frames):
        frame = build_frame(params, known_ref_seq, rng=rng)
        waveform = np.asarray(frame["waveform"], dtype=np.complex64)
        expected_shape = (num_tx_antennas, frame_length)
        if waveform.shape != expected_shape:
            raise ValueError(
                f"source frame {frame_index} has shape {waveform.shape}, "
                f"expected {expected_shape}"
            )
        frames[frame_index] = apply_mrt_precoding(
            waveform[source_tx_index],
            channel,
            num_tx_antennas=num_tx_antennas,
        )
    return frames


def transmit_mrt_frames(
    channel_coefficients: Sequence[complex] | np.ndarray,
    *,
    num_frames: int = 1,
    frame_seed: int | None = None,
    bandwidth_mhz: float = BANDWIDTH,
    stop_event=None,
) -> np.ndarray:
    """Build and transmit a finite MRT frame batch through the configured RU.

    The returned array is the exact complex64 frame batch given to UHD.  This
    makes the function usable by a later predictor worker and straightforward
    to verify in hardware-free tests.
    """
    params = get_system_params(bandwidth_mhz)
    if len(TX_CHANNELS) != int(params["num_tx_ant"]):
        raise ValueError(
            "TX_CHANNELS and params['num_tx_ant'] must describe the same "
            "number of transmit antennas"
        )
    resolved_seed = secrets.randbits(63) if frame_seed is None else frame_seed
    if resolved_seed < 0:
        raise ValueError("frame_seed must be non-negative")

    reference_seed = REFERENCE_SEQUENCE_SEED + int(10 * bandwidth_mhz)
    known_ref_seq = generate_known_reference_sequence(
        params["N"],
        reference_seed,
    )
    frames = build_mrt_frames(
        params,
        known_ref_seq,
        channel_coefficients,
        num_frames=num_frames,
        frame_seed=resolved_seed,
    )

    from USRP.usrp_utils import initialize_tx_usrp

    usrp = initialize_tx_usrp(
        device_args=TX_USRP_DEVICE_ARGS,
        tx_subdev_spec=TX_SUBDEV_SPEC,
        tx_antenna=TX_ANTENNA,
        tx_channels=TX_CHANNELS,
        clock_source=CLOCK_SOURCE,
        time_source=TIME_SOURCE,
    )
    transmitter = USRPTransmitter(
        usrp=usrp,
        frames=frames,
        carrier_frequency=params["carrier_frequency"],
        sampling_rate=params["sampling_rate"],
        gain=Tx_gain,
        channels=TX_CHANNELS,
        transmission_delay=WAIT_TIME,
        stop_event=Event() if stop_event is None else stop_event,
        otw_format=OTW_FORMAT,
    )
    transmitter.run()
    return frames


def _parse_complex(value: str) -> complex:
    try:
        return complex(value.strip().replace("i", "j"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid complex coefficient {value!r}; use a form such as 0.8-0.2j"
        ) from exc


def _load_channel_file(path: Path, key: str) -> np.ndarray:
    """Load a channel vector from a NPY or NPZ predictor output."""
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
            "Apply single-user MRT to an OFDM stream and transmit it from "
            "the configured RU antennas."
        )
    )
    channel_group = parser.add_mutually_exclusive_group(required=True)
    channel_group.add_argument(
        "--channel",
        type=_parse_complex,
        nargs="+",
        help=(
            "Predicted complex channel coefficients, one per TX antenna "
            "(example: --channel 0.8+0.1j 0.3-0.4j)."
        ),
    )
    channel_group.add_argument(
        "--channel-file",
        type=Path,
        help="NPY or NPZ file containing the predicted complex channel vector.",
    )
    parser.add_argument(
        "--channel-key",
        default="predicted_channel",
        help="Array key used with --channel-file when the file is NPZ.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=1,
        help="Number of 5 ms frames to transmit.",
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
        "--dry-run",
        action="store_true",
        help="Build and validate the MRT frames without opening the USRP.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    channel = (
        np.asarray(args.channel, dtype=np.complex64)
        if args.channel is not None
        else _load_channel_file(args.channel_file, args.channel_key)
    )
    channel = np.asarray(channel, dtype=np.complex64).reshape(-1)
    seed = secrets.randbits(63) if args.frame_seed is None else args.frame_seed

    params = get_system_params(args.bandwidth)
    weights = calculate_mrt_weights(
        channel,
        num_tx_antennas=int(params["num_tx_ant"]),
    )
    print(f"Predicted channel: {channel}")
    print(f"MRT weights: {weights}")
    print(f"MRT weight norm: {np.linalg.norm(weights):.6f}")

    if args.dry_run:
        reference_seed = REFERENCE_SEQUENCE_SEED + int(10 * args.bandwidth)
        known_ref_seq = generate_known_reference_sequence(
            params["N"],
            reference_seed,
        )
        frames = build_mrt_frames(
            params,
            known_ref_seq,
            channel,
            num_frames=args.frames,
            frame_seed=seed,
        )
        print(
            "Dry run complete: built MRT frame batch with shape "
            f"{frames.shape} and dtype {frames.dtype}."
        )
        return

    transmit_mrt_frames(
        channel,
        num_frames=args.frames,
        frame_seed=seed,
        bandwidth_mhz=args.bandwidth,
    )


if __name__ == "__main__":
    main()
