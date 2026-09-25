"""Entry point for repeated virtual-pilot OFDM channel capture."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from config.radio_config import (
    CAPTURE_BANDWIDTHS,
    NUM_CHANNEL_CAPTURES,
    NUM_VIRTUAL_PILOTS,
    OTW_FORMAT,
    OUTPUT_DIR,
    REFERENCE_SEQUENCE_SEED,
    RX_ANTENNA,
    RX_CHANNELS,
    RX_SUBDEV_SPEC,
    RX_TRAILING_TIME,
    TX_ANTENNA,
    TX_CHANNELS,
    TX_DELAY_TIME,
    TX_SUBDEV_SPEC,
    USRP_DEVICE_ARGS,
    WAIT_TIME,
)
from USRP.build_frame import build_frame
from USRP.helper_functions import (
    generate_known_reference_sequence,
    get_system_params,
)
from USRP.postprocessing import postprocess
from USRP.results import CaptureAccumulator, plot_results, save_results


CaptureFunction = Callable[[Any, np.ndarray, dict[str, Any]], np.ndarray]


def _initialize_usrp() -> Any:
    """Load UHD only when a real hardware capture is requested."""
    from USRP.usrp_utils import initialize_usrp

    return initialize_usrp(
        USRP_DEVICE_ARGS,
        TX_SUBDEV_SPEC,
        RX_SUBDEV_SPEC,
        TX_ANTENNA,
        RX_ANTENNA,
        tx_channels=TX_CHANNELS,
        rx_channels=RX_CHANNELS,
    )


def _capture_from_usrp(
    usrp: Any,
    waveform: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    """Transmit one frame and return every configured RX stream."""
    from USRP.usrp_utils import sendAndReceive

    rx_channels = None if RX_CHANNELS is None else list(RX_CHANNELS)
    return sendAndReceive(
        usrp,
        waveform,
        1,
        params["carrier_frequency"],
        params["sampling_rate"],
        params["Tx_gain"],
        params["Rx_gain"],
        list(TX_CHANNELS),
        rx_channels,
        wait_time=WAIT_TIME,
        tx_delay_samples=int(params["sampling_rate"] * TX_DELAY_TIME),
        rx_trailing_samples=int(
            params["sampling_rate"] * RX_TRAILING_TIME
        ),
        otw_format=OTW_FORMAT,
    )


def collect_channel_results(
    usrp: Any,
    params: dict[str, Any],
    frame_objects: dict[str, Any],
    known_ref_seq: np.ndarray,
    *,
    num_captures: int,
    capture_function: CaptureFunction | None = None,
) -> dict[str, Any]:
    """Capture, postprocess, and aggregate one bandwidth setting."""
    if num_captures <= 0:
        raise ValueError("num_captures must be positive")

    capture = capture_function or _capture_from_usrp
    accumulator = CaptureAccumulator(params)

    for capture_index in range(num_captures):
        print(
            f"[BW {params['bandwidth_mhz']:>5} MHz] Capture "
            f"{capture_index + 1}/{num_captures}"
        )
        frame_rcv = capture(usrp, frame_objects["waveform"], params)
        received = postprocess(
            params,
            frame_objects["ss_td_with_cp"],
            frame_objects["pdsch_idx"],
            known_ref_seq,
            frame_rcv,
        )
        accumulator.add(received)

    return accumulator.finalize()


def _print_saved_summary(
    npz_path: str,
    json_path: str,
    plot_paths: dict[str, str],
    metadata: dict[str, Any],
) -> None:
    print("Saved results:")
    print(f"  NPZ  : {npz_path}")
    print(f"  JSON : {json_path}")
    for name, path in plot_paths.items():
        print(f"  Plot ({name}): {path}")

    print("Important saved metadata:")
    for key in (
        "bandwidth_mhz",
        "subcarrier_spacing_hz",
        "num_active_subcarriers",
        "fft_size",
        "sampling_rate_hz",
        "frame_length_samples",
        "num_channel_captures",
        "num_receive_antennas",
        "saved_channel_tensor_shape",
        "saved_mean_channel_shape",
        "saved_delay_response_shape",
    ):
        print(f"  {key}: {metadata[key]}")


def run_capture_experiment(
    *,
    bandwidths: Sequence[float] = CAPTURE_BANDWIDTHS,
    num_captures: int = NUM_CHANNEL_CAPTURES,
    output_dir: str | Path = OUTPUT_DIR,
    show_plots: bool = True,
    usrp: Any | None = None,
    capture_function: CaptureFunction | None = None,
) -> list[dict[str, Any]]:
    """Run every requested bandwidth and return its generated artifacts.

    Passing ``capture_function`` permits hardware-free loopback or replay
    tests. The callback receives ``(usrp, waveform, params)`` and must return
    samples shaped ``(num_rx_ant, num_samples)``.
    """
    if not bandwidths:
        raise ValueError("at least one bandwidth must be selected")
    if num_captures <= 0:
        raise ValueError("num_captures must be positive")

    validated_params: list[dict[str, Any]] = []
    for bandwidth_mhz in bandwidths:
        try:
            validated_params.append(get_system_params(bandwidth_mhz))
        except KeyError as exc:
            raise ValueError(
                f"unsupported bandwidth: {bandwidth_mhz} MHz"
            ) from exc

    if usrp is None and capture_function is None:
        usrp = _initialize_usrp()

    artifacts: list[dict[str, Any]] = []
    for params in validated_params:
        bandwidth_mhz = params["bandwidth_mhz"]
        print("=" * 80)
        print(f"Running bandwidth setting: {bandwidth_mhz} MHz")
        print(
            f"Virtual pilot count: {NUM_VIRTUAL_PILOTS} | "
            "expected channel-estimation gain: "
            f"{10 * np.log10(NUM_VIRTUAL_PILOTS):.2f} dB"
        )

        reference_seed = REFERENCE_SEQUENCE_SEED + int(10 * bandwidth_mhz)
        params["reference_sequence_seed"] = reference_seed
        known_ref_seq = generate_known_reference_sequence(
            params["N"],
            seed=reference_seed,
        )
        frame_objects = build_frame(params, known_ref_seq)
        results = collect_channel_results(
            usrp,
            params,
            frame_objects,
            known_ref_seq,
            num_captures=num_captures,
            capture_function=capture_function,
        )

        npz_path, json_path, figure_path, metadata = save_results(
            output_dir,
            params,
            known_ref_seq,
            results,
            reference_sequence_seed=reference_seed,
        )
        plot_paths = plot_results(
            figure_path,
            params,
            results,
            show=show_plots,
        )
        _print_saved_summary(npz_path, json_path, plot_paths, metadata)

        artifacts.append(
            {
                "params": params,
                "results": results,
                "npz_path": npz_path,
                "json_path": json_path,
                "plot_paths": plot_paths,
                "metadata": metadata,
            }
        )

    return artifacts


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture and save virtual-pilot OFDM channel estimates."
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        nargs="+",
        default=list(CAPTURE_BANDWIDTHS),
        help="One or more configured bandwidth values in MHz.",
    )
    parser.add_argument(
        "--captures",
        type=int,
        default=NUM_CHANNEL_CAPTURES,
        help="Number of repeated captures per bandwidth.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Directory for NPZ, JSON, and PNG outputs.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save plots without opening interactive windows.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    run_capture_experiment(
        bandwidths=args.bandwidth,
        num_captures=args.captures,
        output_dir=args.output_dir,
        show_plots=not args.no_show,
    )


if __name__ == "__main__":
    main()
