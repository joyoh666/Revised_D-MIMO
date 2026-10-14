import argparse
import math
from multiprocessing import freeze_support, get_context
from pathlib import Path
import secrets
from typing import Sequence

from config.radio_config import (
    BANDWIDTH,
    CSI_WINDOW_LENGTH,
    NUM_CHANNEL_CAPTURES,
    OTW_FORMAT,
    OUTPUT_DIR,
    REFERENCE_SEQUENCE_SEED,
    RX_ANTENNA,
    RX_CHANNELS,
    RX_SUBDEV_SPEC,
    RX_TRAILING_TIME,
    Rx_gain,
    TX_ANTENNA,
    TX_CHANNELS,
    TX_SUBDEV_SPEC,
    TX_USRP_DEVICE_ARGS,
    Tx_gain,
)
from USRP.build_frame import build_frame
from USRP.helper_functions import (
    generate_known_reference_sequence,
    get_system_params,
)

from .postprocessor import process_received_frames
from .single_usrp import capture_single_usrp


def _wait_for_processes(processes) -> bool:
    """Wait for the workers and return whether the user interrupted the run."""
    try:
        while any(process.is_alive() for process in processes):
            for process in processes:
                process.join(timeout=0.25)
            if any(
                process.exitcode not in (None, 0)
                for process in processes
            ):
                break
    except KeyboardInterrupt:
        print("Stopping runtime processes...")
        return True

    return False


def _stop_processes(processes, stop_event) -> None:
    stop_event.set()

    for process in processes:
        process.join(timeout=5)

    for process in processes:
        if process.is_alive():
            print(f"Terminating unresponsive process: {process.name}")
            process.terminate()

    for process in processes:
        process.join()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect 2-TX/1-RX CSI on one X310 every 5 ms."
    )
    capture_group = parser.add_mutually_exclusive_group()
    capture_group.add_argument(
        "--captures",
        type=int,
        default=None,
        help="Number of 5 ms CSI snapshots to save.",
    )
    capture_group.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Transmit duration in seconds; must be a multiple of 5 ms.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(OUTPUT_DIR),
        help="Directory for NPZ, JSON, and plot artifacts.",
    )
    parser.add_argument(
        "--window-length",
        "--sequence-length",
        dest="window_length",
        type=int,
        default=CSI_WINDOW_LENGTH,
        help="CSI samples per window; must be a positive multiple of 5.",
    )
    parser.add_argument(
        "--frame-seed",
        type=int,
        default=None,
        help=(
            "Seed for random PDSCH frames. By default a new seed is generated "
            "for every run."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Save NPZ and JSON only.",
    )
    return parser.parse_args(argv)


def _resolve_num_captures(args: argparse.Namespace, csi_period_s: float) -> int:
    if args.duration_s is not None:
        if args.duration_s <= 0:
            raise ValueError("--duration-s must be greater than zero")
        exact_frame_count = args.duration_s / csi_period_s
        num_captures = round(exact_frame_count)
        if not math.isclose(
            exact_frame_count,
            num_captures,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "--duration-s must be a multiple of the CSI period "
                f"({csi_period_s * 1e3:g} ms)"
            )
        return num_captures

    return (
        NUM_CHANNEL_CAPTURES
        if args.captures is None
        else args.captures
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    bandwidth_mhz = BANDWIDTH
    params = get_system_params(bandwidth_mhz)
    num_captures = _resolve_num_captures(
        args,
        float(params["csi_sample_period_s"]),
    )
    if num_captures <= 0:
        raise ValueError("--captures must be greater than zero")
    if args.window_length <= 0 or args.window_length % 5:
        raise ValueError("--window-length must be a positive multiple of 5")
    if num_captures < args.window_length:
        raise ValueError("--captures must be at least --window-length")
    frame_seed = (
        secrets.randbits(63)
        if args.frame_seed is None
        else args.frame_seed
    )
    if frame_seed < 0:
        raise ValueError("--frame-seed must be non-negative")

    # Only the hardware subprocess initializes UHD; both TX/RX streamers
    # share a single X310 instance and use one scheduled device timestamp.
    context = get_context("spawn")
    frames_queue = context.Queue(maxsize=8)
    stop_event = context.Event()

    reference_sequence_seed = (
        REFERENCE_SEQUENCE_SEED + int(10 * bandwidth_mhz)
    )
    known_ref_seq = generate_known_reference_sequence(
        params["N"],
        reference_sequence_seed,
    )
    # Synchronization and PDSCH positions are fixed by the frame layout, so a
    # single reference frame supplies all postprocessing workers with them.
    reference_frame = build_frame(params, known_ref_seq)

    hardware = context.Process(
        name="x310-single-usrp",
        target=capture_single_usrp,
        kwargs={
            "frames_queue": frames_queue,
            "stop_event": stop_event,
            "params": params,
            "known_ref_seq": known_ref_seq,
            "num_frames": num_captures,
            "frame_seed": frame_seed,
            "device_args": TX_USRP_DEVICE_ARGS,
            "tx_subdev_spec": TX_SUBDEV_SPEC,
            "rx_subdev_spec": RX_SUBDEV_SPEC,
            "tx_antenna": TX_ANTENNA,
            "rx_antenna": RX_ANTENNA,
            "tx_channels": TX_CHANNELS,
            "rx_channels": RX_CHANNELS,
            "tx_gain": Tx_gain,
            "rx_gain": Rx_gain,
            "otw_format": OTW_FORMAT,
        },
    )
    postprocessor = context.Process(
        name="csi-postprocessor",
        target=process_received_frames,
        kwargs={
            "frames_queue": frames_queue,
            "predictor_input_queue": None,
            "scheduler_input_queue": None,
            "stop_event": stop_event,
            "params": params,
            "ss_td_with_cp": reference_frame["ss_td_with_cp"],
            "pdsch_idx": reference_frame["pdsch_idx"],
            "known_ref_seq": known_ref_seq,
            "trailing_samples": int(
                params["sampling_rate"] * RX_TRAILING_TIME
            ),
            "max_csi_frames": num_captures,
            "capture_output_dir": args.output_dir,
            "reference_sequence_seed": reference_sequence_seed,
            "tx_frame_seed": frame_seed,
            "save_capture_plots": not args.no_plots,
            "window_length": args.window_length,
        },
    )
    processes = (postprocessor, hardware)
    started_processes = []
    interrupted = False

    try:
        for process in processes:
            process.start()
            started_processes.append(process)

        interrupted = _wait_for_processes(processes)
    finally:
        _stop_processes(started_processes, stop_event)
        for process_queue in (frames_queue,):
            process_queue.close()
            process_queue.join_thread()

    if interrupted:
        return

    failures = [
        f"{process.name} (exit code {process.exitcode})"
        for process in processes
        if process.exitcode != 0
    ]
    if failures:
        raise RuntimeError("Worker process failed: " + ", ".join(failures))


if __name__ == "__main__":
    freeze_support()
    main()
