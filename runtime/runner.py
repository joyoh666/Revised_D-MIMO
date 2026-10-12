import argparse
from multiprocessing import freeze_support, get_context
from pathlib import Path
from typing import Sequence

from config.radio_config import (
    BANDWIDTH,
    CSI_SEQUENCE_LENGTH,
    CLOCK_SOURCE,
    NUM_CHANNEL_CAPTURES,
    OTW_FORMAT,
    OUTPUT_DIR,
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
from USRP.build_frame import build_frame
from USRP.helper_functions import (
    generate_known_reference_sequence,
    get_system_params,
)

from .frame_producer import frame_producer
from .postprocessor import process_received_blocks
from .receiver import receive_blocks
from .transmitter import transmit_frames


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
        description="Collect and save simultaneous 6-TX/1-RX channel sounding CSI."
    )
    parser.add_argument(
        "--captures",
        type=int,
        default=NUM_CHANNEL_CAPTURES,
        help="Number of 5 ms CSI snapshots to save.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(OUTPUT_DIR),
        help="Directory for NPZ, JSON, and plot artifacts.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=CSI_SEQUENCE_LENGTH,
        help="CSI samples per sequence; must be a positive multiple of 5.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Save NPZ and JSON only.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.captures <= 0:
        raise ValueError("--captures must be greater than zero")
    if args.sequence_length <= 0 or args.sequence_length % 5:
        raise ValueError("--sequence-length must be a positive multiple of 5")
    if args.captures < args.sequence_length:
        raise ValueError("--captures must be at least --sequence-length")

    # Use spawn so that UHD is initialized only in the hardware child
    # processes. TX and RX rendezvous before scheduling a common start time.
    context = get_context("spawn")
    frame_queue = context.Queue(maxsize=2)
    samples_queue = context.Queue(maxsize=8)
    start_time_queue = context.Queue(maxsize=1)
    stop_event = context.Event()
    receiver_ready_event = context.Event()
    sync_barrier = context.Barrier(2)

    bandwidth_mhz = BANDWIDTH
    params = get_system_params(bandwidth_mhz)
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

    producer = context.Process(
        name="frame-producer",
        target=frame_producer,
        args=(frame_queue, stop_event, params, known_ref_seq),
    )
    receiver = context.Process(
        name="usrp-receiver",
        target=receive_blocks,
        kwargs={
            "samples_queue": samples_queue,
            "stop_event": stop_event,
            "device_args": RX_USRP_DEVICE_ARGS,
            "rx_subdev_spec": RX_SUBDEV_SPEC,
            "rx_antenna": RX_ANTENNA,
            "num_samples": params["frame_length"],
            "carrier_frequency": params["carrier_frequency"],
            "sampling_rate": params["sampling_rate"],
            "gain": Rx_gain,
            "channels": RX_CHANNELS,
            "start_time": None,
            "otw_format": OTW_FORMAT,
            "clock_source": CLOCK_SOURCE,
            "time_source": TIME_SOURCE,
            "sync_barrier": sync_barrier,
            "receiver_ready_event": receiver_ready_event,
            "start_time_queue": start_time_queue,
        },
    )
    transmitter = context.Process(
        name="usrp-transmitter",
        target=transmit_frames,
        kwargs={
            "frame_queue": frame_queue,
            "stop_event": stop_event,
            "device_args": TX_USRP_DEVICE_ARGS,
            "tx_subdev_spec": TX_SUBDEV_SPEC,
            "tx_antenna": TX_ANTENNA,
            "carrier_frequency": params["carrier_frequency"],
            "sampling_rate": params["sampling_rate"],
            "gain": Tx_gain,
            "channels": TX_CHANNELS,
            "otw_format": OTW_FORMAT,
            "transmission_delay": WAIT_TIME,
            "clock_source": CLOCK_SOURCE,
            "time_source": TIME_SOURCE,
            "sync_barrier": sync_barrier,
            "receiver_ready_event": receiver_ready_event,
            "start_time_queue": start_time_queue,
        },
    )
    postprocessor = context.Process(
        name="csi-postprocessor",
        target=process_received_blocks,
        kwargs={
            "samples_queue": samples_queue,
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
            "max_csi_frames": args.captures,
            "capture_output_dir": args.output_dir,
            "reference_sequence_seed": reference_sequence_seed,
            "save_capture_plots": not args.no_plots,
            "sequence_length": args.sequence_length,
        },
    )
    processes = (postprocessor, producer, receiver, transmitter)
    started_processes = []
    interrupted = False

    try:
        for process in processes:
            process.start()
            started_processes.append(process)

        interrupted = _wait_for_processes(processes)
    finally:
        _stop_processes(started_processes, stop_event)
        for process_queue in (
            frame_queue,
            samples_queue,
            start_time_queue,
        ):
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
