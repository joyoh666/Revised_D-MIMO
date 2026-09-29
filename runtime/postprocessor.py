import time
from queue import Empty, Full

import numpy as np

from data.schema import CSIFrame
from USRP.postprocessing import postprocess

from .csi_buffer import ModelInputBuffers


def _validate_samples(block, params) -> np.ndarray:
    samples = np.asarray(block.samples, dtype=np.complex64)
    expected_shape = (
        int(params["num_rx_ant"]),
        int(params["frame_length"]),
    )
    if samples.shape != expected_shape:
        raise ValueError(
            f"RX block {block.block_id} has shape {samples.shape}, "
            f"expected {expected_shape}"
        )
    return samples


def _put_latest(model_queue, value, queue_name: str) -> bool:
    """Enqueue without letting slow inference stall the USRP pipeline.

    A full queue means that an older model input is already stale. Drop that
    input and keep the newest completed window instead.
    """
    try:
        model_queue.put_nowait(value)
        return True
    except Full:
        try:
            model_queue.get_nowait()
        except Empty:
            # multiprocessing.Queue can briefly report Full before its feeder
            # thread makes the oldest item visible to get_nowait().
            print(f"Dropped newest {queue_name} input: queue is busy")
            return False

    try:
        model_queue.put_nowait(value)
        print(f"Dropped stale {queue_name} input: inference is behind")
        return True
    except Full:
        print(f"Dropped newest {queue_name} input: queue remained full")
        return False


def _csi_timestamps(block, received, params) -> np.ndarray | None:
    if block.first_sample_time is None:
        return None

    frame_start_time = (
        float(block.first_sample_time)
        + float(received["sync_idx"]) / float(params["sampling_rate"])
    )
    return frame_start_time + (
        np.asarray(
            params["csi_sample_timestamp_offsets_samples"],
            dtype=np.float64,
        )
        / float(params["sampling_rate"])
    )


def process_received_blocks(
    *,
    samples_queue,
    predictor_input_queue,
    scheduler_input_queue,
    stop_event,
    params,
    ss_td_with_cp,
    pdsch_idx,
    known_ref_seq,
    trailing_samples,
    max_csi_frames: int | None = None,
) -> None:
    """Turn the live RX stream into non-overlapping model-input windows.

    One channel estimate is produced every 5 ms. The same estimate is sent to
    two independent buffers: five frames become a predictor token (25 ms), and
    twenty frames become one scheduler gain segment (100 ms). No NPZ or
    dataset metadata is created on this real-time path.
    """
    if max_csi_frames is not None and max_csi_frames <= 0:
        raise ValueError("max_csi_frames must be greater than zero")
    if trailing_samples <= 0:
        raise ValueError("trailing_samples must be greater than zero")
    if int(params["num_rx_ant"]) != 1:
        raise ValueError(
            "The trained model input currently supports exactly one RX antenna"
        )

    frame_length = int(params["frame_length"])
    trailing_samples = min(int(trailing_samples), frame_length)
    num_csi_samples = int(params["num_csi_samples_per_frame"])
    model_buffers = ModelInputBuffers(
        num_tx_ant=int(params["num_tx_ant"]),
        frame_period_s=float(params["csi_sample_period_s"]),
    )
    pending_block = None
    pending_samples = None
    csi_count = 0
    predictor_count = 0
    scheduler_count = 0
    failure = None

    try:
        while not stop_event.is_set() and (
            max_csi_frames is None or csi_count < max_csi_frames
        ):
            try:
                block = samples_queue.get(timeout=0.2)
            except Empty:
                continue

            samples = _validate_samples(block, params)
            if pending_block is None:
                pending_block = block
                pending_samples = samples
                continue

            if block.block_id != pending_block.block_id + 1:
                print(
                    "Skipping RX block "
                    f"{pending_block.block_id}: next available block is "
                    f"{block.block_id}; clearing partial model windows"
                )
                model_buffers.clear()
                pending_block = block
                pending_samples = samples
                continue

            # postprocess() discards its first 10 samples. Prepending a guard
            # preserves the RX block, while the next block supplies the tail
            # required for a positive timing offset.
            guard = np.zeros((samples.shape[0], 10), dtype=np.complex64)
            capture = np.concatenate(
                (guard, pending_samples, samples[:, :trailing_samples]),
                axis=1,
            )
            received = postprocess(
                params,
                ss_td_with_cp,
                pdsch_idx,
                known_ref_seq,
                capture,
            )
            csi = np.asarray(received["csi_scalar_raw"], dtype=np.complex64)
            expected_csi_shape = (
                num_csi_samples,
                int(params["num_tx_ant"]),
                1,
            )
            if csi.shape != expected_csi_shape:
                raise ValueError(
                    f"csi_scalar_raw has shape {csi.shape}, "
                    f"expected {expected_csi_shape}"
                )

            usrp_timestamps = _csi_timestamps(
                pending_block,
                received,
                params,
            )
            # Monotonic host time is suitable for end-to-end latency and
            # interval measurements; USRP time remains the RF-clock timestamp.
            host_time = time.perf_counter()
            for sample_idx in range(num_csi_samples):
                if max_csi_frames is not None and csi_count >= max_csi_frames:
                    break

                usrp_time = (
                    None
                    if usrp_timestamps is None
                    else float(usrp_timestamps[sample_idx])
                )
                frame = CSIFrame(
                    frame_idx=(
                        int(pending_block.block_id) * num_csi_samples
                        + sample_idx
                    ),
                    host_time=host_time,
                    usrp_time=usrp_time,
                    csi=csi[sample_idx, :, 0],
                )
                outputs = model_buffers.append(frame)
                if outputs.reset_for_discontinuity:
                    print(
                        "CSI discontinuity detected at frame "
                        f"{frame.frame_idx}; restarted both model windows"
                    )
                if outputs.predictor is not None:
                    predictor_count += int(
                        _put_latest(
                            predictor_input_queue,
                            outputs.predictor,
                            "predictor",
                        )
                    )
                if outputs.scheduler is not None:
                    scheduler_count += int(
                        _put_latest(
                            scheduler_input_queue,
                            outputs.scheduler,
                            "scheduler",
                        )
                    )
                csi_count += 1

            if csi_count and csi_count % 100 == 0:
                print(
                    f"Processed {csi_count} CSI frames; dispatched "
                    f"predictor={predictor_count}, scheduler={scheduler_count}"
                )

            pending_block = block
            pending_samples = samples

        if max_csi_frames is not None and csi_count >= max_csi_frames:
            stop_event.set()
    except KeyboardInterrupt:
        stop_event.set()
    except Exception as exc:
        failure = exc
        stop_event.set()

    print(
        "CSI postprocessor stopped: "
        f"frames={csi_count}, predictor_inputs={predictor_count}, "
        f"scheduler_inputs={scheduler_count}"
    )
    if failure is not None:
        raise failure
