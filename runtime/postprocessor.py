import time
from pathlib import Path
from queue import Empty, Full

import numpy as np

from data.schema import CSIFrame
from USRP.postprocessing import postprocess
from USRP.results import CaptureAccumulator, plot_results, save_results

from .csi_buffer import ModelInputBuffers


def _validate_samples(frame, params) -> np.ndarray:
    samples = np.asarray(frame.samples, dtype=np.complex64)
    expected_shape = (
        int(params["num_rx_ant"]),
        int(params["frame_length"]),
    )
    if samples.shape != expected_shape:
        raise ValueError(
            f"RX frame {frame.frame_id} has shape {samples.shape}, "
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


def _csi_timestamps(frame, received, params) -> np.ndarray | None:
    if frame.first_sample_time is None:
        return None

    frame_start_time = (
        float(frame.first_sample_time)
        + float(received["sync_idx"]) / float(params["sampling_rate"])
    )
    return frame_start_time + (
        np.asarray(
            params["csi_sample_timestamp_offsets_samples"],
            dtype=np.float64,
        )
        / float(params["sampling_rate"])
    )


def _pilot_timestamps(frame, received, params) -> np.ndarray | None:
    """Return the USRP timestamp of every repeated sounding pilot."""
    if frame.first_sample_time is None:
        return None

    frame_start_time = (
        float(frame.first_sample_time)
        + float(received["sync_idx"]) / float(params["sampling_rate"])
    )
    return frame_start_time + (
        np.asarray(
            params["pilot_timestamp_offsets_samples"],
            dtype=np.float64,
        )
        / float(params["sampling_rate"])
    )


def build_csi_windows(
    csi,
    period_indices,
    timestamps_usrp_s,
    window_length,
) -> dict[str, np.ndarray]:
    """Build non-overlapping ``(window, sample, TX)`` CSI tensors.

    Incomplete tails are discarded independently for each continuous run, so
    a window never crosses a missing RX period.
    """
    if window_length <= 0 or window_length % 5:
        raise ValueError("window_length must be a positive multiple of 5")

    csi_array = np.asarray(csi, dtype=np.complex64)
    period_array = np.asarray(period_indices, dtype=np.int64)
    timestamp_array = np.asarray(timestamps_usrp_s, dtype=np.float64)
    if csi_array.ndim != 2:
        raise ValueError(f"csi must have shape (time, tx); got {csi_array.shape}")
    num_rows, num_tx = csi_array.shape
    if period_array.shape != (num_rows,):
        raise ValueError(
            "period_indices must have shape (time,); "
            f"got {period_array.shape}, expected ({num_rows},)"
        )
    if timestamp_array.shape != (num_rows,):
        raise ValueError(
            "timestamps_usrp_s must have shape (time,); "
            f"got {timestamp_array.shape}, expected ({num_rows},)"
        )

    empty_window_shape = (0, window_length)
    if num_rows == 0:
        return {
            "csi_windows": np.empty(
                (0, window_length, num_tx),
                dtype=np.complex64,
            ),
            "csi_window_dataset_row_indices": np.empty(
                empty_window_shape,
                dtype=np.int64,
            ),
            "csi_window_period_indices": np.empty(
                empty_window_shape,
                dtype=np.int64,
            ),
            "csi_window_timestamps_usrp_s": np.empty(
                empty_window_shape,
                dtype=np.float64,
            ),
        }

    run_starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(np.diff(period_array) != 1).astype(np.int64) + 1,
        )
    )
    run_ends = np.concatenate(
        (run_starts[1:], np.asarray([num_rows], dtype=np.int64))
    )
    window_rows: list[np.ndarray] = []
    for run_start, run_end in zip(run_starts, run_ends, strict=True):
        usable_end = int(run_start) + (
            (int(run_end) - int(run_start)) // window_length
        ) * window_length
        for start in range(int(run_start), usable_end, window_length):
            window_rows.append(
                np.arange(start, start + window_length, dtype=np.int64)
            )

    if window_rows:
        row_indices = np.stack(window_rows, axis=0)
    else:
        row_indices = np.empty(empty_window_shape, dtype=np.int64)

    return {
        "csi_windows": csi_array[row_indices],
        "csi_window_dataset_row_indices": row_indices,
        "csi_window_period_indices": period_array[row_indices],
        "csi_window_timestamps_usrp_s": timestamp_array[row_indices],
    }


def build_csi_windows_from_results(
    results,
    window_length,
) -> dict[str, np.ndarray]:
    """Create model windows from a finalized :class:`CaptureAccumulator`."""
    csi_by_rx = np.asarray(results["csi"], dtype=np.complex64)
    if csi_by_rx.ndim != 4:
        raise ValueError("results['csi'] must have shape (capture, csi, rx, tx)")
    captures, csi_samples, num_rx, num_tx = csi_by_rx.shape
    if num_rx < 1:
        raise ValueError("results['csi'] must contain at least one RX antenna")

    csi = csi_by_rx.reshape(-1, num_rx, num_tx)[:, 0]
    timestamps = np.asarray(
        results["csi_timestamps_usrp_s"],
        dtype=np.float64,
    ).reshape(-1)
    frame_ids = np.asarray(results["rx_frame_ids"], dtype=np.int64).reshape(-1)
    if frame_ids.size != captures:
        raise ValueError("rx_frame_ids must contain one value per capture")
    if frame_ids.size and np.all(frame_ids >= 0):
        period_indices = (
            (frame_ids[:, None] - frame_ids[0]) * csi_samples
            + np.arange(csi_samples, dtype=np.int64)[None, :]
        ).reshape(-1)
    else:
        period_indices = np.arange(csi.shape[0], dtype=np.int64)

    return build_csi_windows(
        csi,
        period_indices,
        timestamps,
        window_length,
    )


def process_received_frames(
    *,
    frames_queue,
    predictor_input_queue=None,
    scheduler_input_queue=None,
    stop_event,
    params,
    ss_td_with_cp,
    pdsch_idx,
    known_ref_seq,
    trailing_samples,
    max_csi_frames: int | None = None,
    capture_output_dir: str | Path | None = None,
    reference_sequence_seed: int | None = None,
    tx_frame_seed: int | None = None,
    save_capture_plots: bool = True,
    window_length: int = 5,
) -> None:
    """Estimate live CSI and optionally save a finite sounding dataset.

    One channel estimate is produced every 5 ms. The same estimate is sent to
    two independent buffers: five frames become a predictor token (25 ms), and
    twenty frames become one scheduler gain segment (100 ms). When
    ``capture_output_dir`` is provided, every accepted RX frame is also added
    to :class:`CaptureAccumulator` and written as NPZ/JSON artifacts when the
    requested number of CSI frames has been collected.
    """
    if max_csi_frames is not None and max_csi_frames <= 0:
        raise ValueError("max_csi_frames must be greater than zero")
    if capture_output_dir is not None and reference_sequence_seed is None:
        raise ValueError(
            "reference_sequence_seed is required when saving sounding data"
        )
    if window_length <= 0 or window_length % 5:
        raise ValueError("window_length must be a positive multiple of 5")
    if trailing_samples <= 0:
        raise ValueError("trailing_samples must be greater than zero")
    if int(params["num_rx_ant"]) != 1:
        raise ValueError(
            "The trained model input currently supports exactly one RX antenna"
        )

    frame_length = int(params["frame_length"])
    trailing_samples = min(int(trailing_samples), frame_length)
    num_csi_samples = int(params["num_csi_samples_per_frame"])
    model_outputs_enabled = (
        predictor_input_queue is not None or scheduler_input_queue is not None
    )
    model_buffers = (
        ModelInputBuffers(
            num_tx_ant=int(params["num_tx_ant"]),
            frame_period_s=float(params["csi_sample_period_s"]),
        )
        if model_outputs_enabled
        else None
    )
    capture_accumulator = (
        CaptureAccumulator(params) if capture_output_dir is not None else None
    )
    pending_frame = None
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
                rx_frame = frames_queue.get(timeout=0.2)
            except Empty:
                continue

            samples = _validate_samples(rx_frame, params)
            if pending_frame is None:
                pending_frame = rx_frame
                pending_samples = samples
                continue

            if rx_frame.frame_id != pending_frame.frame_id + 1:
                print(
                    "Skipping RX frame "
                    f"{pending_frame.frame_id}: next available frame is "
                    f"{rx_frame.frame_id}; clearing partial model windows"
                )
                if model_buffers is not None:
                    model_buffers.clear()
                pending_frame = rx_frame
                pending_samples = samples
                continue

            # postprocess() discards its first 10 samples. Prepending a guard
            # preserves the RX frame, while the next frame supplies the tail
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
                pending_frame,
                received,
                params,
            )
            pilot_timestamps = _pilot_timestamps(
                pending_frame,
                received,
                params,
            )
            if capture_accumulator is not None:
                capture_accumulator.add(
                    received,
                    csi_timestamps_usrp_s=usrp_timestamps,
                    pilot_timestamps_usrp_s=pilot_timestamps,
                    rx_frame_id=int(pending_frame.frame_id),
                    dropped_frames=int(pending_frame.dropped_frames),
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
                        int(pending_frame.frame_id) * num_csi_samples
                        + sample_idx
                    ),
                    host_time=host_time,
                    usrp_time=usrp_time,
                    csi=csi[sample_idx, :, 0],
                )
                if model_buffers is not None:
                    outputs = model_buffers.append(frame)
                    if outputs.reset_for_discontinuity:
                        print(
                            "CSI discontinuity detected at frame "
                            f"{frame.frame_idx}; restarted both model windows"
                        )
                    if (
                        outputs.predictor is not None
                        and predictor_input_queue is not None
                    ):
                        predictor_count += int(
                            _put_latest(
                                predictor_input_queue,
                                outputs.predictor,
                                "predictor",
                            )
                        )
                    if (
                        outputs.scheduler is not None
                        and scheduler_input_queue is not None
                    ):
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

            pending_frame = rx_frame
            pending_samples = samples

        if max_csi_frames is not None and csi_count >= max_csi_frames:
            stop_event.set()

        if capture_accumulator is not None and csi_count:
            results = capture_accumulator.finalize()
            results.update(
                build_csi_windows_from_results(results, window_length)
            )
            npz_path, json_path, figure_path, _ = save_results(
                capture_output_dir,
                params,
                known_ref_seq,
                results,
                reference_sequence_seed=reference_sequence_seed,
                tx_frame_seed=tx_frame_seed,
                window_length=window_length,
            )
            print(f"Saved channel sounding NPZ: {npz_path}")
            print(f"Saved channel sounding metadata: {json_path}")
            if save_capture_plots:
                plot_paths = plot_results(
                    figure_path,
                    params,
                    results,
                    show=False,
                )
                for name, path in plot_paths.items():
                    print(f"Saved channel sounding plot ({name}): {path}")
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
