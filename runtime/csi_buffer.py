from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from data.schema import (
    CSIFrame,
    PredictorModelInput,
    SchedulerModelInput,
    WindowMetadata,
)


PREDICTOR_WINDOW_FRAMES = 5
SCHEDULER_WINDOW_FRAMES = 20
PREDICTOR_GAIN_SCALE = 20.0
SCHEDULER_HISTORY_SEGMENTS = 3
SCHEDULER_STATE_RUS = {
    0: (0,),
    1: (1,),
    2: (2,),
    3: (0, 1),
    4: (0, 2),
    5: (1, 2),
}


class CSIBuffer:
    """Fixed-size, non-overlapping buffer of consecutive CSI frames."""

    def __init__(self, maxlen: int):
        if int(maxlen) <= 0:
            raise ValueError("maxlen must be greater than zero")
        self.maxlen = int(maxlen)
        self._frames: deque[CSIFrame] = deque(maxlen=self.maxlen)

    def append(self, frame: CSIFrame) -> None:
        self._frames.append(frame)

    def is_ready(self) -> bool:
        return len(self._frames) == self.maxlen

    def get_window(self) -> list[CSIFrame]:
        return list(self._frames)

    def get_len(self) -> int:
        return len(self._frames)

    def clear(self) -> None:
        self._frames.clear()


def _validate_frame(frame: CSIFrame, num_tx_ant: int) -> CSIFrame:
    csi = np.asarray(frame.csi, dtype=np.complex64)
    if csi.shape != (num_tx_ant,):
        raise ValueError(
            f"CSI frame has shape {csi.shape}, expected {(num_tx_ant,)}"
        )
    if not np.all(np.isfinite(csi.real)) or not np.all(np.isfinite(csi.imag)):
        raise ValueError("CSI frame contains a non-finite value")

    return CSIFrame(
        frame_idx=int(frame.frame_idx),
        host_time=float(frame.host_time),
        usrp_time=(
            None if frame.usrp_time is None else float(frame.usrp_time)
        ),
        csi=np.array(csi, copy=True),
    )


def _window_metadata(frames: list[CSIFrame]) -> WindowMetadata:
    first = frames[0]
    last = frames[-1]
    return WindowMetadata(
        start_frame_idx=first.frame_idx,
        end_frame_idx=last.frame_idx,
        start_host_time=first.host_time,
        end_host_time=last.host_time,
        start_usrp_time=first.usrp_time,
        end_usrp_time=last.usrp_time,
    )


def predictor_tokens(
    csi_window: np.ndarray,
    *,
    gain_scale: float = PREDICTOR_GAIN_SCALE,
) -> np.ndarray:
    """Encode a ``(5, 6)`` complex window exactly like predictor training."""
    csi_window = np.asarray(csi_window, dtype=np.complex64)
    expected_shape = (PREDICTOR_WINDOW_FRAMES, 6)
    if csi_window.shape != expected_shape:
        raise ValueError(
            f"csi_window has shape {csi_window.shape}, "
            f"expected {expected_shape}"
        )

    # Training treats each TX stream as one batch item and reshapes every
    # five 5-ms samples into a single token.
    h_taps = np.transpose(csi_window, (1, 0))
    magnitude = np.abs(h_taps) * float(gain_scale)
    phase = np.angle(h_taps)
    return np.concatenate(
        (magnitude, np.cos(phase), np.sin(phase)),
        axis=-1,
    ).astype(np.float32)


def scheduler_ru_gains(csi_window: np.ndarray) -> np.ndarray:
    """Reduce a ``(20, 6)`` CSI window using the training environment rule."""
    csi_window = np.asarray(csi_window, dtype=np.complex64)
    expected_shape = (SCHEDULER_WINDOW_FRAMES, 6)
    if csi_window.shape != expected_shape:
        raise ValueError(
            f"csi_window has shape {csi_window.shape}, "
            f"expected {expected_shape}"
        )

    gains = np.abs(csi_window).astype(np.float32)
    return np.array(
        [
            np.sqrt(gains[:, 0] ** 2 + gains[:, 1] ** 2).mean(),
            np.sqrt(gains[:, 2] ** 2 + gains[:, 3] ** 2).mean(),
            np.sqrt(gains[:, 4] ** 2 + gains[:, 5] ** 2).mean(),
        ],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class BufferOutputs:
    predictor: PredictorModelInput | None
    scheduler: SchedulerModelInput | None
    reset_for_discontinuity: bool


class ModelInputBuffers:
    """Fan one 5 ms CSI stream out to 25 ms and 100 ms model windows."""

    def __init__(
        self,
        *,
        num_tx_ant: int = 6,
        frame_period_s: float = 0.005,
        timestamp_tolerance_s: float = 0.002,
        predictor_window_frames: int = PREDICTOR_WINDOW_FRAMES,
        scheduler_window_frames: int = SCHEDULER_WINDOW_FRAMES,
        predictor_gain_scale: float = PREDICTOR_GAIN_SCALE,
    ):
        if num_tx_ant != 6:
            raise ValueError(
                "The current predictor/scheduler checkpoints require 6 TX CSI"
            )
        if frame_period_s <= 0:
            raise ValueError("frame_period_s must be greater than zero")
        if timestamp_tolerance_s < 0:
            raise ValueError("timestamp_tolerance_s must be non-negative")
        if predictor_window_frames != PREDICTOR_WINDOW_FRAMES:
            raise ValueError("The predictor checkpoint requires 5 CSI frames")
        if scheduler_window_frames != SCHEDULER_WINDOW_FRAMES:
            raise ValueError("The scheduler checkpoint requires 20 CSI frames")

        self.num_tx_ant = int(num_tx_ant)
        self.frame_period_s = float(frame_period_s)
        self.timestamp_tolerance_s = float(timestamp_tolerance_s)
        self.predictor_gain_scale = float(predictor_gain_scale)
        self.predictor_buffer = CSIBuffer(predictor_window_frames)
        self.scheduler_buffer = CSIBuffer(scheduler_window_frames)
        self._last_frame: CSIFrame | None = None

    def clear(self) -> None:
        self.predictor_buffer.clear()
        self.scheduler_buffer.clear()
        self._last_frame = None

    def _is_contiguous(self, frame: CSIFrame) -> bool:
        previous = self._last_frame
        if previous is None:
            return True
        if frame.frame_idx != previous.frame_idx + 1:
            return False
        if previous.usrp_time is None or frame.usrp_time is None:
            return True
        measured_period = frame.usrp_time - previous.usrp_time
        return (
            abs(measured_period - self.frame_period_s)
            <= self.timestamp_tolerance_s
        )

    def append(self, frame: CSIFrame) -> BufferOutputs:
        frame = _validate_frame(frame, self.num_tx_ant)
        reset_for_discontinuity = not self._is_contiguous(frame)
        if reset_for_discontinuity:
            self.clear()

        self.predictor_buffer.append(frame)
        self.scheduler_buffer.append(frame)
        self._last_frame = frame

        predictor_input = None
        if self.predictor_buffer.is_ready():
            frames = self.predictor_buffer.get_window()
            csi = np.stack([item.csi for item in frames], axis=0)
            predictor_input = PredictorModelInput(
                metadata=_window_metadata(frames),
                tokens=predictor_tokens(
                    csi,
                    gain_scale=self.predictor_gain_scale,
                ),
            )
            self.predictor_buffer.clear()

        scheduler_input = None
        if self.scheduler_buffer.is_ready():
            frames = self.scheduler_buffer.get_window()
            csi = np.stack([item.csi for item in frames], axis=0)
            scheduler_input = SchedulerModelInput(
                metadata=_window_metadata(frames),
                ru_gains=scheduler_ru_gains(csi),
            )
            self.scheduler_buffer.clear()

        return BufferOutputs(
            predictor=predictor_input,
            scheduler=scheduler_input,
            reset_for_discontinuity=reset_for_discontinuity,
        )


class SchedulerObservationBuilder:
    """Add scheduler state/history to 100 ms CSI-derived gain segments.

    CSI collection cannot create the final policy observation by itself: the
    masking depends on which RU state was connected while each segment was
    observed. The scheduler worker owns this small stateful adapter.
    """

    def __init__(
        self,
        *,
        initial_connection_state: int = 0,
        history_segments: int = SCHEDULER_HISTORY_SEGMENTS,
        mask_value: float = 0.0,
    ):
        if history_segments <= 0:
            raise ValueError("history_segments must be greater than zero")
        self.history_segments = int(history_segments)
        self.mask_value = float(mask_value)
        # Keep one extra (current) segment. After initialization the training
        # environment builds an observation from the segments preceding the
        # interval being scheduled.
        self._history: deque[tuple[np.ndarray, int]] = deque(
            maxlen=self.history_segments + 1
        )
        self.connection_state = 0
        self.set_connection_state(initial_connection_state)

    def set_connection_state(self, connection_state: int) -> None:
        connection_state = int(connection_state)
        if connection_state not in SCHEDULER_STATE_RUS:
            raise ValueError("connection_state must be in [0, 5]")
        self.connection_state = connection_state

    def clear(self) -> None:
        self._history.clear()

    def append(self, segment: SchedulerModelInput) -> np.ndarray:
        ru_gains = np.asarray(segment.ru_gains, dtype=np.float32)
        if ru_gains.shape != (3,):
            raise ValueError("scheduler ru_gains must have shape (3,)")
        if not np.all(np.isfinite(ru_gains)):
            raise ValueError("scheduler ru_gains contain a non-finite value")

        self._history.append(
            (np.array(ru_gains, copy=True), self.connection_state)
        )
        all_segments = list(self._history)
        history = (
            all_segments
            if len(all_segments) == 1
            else all_segments[:-1]
        )
        history = history[-self.history_segments:]
        if len(history) < self.history_segments:
            padding = [history[0]] * (
                self.history_segments - len(history)
            )
            history = padding + history

        masked = np.full(
            (3, self.history_segments),
            self.mask_value,
            dtype=np.float32,
        )
        for history_idx, (gains, state) in enumerate(history):
            for ru_idx in SCHEDULER_STATE_RUS[state]:
                masked[ru_idx, history_idx] = gains[ru_idx]

        connection_one_hot = np.zeros((6,), dtype=np.float32)
        connection_one_hot[self.connection_state] = 1.0
        return np.concatenate(
            (masked.reshape(-1), connection_one_hot),
            axis=0,
        ).astype(np.float32)
