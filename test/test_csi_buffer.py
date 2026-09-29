import numpy as np

from data.schema import (
    CSIFrame,
    SchedulerModelInput,
    WindowMetadata,
)
from runtime.csi_buffer import (
    CSIBuffer,
    ModelInputBuffers,
    SchedulerObservationBuilder,
)


def _frame(frame_idx: int, csi: np.ndarray) -> CSIFrame:
    return CSIFrame(
        frame_idx=frame_idx,
        host_time=100.0 + frame_idx * 0.005,
        usrp_time=10.0 + frame_idx * 0.005,
        csi=np.asarray(csi, dtype=np.complex64),
    )


def test_buffer():
    buffer = CSIBuffer(maxlen=5)
    for frame_idx in range(5):
        buffer.append(
            _frame(
                frame_idx,
                np.ones(6, dtype=np.complex64) * frame_idx,
            )
        )

    assert buffer.get_len() == 5
    assert buffer.is_ready() is True


def test_two_model_buffers_emit_at_different_periods():
    buffers = ModelInputBuffers()
    csi = np.array([1, 2, 3, 4, 5, 6], dtype=np.complex64)
    predictor_outputs = []
    scheduler_outputs = []

    for frame_idx in range(20):
        outputs = buffers.append(_frame(frame_idx, csi))
        if outputs.predictor is not None:
            predictor_outputs.append(outputs.predictor)
        if outputs.scheduler is not None:
            scheduler_outputs.append(outputs.scheduler)

    assert len(predictor_outputs) == 4
    assert len(scheduler_outputs) == 1

    first_predictor = predictor_outputs[0]
    assert first_predictor.tokens.shape == (6, 15)
    np.testing.assert_allclose(first_predictor.tokens[0, :5], 20.0)
    np.testing.assert_allclose(first_predictor.tokens[0, 5:10], 1.0)
    np.testing.assert_allclose(first_predictor.tokens[0, 10:15], 0.0)
    assert first_predictor.metadata.start_frame_idx == 0
    assert first_predictor.metadata.end_frame_idx == 4

    scheduler = scheduler_outputs[0]
    np.testing.assert_allclose(
        scheduler.ru_gains,
        [np.sqrt(5.0), 5.0, np.sqrt(61.0)],
        rtol=1e-6,
    )
    assert scheduler.metadata.start_frame_idx == 0
    assert scheduler.metadata.end_frame_idx == 19


def test_gap_clears_both_partial_windows():
    buffers = ModelInputBuffers()
    csi = np.ones(6, dtype=np.complex64)
    for frame_idx in range(3):
        buffers.append(_frame(frame_idx, csi))

    outputs = buffers.append(_frame(4, csi))

    assert outputs.reset_for_discontinuity is True
    assert buffers.predictor_buffer.get_len() == 1
    assert buffers.scheduler_buffer.get_len() == 1


def test_scheduler_observation_adds_masked_history_and_state():
    metadata = WindowMetadata(0, 19, 0.0, 0.095, 1.0, 1.095)
    builder = SchedulerObservationBuilder(initial_connection_state=3)
    observation = builder.append(
        SchedulerModelInput(
            metadata=metadata,
            ru_gains=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        )
    )

    assert observation.shape == (15,)
    np.testing.assert_array_equal(
        observation[:9],
        [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0],
    )
    np.testing.assert_array_equal(observation[9:], [0, 0, 0, 1, 0, 0])

    builder.set_connection_state(2)
    next_observation = builder.append(
        SchedulerModelInput(
            metadata=WindowMetadata(20, 39, 0.1, 0.195, 1.1, 1.195),
            ru_gains=np.array([4.0, 5.0, 6.0], dtype=np.float32),
        )
    )
    # As in the training environment, the next decision still observes the
    # completed previous segment; the newest segment becomes history after it.
    np.testing.assert_array_equal(
        next_observation[:9],
        [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0],
    )
    np.testing.assert_array_equal(next_observation[9:], [0, 0, 1, 0, 0, 0])
