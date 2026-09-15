import numpy as np

from runtime.csi_buffer import CSIBuffer
from data.schema import CSIFrame
from data.schema import CSIBlock


def test_csi_block():

    buffer = CSIBuffer(maxlen=5)

    for frame_idx in range(5):

        frame = CSIFrame(
            frame_idx=frame_idx,
            host_time=frame_idx * 0.005,
            usrp_time=None,
            csi=np.ones(6, dtype=np.complex64) * frame_idx
        )

        buffer.append(frame)

    assert buffer.is_ready()

    frames = buffer.get_window()

    block = CSIBlock(
        start_frame_idx=frames[0].frame_idx,
        end_frame_idx=frames[-1].frame_idx,
        start_time=frames[0].host_time,
        end_time=frames[-1].host_time,
        csi=np.stack([frame.csi for frame in frames])
    )

    print("start frame :", block.start_frame_idx)
    print("end frame   :", block.end_frame_idx)
    print("CSI shape   :", block.csi.shape)
    print(block.csi)

    assert block.start_frame_idx == 0
    assert block.end_frame_idx == 4

    # 5 samples × 6 CSI channels
    assert block.csi.shape == (5, 6)
    buffer.clear()

    assert buffer.get_len() == 0
    assert buffer.is_ready() is False

    print("CSIBlock test passed.")


if __name__ == "__main__":
    test_csi_block()