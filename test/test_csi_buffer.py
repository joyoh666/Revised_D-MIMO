import numpy as np

from runtime.csi_buffer import CSIBuffer
from data.schema import CSIFrame


def test_buffer():

    buffer = CSIBuffer(maxlen=5)

    # 가짜 CSI 5개 생성
    for frame_idx in range(5):

        frame = CSIFrame(
            frame_idx=frame_idx,
            host_time=frame_idx * 0.005,
            usrp_time=None,
            csi=np.ones(6, dtype=np.complex64) * frame_idx,
            producer_interval=None,
            buffer_ready=None
        )

        buffer.append(frame)

        print(
            f"frame={frame_idx}, "
            f"buffer_size={buffer.get_len()}, "
            f"ready={buffer.is_ready()}"
        )

    assert buffer.get_len() == 5
    assert buffer.is_ready() is True

    print("Buffer test passed.")


if __name__ == "__main__":
    test_buffer()