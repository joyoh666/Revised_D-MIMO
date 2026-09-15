import numpy as np

from runtime.csi_buffer import CSIBuffer
from data.schema import CSIFrame
from data.schema import CSIBlock

def test_multiple_blocks():

    buffer = CSIBuffer(maxlen=5)

    blocks = []

    for frame_idx in range(10):

        frame = CSIFrame(
            frame_idx=frame_idx,
            host_time=frame_idx * 0.005,
            usrp_time=None,
            csi=np.ones(6, dtype=np.complex64) * frame_idx
        )

        buffer.append(frame)

        if buffer.is_ready():

            frames = buffer.get_window()

            block = CSIBlock(
                start_frame_idx=frames[0].frame_idx,
                end_frame_idx=frames[-1].frame_idx,
                start_time=frames[0].host_time,
                end_time=frames[-1].host_time,
                csi=np.stack([f.csi for f in frames])
            )

            blocks.append(block)

            buffer.clear()

    assert len(blocks) == 2

    assert blocks[0].start_frame_idx == 0
    assert blocks[0].end_frame_idx == 4

    assert blocks[1].start_frame_idx == 5
    assert blocks[1].end_frame_idx == 9

    print(
        f"Block 0: "
        f"{blocks[0].start_frame_idx}~"
        f"{blocks[0].end_frame_idx}"
    )

    print(
        f"Block 1: "
        f"{blocks[1].start_frame_idx}~"
        f"{blocks[1].end_frame_idx}"
    )

    print("Multiple block test passed.")

if __name__ == "__main__":
    test_multiple_blocks()