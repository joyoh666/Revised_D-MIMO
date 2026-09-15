from data.schema import CSIFrame, CSIBlock
from phy.run_frame import run_frame
import time
from csi_buffer import CSIBuffer
import numpy as np

def csi_producer(csi_queue, block_queue):
    buffer = CSIBuffer(maxlen=5)
    period = 0.005
    next_time = time.perf_counter()
    prev_host_time = time.perf_counter()

    for frame_idx in range(1000):

        H_est = run_frame()

        host_time = time.perf_counter()
        producer_interval = host_time - prev_host_time
        prev_host_time = host_time
        buffer_ready = buffer.is_ready()

        frame = CSIFrame(
            frame_idx=frame_idx,
            host_time=host_time,
            usrp_time=None,
            csi=H_est,
            producer_interval=producer_interval,
            buffer_ready=False if not buffer.is_ready() else True
        )

        csi_queue.put(frame)

        if not buffer_ready:
            buffer.append(frame)
        else :
            window = buffer.get_window()
            block = CSIBlock(
                start_frame_idx=window[0].frame_idx,
                end_frame_idx=window[-1].frame_idx,
                start_time=window[0].host_time,
                end_time=window[-1].host_time,
                csi=np.stack([frame.csi for frame in window])
            )
            block_queue.put(block)
            buffer.clear()

        next_time += period
        remaining = next_time - time.perf_counter()

        if remaining > 0:
            time.sleep(remaining)
        else : 
            print(
                f"Frame {frame_idx}",
                f"overrun={-remaining * 1000:.3f} ms"
            )

    csi_queue.put(None)