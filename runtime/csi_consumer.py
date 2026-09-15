from collections import deque
import time

def csi_consumer(csi_queue, result_queue):
    buffer = deque(maxlen=5)

    while True:
        frame = csi_queue.get()

        if frame is None:
            break

        receive_time = time.perf_counter()

        frame_idx = frame.frame_idx
        H_est = frame.csi
        queue_delay = receive_time - frame.host_time

        if len(buffer) < buffer.maxlen:
            buffer.append(frame)

        result = {
            "frame_idx": frame_idx,
            "csi": H_est,
            "queue_delay": queue_delay,
            "producer_interval": frame.producer_interval,
            "buffer_ready": frame.buffer_ready
        }

        result_queue.put(result)

    result_queue.put(None)