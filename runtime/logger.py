def result_logger(result_queue):
    while True:
        result = result_queue.get()

        if result is None:
            break

        print(
            f"frame={result['frame_idx']}, "
            f"CSI shape={result['csi'].shape}, "
            f"queue delay={result['queue_delay'] * 1000:.3f} ms",
            f"producer interval={result['producer_interval'] * 1000:.3f} ms",
            f"buffer ready={result['buffer_ready']}"
        )