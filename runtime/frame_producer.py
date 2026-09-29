from dataclasses import dataclass
from queue import Full

import numpy as np

from USRP.build_frame import build_frame


@dataclass
class TXFrame:
    frame_id: int
    samples: np.ndarray


def frame_producer(frame_queue, stop_event, params, known_ref_seq) -> None:
    frame_id = 0

    try:
        while not stop_event.is_set():
            frame_object = build_frame(params, known_ref_seq)
            tx_frame = TXFrame(
                frame_id=frame_id,
                samples=frame_object["waveform"],
            )

            while not stop_event.is_set():
                try:
                    frame_queue.put(tx_frame, timeout=0.2)
                except Full:
                    continue
                break
            else:
                break

            frame_id += 1
    except KeyboardInterrupt:
        stop_event.set()
    except Exception:
        stop_event.set()
        raise
    finally:
        # Do not wait for buffered frames after the transmitter has stopped.
        frame_queue.cancel_join_thread()
