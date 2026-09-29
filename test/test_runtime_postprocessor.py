import queue
import threading
import unittest

import numpy as np

from USRP.build_frame import build_frame
from USRP.helper_functions import (
    generate_known_reference_sequence,
    get_system_params,
)
from runtime.postprocessor import process_received_blocks
from runtime.receiver import RXBlock


class RuntimePostprocessorTests(unittest.TestCase):
    def test_dispatches_five_csi_frames_as_one_predictor_input(self):
        params = get_system_params(1.4)
        reference_seed = 2040
        known_ref_seq = generate_known_reference_sequence(
            params["N"],
            reference_seed,
        )
        frame = build_frame(params, known_ref_seq)
        rx_frame = np.sum(frame["waveform"], axis=0, keepdims=True)

        samples_queue = queue.Queue()
        for block_id in range(6):
            samples_queue.put(
                RXBlock(
                    block_id=block_id,
                    samples=rx_frame,
                    first_sample_time=1.0 + block_id * 0.005,
                    dropped_blocks=0,
                )
            )

        predictor_input_queue = queue.Queue(maxsize=2)
        scheduler_input_queue = queue.Queue(maxsize=2)
        stop_event = threading.Event()
        process_received_blocks(
            samples_queue=samples_queue,
            predictor_input_queue=predictor_input_queue,
            scheduler_input_queue=scheduler_input_queue,
            stop_event=stop_event,
            params=params,
            ss_td_with_cp=frame["ss_td_with_cp"],
            pdsch_idx=frame["pdsch_idx"],
            known_ref_seq=known_ref_seq,
            max_csi_frames=5,
            trailing_samples=1920,
        )

        predictor_input = predictor_input_queue.get_nowait()
        self.assertEqual(predictor_input.tokens.shape, (6, 15))
        self.assertEqual(predictor_input.metadata.start_frame_idx, 0)
        self.assertEqual(predictor_input.metadata.end_frame_idx, 4)
        self.assertAlmostEqual(
            predictor_input.metadata.start_usrp_time,
            1.0
            + params["csi_sample_timestamp_offsets_samples"][0]
            / params["sampling_rate"],
        )
        self.assertTrue(scheduler_input_queue.empty())
        self.assertTrue(stop_event.is_set())


if __name__ == "__main__":
    unittest.main()
