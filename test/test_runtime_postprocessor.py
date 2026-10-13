import queue
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from USRP.build_frame import build_frame
from USRP.helper_functions import (
    generate_known_reference_sequence,
    get_system_params,
)
from runtime.postprocessor import process_received_frames
from runtime.receiver import RXFrame


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

        frames_queue = queue.Queue()
        for frame_id in range(6):
            frames_queue.put(
                RXFrame(
                    frame_id=frame_id,
                    samples=rx_frame,
                    first_sample_time=1.0 + frame_id * 0.005,
                    dropped_frames=0,
                )
            )

        predictor_input_queue = queue.Queue(maxsize=2)
        scheduler_input_queue = queue.Queue(maxsize=2)
        stop_event = threading.Event()
        process_received_frames(
            frames_queue=frames_queue,
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

    def test_saves_five_six_tx_captures_as_one_window(self):
        params = get_system_params(1.4)
        reference_seed = 2040
        known_ref_seq = generate_known_reference_sequence(
            params["N"],
            reference_seed,
        )
        frame = build_frame(params, known_ref_seq)
        rx_frame = np.sum(frame["waveform"], axis=0, keepdims=True)

        frames_queue = queue.Queue()
        for frame_id in range(6):
            frames_queue.put(
                RXFrame(
                    frame_id=frame_id,
                    samples=rx_frame,
                    first_sample_time=1.0 + frame_id * 0.005,
                    dropped_frames=0,
                )
            )

        stop_event = threading.Event()
        with tempfile.TemporaryDirectory() as temporary_directory:
            process_received_frames(
                frames_queue=frames_queue,
                stop_event=stop_event,
                params=params,
                ss_td_with_cp=frame["ss_td_with_cp"],
                pdsch_idx=frame["pdsch_idx"],
                known_ref_seq=known_ref_seq,
                max_csi_frames=5,
                trailing_samples=1920,
                capture_output_dir=temporary_directory,
                reference_sequence_seed=reference_seed,
                tx_frame_seed=1234,
                save_capture_plots=False,
                window_length=5,
            )

            output_dir = Path(temporary_directory)
            npz_paths = list(output_dir.glob("*.npz"))
            json_paths = list(output_dir.glob("*.json"))
            self.assertEqual(len(npz_paths), 1)
            self.assertEqual(len(json_paths), 1)
            self.assertIn('"tx_frame_seed": 1234', json_paths[0].read_text())
            with np.load(npz_paths[0], allow_pickle=False) as saved:
                self.assertEqual(saved["csi"].shape, (5, 6))
                self.assertEqual(saved["csi_windows"].shape, (1, 5, 6))
                np.testing.assert_array_equal(
                    saved["csi_windows"][0],
                    saved["csi"],
                )
                self.assertEqual(saved["rx_frame_ids"].tolist(), list(range(5)))
                self.assertEqual(saved["rx_dropped_frames"].tolist(), [0] * 5)
                self.assertTrue(
                    np.all(np.isfinite(saved["csi_timestamps_usrp_s"]))
                )
                self.assertTrue(
                    np.all(np.isfinite(saved["pilot_timestamps_usrp_s"]))
                )
            self.assertTrue(stop_event.is_set())


if __name__ == "__main__":
    unittest.main()
