import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from USRP.build_frame import build_frame
from USRP.helper_functions import (
    generate_known_reference_sequence,
    get_system_params,
)
from USRP.postprocessing import postprocess
from USRP.results import (
    CaptureAccumulator,
    _build_csi_sequences,
    plot_results,
    save_results,
)


class ResultsSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = get_system_params(1.4)
        cls.reference_seed = 2041
        cls.known_ref_seq = generate_known_reference_sequence(
            cls.params["N"],
            seed=cls.reference_seed,
        )
        cls.frame = build_frame(cls.params, cls.known_ref_seq)

    def _synthetic_received(self):
        tx_gains = np.exp(
            1j * 0.2 * np.arange(self.params["num_tx_ant"])
        ).astype(np.complex64)
        composite_waveform = np.sum(
            tx_gains[:, None] * self.frame["waveform"],
            axis=0,
        )
        timing_offset = 23
        capture = np.concatenate(
            (
                np.zeros(10 + timing_offset, dtype=np.complex64),
                composite_waveform,
                np.zeros(128, dtype=np.complex64),
            )
        )[None, :]
        return postprocess(
            self.params,
            self.frame["ss_td_with_cp"],
            self.frame["pdsch_idx"],
            self.known_ref_seq,
            capture,
        )

    def test_save_and_plot_preserve_capture_csi_rx_tx_axes(self):
        accumulator = CaptureAccumulator(self.params)
        received = self._synthetic_received()
        for block_id in range(5):
            accumulator.add(received, rx_block_id=block_id)
        results = accumulator.finalize()

        captures = 5
        csi_samples = self.params["num_csi_samples_per_frame"]
        num_rx = self.params["num_rx_ant"]
        num_tx = self.params["num_tx_ant"]
        num_subcarriers = self.params["N"]
        num_pilots = self.params["num_virtual_pilots"]
        fft_size = self.params["FFT_SIZE"]
        expected_grid_shape = (
            captures,
            num_rx,
            num_tx,
            self.params["num_subframe_per_frame"],
            self.params["num_slot_per_subframe"],
            num_subcarriers,
        )
        expected_csi_shape = (
            captures,
            csi_samples,
            num_rx,
            num_tx,
            num_subcarriers,
        )

        self.assertEqual(
            results["channel_estimates_fd"].shape,
            expected_grid_shape,
        )
        self.assertEqual(results["channel_mean_fd"].shape, expected_csi_shape)
        np.testing.assert_allclose(
            results["csi"],
            np.mean(results["channel_mean_fd"], axis=-1),
        )
        self.assertEqual(
            results["channel_estimates_fd_virtual_pilots"].shape,
            (
                captures,
                csi_samples,
                num_rx,
                num_tx,
                num_pilots,
                num_subcarriers,
            ),
        )
        self.assertEqual(
            results["channel_impulse_response_td"].shape,
            expected_csi_shape[:-1] + (fft_size,),
        )
        self.assertEqual(
            results["virtual_pilot_common_phases_rad"].shape,
            expected_csi_shape[:-1] + (num_pilots,),
        )
        self.assertEqual(
            results["virtual_pilot_snr_db"].shape,
            expected_csi_shape[:-1],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            npz_path, json_path, figure_path, metadata = save_results(
                Path(temporary_directory),
                self.params,
                self.known_ref_seq,
                results,
                reference_sequence_seed=self.reference_seed,
                timestamp="20260925_000000",
            )
            plot_paths = plot_results(
                figure_path,
                self.params,
                results,
                show=False,
            )

            self.assertEqual(metadata["result_schema_version"], 5)
            self.assertEqual(metadata["pilot_multiplexing"], "fdm")
            self.assertEqual(metadata["num_transmit_antennas"], num_tx)
            self.assertEqual(metadata["num_csi_samples_per_frame"], csi_samples)
            self.assertEqual(metadata["csi_sample_period_ms"], 5.0)
            self.assertEqual(metadata["csi_sequence_length"], 5)
            self.assertEqual(metadata["num_csi_sequences"], 1)
            self.assertEqual(metadata["num_csi_samples_used_in_sequences"], 5)
            self.assertEqual(
                metadata["num_csi_samples_discarded_from_sequences"],
                0,
            )
            self.assertEqual(
                metadata["saved_csi_channel_shape"],
                list(expected_csi_shape),
            )
            self.assertEqual(
                metadata["array_axis_order"]["channel_mean_fd"],
                [
                    "capture",
                    "csi_sample",
                    "rx_antenna",
                    "tx_antenna",
                    "subcarrier",
                ],
            )

            saved_metadata = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_metadata, metadata)
            with np.load(npz_path, allow_pickle=False) as dataset:
                self.assertEqual(dataset["csi"].shape, (captures, num_tx))
                self.assertEqual(dataset["csi_sequences"].shape, (1, 5, num_tx))
                np.testing.assert_array_equal(
                    dataset["csi_sequences"][0],
                    dataset["csi"],
                )
                self.assertEqual(
                    dataset["csi_repetitions"].shape,
                    (captures, num_pilots, num_tx),
                )
                self.assertEqual(
                    dataset["csi_repetitions_aligned"].shape,
                    (captures, num_pilots, num_tx),
                )
                self.assertEqual(
                    dataset["pilot_received"].shape,
                    (captures, num_pilots, num_tx),
                )
                self.assertEqual(
                    dataset["pilot_snr_db"].shape,
                    (captures, num_pilots, num_tx),
                )
                self.assertEqual(
                    dataset["pilot_centered_bins"].tolist(),
                    [-30, -18, -6, 6, 18, 30],
                )
                self.assertEqual(
                    dataset["channel_mean_fd"].shape,
                    expected_csi_shape,
                )
                self.assertEqual(
                    dataset["channel_estimates_fd_virtual_pilots"].shape,
                    (
                        captures,
                        csi_samples,
                        num_rx,
                        num_tx,
                        num_pilots,
                        num_subcarriers,
                    ),
                )
                self.assertEqual(
                    dataset["example_iq_rcv"].shape,
                    results["example_iq_rcv"].shape,
                )
                self.assertEqual(
                    dataset["example_frame_rcv"].shape,
                    results["example_frame_rcv"].shape,
                )
                embedded_metadata = json.loads(dataset["metadata_json"].item())
                self.assertEqual(embedded_metadata, metadata)

            for plot_path in plot_paths.values():
                self.assertTrue(plot_path.is_file())
                self.assertGreater(plot_path.stat().st_size, 0)

    def test_sequence_builder_does_not_cross_period_gaps(self):
        csi = np.arange(19 * 6, dtype=np.float32).reshape(19, 6).astype(
            np.complex64
        )
        period_indices = np.concatenate(
            (np.arange(7, dtype=np.int64), np.arange(10, 22, dtype=np.int64))
        )
        timestamps = period_indices.astype(np.float64) * 0.005

        sequences = _build_csi_sequences(
            csi,
            period_indices,
            timestamps,
            sequence_length=5,
        )

        self.assertEqual(sequences["csi_sequences"].shape, (3, 5, 6))
        np.testing.assert_array_equal(
            sequences["csi_sequence_dataset_row_indices"],
            np.asarray(
                [
                    [0, 1, 2, 3, 4],
                    [7, 8, 9, 10, 11],
                    [12, 13, 14, 15, 16],
                ]
            ),
        )
        self.assertTrue(
            np.all(
                np.diff(sequences["csi_sequence_period_indices"], axis=1)
                == 1
            )
        )

    def test_sequence_length_must_be_a_positive_multiple_of_five(self):
        csi = np.empty((0, 6), dtype=np.complex64)
        indices = np.empty((0,), dtype=np.int64)
        timestamps = np.empty((0,), dtype=np.float64)
        for invalid_length in (0, -5, 6):
            with self.subTest(sequence_length=invalid_length):
                with self.assertRaises(ValueError):
                    _build_csi_sequences(
                        csi,
                        indices,
                        timestamps,
                        invalid_length,
                    )

if __name__ == "__main__":
    unittest.main()
