import tempfile
import unittest
from pathlib import Path

import numpy as np

from USRP.helper_functions import (
    generate_known_reference_sequence,
    get_system_params,
)
from USRP.ofdm import ofdm_demodulate
from runtime.mrt_transceiver import (
    build_single_stream_mrt_batch,
    calculate_link_metrics,
    equalize_common_pdsch,
    hard_decision,
    plot_constellation_comparison,
    synchronize_and_demodulate,
)


class MRTTransceiverTests(unittest.TestCase):
    def setUp(self):
        self.params = get_system_params(1.4)
        self.known_ref_seq = generate_known_reference_sequence(
            self.params["N"],
            2040,
        )
        num_tx = int(self.params["num_tx_ant"])
        self.channel = np.arange(
            1,
            num_tx + 1,
            dtype=np.float32,
        ).astype(np.complex64)

    def test_every_antenna_carries_one_common_pdsch_stream(self):
        batch = build_single_stream_mrt_batch(
            self.params,
            self.known_ref_seq,
            self.channel,
            num_frames=2,
            frame_seed=17,
        )

        tx_grids = ofdm_demodulate(
            batch.waveforms[0] / np.sqrt(self.params["POWER"]),
            self.params,
        )
        sf_index, slot_index, symbol_index, subcarrier_index = (
            batch.pdsch_indices
        )
        precoded_pdsch = tx_grids[
            :, sf_index, slot_index, symbol_index, subcarrier_index
        ]
        expected = batch.weights[:, None] * batch.pdsch_symbols[0][None, :]
        np.testing.assert_allclose(
            precoded_pdsch,
            expected,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_ideal_mrt_link_recovers_original_symbols(self):
        batch = build_single_stream_mrt_batch(
            self.params,
            self.known_ref_seq,
            self.channel,
            num_frames=2,
            frame_seed=22,
        )
        received = np.sum(
            self.channel[:, None] * batch.waveforms[0],
            axis=0,
            keepdims=True,
        )
        received = np.concatenate(
            (received, received[:, : int(self.params["sampling_rate"] * 1e-3)]),
            axis=1,
        )

        grid, _ = synchronize_and_demodulate(
            received,
            self.params,
            batch.ss_td_with_cp,
        )
        _, equalized, _ = equalize_common_pdsch(
            grid,
            self.params,
            self.known_ref_seq,
            batch.pdsch_indices,
        )

        np.testing.assert_allclose(
            equalized[0],
            batch.pdsch_symbols[0],
            rtol=1e-4,
            atol=1e-4,
        )

    def test_metrics_and_hard_decisions_report_an_ideal_link(self):
        symbols = np.array(
            [
                1 + 1j,
                1 - 1j,
                -1 + 1j,
                -1 - 1j,
            ],
            dtype=np.complex64,
        ) / np.sqrt(2)
        decisions = hard_decision(symbols + 0.01, 4)
        metrics = calculate_link_metrics(
            symbols,
            symbols,
            self.channel,
            4,
        )

        np.testing.assert_allclose(decisions, symbols)
        self.assertEqual(metrics["symbol_error_rate"], 0.0)
        self.assertAlmostEqual(metrics["evm_rms_percent"], 0.0)
        self.assertGreaterEqual(
            metrics["predicted_array_gain_vs_best_single_tx_db"],
            0.0,
        )

    def test_constellation_plot_is_saved(self):
        symbols = np.array([1 + 1j, -1 - 1j], dtype=np.complex64) / np.sqrt(2)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "constellation.png"
            plot_constellation_comparison(
                symbols,
                symbols,
                output_path,
                modulation_order=4,
            )
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
