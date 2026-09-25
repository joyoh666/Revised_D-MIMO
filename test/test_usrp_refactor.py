import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np


# Keep plotting deterministic and hardware/display independent.  This must be
# set before ``plot_results`` imports pyplot lazily.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "revised-d-mimo-matplotlib"),
)

from USRP.build_frame import build_frame
from USRP.helper_functions import (
    generate_known_reference_sequence,
    get_system_params,
)
from USRP.ofdm import ofdm_demodulate
from USRP.postprocessing import postprocess
from USRP.results import CaptureAccumulator, plot_results, save_results


class USRPRefactorTests(unittest.TestCase):
    BANDWIDTH_MHZ = 1.4
    REFERENCE_SEED = 2040
    NUM_RX_ANT = 2
    NUM_CAPTURES = 2
    INITIAL_DISCARD_SAMPLES = 10
    TIMING_OFFSET_SAMPLES = 23
    TRAILING_SAMPLES = 64

    @classmethod
    def setUpClass(cls):
        cls.params = get_system_params(cls.BANDWIDTH_MHZ)
        cls.known_ref_seq = generate_known_reference_sequence(
            cls.params["N"],
            seed=cls.REFERENCE_SEED,
        )

        # ``build_frame`` currently uses NumPy's global RNG for PDSCH bits.
        # Restore the caller's RNG state so importing/running this test has no
        # lasting effect on another test.
        random_state = np.random.get_state()
        np.random.seed(12345)
        try:
            cls.frame_objects = build_frame(cls.params, cls.known_ref_seq)
        finally:
            np.random.set_state(random_state)

        waveform = cls.frame_objects["waveform"][0]
        antenna_gains = np.asarray(
            [1.0, 0.65 * np.exp(0.31j)],
            dtype=np.complex64,
        )
        prefix = np.zeros(
            cls.INITIAL_DISCARD_SAMPLES + cls.TIMING_OFFSET_SAMPLES,
            dtype=np.complex64,
        )
        suffix = np.zeros(cls.TRAILING_SAMPLES, dtype=np.complex64)
        cls.synthetic_capture = np.stack(
            [
                np.concatenate((prefix, gain * waveform, suffix))
                for gain in antenna_gains
            ],
            axis=0,
        ).astype(np.complex64)

        cls.received = postprocess(
            cls.params,
            cls.frame_objects["ss_td_with_cp"],
            cls.frame_objects["pdsch_idx"],
            cls.known_ref_seq,
            cls.synthetic_capture,
        )

    @classmethod
    def _finalized_results(cls):
        accumulator = CaptureAccumulator(cls.params)
        for _ in range(cls.NUM_CAPTURES):
            accumulator.add(cls.received)
        return accumulator.finalize()

    def test_build_frame_waveform_and_ofdm_round_trip(self):
        waveform = self.frame_objects["waveform"]
        self.assertEqual(
            waveform.shape,
            (1, self.params["frame_length"]),
        )
        self.assertEqual(waveform.dtype, np.complex64)

        unscaled_waveform = waveform / np.sqrt(self.params["POWER"])
        recovered_grid = ofdm_demodulate(unscaled_waveform, self.params)
        expected_grid = self.frame_objects["resource_maps_tx"]

        self.assertEqual(recovered_grid.shape, (1,) + expected_grid.shape)
        np.testing.assert_allclose(
            recovered_grid[0],
            expected_grid,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_two_antenna_synthetic_loopback_shapes(self):
        num_subframes = self.params["num_subframe_per_frame"]
        num_slots = self.params["num_slot_per_subframe"]
        num_symbols = self.params["num_symbols_per_slot"]
        num_subcarriers = self.params["N"]
        fft_size = self.params["FFT_SIZE"]
        num_virtual_pilots = self.params["num_virtual_pilots"]
        num_data_symbols = self.frame_objects["pdsch_idx"][0].size
        num_slot_total = num_subframes * num_slots

        expected_grid_shape = (
            self.NUM_RX_ANT,
            num_subframes,
            num_slots,
            num_symbols,
            num_subcarriers,
        )
        expected_channel_shape = (
            self.NUM_RX_ANT,
            num_subframes,
            num_slots,
            num_subcarriers,
        )

        self.assertEqual(
            self.received["frame_rcv_timesync"].shape,
            (self.NUM_RX_ANT, self.params["frame_length"]),
        )
        self.assertEqual(
            self.received["resource_maps_rcv"].shape,
            expected_grid_shape,
        )
        self.assertEqual(self.received["h_fd"].shape, expected_channel_shape)
        self.assertEqual(
            self.received["h_single_fd"].shape,
            expected_channel_shape,
        )
        for key in (
            "h_fd_virtual_avg_raw",
            "h_fd_virtual_avg_aligned",
            "virtual_pilot_variance_raw",
            "virtual_pilot_variance_aligned",
        ):
            self.assertEqual(
                self.received[key].shape,
                (self.NUM_RX_ANT, num_subcarriers),
                key,
            )
        for key in (
            "h_virtual_pilots_fd",
            "h_virtual_pilots_fd_aligned",
        ):
            self.assertEqual(
                self.received[key].shape,
                (num_virtual_pilots, self.NUM_RX_ANT, num_subcarriers),
                key,
            )

        self.assertEqual(
            self.received["virtual_pilot_common_phases_rad"].shape,
            (num_virtual_pilots, self.NUM_RX_ANT),
        )
        self.assertEqual(
            self.received["iq_rcv"].shape,
            (self.NUM_RX_ANT, num_data_symbols),
        )
        self.assertEqual(
            self.received["iq_rcv_single"].shape,
            (self.NUM_RX_ANT, num_data_symbols),
        )
        self.assertEqual(
            self.received["virtual_pilot_correction_phase_rad"].shape,
            (self.NUM_RX_ANT,),
        )
        self.assertEqual(
            self.received["ffo_hz"].shape,
            (self.NUM_RX_ANT,),
        )
        self.assertEqual(
            self.received["rfo_hz_per_slot"].shape,
            (self.NUM_RX_ANT, num_slot_total),
        )
        self.assertEqual(
            self.received["timing_correlation_peak"].shape,
            (self.NUM_RX_ANT,),
        )
        self.assertEqual(
            self.received["sync_idx"],
            self.TIMING_OFFSET_SAMPLES,
        )

        for key in (
            "h_fd",
            "iq_rcv",
            "ffo_hz",
            "rfo_hz_per_slot",
            "timing_correlation_peak",
        ):
            self.assertTrue(np.all(np.isfinite(self.received[key])), key)

        # The delay-domain conversion exercised by CaptureAccumulator uses the
        # FFT size, so retain this assertion beside the receiver shape contract.
        self.assertEqual(fft_size, 128)

    def test_capture_accumulator_finalize_shapes(self):
        results = self._finalized_results()
        num_subframes = self.params["num_subframe_per_frame"]
        num_slots = self.params["num_slot_per_subframe"]
        num_subcarriers = self.params["N"]
        fft_size = self.params["FFT_SIZE"]
        num_virtual_pilots = self.params["num_virtual_pilots"]
        num_slot_total = num_subframes * num_slots

        self.assertEqual(results["num_receive_antennas"], self.NUM_RX_ANT)
        self.assertEqual(
            results["channel_estimates_fd"].shape,
            (
                self.NUM_CAPTURES,
                self.NUM_RX_ANT,
                num_subframes,
                num_slots,
                num_subcarriers,
            ),
        )
        self.assertEqual(
            results["channel_mean_fd"].shape,
            (self.NUM_CAPTURES, self.NUM_RX_ANT, num_subcarriers),
        )
        for key in (
            "channel_impulse_response_td",
            "channel_impulse_response_td_single_pilot",
            "channel_full_spectrum_fd",
            "channel_full_spectrum_fd_single_pilot",
        ):
            self.assertEqual(
                results[key].shape,
                (self.NUM_CAPTURES, self.NUM_RX_ANT, fft_size),
                key,
            )
        for key in (
            "channel_estimates_fd_raw_virtual",
            "channel_estimates_fd_aligned_virtual",
            "virtual_pilot_variance_raw",
            "virtual_pilot_variance_aligned",
        ):
            self.assertEqual(
                results[key].shape,
                (self.NUM_CAPTURES, self.NUM_RX_ANT, num_subcarriers),
                key,
            )
        self.assertEqual(
            results["virtual_pilot_common_phases_rad"].shape,
            (
                self.NUM_CAPTURES,
                num_virtual_pilots,
                self.NUM_RX_ANT,
            ),
        )
        for key in (
            "single_pilot_snr_db",
            "virtual_pilot_snr_db",
            "virtual_pilot_snr_gain_db",
            "virtual_pilot_correction_phase_rad",
            "delay_profile_residual_gain_db",
            "ffo_estimates_hz",
            "timing_correlation_peaks",
        ):
            self.assertEqual(
                results[key].shape,
                (self.NUM_CAPTURES, self.NUM_RX_ANT),
                key,
            )
        self.assertEqual(
            results["rfo_estimates_hz"].shape,
            (
                self.NUM_CAPTURES,
                self.NUM_RX_ANT,
                num_slot_total,
            ),
        )
        self.assertEqual(results["sync_indices"].shape, (self.NUM_CAPTURES,))

    def test_save_and_plot_results(self):
        results = self._finalized_results()
        with tempfile.TemporaryDirectory() as temporary_directory:
            npz_path, json_path, figure_path, metadata = save_results(
                temporary_directory,
                self.params,
                self.known_ref_seq,
                results,
                reference_sequence_seed=self.REFERENCE_SEED,
                timestamp="20260922_120000",
            )
            plot_paths = plot_results(
                figure_path,
                self.params,
                results,
                show=False,
            )

            self.assertTrue(npz_path.is_file())
            self.assertTrue(json_path.is_file())
            self.assertGreater(npz_path.stat().st_size, 0)
            self.assertGreater(json_path.stat().st_size, 0)
            self.assertEqual(
                set(plot_paths),
                {
                    "channel_plots",
                    "delay_profile",
                    "constellation",
                    "equalized_psd",
                },
            )
            for plot_path in plot_paths.values():
                self.assertTrue(plot_path.is_file(), plot_path)
                self.assertGreater(plot_path.stat().st_size, 0, plot_path)

            json_metadata = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(json_metadata, metadata)
            self.assertEqual(metadata["num_channel_captures"], self.NUM_CAPTURES)
            self.assertEqual(
                metadata["num_receive_antennas"],
                self.NUM_RX_ANT,
            )
            self.assertEqual(
                metadata["reference_sequence_seed"],
                self.REFERENCE_SEED,
            )
            self.assertEqual(
                metadata["saved_channel_tensor_shape"],
                list(results["channel_estimates_fd"].shape),
            )

            with np.load(npz_path, allow_pickle=False) as saved:
                self.assertEqual(
                    saved["channel_estimates_fd"].shape,
                    results["channel_estimates_fd"].shape,
                )
                self.assertEqual(
                    saved["known_reference_sequence"].dtype,
                    np.dtype(np.complex64),
                )
                np.testing.assert_array_equal(
                    saved["known_reference_sequence"],
                    self.known_ref_seq,
                )
                archived_metadata = json.loads(saved["metadata_json"].item())
                self.assertEqual(archived_metadata, metadata)


if __name__ == "__main__":
    unittest.main()
