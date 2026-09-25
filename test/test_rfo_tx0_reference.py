import unittest

import numpy as np

from USRP.build_frame import build_frame
from USRP.channelestimation import channelestimation
from USRP.helper_functions import (
    generate_known_reference_sequence,
    get_system_params,
)
from USRP.postprocessing import (
    estimate_and_correct_ffo,
    estimate_and_correct_rfo,
    postprocess,
    synchronize_frame_timing,
)


class TX0ReferenceAndRFOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = get_system_params(1.4)
        cls.known_ref_seq = generate_known_reference_sequence(
            cls.params["N"],
            seed=2041,
        )
        cls.frame = build_frame(cls.params, cls.known_ref_seq)

    def test_only_tx0_transmits_symbol_zero(self):
        resource_maps = self.frame["resource_maps_tx"]
        sync_tx_idx = self.params["sync_tx_idx"]

        np.testing.assert_allclose(
            resource_maps[sync_tx_idx, :, :, 0, :],
            np.broadcast_to(
                self.known_ref_seq,
                resource_maps[sync_tx_idx, :, :, 0, :].shape,
            ),
        )
        other_tx_indices = np.arange(self.params["num_tx_ant"]) != sync_tx_idx
        np.testing.assert_array_equal(
            resource_maps[other_tx_indices, :, :, 0, :],
            0,
        )

    def test_virtual_pilots_use_six_nonzero_symbols_per_tx(self):
        resource_maps = self.frame["resource_maps_tx"]
        positions = self.params["virtual_pilot_positions"]
        num_tx_ant = self.params["num_tx_ant"]
        pilots_per_tx = self.params["num_virtual_pilots"]

        self.assertEqual(len(positions), num_tx_ant * pilots_per_tx)
        for tx_idx in range(num_tx_ant):
            tx_positions = [
                position for position in positions if position[0] == tx_idx
            ]
            self.assertEqual(len(tx_positions), pilots_per_tx)
            self.assertEqual(
                {position[3] for position in tx_positions},
                set(range(1, self.params["num_symbols_per_slot"])),
            )
            for _, sf_idx, slot_idx, symbol_idx in tx_positions:
                np.testing.assert_allclose(
                    resource_maps[tx_idx, sf_idx, slot_idx, symbol_idx, :],
                    self.known_ref_seq,
                )
                other_tx_indices = np.arange(num_tx_ant) != tx_idx
                np.testing.assert_array_equal(
                    resource_maps[
                        other_tx_indices,
                        sf_idx,
                        slot_idx,
                        symbol_idx,
                        :,
                    ],
                    0,
                )

    def test_timing_synchronization_is_callable_independently(self):
        timing_offset = 29
        waveform = self.frame["waveform"][self.params["sync_tx_idx"]]
        gains = np.asarray(
            [1.0, 0.7 * np.exp(0.42j)],
            dtype=np.complex64,
        )
        prefix = np.zeros(timing_offset, dtype=np.complex64)
        suffix = np.zeros(64, dtype=np.complex64)
        capture = np.stack(
            [
                np.concatenate((prefix, gain * waveform, suffix))
                for gain in gains
            ],
            axis=0,
        )

        synchronized, sync_idx, correlation_peaks = (
            synchronize_frame_timing(
                capture,
                self.frame["ss_td_with_cp"],
                self.params,
            )
        )

        self.assertEqual(sync_idx, timing_offset)
        self.assertEqual(
            synchronized.shape,
            (gains.size, self.params["frame_length"]),
        )
        self.assertEqual(correlation_peaks.shape, (gains.size,))
        np.testing.assert_allclose(
            synchronized,
            gains[:, None] * waveform[None, :],
            rtol=1e-6,
            atol=1e-6,
        )

    def test_ffo_is_callable_independently(self):
        waveform = self.frame["waveform"][self.params["sync_tx_idx"]]
        gains = np.asarray(
            [1.0, 0.55 * np.exp(-0.27j)],
            dtype=np.complex64,
        )
        expected_without_ffo = gains[:, None] * waveform[None, :]
        injected_ffo_hz = 1325.5
        sample_indices = np.arange(
            self.params["frame_length"],
            dtype=np.float64,
        )
        frequency_rotation = np.exp(
            1j
            * 2.0
            * np.pi
            * injected_ffo_hz
            * sample_indices
            / self.params["sampling_rate"]
        ).astype(np.complex64)
        frame_with_ffo = expected_without_ffo * frequency_rotation[None, :]

        corrected, estimated_ffo_hz = estimate_and_correct_ffo(
            frame_with_ffo,
            self.params,
        )

        np.testing.assert_allclose(
            estimated_ffo_hz,
            injected_ffo_hz,
            rtol=0,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            corrected,
            expected_without_ffo,
            rtol=2e-5,
            atol=2e-5,
        )

    def test_channel_estimation_preserves_tx_and_rx_axes(self):
        params = dict(self.params)
        params["num_rx_ant"] = 2
        num_tx_ant = params["num_tx_ant"]
        num_rx_ant = params["num_rx_ant"]
        num_pilots = params["num_virtual_pilots"]
        num_subcarriers = params["N"]

        tx_indices = np.arange(num_tx_ant, dtype=np.float32)
        channel_rx_tx = np.stack(
            (
                np.exp(1j * 0.2 * tx_indices),
                0.7 * np.exp(1j * (-0.15 * tx_indices + 0.3)),
            ),
            axis=0,
        ).astype(np.complex64)
        resource_maps_rcv = np.sum(
            channel_rx_tx[:, :, None, None, None, None]
            * self.frame["resource_maps_tx"][None, ...],
            axis=1,
        )

        outputs = channelestimation(
            params,
            resource_maps_rcv,
            self.known_ref_seq,
            self.frame["pdsch_idx"],
        )
        (
            h_fd,
            h_single_fd,
            h_raw,
            h_aligned,
            h_virtual,
            h_virtual_aligned,
            variance_raw,
            variance_aligned,
            common_phases,
            single_snr,
            averaged_snr,
            snr_gain,
            correction_phase,
            iq_rcv,
            iq_rcv_single,
        ) = outputs

        self.assertEqual(
            h_fd.shape,
            (
                num_rx_ant,
                num_tx_ant,
                params["num_subframe_per_frame"],
                params["num_slot_per_subframe"],
                num_subcarriers,
            ),
        )
        self.assertEqual(h_single_fd.shape, h_fd.shape)
        self.assertEqual(h_raw.shape, (num_tx_ant, num_rx_ant, num_subcarriers))
        self.assertEqual(h_aligned.shape, h_raw.shape)
        self.assertEqual(
            h_virtual.shape,
            (num_tx_ant, num_pilots, num_rx_ant, num_subcarriers),
        )
        self.assertEqual(h_virtual_aligned.shape, h_virtual.shape)
        self.assertEqual(variance_raw.shape, h_raw.shape)
        self.assertEqual(variance_aligned.shape, h_raw.shape)
        self.assertEqual(
            common_phases.shape,
            (num_tx_ant, num_pilots, num_rx_ant),
        )
        for metric in (single_snr, averaged_snr, snr_gain, correction_phase):
            self.assertEqual(metric.shape, (num_tx_ant, num_rx_ant))
        self.assertEqual(iq_rcv.shape, (num_rx_ant, 0))
        self.assertEqual(iq_rcv_single.shape, (num_rx_ant, 0))

        expected_tx_rx = np.broadcast_to(
            np.transpose(channel_rx_tx, (1, 0))[:, :, None],
            h_raw.shape,
        )
        np.testing.assert_allclose(
            h_raw,
            expected_tx_rx,
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            h_aligned,
            expected_tx_rx,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_six_tx_one_rx_postprocess_channel_shapes(self):
        num_tx_ant = self.params["num_tx_ant"]
        timing_offset = 23
        channel_tx = np.exp(
            1j * 0.2 * np.arange(num_tx_ant)
        ).astype(np.complex64)
        composite_waveform = np.sum(
            channel_tx[:, None] * self.frame["waveform"],
            axis=0,
        )
        capture = np.concatenate(
            (
                np.zeros(10 + timing_offset, dtype=np.complex64),
                composite_waveform,
                np.zeros(128, dtype=np.complex64),
            )
        )[None, :]

        received = postprocess(
            self.params,
            self.frame["ss_td_with_cp"],
            self.frame["pdsch_idx"],
            self.known_ref_seq,
            capture,
        )

        self.assertEqual(received["sync_idx"], timing_offset)
        self.assertEqual(
            received["h_fd"].shape,
            (
                1,
                num_tx_ant,
                self.params["num_subframe_per_frame"],
                self.params["num_slot_per_subframe"],
                self.params["N"],
            ),
        )
        self.assertEqual(
            received["h_virtual_pilots_fd"].shape,
            (
                num_tx_ant,
                self.params["num_virtual_pilots"],
                1,
                self.params["N"],
            ),
        )
        self.assertEqual(received["iq_rcv"].shape, (1, 0))
        self.assertTrue(np.all(np.isfinite(received["h_fd"])))

    def test_rfo_is_estimated_once_and_corrected_continuously(self):
        rng = np.random.default_rng(2042)
        num_rx_ant = 2
        frame_length = self.params["frame_length"]
        slot_length = self.params["slot_length"]
        fft_size = self.params["FFT_SIZE"]
        first_cp_length = self.params["first_CP_length"]
        sampling_rate = self.params["sampling_rate"]
        num_slots = (
            self.params["num_subframe_per_frame"]
            * self.params["num_slot_per_subframe"]
        )

        base_frame = (
            rng.standard_normal(frame_length)
            + 1j * rng.standard_normal(frame_length)
        ).astype(np.complex64)
        repeated_reference = (
            rng.standard_normal(fft_size)
            + 1j * rng.standard_normal(fft_size)
        ).astype(np.complex64)
        for slot_idx in range(num_slots):
            start = slot_idx * slot_length + first_cp_length
            base_frame[start:start + fft_size] = repeated_reference

        gains = np.asarray(
            [1.0, 0.6 * np.exp(0.73j)],
            dtype=np.complex64,
        )
        expected_without_rfo = gains[:, None] * base_frame[None, :]
        injected_rfo_hz = 173.25
        sample_indices = np.arange(frame_length, dtype=np.float64)
        frequency_rotation = np.exp(
            1j
            * 2.0
            * np.pi
            * injected_rfo_hz
            * sample_indices
            / sampling_rate
        ).astype(np.complex64)
        frame_with_rfo = expected_without_rfo * frequency_rotation[None, :]

        corrected, estimated_rfo_hz, rfo_hz_per_slot = (
            estimate_and_correct_rfo(frame_with_rfo, self.params)
        )

        np.testing.assert_allclose(
            estimated_rfo_hz,
            injected_rfo_hz,
            rtol=0,
            atol=1e-3,
        )
        self.assertEqual(rfo_hz_per_slot.shape, (num_rx_ant, num_slots))
        np.testing.assert_allclose(
            corrected,
            expected_without_rfo,
            rtol=2e-5,
            atol=2e-5,
        )


if __name__ == "__main__":
    unittest.main()
