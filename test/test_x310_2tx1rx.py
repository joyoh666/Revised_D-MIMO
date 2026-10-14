"""Static preflight checks for single X310 2-TX/1-RX capture.

These tests exercise waveform/resource-grid configuration without RF hardware.
"""
import unittest

import numpy as np

from config import radio_config as cfg
from USRP.build_frame import build_frame
from USRP.helper_functions import (
    generate_known_reference_sequence,
    get_system_params,
)


class X310TwoTxOneRxTests(unittest.TestCase):
    def test_radio_mapping(self):
        self.assertEqual(cfg.TX_USRP_DEVICE_ARGS, "addr=192.168.70.3")
        self.assertEqual(cfg.TX_CHANNELS, (0, 1))
        self.assertEqual(cfg.RX_CHANNELS, (1,))
        self.assertEqual(cfg.TX_ANTENNA, "TX/RX")
        self.assertEqual(cfg.RX_ANTENNA, "RX2")
        self.assertEqual(cfg.num_tx_ant, 2)
        self.assertEqual(cfg.num_rx_ant, 1)

    def test_pilots_and_waveforms_cover_two_tx_channels(self):
        params = get_system_params(cfg.BANDWIDTH)
        self.assertEqual(params["num_tx_ant"], 2)
        self.assertEqual(params["num_rx_ant"], 1)
        self.assertEqual(params["frame_length"], 9600)
        self.assertAlmostEqual(params["csi_sample_period_s"], 0.005)
        self.assertEqual(len(params["fdm_pilot_active_indices"]), 2)
        self.assertEqual(len(set(params["fdm_pilot_active_indices"])), 2)

        known_seq = generate_known_reference_sequence(params["N"], 2026)
        frame = build_frame(params, known_seq, rng=np.random.default_rng(2026))
        waveform = frame["waveform"]
        self.assertEqual(waveform.shape, (2, 9600))
        self.assertTrue(np.any(waveform[0]))
        self.assertTrue(np.any(waveform[1]))


if __name__ == "__main__":
    unittest.main()
