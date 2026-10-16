import unittest

import numpy as np

from USRP.helper_functions import (
    generate_known_reference_sequence,
    get_system_params,
)
from runtime.mrt_transmitter import (
    apply_mrt_precoding,
    build_mrt_frames,
    calculate_mrt_weights,
)


class MRTTransmitterTests(unittest.TestCase):
    def test_weights_are_unit_norm_and_coherently_combine(self):
        channel = np.array([1 + 2j, -0.5 + 0.25j], dtype=np.complex64)

        weights = calculate_mrt_weights(channel)

        np.testing.assert_allclose(weights, np.conj(channel) / np.linalg.norm(channel))
        self.assertAlmostEqual(float(np.linalg.norm(weights)), 1.0, places=6)
        effective_channel = np.sum(channel * weights)
        self.assertAlmostEqual(float(effective_channel.imag), 0.0, places=6)
        self.assertAlmostEqual(
            float(effective_channel.real),
            float(np.linalg.norm(channel)),
            places=6,
        )

    def test_precoding_preserves_total_power(self):
        signal = np.array([1 + 1j, -2j, 0.25 - 0.5j], dtype=np.complex64)
        channel = np.array([0.5 + 0.5j, -1 + 0.25j], dtype=np.complex64)

        precoded = apply_mrt_precoding(signal, channel)

        self.assertEqual(precoded.shape, (2, 3))
        np.testing.assert_allclose(
            np.sum(np.abs(precoded) ** 2, axis=0),
            np.abs(signal) ** 2,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_frame_builder_uses_one_common_stream(self):
        params = get_system_params(1.4)
        num_tx = int(params["num_tx_ant"])
        channel = np.arange(1, num_tx + 1, dtype=np.float32).astype(np.complex64)
        known_ref_seq = generate_known_reference_sequence(params["N"], 2040)

        frames = build_mrt_frames(
            params,
            known_ref_seq,
            channel,
            num_frames=2,
            frame_seed=123,
        )
        repeated = build_mrt_frames(
            params,
            known_ref_seq,
            channel,
            num_frames=2,
            frame_seed=123,
        )

        self.assertEqual(
            frames.shape,
            (2, num_tx, int(params["frame_length"])),
        )
        np.testing.assert_array_equal(frames, repeated)
        weights = calculate_mrt_weights(channel)
        nonzero_weight = int(np.flatnonzero(np.abs(weights) > 0)[0])
        source = frames[:, nonzero_weight] / weights[nonzero_weight]
        for tx_index, weight in enumerate(weights):
            np.testing.assert_allclose(
                frames[:, tx_index],
                weight * source,
                rtol=1e-6,
                atol=1e-6,
            )

    def test_rejects_wrong_size_and_zero_channel(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            calculate_mrt_weights([1 + 0j], num_tx_antennas=2)
        with self.assertRaisesRegex(ValueError, "all-zero"):
            calculate_mrt_weights([0j, 0j])


if __name__ == "__main__":
    unittest.main()
