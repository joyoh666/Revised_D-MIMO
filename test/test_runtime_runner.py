import unittest

from config.radio_config import NUM_CHANNEL_CAPTURES
from runtime.runner import _parse_args, _resolve_num_captures


class RuntimeRunnerTests(unittest.TestCase):
    def test_five_second_duration_resolves_to_one_thousand_frames(self):
        args = _parse_args(["--duration-s", "5"])

        self.assertEqual(_resolve_num_captures(args, 0.005), 1000)

    def test_capture_count_remains_available(self):
        args = _parse_args(["--captures", "25"])

        self.assertEqual(_resolve_num_captures(args, 0.005), 25)

    def test_default_capture_count_is_preserved(self):
        args = _parse_args([])

        self.assertEqual(
            _resolve_num_captures(args, 0.005),
            NUM_CHANNEL_CAPTURES,
        )

    def test_duration_must_align_to_csi_period(self):
        args = _parse_args(["--duration-s", "1.001"])

        with self.assertRaises(ValueError):
            _resolve_num_captures(args, 0.005)


if __name__ == "__main__":
    unittest.main()
