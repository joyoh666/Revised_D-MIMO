import threading
import types
import unittest

import numpy as np

from USRP.helper_functions import (
    generate_known_reference_sequence,
    get_system_params,
)
from runtime.transmitter import USRPTransmitter, build_precomputed_frames


class FakeTXMetadata:
    def __init__(self):
        self.has_time_spec = False
        self.time_spec = None
        self.start_of_burst = False
        self.end_of_burst = False


class FakeTimeSpec:
    def __init__(self, seconds):
        self.seconds = seconds


class FakeStreamArgs:
    def __init__(self, cpu_format, otw_format):
        self.cpu_format = cpu_format
        self.otw_format = otw_format
        self.channels = None


class FakeStreamer:
    def __init__(self, stop_event):
        self.stop_event = stop_event
        self.calls = []

    def send(self, samples, metadata):
        self.calls.append(
            {
                "samples": samples.copy(),
                "has_time_spec": metadata.has_time_spec,
                "time_spec": metadata.time_spec,
                "start_of_burst": metadata.start_of_burst,
                "end_of_burst": metadata.end_of_burst,
            }
        )
        return samples.shape[1]


class FakeUSRP:
    def __init__(self, streamer):
        self.streamer = streamer
        self.rates = []
        self.frequencies = []
        self.gains = []

    def set_tx_rate(self, rate, channel):
        self.rates.append((rate, channel))

    def set_tx_freq(self, frequency, channel):
        self.frequencies.append((frequency, channel))

    def set_tx_gain(self, gain, channel):
        self.gains.append((gain, channel))

    def get_tx_stream(self, stream_args):
        return self.streamer

    def get_time_now(self):
        return types.SimpleNamespace(get_real_secs=lambda: 10.0)


def make_fake_uhd():
    return types.SimpleNamespace(
        libpyuhd=types.SimpleNamespace(
            types=types.SimpleNamespace(tune_request=lambda frequency: frequency)
        ),
        usrp=types.SimpleNamespace(StreamArgs=FakeStreamArgs),
        types=types.SimpleNamespace(
            TXMetadata=FakeTXMetadata,
            TimeSpec=FakeTimeSpec,
        ),
    )


class USRPTransmitterTests(unittest.TestCase):
    def test_transmits_tx_frame_and_sends_end_of_burst(self):
        stop_event = threading.Event()
        frames = np.stack(
            (
                np.ones((6, 8), dtype=np.complex64),
                np.full((6, 8), 2 + 1j, dtype=np.complex64),
            )
        )

        streamer = FakeStreamer(stop_event)
        usrp = FakeUSRP(streamer)
        transmitter = USRPTransmitter(
            usrp=usrp,
            frames=frames,
            carrier_frequency=2.2e9,
            sampling_rate=1.92e6,
            gain=30,
            channels=tuple(range(6)),
            transmission_delay=0.2,
            stop_event=stop_event,
            otw_format="sc16",
            uhd_module=make_fake_uhd(),
        )

        transmitter.run()

        self.assertEqual(
            usrp.gains,
            [(30, channel) for channel in range(6)],
        )
        self.assertEqual(len(streamer.calls), 3)
        first_call, second_call, end_call = streamer.calls
        np.testing.assert_array_equal(first_call["samples"], frames[0])
        np.testing.assert_array_equal(second_call["samples"], frames[1])
        self.assertTrue(first_call["has_time_spec"])
        self.assertAlmostEqual(first_call["time_spec"].seconds, 10.2)
        self.assertTrue(first_call["start_of_burst"])
        self.assertFalse(first_call["end_of_burst"])
        self.assertFalse(second_call["has_time_spec"])
        self.assertFalse(second_call["start_of_burst"])
        self.assertFalse(second_call["end_of_burst"])
        self.assertFalse(end_call["start_of_burst"])
        self.assertEqual(end_call["samples"].shape, (6, 0))
        self.assertTrue(end_call["end_of_burst"])

    def test_precomputed_frames_are_random_per_frame_and_seeded_per_run(self):
        params = get_system_params(1.4)
        known_ref_seq = generate_known_reference_sequence(params["N"], 2040)

        first_run = build_precomputed_frames(
            params,
            known_ref_seq,
            num_frames=2,
            frame_seed=100,
        )
        repeated_run = build_precomputed_frames(
            params,
            known_ref_seq,
            num_frames=2,
            frame_seed=100,
        )
        different_run = build_precomputed_frames(
            params,
            known_ref_seq,
            num_frames=2,
            frame_seed=101,
        )

        self.assertEqual(
            first_run.shape,
            (2, params["num_tx_ant"], params["frame_length"]),
        )
        np.testing.assert_array_equal(first_run, repeated_run)
        self.assertFalse(np.array_equal(first_run[0], first_run[1]))
        self.assertFalse(np.array_equal(first_run, different_run))


if __name__ == "__main__":
    unittest.main()
