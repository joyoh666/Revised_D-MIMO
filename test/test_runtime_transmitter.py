import queue
import threading
import types
import unittest

import numpy as np

from runtime.frame_producer import TXFrame
from runtime.transmitter import USRPTransmitter


class FakeTXMetadata:
    def __init__(self):
        self.has_time_spec = False
        self.time_spec = None
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
                "end_of_burst": metadata.end_of_burst,
            }
        )
        if samples.shape[1] > 0:
            self.stop_event.set()
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
        frame_queue = queue.Queue()
        samples = np.ones((2, 8), dtype=np.complex64)
        frame_queue.put(TXFrame(frame_id=7, samples=samples))

        streamer = FakeStreamer(stop_event)
        usrp = FakeUSRP(streamer)
        transmitter = USRPTransmitter(
            usrp=usrp,
            frame_queue=frame_queue,
            carrier_frequency=2.2e9,
            sampling_rate=1.92e6,
            gain=30,
            channels=(0, 1),
            transmission_delay=0.2,
            stop_event=stop_event,
            otw_format="sc16",
            uhd_module=make_fake_uhd(),
        )

        transmitter.run()

        self.assertEqual(usrp.gains, [(30, 0), (30, 1)])
        self.assertEqual(len(streamer.calls), 2)
        data_call, end_call = streamer.calls
        np.testing.assert_array_equal(data_call["samples"], samples)
        self.assertTrue(data_call["has_time_spec"])
        self.assertAlmostEqual(data_call["time_spec"].seconds, 10.2)
        self.assertFalse(data_call["end_of_burst"])
        self.assertEqual(end_call["samples"].shape, (2, 0))
        self.assertTrue(end_call["end_of_burst"])


if __name__ == "__main__":
    unittest.main()
