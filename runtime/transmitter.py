from queue import Empty, Full

import numpy as np


class USRPTransmitter:
    def __init__(
        self,
        usrp,
        frame_queue,
        carrier_frequency,
        sampling_rate,
        gain,
        channels,
        transmission_delay,
        stop_event,
        otw_format,
        transmission_time=None,
        start_time_queue=None,
        uhd_module=None,
    ):
        if uhd_module is None:
            import uhd as uhd_module

        self.uhd = uhd_module
        self.usrp = usrp
        self.frame_queue = frame_queue
        self.carrier_frequency = carrier_frequency
        self.sampling_rate = sampling_rate
        self.channels = channels
        self.gain = (
            [gain] * len(channels)
            if np.isscalar(gain) and not isinstance(gain, bool)
            else gain
        )
        self.transmission_delay = transmission_delay
        self.transmission_time = transmission_time
        self.start_time_queue = start_time_queue
        self.stop_event = stop_event
        self.otw_format = otw_format

        for i, channel in enumerate(channels):
            self.usrp.set_tx_rate(self.sampling_rate, channel)
            self.usrp.set_tx_freq(
                self.uhd.libpyuhd.types.tune_request(self.carrier_frequency),
                channel,
            )
            self.usrp.set_tx_gain(self.gain[i], channel)

        stream_args = self.uhd.usrp.StreamArgs("fc32", otw_format)
        stream_args.channels = channels
        self.streamer = self.usrp.get_tx_stream(stream_args)

    def _send_frame(self, samples, metadata) -> None:
        sent_total = 0
        no_progress_count = 0

        while sent_total < samples.shape[1]:
            chunk = np.ascontiguousarray(samples[:, sent_total:])
            num_sent = self.streamer.send(chunk, metadata)

            if num_sent > 0:
                sent_total += num_sent
                no_progress_count = 0
                metadata.has_time_spec = False
            else:
                no_progress_count += 1

            if no_progress_count >= 10:
                raise TimeoutError(
                    "USRP transmit made no progress after 10 attempts"
                )

    def _send_end_of_burst(self) -> None:
        metadata = self.uhd.types.TXMetadata()
        metadata.end_of_burst = True
        self.streamer.send(
            np.zeros((len(self.channels), 0), dtype=np.complex64),
            metadata,
        )

    def _publish_start_time(self, transmission_time) -> bool:
        if self.start_time_queue is None:
            return True

        while not self.stop_event.is_set():
            try:
                self.start_time_queue.put(transmission_time, timeout=0.2)
                return True
            except Full:
                continue
        return False

    def run(self) -> None:
        metadata = self.uhd.types.TXMetadata()
        first_frame = True
        failure = None

        try:
            while not self.stop_event.is_set():
                try:
                    tx_frame = self.frame_queue.get(timeout=0.2)
                except Empty:
                    continue

                samples = np.ascontiguousarray(
                    tx_frame.samples,
                    dtype=np.complex64,
                )
                if samples.ndim != 2:
                    raise ValueError(
                        "TX frame samples must have shape "
                        "(num_channels, num_samples)"
                    )
                if samples.shape[0] != len(self.channels):
                    raise ValueError(
                        f"TX frame has {samples.shape[0]} channel(s), but "
                        f"{len(self.channels)} channel(s) were configured"
                    )
                if first_frame:
                    transmission_time = self.transmission_time
                    if transmission_time is None:
                        transmission_time = (
                            self.usrp.get_time_now().get_real_secs()
                            + self.transmission_delay
                        )
                    if not self._publish_start_time(transmission_time):
                        break
                    metadata.has_time_spec = True
                    metadata.time_spec = self.uhd.types.TimeSpec(
                        transmission_time
                    )
                    first_frame = False

                self._send_frame(samples, metadata)
        except KeyboardInterrupt:
            self.stop_event.set()
        except Exception as exc:
            failure = exc
            self.stop_event.set()
        finally:
            try:
                self._send_end_of_burst()
            except Exception as exc:
                if failure is None:
                    failure = exc
            finally:
                self.streamer = None

        if failure is not None:
            raise failure


def transmit_frames(
    *,
    frame_queue,
    stop_event,
    device_args,
    tx_subdev_spec,
    tx_antenna,
    carrier_frequency,
    sampling_rate,
    gain,
    channels,
    otw_format,
    transmission_delay,
    clock_source=None,
    time_source=None,
    sync_barrier=None,
    receiver_ready_event=None,
    start_time_queue=None,
) -> None:
    """Initialize UHD and transmit queued frames inside the child process."""
    try:
        from USRP.usrp_utils import initialize_tx_usrp

        usrp = initialize_tx_usrp(
            device_args=device_args,
            tx_subdev_spec=tx_subdev_spec,
            tx_antenna=tx_antenna,
            tx_channels=channels,
            clock_source=clock_source,
            time_source=time_source,
            sync_barrier=sync_barrier,
        )

        if receiver_ready_event is not None:
            while not stop_event.is_set():
                if receiver_ready_event.wait(timeout=0.2):
                    break
            else:
                return

        transmitter = USRPTransmitter(
            usrp=usrp,
            frame_queue=frame_queue,
            carrier_frequency=carrier_frequency,
            sampling_rate=sampling_rate,
            gain=gain,
            channels=channels,
            transmission_delay=transmission_delay,
            stop_event=stop_event,
            otw_format=otw_format,
            start_time_queue=start_time_queue,
        )
        transmitter.run()
    except KeyboardInterrupt:
        stop_event.set()
    except Exception:
        stop_event.set()
        raise
