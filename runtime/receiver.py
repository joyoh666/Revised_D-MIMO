from dataclasses import dataclass
from queue import Empty, Full

import numpy as np


@dataclass
class RXBlock:
    """One complete block of samples captured from the UE USRP."""

    block_id: int
    samples: np.ndarray
    first_sample_time: float | None
    dropped_blocks: int


class USRPReceiver:
    def __init__(
        self,
        usrp,
        samples_queue,
        num_samples,
        carrier_frequency,
        sampling_rate,
        gain,
        channels,
        start_time,
        otw_format,
        stop_event,
        *,
        recv_timeout=0.2,
        max_consecutive_timeouts=10,
        uhd_module=None,
    ):
        if uhd_module is None:
            import uhd as uhd_module

        if num_samples <= 0:
            raise ValueError("num_samples must be greater than zero")
        if not channels:
            raise ValueError("at least one RX channel must be selected")
        if recv_timeout <= 0:
            raise ValueError("recv_timeout must be greater than zero")
        if max_consecutive_timeouts <= 0:
            raise ValueError(
                "max_consecutive_timeouts must be greater than zero"
            )

        self.uhd = uhd_module
        self.usrp = usrp
        self.samples_queue = samples_queue
        self.num_samples = int(num_samples)
        self.carrier_frequency = carrier_frequency
        self.sampling_rate = sampling_rate
        self.channels = tuple(channels)
        self.start_time = start_time
        self.otw_format = otw_format
        self.stop_event = stop_event
        self.recv_timeout = recv_timeout
        self.max_consecutive_timeouts = max_consecutive_timeouts
        self.dropped_blocks = 0

        self.gain = (
            [gain] * len(self.channels)
            if np.isscalar(gain) and not isinstance(gain, bool)
            else list(gain)
        )
        if len(self.gain) != len(self.channels):
            raise ValueError(
                "gain must be a scalar or contain one value per RX channel"
            )

        for index, channel in enumerate(self.channels):
            self.usrp.set_rx_rate(self.sampling_rate, channel)
            self.usrp.set_rx_freq(
                self.uhd.libpyuhd.types.tune_request(
                    self.carrier_frequency
                ),
                channel,
            )
            self.usrp.set_rx_gain(self.gain[index], channel)
            self.usrp.set_rx_dc_offset(False, channel)

        stream_args = self.uhd.usrp.StreamArgs("fc32", self.otw_format)
        stream_args.channels = list(self.channels)
        self.streamer = self.usrp.get_rx_stream(stream_args)
        self.max_samples_per_packet = self.streamer.get_max_num_samps()
        if self.max_samples_per_packet <= 0:
            raise RuntimeError("USRP RX streamer reported an invalid packet size")

    def _metadata_time(self, metadata) -> float | None:
        if not getattr(metadata, "has_time_spec", False):
            return None
        return float(metadata.time_spec.get_real_secs())

    def receive_samples(self, metadata):
        """Receive one complete block, or return ``None`` when stopping."""
        num_channels = len(self.channels)
        packet_buffer = np.empty(
            (num_channels, self.max_samples_per_packet),
            dtype=np.complex64,
        )
        samples = np.empty(
            (num_channels, self.num_samples),
            dtype=np.complex64,
        )

        received_total = 0
        consecutive_timeouts = 0
        first_sample_time = None

        while received_total < self.num_samples:
            if self.stop_event.is_set():
                return None

            remaining = self.num_samples - received_total
            packet_view = packet_buffer[
                :,
                :min(remaining, self.max_samples_per_packet),
            ]
            num_received = self.streamer.recv(
                packet_view,
                metadata,
                self.recv_timeout,
            )

            error_code = metadata.error_code
            if error_code not in (
                self.uhd.types.RXMetadataErrorCode.none,
                self.uhd.types.RXMetadataErrorCode.timeout,
            ):
                raise RuntimeError(f"USRP receive error: {error_code}")

            if num_received < 0 or num_received > packet_view.shape[1]:
                raise RuntimeError(
                    "USRP RX streamer returned an invalid sample count: "
                    f"{num_received}"
                )

            if num_received == 0:
                consecutive_timeouts += 1
                if consecutive_timeouts >= self.max_consecutive_timeouts:
                    raise TimeoutError(
                        "USRP receive made no progress after "
                        f"{consecutive_timeouts} attempts"
                    )
                continue

            if received_total == 0:
                first_sample_time = self._metadata_time(metadata)

            samples[
                :,
                received_total:received_total + num_received,
            ] = packet_view[:, :num_received]
            received_total += num_received
            consecutive_timeouts = 0

        return samples, first_sample_time

    def _enqueue_block(
        self,
        block_id,
        samples,
        first_sample_time,
    ) -> bool:
        """Queue the newest block without stalling the hardware RX loop."""
        while not self.stop_event.is_set():
            block = RXBlock(
                block_id=block_id,
                samples=samples,
                first_sample_time=first_sample_time,
                dropped_blocks=self.dropped_blocks,
            )
            try:
                self.samples_queue.put(block, timeout=0.05)
                return True
            except Full:
                try:
                    self.samples_queue.get_nowait()
                except Empty:
                    continue
                self.dropped_blocks += 1

        return False

    def _start_stream(self) -> None:
        stream_cmd = self.uhd.types.StreamCMD(
            self.uhd.types.StreamMode.start_cont
        )
        stream_cmd.stream_now = self.start_time is None
        if self.start_time is not None:
            stream_cmd.time_spec = self.uhd.types.TimeSpec(self.start_time)
        self.streamer.issue_stream_cmd(stream_cmd)

    def _stop_stream(self) -> None:
        stream_cmd = self.uhd.types.StreamCMD(
            self.uhd.types.StreamMode.stop_cont
        )
        self.streamer.issue_stream_cmd(stream_cmd)

    def run(self) -> None:
        metadata = self.uhd.types.RXMetadata()
        stream_started = False
        failure = None

        try:
            self._start_stream()
            stream_started = True

            block_id = 0
            while not self.stop_event.is_set():
                received = self.receive_samples(metadata)
                if received is None:
                    break

                samples, first_sample_time = received
                if not self._enqueue_block(
                    block_id,
                    samples,
                    first_sample_time,
                ):
                    break
                block_id += 1
        except KeyboardInterrupt:
            self.stop_event.set()
        except Exception as exc:
            failure = exc
            self.stop_event.set()
        finally:
            if stream_started and self.streamer is not None:
                try:
                    self._stop_stream()
                except Exception as exc:
                    if failure is None:
                        failure = exc
            self.streamer = None

        if failure is not None:
            raise failure


def receive_blocks(
    *,
    samples_queue,
    stop_event,
    device_args,
    rx_subdev_spec,
    rx_antenna,
    num_samples,
    carrier_frequency,
    sampling_rate,
    gain,
    channels,
    start_time,
    otw_format,
    clock_source=None,
    time_source=None,
    sync_barrier=None,
    receiver_ready_event=None,
    start_time_queue=None,
) -> None:
    """Initialize the UE USRP and receive blocks inside a child process."""
    try:
        import uhd
        from USRP.usrp_utils import initialize_rx_usrp

        selected_channels = tuple(channels)
        usrp = initialize_rx_usrp(
            device_args=device_args,
            rx_subdev_spec=rx_subdev_spec,
            rx_antenna=rx_antenna,
            rx_channels=selected_channels,
            clock_source=clock_source,
            time_source=time_source,
            sync_barrier=sync_barrier,
        )

        receiver = USRPReceiver(
            usrp=usrp,
            samples_queue=samples_queue,
            num_samples=num_samples,
            carrier_frequency=carrier_frequency,
            sampling_rate=sampling_rate,
            gain=gain,
            channels=selected_channels,
            start_time=start_time,
            otw_format=otw_format,
            stop_event=stop_event,
            uhd_module=uhd,
        )

        if receiver_ready_event is not None:
            receiver_ready_event.set()

        if start_time_queue is not None:
            while not stop_event.is_set():
                try:
                    receiver.start_time = start_time_queue.get(timeout=0.2)
                    break
                except Empty:
                    continue
            else:
                return

        receiver.run()
    except KeyboardInterrupt:
        stop_event.set()
    except Exception:
        stop_event.set()
        raise
    finally:
        cancel_join = getattr(samples_queue, "cancel_join_thread", None)
        if cancel_join is not None:
            cancel_join()
