import numpy as np
import uhd
from threading import Thread

# usrp_utils version 2025-05-20 18:50


def initialize_usrp(
    device_args,
    tx_subdev_spec,
    rx_subdev_spec,
    tx_antenna,
    rx_antenna,
    tx_channels=(0,),
    rx_channels=None,
):
    """Create and configure the MultiUSRP used by the capture pipeline."""
    usrp = uhd.usrp.MultiUSRP(device_args)
    usrp.set_time_unknown_pps(uhd.types.TimeSpec(0.0))
    usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec(tx_subdev_spec), 0)
    usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec(rx_subdev_spec), 0)

    tx_channels = tuple(tx_channels)
    if rx_channels is None:
        rx_channels = tuple(range(usrp.get_rx_num_channels()))
    else:
        rx_channels = tuple(rx_channels)

    if not tx_channels or not rx_channels:
        raise ValueError("at least one TX and RX channel must be selected")
    if max(tx_channels) >= usrp.get_tx_num_channels():
        raise ValueError("configured TX channel is not exposed by the USRP")
    if max(rx_channels) >= usrp.get_rx_num_channels():
        raise ValueError("configured RX channel is not exposed by the USRP")

    for channel in tx_channels:
        usrp.set_tx_antenna(tx_antenna, channel)
    for channel in rx_channels:
        usrp.set_rx_antenna(rx_antenna, channel)

    print("USRP loaded. Session Ready.")
    print(f"Tx: {usrp.get_tx_subdev_spec(0)}")
    print(f"Rx: {usrp.get_rx_subdev_spec(0)}")
    print(f"TX channels: {tx_channels}")
    print(f"RX channels: {rx_channels}")
    return usrp


def sendAndReceive(usrp, signal, power, Fc, Fs,
                   Tx_gain=0, Rx_gain=10, Tx_chan=None, Rx_chan=None,
                   wait_time=0.2, tx_delay_samples=100, rx_trailing_samples=100,
                   otw_format='sc16'):
    '''
    usrp: usrp object.
    signal: (np.array, dtype=complex64) numpy array of complex-valued baseband signal.
    power: (float) constant value to be multiplied in signal.
    Fc: (float) carrier frequency
    Fs: (float) sampling rate
    Tx_gain: (float/list) Tx gain (in dB) for each channels. Typically 0 - 31.5
             If float, the value is broadcasted for all channels.
    Rx_gain: (float/list) Rx gain (in dB) for each channels. Typically 0 - 31.5
             If float, the value is broadcasted for all channels.
    Tx_chan: (list) Tx channels
    Rx_chan: (list) Rx channels
    wait_time: (float) time delay for initial sample reception (waiting for tx/rx thread spawn)
    tx_delay_samples: (int) time delay for initial sample transmission (avoiding large noise in initial rx samples)
    rx_trailing_samples: (int) additional samples received (accounting for tx timing error)
    otw_format: ('sc16' or 'sc8') symbol representation type within connection of PC and USRP.
                'sc16' uses 16-bit complex integer values and provides better resolution,
                while 'sc8' uses 8-bit representations with doubled Msps (samples per second) between USRP and PC links.
                'sc8' not supported in X310.
    '''

    signal = np.asarray(signal, dtype=np.complex64)
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
    if signal.ndim != 2:
        raise ValueError("signal must have shape (num_tx_channels, num_samples)")
    if power <= 0:
        raise ValueError("power must be positive")

    # Timing control
    current_time = usrp.get_time_now().get_real_secs()
    rx_time = current_time +  wait_time
    tx_time = rx_time # + tx_delay_samples / Fs

    # Tx/Rx channel/gain sanity check
    if Rx_chan is None:
        Rx_chan = list(range(usrp.get_rx_num_channels()))

    if Tx_chan is None:
        Tx_chan = list(range(usrp.get_tx_num_channels()))

    if signal.shape[0] != len(Tx_chan):
        raise ValueError(
            f"signal has {signal.shape[0]} row(s), but {len(Tx_chan)} "
            "TX channel(s) were selected"
        )
    
    if isinstance(Tx_gain, (int, float, complex)) and not isinstance(Tx_gain, bool):
        # If Tx_gain is constant:
        Tx_gain = [Tx_gain] * len(Tx_chan)

    if isinstance(Rx_gain, (int, float, complex)) and not isinstance(Rx_gain, bool):
        # If Rx_gain is constant:
        Rx_gain = [Rx_gain] * len(Rx_chan)
        
    # Rx
    rx_thread = USRPReceiver(
        usrp=usrp,
        num_samples=int(signal.shape[-1] + tx_delay_samples + rx_trailing_samples),
        carrier_frequency=Fc,
        sampling_rate=Fs, 
        gain=Rx_gain,
        channels=Rx_chan,
        transmission_time=rx_time,
        otw_format=otw_format
    )

    # Tx
    signal = signal.copy()
    signal *= np.sqrt(power)

    zero_pad = np.zeros((signal.shape[0], tx_delay_samples), dtype=np.complex64)
    signal = np.concatenate([zero_pad, signal], axis=1)

    tx_thread = USRPTransmitter(
        usrp=usrp,
        samples=signal.astype(np.complex64),
        carrier_frequency=Fc,
        sampling_rate=Fs, 
        gain=Tx_gain,
        channels=Tx_chan,
        transmission_time=tx_time,
        otw_format=otw_format
    )

    rx_thread.start()
    tx_thread.start()
    tx_thread.join()
    rx_thread.join()

    if tx_thread.error is not None:
        raise RuntimeError("USRP transmit failed") from tx_thread.error
    if rx_thread.error is not None:
        raise RuntimeError("USRP receive failed") from rx_thread.error
    if rx_thread.rcv_samples is None:
        raise RuntimeError("USRP receive completed without samples")

    rcv_signal = rx_thread.rcv_samples
    rcv_signal /= np.sqrt(power)
    
    return rcv_signal


class USRPReceiver(Thread):
    def __init__(self, usrp, num_samples, carrier_frequency, sampling_rate, gain, channels, transmission_time, otw_format):
        super().__init__()
        self.usrp = usrp
        self.num_samples = num_samples
        self.carrier_frequency = carrier_frequency
        self.sampling_rate = sampling_rate
        self.gain = gain
        self.channels = channels
        self.transmission_time = transmission_time
        self.otw_format = otw_format
        self.rcv_samples = None
        self.error = None
        
        for i, c in enumerate(channels):
          usrp.set_rx_rate(self.sampling_rate, c)
          usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(self.carrier_frequency), c)
          usrp.set_rx_gain(self.gain[i], c)
          usrp.set_rx_dc_offset(False, c)
        
        # Set up the stream and receive buffer
        st_args = uhd.usrp.StreamArgs("fc32", otw_format) # possibly 'sc8' for more data rate instead of more quantization error
        st_args.channels = channels
        self.streamer = self.usrp.get_rx_stream(st_args)
        self.num_samples_per_frame = self.streamer.get_max_num_samps()
    

    def receiveUSRP(self, num_samples, metadata):
        num_channels = len(self.channels)

        recv_buffer = np.zeros((num_channels, self.num_samples_per_frame), dtype=np.complex64)
        samples = np.zeros((num_channels, num_samples), dtype=np.complex64)

        tail = 0
        no_progress_count = 0
        while tail < num_samples:
            remaining = num_samples - tail
            recv_view = recv_buffer[
                :,
                :min(remaining, self.num_samples_per_frame),
            ]
            num_received = self.streamer.recv(recv_view, metadata)

            if num_received > 0:
                samples[:, tail:tail + num_received] = recv_view[
                    :,
                    :num_received,
                ]
                tail += num_received
                no_progress_count = 0
            else:
                no_progress_count += 1

            if (
                metadata.error_code
                not in (
                    uhd.types.RXMetadataErrorCode.none,
                    uhd.types.RXMetadataErrorCode.timeout,
                )
            ):
                raise RuntimeError(f"USRP receive error: {metadata.error_code}")
            if no_progress_count >= 10:
                raise TimeoutError(
                    "USRP receive made no progress after 10 attempts"
                )
        
        return samples
    
    
    def run(self):
        try:
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
            stream_cmd.stream_now = False
            stream_cmd.time_spec = uhd.types.TimeSpec(self.transmission_time)
            stream_cmd.num_samps = self.num_samples
            self.streamer.issue_stream_cmd(stream_cmd)

            metadata = uhd.types.RXMetadata()
            self.rcv_samples = self.receiveUSRP(self.num_samples, metadata)
        except Exception as exc:
            self.error = exc
        finally:
            if self.streamer is not None:
                try:
                    stream_cmd = uhd.types.StreamCMD(
                        uhd.types.StreamMode.stop_cont
                    )
                    self.streamer.issue_stream_cmd(stream_cmd)
                except Exception as exc:
                    if self.error is None:
                        self.error = exc
                finally:
                    self.streamer = None
        
        
class USRPTransmitter(Thread):
    def __init__(self, usrp, samples, carrier_frequency, sampling_rate, gain, channels, transmission_time, otw_format):
        super().__init__()
        self.usrp = usrp
        assert samples.shape[0] == len(channels), "sample.shape[0] != # channel"
        self.samples = samples
        self.carrier_frequency = carrier_frequency
        self.sampling_rate = sampling_rate
        self.gain = gain
        self.channels = channels
        self.transmission_time = transmission_time
        self.otw_format = otw_format
        self.error = None
        
        for i, c in enumerate(channels):
            self.usrp.set_tx_rate(self.sampling_rate, c)
            self.usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(self.carrier_frequency), c)
            self.usrp.set_tx_gain(self.gain[i], c)
        
        # Set up the stream and receive buffer
        st_args = uhd.usrp.StreamArgs("fc32", otw_format)
        st_args.channels = channels
        self.streamer = self.usrp.get_tx_stream(st_args)
        
    
    def run(self):
        try:
            metadata = uhd.types.TXMetadata()
            metadata.has_time_spec = True
            metadata.time_spec = uhd.types.TimeSpec(self.transmission_time)

            sent_total = 0
            no_progress_count = 0
            while sent_total < self.samples.shape[1]:
                num_sent = self.streamer.send(
                    self.samples[:, sent_total:],
                    metadata,
                )
                metadata.has_time_spec = False
                if num_sent > 0:
                    sent_total += num_sent
                    no_progress_count = 0
                else:
                    no_progress_count += 1
                if no_progress_count >= 10:
                    raise TimeoutError(
                        "USRP transmit made no progress after 10 attempts"
                    )
        except Exception as exc:
            self.error = exc
        finally:
            if self.streamer is not None:
                try:
                    metadata = uhd.types.TXMetadata()
                    metadata.end_of_burst = True
                    self.streamer.send(
                        np.zeros(
                            (self.samples.shape[0], 0),
                            dtype=np.complex64,
                        ),
                        metadata,
                    )
                except Exception as exc:
                    if self.error is None:
                        self.error = exc
                finally:
                    self.streamer = None
