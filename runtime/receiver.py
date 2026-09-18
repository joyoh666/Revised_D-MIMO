import uhd
from threading import Thread
import numpy as np

class USRPReceiver(Thread):
    def __init__(self, usrp, num_samples, carrier_frequency, sampling_rate, gain, channels, transmission_time, stop_event, otw_format):
        super().__init__()
        self.usrp = usrp
        self.num_samples = num_samples
        self.carrier_frequency = carrier_frequency
        self.sampling_rate = sampling_rate
        self.gain = gain
        self.channels = channels
        self.transmission_time = transmission_time
        self.stop_event = stop_event
        self.otw_format = otw_format
        
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

        head, tail = 0, 0
        try:
            while not self.stop_event.is_set():
                while tail < num_samples:
                    tail += self.streamer.recv(recv_buffer, metadata)
                    tail = min(tail, num_samples)
                    samples[:, head:tail] = recv_buffer[:, :tail-head]
                    head = tail

                    if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                        # print(metadata.error_code)
                        if metadata.error_code != uhd.types.RXMetadataErrorCode.timeout:
                            break
        except RuntimeError as ex:
            print(ex)
        
        return samples
    
    
    def run(self):
        # Start Stream
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done) # num_done, stop_cont, start_cont
        stream_cmd.stream_now = False
        stream_cmd.time_spec = uhd.types.TimeSpec(self.transmission_time)
        stream_cmd.num_samps = self.num_samples
        self.streamer.issue_stream_cmd(stream_cmd)
        
        metadata = uhd.types.RXMetadata()

        self.rcv_samples = self.receiveUSRP(self.num_samples, metadata)
        
        # Close streamer
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        self.streamer.issue_stream_cmd(stream_cmd)
        self.streamer = None