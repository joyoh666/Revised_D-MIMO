import uhd
from threading import Thread
import numpy as np

class USRPTransmitter(Thread):
    def __init__(self, usrp, samples, carrier_frequency, sampling_rate, gain, channels, transmission_time, stop_event, otw_format):
        super().__init__()
        self.usrp = usrp
        assert samples.shape[0] == len(channels), "sample.shape[0] != # channel"
        self.samples = samples
        self.carrier_frequency = carrier_frequency
        self.sampling_rate = sampling_rate
        self.gain = gain
        self.channels = channels
        self.transmission_time = transmission_time
        self.stop_event = stop_event
        self.otw_format = otw_format
        
        for i, c in enumerate(channels):
            self.usrp.set_tx_rate(self.sampling_rate, c)
            self.usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(self.carrier_frequency), c)
            self.usrp.set_tx_gain(self.gain[i], c)
        
        # Set up the stream and receive buffer
        st_args = uhd.usrp.StreamArgs("fc32", otw_format)
        st_args.channels = channels
        self.streamer = self.usrp.get_tx_stream(st_args)
        
    
    def run(self):
        # Transmit Samples
        metadata = uhd.types.TXMetadata()
        metadata.has_time_spec = True
        metadata.time_spec = uhd.types.TimeSpec(self.transmission_time)

        try:
            while not self.stop_event.is_set():
                i = self.streamer.send(self.samples, metadata)
                while i < self.samples.shape[1]:
                    metadata.has_time_spec = False
                    i += self.streamer.send(self.samples[:, i:], metadata)
        except RuntimeError as ex:
            print(ex)
    
        # End
        metadata.end_of_burst = True
        self.streamer.send(np.zeros((self.samples.shape[0], 0), dtype=np.complex64), metadata)
        self.streamer = None

