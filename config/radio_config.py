import numpy as np

center_frequency = 2.2e9
bandwidth = 1.4e6

#resource grid parameters
fft_size = 128
subcarrier_spacing = 15e3
cp_length = 9
N_ofdm_symbols = 14

pilot_spacing = 8
pilot_indices = np.arange(0, fft_size, pilot_spacing)

all_indices = np.arange(0, fft_size)

data_indices = np.setdiff1d(all_indices, pilot_indices)

#channel parameters
channel_type = "multipath"

channel_delays = [0,1,4,6]
channel_gains_db = [0, -3, -10, -15]