import numpy as np
from config import radio_config as rc

def generate_channel():
    max_delay = max(rc.channel_delays)

    h_td = np.zeros(max_delay + 1, dtype=complex)

    gains = 10 ** (np.array(rc.channel_gains_db) / 20)

    phases = np.random.uniform(0, 2 * np.pi, len(gains))

    h_td[rc.channel_delays] = gains * np.exp(1j * phases)

    return h_td

def td_to_fd(h_td):
    return np.fft.fft(h_td, n=rc.fft_size)

def apply_channel(x, h_td):
    rx_data = np.convolve(x, h_td, mode='full')
    rx = rx_data[:-(len(h_td)-1)]

    return rx