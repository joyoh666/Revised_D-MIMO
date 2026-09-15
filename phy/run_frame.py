import numpy as np
from config import radio_config as rc
from phy.qpsk import qpsk_modulate
from phy.ofdm import ofdm_modulate, ofdm_demodulate
from phy.resource_grid import resource_grid
from phy.channel import generate_channel, apply_channel
from phy.channel_estimation import LSestimation

def run_frame():
    bits = np.random.randint(0, 2, 2 * len(rc.data_indices) * rc.N_ofdm_symbols)
    expected_bits = 2 * len(rc.data_indices) * rc.N_ofdm_symbols
    if bits.size != expected_bits:
        raise ValueError(
            f"Expected {expected_bits} bits for one frame, got {bits.size}"
        )

    symbols = qpsk_modulate(bits)

    pilots = np.ones((len(rc.pilot_indices), rc.N_ofdm_symbols))
    symbols = symbols.reshape(len(rc.data_indices), rc.N_ofdm_symbols)
    
    rg = resource_grid(symbols, pilots)

    tx_data = ofdm_modulate(rg, rc.fft_size, rc.cp_length)

    channel = generate_channel()
    # Serialize OFDM symbols column by column before applying the time-domain
    # channel. np.convolve (used by apply_channel) only accepts 1-D signals.
    tx_shape = tx_data.shape
    tx_data = tx_data.reshape(-1, order="F")
    noise = (
        np.random.randn(*tx_data.shape) + 1j * np.random.randn(*tx_data.shape)
    ) * 0.01

    rx_data = apply_channel(tx_data, channel) + noise
    rx_data = rx_data.reshape(tx_shape, order="F")

    demodulated_rg = ofdm_demodulate(rx_data, rc.fft_size, rc.cp_length)

    channel_LS = LSestimation(demodulated_rg, pilots)

    return channel_LS
