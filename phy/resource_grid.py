import numpy as np
from config import radio_config as rc

def resource_grid(data,pilot):
    N_symbol = rc.N_ofdm_symbols
    fft_size = rc.fft_size
    data_indices = rc.data_indices
    pilot_indices = rc.pilot_indices

    grid = np.zeros((fft_size, N_symbol), dtype=np.complex128)
    grid[data_indices, :] = data
    grid[pilot_indices, :] = pilot

    return grid

def rg_to_data(rg):
    return rg[rc.data_indices, :]