import numpy as np
from config import radio_config as rc

def rg_to_pilot(rg):
    return rg[rc.pilot_indices, :]

def LSestimation(rg, pilot):
    pilot_received = rg_to_pilot(rg)
    h_LS = pilot_received / pilot

    H_LS = np.zeros((len(rc.data_indices), rc.N_ofdm_symbols), dtype=complex)

    for i in range(rc.N_ofdm_symbols): 
        h_real = np.interp(rc.data_indices,rc.pilot_indices, h_LS[:,i].real)
        h_imag = np.interp(rc.data_indices,rc.pilot_indices, h_LS[:,i].imag)

        H_LS[:, i] = h_real + 1j * h_imag

    return H_LS