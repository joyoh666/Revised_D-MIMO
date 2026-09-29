import numpy as np
from config import radio_config as rc
from phy.qpsk import qpsk_modulate
from phy.ofdm import ofdm_modulate, ofdm_demodulate
from phy.resource_grid import resource_grid
from phy.channel import generate_channel, apply_channel
from phy.channel_estimation import LSestimation
from USRP.helper_functions import get_zc_sequence, sss_sequence, get_virtual_pilot_positions
from USRP.modulate import modulate

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

def build_frame(params, known_ref_seq):
    N = params["N"]
    FFT_SIZE = params["FFT_SIZE"]
    modulation_order = params["modulation_order"]
    first_CP_length = params["first_CP_length"]
    normal_CP_length = params["normal_CP_length"]
    num_subframe_per_frame = params["num_subframe_per_frame"]
    num_slot_per_subframe = params["num_slot_per_subframe"]
    num_symbols_per_slot = params["num_symbols_per_slot"]
    N_PSS = params["N_PSS"]

    PDSCH_PLACEHOLDER = 999

    resource_maps = np.ones(
        (num_subframe_per_frame, num_slot_per_subframe, num_symbols_per_slot, N)
        ,dtype=np.complex64,
    ) * PDSCH_PLACEHOLDER

    zc_seq = get_zc_sequence(N_PSS, 25)
    pss_start_idx = N - (N_PSS + 1) - (N - (N_PSS + 1) // 2)

    pss = np.zeros(N, dtype=np.complex64)
    pss[pss_start_idx:pss_start_idx + N_PSS] = zc_seq

    sss = np.zeros(N, dtype=np.complex64)
    sss[pss_start_idx:pss_start_idx + N_PSS] = sss_sequence(N_PSS, 0)

    resource_maps[0, 0, 6, :] = pss
    resource_maps[0, 0, 5, :] = sss

    known_ref_stacked = np.tile(
        known_ref_seq.reshape(1, 1, N),
        [num_subframe_per_frame, num_slot_per_subframe, 1]
    )
    resource_maps[..., 0, :] = known_ref_stacked

    pilot_positions = get_virtual_pilot_positions()
    for sf_idx, slot_idx, sym_idx in {
        position[1:] for position in pilot_positions
    }:
        if not (0 <= sf_idx < num_subframe_per_frame and 0 <= slot_idx < num_slot_per_subframe):
            continue
        if not (0 <= sym_idx < num_symbols_per_slot):
            continue
        resource_maps[sf_idx, slot_idx, sym_idx, :] = 0

    for tx_idx, sf_idx, slot_idx, sym_idx in pilot_positions:
        active_idx = params["fdm_pilot_active_indices"][tx_idx]
        resource_maps[sf_idx, slot_idx, sym_idx, active_idx] = (
            known_ref_seq[active_idx]
        )

    pdsch_idx = np.where(resource_maps == PDSCH_PLACEHOLDER)
    num_data_symbols = pdsch_idx[0].size

    data_bits = np.random.randint(
        0,
        2,
        num_data_symbols * int(np.log2(modulation_order)),
        dtype=np.uint8
    )
    iq = modulate(data_bits, modulation_order)

    resource_maps[pdsch_idx] = iq
