from USRP.helper_functions import get_zc_sequence, sss_sequence, get_virtual_pilot_positions
from USRP.modulate import modulate

import numpy as np
from .ofdm import ofdm_modulate

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
    POWER = params["POWER"]

    PDSCH_PLACEHOLDER = 999

    resource_maps = np.ones(
        (num_subframe_per_frame, num_slot_per_subframe, num_symbols_per_slot, N)
        ,dtype=np.complex64,
    ) * PDSCH_PLACEHOLDER

    zc_seq = get_zc_sequence(N_PSS, 25)
    pss_start_idx = N - (N_PSS + 1) - (N - (N_PSS + 1) // 2)        #Place zc_seq into middle of the active subcarriers

    pss = np.zeros(N, dtype=np.complex64)
    pss[pss_start_idx:pss_start_idx + N_PSS] = zc_seq

    sss_first = np.zeros(N, dtype=np.complex64)
    sss_first[pss_start_idx:pss_start_idx + N_PSS] = sss_sequence(N_PSS, 0)     #Place sss into middle of the active subcarriers

    sss_second = np.zeros(N, dtype=np.complex64)
    sss_second[pss_start_idx:pss_start_idx + N_PSS] = sss_sequence(N_PSS, 1)

    for sfn in [0, 5]:
        resource_maps[sfn, 0, 6, :] = pss

    resource_maps[0, 0, 5, :] = sss_first
    resource_maps[5, 0, 5, :] = sss_second

    # Stack known reference sequence and place reference sequence into every first symbol in each slot
    known_ref_stacked = np.tile(
        known_ref_seq.reshape(1, 1, N),
        [num_subframe_per_frame, num_slot_per_subframe, 1]
    )
    resource_maps[..., 0, :] = known_ref_stacked

    extra_virtual_pilot_symbols = 0
    for sf_idx, slot_idx, sym_idx in get_virtual_pilot_positions():
        if not (0 <= sf_idx < num_subframe_per_frame and 0 <= slot_idx < num_slot_per_subframe):
            continue
        if not (0 <= sym_idx < num_symbols_per_slot):
            continue
        resource_maps[sf_idx, slot_idx, sym_idx, :] = known_ref_seq
        if sym_idx != 0:
            extra_virtual_pilot_symbols += 1

    num_data_symbols = (
        (num_symbols_per_slot - 1)      # Exclude first symbol(reference symbol)
        * num_slot_per_subframe 
        * num_subframe_per_frame
        - 4                             # Exclude two PSS, SSS_first, SSS_second
        - extra_virtual_pilot_symbols   # Exclude virtual pilot 
    ) * N

    data_bits = np.random.randint(
        0,
        2,
        num_data_symbols * int(np.log2(modulation_order)),
        dtype=np.uint8
    )
    iq = modulate(data_bits, modulation_order)

    pdsch_idx = np.where(resource_maps == PDSCH_PLACEHOLDER)
    resource_maps[pdsch_idx] = iq

    td_symbols_with_CP, td_symbols_with_cp_normal = ofdm_modulate(params, resource_maps)

    # SSS & PSS time domain symbol with CP : find frame start idx in receiver
    ss_td_with_cp = td_symbols_with_cp_normal[0, 0, 4:, :].flatten()

    td_symbols_with_CP *= np.sqrt(POWER)
    waveform = np.reshape(td_symbols_with_CP, (-1,1)).astype(np.complex64)

    return {
            "resource_maps_tx": resource_maps,
            "waveform": waveform,
            "pdsch_idx": pdsch_idx,
            "ss_td_with_cp": ss_td_with_cp,
            "data_bits": data_bits,
        }