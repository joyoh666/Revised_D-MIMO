from USRP.helper_functions import get_zc_sequence, sss_sequence, place_virtual_pilots
from USRP.modulate import modulate

import numpy as np
from .ofdm import ofdm_modulate

def build_frame(params, known_ref_seq):
    N = params["N"]
    modulation_order = params["modulation_order"]
    num_subframe_per_frame = params["num_subframe_per_frame"]
    num_slot_per_subframe = params["num_slot_per_subframe"]
    num_symbols_per_slot = params["num_symbols_per_slot"]
    num_tx_ant = params["num_tx_ant"]
    N_PSS = params["N_PSS"]
    POWER = params["POWER"]
    SYNC_TX_IDX = params["sync_tx_idx"]

    PDSCH_PLACEHOLDER = 999

    resource_maps = np.ones(
        (num_tx_ant, num_subframe_per_frame, num_slot_per_subframe, num_symbols_per_slot, N)
        ,dtype=np.complex64,
    ) * PDSCH_PLACEHOLDER

    zc_seq = get_zc_sequence(N_PSS, 25)
    pss_start_idx = N - (N_PSS + 1) - ((N - (N_PSS + 1)) // 2)

    pss = np.zeros(N, dtype=np.complex64)
    pss[pss_start_idx:pss_start_idx + N_PSS] = zc_seq

    sss_first = np.zeros(N, dtype=np.complex64)
    sss_first[pss_start_idx:pss_start_idx + N_PSS] = sss_sequence(N_PSS, 0)     #Place sss into middle of the active subcarriers

    sss_second = np.zeros(N, dtype=np.complex64)
    sss_second[pss_start_idx:pss_start_idx + N_PSS] = sss_sequence(N_PSS, 1)

    for sfn in [0, 5]:
        resource_maps[:, sfn, 0, 6, :] = 0
        resource_maps[SYNC_TX_IDX, sfn, 0, 6, :] = pss

    resource_maps[:, 0, 0, 5, :] = 0
    resource_maps[SYNC_TX_IDX, 0, 0, 5, :] = sss_first

    resource_maps[:, 5, 0, 5, :] = 0
    resource_maps[SYNC_TX_IDX, 5, 0, 5, :] = sss_second

    # Stack known reference sequence and place reference sequence into every first symbol in each slot
    resource_maps[:, :, :, 0, :] = 0
    known_ref_stacked = np.tile(
        known_ref_seq.reshape(1, 1, N),
        [num_subframe_per_frame, num_slot_per_subframe, 1]
    )
    resource_maps[SYNC_TX_IDX, :, :, 0, :] = known_ref_stacked

    resource_maps = place_virtual_pilots(resource_maps, known_ref_seq)

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

    td_symbols_with_CP, td_symbols_with_cp_normal = ofdm_modulate(params, resource_maps)

    # SSS & PSS time domain symbol with CP : find frame start idx in receiver
    ss_td_with_cp = td_symbols_with_cp_normal[SYNC_TX_IDX, 0, 0, 4:, :].flatten()

    td_symbols_with_CP *= np.sqrt(POWER)
    # Streaming helpers use (num_tx_channels, num_samples).
    waveform = np.reshape(td_symbols_with_CP, (num_tx_ant, -1)).astype(np.complex64)

    return {
            "resource_maps_tx": resource_maps,
            "waveform": waveform,
            "pdsch_idx": pdsch_idx,
            "ss_td_with_cp": ss_td_with_cp,
            "data_bits": data_bits,
        }
