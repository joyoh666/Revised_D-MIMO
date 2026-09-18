import numpy as np
from config.radio_config import (VIRTUAL_PILOT_SUBFRAME, 
                                VIRTUAL_PILOT_SLOT, VIRTUAL_PILOT_SYMBOLS,
                                bandwidth_options, DELTA_F,
                                normal_CP_time, first_CP_time, 
                                num_symbols_per_slot, num_slot_per_subframe, num_subframe_per_frame,
                                num_symbols_frame,carrier_frequency, Tx_gain, Rx_gain,
                                modulation_order, POWER,
                                N_PSS, NUM_VIRTUAL_PILOTS)

def get_zc_sequence(N, q=25):
    m = np.arange(N)
    seq = -np.pi * q * m * (m + 1) / N
    i = np.cos(seq)
    q = np.sin(seq)
    return (i + 1j * q).astype(np.complex64)

def sss_sequence(N, q):
    rng = np.random.default_rng(q)
    seq = rng.integers(0,2, size=N)
    return (2 * seq - 1 + 0.0j).astype(np.complex64)

def generate_known_reference_sequence(num_subcarriers, seed=2026):
    rng = np.random.default_rng(seed)
    qpsk_constellation = np.array(
            [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j],
            dtype=np.complex64,
        ) / np.sqrt(2)
    seq = rng.choice(qpsk_constellation, size=num_subcarriers)

    seq = seq / np.sqrt(np.mean(np.abs(seq) ** 2))
    return seq.astype(np.complex64)

def get_virtual_pilot_positions():
    return [
        (VIRTUAL_PILOT_SUBFRAME, VIRTUAL_PILOT_SLOT, sym) 
        for sym in VIRTUAL_PILOT_SYMBOLS
    ]

def phase_align_channel_estimates(h_virtual_pilots):
    ref = h_virtual_pilots[0]
    aligned = [ref.astype(np.complex64)]
    common_phases = [0.0]
    for idx in range(1, h_virtual_pilots.shape[0]):
        h_i = h_virtual_pilots[idx]
        phase = np.angle(np.vdot(ref, h_i))
        aligned_h = h_i * np.exp(-1j * phase)
        aligned.append(aligned_h.astype(np.complex64))
        common_phases.append(float(phase))
    return np.stack(aligned, axis=0).astype(np.complex64), np.asarray(common_phases, dtype=np.float32)

def pilot_snr_db_from_equalized(equalized_pilots_fd, known_ref_seq):
    if equalized_pilots_fd.size == 0:
        return 0.0
    equalized_unit = equalized_pilots_fd / known_ref_seq.reshape(1, -1).astype(np.complex64)
    mse = np.mean(np.abs(equalized_unit - 1.0) ** 2)
    return float(10.0 * np.log10(1.0 / (mse + 1e-15)))

def get_system_params(bandwidth_mhz):
    N, FFT_SIZE = bandwidth_options[bandwidth_mhz]
    sampling_rate = FFT_SIZE * DELTA_F
    T = 1 / DELTA_F

    normal_CP_length = round(normal_CP_time * sampling_rate)
    first_CP_length = round(first_CP_time * sampling_rate)

    slot_length = (
        FFT_SIZE * num_symbols_per_slot
        + normal_CP_length * (num_symbols_per_slot - 1)
        + first_CP_length
    )
    frame_length = slot_length * num_slot_per_subframe * num_subframe_per_frame

    return {
        "bandwidth_mhz": bandwidth_mhz,
        "N": N,
        "FFT_SIZE": FFT_SIZE,
        "sampling_rate": sampling_rate,
        "T": T,
        "normal_CP_length": normal_CP_length,
        "first_CP_length": first_CP_length,
        "slot_length": slot_length,
        "frame_length": frame_length,
        "carrier_frequency": carrier_frequency,
        "Tx_gain": Tx_gain,
        "Rx_gain": Rx_gain,
        "modulation_order": modulation_order,
        "POWER": POWER,
        "num_symbols_per_slot": num_symbols_per_slot,
        "num_slot_per_subframe": num_slot_per_subframe,
        "num_subframe_per_frame": num_subframe_per_frame,
        "num_symbols_frame": num_symbols_frame,
        "N_PSS": N_PSS,
        "num_virtual_pilots": NUM_VIRTUAL_PILOTS,
        "virtual_pilot_positions": get_virtual_pilot_positions()
    }