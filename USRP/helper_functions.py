import numpy as np
from config.radio_config import (VIRTUAL_PILOT_SUBFRAME, 
                                VIRTUAL_PILOT_SLOT, VIRTUAL_PILOT_SYMBOLS,
                                bandwidth_options, DELTA_F,
                                normal_CP_time, first_CP_time, 
                                num_symbols_per_slot, num_slot_per_subframe, num_subframe_per_frame,
                                num_symbols_frame,carrier_frequency, Tx_gain, Rx_gain,
                                modulation_order, POWER,
                                N_PSS, NUM_VIRTUAL_PILOTS,
                                PHASE_ALIGN_VIRTUAL_PILOTS,
                                SYNC_TX_IDX, num_tx_ant, num_rx_ant,
                                VIRTUAL_PILOT_GLOBAL_SLOTS
                                )

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
    num_slots = num_subframe_per_frame * num_slot_per_subframe
    virtual_pilot_symbols = tuple(VIRTUAL_PILOT_SYMBOLS)

    if len(VIRTUAL_PILOT_GLOBAL_SLOTS) != num_tx_ant:
        raise ValueError(
            "VIRTUAL_PILOT_GLOBAL_SLOTS must contain exactly one slot "
            f"for each TX antenna; got {len(VIRTUAL_PILOT_GLOBAL_SLOTS)} "
            f"slots for {num_tx_ant} antennas"
        )
    if len(set(VIRTUAL_PILOT_GLOBAL_SLOTS)) != num_tx_ant:
        raise ValueError("Each TX antenna must use a different virtual-pilot slot")
    if len(virtual_pilot_symbols) != NUM_VIRTUAL_PILOTS:
        raise ValueError(
            "NUM_VIRTUAL_PILOTS does not match VIRTUAL_PILOT_SYMBOLS"
        )
    if any(
        symbol_idx <= 0 or symbol_idx >= num_symbols_per_slot
        for symbol_idx in virtual_pilot_symbols
    ):
        raise ValueError(
            "Virtual pilots must use symbols 1..num_symbols_per_slot-1; "
            "symbol 0 is reserved for the TX0 RFO reference"
        )

    positions = []

    for tx_idx, global_slot_idx in enumerate(VIRTUAL_PILOT_GLOBAL_SLOTS):
        if not 0 <= global_slot_idx < num_slots:
            raise ValueError(
                f"Invalid global slot: {global_slot_idx}"
            )

        sf_idx , slot_idx = divmod(
            global_slot_idx,
            num_slot_per_subframe
        )

        for sym_idx in virtual_pilot_symbols:
            positions.append((tx_idx, sf_idx, slot_idx, sym_idx))

    return positions

def place_virtual_pilots(resource_maps, known_ref_seq):
    resource_maps = np.asarray(resource_maps)
    known_ref_seq = np.asarray(known_ref_seq, dtype=np.complex64)
    expected_shape = (
        num_tx_ant,
        num_subframe_per_frame,
        num_slot_per_subframe,
        num_symbols_per_slot,
    )
    if resource_maps.ndim != 5 or resource_maps.shape[:4] != expected_shape:
        raise ValueError(
            "resource_maps must have shape "
            "(num_tx_ant, num_subframes, num_slots, num_symbols, N)"
        )
    if known_ref_seq.shape != (resource_maps.shape[-1],):
        raise ValueError(
            f"known_ref_seq has shape {known_ref_seq.shape}, expected "
            f"{(resource_maps.shape[-1],)}"
        )

    virtual_pilot_positions = get_virtual_pilot_positions()
    for tx_idx, global_slot_idx in enumerate(VIRTUAL_PILOT_GLOBAL_SLOTS):
        sf_idx, slot_idx = divmod(
            global_slot_idx,
            num_slot_per_subframe,
        )

        # Symbol 0 is intentionally untouched: TX0's RFO reference was placed
        # there by build_frame().  Only symbols 1..6 belong to the TDM pilot.
        pilot_symbols = [
            symbol_idx
            for pilot_tx_idx, pilot_sf_idx, pilot_slot_idx, symbol_idx
            in virtual_pilot_positions
            if (
                pilot_tx_idx == tx_idx
                and pilot_sf_idx == sf_idx
                and pilot_slot_idx == slot_idx
            )
        ]
        resource_maps[:, sf_idx, slot_idx, pilot_symbols, :] = 0
        resource_maps[
            tx_idx, sf_idx, slot_idx, pilot_symbols, :
        ] = known_ref_seq[None, :]

    return resource_maps

def phase_align_channel_estimates(h_virtual_pilots):
    """Align pilot estimates to the first pilot for each antenna.

    The first axis indexes virtual pilots and the last axis indexes
    subcarriers. Any axes between them (currently the antenna axis) are
    preserved and aligned independently.
    """
    h_virtual_pilots = np.asarray(h_virtual_pilots)
    if h_virtual_pilots.ndim < 2 or h_virtual_pilots.shape[0] == 0:
        raise ValueError(
            "h_virtual_pilots must have a non-empty pilot axis"
        )

    ref = h_virtual_pilots[0]
    correlations = np.sum(
        np.conj(ref)[None, ...] * h_virtual_pilots,
        axis=-1,
    )
    common_phases = np.angle(correlations)
    aligned = h_virtual_pilots * np.exp(-1j * common_phases[..., None])
    return aligned.astype(np.complex64), common_phases.astype(np.float32)

def pilot_snr_db_from_equalized(equalized_pilots_fd, known_ref_seq):
    """Estimate pilot-domain SNR while preserving receiver axes.

    The first axis contains repeated pilots and the last axis contains
    subcarriers. Any axes between them (currently the RX antenna axis) are
    retained in the returned SNR array.
    """
    equalized_pilots_fd = np.asarray(equalized_pilots_fd)
    if equalized_pilots_fd.size == 0:
        return 0.0
    known_ref_seq = np.asarray(known_ref_seq, dtype=np.complex64)
    reference_shape = (1,) * (equalized_pilots_fd.ndim - 1) + (-1,)
    equalized_unit = equalized_pilots_fd / known_ref_seq.reshape(reference_shape)
    mse = np.mean(
        np.abs(equalized_unit - 1.0) ** 2,
        axis=(0, equalized_pilots_fd.ndim - 1),
    )
    snr_db = np.asarray(10.0 * np.log10(1.0 / (mse + 1e-15)), dtype=np.float32)
    if snr_db.ndim == 0:
        return float(snr_db)
    return snr_db

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
        "delta_f": DELTA_F,
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
        "num_tx_ant": num_tx_ant,
        "num_rx_ant": num_rx_ant,
        "N_PSS": N_PSS,
        "num_virtual_pilots": NUM_VIRTUAL_PILOTS,
        "virtual_pilot_positions": get_virtual_pilot_positions(),
        "phase_align_virtual_pilots": PHASE_ALIGN_VIRTUAL_PILOTS,
        "sync_tx_idx": SYNC_TX_IDX
    }
