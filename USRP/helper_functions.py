import numpy as np
from config.radio_config import (
    CSI_SAMPLE_PERIOD_SUBFRAMES,
    DELTA_F,
    FDM_PILOT_CENTERED_BINS,
    FDM_PILOT_GLOBAL_SLOT,
    N_PSS,
    NUM_VIRTUAL_PILOTS,
    PHASE_ALIGN_VIRTUAL_PILOTS,
    POWER,
    SYNC_TX_IDX,
    VIRTUAL_PILOT_SYMBOL_START,
    bandwidth_options,
    carrier_frequency,
    first_CP_time,
    modulation_order,
    normal_CP_time,
    num_rx_ant,
    num_slot_per_subframe,
    num_subframe_per_frame,
    num_symbols_frame,
    num_symbols_per_slot,
    num_tx_ant,
    Rx_gain,
    Tx_gain,
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

def centered_bin_to_active_index(centered_bin, num_subcarriers):
    """Map a non-DC centered FFT bin to the active-subcarrier axis."""
    half = int(num_subcarriers) // 2
    centered_bin = int(centered_bin)
    if centered_bin == 0 or abs(centered_bin) > half:
        raise ValueError(
            f"centered bin {centered_bin} is outside the active allocation"
        )
    if centered_bin < 0:
        return centered_bin + half
    return half + centered_bin - 1


def get_fdm_pilot_active_indices(num_subcarriers):
    """Return the active-axis pilot index assigned to every TX antenna."""
    if len(FDM_PILOT_CENTERED_BINS) != num_tx_ant:
        raise ValueError(
            "FDM_PILOT_CENTERED_BINS must contain one bin per TX antenna"
        )
    if len(set(FDM_PILOT_CENTERED_BINS)) != num_tx_ant:
        raise ValueError("FDM pilot bins must be mutually distinct")
    return tuple(
        centered_bin_to_active_index(centered_bin, num_subcarriers)
        for centered_bin in FDM_PILOT_CENTERED_BINS
    )


def get_virtual_pilot_positions():
    """Return every FDM pilot as ``(tx, subframe, slot, symbol)``."""
    first_symbol = VIRTUAL_PILOT_SYMBOL_START
    last_symbol_exclusive = first_symbol + NUM_VIRTUAL_PILOTS
    if first_symbol <= 0 or last_symbol_exclusive > num_symbols_per_slot:
        raise ValueError(
            "The FDM virtual-pilot block must fit in symbols "
            "1..num_symbols_per_slot-1; symbol 0 is reserved for the "
            "TX0 RFO reference"
        )
    if num_subframe_per_frame % CSI_SAMPLE_PERIOD_SUBFRAMES:
        raise ValueError(
            "num_subframe_per_frame must be divisible by "
            "CSI_SAMPLE_PERIOD_SUBFRAMES"
        )

    slots_per_csi_sample = (
        CSI_SAMPLE_PERIOD_SUBFRAMES * num_slot_per_subframe
    )
    if not 0 <= FDM_PILOT_GLOBAL_SLOT < slots_per_csi_sample:
        raise ValueError(
            "FDM_PILOT_GLOBAL_SLOT must lie in the first CSI interval"
        )

    positions = []
    num_csi_samples = (
        num_subframe_per_frame // CSI_SAMPLE_PERIOD_SUBFRAMES
    )
    for csi_sample_idx in range(num_csi_samples):
        global_slot_idx = (
            FDM_PILOT_GLOBAL_SLOT
            + csi_sample_idx * slots_per_csi_sample
        )
        sf_idx, slot_idx = divmod(global_slot_idx, num_slot_per_subframe)
        for tx_idx in range(num_tx_ant):
            for sym_idx in range(first_symbol, last_symbol_exclusive):
                positions.append((tx_idx, sf_idx, slot_idx, sym_idx))

    return positions


def place_virtual_pilots(resource_maps, known_ref_seq, params):
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

    positions = params["virtual_pilot_positions"]

    # Clear every pilot-bearing OFDM symbol once before placing per-TX pilots.
    for sf_idx, slot_idx, symbol_idx in {
        position[1:] for position in positions
    }:
        resource_maps[:, sf_idx, slot_idx, symbol_idx, :] = 0

    active_indices = params["fdm_pilot_active_indices"]
    for tx_idx, sf_idx, slot_idx, symbol_idx in positions:
        active_idx = active_indices[tx_idx]
        resource_maps[
            tx_idx,
            sf_idx,
            slot_idx,
            symbol_idx,
            active_idx,
        ] = known_ref_seq[active_idx]

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

def _symbol_useful_center_sample(
    global_slot_idx,
    symbol_idx,
    *,
    slot_length,
    fft_size,
    first_cp_length,
    normal_cp_length,
):
    slot_start = global_slot_idx * slot_length
    if symbol_idx == 0:
        return slot_start + first_cp_length + fft_size / 2
    return (
        slot_start
        + first_cp_length
        + fft_size
        + (symbol_idx - 1) * (normal_cp_length + fft_size)
        + normal_cp_length
        + fft_size / 2
    )


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

    virtual_pilot_positions = get_virtual_pilot_positions()
    num_csi_samples = (
        num_subframe_per_frame // CSI_SAMPLE_PERIOD_SUBFRAMES
    )
    pilot_offsets = np.empty(
        (num_csi_samples, num_tx_ant, NUM_VIRTUAL_PILOTS),
        dtype=np.float64,
    )
    pilot_offsets.fill(np.nan)
    counts = np.zeros((num_csi_samples, num_tx_ant), dtype=np.int64)
    for tx_idx, sf_idx, slot_idx, symbol_idx in virtual_pilot_positions:
        csi_sample_idx = sf_idx // CSI_SAMPLE_PERIOD_SUBFRAMES
        repetition = counts[csi_sample_idx, tx_idx]
        global_slot_idx = sf_idx * num_slot_per_subframe + slot_idx
        pilot_offsets[csi_sample_idx, tx_idx, repetition] = (
            _symbol_useful_center_sample(
                global_slot_idx,
                symbol_idx,
                slot_length=slot_length,
                fft_size=FFT_SIZE,
                first_cp_length=first_CP_length,
                normal_cp_length=normal_CP_length,
            )
        )
        counts[csi_sample_idx, tx_idx] += 1
    if np.any(counts != NUM_VIRTUAL_PILOTS) or np.any(~np.isfinite(pilot_offsets)):
        raise ValueError("Every TX must have all repeated pilot timestamps")

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
        "num_csi_samples_per_frame": num_csi_samples,
        "csi_sample_period_subframes": CSI_SAMPLE_PERIOD_SUBFRAMES,
        "csi_sample_period_s": CSI_SAMPLE_PERIOD_SUBFRAMES * 1e-3,
        "virtual_pilot_positions": virtual_pilot_positions,
        "fdm_pilot_centered_bins": tuple(FDM_PILOT_CENTERED_BINS),
        "fdm_pilot_active_indices": get_fdm_pilot_active_indices(N),
        "pilot_timestamp_offsets_samples_by_tx": pilot_offsets.tolist(),
        "pilot_timestamp_offsets_samples": np.mean(
            pilot_offsets,
            axis=1,
        ).tolist(),
        "csi_sample_timestamp_offsets_samples": np.mean(
            pilot_offsets,
            axis=(1, 2),
        ).tolist(),
        "phase_align_virtual_pilots": PHASE_ALIGN_VIRTUAL_PILOTS,
        "sync_tx_idx": SYNC_TX_IDX
    }
