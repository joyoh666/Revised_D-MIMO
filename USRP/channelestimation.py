import numpy as np

from config.radio_config import PHASE_ALIGN_VIRTUAL_PILOTS
from .helper_functions import phase_align_channel_estimates


def channelestimation(params, resource_maps_rcv, known_ref_seq, pdsch_idx):
    """Estimate every TX-to-RX channel from simultaneous FDM pilots.

    ``resource_maps_rcv`` has shape
    ``(num_rx_ant, num_subframes, num_slots, num_symbols, N)``.
    ``pdsch_idx`` is retained for API compatibility, but independent PDSCH
    streams are not equalized because a single RX observation has no TX axis.
    """
    num_subframe_per_frame = params["num_subframe_per_frame"]
    num_slot_per_subframe = params["num_slot_per_subframe"]
    num_symbols_per_slot = params["num_symbols_per_slot"]
    N = params["N"]
    num_tx_ant = params["num_tx_ant"]
    num_rx_ant = params["num_rx_ant"]
    num_virtual_pilots = params["num_virtual_pilots"]
    num_csi_samples = params["num_csi_samples_per_frame"]
    csi_sample_period_subframes = params["csi_sample_period_subframes"]

    if resource_maps_rcv.ndim != 5:
        raise ValueError(
            "resource_maps_rcv must have shape "
            "(num_rx_ant, num_subframes, num_slots, num_symbols, N)"
        )

    expected_shape = (
        num_rx_ant,
        num_subframe_per_frame,
        num_slot_per_subframe,
        num_symbols_per_slot,
        N,
    )
    if resource_maps_rcv.shape != expected_shape:
        raise ValueError(
            f"resource_maps_rcv has shape {resource_maps_rcv.shape}, "
            f"expected {expected_shape}"
        )

    known_ref_seq = np.asarray(known_ref_seq, dtype=np.complex64)
    if known_ref_seq.shape != (N,):
        raise ValueError(
            f"known_ref_seq has shape {known_ref_seq.shape}, expected {(N,)}"
        )

    pilot_counts = np.zeros(
        (num_csi_samples, num_tx_ant),
        dtype=np.int64,
    )
    h_scalar = np.empty(
        (num_csi_samples, num_tx_ant, num_virtual_pilots, num_rx_ant),
        dtype=np.complex64,
    )
    pilot_received_scalar = np.empty_like(h_scalar)
    pilot_noise_power = np.full(
        (num_csi_samples, num_virtual_pilots, num_rx_ant),
        np.nan,
        dtype=np.float32,
    )

    active_indices = np.asarray(
        params["fdm_pilot_active_indices"],
        dtype=np.int64,
    )
    if active_indices.shape != (num_tx_ant,):
        raise ValueError("FDM requires one active pilot index per TX")
    noise_indices = np.setdiff1d(
        np.arange(N, dtype=np.int64),
        active_indices,
        assume_unique=True,
    )
    measured_noise_positions = set()
    for position in params["virtual_pilot_positions"]:
        tx_idx, sf_idx, slot_idx, sym_idx = position
        csi_sample_idx = sf_idx // csi_sample_period_subframes
        repetition = int(pilot_counts[csi_sample_idx, tx_idx])
        active_idx = int(active_indices[tx_idx])
        received_symbol = resource_maps_rcv[
            :, sf_idx, slot_idx, sym_idx, :
        ]
        received_pilot = received_symbol[:, active_idx]
        pilot_received_scalar[
            csi_sample_idx, tx_idx, repetition
        ] = received_pilot
        h_scalar[csi_sample_idx, tx_idx, repetition] = (
            received_pilot / known_ref_seq[active_idx]
        )
        pilot_counts[csi_sample_idx, tx_idx] += 1

        noise_key = (csi_sample_idx, repetition)
        if noise_key not in measured_noise_positions:
            pilot_noise_power[csi_sample_idx, repetition] = np.mean(
                np.abs(received_symbol[:, noise_indices]) ** 2,
                axis=-1,
            ).astype(np.float32)
            measured_noise_positions.add(noise_key)

    expected_counts = np.full_like(pilot_counts, num_virtual_pilots)
    if not np.array_equal(pilot_counts, expected_counts):
        raise ValueError(
            "Each TX antenna in each CSI interval must have exactly "
            f"{num_virtual_pilots} virtual pilots; got {pilot_counts.tolist()}"
        )

    aligned_scalar = np.empty_like(h_scalar)
    virtual_pilot_common_phases = np.empty(
        h_scalar.shape,
        dtype=np.float32,
    )
    for csi_sample_idx in range(num_csi_samples):
        for tx_idx in range(num_tx_ant):
            aligned, common_phases = phase_align_channel_estimates(
                h_scalar[csi_sample_idx, tx_idx, :, :, None]
            )
            aligned_scalar[csi_sample_idx, tx_idx] = aligned[..., 0]
            virtual_pilot_common_phases[csi_sample_idx, tx_idx] = (
                common_phases
            )

    h_scalar_avg_raw = np.mean(h_scalar, axis=2).astype(np.complex64)
    h_scalar_avg_aligned = np.mean(
        aligned_scalar,
        axis=2,
    ).astype(np.complex64)

    # Under the paper's frequency-flat assumption, expose the scalar FDM
    # estimate on every active subcarrier so legacy full-band consumers keep
    # their established shapes. The scalar arrays remain authoritative.
    h_virtual_pilots = np.broadcast_to(
        h_scalar[..., None],
        h_scalar.shape + (N,),
    ).copy()
    h_virtual_pilots_aligned = np.broadcast_to(
        aligned_scalar[..., None],
        aligned_scalar.shape + (N,),
    ).copy()

    h_fd_virtual_avg_raw = np.mean(
        h_virtual_pilots,
        axis=2,
    ).astype(np.complex64)
    h_fd_virtual_avg_aligned = np.mean(
        h_virtual_pilots_aligned,
        axis=2,
    ).astype(np.complex64)
    phase_align_virtual_pilots = params.get(
        "phase_align_virtual_pilots",
        PHASE_ALIGN_VIRTUAL_PILOTS,
    )
    if phase_align_virtual_pilots:
        h_fd_for_equalization = h_fd_virtual_avg_aligned
    else:
        h_fd_for_equalization = h_fd_virtual_avg_raw

    # Per-TX, per-RX, per-subcarrier estimator variance across pilots. In FDM
    # mode this is the scalar estimator variance repeated over the active band.
    virtual_pilot_variance_raw = np.mean(
        np.abs(
            h_virtual_pilots
            - h_fd_virtual_avg_raw[:, :, None, :, :]
        ) ** 2,
        axis=2,
    ).astype(np.float32)
    virtual_pilot_variance_aligned = np.mean(
        np.abs(
            h_virtual_pilots_aligned
            - h_fd_virtual_avg_aligned[:, :, None, :, :]
        ) ** 2,
        axis=2,
    ).astype(np.float32)

    # Map each 5 ms CSI sample to the subframes in its own interval while
    # retaining the existing (RX, TX, subframe, slot, subcarrier) interface.
    h_fd = np.empty(
        (
            num_rx_ant,
            num_tx_ant,
            num_subframe_per_frame,
            num_slot_per_subframe,
            N,
        ),
        dtype=np.complex64,
    )
    for sf_idx in range(num_subframe_per_frame):
        csi_sample_idx = sf_idx // csi_sample_period_subframes
        h_fd_rx_tx = np.transpose(
            h_fd_for_equalization[csi_sample_idx],
            (1, 0, 2),
        )
        h_fd[:, :, sf_idx, :, :] = h_fd_rx_tx[:, :, None, :]

    # Baseline estimate from the first virtual pilot of every TX.
    h_single_fd = h_virtual_pilots[:, :, 0, :, :]
    h_single_fd_full = np.empty(
        (
            num_rx_ant,
            num_tx_ant,
            num_subframe_per_frame,
            num_slot_per_subframe,
            N,
        ),
        dtype=np.complex64,
    )
    for sf_idx in range(num_subframe_per_frame):
        csi_sample_idx = sf_idx // csi_sample_period_subframes
        h_single_fd_rx_tx = np.transpose(
            h_single_fd[csi_sample_idx],
            (1, 0, 2),
        )
        h_single_fd_full[:, :, sf_idx, :, :] = (
            h_single_fd_rx_tx[:, :, None, :]
        )

    # The received grid has no separable TX data axis. Keep this path focused
    # on channel sounding instead of indexing it with TX-domain pdsch_idx.
    del pdsch_idx
    iq_rcv = np.empty((num_rx_ant, 0), dtype=np.complex64)
    iq_rcv_single = np.empty((num_rx_ant, 0), dtype=np.complex64)
    correction_phase = np.zeros(
        (num_csi_samples, num_tx_ant, num_rx_ant),
        dtype=np.float32,
    )

    signal_power = np.abs(pilot_received_scalar) ** 2
    pilot_snr_repetitions_db = (
        10.0
        * np.log10(
            (signal_power + 1e-20)
            / (pilot_noise_power[:, None, :, :] + 1e-20)
        )
    ).astype(np.float32)
    single_pilot_snr_db = pilot_snr_repetitions_db[:, :, 0, :]
    virtual_pilot_snr_db = (
        10.0
        * np.log10(
            (np.mean(signal_power, axis=2) + 1e-20)
            / (
                np.mean(pilot_noise_power, axis=1)[:, None, :]
                / num_virtual_pilots
                + 1e-20
            )
        )
    ).astype(np.float32)
    snr_gain_db = virtual_pilot_snr_db - single_pilot_snr_db

    return (
        h_fd,
        h_single_fd_full,
        h_fd_virtual_avg_raw,
        h_fd_virtual_avg_aligned,
        h_virtual_pilots,
        h_virtual_pilots_aligned,
        virtual_pilot_variance_raw,
        virtual_pilot_variance_aligned,
        virtual_pilot_common_phases,
        single_pilot_snr_db,
        virtual_pilot_snr_db,
        snr_gain_db,
        correction_phase,
        iq_rcv,
        iq_rcv_single,
        h_scalar_avg_raw,
        h_scalar_avg_aligned,
        h_scalar,
        aligned_scalar,
        pilot_received_scalar,
        pilot_snr_repetitions_db,
        pilot_noise_power,
    )
