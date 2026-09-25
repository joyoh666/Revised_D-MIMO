import numpy as np

from config.radio_config import PHASE_ALIGN_VIRTUAL_PILOTS
from .helper_functions import (
    phase_align_channel_estimates,
    pilot_snr_db_from_equalized,
)


def channelestimation(params, resource_maps_rcv, known_ref_seq, pdsch_idx):
    """Estimate every TX-to-RX channel from TDM virtual pilots.

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

    h_virtual_pilots_by_tx = [[] for _ in range(num_tx_ant)]
    virtual_pilot_symbols_by_tx = [[] for _ in range(num_tx_ant)]

    for position in params["virtual_pilot_positions"]:
        tx_idx, sf_idx, slot_idx, sym_idx = position
        if not (
            0 <= tx_idx < num_tx_ant
            and 0 <= sf_idx < num_subframe_per_frame
            and 0 <= slot_idx < num_slot_per_subframe
            and 0 <= sym_idx < num_symbols_per_slot
        ):
            raise ValueError(f"Invalid virtual-pilot position: {position}")

        pilot_received = resource_maps_rcv[:, sf_idx, slot_idx, sym_idx, :]
        h_virtual_pilots_by_tx[tx_idx].append(
            pilot_received / known_ref_seq[None, :]
        )
        virtual_pilot_symbols_by_tx[tx_idx].append(pilot_received)

    pilot_counts = [len(pilots) for pilots in h_virtual_pilots_by_tx]
    expected_counts = [num_virtual_pilots] * num_tx_ant
    if pilot_counts != expected_counts:
        raise ValueError(
            "Each TX antenna must have exactly "
            f"{num_virtual_pilots} virtual pilots; got {pilot_counts}"
        )

    # (num_tx_ant, num_virtual_pilots, num_rx_ant, N)
    h_virtual_pilots = np.stack(
        [np.stack(pilots, axis=0) for pilots in h_virtual_pilots_by_tx],
        axis=0,
    ).astype(np.complex64)
    virtual_pilot_received = np.stack(
        [
            np.stack(pilots, axis=0)
            for pilots in virtual_pilot_symbols_by_tx
        ],
        axis=0,
    ).astype(np.complex64)

    # Average only the repeated-pilot axis.
    # Shape: (num_tx_ant, num_rx_ant, N)
    h_fd_virtual_avg_raw = np.mean(h_virtual_pilots, axis=1).astype(np.complex64)
    aligned_per_tx = []
    phases_per_tx = []
    for tx_idx in range(num_tx_ant):
        aligned, common_phases = phase_align_channel_estimates(
            h_virtual_pilots[tx_idx]
        )
        aligned_per_tx.append(aligned)
        phases_per_tx.append(common_phases)
    h_virtual_pilots_aligned = np.stack(
        aligned_per_tx,
        axis=0,
    ).astype(np.complex64)
    # Shape: (num_tx_ant, num_virtual_pilots, num_rx_ant)
    virtual_pilot_common_phases = np.stack(
        phases_per_tx,
        axis=0,
    ).astype(np.float32)

    h_fd_virtual_avg_aligned = np.mean(
        h_virtual_pilots_aligned,
        axis=1,
    ).astype(np.complex64)
    phase_align_virtual_pilots = params.get(
        "phase_align_virtual_pilots",
        PHASE_ALIGN_VIRTUAL_PILOTS,
    )
    if phase_align_virtual_pilots:
        h_fd_for_equalization = h_fd_virtual_avg_aligned
    else:
        h_fd_for_equalization = h_fd_virtual_avg_raw

    # Per-TX, per-RX, per-subcarrier estimator variance across pilots.
    virtual_pilot_variance_raw = np.mean(
        np.abs(
            h_virtual_pilots
            - h_fd_virtual_avg_raw[:, None, :, :]
        ) ** 2,
        axis=1,
    ).astype(np.float32)
    virtual_pilot_variance_aligned = np.mean(
        np.abs(
            h_virtual_pilots_aligned
            - h_fd_virtual_avg_aligned[:, None, :, :]
        ) ** 2,
        axis=1,
    ).astype(np.float32)

    # Expose the full channel using conventional (RX, TX, ...) ordering.
    h_fd_rx_tx = np.transpose(h_fd_for_equalization, (1, 0, 2))
    h_fd = np.broadcast_to(
        h_fd_rx_tx[:, :, None, None, :],
        (
            num_rx_ant,
            num_tx_ant,
            num_subframe_per_frame,
            num_slot_per_subframe,
            N,
        ),
    ).copy()

    # Baseline estimate from the first virtual pilot of every TX.
    h_single_fd = h_virtual_pilots[:, 0, :, :]
    h_single_fd_rx_tx = np.transpose(h_single_fd, (1, 0, 2))
    h_single_fd_full = np.broadcast_to(
        h_single_fd_rx_tx[:, :, None, None, :],
        (
            num_rx_ant,
            num_tx_ant,
            num_subframe_per_frame,
            num_slot_per_subframe,
            N,
        ),
    ).copy()

    # The received grid has no separable TX data axis. Keep this path focused
    # on channel sounding instead of indexing it with TX-domain pdsch_idx.
    del pdsch_idx
    iq_rcv = np.empty((num_rx_ant, 0), dtype=np.complex64)
    iq_rcv_single = np.empty((num_rx_ant, 0), dtype=np.complex64)
    correction_phase = np.zeros(
        (num_tx_ant, num_rx_ant),
        dtype=np.float32,
    )

    # Pilot-domain quality metrics are evaluated independently per TX.
    single_pilot_snr_db = []
    virtual_pilot_snr_db = []
    for tx_idx in range(num_tx_ant):
        h_gain_raw = virtual_pilot_received[tx_idx] / (
            h_fd_for_equalization[tx_idx][None, ...] + 1e-12
        )
        h_gain_single = virtual_pilot_received[tx_idx] / (
            h_single_fd[tx_idx][None, ...] + 1e-12
        )
        single_pilot_snr_db.append(
            pilot_snr_db_from_equalized(h_gain_single, known_ref_seq)
        )
        virtual_pilot_snr_db.append(
            pilot_snr_db_from_equalized(h_gain_raw, known_ref_seq)
        )
    single_pilot_snr_db = np.asarray(
        single_pilot_snr_db,
        dtype=np.float32,
    )
    virtual_pilot_snr_db = np.asarray(
        virtual_pilot_snr_db,
        dtype=np.float32,
    )
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
    )
