import numpy as np
from scipy.signal import correlate
from .ofdm import ofdm_demodulate
from .channelestimation import channelestimation


def active_to_full_spectrum(channel_active, N, FFT_SIZE):
    """Map active-subcarrier channels to FFT bins along the last axis."""
    channel_active = np.asarray(channel_active, dtype=np.complex64)
    if channel_active.shape[-1] != N:
        raise ValueError(
            f"channel_active has {channel_active.shape[-1]} subcarriers, "
            f"expected {N}"
        )

    full_spectrum = np.zeros(
        channel_active.shape[:-1] + (FFT_SIZE,),
        dtype=np.complex64,
    )
    full_spectrum[..., 1:N // 2 + 1] = channel_active[..., N // 2:]
    full_spectrum[..., -(N // 2):] = channel_active[..., :N // 2]
    return full_spectrum


def channel_to_delay_response(channel_active, N, FFT_SIZE):
    """Return the channel impulse response while preserving leading axes."""
    full_spectrum = active_to_full_spectrum(channel_active, N, FFT_SIZE)
    impulse_response = np.fft.ifft(
        full_spectrum,
        n=FFT_SIZE,
        axis=-1,
        norm="ortho",
    )
    return (
        impulse_response.astype(np.complex64),
        full_spectrum.astype(np.complex64),
    )


def delay_residual_power(impulse_response, guard_taps=2):
    """Measure main- and residual-tap power along the last array axis.

    Leading axes, such as the RX antenna axis, are preserved. A one-dimensional
    input returns scalar values for compatibility with the original script.
    """
    impulse_response = np.asarray(impulse_response)
    if impulse_response.ndim == 0:
        raise ValueError("impulse_response must have a delay-sample axis")
    if guard_taps < 0:
        raise ValueError("guard_taps must be non-negative")
    if impulse_response.shape[-1] == 0:
        empty_shape = impulse_response.shape[:-1]
        zeros = np.zeros(empty_shape, dtype=np.float64)
        reductions = np.full(empty_shape, -np.inf, dtype=np.float64)
        if not empty_shape:
            return 0.0, 0.0, -np.inf
        return zeros, zeros.copy(), reductions

    power = np.abs(impulse_response) ** 2
    flat_power = power.reshape(-1, power.shape[-1])
    main_power = np.empty(flat_power.shape[0], dtype=np.float64)
    residual_power = np.empty(flat_power.shape[0], dtype=np.float64)

    for row_index, row_power in enumerate(flat_power):
        peak_index = int(np.argmax(row_power))
        lower = max(0, peak_index - guard_taps)
        upper = min(row_power.size, peak_index + guard_taps + 1)
        main_power[row_index] = float(np.mean(row_power[lower:upper]) + 1e-15)
        residual_samples = np.concatenate(
            (row_power[:lower], row_power[upper:])
        )
        residual_power[row_index] = float(
            (np.mean(residual_samples) if residual_samples.size else 0.0)
            + 1e-15
        )

    reduction_db = 10.0 * np.log10(main_power / residual_power)
    leading_shape = power.shape[:-1]
    main_power = main_power.reshape(leading_shape)
    residual_power = residual_power.reshape(leading_shape)
    reduction_db = reduction_db.reshape(leading_shape)
    if not leading_shape:
        return (
            float(main_power),
            float(residual_power),
            float(reduction_db),
        )
    return main_power, residual_power, reduction_db


def synchronize_frame_timing(frame_rcv_ifosync, ss_td_with_cp, params):
    """Find the frame start from the TX0 synchronization symbols.

    The same timing index is used for all RX chains.  Their correlation powers
    are combined before locating the peak, while the per-RX peak magnitudes
    are returned for diagnostics.
    """
    frame_rcv_ifosync = np.asarray(frame_rcv_ifosync, dtype=np.complex64)
    if frame_rcv_ifosync.ndim != 2:
        raise ValueError(
            "frame_rcv_ifosync must have shape (num_rx_ant, num_samples)"
        )

    ss_td_with_cp = np.asarray(ss_td_with_cp, dtype=np.complex64).reshape(-1)
    if ss_td_with_cp.size == 0:
        raise ValueError("ss_td_with_cp must not be empty")
    if frame_rcv_ifosync.shape[1] < ss_td_with_cp.size:
        raise ValueError("received frame is shorter than the sync sequence")

    FFT_SIZE = int(params["FFT_SIZE"])
    first_CP_length = int(params["first_CP_length"])
    normal_CP_length = int(params["normal_CP_length"])
    frame_length = int(params["frame_length"])

    correlations = np.stack(
        [
            correlate(
                frame_rcv_ifosync[rx_ant_idx],
                ss_td_with_cp,
                mode="valid",
                method="fft",
            )
            for rx_ant_idx in range(frame_rcv_ifosync.shape[0])
        ]
    )
    combined_correlation_power = np.sum(
        np.abs(correlations) ** 2,
        axis=0,
    )
    max_corr_idx = int(np.argmax(combined_correlation_power))
    timing_correlation_peaks = np.abs(
        correlations[:, max_corr_idx]
    ).astype(np.float32)

    # The timing template begins at symbol 5 and contains SSS and PSS. Convert
    # that template position back to the beginning of the frame.
    ss_start_idx = (
        FFT_SIZE * 5
        + first_CP_length
        + normal_CP_length * 4
    )
    sync_idx = int(max_corr_idx - ss_start_idx)
    if (
        sync_idx < 0
        or sync_idx + frame_length > frame_rcv_ifosync.shape[1]
    ):
        print("sync not found, using sync_idx = 0")
        sync_idx = 0

    frame_rcv_timesync = frame_rcv_ifosync[
        :,
        sync_idx:sync_idx + frame_length,
    ]
    if frame_rcv_timesync.shape[1] < frame_length:
        pad_length = frame_length - frame_rcv_timesync.shape[1]
        frame_rcv_timesync = np.concatenate(
            (
                frame_rcv_timesync.astype(np.complex64, copy=False),
                np.zeros(
                    (frame_rcv_ifosync.shape[0], pad_length),
                    dtype=np.complex64,
                ),
            ),
            axis=1,
        )

    return frame_rcv_timesync, sync_idx, timing_correlation_peaks


def estimate_and_correct_ffo(frame_rcv_timesync, params):
    """Estimate and correct fractional frequency offset using all CPs."""
    frame_rcv_timesync = np.asarray(frame_rcv_timesync, dtype=np.complex64)
    if frame_rcv_timesync.ndim != 2:
        raise ValueError(
            "frame_rcv_timesync must have shape (num_rx_ant, frame_length)"
        )

    FFT_SIZE = int(params["FFT_SIZE"])
    first_CP_length = int(params["first_CP_length"])
    normal_CP_length = int(params["normal_CP_length"])
    num_symbols_per_slot = int(params["num_symbols_per_slot"])
    sampling_rate = float(params["sampling_rate"])
    slot_length = int(params["slot_length"])
    num_slots = (
        int(params["num_subframe_per_frame"])
        * int(params["num_slot_per_subframe"])
    )
    expected_frame_length = num_slots * slot_length
    if frame_rcv_timesync.shape[1] != expected_frame_length:
        raise ValueError(
            f"frame_rcv_timesync has {frame_rcv_timesync.shape[1]} samples, "
            f"expected {expected_frame_length}"
        )

    slot_waveforms = frame_rcv_timesync.reshape(
        frame_rcv_timesync.shape[0],
        num_slots,
        slot_length,
    )
    first_cps = slot_waveforms[:, :, :first_CP_length]
    first_symbol_tails = slot_waveforms[
        :,
        :,
        FFT_SIZE:FFT_SIZE + first_CP_length,
    ]

    remaining_symbols = slot_waveforms[
        :,
        :,
        FFT_SIZE + first_CP_length:,
    ].reshape(
        frame_rcv_timesync.shape[0],
        num_slots * (num_symbols_per_slot - 1),
        FFT_SIZE + normal_CP_length,
    )
    other_cps = remaining_symbols[:, :, :normal_CP_length]
    other_symbol_tails = remaining_symbols[:, :, -normal_CP_length:]

    first_cp_correlation = np.sum(
        np.conj(first_cps) * first_symbol_tails,
        axis=(1, 2),
    )
    other_cp_correlation = np.sum(
        np.conj(other_cps) * other_symbol_tails,
        axis=(1, 2),
    )
    phase_difference = np.angle(
        first_cp_correlation + other_cp_correlation
    )
    ffo_hz = (
        phase_difference * sampling_rate / (2.0 * np.pi * FFT_SIZE)
    ).astype(np.float32)

    sample_indices = np.arange(
        expected_frame_length,
        dtype=np.float64,
    )[None, :]
    compensation_ffo = np.exp(
        -1j
        * 2.0
        * np.pi
        * ffo_hz[:, None]
        * sample_indices
        / sampling_rate
    ).astype(np.complex64)
    frame_rcv_ffosync = frame_rcv_timesync * compensation_ffo

    return frame_rcv_ffosync, ffo_hz


def estimate_and_correct_rfo(frame_rcv_ffosync, params):
    """Estimate residual CFO from TX0's symbol-0 references.

    Symbol 0 of every slot is transmitted by the same RF chain (TX0), so the
    static TX0-to-RX channel phase cancels when adjacent reference symbols are
    correlated.  One residual-frequency estimate is formed per RX chain by
    coherently combining all adjacent-slot correlations.  That estimate is
    then applied with a continuous sample index over the complete frame.

    Returns
    -------
    frame_rcv_rfosync : np.ndarray
        RFO-corrected frame with shape ``(num_rx_ant, frame_length)``.
    rfo_hz : np.ndarray
        Frame-level RFO estimate for every RX chain.
    rfo_hz_per_slot : np.ndarray
        Adjacent-slot diagnostic estimates.  The last estimate is repeated so
        that the slot axis has the same length as the frame.
    """
    frame_rcv_ffosync = np.asarray(frame_rcv_ffosync, dtype=np.complex64)
    if frame_rcv_ffosync.ndim != 2:
        raise ValueError(
            "frame_rcv_ffosync must have shape (num_rx_ant, frame_length)"
        )

    FFT_SIZE = int(params["FFT_SIZE"])
    first_CP_length = int(params["first_CP_length"])
    slot_length = int(params["slot_length"])
    sampling_rate = float(params["sampling_rate"])
    num_slots = (
        int(params["num_subframe_per_frame"])
        * int(params["num_slot_per_subframe"])
    )
    expected_frame_length = num_slots * slot_length
    if frame_rcv_ffosync.shape[1] != expected_frame_length:
        raise ValueError(
            f"frame_rcv_ffosync has {frame_rcv_ffosync.shape[1]} samples, "
            f"expected {expected_frame_length}"
        )
    if num_slots < 2:
        raise ValueError("RFO estimation requires at least two slots")

    slot_waveforms = frame_rcv_ffosync.reshape(
        frame_rcv_ffosync.shape[0],
        num_slots,
        slot_length,
    )
    reference_symbols = slot_waveforms[
        :,
        :,
        first_CP_length:first_CP_length + FFT_SIZE,
    ]

    adjacent_slot_correlations = np.sum(
        np.conj(reference_symbols[:, :-1, :])
        * reference_symbols[:, 1:, :],
        axis=-1,
    )
    frequency_scale = sampling_rate / (2.0 * np.pi * slot_length)
    rfo_hz_per_pair = (
        np.angle(adjacent_slot_correlations) * frequency_scale
    )
    rfo_hz_per_slot = np.concatenate(
        (rfo_hz_per_pair, rfo_hz_per_pair[:, -1:]),
        axis=1,
    ).astype(np.float32)

    # A magnitude-weighted circular average avoids averaging wrapped phase
    # angles directly.  The remaining offset after CP-based FFO correction is
    # expected to lie inside the adjacent-slot unambiguous range.
    combined_correlation = np.sum(adjacent_slot_correlations, axis=1)
    rfo_hz = (
        np.angle(combined_correlation) * frequency_scale
    ).astype(np.float32)

    sample_indices = np.arange(
        expected_frame_length,
        dtype=np.float64,
    )[None, :]
    compensation_rfo = np.exp(
        -1j
        * 2.0
        * np.pi
        * rfo_hz[:, None]
        * sample_indices
        / sampling_rate
    ).astype(np.complex64)
    frame_rcv_rfosync = frame_rcv_ffosync * compensation_rfo

    return frame_rcv_rfosync, rfo_hz, rfo_hz_per_slot


def postprocess(params,
                ss_td_with_cp,
                pdsch_idx,
                known_ref_seq,
                frame_rcv):

    # Shape: (num_rx_ant, num_received_samples).
    frame_rcv = np.asarray(frame_rcv, dtype=np.complex64)
    if frame_rcv.ndim == 1:
        frame_rcv = frame_rcv.reshape(1, -1)
    if frame_rcv.ndim != 2:
        raise ValueError(
            "frame_rcv must have shape (num_rx_ant, num_samples)"
        )
    if frame_rcv.shape[1] <= 10:
        raise ValueError("frame_rcv must contain more than 10 samples")
    frame_rcv = frame_rcv[:, 10:].copy()
    frame_rcv -= np.mean(frame_rcv, axis=1, keepdims=True)

    # Integer FO sync left disabled as in original code
    frame_rcv_ifosync = frame_rcv

    (
        frame_rcv_timesync,
        sync_idx,
        timing_correlation_peaks,
    ) = synchronize_frame_timing(
        frame_rcv_ifosync,
        ss_td_with_cp,
        params,
    )

    frame_rcv_ffosync, ffo_hz = estimate_and_correct_ffo(
        frame_rcv_timesync,
        params,
    )

    # TX0 alone transmits symbol 0 in every slot.  Use those repeated symbols
    # to estimate one residual frequency offset and compensate continuously
    # over the complete frame.
    (
        frame_rcv_rfosync,
        rfo_hz,
        rfo_hz_per_slot,
    ) = estimate_and_correct_rfo(frame_rcv_ffosync, params)

    #Frequency-domain demodulation
    resource_maps_rcv = ofdm_demodulate(frame_rcv_rfosync, params)

    # Channel estimation using consecutive identical pilots.
    (
        h_fd,
        h_single_fd_full,
        h_fd_virtual_avg_raw,
        h_fd_virtual_avg,
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
        iq_rcv_single
    ) = channelestimation(params, resource_maps_rcv, known_ref_seq, pdsch_idx)

    return {
            "frame_rcv": frame_rcv,
            "frame_rcv_timesync": frame_rcv_timesync,
            "frame_rcv_rfosync": frame_rcv_rfosync,
            "resource_maps_rcv": resource_maps_rcv,
            "h_fd": h_fd.astype(np.complex64),
            "h_single_fd": h_single_fd_full.astype(np.complex64),
            "h_fd_virtual_avg_raw": h_fd_virtual_avg_raw.astype(np.complex64),
            "h_fd_virtual_avg_aligned": h_fd_virtual_avg.astype(np.complex64),
            "h_virtual_pilots_fd": h_virtual_pilots,
            "h_virtual_pilots_fd_aligned": h_virtual_pilots_aligned,
            "virtual_pilot_variance_raw": virtual_pilot_variance_raw,
            "virtual_pilot_variance_aligned": virtual_pilot_variance_aligned,
            "virtual_pilot_common_phases_rad": virtual_pilot_common_phases,
            "single_pilot_snr_db": single_pilot_snr_db,
            "virtual_pilot_snr_db": virtual_pilot_snr_db,
            "virtual_pilot_snr_gain_db": snr_gain_db,
            "virtual_pilot_correction_phase_rad": np.asarray(
                correction_phase,
                dtype=np.float32,
            ),
            "iq_rcv": iq_rcv.astype(np.complex64),
            "iq_rcv_single": iq_rcv_single.astype(np.complex64),
            "sync_idx": sync_idx,
            "ffo_hz": np.asarray(ffo_hz, dtype=np.float32),
            "rfo_hz": np.asarray(rfo_hz, dtype=np.float32),
            "rfo_hz_per_slot": np.asarray(
                rfo_hz_per_slot,
                dtype=np.float32,
            ),
            "timing_correlation_peak": timing_correlation_peaks,
        }
