import numpy as np
from scipy.signal import correlate
def postprocess(params,
                ss_td_with_cp,
                pdsch_idx,
                known_ref_seq,
                frame_rcv):

    N = params["N"]
    FFT_SIZE = params["FFT_SIZE"]
    frame_length = params["frame_length"]
    slot_length = params["slot_length"]
    first_CP_length = params["first_CP_length"]
    normal_CP_length = params["normal_CP_length"]
    num_subframe_per_frame = params["num_subframe_per_frame"]
    num_slot_per_subframe = params["num_slot_per_subframe"]
    num_symbols_per_slot = params["num_symbols_per_slot"]
    sampling_rate = params["sampling_rate"]

    #shape : (num_antennas, frame_length)
    frame_rcv = frame_rcv[:, 10:]
    frame_rcv -= np.mean(frame_rcv, axis=1, keepdims=True)

    # Integer FO sync left disabled as in original code
    frame_rcv_ifosync = frame_rcv

    num_ant = frame_rcv_ifosync.shape[0]

    #Timing synchronization
    corr = np.stack([
        correlate(frame_rcv_ifosync[ant], ss_td_with_cp,
                  mode="valid", method="fft")
        for ant in range(num_ant)
    ])

    max_corr_idx = np.argmax(np.sum(np.abs(corr)**2, axis=0))
    #SSS is placed in symbol 5 & PSS is placed in symbol 6
    ss_start_idx = FFT_SIZE * 5 + first_CP_length + normal_CP_length * 4
    sync_idx = int(max_corr_idx - ss_start_idx)

    if sync_idx + frame_length > frame_rcv.size or sync_idx < 0:
        print("sync not found, using sync_idx = 0")
        sync_idx = 0

    frame_rcv_timesync = frame_rcv_ifosync[:, sync_idx:sync_idx + frame_length]
    if frame_rcv_timesync[1].size < frame_length:
        pad = frame_length - frame_rcv_timesync[1].size
        frame_rcv_timesync = np.concatenate(
            [frame_rcv_timesync.astype(np.complex64), np.zeros((num_ant, pad), dtype=np.complex64)],
            axis=1
        )

    # Fractional frequency offset synchronization by CP correlation
    td_symbols_rcv = np.reshape(
        frame_rcv_timesync,
        (num_ant, num_subframe_per_frame * num_slot_per_subframe, -1),
        )

    first_CPs = td_symbols_rcv[:, :, :first_CP_length]
    first_signal_for_CPs = td_symbols_rcv[:, :, FFT_SIZE: FFT_SIZE + first_CP_length]
    
    td_symbols_rcv_wo_first = td_symbols_rcv[:, :, FFT_SIZE + first_CP_length:]
    td_symbols_rcv_wo_first = np.reshape(
        td_symbols_rcv_wo_first,
        (num_ant, num_subframe_per_frame * num_slot_per_subframe * (num_symbols_per_slot - 1), -1)
    )
    other_CPs = td_symbols_rcv_wo_first[:, :, :normal_CP_length]
    other_signals_for_CPs = td_symbols_rcv_wo_first[:,:, -normal_CP_length:]

    first_cp_corr = np.sum(
        np.conj(first_CPs) * first_signal_for_CPs,
        axis=(1,2)
    )
    other_cp_corr = np.sum(
        np.conj(other_CPs) * other_signals_for_CPs,
        axis=(1,2)
    )
    phase_diff = np.angle(other_cp_corr + first_cp_corr)

    sample_idx_diff = FFT_SIZE
    ffo = phase_diff * sampling_rate / (2 * np.pi * sample_idx_diff)

    n = np.arange(frame_rcv_timesync.shape[1], dtype=np.float64)[None, :]
    compensation_ffo = np.exp(-1j * ffo[:, None] * n / sampling_rate * 2 * np.pi).astype(np.complex64)
    frame_rcv_ffosync = frame_rcv_timesync * compensation_ffo

    # Residual frequency offset estimation with known reference symbol (symbol 0)
    refsym_rcv_td = np.reshape(
        frame_rcv_ffosync,
        (num_ant, num_subframe_per_frame * num_slot_per_subframe, -1)
    )
    refsym_rcv_td = refsym_rcv_td[:, :, first_CP_length:first_CP_length + FFT_SIZE]

    phase_diff = np.angle(
        np.sum(np.conj(refsym_rcv_td[ant, :-1]) * refsym_rcv_td[ant, 1:], axis=-1) 
        for ant in range(num_ant)
    )
    phase_diff = np.reshape(phase_diff, (num_ant, -1))

    sample_idx_diff = slot_length
    rfo = phase_diff * sampling_rate / (2 * np.pi * sample_idx_diff)
    rfo = np.concatenate([rfo, rfo[:, -1:]], axis=0)
    rfo = np.expand_dims(rfo, axis=1)

    n = np.expand_dims(np.arange(slot_length), axis=0)
    compensation_rfo = np.exp(-1j * rfo * n / sampling_rate * 2 * np.pi).astype(np.complex64)
    compensation_rfo = compensation_rfo.flatten()
    frame_rcv_rfosync = frame_rcv_ffosync * compensation_rfo

