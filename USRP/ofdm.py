import numpy as np

def ofdm_modulate(params, resource_maps):
    num_subframe_per_frame = params["num_subframe_per_frame"]
    num_slot_per_subframe = params["num_slot_per_subframe"]
    num_symbols_per_slot = params["num_symbols_per_slot"]
    num_tx_ant = params["num_tx_ant"]
    FFT_SIZE = params["FFT_SIZE"]
    N = params["N"]
    first_CP_length = params["first_CP_length"]
    normal_CP_length = params["normal_CP_length"]

    fft_symbols = np.zeros(
        (num_tx_ant, num_subframe_per_frame, num_slot_per_subframe, num_symbols_per_slot, FFT_SIZE),
        dtype=np.complex64
    )

    fft_symbols[..., 1:N // 2 + 1] = resource_maps[..., N // 2:]
    fft_symbols[..., -(N // 2):] = resource_maps[..., :N // 2]

    td_symbols = np.fft.ifft(fft_symbols, axis=-1, norm="ortho").astype(np.complex64)

    td_symbols_with_cp_0th = np.concatenate(
        [
            td_symbols[..., 0:1, -first_CP_length:],
            td_symbols[..., 0:1, :],
        ],
        axis=-1
    )

    td_symbols_with_cp_normal = np.concatenate(
            [
                td_symbols[..., 1:, -normal_CP_length:],
                td_symbols[..., 1:, :],
            ],
            axis=-1
        )

    td_symbols_with_cp = np.concatenate(
        [
            np.reshape(
                td_symbols_with_cp_0th,
                (num_tx_ant, num_subframe_per_frame, num_slot_per_subframe, -1)
            ),
            np.reshape(
                td_symbols_with_cp_normal,
                (num_tx_ant, num_subframe_per_frame, num_slot_per_subframe, -1)
            )
        ],
        axis=-1
    )

    return td_symbols_with_cp, td_symbols_with_cp_normal

def ofdm_demodulate(frame_rcv, params):
    num_subframe_per_frame = params["num_subframe_per_frame"]
    num_slot_per_subframe = params["num_slot_per_subframe"]
    num_symbols_per_slot = params["num_symbols_per_slot"]
    FFT_SIZE = params["FFT_SIZE"]
    first_CP_length = params["first_CP_length"]
    normal_CP_length = params["normal_CP_length"]
    N = params["N"]
    num_rx_ant = frame_rcv.shape[0]

    slots_rcv = np.reshape(
        frame_rcv,
        (num_rx_ant, num_subframe_per_frame, num_slot_per_subframe, -1),
    )

    # Drop only the excess part of the longer first-symbol CP. Every symbol
    # in a slot then has the common length normal_CP_length + FFT_SIZE.
    slots_rcv = slots_rcv[..., first_CP_length - normal_CP_length:]
    symbols_with_cp = np.reshape(
        slots_rcv,
        (
            num_rx_ant,
            num_subframe_per_frame,
            num_slot_per_subframe,
            num_symbols_per_slot,
            normal_CP_length + FFT_SIZE,
        ),
    )
    symbols_rcv = symbols_with_cp[..., normal_CP_length:]

    fft_symbols_rcv = np.fft.fft(symbols_rcv, axis=-1, norm="ortho")

    resource_maps_rcv = np.zeros(
        (
            num_rx_ant,
            num_subframe_per_frame,
            num_slot_per_subframe,
            num_symbols_per_slot,
            N,
        ),
        dtype=np.complex64,
    )
    resource_maps_rcv[..., N // 2:] = fft_symbols_rcv[..., 1:N // 2 + 1]
    resource_maps_rcv[..., :N // 2] = fft_symbols_rcv[..., -(N // 2):]

    return resource_maps_rcv
