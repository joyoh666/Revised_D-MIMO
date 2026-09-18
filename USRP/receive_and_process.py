from .usrp_utils import sendAndReceive

def receive_and_process(
        usrp,
        params,
        waveform,
        ss_td_with_cp,
        pdsch_idx,
        known_ref_seq
):
    N = params["N"]
    FFT_SIZE = params["FFT_SIZE"]
    sampling_rate = params["sampling_rate"]
    frame_length = params["frame_length"]
    slot_length = params["slot_length"]
    first_CP_length = params["first_CP_length"]
    normal_CP_length = params["normal_CP_length"]
    Tx_gain = params["Tx_gain"]
    Rx_gain = params["Rx_gain"]

    frame_rcv = sendAndReceive(
        
    )