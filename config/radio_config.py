import numpy as np

carrier_frequency = 2.2e9
Tx_gain, Rx_gain = 30, 30

bandwidth_options = {
    # BW [MHz] : (# active subcarriers, FFT size)
    1.4: (72, 128),
    3:   (180, 256),
    5:   (300, 512),
    10:  (600, 1024),
    15:  (900, 1536),
    20:  (1200, 2048),
    50:  (3000, 4096),
    400: (24000, 32768),
}

#resource grid parameters
DELTA_F = 15e3
cp_length = 9
NUM_VIRTUAL_PILOTS = 7
VIRTUAL_PILOT_SUBFRAME = 0
VIRTUAL_PILOT_SLOT = 1
VIRTUAL_PILOT_SYMBOL_START = 0
VIRTUAL_PILOT_SYMBOLS = list(range(VIRTUAL_PILOT_SYMBOL_START, VIRTUAL_PILOT_SYMBOL_START + NUM_VIRTUAL_PILOTS))
PHASE_ALIGN_VIRTUAL_PILOTS = True
normal_CP_time = 4.7e-6
first_CP_time = 5.2e-6
N_PSS = 62
num_symbols_per_slot = 7
num_slot_per_subframe = 2
num_subframe_per_frame = 10
num_symbols_frame = num_symbols_per_slot * num_slot_per_subframe * num_subframe_per_frame

#channel parameters
channel_type = "multipath"

channel_delays = [0,1,4,6]
channel_gains_db = [0, -3, -10, -15]

#USRP Parameters
POWER = 4
modulation_order = 4