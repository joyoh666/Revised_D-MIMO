import numpy as np

carrier_frequency = 2.2e9
Tx_gain, Rx_gain = 30, 30

# Capture settings
BANDWIDTH = 10
CAPTURE_BANDWIDTHS = (BANDWIDTH,)
NUM_CHANNEL_CAPTURES = 50
REFERENCE_SEQUENCE_SEED = 2026
OUTPUT_DIR = "saved_channel_estimates"

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
NUM_VIRTUAL_PILOTS = 6
VIRTUAL_PILOT_SUBFRAME = 0
VIRTUAL_PILOT_SLOT = 1
VIRTUAL_PILOT_SYMBOL_START = 1
VIRTUAL_PILOT_SYMBOLS = list(range(VIRTUAL_PILOT_SYMBOL_START, VIRTUAL_PILOT_SYMBOL_START + NUM_VIRTUAL_PILOTS))
PHASE_ALIGN_VIRTUAL_PILOTS = True
SYNC_TX_IDX = 0
VIRTUAL_PILOT_GLOBAL_SLOTS = (
    1,   # TX0: SF0, slot1
    3,   # TX1: SF1, slot1
    5,   # TX2: SF2, slot1
    7,   # TX3: SF3, slot1
    9,   # TX4: SF4, slot1
    11,  # TX5: SF5, slot1
)
normal_CP_time = 4.7e-6
first_CP_time = 5.2e-6
N_PSS = 62
num_RU = 3
num_tx_ant_per_RU = 2
num_tx_ant = num_RU * num_tx_ant_per_RU
num_rx_ant = 1
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

USRP_DEVICE_ARGS = (
    "addr0=192.168.10.2,second_addr=192.168.11.2,"
    "third_addr=192.168.12.2,fourth_addr=192.168.13.2"
)
TX_SUBDEV_SPEC = "A:0"
RX_SUBDEV_SPEC = "B:0"
TX_ANTENNA = "TX/RX"
RX_ANTENNA = "TX/RX"
TX_CHANNELS = (0, 1)
# Add channels here for multi-RX capture; the DSP path preserves this axis.
RX_CHANNELS = (0,)
WAIT_TIME = 0.2
TX_DELAY_TIME = 1e-4
RX_TRAILING_TIME = 1e-3
OTW_FORMAT = "sc16"
