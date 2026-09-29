import numpy as np

carrier_frequency = 2.2e9
Tx_gain, Rx_gain = 30, 30

# Capture settings
BANDWIDTH = 1.4
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
NUM_VIRTUAL_PILOTS = 3
CSI_SAMPLE_PERIOD_SUBFRAMES = 5
VIRTUAL_PILOT_SYMBOL_START = 1
# Match the original training-data convention: preserve physical phase
# evolution and average the three repeated pilots without phase alignment.
# The aligned repetitions are still saved as a separate diagnostic array.
PHASE_ALIGN_VIRTUAL_PILOTS = False
SYNC_TX_IDX = 0
# All six antennas transmit three FDM pilots in the same OFDM symbols, each
# on its own subcarrier.
FDM_PILOT_GLOBAL_SLOT = 1
FDM_PILOT_CENTERED_BINS = (-30, -18, -6, 6, 18, 30)
normal_CP_time = 4.7e-6
first_CP_time = 5.2e-6
N_PSS = 62
num_RU = 3
num_tx_ant_per_RU = 2
num_tx_ant = num_RU * num_tx_ant_per_RU
num_rx_ant = 1
num_symbols_per_slot = 7
num_slot_per_subframe = 2
# One radio frame is also one CSI scheduling interval: 5 subframes = 5 ms.
num_subframe_per_frame = CSI_SAMPLE_PERIOD_SUBFRAMES
num_symbols_frame = num_symbols_per_slot * num_slot_per_subframe * num_subframe_per_frame

#channel parameters
channel_type = "multipath"

channel_delays = [0,1,4,6]
channel_gains_db = [0, -3, -10, -15]

#USRP Parameters
POWER = 4
modulation_order = 4

USRP_DEVICE_ARGS = (
    "addr0=192.168.10.2,addr1=192.168.11.2,"
    "addr2=192.168.12.2,addr3=192.168.13.2"
)
# The runtime pipeline opens the three RU radios and the UE radio in separate
# processes. UHD multi-device argument keys must be indexed as addr0, addr1,
# and so on; a single device uses addr.
TX_USRP_DEVICE_ARGS = (
    "addr0=192.168.10.2,addr1=192.168.11.2,addr2=192.168.12.2"
)
RX_USRP_DEVICE_ARGS = "addr=192.168.13.2"
CLOCK_SOURCE = "external"
TIME_SOURCE = "external"
TX_SUBDEV_SPEC = "A:0 B:0"
RX_SUBDEV_SPEC = "B:0"
TX_ANTENNA = "TX/RX"
RX_ANTENNA = "TX/RX"
TX_CHANNELS = tuple(range(num_tx_ant))
# Add channels here for multi-RX capture; the DSP path preserves this axis.
RX_CHANNELS = (0,)
WAIT_TIME = 0.2
TX_DELAY_TIME = 1e-4
RX_TRAILING_TIME = 1e-3
OTW_FORMAT = "sc16"
