import numpy as np

carrier_frequency = 2.2e9
Tx_gain, Rx_gain = 30, 30

# Capture settings
BANDWIDTH = 1.4
CAPTURE_BANDWIDTHS = (BANDWIDTH,)
NUM_CHANNEL_CAPTURES = 50
CSI_WINDOW_LENGTH = 5
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
# Each enabled TX antenna transmits on its own subcarrier.  Keep the complete
# three-RU allocation here and select only the bins needed by ``num_RU``.
FDM_PILOT_GLOBAL_SLOT = 1
ALL_FDM_PILOT_CENTERED_BINS = (-30, -18, -6, 6, 18, 30)
normal_CP_time = 4.7e-6
first_CP_time = 5.2e-6
N_PSS = 62
num_RU = 1
num_tx_ant_per_RU = 2
if num_RU not in (1, 2, 3):
    raise ValueError("num_RU must be 1, 2, or 3")
num_tx_ant = num_RU * num_tx_ant_per_RU
FDM_PILOT_CENTERED_BINS = ALL_FDM_PILOT_CENTERED_BINS[:num_tx_ant]
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

RU_USRP_ADDRESSES = (
    "192.168.10.2",
    "192.168.11.2",
    "192.168.12.2",
)
RX_USRP_ADDRESS = "192.168.13.2"
SELECTED_RU_USRP_ADDRESSES = RU_USRP_ADDRESSES[:num_RU]

# The runtime pipeline opens the selected RU radios and the UE radio in
# separate processes. UHD uses ``addr`` for one device and indexed ``addrN``
# keys when several devices form one MultiUSRP.
TX_USRP_DEVICE_ARGS = (
    f"addr={SELECTED_RU_USRP_ADDRESSES[0]}"
    if num_RU == 1
    else ",".join(
        f"addr{index}={address}"
        for index, address in enumerate(SELECTED_RU_USRP_ADDRESSES)
    )
)
RX_USRP_DEVICE_ARGS = f"addr={RX_USRP_ADDRESS}"

# Legacy single-process device list retained for ``main.py``.  New captures
# should use ``python -m runtime.runner``, which uses the TX/RX arguments above.
USRP_DEVICE_ARGS = ",".join(
    f"addr{index}={address}"
    for index, address in enumerate(
        SELECTED_RU_USRP_ADDRESSES + (RX_USRP_ADDRESS,)
    )
)
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
