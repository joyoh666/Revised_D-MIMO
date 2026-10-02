from USRP.build_frame import build_frame
from USRP.helper_functions import get_system_params, generate_known_reference_sequence
from config.radio_config import BANDWIDTH, REFERENCE_SEQUENCE_SEED, Rx_gain, RX_CHANNELS, OTW_FORMAT, carrier_frequency
from USRP.usrp_utils import USRPReceiver, USRPTransmitter
import uhd

bandwidth_mhz = BANDWIDTH 
params = get_system_params(bandwidth_mhz)
params = get_system_params(bandwidth_mhz)
reference_sequence_seed = (
        REFERENCE_SEQUENCE_SEED + int(10 * bandwidth_mhz)
    )
known_ref_seq = generate_known_reference_sequence(params["N"], reference_sequence_seed)

frames = []
for _ in range(5):
    frame = build_frame(params, known_ref_seq)
    frames.append(frame)

# USRP initialization
usrp = uhd.usrp.MultiUSRP(
    "addr0=192.168.10.2,second_addr=192.168.11.2,third_addr=192.168.12.2,fourth_addr=192.168.13.2"
)


    