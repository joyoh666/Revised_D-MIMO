"""Single-X310, simultaneous TX/RX sounding worker.

One MultiUSRP object is shared by TX and RX streamers. A receiver thread runs
concurrently with timed TX streaming; only the postprocessor is a subprocess.
"""
import threading
import traceback
import numpy as np

from .receiver import USRPReceiver
from .transmitter import USRPTransmitter, build_precomputed_frames


def capture_single_usrp(
    *, frames_queue, stop_event, params, known_ref_seq, num_frames,
    frame_seed, device_args, tx_subdev_spec, rx_subdev_spec,
    tx_antenna, rx_antenna, tx_channels, rx_channels,
    tx_gain, rx_gain, otw_format, start_delay=0.8,
):
    import uhd

    from USRP.usrp_utils import configure_usrp_timing

    # Open the hardware exactly once, before constructing both streamers.
    usrp = uhd.usrp.MultiUSRP(device_args)
    configure_usrp_timing(usrp)  # Shared internal sample clock and device time
    usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec(tx_subdev_spec), 0)
    usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec(rx_subdev_spec), 0)
    for ch in tx_channels:
        usrp.set_tx_antenna(tx_antenna, ch)
    for ch in rx_channels:
        usrp.set_rx_antenna(rx_antenna, ch)

    print(f"Using one X310: {device_args}; TX={tx_channels}, RX={rx_channels}", flush=True)
    frames = build_precomputed_frames(
        params, known_ref_seq, num_frames=num_frames, frame_seed=frame_seed
    )
    # Extra silent 5ms segment lets the postprocessor consume the final frame's
    # trailing samples without attempting to extend a finished TX burst.
    guard = np.zeros_like(frames[:1])
    frames = np.concatenate((frames, guard), axis=0)
    start_time = usrp.get_time_now().get_real_secs() + start_delay

    receiver = USRPReceiver(
        usrp=usrp, frames_queue=frames_queue,
        num_samples=int(params["frame_length"]),
        carrier_frequency=params["carrier_frequency"],
        sampling_rate=params["sampling_rate"], gain=rx_gain,
        channels=tuple(rx_channels), start_time=start_time,
        otw_format=otw_format, stop_event=stop_event,
        recv_timeout=0.5, max_consecutive_timeouts=10, uhd_module=uhd,
    )
    transmitter = USRPTransmitter(
        usrp=usrp, frames=frames,
        carrier_frequency=params["carrier_frequency"],
        sampling_rate=params["sampling_rate"], gain=tx_gain,
        channels=list(tx_channels), transmission_delay=start_delay,
        transmission_time=start_time, stop_event=stop_event,
        otw_format=otw_format, uhd_module=uhd,
    )
    errors = []
    def collect():
        try:
            metadata = uhd.types.RXMetadata()
            receiver._start_stream()
            try:
                # One extra RX frame is required for the last CSI tail.
                for frame_id in range(num_frames + 1):
                    if stop_event.is_set():
                        break
                    value = receiver.receive_frame(metadata)
                    if value is None:
                        break
                    samples, timestamp = value
                    if not receiver._enqueue_frame(frame_id, samples, timestamp):
                        break
            finally:
                receiver._stop_stream()
        except BaseException as exc:
            errors.append(exc)
            traceback.print_exc()
            stop_event.set()

    rx_thread = threading.Thread(target=collect, name="x310-rx", daemon=True)
    rx_thread.start()
    try:
        transmitter.run()
        rx_thread.join(timeout=max(5.0, num_frames * float(params["csi_sample_period_s"]) + 5))
        if rx_thread.is_alive():
            stop_event.set()
            raise TimeoutError("X310 receiver did not finish collecting the expected frames")
        if errors:
            raise RuntimeError("X310 RX failed") from errors[0]
    finally:
        if rx_thread.is_alive():
            stop_event.set()
            rx_thread.join(timeout=2)
