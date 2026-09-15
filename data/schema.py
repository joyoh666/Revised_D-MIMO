from dataclasses import dataclass
import numpy as np

@dataclass
class CSIFrame:
    frame_idx : int
    host_time : int
    usrp_time : float | None
    csi : np.ndarray
    producer_interval : float | None = None
    buffer_ready : bool | None = None

    #optional metadata
    position : np.ndarray | None = None
    velocity : np.ndarray | None = None
    snr_db : float | None = None

@dataclass
class CSIBlock:
    start_frame_idx : int
    end_frame_idx : int
    start_time : float
    end_time : float
    csi : np.ndarray