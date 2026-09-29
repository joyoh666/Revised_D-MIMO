from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CSIFrame:
    """One 5 ms CSI vector and only the metadata needed online."""

    frame_idx: int
    host_time: float
    usrp_time: float | None
    csi: np.ndarray


@dataclass(frozen=True)
class WindowMetadata:
    """Time/index range shared by a completed model-input window."""

    start_frame_idx: int
    end_frame_idx: int
    start_host_time: float
    end_host_time: float
    start_usrp_time: float | None
    end_usrp_time: float | None


@dataclass(frozen=True)
class PredictorModelInput:
    """One 25 ms predictor input.

    ``tokens`` has shape ``(num_tx_ant, 15)``.  Each row contains the
    gain/cos/sin representation of five consecutive CSI samples.
    """

    metadata: WindowMetadata
    tokens: np.ndarray


@dataclass(frozen=True)
class SchedulerModelInput:
    """One 100 ms scheduler segment before connection-state masking.

    ``ru_gains`` has shape ``(3,)`` for RU1, RU2, and RU3.  The scheduler
    process combines these segments with its current connection state and
    history to form the final 15-dimensional policy observation.
    """

    metadata: WindowMetadata
    ru_gains: np.ndarray


@dataclass
class CSIBlock:
    """Legacy raw-CSI block used by the small buffer examples/tests."""

    start_frame_idx: int
    end_frame_idx: int
    start_time: float
    end_time: float
    csi: np.ndarray
