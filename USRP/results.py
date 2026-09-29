from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .modulate import modulations
from .postprocessing import channel_to_delay_response, delay_residual_power


_EPSILON = 1e-15


class CaptureAccumulator:
    """Collect per-capture receiver outputs and build the saved result schema.

    Saved CSI-domain arrays use ``(CSI sample, RX, TX, value)`` ordering.  The
    capture axis is prepended by :meth:`finalize`.
    """

    def __init__(self, params: dict[str, Any]):
        self.params = params
        self._captures: dict[str, list[np.ndarray]] = {}
        self._num_receive_antennas: int | None = None
        self._example_iq_rcv: np.ndarray | None = None
        self._example_iq_rcv_single: np.ndarray | None = None
        self._example_frame_rcv: np.ndarray | None = None

    def _append(
        self,
        key: str,
        value: Any,
        *,
        dtype: np.dtype[Any] | type[Any] | None = None,
    ) -> None:
        array = np.asarray(value, dtype=dtype)
        self._captures.setdefault(key, []).append(array.copy())

    def _channel_grid(self, value: Any, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.complex64)
        expected_shape = (
            int(self.params["num_rx_ant"]),
            int(self.params["num_tx_ant"]),
            int(self.params["num_subframe_per_frame"]),
            int(self.params["num_slot_per_subframe"]),
            int(self.params["N"]),
        )
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape "
                "(rx, tx, subframe, slot, subcarrier); "
                f"got {array.shape}, expected {expected_shape}"
            )
        self._check_antenna_count(array.shape[0], name)
        return array

    def _csi_channel(self, value: Any, name: str) -> np.ndarray:
        """Validate ``(CSI, TX, RX, N)`` and return ``(CSI, RX, TX, N)``."""
        array = np.asarray(value)
        expected_shape = (
            int(self.params["num_csi_samples_per_frame"]),
            int(self.params["num_tx_ant"]),
            int(self.params["num_rx_ant"]),
            int(self.params["N"]),
        )
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape (csi, tx, rx, subcarrier); "
                f"got {array.shape}, expected {expected_shape}"
            )
        self._check_antenna_count(array.shape[2], name)
        return np.transpose(array, (0, 2, 1, 3))

    def _virtual_pilot_channels(self, value: Any, name: str) -> np.ndarray:
        """Return pilots as ``(CSI, RX, TX, pilot, subcarrier)``."""
        array = np.asarray(value, dtype=np.complex64)
        expected_shape = (
            int(self.params["num_csi_samples_per_frame"]),
            int(self.params["num_tx_ant"]),
            int(self.params["num_virtual_pilots"]),
            int(self.params["num_rx_ant"]),
            int(self.params["N"]),
        )
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape (csi, tx, pilot, rx, subcarrier); "
                f"got {array.shape}, expected {expected_shape}"
            )
        self._check_antenna_count(array.shape[3], name)
        return np.transpose(array, (0, 3, 1, 2, 4))

    def _pilot_phases(self, value: Any, name: str) -> np.ndarray:
        """Return phases as ``(CSI, RX, TX, pilot)``."""
        array = np.asarray(value, dtype=np.float32)
        expected_shape = (
            int(self.params["num_csi_samples_per_frame"]),
            int(self.params["num_tx_ant"]),
            int(self.params["num_virtual_pilots"]),
            int(self.params["num_rx_ant"]),
        )
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape (csi, tx, pilot, rx); "
                f"got {array.shape}, expected {expected_shape}"
            )
        self._check_antenna_count(array.shape[3], name)
        return np.transpose(array, (0, 3, 1, 2))

    def _csi_metric(self, value: Any, name: str) -> np.ndarray:
        """Return a scalar CSI metric as ``(CSI, RX, TX)``."""
        array = np.asarray(value, dtype=np.float32)
        expected_shape = (
            int(self.params["num_csi_samples_per_frame"]),
            int(self.params["num_tx_ant"]),
            int(self.params["num_rx_ant"]),
        )
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape (csi, tx, rx); "
                f"got {array.shape}, expected {expected_shape}"
            )
        self._check_antenna_count(array.shape[2], name)
        return np.transpose(array, (0, 2, 1))

    def _scalar_csi(self, value: Any, name: str) -> np.ndarray:
        """Return scalar FDM-compatible CSI as ``(CSI, RX, TX)``."""
        array = np.asarray(value, dtype=np.complex64)
        expected_shape = (
            int(self.params["num_csi_samples_per_frame"]),
            int(self.params["num_tx_ant"]),
            int(self.params["num_rx_ant"]),
        )
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape (csi, tx, rx); "
                f"got {array.shape}, expected {expected_shape}"
            )
        self._check_antenna_count(array.shape[2], name)
        return np.transpose(array, (0, 2, 1))

    def _scalar_pilots(self, value: Any, name: str) -> np.ndarray:
        """Return scalar repeated pilots as ``(CSI, RX, TX, pilot)``."""
        array = np.asarray(value)
        expected_shape = (
            int(self.params["num_csi_samples_per_frame"]),
            int(self.params["num_tx_ant"]),
            int(self.params["num_virtual_pilots"]),
            int(self.params["num_rx_ant"]),
        )
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape (csi, tx, pilot, rx); "
                f"got {array.shape}, expected {expected_shape}"
            )
        self._check_antenna_count(array.shape[3], name)
        return np.transpose(array, (0, 3, 1, 2))

    def _pilot_noise(self, value: Any, name: str) -> np.ndarray:
        """Return repeated-pilot noise as ``(CSI, RX, pilot)``."""
        array = np.asarray(value, dtype=np.float32)
        expected_shape = (
            int(self.params["num_csi_samples_per_frame"]),
            int(self.params["num_virtual_pilots"]),
            int(self.params["num_rx_ant"]),
        )
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape (csi, pilot, rx); "
                f"got {array.shape}, expected {expected_shape}"
            )
        self._check_antenna_count(array.shape[2], name)
        return np.transpose(array, (0, 2, 1))

    def _antenna_samples(self, value: Any, name: str) -> np.ndarray:
        array = np.asarray(value)
        if array.ndim == 1:
            array = array[np.newaxis, ...]
        if array.ndim != 2:
            raise ValueError(
                f"{name} must have shape (ant, sample); got {array.shape}"
            )
        self._check_antenna_count(array.shape[0], name)
        return array

    def _check_antenna_count(self, count: int, name: str) -> None:
        if self._num_receive_antennas is None:
            self._num_receive_antennas = int(count)
        elif count != self._num_receive_antennas:
            raise ValueError(
                f"{name} has {count} receive antennas, but previous captures "
                f"have {self._num_receive_antennas}"
            )

    def add(
        self,
        received: dict[str, Any],
        *,
        csi_timestamps_usrp_s: Any | None = None,
        pilot_timestamps_usrp_s: Any | None = None,
        rx_block_id: int | None = None,
        dropped_blocks: int | None = None,
    ) -> None:
        """Add one dictionary returned by ``USRP.postprocessing.postprocess``."""

        h_fd = self._channel_grid(received["h_fd"], "h_fd")
        h_single_fd = self._channel_grid(
            received["h_single_fd"],
            "h_single_fd",
        )
        if h_single_fd.shape != h_fd.shape:
            raise ValueError(
                "h_single_fd and h_fd must have the same shape; "
                f"got {h_single_fd.shape} and {h_fd.shape}"
            )

        h_raw = self._csi_channel(
            received["h_fd_virtual_avg_raw"],
            "h_fd_virtual_avg_raw",
        ).astype(np.complex64, copy=False)
        h_aligned = self._csi_channel(
            received["h_fd_virtual_avg_aligned"],
            "h_fd_virtual_avg_aligned",
        ).astype(np.complex64, copy=False)
        variance_raw = self._csi_channel(
            received["virtual_pilot_variance_raw"],
            "virtual_pilot_variance_raw",
        ).astype(np.float32, copy=False)
        variance_aligned = self._csi_channel(
            received["virtual_pilot_variance_aligned"],
            "virtual_pilot_variance_aligned",
        ).astype(np.float32, copy=False)
        h_virtual = self._virtual_pilot_channels(
            received["h_virtual_pilots_fd"],
            "h_virtual_pilots_fd",
        )
        h_virtual_aligned = self._virtual_pilot_channels(
            received["h_virtual_pilots_fd_aligned"],
            "h_virtual_pilots_fd_aligned",
        )
        if h_raw.shape != h_aligned.shape or h_raw.shape != variance_raw.shape:
            raise ValueError("CSI channel and variance arrays must have equal shapes")
        if variance_aligned.shape != variance_raw.shape:
            raise ValueError("Raw and aligned variance arrays must have equal shapes")

        csi_raw = self._scalar_csi(
            received["csi_scalar_raw"],
            "csi_scalar_raw",
        )
        csi_aligned = self._scalar_csi(
            received["csi_scalar_aligned"],
            "csi_scalar_aligned",
        )
        csi_repetitions = self._scalar_pilots(
            received["csi_repetitions_scalar"],
            "csi_repetitions_scalar",
        ).astype(np.complex64, copy=False)
        csi_repetitions_aligned = self._scalar_pilots(
            received["csi_repetitions_scalar_aligned"],
            "csi_repetitions_scalar_aligned",
        ).astype(np.complex64, copy=False)
        pilot_received = self._scalar_pilots(
            received["pilot_received_scalar"],
            "pilot_received_scalar",
        ).astype(np.complex64, copy=False)
        pilot_snr_repetitions = self._scalar_pilots(
            received["pilot_snr_repetitions_db"],
            "pilot_snr_repetitions_db",
        ).astype(np.float32, copy=False)
        pilot_noise_power = self._pilot_noise(
            received["pilot_noise_power"],
            "pilot_noise_power",
        )

        # Match the configured training-data convention while retaining both
        # raw and aligned repeated-pilot estimates in the dataset.
        h_mean = (
            h_aligned
            if self.params.get("phase_align_virtual_pilots", False)
            else h_raw
        )
        h_single_mean = h_virtual[:, :, :, 0, :]

        h_delay, h_full_spectrum = channel_to_delay_response(
            h_mean,
            int(self.params["N"]),
            int(self.params["FFT_SIZE"]),
        )
        h_delay_single, h_full_spectrum_single = channel_to_delay_response(
            h_single_mean,
            int(self.params["N"]),
            int(self.params["FFT_SIZE"]),
        )

        guard_taps = max(1, int(round(int(self.params["FFT_SIZE"]) / 200)))
        _, residual_single, _ = delay_residual_power(
            h_delay_single,
            guard_taps=guard_taps,
        )
        _, residual_averaged, _ = delay_residual_power(
            h_delay,
            guard_taps=guard_taps,
        )
        residual_gain_db = 10.0 * np.log10(
            (np.asarray(residual_single) + _EPSILON)
            / (np.asarray(residual_averaged) + _EPSILON)
        )

        self._append("channel_estimates_fd", h_fd, dtype=np.complex64)
        self._append(
            "channel_estimates_fd_single_pilot",
            h_single_fd,
            dtype=np.complex64,
        )
        self._append(
            "channel_estimates_fd_raw_virtual",
            h_raw,
            dtype=np.complex64,
        )
        self._append(
            "channel_estimates_fd_aligned_virtual",
            h_aligned,
            dtype=np.complex64,
        )
        self._append(
            "channel_estimates_fd_virtual_pilots",
            h_virtual,
            dtype=np.complex64,
        )
        self._append(
            "channel_estimates_fd_virtual_pilots_aligned",
            h_virtual_aligned,
            dtype=np.complex64,
        )
        self._append("channel_mean_fd", h_mean, dtype=np.complex64)
        self._append(
            "csi",
            csi_aligned
            if self.params.get("phase_align_virtual_pilots", False)
            else csi_raw,
            dtype=np.complex64,
        )
        self._append(
            "csi_repetitions",
            csi_repetitions,
            dtype=np.complex64,
        )
        self._append(
            "csi_repetitions_aligned",
            csi_repetitions_aligned,
            dtype=np.complex64,
        )
        self._append(
            "pilot_received",
            pilot_received,
            dtype=np.complex64,
        )
        self._append(
            "pilot_snr_db_per_repetition",
            pilot_snr_repetitions,
            dtype=np.float32,
        )
        self._append(
            "pilot_noise_power",
            pilot_noise_power,
            dtype=np.float32,
        )
        self._append(
            "channel_impulse_response_td",
            h_delay,
            dtype=np.complex64,
        )
        self._append(
            "channel_impulse_response_td_single_pilot",
            h_delay_single,
            dtype=np.complex64,
        )
        self._append(
            "channel_full_spectrum_fd",
            h_full_spectrum,
            dtype=np.complex64,
        )
        self._append(
            "channel_full_spectrum_fd_single_pilot",
            h_full_spectrum_single,
            dtype=np.complex64,
        )
        self._append(
            "delay_profile_residual_gain_db",
            residual_gain_db,
            dtype=np.float32,
        )
        self._append(
            "virtual_pilot_variance_raw",
            variance_raw,
            dtype=np.float32,
        )
        self._append(
            "virtual_pilot_variance_aligned",
            variance_aligned,
            dtype=np.float32,
        )
        self._append(
            "virtual_pilot_common_phases_rad",
            self._pilot_phases(
                received["virtual_pilot_common_phases_rad"],
                "virtual_pilot_common_phases_rad",
            ),
            dtype=np.float32,
        )
        for key in (
            "single_pilot_snr_db",
            "virtual_pilot_snr_db",
            "virtual_pilot_snr_gain_db",
            "virtual_pilot_correction_phase_rad",
        ):
            self._append(
                key,
                self._csi_metric(received[key], key),
                dtype=np.float32,
            )
        self._append("sync_indices", received["sync_idx"], dtype=np.int32)
        self._append("ffo_estimates_hz", received["ffo_hz"], dtype=np.float32)
        self._append(
            "rfo_estimates_hz",
            received["rfo_hz_per_slot"],
            dtype=np.float32,
        )
        self._append(
            "timing_correlation_peaks",
            received["timing_correlation_peak"],
            dtype=np.float32,
        )

        num_csi_samples = int(self.params["num_csi_samples_per_frame"])
        num_pilots = int(self.params["num_virtual_pilots"])
        if csi_timestamps_usrp_s is None:
            csi_timestamps_usrp_s = np.full(
                num_csi_samples,
                np.nan,
                dtype=np.float64,
            )
        if pilot_timestamps_usrp_s is None:
            pilot_timestamps_usrp_s = np.full(
                (num_csi_samples, num_pilots),
                np.nan,
                dtype=np.float64,
            )
        csi_timestamps_usrp_s = np.asarray(
            csi_timestamps_usrp_s,
            dtype=np.float64,
        )
        pilot_timestamps_usrp_s = np.asarray(
            pilot_timestamps_usrp_s,
            dtype=np.float64,
        )
        if csi_timestamps_usrp_s.shape != (num_csi_samples,):
            raise ValueError(
                "csi_timestamps_usrp_s must have shape "
                f"({num_csi_samples},)"
            )
        if pilot_timestamps_usrp_s.shape != (num_csi_samples, num_pilots):
            raise ValueError(
                "pilot_timestamps_usrp_s must have shape "
                f"({num_csi_samples}, {num_pilots})"
            )
        self._append(
            "csi_timestamps_usrp_s",
            csi_timestamps_usrp_s,
            dtype=np.float64,
        )
        self._append(
            "pilot_timestamps_usrp_s",
            pilot_timestamps_usrp_s,
            dtype=np.float64,
        )
        self._append(
            "rx_block_ids",
            -1 if rx_block_id is None else rx_block_id,
            dtype=np.int64,
        )
        self._append(
            "rx_dropped_blocks",
            -1 if dropped_blocks is None else dropped_blocks,
            dtype=np.int64,
        )

        if self._example_iq_rcv is None:
            self._example_iq_rcv = self._antenna_samples(
                received["iq_rcv"],
                "iq_rcv",
            ).astype(np.complex64, copy=True)
            self._example_iq_rcv_single = self._antenna_samples(
                received["iq_rcv_single"],
                "iq_rcv_single",
            ).astype(np.complex64, copy=True)
            self._example_frame_rcv = self._antenna_samples(
                received["frame_rcv"],
                "frame_rcv",
            ).astype(np.complex64, copy=True)

    def finalize(self) -> dict[str, Any]:
        """Stack captures using the version-3 result dictionary schema."""

        if not self._captures:
            raise RuntimeError("Cannot finalize results before adding a capture")
        if (
            self._example_iq_rcv is None
            or self._example_iq_rcv_single is None
            or self._example_frame_rcv is None
        ):
            raise RuntimeError("The first capture did not provide example arrays")

        results: dict[str, Any] = {
            key: np.stack(values, axis=0)
            for key, values in self._captures.items()
        }
        results.update(
            {
                "example_iq_rcv": self._example_iq_rcv.copy(),
                "example_iq_rcv_single": self._example_iq_rcv_single.copy(),
                "example_frame_rcv": self._example_frame_rcv.copy(),
                "num_receive_antennas": int(self._num_receive_antennas or 0),
                "num_transmit_antennas": int(self.params["num_tx_ant"]),
                "num_csi_samples_per_frame": int(
                    self.params["num_csi_samples_per_frame"]
                ),
            }
        )
        return results


def _timestamp_string(value: str | datetime | None) -> str:
    if value is None:
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    if isinstance(value, datetime):
        return value.strftime("%Y%m%d_%H%M%S")
    return str(value)


def _json_compatible(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    return value


def _capture_channel_grid(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 6:
        raise ValueError(
            f"{name} must have shape "
            "(capture, rx, tx, subframe, slot, subcarrier); "
            f"got {array.shape}"
        )
    return array


def _capture_csi_rx_tx_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 5:
        raise ValueError(
            f"{name} must have shape (capture, csi, rx, tx, value); "
            f"got {array.shape}"
        )
    return array


def _example_antenna_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (ant, value); got {array.shape}")
    return array


def _original_compatible_arrays(
    params: dict[str, Any],
    known_ref_seq: np.ndarray,
    results: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Flatten capture/CSI axes into the original ``(time, ...)`` schema."""
    csi_by_rx = np.asarray(results["csi"], dtype=np.complex64)
    repetitions_by_rx = np.asarray(
        results["csi_repetitions"],
        dtype=np.complex64,
    )
    repetitions_aligned_by_rx = np.asarray(
        results["csi_repetitions_aligned"],
        dtype=np.complex64,
    )
    pilot_received_by_rx = np.asarray(
        results["pilot_received"],
        dtype=np.complex64,
    )
    pilot_snr_by_rx = np.asarray(
        results["pilot_snr_db_per_repetition"],
        dtype=np.float32,
    )
    noise_by_rx = np.asarray(
        results["pilot_noise_power"],
        dtype=np.float32,
    )

    if csi_by_rx.ndim != 4:
        raise ValueError("csi must have shape (capture, csi, rx, tx)")
    captures, csi_samples, num_rx, num_tx = csi_by_rx.shape
    num_pilots = int(params["num_virtual_pilots"])
    expected_pilot_shape = (
        captures,
        csi_samples,
        num_rx,
        num_tx,
        num_pilots,
    )
    for name, array in (
        ("csi_repetitions", repetitions_by_rx),
        ("csi_repetitions_aligned", repetitions_aligned_by_rx),
        ("pilot_received", pilot_received_by_rx),
        ("pilot_snr_db_per_repetition", pilot_snr_by_rx),
    ):
        if array.shape != expected_pilot_shape:
            raise ValueError(
                f"{name} has shape {array.shape}, expected "
                f"{expected_pilot_shape}"
            )
    if noise_by_rx.shape != (
        captures,
        csi_samples,
        num_rx,
        num_pilots,
    ):
        raise ValueError("pilot_noise_power has an invalid shape")

    # The paper-compatible dataset has one UE/RX and omits that singleton
    # axis. Preserve every RX in ``*_by_rx`` extensions for future use.
    flattened_csi_by_rx = csi_by_rx.reshape(-1, num_rx, num_tx)
    flattened_repetitions_by_rx = np.transpose(
        repetitions_by_rx,
        (0, 1, 2, 4, 3),
    ).reshape(-1, num_rx, num_pilots, num_tx)
    flattened_repetitions_aligned_by_rx = np.transpose(
        repetitions_aligned_by_rx,
        (0, 1, 2, 4, 3),
    ).reshape(-1, num_rx, num_pilots, num_tx)
    flattened_pilot_received_by_rx = np.transpose(
        pilot_received_by_rx,
        (0, 1, 2, 4, 3),
    ).reshape(-1, num_rx, num_pilots, num_tx)
    flattened_pilot_snr_by_rx = np.transpose(
        pilot_snr_by_rx,
        (0, 1, 2, 4, 3),
    ).reshape(-1, num_rx, num_pilots, num_tx)
    flattened_noise_by_rx = noise_by_rx.reshape(-1, num_rx, num_pilots)
    num_snapshots = flattened_csi_by_rx.shape[0]

    centered_bins = np.asarray(
        params["fdm_pilot_centered_bins"],
        dtype=np.int16,
    )
    active_indices = np.asarray(
        params["fdm_pilot_active_indices"],
        dtype=np.int16,
    )
    transmitted_pilots = np.broadcast_to(
        np.asarray(known_ref_seq, dtype=np.complex64)[active_indices][None, :],
        (num_pilots, num_tx),
    ).copy()

    csi_timestamps = np.asarray(
        results["csi_timestamps_usrp_s"],
        dtype=np.float64,
    ).reshape(-1)
    pilot_timestamps = np.asarray(
        results["pilot_timestamps_usrp_s"],
        dtype=np.float64,
    ).reshape(-1, num_pilots)
    block_ids = np.asarray(results["rx_block_ids"], dtype=np.int64).reshape(-1)
    dropped_blocks = np.asarray(
        results["rx_dropped_blocks"],
        dtype=np.int64,
    ).reshape(-1)
    if block_ids.size == captures and np.all(block_ids >= 0):
        period_indices = (
            (block_ids[:, None] - block_ids[0]) * csi_samples
            + np.arange(csi_samples, dtype=np.int64)[None, :]
        ).reshape(-1)
    else:
        period_indices = np.arange(num_snapshots, dtype=np.int64)

    sync_indices = np.repeat(
        np.asarray(results["sync_indices"], dtype=np.int64),
        csi_samples,
    )
    correlation = np.repeat(
        np.asarray(results["timing_correlation_peaks"], dtype=np.float32),
        csi_samples,
        axis=0,
    )
    phases = np.asarray(
        results["virtual_pilot_common_phases_rad"],
        dtype=np.float32,
    )
    phases = np.transpose(phases, (0, 1, 2, 4, 3)).reshape(
        -1,
        num_rx,
        num_pilots,
        num_tx,
    )

    return {
        "dataset_row_indices": np.arange(num_snapshots, dtype=np.int64),
        "csi_period_indices": period_indices,
        "csi": flattened_csi_by_rx[:, 0],
        "csi_by_rx": flattened_csi_by_rx,
        "csi_repetitions": flattened_repetitions_by_rx[:, 0],
        "csi_repetitions_by_rx": flattened_repetitions_by_rx,
        "csi_repetitions_aligned": (
            flattened_repetitions_aligned_by_rx[:, 0]
        ),
        "csi_repetitions_aligned_by_rx": (
            flattened_repetitions_aligned_by_rx
        ),
        "pilot_received": flattened_pilot_received_by_rx[:, 0],
        "pilot_received_by_rx": flattened_pilot_received_by_rx,
        "transmitted_pilots": transmitted_pilots,
        "csi_timestamps_usrp_s": csi_timestamps,
        "pilot_timestamps_usrp_s": pilot_timestamps,
        "frame_start_sample_indices_in_processing_buffer": sync_indices,
        "frame_sequence_numbers_in_processing_buffer": period_indices,
        "timing_error_samples": sync_indices,
        "zc_correlation_scores": correlation[:, 0],
        "pilot_noise_power": flattened_noise_by_rx[:, 0],
        "pilot_snr_db": flattened_pilot_snr_by_rx[:, 0],
        "repetition_phase_corrections_rad": phases[:, 0],
        "pilot_centered_bins": centered_bins,
        "pilot_active_subcarrier_indices": active_indices,
        "rx_block_ids": np.repeat(block_ids, csi_samples),
        "rx_dropped_blocks": np.repeat(dropped_blocks, csi_samples),
    }


def save_results(
    output_dir: str | Path,
    params: dict[str, Any],
    known_ref_seq: np.ndarray,
    results: dict[str, Any],
    *,
    reference_sequence_seed: int,
    timestamp: str | datetime | None = None,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    """Save the capture dataset and its human-readable metadata."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    timestamp_text = _timestamp_string(timestamp)
    bandwidth_tag = str(params["bandwidth_mhz"]).replace(".", "p")
    prefix = f"BW_{bandwidth_tag}MHz_{timestamp_text}"
    npz_path = destination / f"channel_dataset_{prefix}.npz"
    json_path = destination / f"channel_dataset_{prefix}_metadata.json"
    figure_path = destination / f"channel_plots_{prefix}.png"

    channel_estimates = _capture_channel_grid(
        results["channel_estimates_fd"],
        "channel_estimates_fd",
    )
    channel_mean = _capture_csi_rx_tx_array(
        results["channel_mean_fd"],
        "channel_mean_fd",
    )
    delay_response = _capture_csi_rx_tx_array(
        results["channel_impulse_response_td"],
        "channel_impulse_response_td",
    )
    delay_response_single = _capture_csi_rx_tx_array(
        results["channel_impulse_response_td_single_pilot"],
        "channel_impulse_response_td_single_pilot",
    )
    example_iq_single = _example_antenna_array(
        results["example_iq_rcv_single"],
        "example_iq_rcv_single",
    )
    num_receive_antennas = int(
        results.get("num_receive_antennas", channel_mean.shape[2])
    )
    num_transmit_antennas = int(
        results.get("num_transmit_antennas", channel_mean.shape[3])
    )
    num_csi_samples = int(
        results.get("num_csi_samples_per_frame", channel_mean.shape[1])
    )
    num_virtual_pilots = int(params["num_virtual_pilots"])
    expected_grid_shape = (
        channel_estimates.shape[0],
        num_receive_antennas,
        num_transmit_antennas,
        int(params["num_subframe_per_frame"]),
        int(params["num_slot_per_subframe"]),
        int(params["N"]),
    )
    expected_csi_shape = (
        channel_estimates.shape[0],
        num_csi_samples,
        num_receive_antennas,
        num_transmit_antennas,
        int(params["N"]),
    )
    if channel_estimates.shape != expected_grid_shape:
        raise ValueError(
            "channel_estimates_fd shape does not match the saved dimensions; "
            f"got {channel_estimates.shape}, expected {expected_grid_shape}"
        )
    if channel_mean.shape != expected_csi_shape:
        raise ValueError(
            "channel_mean_fd shape does not match the saved dimensions; "
            f"got {channel_mean.shape}, expected {expected_csi_shape}"
        )

    compatible = _original_compatible_arrays(
        params,
        known_ref_seq,
        results,
    )

    saved_result_keys = (
        "channel_estimates_fd",
        "channel_estimates_fd_single_pilot",
        "channel_estimates_fd_raw_virtual",
        "channel_estimates_fd_aligned_virtual",
        "channel_estimates_fd_virtual_pilots",
        "channel_estimates_fd_virtual_pilots_aligned",
        "channel_mean_fd",
        "csi",
        "csi_repetitions",
        "csi_repetitions_aligned",
        "pilot_received",
        "pilot_snr_db_per_repetition",
        "pilot_noise_power",
        "csi_timestamps_usrp_s",
        "pilot_timestamps_usrp_s",
        "rx_block_ids",
        "rx_dropped_blocks",
        "virtual_pilot_variance_raw",
        "virtual_pilot_variance_aligned",
        "virtual_pilot_common_phases_rad",
        "channel_impulse_response_td",
        "channel_impulse_response_td_single_pilot",
        "channel_full_spectrum_fd",
        "channel_full_spectrum_fd_single_pilot",
        "sync_indices",
        "ffo_estimates_hz",
        "rfo_estimates_hz",
        "timing_correlation_peaks",
        "single_pilot_snr_db",
        "virtual_pilot_snr_db",
        "virtual_pilot_snr_gain_db",
        "delay_profile_residual_gain_db",
        "virtual_pilot_correction_phase_rad",
        "example_iq_rcv",
        "example_iq_rcv_single",
        "example_frame_rcv",
    )

    metadata = {
        "result_schema_version": 4,
        "array_axis_order": {
            "csi": ["time", "tx_antenna"],
            "csi_by_rx": ["time", "rx_antenna", "tx_antenna"],
            "csi_repetitions": [
                "time",
                "virtual_pilot",
                "tx_antenna",
            ],
            "csi_repetitions_aligned": [
                "time",
                "virtual_pilot",
                "tx_antenna",
            ],
            "pilot_received": [
                "time",
                "virtual_pilot",
                "tx_antenna",
            ],
            "pilot_snr_db": [
                "time",
                "virtual_pilot",
                "tx_antenna",
            ],
            "pilot_noise_power": ["time", "virtual_pilot"],
            "csi_timestamps_usrp_s": ["time"],
            "pilot_timestamps_usrp_s": ["time", "virtual_pilot"],
            "channel_estimates_fd": [
                "capture",
                "rx_antenna",
                "tx_antenna",
                "subframe",
                "slot",
                "subcarrier",
            ],
            "channel_estimates_fd_single_pilot": [
                "capture",
                "rx_antenna",
                "tx_antenna",
                "subframe",
                "slot",
                "subcarrier",
            ],
            "channel_estimates_fd_raw_virtual": [
                "capture",
                "csi_sample",
                "rx_antenna",
                "tx_antenna",
                "subcarrier",
            ],
            "channel_estimates_fd_aligned_virtual": [
                "capture",
                "csi_sample",
                "rx_antenna",
                "tx_antenna",
                "subcarrier",
            ],
            "channel_estimates_fd_virtual_pilots": [
                "capture",
                "csi_sample",
                "rx_antenna",
                "tx_antenna",
                "virtual_pilot",
                "subcarrier",
            ],
            "channel_estimates_fd_virtual_pilots_aligned": [
                "capture",
                "csi_sample",
                "rx_antenna",
                "tx_antenna",
                "virtual_pilot",
                "subcarrier",
            ],
            "channel_mean_fd": [
                "capture",
                "csi_sample",
                "rx_antenna",
                "tx_antenna",
                "subcarrier",
            ],
            "channel_impulse_response_td": [
                "capture",
                "csi_sample",
                "rx_antenna",
                "tx_antenna",
                "delay_sample",
            ],
            "channel_impulse_response_td_single_pilot": [
                "capture",
                "csi_sample",
                "rx_antenna",
                "tx_antenna",
                "delay_sample",
            ],
            "channel_full_spectrum_fd": [
                "capture",
                "csi_sample",
                "rx_antenna",
                "tx_antenna",
                "fft_bin",
            ],
            "channel_full_spectrum_fd_single_pilot": [
                "capture",
                "csi_sample",
                "rx_antenna",
                "tx_antenna",
                "fft_bin",
            ],
            "virtual_pilot_variance_raw": [
                "capture",
                "csi_sample",
                "rx_antenna",
                "tx_antenna",
                "subcarrier",
            ],
            "virtual_pilot_variance_aligned": [
                "capture",
                "csi_sample",
                "rx_antenna",
                "tx_antenna",
                "subcarrier",
            ],
            "virtual_pilot_common_phases_rad": [
                "capture",
                "csi_sample",
                "rx_antenna",
                "tx_antenna",
                "virtual_pilot",
            ],
            "virtual_pilot_snr_db": [
                "capture",
                "csi_sample",
                "rx_antenna",
                "tx_antenna",
            ],
            "single_pilot_snr_db": [
                "capture",
                "csi_sample",
                "rx_antenna",
                "tx_antenna",
            ],
            "virtual_pilot_snr_gain_db": [
                "capture",
                "csi_sample",
                "rx_antenna",
                "tx_antenna",
            ],
            "delay_profile_residual_gain_db": [
                "capture",
                "csi_sample",
                "rx_antenna",
                "tx_antenna",
            ],
            "virtual_pilot_correction_phase_rad": [
                "capture",
                "csi_sample",
                "rx_antenna",
                "tx_antenna",
            ],
            "sync_indices": ["capture"],
            "ffo_estimates_hz": ["capture", "rx_antenna"],
            "rfo_estimates_hz": ["capture", "rx_antenna", "slot"],
            "timing_correlation_peaks": ["capture", "rx_antenna"],
            "example_iq_rcv": ["rx_antenna", "data_symbol"],
            "example_iq_rcv_single": ["rx_antenna", "data_symbol"],
            "example_frame_rcv": ["rx_antenna", "time_sample"],
            "known_reference_sequence": ["subcarrier"],
        },
        "timestamp": timestamp_text,
        "bandwidth_mhz": params["bandwidth_mhz"],
        "subcarrier_spacing_hz": params["delta_f"],
        "num_active_subcarriers": params["N"],
        "fft_size": params["FFT_SIZE"],
        "sampling_rate_hz": params["sampling_rate"],
        "carrier_frequency_hz": params["carrier_frequency"],
        "tx_gain_db": params["Tx_gain"],
        "rx_gain_db": params["Rx_gain"],
        "num_symbols_per_slot": params["num_symbols_per_slot"],
        "num_slots_per_subframe": params["num_slot_per_subframe"],
        "num_subframes_per_frame": params["num_subframe_per_frame"],
        "num_symbols_per_frame": params["num_symbols_frame"],
        "normal_cp_length_samples": params["normal_CP_length"],
        "first_cp_length_samples": params["first_CP_length"],
        "slot_length_samples": params["slot_length"],
        "frame_length_samples": params["frame_length"],
        "num_channel_captures": int(channel_estimates.shape[0]),
        "num_receive_antennas": num_receive_antennas,
        "num_transmit_antennas": num_transmit_antennas,
        "num_csi_samples_per_frame": num_csi_samples,
        "pilot_multiplexing": "fdm",
        "primary_csi_array": "csi",
        "primary_csi_interpretation": (
            "one complex estimate per TX at its assigned FDM pilot bin"
        ),
        "fullband_csi_interpretation": (
            "frequency-flat broadcast of the scalar FDM estimate"
        ),
        "csi_sample_period_subframes": int(
            params["csi_sample_period_subframes"]
        ),
        "csi_sample_period_s": float(params["csi_sample_period_s"]),
        "csi_sample_period_ms": float(params["csi_sample_period_s"] * 1e3),
        "csi_sample_start_subframes": [
            sample_idx * int(params["csi_sample_period_subframes"])
            for sample_idx in range(num_csi_samples)
        ],
        "saved_channel_tensor_shape": list(channel_estimates.shape),
        "saved_csi_channel_shape": list(channel_mean.shape),
        "saved_original_compatible_csi_shape": list(
            compatible["csi"].shape
        ),
        "saved_mean_channel_shape": list(channel_mean.shape),
        "saved_delay_response_shape": list(delay_response.shape),
        "saved_single_pilot_delay_response_shape": list(
            delay_response_single.shape
        ),
        "saved_example_single_pilot_iq_shape": list(example_iq_single.shape),
        "saved_array_shapes": {
            key: list(np.asarray(results[key]).shape)
            for key in saved_result_keys
        }
        | {
            key: list(np.asarray(value).shape)
            for key, value in compatible.items()
        }
        | {"known_reference_sequence": list(np.asarray(known_ref_seq).shape)},
        "samples_per_channel_vector": params["N"],
        "samples_per_delay_response": params["FFT_SIZE"],
        "reference_sequence_type": "seeded_unit_power_qpsk",
        "reference_sequence_seed": int(reference_sequence_seed),
        "virtual_pilots_per_tx_per_csi_sample": num_virtual_pilots,
        "fdm_pilot_centered_bins": params["fdm_pilot_centered_bins"],
        "fdm_pilot_active_indices": params["fdm_pilot_active_indices"],
        "csi_timestamps_available": bool(
            np.all(np.isfinite(compatible["csi_timestamps_usrp_s"]))
        ),
        "virtual_pilot_count": num_virtual_pilots,
        "virtual_pilot_positions": params["virtual_pilot_positions"],
        "phase_align_virtual_pilots": bool(
            params.get("phase_align_virtual_pilots", True)
        ),
        "expected_channel_estimation_snr_gain_db": float(
            10.0 * np.log10(max(num_virtual_pilots, 1))
        ),
        "achieved_channel_estimation_snr_gain_db_mean": float(
            np.mean(results["virtual_pilot_snr_gain_db"])
        ),
        "mean_single_to_averaged_residual_delay_gain_db": float(
            np.mean(results["delay_profile_residual_gain_db"])
        ),
        "modulation_order": params["modulation_order"],
        "digital_power_scale": params["POWER"],
    }
    metadata = _json_compatible(metadata)
    metadata_json = json.dumps(metadata)

    np.savez_compressed(
        npz_path,
        channel_estimates_fd=results["channel_estimates_fd"],
        channel_estimates_fd_single_pilot=results[
            "channel_estimates_fd_single_pilot"
        ],
        channel_estimates_fd_raw_virtual=results[
            "channel_estimates_fd_raw_virtual"
        ],
        channel_estimates_fd_aligned_virtual=results[
            "channel_estimates_fd_aligned_virtual"
        ],
        channel_estimates_fd_virtual_pilots=results[
            "channel_estimates_fd_virtual_pilots"
        ],
        channel_estimates_fd_virtual_pilots_aligned=results[
            "channel_estimates_fd_virtual_pilots_aligned"
        ],
        channel_mean_fd=results["channel_mean_fd"],
        virtual_pilot_variance_raw=results["virtual_pilot_variance_raw"],
        virtual_pilot_variance_aligned=results[
            "virtual_pilot_variance_aligned"
        ],
        virtual_pilot_common_phases_rad=results[
            "virtual_pilot_common_phases_rad"
        ],
        channel_impulse_response_td=results["channel_impulse_response_td"],
        channel_impulse_response_td_single_pilot=results[
            "channel_impulse_response_td_single_pilot"
        ],
        channel_full_spectrum_fd_single_pilot=results[
            "channel_full_spectrum_fd_single_pilot"
        ],
        channel_full_spectrum_fd=results["channel_full_spectrum_fd"],
        known_reference_sequence=np.asarray(known_ref_seq, dtype=np.complex64),
        num_receive_antennas=np.int32(num_receive_antennas),
        num_transmit_antennas=np.int32(num_transmit_antennas),
        num_csi_samples_per_frame=np.int32(num_csi_samples),
        csi_sample_period_s=np.float64(params["csi_sample_period_s"]),
        sync_indices=results["sync_indices"],
        ffo_estimates_hz=results["ffo_estimates_hz"],
        rfo_estimates_hz=results["rfo_estimates_hz"],
        timing_correlation_peaks=results["timing_correlation_peaks"],
        single_pilot_snr_db=results["single_pilot_snr_db"],
        virtual_pilot_snr_db=results["virtual_pilot_snr_db"],
        virtual_pilot_snr_gain_db=results["virtual_pilot_snr_gain_db"],
        delay_profile_residual_gain_db=results[
            "delay_profile_residual_gain_db"
        ],
        virtual_pilot_correction_phase_rad=results[
            "virtual_pilot_correction_phase_rad"
        ],
        example_rx_constellation=results["example_iq_rcv"],
        example_iq_rcv=results["example_iq_rcv"],
        example_rx_constellation_single_pilot=results[
            "example_iq_rcv_single"
        ],
        example_iq_rcv_single=results["example_iq_rcv_single"],
        example_rx_frame=results["example_frame_rcv"],
        example_frame_rcv=results["example_frame_rcv"],
        metadata_json=metadata_json,
        **compatible,
    )
    json_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return npz_path, json_path, figure_path, metadata


def _companion_figure_path(figure_path: Path, label: str) -> Path:
    prefix = "channel_plots_"
    if figure_path.stem.startswith(prefix):
        stem = f"{prefix}{label}_{figure_path.stem[len(prefix):]}"
    else:
        stem = f"{figure_path.stem}_{label}"
    return figure_path.with_name(f"{stem}{figure_path.suffix}")


def _psd_nfft(num_samples: int) -> int:
    next_power_of_two = 1 << int(np.ceil(np.log2(max(1, num_samples))))
    return min(4096, max(256, next_power_of_two))


def _plot_rx_tx_lines(
    axis: Any,
    values: np.ndarray,
    *,
    label_suffix: str = "",
    linestyle: str = "-",
) -> None:
    """Plot an ``(RX, TX, value)`` array with explicit link labels."""
    if values.ndim != 3:
        raise ValueError(
            "values must have shape (rx, tx, value); "
            f"got {values.shape}"
        )
    for rx_ant_idx in range(values.shape[0]):
        for tx_ant_idx in range(values.shape[1]):
            axis.plot(
                values[rx_ant_idx, tx_ant_idx],
                linestyle=linestyle,
                label=(
                    f"Rx {rx_ant_idx} / Tx {tx_ant_idx}{label_suffix}"
                ),
            )
    if values.shape[0] * values.shape[1] > 1:
        axis.legend(fontsize="small", ncol=2)


def plot_results(
    figure_path: str | Path,
    params: dict[str, Any],
    results: dict[str, Any],
    *,
    show: bool = True,
) -> dict[str, Path]:
    """Create and save capture summary plots for one or more antennas."""

    import matplotlib

    if not show:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    summary_path = Path(figure_path)
    if not summary_path.suffix:
        summary_path = summary_path.with_suffix(".png")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    delay_path = _companion_figure_path(summary_path, "delay")
    constellation_path = _companion_figure_path(summary_path, "constellation")
    equalized_psd_path = _companion_figure_path(summary_path, "eqpsd")

    channel_mean = _capture_csi_rx_tx_array(
        results["channel_mean_fd"],
        "channel_mean_fd",
    )
    delay_response = _capture_csi_rx_tx_array(
        results["channel_impulse_response_td"],
        "channel_impulse_response_td",
    )
    delay_response_single = _capture_csi_rx_tx_array(
        results["channel_impulse_response_td_single_pilot"],
        "channel_impulse_response_td_single_pilot",
    )
    variance_raw = _capture_csi_rx_tx_array(
        results["virtual_pilot_variance_raw"],
        "virtual_pilot_variance_raw",
    )
    variance_aligned = _capture_csi_rx_tx_array(
        results["virtual_pilot_variance_aligned"],
        "virtual_pilot_variance_aligned",
    )
    example_iq = _example_antenna_array(
        results["example_iq_rcv"],
        "example_iq_rcv",
    )
    example_iq_single = _example_antenna_array(
        results.get("example_iq_rcv_single", results["example_iq_rcv"]),
        "example_iq_rcv_single",
    )
    example_frame = _example_antenna_array(
        results["example_frame_rcv"],
        "example_frame_rcv",
    )

    (
        num_captures,
        num_csi_samples,
        num_rx_ant,
        num_tx_ant,
        num_subcarriers,
    ) = channel_mean.shape
    expected_channel_shape = (
        num_captures,
        num_csi_samples,
        num_rx_ant,
        num_tx_ant,
        num_subcarriers,
    )
    if delay_response.shape[:4] != expected_channel_shape[:4]:
        raise ValueError("Channel and delay-response CSI/RX/TX axes do not match")
    if delay_response_single.shape != delay_response.shape:
        raise ValueError(
            "Single-pilot and averaged delay responses must have equal shapes"
        )
    if variance_raw.shape != expected_channel_shape:
        raise ValueError(
            "virtual_pilot_variance_raw must match channel_mean_fd; "
            f"got {variance_raw.shape} and {expected_channel_shape}"
        )
    if variance_raw.shape != variance_aligned.shape:
        raise ValueError(
            "Raw and aligned pilot variances must have the same shape"
        )
    if example_iq.shape[0] != num_rx_ant:
        raise ValueError("Example IQ and channel RX antenna axes do not match")

    average_channel_magnitude = np.mean(
        np.abs(channel_mean),
        axis=(0, 1),
    )
    average_delay_magnitude = np.mean(
        np.abs(delay_response),
        axis=(0, 1),
    )
    average_delay_magnitude_single = np.mean(
        np.abs(delay_response_single),
        axis=(0, 1),
    )
    average_variance_raw = np.mean(variance_raw, axis=(0, 1, 2, 3))
    average_variance_aligned = np.mean(
        variance_aligned,
        axis=(0, 1, 2, 3),
    )
    num_virtual_pilots = max(int(params["num_virtual_pilots"]), 1)
    variance_gain_db = 10.0 * np.log10(
        (average_variance_raw + _EPSILON)
        / (
            average_variance_aligned / num_virtual_pilots
            + _EPSILON
        )
    )
    measured_gain_db = float(np.mean(results["virtual_pilot_snr_gain_db"]))

    figures: list[Any] = []

    summary_figure, axes = plt.subplots(2, 3, figsize=(18, 9))
    figures.append(summary_figure)
    summary_figure.suptitle(
        "Channel capture summary | "
        f"BW={params['bandwidth_mhz']} MHz | N={num_subcarriers} | "
        f"FFT={params['FFT_SIZE']} | captures={num_captures} | "
        f"CSI/frame={num_csi_samples} | Rx={num_rx_ant} | "
        f"Tx={num_tx_ant} | VP={num_virtual_pilots} | "
        f"measured pilot gain={measured_gain_db:.2f} dB"
    )

    _plot_rx_tx_lines(axes[0, 0], average_channel_magnitude)
    axes[0, 0].set_title("Average |H[k]| across captures and CSI samples")
    axes[0, 0].set_xlabel("Subcarrier index")
    axes[0, 0].set_ylabel("Magnitude")
    axes[0, 0].grid(True, alpha=0.3)

    image = axes[0, 1].imshow(
        np.mean(np.abs(channel_mean), axis=(2, 3)).reshape(
            num_captures * num_csi_samples,
            num_subcarriers,
        ),
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    axes[0, 1].set_title("Mean-link |H[k]| over captures and CSI samples")
    axes[0, 1].set_xlabel("Subcarrier index")
    axes[0, 1].set_ylabel("Capture × CSI sample index")
    summary_figure.colorbar(image, ax=axes[0, 1], fraction=0.046, pad=0.04)

    _plot_rx_tx_lines(axes[0, 2], average_delay_magnitude)
    axes[0, 2].set_title("Average |h[n]| in delay domain")
    axes[0, 2].set_xlabel("Delay sample")
    axes[0, 2].set_ylabel("Magnitude")
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(
        10.0 * np.log10(average_variance_raw + _EPSILON),
        label="Raw virtual-pilot variance",
    )
    axes[1, 0].plot(
        10.0 * np.log10(average_variance_aligned + _EPSILON),
        label="Phase-aligned virtual-pilot variance",
    )
    axes[1, 0].set_title("Virtual pilot estimator variance")
    axes[1, 0].set_xlabel("Subcarrier index")
    axes[1, 0].set_ylabel("Variance (dB)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # variance_gain_db is already in dB; do not apply a second logarithm.
    axes[1, 1].plot(variance_gain_db)
    axes[1, 1].axhline(
        10.0 * np.log10(num_virtual_pilots),
        color="tab:orange",
        linestyle="--",
        label="Ideal averaging gain",
    )
    axes[1, 1].set_title("Estimated variance reduction from averaging")
    axes[1, 1].set_xlabel("Subcarrier index")
    axes[1, 1].set_ylabel("Gain (dB)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    for antenna_idx, samples in enumerate(example_frame):
        axes[1, 2].psd(
            samples,
            NFFT=_psd_nfft(samples.size),
            Fs=float(params["sampling_rate"]),
            scale_by_freq=False,
            label=f"Rx {antenna_idx}",
        )
    axes[1, 2].set_title("Example received-frame spectrum")
    axes[1, 2].set_xlabel("Frequency (Hz)")
    axes[1, 2].set_ylabel("Power spectrum (dB)")
    if num_rx_ant > 1:
        axes[1, 2].legend()

    summary_figure.tight_layout(rect=(0, 0, 1, 0.95))
    summary_figure.savefig(summary_path, dpi=150, bbox_inches="tight")

    delay_figure, delay_axes = plt.subplots(1, 2, figsize=(13, 5))
    figures.append(delay_figure)
    _plot_rx_tx_lines(
        delay_axes[0],
        average_delay_magnitude_single,
        label_suffix=" single pilot",
    )
    _plot_rx_tx_lines(
        delay_axes[0],
        average_delay_magnitude,
        label_suffix=f" {num_virtual_pilots}-pilot average",
        linestyle="--",
    )
    delay_axes[0].set_title("Delay-domain mean profile")
    delay_axes[0].set_xlabel("Delay sample")
    delay_axes[0].set_ylabel("Magnitude")
    delay_axes[0].grid(True, alpha=0.3)
    delay_axes[0].legend(fontsize="x-small", ncol=2)

    residual_gains = np.asarray(
        results["delay_profile_residual_gain_db"],
        dtype=np.float32,
    )
    expected_metric_shape = (
        num_captures,
        num_csi_samples,
        num_rx_ant,
        num_tx_ant,
    )
    if residual_gains.shape != expected_metric_shape:
        raise ValueError(
            "delay_profile_residual_gain_db must have shape "
            "(capture, csi, rx, tx); "
            f"got {residual_gains.shape}, expected {expected_metric_shape}"
        )
    mean_residual_gain = np.mean(residual_gains, axis=(0, 1))
    flattened_gain = mean_residual_gain.reshape(-1)
    link_indices = np.arange(flattened_gain.size)
    delay_axes[1].bar(link_indices, flattened_gain)
    delay_axes[1].set_xticks(link_indices)
    delay_axes[1].set_xticklabels(
        [
            f"R{rx_idx}/T{tx_idx}"
            for rx_idx in range(num_rx_ant)
            for tx_idx in range(num_tx_ant)
        ],
        rotation=45,
        ha="right",
    )
    delay_axes[1].set_ylabel("Residual suppression gain (dB)")
    delay_axes[1].set_title("Single-pilot to averaged delay residual gain")
    delay_axes[1].grid(True, axis="y", alpha=0.3)
    delay_figure.suptitle(
        "Delay profile comparison | "
        f"mean residual suppression={np.mean(flattened_gain):.2f} dB"
    )
    delay_figure.tight_layout(rect=(0, 0, 1, 0.94))
    delay_figure.savefig(delay_path, dpi=150, bbox_inches="tight")

    constellation_figure, constellation_axes = plt.subplots(
        1,
        2,
        figsize=(12, 6),
    )
    figures.append(constellation_figure)
    modulation_order = int(params["modulation_order"])
    if modulation_order in modulations:
        peak_amplitude = 2.0 * float(
            np.max(np.abs(np.real(modulations[modulation_order])))
        )
    else:
        peak_amplitude = 1.1 * float(
            max(
                np.max(np.abs(example_iq_single)),
                np.max(np.abs(example_iq)),
                1.0,
            )
        )
    for rx_ant_idx in range(num_rx_ant):
        constellation_axes[0].scatter(
            np.real(example_iq_single[rx_ant_idx]),
            np.imag(example_iq_single[rx_ant_idx]),
            s=1.0,
            marker="o",
            alpha=0.6,
            label=f"Rx {rx_ant_idx}",
        )
        constellation_axes[1].scatter(
            np.real(example_iq[rx_ant_idx]),
            np.imag(example_iq[rx_ant_idx]),
            s=1.0,
            marker="o",
            alpha=0.6,
            label=f"Rx {rx_ant_idx}",
        )
    for axis, title in zip(
        constellation_axes,
        (
            "Single-pilot equalized constellation",
            f"{num_virtual_pilots}-pilot averaged equalized constellation",
        ),
        strict=True,
    ):
        axis.set_xlim([-peak_amplitude, peak_amplitude])
        axis.set_ylim([-peak_amplitude, peak_amplitude])
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("In-Phase")
        axis.set_ylabel("Quadrature")
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        if num_rx_ant > 1:
            axis.legend(markerscale=5)
    constellation_figure.tight_layout()
    constellation_figure.savefig(
        constellation_path,
        dpi=150,
        bbox_inches="tight",
    )

    psd_figure, psd_axes = plt.subplots(1, 2, figsize=(12, 5))
    figures.append(psd_figure)
    for rx_ant_idx in range(num_rx_ant):
        single_samples = example_iq_single[rx_ant_idx]
        averaged_samples = example_iq[rx_ant_idx]
        if single_samples.size:
            psd_axes[0].psd(
                single_samples,
                NFFT=_psd_nfft(single_samples.size),
                Fs=float(params["sampling_rate"]),
                scale_by_freq=False,
                label=f"Rx {rx_ant_idx}",
            )
        if averaged_samples.size:
            psd_axes[1].psd(
                averaged_samples,
                NFFT=_psd_nfft(averaged_samples.size),
                Fs=float(params["sampling_rate"]),
                scale_by_freq=False,
                label=f"Rx {rx_ant_idx}",
            )
    for axis, title in zip(
        psd_axes,
        (
            "Single-pilot equalized PSD",
            f"{num_virtual_pilots}-pilot averaged equalized PSD",
        ),
        strict=True,
    ):
        axis.set_title(title)
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("Power spectrum (dB)")
        axis.grid(True, alpha=0.3)
        if len(axis.lines) > 0 and num_rx_ant > 1:
            axis.legend()
        if len(axis.lines) == 0:
            axis.text(
                0.5,
                0.5,
                "No equalized data symbols\n(channel-sounding mode)",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
    psd_figure.suptitle(
        "Equalized spectrum | "
        f"measured equalization gain={measured_gain_db:.2f} dB | "
        f"VP={num_virtual_pilots}"
    )
    psd_figure.tight_layout(rect=(0, 0, 1, 0.94))
    psd_figure.savefig(equalized_psd_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    for figure in figures:
        plt.close(figure)

    return {
        "channel_plots": summary_path,
        "delay_profile": delay_path,
        "constellation": constellation_path,
        "equalized_psd": equalized_psd_path,
    }
