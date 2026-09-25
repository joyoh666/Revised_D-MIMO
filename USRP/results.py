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

    All channel-domain arrays keep an explicit receive-antenna axis.  Legacy
    single-antenna receiver outputs without that axis are accepted and promoted
    to an antenna dimension of length one.
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
        if array.ndim == 3:
            array = array[np.newaxis, ...]
        if array.ndim != 4:
            raise ValueError(
                f"{name} must have shape (ant, subframe, slot, subcarrier); "
                f"got {array.shape}"
            )
        if array.shape[-1] != int(self.params["N"]):
            raise ValueError(
                f"{name} has {array.shape[-1]} subcarriers; "
                f"expected {self.params['N']}"
            )
        self._check_antenna_count(array.shape[0], name)
        return array

    def _antenna_subcarriers(self, value: Any, name: str) -> np.ndarray:
        array = np.asarray(value)
        if array.ndim == 1:
            array = array[np.newaxis, ...]
        if array.ndim != 2:
            raise ValueError(
                f"{name} must have shape (ant, subcarrier); got {array.shape}"
            )
        if array.shape[-1] != int(self.params["N"]):
            raise ValueError(
                f"{name} has {array.shape[-1]} subcarriers; "
                f"expected {self.params['N']}"
            )
        self._check_antenna_count(array.shape[0], name)
        return array

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

    def add(self, received: dict[str, Any]) -> None:
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

        h_raw = self._antenna_subcarriers(
            received["h_fd_virtual_avg_raw"],
            "h_fd_virtual_avg_raw",
        ).astype(np.complex64, copy=False)
        h_aligned = self._antenna_subcarriers(
            received["h_fd_virtual_avg_aligned"],
            "h_fd_virtual_avg_aligned",
        ).astype(np.complex64, copy=False)
        variance_raw = self._antenna_subcarriers(
            received["virtual_pilot_variance_raw"],
            "virtual_pilot_variance_raw",
        ).astype(np.float32, copy=False)
        variance_aligned = self._antenna_subcarriers(
            received["virtual_pilot_variance_aligned"],
            "virtual_pilot_variance_aligned",
        ).astype(np.float32, copy=False)

        # Average over subframes and slots independently for every antenna.
        h_mean = np.mean(h_fd, axis=(1, 2), dtype=np.complex128).astype(
            np.complex64
        )
        h_single_mean = np.mean(
            h_single_fd,
            axis=(1, 2),
            dtype=np.complex128,
        ).astype(np.complex64)

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
        self._append("channel_mean_fd", h_mean, dtype=np.complex64)
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
            received["virtual_pilot_common_phases_rad"],
            dtype=np.float32,
        )
        self._append(
            "single_pilot_snr_db",
            received["single_pilot_snr_db"],
            dtype=np.float32,
        )
        self._append(
            "virtual_pilot_snr_db",
            received["virtual_pilot_snr_db"],
            dtype=np.float32,
        )
        self._append(
            "virtual_pilot_snr_gain_db",
            received["virtual_pilot_snr_gain_db"],
            dtype=np.float32,
        )
        self._append(
            "virtual_pilot_correction_phase_rad",
            received["virtual_pilot_correction_phase_rad"],
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
        """Stack all captures and return the original result dictionary schema."""

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


def _capture_antenna_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2:
        array = array[:, np.newaxis, :]
    if array.ndim != 3:
        raise ValueError(
            f"{name} must have shape (capture, ant, value); got {array.shape}"
        )
    return array


def _example_antenna_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (ant, value); got {array.shape}")
    return array


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

    channel_estimates = np.asarray(results["channel_estimates_fd"])
    channel_mean = _capture_antenna_array(
        results["channel_mean_fd"],
        "channel_mean_fd",
    )
    delay_response = _capture_antenna_array(
        results["channel_impulse_response_td"],
        "channel_impulse_response_td",
    )
    delay_response_single = _capture_antenna_array(
        results["channel_impulse_response_td_single_pilot"],
        "channel_impulse_response_td_single_pilot",
    )
    example_iq_single = _example_antenna_array(
        results["example_iq_rcv_single"],
        "example_iq_rcv_single",
    )
    num_receive_antennas = int(
        results.get("num_receive_antennas", channel_mean.shape[1])
    )
    num_virtual_pilots = int(params["num_virtual_pilots"])

    metadata = {
        "result_schema_version": 2,
        "array_axis_order": {
            "channel_estimates_fd": [
                "capture",
                "rx_antenna",
                "subframe",
                "slot",
                "subcarrier",
            ],
            "channel_mean_fd": ["capture", "rx_antenna", "subcarrier"],
            "channel_impulse_response_td": [
                "capture",
                "rx_antenna",
                "delay_sample",
            ],
            "virtual_pilot_common_phases_rad": [
                "capture",
                "virtual_pilot",
                "rx_antenna",
            ],
            "rfo_estimates_hz": ["capture", "rx_antenna", "slot"],
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
        "saved_channel_tensor_shape": list(channel_estimates.shape),
        "saved_mean_channel_shape": list(channel_mean.shape),
        "saved_delay_response_shape": list(delay_response.shape),
        "saved_single_pilot_delay_response_shape": list(
            delay_response_single.shape
        ),
        "saved_example_single_pilot_iq_shape": list(example_iq_single.shape),
        "samples_per_channel_vector": params["N"],
        "samples_per_delay_response": params["FFT_SIZE"],
        "reference_sequence_type": "seeded_unit_power_qpsk",
        "reference_sequence_seed": int(reference_sequence_seed),
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
        example_rx_constellation_single_pilot=results[
            "example_iq_rcv_single"
        ],
        example_iq_rcv_single=results["example_iq_rcv_single"],
        example_rx_frame=results["example_frame_rcv"],
        metadata_json=metadata_json,
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


def _plot_antenna_lines(
    axis: Any,
    values: np.ndarray,
    *,
    label_prefix: str = "Rx",
    linestyle: str = "-",
) -> None:
    for antenna_idx, row in enumerate(values):
        axis.plot(
            row,
            linestyle=linestyle,
            label=f"{label_prefix} {antenna_idx}",
        )
    if values.shape[0] > 1:
        axis.legend()


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

    channel_mean = _capture_antenna_array(
        results["channel_mean_fd"],
        "channel_mean_fd",
    )
    delay_response = _capture_antenna_array(
        results["channel_impulse_response_td"],
        "channel_impulse_response_td",
    )
    delay_response_single = _capture_antenna_array(
        results["channel_impulse_response_td_single_pilot"],
        "channel_impulse_response_td_single_pilot",
    )
    variance_raw = _capture_antenna_array(
        results["virtual_pilot_variance_raw"],
        "virtual_pilot_variance_raw",
    )
    variance_aligned = _capture_antenna_array(
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

    num_captures, num_rx_ant, num_subcarriers = channel_mean.shape
    if delay_response.shape[:2] != (num_captures, num_rx_ant):
        raise ValueError("Channel and delay-response antenna axes do not match")
    if variance_raw.shape != variance_aligned.shape:
        raise ValueError("Raw and aligned pilot variances must have the same shape")

    average_channel_magnitude = np.mean(np.abs(channel_mean), axis=0)
    average_delay_magnitude = np.mean(np.abs(delay_response), axis=0)
    average_delay_magnitude_single = np.mean(
        np.abs(delay_response_single),
        axis=0,
    )
    average_variance_raw = np.mean(variance_raw, axis=(0, 1))
    average_variance_aligned = np.mean(variance_aligned, axis=(0, 1))
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
        f"Rx={num_rx_ant} | VP={num_virtual_pilots} | "
        f"measured pilot gain={measured_gain_db:.2f} dB"
    )

    _plot_antenna_lines(axes[0, 0], average_channel_magnitude)
    axes[0, 0].set_title("Average |H[k]| across captures")
    axes[0, 0].set_xlabel("Subcarrier index")
    axes[0, 0].set_ylabel("Magnitude")
    axes[0, 0].grid(True, alpha=0.3)

    image = axes[0, 1].imshow(
        np.mean(np.abs(channel_mean), axis=1),
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    axes[0, 1].set_title("Mean-Rx |H[k]| for repeated captures")
    axes[0, 1].set_xlabel("Subcarrier index")
    axes[0, 1].set_ylabel("Capture index")
    summary_figure.colorbar(image, ax=axes[0, 1], fraction=0.046, pad=0.04)

    _plot_antenna_lines(axes[0, 2], average_delay_magnitude)
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
    for rx_ant_idx in range(num_rx_ant):
        delay_axes[0].plot(
            average_delay_magnitude_single[rx_ant_idx],
            label=f"Rx {rx_ant_idx} single pilot",
        )
        delay_axes[0].plot(
            average_delay_magnitude[rx_ant_idx],
            linestyle="--",
            label=f"Rx {rx_ant_idx} {num_virtual_pilots}-pilot average",
        )
    delay_axes[0].set_title("Delay-domain mean profile")
    delay_axes[0].set_xlabel("Delay sample")
    delay_axes[0].set_ylabel("Magnitude")
    delay_axes[0].grid(True, alpha=0.3)
    delay_axes[0].legend(fontsize="small")

    residual_gains = np.asarray(results["delay_profile_residual_gain_db"])
    if residual_gains.ndim == 1:
        residual_gains = residual_gains[:, np.newaxis]
    if residual_gains.ndim != 2 or residual_gains.shape[1] != num_rx_ant:
        raise ValueError(
            "delay_profile_residual_gain_db must have shape "
            "(capture, num_rx_ant); "
            f"got {residual_gains.shape}"
        )
    mean_residual_gain = np.mean(residual_gains, axis=0)
    delay_axes[1].bar(np.arange(num_rx_ant), mean_residual_gain)
    delay_axes[1].set_xticks(np.arange(num_rx_ant))
    delay_axes[1].set_xticklabels(
        [f"Rx {idx}" for idx in range(num_rx_ant)]
    )
    delay_axes[1].set_ylabel("Residual suppression gain (dB)")
    delay_axes[1].set_title("Single-pilot to averaged delay residual gain")
    delay_axes[1].grid(True, axis="y", alpha=0.3)
    delay_figure.suptitle(
        "Delay profile comparison | "
        f"mean residual suppression={np.mean(mean_residual_gain):.2f} dB"
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
        psd_axes[0].psd(
            single_samples,
            NFFT=_psd_nfft(single_samples.size),
            Fs=float(params["sampling_rate"]),
            scale_by_freq=False,
            label=f"Rx {rx_ant_idx}",
        )
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
        if num_rx_ant > 1:
            axis.legend()
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
