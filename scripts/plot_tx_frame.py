"""Generate documentation figures for the six-antenna sounding frame."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from config.radio_config import (  # noqa: E402
    BANDWIDTH,
    REFERENCE_SEQUENCE_SEED,
)
from USRP.build_frame import build_frame  # noqa: E402
from USRP.helper_functions import (  # noqa: E402
    generate_known_reference_sequence,
    get_system_params,
)


MUTED = 0
DATA = 1
RFO_REFERENCE = 2
VIRTUAL_PILOT = 3
PSS = 4
SSS = 5

CATEGORY_COLORS = (
    "#E5E7EB",  # muted
    "#60A5FA",  # data
    "#F59E0B",  # RFO reference
    "#10B981",  # virtual pilot
    "#EF4444",  # PSS
    "#8B5CF6",  # SSS
)
CATEGORY_LABELS = (
    "Muted / zero",
    "DATA (QPSK)",
    "TX0 RFO reference",
    "Virtual pilot",
    "PSS (62 central REs)",
    "SSS (62 central REs)",
)


def _frame_category_map(params: dict[str, object]) -> np.ndarray:
    """Return one categorical value per TX antenna and OFDM symbol."""
    num_tx = int(params["num_tx_ant"])
    num_subframes = int(params["num_subframe_per_frame"])
    slots_per_subframe = int(params["num_slot_per_subframe"])
    symbols_per_slot = int(params["num_symbols_per_slot"])
    sync_tx = int(params["sync_tx_idx"])

    num_slots = num_subframes * slots_per_subframe
    categories = np.full(
        (num_tx, num_slots, symbols_per_slot),
        DATA,
        dtype=np.int8,
    )

    # Symbol 0 in every slot is an RFO reference transmitted only by TX0.
    categories[:, :, 0] = MUTED
    categories[sync_tx, :, 0] = RFO_REFERENCE

    pilot_positions = params["virtual_pilot_positions"]
    for sf_idx, slot_idx, symbol_idx in {
        position[1:] for position in pilot_positions
    }:
        global_slot = int(sf_idx) * slots_per_subframe + int(slot_idx)
        categories[:, global_slot, int(symbol_idx)] = MUTED
    for tx_idx, sf_idx, slot_idx, symbol_idx in pilot_positions:
        global_slot = int(sf_idx) * slots_per_subframe + int(slot_idx)
        categories[int(tx_idx), global_slot, int(symbol_idx)] = VIRTUAL_PILOT

    # Each 5 ms frame has one SSS/PSS pair on TX0 at its beginning.
    categories[:, 0, 6] = MUTED
    categories[sync_tx, 0, 6] = PSS
    categories[:, 0, 5] = MUTED
    categories[sync_tx, 0, 5] = SSS

    return categories.reshape(num_tx, -1)


def _plot_frame_layout(params: dict[str, object], output_path: Path) -> None:
    categories = _frame_category_map(params)
    num_tx, num_symbols = categories.shape
    symbols_per_slot = int(params["num_symbols_per_slot"])
    slots_per_subframe = int(params["num_slot_per_subframe"])
    symbols_per_subframe = symbols_per_slot * slots_per_subframe
    num_subframes = int(params["num_subframe_per_frame"])

    cmap = ListedColormap(CATEGORY_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, len(CATEGORY_COLORS) + 0.5), cmap.N)

    fig, ax = plt.subplots(figsize=(19, 5.6), constrained_layout=True)
    ax.imshow(
        categories,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )

    for slot_boundary in range(symbols_per_slot, num_symbols, symbols_per_slot):
        ax.axvline(slot_boundary - 0.5, color="white", linewidth=0.7, alpha=0.9)
    for sf_boundary in range(
        symbols_per_subframe,
        num_symbols,
        symbols_per_subframe,
    ):
        ax.axvline(sf_boundary - 0.5, color="#111827", linewidth=1.5)

    ax.set_yticks(np.arange(num_tx), [f"TX{tx_idx}" for tx_idx in range(num_tx)])
    ax.set_xticks(
        np.arange(num_subframes) * symbols_per_subframe
        + (symbols_per_subframe - 1) / 2,
        [f"SF{sf_idx}" for sf_idx in range(num_subframes)],
    )
    ax.set_xlabel(
        f"Time: {num_subframes} subframes × 2 slots × 7 OFDM symbols "
        "(thin line = slot, dark line = subframe)"
    )
    ax.set_ylabel("Transmit antenna")
    ax.set_title("Six-antenna channel-sounding frame: FDM pilot allocation")
    ax.set_xlim(-0.5, num_symbols - 0.5)

    legend_handles = [
        Patch(facecolor=color, edgecolor="#374151", label=label)
        for color, label in zip(CATEGORY_COLORS, CATEGORY_LABELS)
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=3,
        frameon=False,
    )

    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_symbol_values(
    params: dict[str, object],
    resource_maps: np.ndarray,
    output_path: Path,
) -> None:
    num_subcarriers = int(params["N"])
    sync_tx = int(params["sync_tx_idx"])
    slots_per_subframe = int(params["num_slot_per_subframe"])

    first_virtual_position = params["virtual_pilot_positions"][0]
    pilot_tx, pilot_sf, pilot_slot, pilot_symbol = first_virtual_position

    examples = (
        (
            "DATA (TX0, SF0/S0/L1)",
            resource_maps[sync_tx, 0, 0, 1],
        ),
        (
            "RFO reference (TX0, every slot/L0)",
            resource_maps[sync_tx, 0, 0, 0],
        ),
        (
            f"Virtual pilot (TX{pilot_tx}, SF{pilot_sf}/S{pilot_slot}/L{pilot_symbol})",
            resource_maps[pilot_tx, pilot_sf, pilot_slot, pilot_symbol],
        ),
        (
            "SSS (TX0, SF0/S0/L5)",
            resource_maps[sync_tx, 0, 0, 5],
        ),
        (
            "PSS (TX0, SF0/S0/L6)",
            resource_maps[sync_tx, 0, 0, 6],
        ),
        (
            "Muted (TX1, SF0/S0/L0)",
            resource_maps[1, 0, 0, 0],
        ),
    )

    del slots_per_subframe  # retained in the labels above through sf/slot indices
    x = np.arange(num_subcarriers)
    fig, axes = plt.subplots(
        len(examples),
        1,
        figsize=(16, 11),
        sharex=True,
        constrained_layout=True,
    )
    for ax, (label, values) in zip(axes, examples):
        ax.plot(x, values.real, color="#2563EB", linewidth=0.75, label="Real")
        ax.plot(x, values.imag, color="#DC2626", linewidth=0.75, label="Imag")
        ax.axhline(0.0, color="#6B7280", linewidth=0.5)
        ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=16)
        ax.set_ylim(-1.15, 1.15)
        ax.grid(axis="y", alpha=0.2)

    axes[0].legend(loc="upper right", ncol=2, frameon=False)
    axes[-1].set_xlabel(
        f"Active-subcarrier index (N = {num_subcarriers}; DC itself is not transmitted)"
    )
    fig.suptitle(
        "Complex frequency-domain values in representative OFDM symbols",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    output_dir = REPOSITORY_ROOT / "docs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    params = get_system_params(BANDWIDTH)
    reference_seed = REFERENCE_SEQUENCE_SEED + int(10 * BANDWIDTH)
    known_ref_seq = generate_known_reference_sequence(
        int(params["N"]),
        seed=reference_seed,
    )

    random_state = np.random.get_state()
    np.random.seed(2026)
    try:
        frame = build_frame(params, known_ref_seq)
    finally:
        np.random.set_state(random_state)

    _plot_frame_layout(params, output_dir / "tx_frame_layout.png")
    _plot_symbol_values(
        params,
        frame["resource_maps_tx"],
        output_dir / "tx_symbol_values.png",
    )

    print(output_dir / "tx_frame_layout.png")
    print(output_dir / "tx_symbol_values.png")


if __name__ == "__main__":
    main()
