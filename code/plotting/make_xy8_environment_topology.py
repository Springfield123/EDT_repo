#!/usr/bin/env python3
"""Create the Stage-6 strain/temperature-conditioned EDT summary figure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED = ROOT / "data" / "processed"
DEFAULT_OUTDIR = ROOT / "outputs" / "reproduced_figures"

SCORE_SUMMARY = "xy8_environment_score_summary.csv"
CONFIRMATION_SELECTION = "xy8_environment_confirmation_selection.csv"
CONFIRMATION_RESULTS = "xy8_environment_confirmation_results.csv"
CONFIRMATION_SURFACES = "xy8_environment_confirmation_surfaces.npz"
TOPOLOGY_SUMMARY = "xy8_environment_topology_summary.json"

FIGURE_BASENAME = "fig_xy8_environment_topology"
DEPTH_SOURCE = "xy8_environment_depth_maps.csv"
CONTOUR_SOURCE = "xy8_environment_contour_source.csv"
FIGURE_SUMMARY = "xy8_environment_figure_summary.json"

COHERENCE_VARIANT = "jarmola_experimental_S8"
STRAIN_PATH = "axial_zz"
REFERENCE_CONDITION = "zero_strain_T300K"
SHIFTED_CONDITION = "axial_zz_T330K_A+20us"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path.name}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def categorical_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or np.any(np.diff(values) <= 0.0):
        raise ValueError("Coordinates must be a strictly increasing one-dimensional array.")
    middle = 0.5 * (values[:-1] + values[1:])
    return np.concatenate(
        ([values[0] - (middle[0] - values[0])], middle, [values[-1] + (values[-1] - middle[-1])])
    )


def build_depth_map_rows(score_rows: list[dict[str, str]]) -> list[dict]:
    """Return the frozen operational axial-strain slice used in panels A and B."""
    selected = [
        row
        for row in score_rows
        if row["strain_path"] == STRAIN_PATH
        and row["coherence_variant"] == COHERENCE_VARIANT
    ]
    policies = ("fixed_reference_carrier", "ideal_retracked_carrier")
    output = []
    for policy in policies:
        policy_rows = [row for row in selected if row["carrier_policy"] == policy]
        temperatures = sorted({float(row["temperature_K"]) for row in policy_rows})
        amplitudes = sorted({float(row["amplitude_microstrain"]) for row in policy_rows})
        if len(policy_rows) != len(temperatures) * len(amplitudes):
            raise ValueError(f"Incomplete or duplicated {policy} temperature/strain map.")
        for row in policy_rows:
            output.append(
                {
                    "carrier_policy": policy,
                    "coherence_variant": COHERENCE_VARIANT,
                    "strain_path": STRAIN_PATH,
                    "temperature_K": float(row["temperature_K"]),
                    "amplitude_microstrain": float(row["amplitude_microstrain"]),
                    "transition_shift_kHz": float(row["transition_shift_kHz"]),
                    "applied_detuning_kHz": float(row["applied_detuning_Hz"]) / 1e3,
                    "best_N": int(row["best_N"]),
                    "window_N_min": int(row["window_N_min"]),
                    "window_N_max": int(row["window_N_max"]),
                    "N20_penalty_percent": 100.0 * (float(row["N20_relative_penalty"]) - 1.0),
                    "boundary_status": row["boundary_status"],
                }
            )
    return sorted(
        output,
        key=lambda row: (
            policies.index(row["carrier_policy"]),
            row["temperature_K"],
            row["amplitude_microstrain"],
        ),
    )


def _find_confirmation_index(
    selection_rows: list[dict[str, str]], condition_id: str
) -> int:
    matches = [
        index
        for index, row in enumerate(selection_rows)
        if row["condition_id"] == condition_id
        and row["carrier_policy"] == "fixed_reference_carrier"
        and row["coherence_variant"] == COHERENCE_VARIANT
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one confirmed surface for {condition_id}; found {len(matches)}.")
    return matches[0]


def build_contour_rows(
    selection_rows: list[dict[str, str]],
    confirmation_rows: list[dict[str, str]],
    archive: np.lib.npyio.NpzFile,
    *,
    tau_min_us: float = 1.225,
    tau_max_us: float = 1.275,
    max_depth: int = 48,
) -> tuple[list[dict], list[dict]]:
    """Extract two fine-grid surfaces without recomputing or refitting them."""
    eta = np.asarray(archive["eta_fine"], dtype=float)
    taus_s = np.asarray(archive["taus_fine_s"], dtype=float)
    depths = np.asarray(archive["depths_N"], dtype=int)
    source_indices = np.asarray(archive["discovery_surface_indices"], dtype=int)
    if eta.shape != (len(selection_rows), taus_s.size, depths.size):
        raise ValueError("Fine-grid surface archive does not match its frozen selection.")
    if not np.array_equal(
        source_indices,
        np.asarray([int(row["discovery_surface_index"]) for row in selection_rows]),
    ):
        raise ValueError("Fine-grid surfaces are not aligned to the frozen selection.")

    confirmation_by_index = {
        int(row["confirmation_surface_index"]): row for row in confirmation_rows
    }
    tau_mask = (taus_s * 1e6 >= tau_min_us) & (taus_s * 1e6 <= tau_max_us)
    depth_mask = depths <= max_depth
    output = []
    examples = []
    for label, condition_id in (
        ("Nominal reference", REFERENCE_CONDITION),
        ("Environment-shifted", SHIFTED_CONDITION),
    ):
        index = _find_confirmation_index(selection_rows, condition_id)
        score = eta[index]
        eta_min = float(np.min(score))
        confirmation = confirmation_by_index[index]
        selection = selection_rows[index]
        examples.append(
            {
                "example": label,
                "condition_id": condition_id,
                "confirmation_surface_index": index,
                "temperature_K": float(selection["temperature_K"]),
                "amplitude_microstrain": float(selection["amplitude_microstrain"]),
                "applied_detuning_kHz": float(selection["applied_detuning_Hz"]) / 1e3,
                "best_N": int(confirmation["fine_best_N"]),
                "window_N_min": int(confirmation["fine_primary_N_min"]),
                "window_N_max": int(confirmation["fine_primary_N_max"]),
                "beta_0_four": int(confirmation["fine_beta_0_four"]),
                "beta_0_eight": int(confirmation["fine_beta_0_eight"]),
                "timing_width_ns": 1e9 * float(confirmation["fine_timing_width_estimate_s"]),
                "confirmation_passed": confirmation["confirmation_passed"] == "True",
            }
        )
        for tau_index in np.flatnonzero(tau_mask):
            for depth_index in np.flatnonzero(depth_mask):
                relative = float(score[tau_index, depth_index] / eta_min)
                output.append(
                    {
                        "example": label,
                        "condition_id": condition_id,
                        "temperature_K": float(selection["temperature_K"]),
                        "amplitude_microstrain": float(selection["amplitude_microstrain"]),
                        "applied_detuning_kHz": float(selection["applied_detuning_Hz"]) / 1e3,
                        "tau_us": float(taus_s[tau_index] * 1e6),
                        "N": int(depths[depth_index]),
                        "eta_proxy": float(score[tau_index, depth_index]),
                        "relative_to_best": relative,
                        "penalty_percent": 100.0 * (relative - 1.0),
                        "inside_10pct_region": relative <= 1.10 + 1e-12,
                    }
                )
    return output, examples


def _grid(rows: list[dict], value_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    temperatures = np.asarray(sorted({row["temperature_K"] for row in rows}), dtype=float)
    amplitudes = np.asarray(sorted({row["amplitude_microstrain"] for row in rows}), dtype=float)
    by_coordinate = {
        (row["temperature_K"], row["amplitude_microstrain"]): row for row in rows
    }
    values = np.asarray(
        [[by_coordinate[(temperature, amplitude)][value_key] for temperature in temperatures]
         for amplitude in amplitudes],
        dtype=float,
    )
    return temperatures, amplitudes, values


def _plot(
    depth_rows: list[dict],
    contour_rows: list[dict],
    examples: list[dict],
    topology_summary: dict,
    png_path: Path,
    pdf_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    depth_values = [12, 14, 16, 20]
    depth_colors = ["#5a189a", "#2c7bb6", "#00a6ca", "#f6c85f"]
    depth_cmap = ListedColormap(depth_colors)
    depth_norm = BoundaryNorm([11, 13, 15, 18, 21], depth_cmap.N)

    fig = plt.figure(figsize=(13.6, 8.6))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.02, 1.0), hspace=0.38, wspace=0.24)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]

    for panel_index, (axis, policy, title) in enumerate(
        (
            (axes[0], "fixed_reference_carrier", "A  Fixed carrier: environment selects shallower depth"),
            (axes[1], "ideal_retracked_carrier", "B  Carrier retracking preserves the nominal depth"),
        )
    ):
        subset = [row for row in depth_rows if row["carrier_policy"] == policy]
        temperatures, amplitudes, best_n = _grid(subset, "best_N")
        axis.pcolormesh(
            categorical_edges(temperatures),
            categorical_edges(amplitudes),
            best_n,
            cmap=depth_cmap,
            norm=depth_norm,
            shading="flat",
        )
        axis.set_xlabel("Temperature (K)")
        axis.set_ylabel(r"Axial strain amplitude ($\mu\varepsilon$)")
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
        axis.set_xticks([280, 290, 300, 310, 320, 330])
        axis.set_yticks([-100, -50, 0, 50, 100])
        axis.scatter([300], [0], marker="*", s=115, facecolor="white", edgecolor="black", zorder=5)
        if policy == "fixed_reference_carrier":
            axis.scatter([330], [20], marker="o", s=75, facecolor="none", edgecolor="white", linewidth=1.5)
            axis.annotate(
                "N=12 example",
                xy=(330, 20),
                xytext=(309, 47),
                arrowprops={"arrowstyle": "->", "color": "white", "lw": 0.8},
                color="white",
                fontsize=8,
            )
        axis.grid(False)

    legend = [Patch(facecolor=color, edgecolor="none", label=f"N={depth}")
              for depth, color in zip(depth_values, depth_colors)]
    axes[1].legend(handles=legend, title="Recommended depth", loc="lower left", ncol=2,
                   fontsize=8, title_fontsize=8, frameon=True)

    penalty_cmap = plt.get_cmap("viridis_r")
    last_mesh = None
    for axis, example, panel_letter in zip(axes[2:], examples, ("C", "D")):
        subset = [row for row in contour_rows if row["example"] == example["example"]]
        taus = np.asarray(sorted({row["tau_us"] for row in subset}), dtype=float)
        depths = np.asarray(sorted({row["N"] for row in subset}), dtype=int)
        by_coordinate = {(row["tau_us"], row["N"]): row for row in subset}
        penalties = np.asarray(
            [[by_coordinate[(tau, int(depth))]["penalty_percent"] for tau in taus]
             for depth in depths],
            dtype=float,
        )
        clipped = np.minimum(penalties, 50.0)
        depth_positions = np.arange(depths.size, dtype=float)
        last_mesh = axis.pcolormesh(
            categorical_edges(taus),
            np.arange(-0.5, depths.size + 0.5, 1.0),
            clipped,
            cmap=penalty_cmap,
            vmin=0.0,
            vmax=50.0,
            shading="flat",
            rasterized=True,
        )
        mask = (penalties <= 10.0).astype(float)
        axis.contour(taus, depth_positions, mask, levels=[0.5], colors="#00ffff", linewidths=2.0)
        best_position = int(np.flatnonzero(depths == example["best_N"])[0])
        axis.scatter([1.25], [best_position], marker="*", s=105, facecolor="white",
                     edgecolor="black", zorder=5)
        tick_depths = [value for value in (8, 12, 16, 20, 28, 40, 48) if value in depths]
        tick_positions = [int(np.flatnonzero(depths == value)[0]) for value in tick_depths]
        axis.set_yticks(tick_positions, tick_depths)
        axis.set_xlabel(r"Model timing parameter $\tau$ ($\mu$s)")
        axis.set_ylabel("XY8 block count N")
        axis.set_title(
            f"{panel_letter}  {example['example']}: one connected 10% region",
            loc="left",
            fontsize=11,
            fontweight="bold",
        )
        axis.text(
            0.02,
            0.97,
            (
                f"{example['temperature_K']:.0f} K, "
                f"{example['amplitude_microstrain']:+.0f} µε\n"
                f"N={example['best_N']}; window {example['window_N_min']}–{example['window_N_max']}\n"
                f"β₀={example['beta_0_four']}; width {example['timing_width_ns']:.2f} ns"
            ),
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "0.75", "pad": 3},
        )
        axis.set_xlim(1.225, 1.275)

    colorbar_axis = fig.add_axes([0.36, 0.105, 0.28, 0.018])
    colorbar = fig.colorbar(last_mesh, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("Modeled sensitivity-proxy penalty (%) — lower is better")
    colorbar.set_ticks([0, 10, 25, 50], labels=["0", "10", "25", "≥50"])

    confirmation = topology_summary["confirmation"]
    fig.suptitle(
        "Environmental detuning deforms one persistent EDT region and changes control depth",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.01,
        (
            f"Fine-grid confirmation: {confirmation['confirmation_pass_count']}/"
            f"{confirmation['selected_surface_count']} selected surfaces passed; "
            "all retained β₀=1 under four- and eight-neighbour connectivity. "
            "Maps show the preregistered axial-strain/Jarmola operational slice."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.subplots_adjust(top=0.93, bottom=0.22)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def generate(
    processed_dir: str | Path = DEFAULT_PROCESSED,
    outdir: str | Path = DEFAULT_OUTDIR,
) -> tuple[dict[str, Path], dict]:
    processed = Path(processed_dir)
    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)

    score_path = processed / SCORE_SUMMARY
    selection_path = processed / CONFIRMATION_SELECTION
    results_path = processed / CONFIRMATION_RESULTS
    surfaces_path = processed / CONFIRMATION_SURFACES
    topology_path = processed / TOPOLOGY_SUMMARY
    score_rows = _read_csv(score_path)
    selection_rows = _read_csv(selection_path)
    confirmation_rows = _read_csv(results_path)
    topology_summary = json.loads(topology_path.read_text(encoding="utf-8"))

    depth_rows = build_depth_map_rows(score_rows)
    with np.load(surfaces_path) as archive:
        contour_rows, examples = build_contour_rows(
            selection_rows, confirmation_rows, archive
        )

    paths = {
        "png": destination / f"{FIGURE_BASENAME}.png",
        "pdf": destination / f"{FIGURE_BASENAME}.pdf",
        "depth_source": destination / DEPTH_SOURCE,
        "contour_source": destination / CONTOUR_SOURCE,
        "summary": destination / FIGURE_SUMMARY,
    }
    _write_csv(paths["depth_source"], depth_rows)
    _write_csv(paths["contour_source"], contour_rows)
    _plot(depth_rows, contour_rows, examples, topology_summary, paths["png"], paths["pdf"])

    fixed_depths = sorted({row["best_N"] for row in depth_rows
                           if row["carrier_policy"] == "fixed_reference_carrier"})
    retracked_depths = sorted({row["best_N"] for row in depth_rows
                               if row["carrier_policy"] == "ideal_retracked_carrier"})
    summary = {
        "schema_version": "1.0",
        "stage": "stage6_environment_topology_communication",
        "figure_title": (
            "Environmental detuning deforms one persistent EDT region and changes control depth"
        ),
        "frozen_slice": {
            "strain_path": STRAIN_PATH,
            "coherence_variant": COHERENCE_VARIANT,
            "fixed_carrier_best_depths": fixed_depths,
            "retracked_carrier_best_depths": retracked_depths,
        },
        "examples": examples,
        "confirmed_result": {
            "discovery_surface_count": topology_summary["discovery"]["surface_count"],
            "selected_fine_grid_surface_count": topology_summary["confirmation"]["selected_surface_count"],
            "confirmation_pass_count": topology_summary["confirmation"]["confirmation_pass_count"],
            "confirmation_failure_count": topology_summary["confirmation"]["confirmation_failure_count"],
            "fine_beta_0_distribution": topology_summary["confirmation"]["fine_primary_beta_0_four_distribution"],
            "four_eight_disagreement_count": topology_summary["confirmation"]["fine_four_eight_disagreement_count"],
        },
        "interpretation": (
            "Within the frozen environmental range, temperature and strain do not split the "
            "near-optimal EDT set into additional resonant components. Fixed-carrier detuning "
            "deforms the same connected region and can move the recommended depth from N=20 "
            "to N=12, whereas ideal carrier retracking preserves N=20 in the displayed slice."
        ),
        "source_artifacts": {
            path.name: {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
            for path in (score_path, selection_path, results_path, surfaces_path, topology_path)
        },
    }
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for key in ("depth_source", "contour_source", "summary"):
        shutil.copy2(paths[key], processed / paths[key].name)
    return paths, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    paths, summary = generate(args.processed_dir, args.outdir)
    for path in paths.values():
        print(f"Wrote {path}")
    confirmed = summary["confirmed_result"]
    print(
        f"confirmation={confirmed['confirmation_pass_count']}/"
        f"{confirmed['selected_fine_grid_surface_count']}, "
        f"beta0={confirmed['fine_beta_0_distribution']}"
    )


if __name__ == "__main__":
    main()
