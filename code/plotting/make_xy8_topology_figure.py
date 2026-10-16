#!/usr/bin/env python3
"""Visualize persistent connected regions in the nominal XY8 EDT landscape."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / "code", ROOT / "code" / "analysis", ROOT / "code" / "plotting"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import numpy as np

import analyze_xy8_resonant_topology as stage2
import analyze_xy8_topology_persistence as stage4


DEFAULT_CALIBRATION = ROOT / "data" / "raw" / "nv" / "xy8_scenario_nominal.json"
DEFAULT_CANONICAL = ROOT / "data" / "processed" / "xy8_qre_nv_results.csv"
DEFAULT_REFINEMENT_SUMMARY = ROOT / "data" / "processed" / "xy8_topology_refinement_summary.json"
DEFAULT_PERSISTENCE_SUMMARY = ROOT / "data" / "processed" / "xy8_topology_persistence_summary.json"
DEFAULT_THRESHOLD_CURVE = ROOT / "data" / "processed" / "xy8_topology_threshold_curve.csv"

FIGURE_BASENAME = "fig_xy8_topology_persistence"
NESTED_SOURCE_FILENAME = "xy8_topology_nested_region_source.csv"
FIGURE_SUMMARY_FILENAME = "xy8_topology_figure_summary.json"
DISPLAY_THRESHOLDS = (1, 5, 10, 20, 50, 100)
CHECKPOINTS = (1, 5, 10, 20, 50, 100)


def load_threshold_curve(path: str | Path) -> list[dict]:
    numeric_float = {
        "threshold_percent",
        "relative_limit",
        "accepted_grid_fraction",
        "primary_tau_min_us",
        "primary_tau_max_us",
        "primary_timing_width_estimate_us",
        "primary_timing_width_half_interval_us",
    }
    numeric_int = {
        "beta_0_four_neighbour",
        "beta_0_eight_neighbour",
        "accepted_point_count",
        "primary_N_min",
        "primary_N_max",
        "primary_N_span_blocks",
    }
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows = []
    for raw in raw_rows:
        row = dict(raw)
        for key in numeric_float:
            row[key] = float(row[key])
        for key in numeric_int:
            row[key] = int(row[key])
        rows.append(row)
    if not rows:
        raise ValueError("The threshold curve is empty.")
    return rows


def categorical_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or np.any(np.diff(values) <= 0.0):
        raise ValueError("At least two strictly increasing coordinates are required.")
    midpoints = 0.5 * (values[:-1] + values[1:])
    return np.concatenate(
        ([values[0] - (midpoints[0] - values[0])], midpoints, [values[-1] + (values[-1] - midpoints[-1])])
    )


def nested_region_rows(surface, curve: list[dict]) -> tuple[list[dict], dict]:
    last = curve[-1]
    full_width = last["primary_tau_max_us"] - last["primary_tau_min_us"]
    margin = max(0.005, 0.12 * full_width)
    tau_lower = last["primary_tau_min_us"] - margin
    tau_upper = last["primary_tau_max_us"] + margin
    tau_mask = (surface.taus_s * 1e6 >= tau_lower) & (surface.taus_s * 1e6 <= tau_upper)
    depth_mask = surface.depths <= max(80, last["primary_N_max"])
    tau_indices = np.flatnonzero(tau_mask)
    depth_indices = np.flatnonzero(depth_mask)
    rows = []
    for tau_index in tau_indices:
        for depth_index in depth_indices:
            relative = float(surface.eta[tau_index, depth_index] / surface.eta_min)
            penalty = 100.0 * (relative - 1.0)
            eligible = [threshold for threshold in DISPLAY_THRESHOLDS if penalty <= threshold]
            rows.append(
                {
                    "tau_s": float(surface.taus_s[tau_index]),
                    "tau_us": float(surface.taus_s[tau_index] * 1e6),
                    "N": int(surface.depths[depth_index]),
                    "eta_proxy": float(surface.eta[tau_index, depth_index]),
                    "relative_to_best": relative,
                    "penalty_percent": penalty,
                    "lowest_display_threshold_percent": min(eligible) if eligible else "",
                }
            )
    crop = {
        "tau_min_us": float(surface.taus_s[tau_indices[0]] * 1e6),
        "tau_max_us": float(surface.taus_s[tau_indices[-1]] * 1e6),
        "N_min": int(surface.depths[depth_indices[0]]),
        "N_max": int(surface.depths[depth_indices[-1]]),
        "tau_count": int(tau_indices.size),
        "N_count": int(depth_indices.size),
    }
    return rows, crop


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path.name}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_figure(
    surface,
    curve: list[dict],
    persistence_summary: dict,
    crop: dict,
    output_png: Path,
    output_pdf: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    band_colors = ["#173f5f", "#20639b", "#3caea3", "#73b99a", "#edc949", "#e9844a"]
    band_labels = ["<=1%", "1-5%", "5-10%", "10-20%", "20-50%", "50-100%"]
    band_boundaries = [0, 1, 5, 10, 20, 50, 100]
    cmap = ListedColormap(band_colors)
    cmap.set_bad("#eeeeee")
    norm = BoundaryNorm(band_boundaries, cmap.N)

    tau_mask = (surface.taus_s * 1e6 >= crop["tau_min_us"]) & (
        surface.taus_s * 1e6 <= crop["tau_max_us"]
    )
    depth_mask = (surface.depths >= crop["N_min"]) & (surface.depths <= crop["N_max"])
    taus_us = surface.taus_s[tau_mask] * 1e6
    depths = surface.depths[depth_mask]
    scores = surface.eta[np.ix_(tau_mask, depth_mask)]
    penalties = 100.0 * (scores / surface.eta_min - 1.0)
    displayed_penalties = np.ma.masked_where(penalties > 100.0, penalties)
    depth_positions = np.arange(depths.size, dtype=float)

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1,
        3,
        figsize=(13.5, 4.8),
        gridspec_kw={"width_ratios": [1.5, 0.75, 1.15]},
    )
    ax_a.set_facecolor("#eeeeee")
    ax_a.pcolormesh(
        categorical_edges(taus_us),
        np.arange(-0.5, depths.size + 0.5, 1.0),
        displayed_penalties.T,
        cmap=cmap,
        norm=norm,
        shading="flat",
        rasterized=True,
    )
    for threshold in DISPLAY_THRESHOLDS:
        mask = (penalties <= threshold).T.astype(float)
        if np.any(mask == 1.0) and np.any(mask == 0.0):
            ax_a.contour(
                taus_us,
                depth_positions,
                mask,
                levels=[0.5],
                colors="#202020",
                linewidths=0.45 if threshold != 10 else 1.2,
                alpha=0.45 if threshold != 10 else 0.95,
            )
    best_depth_position = int(np.flatnonzero(depths == 20)[0])
    practical_depth_position = int(np.flatnonzero(depths == 16)[0])
    ax_a.scatter(
        [1.25],
        [best_depth_position],
        marker="*",
        s=90,
        facecolor="white",
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
        label="Best: N=20",
    )
    ax_a.scatter(
        [1.25],
        [practical_depth_position],
        marker="o",
        s=32,
        facecolor="none",
        edgecolor="white",
        linewidth=1.2,
        zorder=5,
        label="Practical: N=16",
    )
    requested_ticks = [1, 8, 16, 32, 64]
    tick_depths = [value for value in requested_ticks if value in set(depths.tolist())]
    tick_positions = [int(np.flatnonzero(depths == value)[0]) for value in tick_depths]
    ax_a.set_yticks(tick_positions, labels=tick_depths)
    ax_a.set_xlabel(r"Half-spacing $\tau$ ($\mu$s)")
    ax_a.set_ylabel("XY8 block count N (sampled categories)")
    ax_a.set_title("Nested EDT operating regions")
    marker_legend = ax_a.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="none",
                markersize=9,
                markerfacecolor="white",
                markeredgecolor="black",
                label="Optimum N=20",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markersize=6,
                markerfacecolor="none",
                markeredgecolor="black",
                label="Practical N=16",
            ),
        ],
        frameon=False,
        fontsize=7.5,
        loc="upper left",
    )
    ax_a.add_artist(marker_legend)
    ax_a.legend(
        handles=[Patch(facecolor=color, label=label) for color, label in zip(band_colors, band_labels)],
        title="Penalty band",
        frameon=False,
        fontsize=7,
        title_fontsize=8,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
    )
    ax_a.text(0.0, 1.04, "A", transform=ax_a.transAxes, fontsize=14, fontweight="bold")

    thresholds = np.array([row["threshold_percent"] for row in curve], dtype=float)
    beta0 = np.array([row["beta_0_four_neighbour"] for row in curve], dtype=float)
    checkpoint_mask = np.isin(thresholds, CHECKPOINTS)
    ax_b.step(thresholds, beta0, where="post", color="#20639b", linewidth=2.2)
    ax_b.scatter(thresholds[checkpoint_mask], beta0[checkpoint_mask], color="#20639b", s=22, zorder=3)
    ax_b.axvline(10, color="#555555", linestyle="--", linewidth=0.9)
    ax_b.text(12, 1.72, "10% EDT\nwindow", fontsize=8, va="center")
    ax_b.text(50, 1.16, r"$\beta_0=1$ throughout", ha="center", fontsize=9, fontweight="bold")
    ax_b.text(
        50,
        0.68,
        "No births or mergers\n3.75 µs requires 134%",
        ha="center",
        fontsize=8,
    )
    ax_b.set_xlim(1, 100)
    ax_b.set_ylim(0.5, 2.05)
    ax_b.set_yticks([1, 2])
    ax_b.set_xlabel("Allowed performance penalty (%)")
    ax_b.set_ylabel(r"Connected regions, $\beta_0$")
    ax_b.set_title("One region persists")
    ax_b.grid(axis="x", alpha=0.18)
    ax_b.text(-0.18, 1.04, "B", transform=ax_b.transAxes, fontsize=14, fontweight="bold")

    widths_ns = np.array(
        [row["primary_timing_width_estimate_us"] * 1e3 for row in curve], dtype=float
    )
    width_error_ns = np.array(
        [row["primary_timing_width_half_interval_us"] * 1e3 for row in curve], dtype=float
    )
    n_min = np.array([row["primary_N_min"] for row in curve], dtype=float)
    n_max = np.array([row["primary_N_max"] for row in curve], dtype=float)
    ax_c.fill_between(
        thresholds,
        widths_ns - width_error_ns,
        widths_ns + width_error_ns,
        color="#20639b",
        alpha=0.18,
        linewidth=0,
    )
    width_line = ax_c.plot(
        thresholds,
        widths_ns,
        color="#20639b",
        linewidth=2.0,
        label="Timing width",
    )[0]
    ax_c.scatter(thresholds[checkpoint_mask], widths_ns[checkpoint_mask], color="#20639b", s=18)
    ax_c.axvline(10, color="#555555", linestyle="--", linewidth=0.9)
    ax_c.set_xlim(1, 100)
    ax_c.set_ylim(bottom=0)
    ax_c.set_xlabel("Allowed performance penalty (%)")
    ax_c.set_ylabel("Timing width (ns)", color="#20639b")
    ax_c.tick_params(axis="y", colors="#20639b")
    ax_c.grid(axis="both", alpha=0.16)

    depth_axis = ax_c.twinx()
    depth_axis.fill_between(
        thresholds,
        n_min,
        n_max,
        step="post",
        color="#e9844a",
        alpha=0.18,
        label="Depth range",
    )
    lower_line = depth_axis.step(
        thresholds,
        n_min,
        where="post",
        color="#e9844a",
        linewidth=1.5,
        label="Depth limits",
    )[0]
    depth_axis.step(
        thresholds,
        n_max,
        where="post",
        color="#e9844a",
        linewidth=1.5,
    )
    depth_axis.set_ylim(0, max(n_max) * 1.12)
    depth_axis.set_ylabel("Accepted XY8 depth N", color="#c7632a")
    depth_axis.tick_params(axis="y", colors="#c7632a")
    ax_c.legend(
        [width_line, lower_line],
        ["Timing width", "Depth range"],
        frameon=False,
        fontsize=8,
        loc="upper left",
    )
    ten_percent = next(row for row in curve if row["threshold_percent"] == 10.0)
    ax_c.annotate(
        "10%: 9 ns\nN=10-34",
        xy=(10, ten_percent["primary_timing_width_estimate_us"] * 1e3),
        xytext=(26, 8),
        textcoords="data",
        fontsize=8,
        arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.8},
    )
    ax_c.set_title("Tolerance expands the same region")
    ax_c.text(-0.14, 1.04, "C", transform=ax_c.transAxes, fontsize=14, fontweight="bold")

    fig.suptitle(
        "One connected EDT region persists while its operating tolerance expands",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        0.005,
        "Nominal 200 kHz model; lower score is better. N is shown as sampled categorical settings; the 3.75 µs resonance remains outside the 100% set.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--refinement-summary", type=Path, default=DEFAULT_REFINEMENT_SUMMARY)
    parser.add_argument("--persistence-summary", type=Path, default=DEFAULT_PERSISTENCE_SUMMARY)
    parser.add_argument("--threshold-curve", type=Path, default=DEFAULT_THRESHOLD_CURVE)
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "reproduced_figures")
    parser.add_argument("--mirror-data-dir", type=Path)
    parser.add_argument("--frequency-khz", type=float, default=200.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    persistence_summary = json.loads(args.persistence_summary.read_text(encoding="utf-8"))
    if persistence_summary.get("status") != "threshold_persistence_passed":
        raise ValueError("The figure requires a passed Stage 4 persistence summary.")
    curve = load_threshold_curve(args.threshold_curve)
    surface, provenance = stage4.load_fine_surface(
        args.calibration,
        args.canonical,
        args.refinement_summary,
        target_frequency_hz=args.frequency_khz * 1e3,
    )
    nested_rows, crop = nested_region_rows(surface, curve)
    args.outdir.mkdir(parents=True, exist_ok=True)
    nested_path = args.outdir / NESTED_SOURCE_FILENAME
    write_csv(nested_path, nested_rows)
    plot_figure(
        surface,
        curve,
        persistence_summary,
        crop,
        args.outdir / f"{FIGURE_BASENAME}.png",
        args.outdir / f"{FIGURE_BASENAME}.pdf",
    )
    summary = {
        "schema_version": "1.0",
        "status": "topology_figure_generated",
        "figure_title": "One connected EDT region persists while its operating tolerance expands",
        "persistence_configuration_digest": persistence_summary["configuration_digest"],
        "threshold_curve_sha256": hashlib.sha256(args.threshold_curve.read_bytes()).hexdigest(),
        "nested_source_sha256": hashlib.sha256(nested_path.read_bytes()).hexdigest(),
        "display_thresholds_percent": list(DISPLAY_THRESHOLDS),
        "crop": crop,
        "provenance": provenance,
        "panel_messages": {
            "A": "Nested sampled-cell sublevel sets remain centered on the fundamental resonance.",
            "B": "beta_0 remains one from 1% through 100%; no births or mergers occur.",
            "C": "The same region expands from 2.5 to 62.75 ns and from N=16-20 to N=3-64.",
        },
    }
    summary_path = args.outdir / FIGURE_SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.mirror_data_dir is not None:
        args.mirror_data_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.mirror_data_dir / NESTED_SOURCE_FILENAME, nested_rows)
        mirrored_summary = dict(summary)
        mirrored_summary["nested_source_sha256"] = hashlib.sha256(
            (args.mirror_data_dir / NESTED_SOURCE_FILENAME).read_bytes()
        ).hexdigest()
        (args.mirror_data_dir / FIGURE_SUMMARY_FILENAME).write_text(
            json.dumps(mirrored_summary, indent=2) + "\n", encoding="utf-8"
        )
    print(f"Wrote {args.outdir / (FIGURE_BASENAME + '.png')}")
    print(f"Wrote {args.outdir / (FIGURE_BASENAME + '.pdf')}")
    print(f"Wrote {nested_path} and {summary_path}")


if __name__ == "__main__":
    main()
