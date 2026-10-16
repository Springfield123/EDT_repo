#!/usr/bin/env python3
"""Generate the frozen DeVience XY8 sparse-surface topology artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "code" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import xy8_devience_topology as topology


DEFAULT_MANIFEST = (
    ROOT / "data" / "raw" / "nv" / "literature_validation" / "devience_2015_fig2b_manifest.json"
)
DEFAULT_DATA = (
    ROOT / "data" / "raw" / "nv" / "literature_validation" / "devience_2015_fig2b_xy8.csv"
)
DEFAULT_OUTDIR = ROOT / "outputs" / "reproduced_figures"
FILENAMES = {
    "surface": "xy8_devience_sparse_surface.csv",
    "thresholds": "xy8_devience_topology_thresholds.csv",
    "components": "xy8_devience_topology_components.csv",
    "summary": "xy8_devience_topology_summary.json",
    "figure": "fig_xy8_devience_topology",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path.name}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def surface_rows(result: dict) -> list[dict]:
    rows = []
    for depth_index, depth in enumerate(result["depths"]):
        for frequency_index, frequency in enumerate(result["frequency_grid_MHz"]):
            rows.append(
                {
                    "frequency_MHz": float(frequency),
                    "project_tau_us": float(0.25 / frequency),
                    "xy8_blocks": int(depth),
                    "total_pi_pulses": int(8 * depth),
                    "normalized_resonant_response": float(
                        result["normalized_response"][depth_index, frequency_index]
                    ),
                    "source": "linear interpolation of digitized DeVience Figure 2b markers",
                }
            )
    return rows


def make_figure(result: dict, measurements: list[dict], outdir: Path) -> None:
    depths = result["depths"]
    grid = result["frequency_grid_MHz"]
    response = result["normalized_response"]
    threshold_rows = result["threshold_rows"]
    summary = result["summary"]

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.0,
            "figure.dpi": 150,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.2), constrained_layout=True)
    fig.suptitle(
        "Published XY8 timing sweeps reveal depth-dependent resonant topology",
        fontsize=14,
        fontweight="bold",
    )

    colors = {10: "#3b4ba7", 20: "#267f39", 25: "#643fa2", 40: "#369c9a"}
    markers = {10: "s", 20: "^", 25: "d", 40: "o"}
    ax = axes[0, 0]
    for depth in depths:
        selected = [row for row in measurements if row["xy8_blocks"] == depth]
        frequency = [row["frequency_MHz"] for row in selected]
        contrast = [row["corrected_contrast"] for row in selected]
        ax.plot(
            frequency,
            contrast,
            marker=markers[int(depth)],
            color=colors[int(depth)],
            linewidth=1.0,
            markersize=4.8,
            label=f"XY8-{int(depth)} ({8 * int(depth)} pulses)",
        )
    ax.axhline(1.0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Detection frequency (MHz)")
    ax.set_ylabel("Offset-corrected normalized contrast")
    ax.set_xlim(0.82, 1.02)
    ax.set_ylim(0.30, 1.10)
    ax.grid(alpha=0.18)
    ax.legend(ncol=2, loc="lower left")
    ax.set_title("A  Vector-digitized experimental points")

    ax = axes[0, 1]
    depth_positions = np.arange(depths.size)
    mesh = ax.pcolormesh(
        grid,
        depth_positions,
        response,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        shading="nearest",
    )
    contours = ax.contour(
        grid,
        depth_positions,
        response,
        levels=[0.5, 0.6, 0.75, 0.9],
        colors=["#ffffff", "#f0e442", "#56b4e9", "#00e5ff"],
        linewidths=1.15,
    )
    ax.clabel(contours, fmt=lambda value: f"{100 * (1 - value):.0f}%", fontsize=7.5)
    ax.set_yticks(depth_positions, [f"N={int(depth)}" for depth in depths])
    ax.set_xlabel("Detection frequency (MHz)")
    ax.set_ylabel("XY8 depth")
    ax.set_xlim(grid[0], grid[-1])
    ax.set_title("B  Sparse response surface and penalty contours")
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_label("Response / global measured maximum")

    ax = axes[1, 0]
    interpreted = [row for row in threshold_rows if row["threshold_percent"] <= 60]
    threshold = np.asarray([row["threshold_percent"] for row in interpreted])
    beta = np.asarray([row["central_beta_0"] for row in interpreted])
    agreement = np.asarray(
        [row["bootstrap_probability_matches_central_digitization_only"] for row in interpreted]
    )
    ax.step(threshold, beta, where="post", color="#1f4e79", linewidth=2.2, label=r"Central $\beta_0$")
    ax.set_xlabel("Allowed loss from strongest response (%)")
    ax.set_ylabel(r"Connected regions, $\beta_0$")
    ax.set_xlim(1, 60)
    ax.set_ylim(0.7, max(beta) + 0.45)
    ax.set_yticks(range(1, int(max(beta)) + 1))
    ax.grid(alpha=0.2)
    twin = ax.twinx()
    twin.plot(threshold, agreement, color="#d95f02", linewidth=1.6, label="Digitization-only agreement")
    twin.set_ylabel("Probability of same count")
    twin.set_ylim(0, 1.04)
    lines = ax.get_lines() + twin.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="upper left")
    ax.set_title("C  Component-count persistence")

    ax = axes[1, 1]
    checkpoints = [10, 25, 40, 50]
    y_lookup = {value: index for index, value in enumerate(checkpoints)}
    for component in result["component_rows"]:
        threshold_value = int(component["threshold_percent"])
        if threshold_value not in y_lookup:
            continue
        identity = component["resonance_identity"]
        if identity.startswith("19F"):
            color = "#267f39"
        elif identity.startswith("1H"):
            color = "#3b4ba7"
        else:
            color = "#7a5195"
        y = y_lookup[threshold_value]
        ax.plot(
            [component["frequency_min_MHz"], component["frequency_max_MHz"]],
            [y, y],
            color=color,
            linewidth=8,
            solid_capstyle="butt",
        )
        ax.scatter(component["best_frequency_MHz"], y, color="white", edgecolor=color, s=28, zorder=3)
    ax.axvline(0.915, color="#777777", linestyle=":", linewidth=1.0)
    ax.set_yticks(range(len(checkpoints)), [f"{value}%" for value in checkpoints])
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(-0.6, len(checkpoints) - 0.4)
    ax.set_xlabel("Detection frequency (MHz)")
    ax.set_ylabel("Allowed loss")
    ax.grid(axis="x", alpha=0.2)
    ax.set_title("D  Resonance regions separate, then merge")
    ax.text(0.885, 3.45, "$^{19}$F-like", color="#267f39", ha="center")
    ax.text(0.95, 3.45, "$^1$H-like", color="#3b4ba7", ha="center")

    strongest = summary["strongest_measured_response"]
    fig.text(
        0.5,
        -0.012,
        (
            f"Strongest sampled response: {strongest['frequency_MHz']:.3f} MHz at XY8-{strongest['xy8_blocks']}. "
            "Offsets are removed exactly; one global measured normalization is used. "
            "No timing shift or model fitting. Error propagation covers digitization only."
        ),
        ha="center",
        fontsize=8.4,
        color="#444444",
    )
    outdir.mkdir(parents=True, exist_ok=True)
    basename = FILENAMES["figure"]
    fig.savefig(outdir / f"{basename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"{basename}.pdf", bbox_inches="tight")
    plt.close(fig)


def run(
    manifest_path: Path,
    data_path: Path,
    outdir: Path,
    *,
    mirror_data_dir: Path | None = None,
    bootstrap_trials: int = 10_000,
    seed: int = 2014313,
) -> dict:
    manifest = topology.load_manifest(manifest_path)
    measurements = topology.load_measurements(data_path, manifest)
    result = topology.analyze(
        manifest,
        measurements,
        bootstrap_trials=bootstrap_trials,
        seed=seed,
    )
    result["summary"]["source"] = manifest["source"]
    result["summary"]["input_digests"] = {
        "manifest_sha256": sha256(manifest_path),
        "digitized_measurements_sha256": sha256(data_path),
    }
    outdir.mkdir(parents=True, exist_ok=True)
    write_rows(outdir / FILENAMES["surface"], surface_rows(result))
    write_rows(outdir / FILENAMES["thresholds"], result["threshold_rows"])
    write_rows(outdir / FILENAMES["components"], result["component_rows"])
    (outdir / FILENAMES["summary"]).write_text(
        json.dumps(result["summary"], indent=2) + "\n", encoding="utf-8"
    )
    make_figure(result, measurements, outdir)
    if mirror_data_dir is not None:
        mirror_data_dir.mkdir(parents=True, exist_ok=True)
        for key in ("surface", "thresholds", "components", "summary"):
            shutil.copy2(outdir / FILENAMES[key], mirror_data_dir / FILENAMES[key])
    return result["summary"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--mirror-data-dir", type=Path)
    parser.add_argument("--bootstrap-trials", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2014313)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(
        args.manifest,
        args.data,
        args.outdir,
        mirror_data_dir=args.mirror_data_dir,
        bootstrap_trials=args.bootstrap_trials,
        seed=args.seed,
    )
    print(
        "DeVience topology:",
        summary["primary_range_1_to_60pct"]["central_events"],
        "strongest=",
        summary["strongest_measured_response"],
    )


if __name__ == "__main__":
    main()
