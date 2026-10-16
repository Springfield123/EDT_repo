#!/usr/bin/env python3
"""Bootstrap connected-region structure in the frozen Pham depth curve."""

from __future__ import annotations

import argparse
import csv
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

import xy8_experimental_depth_topology as topology
import xy8_literature_validation as validation


DEFAULT_MANIFEST = (
    ROOT / "data" / "raw" / "nv" / "literature_validation" / "pham_2012_manifest.json"
)
DEFAULT_DATA = (
    ROOT / "data" / "raw" / "nv" / "literature_validation" / "pham_2012_fig3b.csv"
)
DEFAULT_OUTDIR = ROOT / "outputs" / "reproduced_figures"
NOMINAL_SCENARIO = ROOT / "data" / "raw" / "nv" / "xy8_scenario_nominal.json"

THRESHOLD_FILENAME = "xy8_pham_depth_topology_thresholds.csv"
DISTRIBUTION_FILENAME = "xy8_pham_depth_topology_distributions.csv"
SUMMARY_FILENAME = "xy8_pham_depth_topology_summary.json"
FIGURE_BASENAME = "fig_xy8_pham_depth_topology"


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(result: dict, outdir: Path) -> None:
    rows = result["threshold_rows"]
    plot_data = result["plot_data"]
    summary = result["summary"]
    thresholds = np.asarray([row["threshold_percent"] for row in rows])

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.2,
            "figure.dpi": 150,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.2), constrained_layout=True)
    fig.suptitle(
        "Independent depth data suggest one region, but uncertainty limits topology confidence",
        fontsize=14,
        fontweight="bold",
    )

    ax = axes[0, 0]
    observed_min = float(np.min(plot_data["observed"]))
    ax.plot(
        plot_data["dense_pulses"],
        plot_data["model_dense_relative"],
        color="#1f4e79",
        linewidth=2.0,
        label="Frozen model (no fit)",
    )
    ax.errorbar(
        plot_data["observed_pulses"],
        plot_data["observed_relative"],
        yerr=plot_data["observed_error"] / observed_min,
        fmt="o-",
        color="#b33b2e",
        markerfacecolor="white",
        markeredgewidth=1.4,
        capsize=2.5,
        linewidth=1.1,
        label="Pham central curve ± combined error",
        zorder=3,
    )
    ax.axhline(1.1, color="#666666", linestyle="--", linewidth=1.0, label="10% threshold")
    ax.set_xlabel("Total control pulses, n")
    ax.set_ylabel("Relative sensitivity (lower is better)")
    ax.set_xlim(20, 285)
    ax.set_ylim(0.9, 1.65)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    ax.set_title("A  Measured and predicted depth dependence")

    ax = axes[0, 1]
    probability = np.asarray(
        [row["bootstrap_probability_beta_0_equals_1"] for row in rows]
    )
    ax.plot(thresholds, probability, color="#6a3d9a", linewidth=2.3)
    ten = summary["ten_percent_comparison"]
    ax.scatter([10], [ten["bootstrap_probability_beta_0_equals_1"]], s=55, color="#d95f02", zorder=3)
    ax.annotate(
        f"10%: {100 * ten['bootstrap_probability_beta_0_equals_1']:.1f}%",
        xy=(10, ten["bootstrap_probability_beta_0_equals_1"]),
        xytext=(18, 0.43),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
    )
    ax.set_xlabel("Allowed penalty above minimum (%)")
    ax.set_ylabel(r"Bootstrap probability $P(\beta_0=1)$")
    ax.set_xlim(1, 100)
    ax.set_ylim(0, 1.04)
    ax.grid(alpha=0.2)
    ax.set_title("B  Confidence in one connected region")

    ax = axes[1, 0]
    lower_med = np.asarray([row["bootstrap_lower_boundary_median_pulses"] for row in rows])
    lower_lo = np.asarray([row["bootstrap_lower_boundary_q2_5_pulses"] for row in rows])
    lower_hi = np.asarray([row["bootstrap_lower_boundary_q97_5_pulses"] for row in rows])
    upper_med = np.asarray([row["bootstrap_upper_boundary_median_pulses"] for row in rows])
    upper_lo = np.asarray([row["bootstrap_upper_boundary_q2_5_pulses"] for row in rows])
    upper_hi = np.asarray([row["bootstrap_upper_boundary_q97_5_pulses"] for row in rows])
    ax.fill_between(thresholds, lower_lo, lower_hi, color="#4c78a8", alpha=0.18)
    ax.fill_between(thresholds, upper_lo, upper_hi, color="#e45756", alpha=0.18)
    ax.plot(thresholds, lower_med, color="#4c78a8", linewidth=2.0, label="Lower boundary, median")
    ax.plot(thresholds, upper_med, color="#e45756", linewidth=2.0, label="Upper boundary, median")
    ax.plot(
        thresholds,
        [row["model_observed_grid_lower_boundary_pulses"] for row in rows],
        color="#4c78a8",
        linestyle="--",
        linewidth=1.0,
        label="Frozen-model boundaries",
    )
    ax.plot(
        thresholds,
        [row["model_observed_grid_upper_boundary_pulses"] for row in rows],
        color="#e45756",
        linestyle="--",
        linewidth=1.0,
    )
    ax.set_xlabel("Allowed penalty above minimum (%)")
    ax.set_ylabel("Boundary (total pulses)")
    ax.set_xlim(1, 100)
    ax.set_ylim(15, 290)
    ax.grid(alpha=0.2)
    ax.legend(loc="center right")
    ax.set_title("C  Boundary distributions (95% intervals)")

    ax = axes[1, 1]
    width_med = np.asarray([row["bootstrap_window_width_median_pulses"] for row in rows])
    width_lo = np.asarray([row["bootstrap_window_width_q2_5_pulses"] for row in rows])
    width_hi = np.asarray([row["bootstrap_window_width_q97_5_pulses"] for row in rows])
    ax.fill_between(thresholds, width_lo, width_hi, color="#59a14f", alpha=0.22, label="Bootstrap 95% interval")
    ax.plot(thresholds, width_med, color="#2f7d32", linewidth=2.2, label="Bootstrap median")
    ax.step(
        thresholds,
        [row["central_window_width_pulses"] for row in rows],
        where="mid",
        color="#b33b2e",
        linewidth=1.5,
        label="Central measured curve",
    )
    ax.step(
        thresholds,
        [row["model_observed_grid_window_width_pulses"] for row in rows],
        where="mid",
        color="#1f4e79",
        linestyle="--",
        linewidth=1.5,
        label="Frozen model on sampled depths",
    )
    ax.set_xlabel("Allowed penalty above minimum (%)")
    ax.set_ylabel("Primary-region width (pulses)")
    ax.set_xlim(1, 100)
    ax.set_ylim(-5, 265)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    ax.set_title("D  Near-optimal window growth")

    fig.text(
        0.5,
        -0.015,
        (
            "One-dimensional adjacency joins consecutive published pulse counts. "
            "Parametric bootstrap; no parameters fitted to the holdout. "
            "This n-pulse XY dataset does not test two-dimensional XY8 timing topology."
        ),
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    fig.savefig(outdir / f"{FIGURE_BASENAME}.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"{FIGURE_BASENAME}.pdf", bbox_inches="tight")
    plt.close(fig)


def run(
    manifest_path: Path,
    data_path: Path,
    outdir: Path,
    *,
    mirror_data_dir: Path | None = None,
    trials: int = 100_000,
    seed: int = 12015686,
) -> dict:
    manifest = validation.load_manifest(manifest_path)
    measurements = validation.load_measurements(data_path)
    result = topology.analyze_pham_topology(
        manifest, measurements, trials=trials, seed=seed
    )
    summary = result["summary"]
    summary["source"] = manifest["source"]
    summary["independence_statement"] = manifest["independence_statement"]
    summary["input_digests"] = {
        "manifest_sha256": validation.file_sha256(manifest_path),
        "measurement_csv_sha256": validation.file_sha256(data_path),
        "nominal_scenario_sha256": validation.file_sha256(NOMINAL_SCENARIO),
    }

    outdir.mkdir(parents=True, exist_ok=True)
    write_rows(outdir / THRESHOLD_FILENAME, result["threshold_rows"])
    write_rows(outdir / DISTRIBUTION_FILENAME, result["distribution_rows"])
    with (outdir / SUMMARY_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    make_figure(result, outdir)

    if mirror_data_dir is not None:
        mirror_data_dir.mkdir(parents=True, exist_ok=True)
        for filename in (THRESHOLD_FILENAME, DISTRIBUTION_FILENAME, SUMMARY_FILENAME):
            shutil.copy2(outdir / filename, mirror_data_dir / filename)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--mirror-data-dir", type=Path)
    parser.add_argument("--trials", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=12015686)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(
        args.manifest,
        args.data,
        args.outdir,
        mirror_data_dir=args.mirror_data_dir,
        trials=args.trials,
        seed=args.seed,
    )
    ten = summary["ten_percent_comparison"]
    print(
        "Pham depth topology:",
        f"P(beta0=1 | 10%)={ten['bootstrap_probability_beta_0_equals_1']:.3f},",
        f"central window={ten['central_lower_boundary_pulses']}-"
        f"{ten['central_upper_boundary_pulses']} pulses,",
        f"persistent={summary['bootstrap_probability_beta_0_equals_1_at_every_threshold']:.3f}",
    )


if __name__ == "__main__":
    main()
