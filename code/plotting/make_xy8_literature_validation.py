#!/usr/bin/env python3
"""Validate the scalar NV control-depth model against a frozen holdout curve."""

from __future__ import annotations

import argparse
import csv
import json
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

import xy8_literature_validation as validation


DEFAULT_MANIFEST = (
    ROOT / "data" / "raw" / "nv" / "literature_validation" / "pham_2012_manifest.json"
)
DEFAULT_DATA = (
    ROOT / "data" / "raw" / "nv" / "literature_validation" / "pham_2012_fig3b.csv"
)
DEFAULT_OUTDIR = ROOT / "outputs" / "reproduced_figures"
NOMINAL_SCENARIO = ROOT / "data" / "raw" / "nv" / "xy8_scenario_nominal.json"


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(result: dict, manifest: dict, outdir: Path) -> None:
    metrics = result["metrics"]
    pulses = result["dense_pulses"]
    measured_pulses = result["observed_pulses"]
    measured = result["observed_relative"]
    measured_error = result["observed_relative_error"]
    model_at_data = np.interp(measured_pulses, pulses, result["dense_relative"])
    residual = measured - model_at_data

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 150,
        }
    )
    fig = plt.figure(figsize=(11.6, 5.2))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.6, 1.0],
        height_ratios=[1.0, 0.72],
        hspace=0.38,
        wspace=0.2,
    )
    ax = fig.add_subplot(grid[:, 0])
    residual_ax = fig.add_subplot(grid[0, 1])
    info_ax = fig.add_subplot(grid[1, 1])
    fig.suptitle(metrics["figure_title"], fontsize=15, fontweight="bold", y=0.975)

    ax.fill_between(
        pulses,
        result["uncertainty_lower"],
        result["uncertainty_upper"],
        color="#9ecae1",
        alpha=0.42,
        linewidth=0,
        label="95% range from unreported fidelity and overhead",
    )
    ax.plot(
        pulses,
        result["dense_relative"],
        color="#1f4e79",
        linewidth=2.2,
        label="Model (predeclared assumptions; no optimum fit)",
    )
    ax.axvspan(
        metrics["predicted_baseline_10pct_window_min_pulses"],
        metrics["predicted_baseline_10pct_window_max_pulses"],
        color="#f6c85f",
        alpha=0.2,
        label="Predicted 10% operating window",
    )
    ax.errorbar(
        measured_pulses,
        measured,
        yerr=measured_error,
        fmt="o",
        markersize=5.8,
        markerfacecolor="white",
        markeredgewidth=1.7,
        color="#b33b2e",
        ecolor="#b33b2e",
        capsize=3,
        linewidth=1.2,
        label="Pham et al. experiment (digitized Fig. 3b)",
        zorder=4,
    )
    ax.scatter(
        [metrics["predicted_baseline_optimum_total_pi_pulses"]],
        [1.0],
        marker="*",
        s=130,
        color="#1f4e79",
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
    )
    observed_index = int(np.argmin(measured))
    ax.annotate(
        f"Observed minimum: {int(measured_pulses[observed_index])} pulses",
        xy=(measured_pulses[observed_index], measured[observed_index]),
        xytext=(148, 1.32),
        arrowprops={"arrowstyle": "->", "color": "#7f2720"},
        color="#7f2720",
    )
    ax.annotate(
        f"Predicted minimum: {metrics['predicted_baseline_optimum_total_pi_pulses']} pulses",
        xy=(metrics["predicted_baseline_optimum_total_pi_pulses"], 1.0),
        xytext=(25, 1.42),
        arrowprops={"arrowstyle": "->", "color": "#1f4e79"},
        color="#1f4e79",
    )
    ax.set_xlim(15, 300)
    ax.set_ylim(0.88, max(1.65, float(np.max(measured + measured_error)) + 0.05))
    ax.set_xlabel("Total control pulses, n")
    ax.set_ylabel("Sensitivity relative to each curve's minimum\n(lower is better)")
    ax.set_title("A  Independent control-depth trend")
    ax.grid(axis="y", alpha=0.22)
    handles, labels = ax.get_legend_handles_labels()
    order = [2, 1, 0, 3]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper right")

    residual_ax.axhspan(-0.1, 0.1, color="#d9ead3", alpha=0.6, zorder=0)
    residual_ax.axhline(0.0, color="#444444", linewidth=1.0)
    residual_ax.errorbar(
        measured_pulses,
        residual,
        yerr=measured_error,
        fmt="o",
        markersize=5.2,
        color="#6a3d9a",
        ecolor="#6a3d9a",
        capsize=2.5,
    )
    residual_ax.set_xlim(15, 300)
    residual_ax.set_ylim(-0.32, 0.40)
    residual_ax.set_xlabel("Total control pulses, n")
    residual_ax.set_ylabel("Experiment − model (relative score)")
    residual_ax.set_title("B  Agreement and remaining mismatch")
    residual_ax.grid(axis="y", alpha=0.22)

    baseline = manifest["predeclared_baseline_assumptions"]
    reported = manifest["reported_parameters"]
    provenance_text = (
        "Reported by holdout paper\n"
        f"f = {manifest['source']['target_frequency_Hz'] / 1e3:.1f} kHz; "
        f"T₂ = {reported['hahn_echo_T2_s'] * 1e6:.0f} µs; "
        f"p = {reported['coherence_stretch_power_p']:.0f}; "
        f"s = {reported['coherence_depth_scaling_s']:.2f}\n\n"
        "Predeclared, not fitted\n"
        f"Fπ = {baseline['per_pulse_fidelity']:.3f}; "
        f"overhead = {baseline['preparation_readout_overhead_s'] * 1e6:.1f} µs\n\n"
        f"ρ = {metrics['spearman_rank_correlation']:.2f}; "
        f"normalized RMSE = {metrics['normalized_rmse']:.2f}\n"
        f"Observed minimum is inside {metrics['predicted_baseline_10pct_window_min_pulses']}–"
        f"{metrics['predicted_baseline_10pct_window_max_pulses']} pulse window"
    )
    info_ax.axis("off")
    info_ax.text(
        0.02,
        0.98,
        provenance_text,
        transform=info_ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#bbbbbb"},
    )

    fig.text(
        0.5,
        0.008,
        "The finite-depth trend transfers, but the simple pulse-error term understates the low-depth penalty and overstates part of the high-depth penalty.",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#333333",
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.84, bottom=0.17)
    fig.savefig(outdir / "fig_xy8_literature_validation.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "fig_xy8_literature_validation.pdf", bbox_inches="tight")
    plt.close(fig)


def run(manifest_path: Path, data_path: Path, outdir: Path) -> dict:
    manifest = validation.load_manifest(manifest_path)
    measurements = validation.load_measurements(data_path)
    result = validation.evaluate_holdout(manifest, measurements)
    outdir.mkdir(parents=True, exist_ok=True)

    result["metrics"]["input_digests"] = {
        "manifest_sha256": validation.file_sha256(manifest_path),
        "measurement_csv_sha256": validation.file_sha256(data_path),
        "nominal_scenario_sha256": validation.file_sha256(NOMINAL_SCENARIO),
    }
    result["metrics"]["source"] = manifest["source"]
    result["metrics"]["independence_statement"] = manifest["independence_statement"]
    result["metrics"]["no_fit_statement"] = manifest["predeclared_baseline_assumptions"][
        "no_fit_statement"
    ]

    write_rows(outdir / "xy8_literature_validation_curve.csv", result["curve_rows"])
    write_rows(outdir / "xy8_literature_validation_uncertainty.csv", result["uncertainty_rows"])
    with (outdir / "xy8_literature_validation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result["metrics"], handle, indent=2)
        handle.write("\n")
    make_figure(result, manifest, outdir)
    return result["metrics"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = run(args.manifest, args.data, args.outdir)
    print(
        "Literature validation:",
        metrics["assessment"],
        f"predicted={metrics['predicted_baseline_optimum_total_pi_pulses']} pulses,",
        f"observed={metrics['observed_central_value_optimum_total_pi_pulses']} pulses,",
        f"rho={metrics['spearman_rank_correlation']:.3f}",
    )


if __name__ == "__main__":
    main()
