#!/usr/bin/env python3
"""Compare scalar and coherent-control models with a frozen XY8 dataset."""

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

import xy8_specific_literature_validation as validation


DEFAULT_MANIFEST = (
    ROOT
    / "data"
    / "raw"
    / "nv"
    / "literature_validation"
    / "farfurnik_2015_xy8_manifest.json"
)
DEFAULT_DATA = (
    ROOT
    / "data"
    / "raw"
    / "nv"
    / "literature_validation"
    / "farfurnik_2015_fig3b_xy8.csv"
)
DEFAULT_OUTDIR = ROOT / "outputs" / "reproduced_figures"


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(result: dict, outdir: Path) -> None:
    metrics = result["metrics"]
    pulses = result["pulses"]

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 150,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), gridspec_kw={"width_ratios": [1.35, 1.0]})
    fig.suptitle(metrics["figure_title"], fontsize=15, fontweight="bold", y=0.98)

    ax = axes[0]
    ax.fill_between(
        result["dense_pulses"],
        result["dense_low"],
        result["dense_high"],
        color="#9ecae1",
        alpha=0.48,
        linewidth=0,
        label="Scalar uncertainty",
    )
    ax.plot(
        result["dense_pulses"],
        result["dense_baseline"],
        color="#1f4e79",
        linewidth=2.2,
        label="Scalar model (not fitted)",
    )
    ax.fill_between(
        result["dense_pulses"],
        result["coherent_dense_low"],
        result["coherent_dense_high"],
        color="#fdbb84",
        alpha=0.38,
        linewidth=0,
        label="Coherent parameter range",
    )
    ax.plot(
        result["dense_pulses"],
        result["coherent_dense_baseline"],
        color="#d95f0e",
        linewidth=2.2,
        label="Coherent XY8 model (not fitted)",
    )
    ax.errorbar(
        pulses,
        result["observed"],
        yerr=np.vstack([result["error_lower"], result["error_upper"]]),
        fmt="*",
        markersize=12,
        color="#d7301f",
        ecolor="#d7301f",
        capsize=3,
        linewidth=1.3,
        label="Farfurnik et al., XY8 experiment",
        zorder=5,
    )
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlim(6, 1050)
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel("Total π pulses (8N)")
    ax.set_ylabel("Relative fluorescence contrast")
    ax.set_title("A  Frozen predictions compared with experiment")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper right", frameon=True)

    residual_ax = axes[1]
    positions = np.arange(pulses.size)
    width = 0.34
    residual_ax.axhline(0.0, color="#666666", linewidth=1.0)
    residual_ax.bar(
        positions - width / 2.0,
        result["predicted"] - result["observed"],
        width,
        color="#1f4e79",
        label="Scalar",
    )
    residual_ax.bar(
        positions + width / 2.0,
        result["coherent_predicted"] - result["observed"],
        width,
        color="#d95f0e",
        label="Coherent XY8",
    )
    residual_ax.set_xticks(positions, [str(value) for value in pulses])
    residual_ax.set_xlabel("Total π pulses (8N)")
    residual_ax.set_ylabel("Prediction − measurement")
    residual_ax.set_title("B  Residuals reveal unmodelled high-depth loss")
    residual_ax.grid(axis="y", alpha=0.22)
    residual_ax.legend(loc="upper left")
    residual_ax.text(
        0.97,
        0.05,
        (
            f"Normalized RMSE\n"
            f"Scalar: {metrics['normalized_rmse_by_measured_span']:.3f}\n"
            f"Coherent: {metrics['coherent_model']['normalized_rmse_by_measured_span']:.3f}"
        ),
        transform=residual_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9, "edgecolor": "#bbbbbb"},
    )

    fig.text(
        0.5,
        0.015,
        (
            f"No fitting • both rank correlations ρ = 1.00 • uncertainty coverage: "
            f"scalar {metrics['points_overlapping_predeclared_model_range']}/3, "
            f"coherent {metrics['coherent_model']['points_overlapping_predeclared_model_range']}/3"
        ),
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#333333",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.17, wspace=0.26)
    fig.savefig(outdir / "fig_xy8_specific_literature_validation.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "fig_xy8_specific_literature_validation.pdf", bbox_inches="tight")
    plt.close(fig)


def run(manifest_path: Path, data_path: Path, outdir: Path) -> dict:
    manifest = validation.load_manifest(manifest_path)
    measurements = validation.load_measurements(data_path)
    result = validation.evaluate_holdout(manifest, measurements)
    outdir.mkdir(parents=True, exist_ok=True)

    result["metrics"]["input_digests"] = {
        "manifest_sha256": validation.file_sha256(manifest_path),
        "measurements_sha256": validation.file_sha256(data_path),
    }
    write_rows(outdir / "xy8_specific_literature_validation_curve.csv", result["curve_rows"])
    with (outdir / "xy8_specific_literature_validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result["metrics"], handle, indent=2)
        handle.write("\n")
    make_figure(result, outdir)
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
        "XY8-specific holdout:",
        metrics["validation_status"],
        f"scalar_NRMSE={metrics['normalized_rmse_by_measured_span']:.3f}",
        f"coherent_NRMSE={metrics['coherent_model']['normalized_rmse_by_measured_span']:.3f}",
        f"winner={metrics['model_comparison']['lower_normalized_rmse_model']}",
    )


if __name__ == "__main__":
    main()
