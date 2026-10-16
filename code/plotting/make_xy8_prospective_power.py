#!/usr/bin/env python3
"""Generate prospective power and sample-size artifacts for the XY8 protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "code" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import xy8_prospective_power as power_analysis


DEFAULT_PROTOCOL = ROOT / "outputs/reproduced_figures/xy8_prospective_validation_protocol.json"


PLOTTED_METRICS = (
    "timing_selectivity",
    "depth_controls",
    "operating_window",
    "n16_n20_equivalent_10pct",
    "exact_optimum",
)

COLORS = {
    "timing_selectivity": "#4c78a8",
    "depth_controls": "#e45756",
    "operating_window": "#54a24b",
    "n16_n20_equivalent_10pct": "#f58518",
    "exact_optimum": "#7a5195",
}

SHORT_LABELS = {
    "timing_selectivity": "Timing selectivity",
    "depth_controls": "Interior beats N=8 and 64",
    "operating_window": "Best lies in 10% window",
    "n16_n20_equivalent_10pct": "N=16 and 20 equivalent",
    "exact_optimum": "Exact N=20 optimum",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_power(rows: list[dict], summary: dict, output_png: Path, output_pdf: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.55, 1.0]}
    )
    for metric_id in PLOTTED_METRICS:
        selected = [row for row in rows if row["metric_id"] == metric_id]
        x = [row["independent_batches_per_control"] for row in selected]
        y = [row["power"] for row in selected]
        ax_a.plot(
            x,
            y,
            marker="o",
            markersize=3.2,
            linewidth=1.7,
            color=COLORS[metric_id],
            label=SHORT_LABELS[metric_id],
        )
    ax_a.axhline(summary["target_power"], color="black", linestyle="--", linewidth=1.0)
    ax_a.text(
        0.985,
        summary["target_power"] + 0.015,
        f"{summary['target_power']:.0%} target",
        transform=ax_a.get_yaxis_transform(),
        ha="right",
        fontsize=8,
    )
    ax_a.set_xscale("log")
    ax_a.set_ylim(0.0, 1.03)
    ax_a.set_xlabel("Independent one-second batches per control")
    ax_a.set_ylabel("Probability of reaching the stated conclusion")
    ax_a.set_title("Detection probability versus sample size")
    ax_a.grid(axis="both", alpha=0.18)
    ax_a.legend(frameon=False, fontsize=8, loc="lower right")
    ax_a.text(0.0, 1.04, "A", transform=ax_a.transAxes, ha="right", fontsize=14, fontweight="bold")

    required = summary["required_for_target_power"]
    labels = [SHORT_LABELS[metric_id] for metric_id in PLOTTED_METRICS]
    times = [required[metric_id]["metric_specific_acquisition_time_h"] for metric_id in PLOTTED_METRICS]
    maximum_resolved = max(value for value in times if value is not None)
    displayed = [maximum_resolved * 1.25 if value is None else value for value in times]
    y_positions = np.arange(len(labels))
    bars = ax_b.barh(
        y_positions,
        displayed,
        color=[COLORS[metric_id] for metric_id in PLOTTED_METRICS],
        alpha=0.9,
    )
    ax_b.set_yticks(y_positions)
    ax_b.set_yticklabels(labels, fontsize=8)
    ax_b.invert_yaxis()
    ax_b.set_xscale("log")
    ax_b.set_xlabel("Batch time for controls used by each test (hours)")
    ax_b.set_title("Time required for 90% decision power")
    ax_b.grid(axis="x", alpha=0.18)
    for bar, metric_id, value in zip(bars, PLOTTED_METRICS, times):
        count = required[metric_id]["required_independent_batches_per_control"]
        label = f">{max(summary['candidate_batch_grid']):,} batches" if value is None else f"{count:,} batches"
        ax_b.text(
            bar.get_width() * 1.04,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=8,
        )
    ax_b.text(-0.10, 1.04, "B", transform=ax_b.transAxes, ha="right", fontsize=14, fontweight="bold")

    fig.suptitle(
        "The operating window is easier to test than the exact optimum",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        0.005,
        "Prospective calculation under nominal predictions; one-second independent batches and Gaussian field-estimate noise assumed.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs/reproduced_figures")
    parser.add_argument("--trials", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--target-power", type=float, default=0.90)
    parser.add_argument("--batch-duration-s", type=float, default=1.0)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    protocol = power_analysis.load_protocol(arguments.protocol)
    rows, summary = power_analysis.build_power_curve(
        protocol,
        trials=arguments.trials,
        seed=arguments.seed,
        target_power=arguments.target_power,
        batch_duration_s=arguments.batch_duration_s,
    )
    arguments.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(arguments.outdir / "xy8_prospective_power_curve.csv", rows)
    (arguments.outdir / "xy8_prospective_power_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    plot_power(
        rows,
        summary,
        arguments.outdir / "fig_xy8_prospective_power.png",
        arguments.outdir / "fig_xy8_prospective_power.pdf",
    )
    print("Wrote prospective XY8 power analysis to", arguments.outdir)


if __name__ == "__main__":
    main()
