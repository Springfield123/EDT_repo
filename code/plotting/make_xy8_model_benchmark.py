#!/usr/bin/env python3
"""Benchmark nested NV depth models against the frozen Pham holdout curve."""

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


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_metric(value, digits=3) -> str:
    if value == "" or value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def make_figure(result: dict, outdir: Path) -> None:
    rows = result["rows"]
    summary = result["summary"]
    by_id = {row["method_id"]: row for row in rows}

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8.7,
            "figure.dpi": 150,
        }
    )
    fig, (curve_ax, table_ax) = plt.subplots(
        1,
        2,
        figsize=(12.6, 5.15),
        gridspec_kw={"width_ratios": [1.45, 1.15]},
    )
    fig.suptitle(summary["figure_title"], fontsize=15, fontweight="bold", y=0.975)

    experimental_window = (
        summary["observed_empirical_10pct_window_min_pulses"],
        summary["observed_empirical_10pct_window_max_pulses"],
    )
    curve_ax.axvspan(
        *experimental_window,
        color="#d9d9d9",
        alpha=0.38,
        label="Experimental 10% window",
    )
    styles = {
        "coherence_only": {"color": "#3a923a", "linestyle": "--", "linewidth": 2.0},
        "coherence_plus_pulse_fidelity": {
            "color": "#e07a1f",
            "linestyle": "-.",
            "linewidth": 2.0,
        },
        "full_resource_score": {"color": "#1f4e79", "linestyle": "-", "linewidth": 2.5},
    }
    for method_id in [
        "coherence_only",
        "coherence_plus_pulse_fidelity",
        "full_resource_score",
    ]:
        curve_ax.plot(
            result["dense_pulses"],
            result["curves"][method_id],
            label=by_id[method_id]["method"],
            **styles[method_id],
        )
    curve_ax.axvline(
        summary["fixed_xy8_N16_total_pi_pulses"],
        color="#7b4ab5",
        linestyle=":",
        linewidth=2.2,
        label="Fixed XY8 N=16 (128 pulses)",
    )
    curve_ax.errorbar(
        result["observed_pulses"],
        result["observed_relative"],
        yerr=result["observed_relative_error"],
        fmt="o",
        markersize=5.5,
        markerfacecolor="white",
        markeredgewidth=1.6,
        color="#b33b2e",
        ecolor="#b33b2e",
        capsize=2.8,
        label="Experiment",
        zorder=5,
    )
    curve_ax.annotate(
        "Observed minimum\n106 pulses",
        xy=(106, 1.0),
        xytext=(145, 1.27),
        arrowprops={"arrowstyle": "->", "color": "#8f2d22"},
        color="#8f2d22",
        ha="center",
    )
    curve_ax.set_xlim(15, 300)
    curve_ax.set_ylim(0.89, 1.76)
    curve_ax.set_xlabel("Total control pulses, n")
    curve_ax.set_ylabel("Sensitivity relative to each curve's minimum\n(lower is better)")
    curve_ax.set_title("A  Nested model comparison")
    curve_ax.grid(axis="y", alpha=0.22)
    curve_ax.legend(loc="upper right", framealpha=0.94)

    table_ax.axis("off")
    table_ax.set_title("B  Decision and shape metrics", pad=12)
    row_order = [
        "coherence_only",
        "coherence_plus_pulse_fidelity",
        "full_resource_score",
        "fixed_xy8_N16",
        "experimental_reference",
    ]
    table_rows = [by_id[method_id] for method_id in row_order]
    cell_text = []
    for row in table_rows:
        if row["method_type"] == "fixed_decision":
            inside = "N/A"
        elif row["method_type"] == "reference_curve":
            inside = "Reference"
        else:
            inside = "Yes" if row["observed_optimum_inside_10pct_window"] else "No"
        cell_text.append(
            [
                str(row["predicted_optimum_total_pi_pulses"]),
                str(row["absolute_error_vs_observed_optimum_pulses"]),
                inside,
                _format_metric(row["normalized_rmse"]),
                _format_metric(row["spearman_rank_correlation"]),
            ]
        )
    row_labels = [
        "Coherence only",
        "Coherence + Fπ",
        "Full resource score",
        "Fixed XY8 N=16",
        "Experiment",
    ]
    column_labels = ["Optimum\n(pulses)", "Error\n(pulses)", "Observed optimum\nin 10% window", "Norm.\nRMSE", "Rank\nρ"]
    table = table_ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=column_labels,
        cellLoc="center",
        rowLoc="left",
        loc="upper center",
        bbox=[0.02, 0.40, 0.96, 0.53],
        colWidths=[0.15, 0.14, 0.27, 0.14, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        if row_index == 0:
            cell.set_facecolor("#d9e8f5")
            cell.set_text_props(weight="bold")
        elif row_index == 3:
            cell.set_facecolor("#e8f1f8")
        elif row_index == 5:
            cell.set_facecolor("#f2f2f2")

    table_ax.text(
        0.02,
        0.32,
        "What improves",
        transform=table_ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color="#1f4e79",
        va="top",
    )
    table_ax.text(
        0.02,
        0.255,
        "Adding pulse fidelity moves the prediction from 264 pulses into the measured finite region.",
        transform=table_ax.transAxes,
        fontsize=9.2,
        wrap=True,
        va="top",
    )
    table_ax.text(
        0.02,
        0.13,
        "What remains",
        transform=table_ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color="#8f2d22",
        va="top",
    )
    table_ax.text(
        0.02,
        0.065,
        "Coherence alone has lower RMSE, and fixed N=16 is only 0.55% above the interpolated measured minimum.",
        transform=table_ax.transAxes,
        fontsize=9.2,
        wrap=True,
        va="top",
    )

    fig.text(
        0.5,
        0.025,
        "Conclusion: pulse fidelity is required to recover a finite optimum; stronger data are needed to distinguish the full score from simpler choices.",
        ha="center",
        fontsize=10.2,
        color="#333333",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.16, wspace=0.25)
    fig.savefig(outdir / "fig_xy8_literature_model_benchmark.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "fig_xy8_literature_model_benchmark.pdf", bbox_inches="tight")
    plt.close(fig)


def run(manifest_path: Path, data_path: Path, outdir: Path) -> dict:
    manifest = validation.load_manifest(manifest_path)
    measurements = validation.load_measurements(data_path)
    result = validation.benchmark_model_alternatives(manifest, measurements)
    outdir.mkdir(parents=True, exist_ok=True)

    result["summary"]["input_digests"] = {
        "manifest_sha256": validation.file_sha256(manifest_path),
        "measurement_csv_sha256": validation.file_sha256(data_path),
    }
    result["summary"]["source"] = manifest["source"]
    result["summary"]["no_fit_statement"] = manifest["predeclared_baseline_assumptions"][
        "no_fit_statement"
    ]
    write_rows(outdir / "xy8_literature_model_benchmark.csv", result["rows"])
    write_rows(outdir / "xy8_literature_model_benchmark_curves.csv", result["curve_rows"])
    with (outdir / "xy8_literature_model_benchmark_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result["summary"], handle, indent=2)
        handle.write("\n")
    make_figure(result, outdir)
    return result["summary"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args.manifest, args.data, args.outdir)
    print(
        "Model benchmark:",
        summary["assessment"],
        f"best optimum method={summary['best_predictive_optimum_method']},",
        f"error={summary['best_predictive_optimum_error_pulses']} pulses",
    )


if __name__ == "__main__":
    main()
