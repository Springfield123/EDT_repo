#!/usr/bin/env python3
"""Compute local-significance metrics for the Na EDT window."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from wannier_geometry_pipeline import parse_agr


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def finite_mean(values: Any) -> float | None:
    arr = finite_array(values)
    return float(np.mean(arr)) if arr.size else None


def finite_std(values: Any) -> float | None:
    arr = finite_array(values)
    return float(np.std(arr, ddof=1)) if arr.size > 1 else None


def coefficient_of_variation(values: Any) -> float | None:
    arr = finite_array(values)
    if arr.size <= 1:
        return None
    mean = float(np.mean(arr))
    if abs(mean) < 1e-15:
        return None
    return float(np.std(arr, ddof=1) / abs(mean))


def contribution_fraction(values: np.ndarray, mask: np.ndarray) -> float | None:
    total = float(np.nansum(values))
    if total == 0.0:
        return None
    return float(np.nansum(values[mask]) / total)


def window_masks(k: np.ndarray, k_left: float, k_right: float) -> dict[str, np.ndarray]:
    width = k_right - k_left
    left_start = max(float(np.min(k)), k_left - width)
    right_end = min(float(np.max(k)), k_right + width)
    return {
        "inside": (k >= k_left) & (k <= k_right),
        "outside": (k < k_left) | (k > k_right),
        "left_neighbor": (k >= left_start) & (k < k_left),
        "right_neighbor": (k > k_right) & (k <= right_end),
    }


def interp_at(k: np.ndarray, y: np.ndarray, x0: float) -> float | None:
    finite = np.isfinite(k) & np.isfinite(y)
    if not np.any(finite):
        return None
    kk = k[finite]
    yy = y[finite]
    order = np.argsort(kk)
    kk = kk[order]
    yy = yy[order]
    if x0 < kk[0] or x0 > kk[-1]:
        return None
    return float(np.interp(x0, kk, yy))


def summarize_channel(k: np.ndarray, values: np.ndarray, masks: dict[str, np.ndarray], k0: float) -> dict[str, Any]:
    abs_values = np.abs(values)
    inside = masks["inside"]
    outside = masks["outside"]
    left = masks["left_neighbor"]
    right = masks["right_neighbor"]
    neighbor_values = np.concatenate([values[left], values[right]])
    neighbor_abs = np.concatenate([abs_values[left], abs_values[right]])
    inside_mean_abs = finite_mean(abs_values[inside])
    neighbor_mean_abs = finite_mean(neighbor_abs)
    return {
        "count_inside": int(np.sum(inside)),
        "count_outside": int(np.sum(outside)),
        "count_left_neighbor": int(np.sum(left)),
        "count_right_neighbor": int(np.sum(right)),
        "contribution_fraction_abs": contribution_fraction(abs_values, inside),
        "inside_mean": finite_mean(values[inside]),
        "outside_mean": finite_mean(values[outside]),
        "inside_mean_abs": inside_mean_abs,
        "outside_mean_abs": finite_mean(abs_values[outside]),
        "left_neighbor_mean_abs": finite_mean(abs_values[left]),
        "right_neighbor_mean_abs": finite_mean(abs_values[right]),
        "neighbor_mean_abs": neighbor_mean_abs,
        "local_background_contrast_abs": (
            inside_mean_abs / neighbor_mean_abs if inside_mean_abs is not None and neighbor_mean_abs not in (None, 0.0) else None
        ),
        "inside_std": finite_std(values[inside]),
        "outside_std": finite_std(values[outside]),
        "inside_abs_std": finite_std(abs_values[inside]),
        "outside_abs_std": finite_std(abs_values[outside]),
        "inside_abs_cv": coefficient_of_variation(abs_values[inside]),
        "outside_abs_cv": coefficient_of_variation(abs_values[outside]),
        "neighbor_mean": finite_mean(neighbor_values),
        "k0_value": interp_at(k, values, k0),
        "k0_abs_value": abs(interp_at(k, values, k0)) if interp_at(k, values, k0) is not None else None,
    }


def metric_by_k(metric_rows: list[dict[str, str]]) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray, np.ndarray]:
    grouped: dict[float, dict[int, float]] = {}
    for row in metric_rows:
        k_coord = float(row["k_coord"])
        band = int(row["band"])
        value = float(row["quantum_metric_parallel"])
        grouped.setdefault(k_coord, {})[band] = value
    k = np.array(sorted(grouped), dtype=float)
    bands = sorted({band for values in grouped.values() for band in values})
    by_band = {
        band: np.array([grouped[kk].get(band, math.nan) for kk in k], dtype=float)
        for band in bands
    }
    matrix = np.vstack([by_band[band] for band in bands])
    metric_sum = np.nansum(matrix, axis=0)
    metric_mean = np.nanmean(matrix, axis=0)
    return k, by_band, metric_sum, metric_mean


def epc_placeholder(analysis_dir: Path, masks: dict[str, np.ndarray], metric_k: np.ndarray) -> dict[str, Any]:
    epc_path = analysis_dir / "epc_extracted.csv"
    if not epc_path.exists():
        return {"status": "missing_epc_extracted_csv", "F_EPC": None, "R_EPC_local": None}
    rows = read_csv(epc_path)
    values = []
    coords = []
    for row in rows:
        value = row.get("epc_g2", "")
        coord = row.get("k_coord", "")
        if value and coord:
            try:
                values.append(float(value))
                coords.append(float(coord))
            except ValueError:
                pass
    if not values:
        return {
            "status": "no_resolved_epc_g2_available",
            "F_EPC": None,
            "R_EPC_local": None,
            "formula": {
                "F_EPC": "sum_inside |g(k)|^2 / sum_all |g(k)|^2",
                "R_EPC_local": "mean_inside |g(k)|^2 / mean_neighboring_background |g(k)|^2",
            },
        }
    coords_arr = np.array(coords, dtype=float)
    values_arr = np.array(values, dtype=float)
    masks_epc = window_masks(coords_arr, float(metric_k[masks["inside"]][0]), float(metric_k[masks["inside"]][-1]))
    inside = masks_epc["inside"]
    neighbor = masks_epc["left_neighbor"] | masks_epc["right_neighbor"]
    inside_mean = finite_mean(values_arr[inside])
    neighbor_mean = finite_mean(values_arr[neighbor])
    return {
        "status": "computed",
        "F_EPC": contribution_fraction(values_arr, inside),
        "R_EPC_local": inside_mean / neighbor_mean if inside_mean is not None and neighbor_mean not in (None, 0.0) else None,
        "inside_mean": inside_mean,
        "neighbor_mean": neighbor_mean,
    }


def save_bar_plot(path: Path, title: str, labels: list[str], values: list[float | None], ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    clean_values = [0.0 if value is None else value for value in values]
    ax.bar(labels, clean_values, color="#3b7a8f")
    ax.set_ylim(0, max(1.0, max(clean_values) * 1.15 if clean_values else 1.0))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_local_background(
    path: Path,
    curvature_stats: dict[str, Any],
    metric_stats: dict[str, Any],
) -> None:
    labels = ["left neighbor", "EDT window", "right neighbor"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    curv = curvature_stats["combined_abs"]
    axes[0].bar(
        labels,
        [
            curv["left_neighbor_mean_abs"] or 0.0,
            curv["inside_mean_abs"] or 0.0,
            curv["right_neighbor_mean_abs"] or 0.0,
        ],
        color=["#8f8f8f", "#d8863b", "#8f8f8f"],
    )
    axes[0].set_title("Curvature local background")
    axes[0].set_ylabel("mean combined |kappa|")
    metric = metric_stats["all_bands_sum"]
    axes[1].bar(
        labels,
        [
            metric["left_neighbor_mean_abs"] or 0.0,
            metric["inside_mean_abs"] or 0.0,
            metric["right_neighbor_mean_abs"] or 0.0,
        ],
        color=["#8f8f8f", "#d8863b", "#8f8f8f"],
    )
    axes[1].set_title("Metric local background")
    axes[1].set_ylabel("mean sum_bands g_parallel")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_k0_comparison(
    path: Path,
    k: np.ndarray,
    curvature_minus: np.ndarray,
    curvature_plus: np.ndarray,
    metric_k: np.ndarray,
    metric_sum: np.ndarray,
    k_left: float,
    k_right: float,
    k0: float,
) -> None:
    width = k_right - k_left
    x_min = max(float(np.min(k)), k0 - width)
    x_max = min(float(np.max(k)), k0 + width)
    mask_c = (k >= x_min) & (k <= x_max)
    mask_m = (metric_k >= x_min) & (metric_k <= x_max)
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.2), sharex=True)
    axes[0].plot(k[mask_c], curvature_minus[mask_c], label="kappa branch -", lw=1.4)
    axes[0].plot(k[mask_c], curvature_plus[mask_c], label="kappa branch +", lw=1.4)
    axes[0].set_ylabel("kappa")
    axes[0].legend(frameon=False)
    axes[1].plot(metric_k[mask_m], metric_sum[mask_m], color="#3b7a8f", lw=1.4, label="sum_bands g_parallel")
    axes[1].set_ylabel("metric sum")
    axes[1].set_xlabel("AGR path coordinate")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.axvspan(k_left, k_right, color="tab:orange", alpha=0.16)
        ax.axvline(k0, color="tab:orange", lw=1.2)
    fig.suptitle("Na k0-centered local comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run(
    agr_path: Path,
    summary_path: Path,
    analysis_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    summary = read_json(summary_path)
    nominal = summary["nominal_result"]
    k_left = float(nominal["k_left"])
    k_right = float(nominal["k_right"])
    k0 = float(nominal["k0"])
    datasets = parse_agr(agr_path)
    minus_label = f"G0.S{summary['dataset_minus']}"
    plus_label = f"G0.S{summary['dataset_plus']}"
    minus = datasets[minus_label]
    plus = datasets[plus_label]
    k = np.array(minus.x, dtype=float)
    kappa_minus = np.array(minus.y, dtype=float)
    kappa_plus = np.array(plus.y, dtype=float)
    masks = window_masks(k, k_left, k_right)

    metric_rows = read_csv(analysis_dir / "quantum_metric_on_agr_path.csv")
    metric_k, metric_by_band_values, metric_sum, metric_mean = metric_by_k(metric_rows)
    metric_masks = window_masks(metric_k, k_left, k_right)

    curvature_stats = {
        "minus_branch": summarize_channel(k, kappa_minus, masks, k0),
        "plus_branch": summarize_channel(k, kappa_plus, masks, k0),
        "combined_abs": summarize_channel(k, np.abs(kappa_minus) + np.abs(kappa_plus), masks, k0),
    }
    metric_stats = {
        "all_bands_sum": summarize_channel(metric_k, metric_sum, metric_masks, k0),
        "all_bands_mean": summarize_channel(metric_k, metric_mean, metric_masks, k0),
        "bands": {
            str(band): summarize_channel(metric_k, values, metric_masks, k0)
            for band, values in metric_by_band_values.items()
        },
    }
    epc = epc_placeholder(analysis_dir, metric_masks, metric_k)
    result = {
        "material": "Na",
        "mode_label": summary.get("mode_label"),
        "edt_window": {
            "k0": k0,
            "k_left": k_left,
            "k_right": k_right,
            "edt_width": k_right - k_left,
        },
        "curvature_contribution_fraction": {
            "minus_branch_abs": curvature_stats["minus_branch"]["contribution_fraction_abs"],
            "plus_branch_abs": curvature_stats["plus_branch"]["contribution_fraction_abs"],
            "combined_abs": curvature_stats["combined_abs"]["contribution_fraction_abs"],
        },
        "metric_contribution_fraction": {
            "all_bands_sum": metric_stats["all_bands_sum"]["contribution_fraction_abs"],
            "all_bands_mean": metric_stats["all_bands_mean"]["contribution_fraction_abs"],
            "by_band": {
                band: stats["contribution_fraction_abs"]
                for band, stats in metric_stats["bands"].items()
            },
        },
        "local_background_contrast": {
            "curvature_combined_abs": curvature_stats["combined_abs"]["local_background_contrast_abs"],
            "metric_all_bands_sum": metric_stats["all_bands_sum"]["local_background_contrast_abs"],
            "definition": "inside mean divided by mean over immediately neighboring same-width left/right windows",
        },
        "k0_centered_values": {
            "curvature_minus": curvature_stats["minus_branch"]["k0_value"],
            "curvature_plus": curvature_stats["plus_branch"]["k0_value"],
            "curvature_combined_abs": curvature_stats["combined_abs"]["k0_value"],
            "metric_all_bands_sum": metric_stats["all_bands_sum"]["k0_value"],
            "metric_all_bands_mean": metric_stats["all_bands_mean"]["k0_value"],
            "metric_by_band": {
                band: stats["k0_value"]
                for band, stats in metric_stats["bands"].items()
            },
            "epc": None,
        },
        "stability": {
            "curvature_minus": {
                "inside_std": curvature_stats["minus_branch"]["inside_std"],
                "outside_std": curvature_stats["minus_branch"]["outside_std"],
                "inside_abs_cv": curvature_stats["minus_branch"]["inside_abs_cv"],
                "outside_abs_cv": curvature_stats["minus_branch"]["outside_abs_cv"],
            },
            "curvature_plus": {
                "inside_std": curvature_stats["plus_branch"]["inside_std"],
                "outside_std": curvature_stats["plus_branch"]["outside_std"],
                "inside_abs_cv": curvature_stats["plus_branch"]["inside_abs_cv"],
                "outside_abs_cv": curvature_stats["plus_branch"]["outside_abs_cv"],
            },
            "curvature_combined_abs": {
                "inside_abs_std": curvature_stats["combined_abs"]["inside_abs_std"],
                "outside_abs_std": curvature_stats["combined_abs"]["outside_abs_std"],
                "inside_abs_cv": curvature_stats["combined_abs"]["inside_abs_cv"],
                "outside_abs_cv": curvature_stats["combined_abs"]["outside_abs_cv"],
            },
        },
        "detailed_curvature": curvature_stats,
        "detailed_metric": metric_stats,
        "epc_future_metrics": epc,
        "interpretation": "Na's EDT window is not a global quantum-metric maximum, but it may still be a physically significant local curvature-stable region.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_out = output_dir / "edt_local_significance_summary.json"
    summary_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    save_bar_plot(
        output_dir / "fig_na_curvature_contribution_fraction.png",
        "Na curvature contribution fraction",
        ["minus |kappa|", "plus |kappa|", "combined |kappa|"],
        [
            result["curvature_contribution_fraction"]["minus_branch_abs"],
            result["curvature_contribution_fraction"]["plus_branch_abs"],
            result["curvature_contribution_fraction"]["combined_abs"],
        ],
        "sum_inside / sum_all",
    )
    save_bar_plot(
        output_dir / "fig_na_metric_contribution_fraction.png",
        "Na metric contribution fraction",
        ["all-band sum", *[f"band {band}" for band in sorted(metric_stats["bands"], key=int)]],
        [
            result["metric_contribution_fraction"]["all_bands_sum"],
            *[metric_stats["bands"][band]["contribution_fraction_abs"] for band in sorted(metric_stats["bands"], key=int)],
        ],
        "sum_inside / sum_all",
    )
    plot_local_background(
        output_dir / "fig_na_local_background_comparison.png",
        curvature_stats,
        metric_stats,
    )
    plot_k0_comparison(
        output_dir / "fig_na_k0_centered_comparison.png",
        k,
        kappa_minus,
        kappa_plus,
        metric_k,
        metric_sum,
        k_left,
        k_right,
        k0,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agr", type=Path, default=Path("data/na/Na-m5.agr"))
    parser.add_argument("--summary", type=Path, default=Path("data/processed/S1_Na_summary.json"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("epw_analysis/Na"))
    args = parser.parse_args(argv)

    result = run(args.agr, args.summary, args.analysis_dir, args.analysis_dir)
    print("Na local significance:")
    print(f"  F_kappa combined = {result['curvature_contribution_fraction']['combined_abs']:.8g}")
    print(f"  F_metric all bands = {result['metric_contribution_fraction']['all_bands_sum']:.8g}")
    print(f"  local curvature contrast = {result['local_background_contrast']['curvature_combined_abs']:.8g}")
    print(f"  local metric contrast = {result['local_background_contrast']['metric_all_bands_sum']:.8g}")
    print(f"  wrote {args.analysis_dir / 'edt_local_significance_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
