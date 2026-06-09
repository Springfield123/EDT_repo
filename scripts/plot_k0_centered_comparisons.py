#!/usr/bin/env python3
"""Make k0-centered EDT local comparison plots for non-Na materials."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from wannier_geometry_pipeline import parse_agr


DEFAULT_AGR = {
    "V": Path("data/v/V-EP-F1.agr"),
    "Nb": Path("data/nb/bands-Nb.agr"),
    "Nb_m4": Path("data/nb/Nb-second-m4.agr"),
    "Nb_m5": Path("data/nb/Nb-seconddri_m5.agr"),
    "Nb_m6": Path("data/nb/Nb-seconddri_m6.agr"),
    "YBaCuO": Path("data/ybacu3o7/YBa2Cu3O7_m33.agr"),
    "K3C60": Path("data/k3c60/tmp.agr"),
}
EXCHANGE_SCORE_SMALL_THRESHOLD = 0.25


@dataclass
class WindowInfo:
    material: str
    mode_label: str | None
    k_left: float
    k0: float
    k_right: float
    delta_k_edt: float
    path_labels: list[str]
    kappa_minus_label: str
    kappa_plus_label: str
    band_pair: str | None = None


@dataclass
class PlotResult:
    material: str
    status: str
    summary: dict[str, Any]
    row: dict[str, Any]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return ""
        return f"{float(value):.12g}"
    return str(value)


def finite_values(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def finite_mean(values: Any) -> float | None:
    arr = finite_values(values)
    return float(np.mean(arr)) if arr.size else None


def finite_max(values: Any) -> float | None:
    arr = finite_values(values)
    return float(np.max(arr)) if arr.size else None


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0.0):
        return None
    return numerator / denominator


def required_input_paths(material: str, analysis_dir: Path, agr_path: Path | None) -> dict[str, Path | None]:
    return {
        "edt_geometry_summary": analysis_dir / "edt_geometry_summary.json",
        "edt_geometry_comparison": analysis_dir / "edt_geometry_comparison.csv",
        "quantum_metric": analysis_dir / "quantum_metric_on_agr_path.csv",
        "wannier_bands": analysis_dir / "wannier_bands_on_agr_path.csv",
        "agr_curvature": agr_path,
    }


def missing_inputs(paths: dict[str, Path | None]) -> list[str]:
    missing = []
    for label, path in paths.items():
        if path is None or not path.exists():
            missing.append(label)
    return missing


def agr_yaxis_label(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re_search_yaxis_label(text)
    return match


def re_search_yaxis_label(text: str) -> str | None:
    import re

    match = re.search(r'@\s+yaxis\s+label\s+"([^"]*)"', text)
    return match.group(1) if match else None


def input_warnings(paths: dict[str, Path | None]) -> list[str]:
    warnings = []
    agr_path = paths.get("agr_curvature")
    label = agr_yaxis_label(agr_path)
    if label and "band energy" in label.lower():
        warnings.append(f"AGR file is labeled {label!r}; it is not an EDT curvature AGR file.")
    summary_path = paths.get("edt_geometry_summary")
    if summary_path is not None and summary_path.exists():
        try:
            summary = read_json(summary_path)
        except json.JSONDecodeError:
            warnings.append("EDT geometry summary exists but is not valid JSON.")
        else:
            status = summary.get("status")
            if status not in (None, "completed"):
                warnings.append(f"EDT geometry summary status is {status!r}, not 'completed'.")
    return warnings


def surface_availability(analysis_dir: Path) -> str:
    summary = analysis_dir / "berry_metric_surface_summary.json"
    if not summary.exists():
        return "not_available"
    try:
        data = read_json(summary)
    except json.JSONDecodeError:
        return "summary_not_valid_json"
    return data.get("status") or "available"


def epc_availability(analysis_dir: Path) -> str:
    for path in (analysis_dir / "epc_extraction_status.json", analysis_dir / "edt_epc_summary.json", analysis_dir / "epc_edt_window_summary.json"):
        if not path.exists():
            continue
        try:
            data = read_json(path)
        except json.JSONDecodeError:
            return f"{path.name}:not_valid_json"
        status = data.get("status") or data.get("resolution")
        if status:
            return str(status)
    return "not_available"


def detect_curvature_labels(summary: dict[str, Any]) -> tuple[str, str]:
    datasets = summary.get("curvature_datasets") or {}
    if len(datasets) < 2:
        raise ValueError("EDT geometry summary does not contain two curvature dataset labels.")
    scored = []
    for label, stats in datasets.items():
        inside_mean = stats.get("inside_mean")
        score = float(inside_mean) if inside_mean is not None else 0.0
        scored.append((score, label))
    scored.sort()
    return scored[0][1], scored[-1][1]


def format_band_pair(summary: dict[str, Any]) -> str | None:
    pair = summary.get("selected_pair") or {}
    minus = pair.get("band_minus_label")
    plus = pair.get("band_plus_label")
    if minus and plus:
        return f"{minus}/{plus}"
    if pair.get("dataset_minus") is not None and pair.get("dataset_plus") is not None:
        return f"Band {int(pair['dataset_minus']) + 1}/Band {int(pair['dataset_plus']) + 1}"
    return None


def load_window_info(material: str, summary: dict[str, Any], allow_pending: bool = False) -> WindowInfo:
    if summary.get("status") not in (None, "completed") and not allow_pending:
        raise ValueError(f"{material} EDT geometry is not completed: {summary.get('status')}")
    window = summary.get("edt_window") or {}
    k_left = float(window["k_left"])
    k0 = float(window["k0"])
    k_right = float(window["k_right"])
    delta = float(window.get("delta_k_EDT") or window.get("edt_width") or (k_right - k_left))
    minus_label, plus_label = detect_curvature_labels(summary)
    return WindowInfo(
        material=material,
        mode_label=summary.get("mode_label"),
        k_left=k_left,
        k0=k0,
        k_right=k_right,
        delta_k_edt=delta,
        path_labels=list(summary.get("path_labels") or []),
        kappa_minus_label=minus_label,
        kappa_plus_label=plus_label,
        band_pair=format_band_pair(summary),
    )


def load_curvature(agr_path: Path, window: WindowInfo) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    datasets = parse_agr(agr_path)
    if window.kappa_minus_label not in datasets or window.kappa_plus_label not in datasets:
        raise ValueError(
            f"{agr_path} does not contain curvature labels "
            f"{window.kappa_minus_label!r} and {window.kappa_plus_label!r}."
        )
    minus = datasets[window.kappa_minus_label]
    plus = datasets[window.kappa_plus_label]
    k = np.array(minus.x, dtype=float)
    k_plus = np.array(plus.x, dtype=float)
    if len(k) != len(k_plus) or np.max(np.abs(k - k_plus)) > 1e-8:
        plus_y = np.interp(k, k_plus, np.array(plus.y, dtype=float))
    else:
        plus_y = np.array(plus.y, dtype=float)
    return k, np.array(minus.y, dtype=float), plus_y


def load_metric(metric_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    rows = read_csv(metric_path)
    if not rows:
        raise ValueError(f"{metric_path} has no rows.")
    fieldnames = set(rows[0])
    if {"k_coord", "band", "quantum_metric_parallel"}.issubset(fieldnames):
        grouped: dict[float, float] = {}
        for row in rows:
            k = float(row["k_coord"])
            value = float(row["quantum_metric_parallel"])
            if math.isfinite(value):
                grouped[k] = grouped.get(k, 0.0) + value
        k_values = np.array(sorted(grouped), dtype=float)
        return k_values, np.array([grouped[k] for k in k_values], dtype=float), "sum_bands_g_parallel"

    for column in ("sum_bands_g_parallel", "metric_sum", "quantum_metric_parallel", "metric"):
        if {"k_coord", column}.issubset(fieldnames):
            grouped = {}
            for row in rows:
                grouped[float(row["k_coord"])] = float(row[column])
            k_values = np.array(sorted(grouped), dtype=float)
            return k_values, np.array([grouped[k] for k in k_values], dtype=float), column
    raise ValueError(f"{metric_path} does not contain a recognized metric column.")


def mask_between(k: np.ndarray, left: float, right: float) -> np.ndarray:
    return (k >= left) & (k <= right)


def local_plot_limits(k: np.ndarray, window: WindowInfo, factor: float) -> tuple[float, float]:
    x_min = max(float(np.min(k)), window.k_left - factor * window.delta_k_edt)
    x_max = min(float(np.max(k)), window.k_right + factor * window.delta_k_edt)
    return x_min, x_max


def save_k0_plot(
    output: Path,
    material: str,
    window: WindowInfo,
    k_curv: np.ndarray,
    kappa_minus: np.ndarray,
    kappa_plus: np.ndarray,
    k_metric: np.ndarray,
    metric_sum: np.ndarray,
    factor: float,
) -> None:
    x_min, x_max = local_plot_limits(k_curv, window, factor)
    curv_mask = mask_between(k_curv, x_min, x_max)
    metric_mask = mask_between(k_metric, x_min, x_max)

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.2), sharex=True)
    axes[0].plot(k_curv[curv_mask], kappa_minus[curv_mask], label="kappa branch -", lw=1.4)
    axes[0].plot(k_curv[curv_mask], kappa_plus[curv_mask], label="kappa branch +", lw=1.4)
    axes[0].set_ylabel("kappa")
    axes[0].legend(frameon=False)

    axes[1].plot(k_metric[metric_mask], metric_sum[metric_mask], color="#3b7a8f", lw=1.4, label="sum_bands g_parallel")
    axes[1].set_ylabel("metric sum")
    axes[1].set_xlabel("AGR path coordinate")
    axes[1].legend(frameon=False)

    for ax in axes:
        ax.axvspan(window.k_left, window.k_right, color="tab:orange", alpha=0.16)
        ax.axvline(window.k0, color="tab:orange", lw=1.2)
        ax.set_xlim(x_min, x_max)
    fig.suptitle(f"{material} k0-centered local comparison")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def local_significance(
    window: WindowInfo,
    k_curv: np.ndarray,
    kappa_minus: np.ndarray,
    kappa_plus: np.ndarray,
    k_metric: np.ndarray,
    metric_sum: np.ndarray,
) -> dict[str, Any]:
    inside_curv = mask_between(k_curv, window.k_left, window.k_right)
    outside_curv = ~inside_curv
    inside_metric = mask_between(k_metric, window.k_left, window.k_right)
    outside_metric = ~inside_metric
    boundary_half_width = 0.1 * window.delta_k_edt
    left_boundary = mask_between(k_metric, window.k_left - boundary_half_width, window.k_left + boundary_half_width)
    right_boundary = mask_between(k_metric, window.k_right - boundary_half_width, window.k_right + boundary_half_width)

    mean_inside_abs_kappa_minus = finite_mean(np.abs(kappa_minus[inside_curv]))
    mean_inside_abs_kappa_plus = finite_mean(np.abs(kappa_plus[inside_curv]))
    mean_outside_abs_kappa_minus = finite_mean(np.abs(kappa_minus[outside_curv]))
    mean_outside_abs_kappa_plus = finite_mean(np.abs(kappa_plus[outside_curv]))
    exchange_numerator = finite_mean(np.abs(kappa_plus[inside_curv] + kappa_minus[inside_curv]))
    exchange_denominator = finite_mean(np.abs(kappa_plus[inside_curv] - kappa_minus[inside_curv]))
    exchange_score = ratio(exchange_numerator, exchange_denominator)

    metric_inside_mean = finite_mean(metric_sum[inside_metric])
    metric_outside_mean = finite_mean(metric_sum[outside_metric])
    metric_ratio = ratio(metric_inside_mean, metric_outside_mean)
    max_metric_inside = finite_max(metric_sum[inside_metric])
    max_metric_left_boundary = finite_max(metric_sum[left_boundary])
    max_metric_right_boundary = finite_max(metric_sum[right_boundary])
    boundary_max = finite_max([value for value in (max_metric_left_boundary, max_metric_right_boundary) if value is not None])

    return {
        "mean_inside_abs_kappa_minus": mean_inside_abs_kappa_minus,
        "mean_inside_abs_kappa_plus": mean_inside_abs_kappa_plus,
        "mean_outside_abs_kappa_minus": mean_outside_abs_kappa_minus,
        "mean_outside_abs_kappa_plus": mean_outside_abs_kappa_plus,
        "exchange_score": exchange_score,
        "exchange_score_definition": "mean_inside_abs(kappa_plus + kappa_minus) / mean_inside_abs(kappa_plus - kappa_minus)",
        "exchange_score_small_threshold": EXCHANGE_SCORE_SMALL_THRESHOLD,
        "metric_inside_mean": metric_inside_mean,
        "metric_outside_mean": metric_outside_mean,
        "metric_inside_outside_ratio": metric_ratio,
        "max_metric_inside_window": max_metric_inside,
        "max_metric_left_boundary": max_metric_left_boundary,
        "max_metric_right_boundary": max_metric_right_boundary,
        "boundary_metric_spikes_larger_than_inside": (
            boundary_max is not None and max_metric_inside is not None and boundary_max > max_metric_inside
        ),
        "left_boundary_region": [window.k_left - boundary_half_width, window.k_left + boundary_half_width],
        "right_boundary_region": [window.k_right - boundary_half_width, window.k_right + boundary_half_width],
    }


def interpretation(metrics: dict[str, Any]) -> str:
    metric_ratio = metrics.get("metric_inside_outside_ratio")
    boundary_spikes = bool(metrics.get("boundary_metric_spikes_larger_than_inside"))
    exchange_score = metrics.get("exchange_score")
    pieces = []
    if metric_ratio is not None and metric_ratio < 1.0 and boundary_spikes:
        pieces.append(
            "EDT window is not a global metric maximum; metric spikes appear near boundaries, suggesting rapid "
            "wavefunction-character changes at the edge of the structured curvature manifold."
        )
    elif metric_ratio is not None and metric_ratio > 1.0:
        pieces.append(
            "EDT window overlaps enhanced path-projected quantum metric, suggesting stronger quantum-geometric "
            "character inside the curvature-stable window."
        )
    elif metric_ratio is not None and metric_ratio < 1.0:
        pieces.append(
            "EDT window is not a global path-projected metric maximum; the available metric is lower inside on average, "
            "so any significance is more local than global."
        )
    else:
        pieces.append("EDT local metric contrast is inconclusive from the available path-projected metric.")
    if exchange_score is not None and exchange_score < EXCHANGE_SCORE_SMALL_THRESHOLD:
        pieces.append("Paired curvature branches show strong opposite-sign curvature exchange.")
    return " ".join(pieces)


def build_missing_result(material: str, analysis_dir: Path, paths: dict[str, Path | None], missing: list[str]) -> PlotResult:
    warnings = input_warnings(paths)
    interpretation_text = "Required EDT curvature/window and Wannier metric files are missing, so no k0-centered comparison was generated."
    if warnings:
        interpretation_text = f"{interpretation_text} {' '.join(warnings)}"
    window = None
    summary_path = paths.get("edt_geometry_summary")
    if summary_path is not None and summary_path.exists():
        try:
            window = load_window_info(material, read_json(summary_path), allow_pending=True)
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            window = None
    metrics: dict[str, Any] = {}
    agr_path = paths.get("agr_curvature")
    if window is not None and agr_path is not None and agr_path.exists():
        try:
            k_curv, kappa_minus, kappa_plus = load_curvature(agr_path, window)
            metrics = local_significance(window, k_curv, kappa_minus, kappa_plus, k_curv, np.full_like(k_curv, np.nan))
        except (KeyError, ValueError, TypeError):
            metrics = {}
    summary = {
        "material": material,
        "status": "missing_required_inputs",
        "analysis_dir": str(analysis_dir),
        "required_inputs": {label: str(path) if path is not None else None for label, path in paths.items()},
        "missing_inputs": missing,
        "input_warnings": warnings,
        "curvature_only_metrics": metrics,
        "interpretation": interpretation_text,
    }
    write_json(analysis_dir / "edt_local_significance_summary.json", summary)
    row = combined_row(
        material,
        window,
        metrics,
        interpretation_text,
        status="missing_required_inputs",
        warnings=warnings + [f"missing {name}" for name in missing],
        berry_metric_surface_available=surface_availability(analysis_dir),
        epc_available=epc_availability(analysis_dir),
    )
    return PlotResult(material=material, status="missing_required_inputs", summary=summary, row=row)


def combined_row(
    material: str,
    window: WindowInfo | None,
    metrics: dict[str, Any],
    interpretation_text: str,
    status: str = "completed",
    warnings: list[str] | None = None,
    berry_metric_surface_available: str | None = None,
    epc_available: str | None = None,
) -> dict[str, Any]:
    return {
        "material": material,
        "status": status,
        "mode_label": window.mode_label if window else None,
        "band_pair": window.band_pair if window else None,
        "k_left": window.k_left if window else None,
        "k0": window.k0 if window else None,
        "k_right": window.k_right if window else None,
        "delta_k_EDT": window.delta_k_edt if window else None,
        "mean_inside_abs_kappa_minus": metrics.get("mean_inside_abs_kappa_minus"),
        "mean_inside_abs_kappa_plus": metrics.get("mean_inside_abs_kappa_plus"),
        "exchange_score": metrics.get("exchange_score"),
        "curvature_exchange_score": metrics.get("exchange_score"),
        "metric_inside_mean": metrics.get("metric_inside_mean"),
        "metric_outside_mean": metrics.get("metric_outside_mean"),
        "metric_inside_outside_ratio": metrics.get("metric_inside_outside_ratio"),
        "max_metric_inside_window": metrics.get("max_metric_inside_window"),
        "max_metric_left_boundary": metrics.get("max_metric_left_boundary"),
        "max_metric_right_boundary": metrics.get("max_metric_right_boundary"),
        "berry_metric_surface_available": berry_metric_surface_available,
        "epc_available": epc_available,
        "warnings": " | ".join(warnings or []),
        "interpretation": interpretation_text,
    }


def plot_k0_centered_comparison(
    material: str,
    analysis_dir: str | Path,
    agr_path: str | Path | None = None,
) -> PlotResult:
    analysis_dir = Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    resolved_agr = Path(agr_path) if agr_path is not None else DEFAULT_AGR.get(material)
    paths = required_input_paths(material, analysis_dir, resolved_agr)
    missing = missing_inputs(paths)
    if missing:
        return build_missing_result(material, analysis_dir, paths, missing)

    assert resolved_agr is not None
    summary = read_json(analysis_dir / "edt_geometry_summary.json")
    window = load_window_info(material, summary)
    k_curv, kappa_minus, kappa_plus = load_curvature(resolved_agr, window)
    k_metric, metric_sum, metric_source = load_metric(analysis_dir / "quantum_metric_on_agr_path.csv")

    short_plot = analysis_dir / f"fig_{material.lower()}_k0_centered_comparison.png"
    wide_plot = analysis_dir / f"fig_{material.lower()}_k0_centered_comparison_wide.png"
    save_k0_plot(short_plot, material, window, k_curv, kappa_minus, kappa_plus, k_metric, metric_sum, factor=0.5)
    save_k0_plot(wide_plot, material, window, k_curv, kappa_minus, kappa_plus, k_metric, metric_sum, factor=1.0)

    metrics = local_significance(window, k_curv, kappa_minus, kappa_plus, k_metric, metric_sum)
    interpretation_text = interpretation(metrics)
    result = {
        "material": material,
        "status": "completed",
        "mode_label": window.mode_label,
        "edt_window": {
            "k_left": window.k_left,
            "k0": window.k0,
            "k_right": window.k_right,
            "delta_k_EDT": window.delta_k_edt,
            "path_labels": window.path_labels,
        },
        "curvature_branch_labels": {
            "kappa_minus": window.kappa_minus_label,
            "kappa_plus": window.kappa_plus_label,
        },
        "metric_source": metric_source,
        "local_significance": metrics,
        "interpretation": interpretation_text,
        "plots": {
            "k0_centered": str(short_plot),
            "k0_centered_wide": str(wide_plot),
        },
        "parsing_assumptions": {
            "delta_k_EDT": "Taken from edt_window.edt_width when delta_k_EDT is absent.",
            "curvature_branches": "Detected from edt_geometry_summary.curvature_datasets by ordering inside_mean; lower is kappa_minus and higher is kappa_plus.",
            "metric": "Band-resolved quantum_metric_parallel values are summed over bands at each k_coord.",
        },
    }
    write_json(analysis_dir / "edt_local_significance_summary.json", result)
    row = combined_row(
        material,
        window,
        metrics,
        interpretation_text,
        status="completed",
        warnings=input_warnings(paths),
        berry_metric_surface_available=surface_availability(analysis_dir),
        epc_available=epc_availability(analysis_dir),
    )
    return PlotResult(material=material, status="completed", summary=result, row=row)


def write_combined_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "material",
        "status",
        "mode_label",
        "band_pair",
        "k_left",
        "k0",
        "k_right",
        "delta_k_EDT",
        "mean_inside_abs_kappa_minus",
        "mean_inside_abs_kappa_plus",
        "exchange_score",
        "curvature_exchange_score",
        "metric_inside_mean",
        "metric_outside_mean",
        "metric_inside_outside_ratio",
        "max_metric_inside_window",
        "max_metric_left_boundary",
        "max_metric_right_boundary",
        "berry_metric_surface_available",
        "epc_available",
        "warnings",
        "interpretation",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fieldnames})


def run(materials: list[str], output_root: Path) -> list[PlotResult]:
    results = []
    for material in materials:
        result = plot_k0_centered_comparison(material, output_root / material)
        results.append(result)
    write_combined_table(output_root / "edt_local_significance_all_materials.csv", [result.row for result in results])
    write_combined_table(output_root / "edt_geometry_summary_all_materials.csv", [result.row for result in results])
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materials", nargs="*", default=["V", "Nb", "Nb_m4", "Nb_m5", "Nb_m6", "YBaCuO", "K3C60"])
    parser.add_argument("--output-root", type=Path, default=Path("epw_analysis"))
    args = parser.parse_args(argv)

    results = run(args.materials, args.output_root)
    for result in results:
        if result.status == "completed":
            window = result.summary["edt_window"]
            metrics = result.summary["local_significance"]
            print(
                f"{result.material}: completed "
                f"k=[{window['k_left']:.6g}, {window['k0']:.6g}, {window['k_right']:.6g}], "
                f"metric ratio={metrics['metric_inside_outside_ratio']:.6g}, "
                f"exchange={metrics['exchange_score']:.6g}"
            )
        else:
            print(f"{result.material}: {result.status}; missing {', '.join(result.summary['missing_inputs'])}")
    print(f"Wrote {args.output_root / 'edt_local_significance_all_materials.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
