#!/usr/bin/env python3
"""Generate EDT summary JSON files for the three Nb curvature modes."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from make_material_figure2_panels import (
    MaterialConfig,
    analyze_pair,
    bootstrap_widths,
    clean_grace_label,
    parse_agr,
    parse_agr_metadata,
    threshold_sweep,
)


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class NbMode:
    key: str
    mode_label: str
    agr_path: Path
    summary_path: Path


NB_MODES = [
    NbMode("Nb_m4", "m4", ROOT / "data" / "nb" / "Nb-second-m4.agr", ROOT / "data" / "processed" / "Nb_m4_summary.json"),
    NbMode("Nb_m5", "m5", ROOT / "data" / "nb" / "Nb-seconddri_m5.agr", ROOT / "data" / "processed" / "Nb_m5_summary.json"),
    NbMode("Nb_m6", "m6", ROOT / "data" / "nb" / "Nb-seconddri_m6.agr", ROOT / "data" / "processed" / "Nb_m6_summary.json"),
]

MANUAL_WINDOW_OVERRIDES = {
    "Nb_m5": {
        "k_right": 2.880955,
        "reason": "User-corrected EDT right edge: stop just before the major path-projected metric-sum spike near k=2.91-2.93.",
    },
    "Nb_m6": {
        "k_right": 2.880955,
        "reason": "User-corrected EDT right edge: stop just before the major path-projected metric-sum spike near k=2.91-2.93.",
    },
}


def auto_select_adjacent_pair(
    mode: NbMode,
    datasets: list[np.ndarray],
    fit_half_window: int,
    threshold: float,
) -> tuple[int, int, dict[str, Any]]:
    """Select the strongest adjacent pair with a finite extracted EDT window."""

    best: tuple[float, int, int, dict[str, Any]] | None = None
    for idx in range(len(datasets) - 1):
        x = datasets[idx][:, 0]
        y_a = datasets[idx][:, 1]
        y_b = datasets[idx + 1][:, 1]
        candidates = [
            {
                "dataset_minus": idx,
                "dataset_plus": idx + 1,
                "k_minus_extreme": float(x[int(np.argmin(y_a))]),
                "k_plus_extreme": float(x[int(np.argmax(y_b))]),
                "minus_extreme": float(np.min(y_a)),
                "plus_extreme": float(np.max(y_b)),
            },
            {
                "dataset_minus": idx + 1,
                "dataset_plus": idx,
                "k_minus_extreme": float(x[int(np.argmin(y_b))]),
                "k_plus_extreme": float(x[int(np.argmax(y_a))]),
                "minus_extreme": float(np.min(y_b)),
                "plus_extreme": float(np.max(y_a)),
            },
        ]
        for candidate in candidates:
            config = MaterialConfig(
                key=mode.key,
                figure_tag="Nb",
                title=f"Nb {mode.mode_label}",
                mode_label=mode.mode_label,
                source_file=mode.agr_path,
                band_figure=ROOT / "figures" / "supplementary" / "Nb_bands.pdf",
                dataset_minus=int(candidate["dataset_minus"]),
                dataset_plus=int(candidate["dataset_plus"]),
                fit_half_window=fit_half_window,
                threshold=threshold,
            )
            try:
                analysis = analyze_pair(config, datasets)
            except Exception:
                continue
            if analysis.edt_width <= 0.0:
                continue
            inside = (analysis.k >= analysis.k_left) & (analysis.k <= analysis.k_right)
            strength = abs(candidate["minus_extreme"]) + abs(candidate["plus_extreme"])
            exchange_mean_abs = float(np.mean(np.abs(analysis.y_plus[inside] + analysis.y_minus[inside])))
            separation = abs(candidate["k_minus_extreme"] - candidate["k_plus_extreme"])
            score = analysis.edt_width * strength / (1.0 + exchange_mean_abs)
            if best is None or score > best[0]:
                best = (
                    score,
                    int(candidate["dataset_minus"]),
                    int(candidate["dataset_plus"]),
                    {
                        **candidate,
                        "score": float(score),
                        "edt_width": float(analysis.edt_width),
                        "exchange_mean_abs_inside": exchange_mean_abs,
                        "paired_extreme_separation": float(separation),
                        "selection_rule": (
                            "max finite-window score = edt_width * (|min_minus| + |max_plus|) "
                            "/ (1 + mean_inside_abs(kappa_plus + kappa_minus))"
                        ),
                    },
                )
    if best is None:
        raise ValueError("No adjacent dataset pair produced a finite EDT window.")
    _score, dataset_minus, dataset_plus, metadata = best
    return dataset_minus, dataset_plus, metadata


def build_summary(mode: NbMode, fit_half_window: int = 8, threshold: float = 0.08) -> dict[str, Any]:
    datasets = parse_agr(mode.agr_path)
    metadata = parse_agr_metadata(mode.agr_path)
    dataset_minus, dataset_plus, selection = auto_select_adjacent_pair(mode, datasets, fit_half_window, threshold)
    config = MaterialConfig(
        key=mode.key,
        figure_tag="Nb",
        title=f"Nb {mode.mode_label}",
        mode_label=mode.mode_label,
        source_file=mode.agr_path,
        band_figure=ROOT / "figures" / "supplementary" / "Nb_bands.pdf",
        dataset_minus=dataset_minus,
        dataset_plus=dataset_plus,
        fit_half_window=fit_half_window,
        threshold=threshold,
    )
    analysis = analyze_pair(config, datasets)
    thresholds, widths = threshold_sweep(config, datasets)
    bootstrap = bootstrap_widths(analysis)
    ci_low, ci_high = np.percentile(bootstrap, [2.5, 97.5])
    nominal_result = {
        "k0": analysis.k0,
        "k_left": analysis.k_left,
        "k_right": analysis.k_right,
        "edt_width": analysis.edt_width,
        "core_width": analysis.core_width,
    }
    manual_override = MANUAL_WINDOW_OVERRIDES.get(mode.key)
    if manual_override:
        original_nominal_result = dict(nominal_result)
        nominal_result["k_right"] = float(manual_override["k_right"])
        nominal_result["edt_width"] = float(nominal_result["k_right"] - nominal_result["k_left"])
        nominal_result["core_width"] = nominal_result["edt_width"]
        nominal_result["k0"] = 0.5 * float(nominal_result["k_left"] + nominal_result["k_right"])
    else:
        original_nominal_result = None

    summary = {
        "material": mode.key,
        "title": f"Nb {mode.mode_label}",
        "mode_label": mode.mode_label,
        "data_file": str(mode.agr_path.relative_to(ROOT)),
        "band_figure": str(config.band_figure.relative_to(ROOT)),
        "dataset_minus": dataset_minus,
        "dataset_plus": dataset_plus,
        "auto_pair_selection": selection,
        "analysis_window_indices": [config.start_idx, config.end_idx],
        "fit_half_window_points": fit_half_window,
        "residual_threshold": threshold,
        "nominal_result": nominal_result,
        "bootstrap": {
            "samples": int(bootstrap.size),
            "width_mean": float(np.mean(bootstrap)),
            "width_std": float(np.std(bootstrap, ddof=1)),
            "width_ci_95": [float(ci_low), float(ci_high)],
        },
        "threshold_scan": {
            "thresholds": thresholds.tolist(),
            "widths": widths.tolist(),
        },
        "archived_y_label": clean_grace_label(str(metadata["y_label_raw"])),
        "path_labels": metadata["path_labels"],
    }
    if manual_override:
        summary["manual_window_override"] = {
            **manual_override,
            "original_nominal_result": original_nominal_result,
            "corrected_k0_rule": "midpoint of corrected [k_left, k_right] interval",
        }
    return summary


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-half-window", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.08)
    args = parser.parse_args(argv)

    for mode in NB_MODES:
        summary = build_summary(mode, fit_half_window=args.fit_half_window, threshold=args.threshold)
        write_summary(mode.summary_path, summary)
        window = summary["nominal_result"]
        print(
            f"{mode.key}: pair S{summary['dataset_minus']}/S{summary['dataset_plus']} "
            f"k=[{window['k_left']:.6g}, {window['k0']:.6g}, {window['k_right']:.6g}] "
            f"width={window['edt_width']:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
