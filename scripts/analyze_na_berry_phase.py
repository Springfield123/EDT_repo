#!/usr/bin/env python3
"""Compute Berry-phase diagnostics for the Na EDT window along the AGR path."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from wannier_geometry_pipeline import (
    diagonalize_path,
    interpolate_fractional_path,
    load_hr,
    load_summary,
    parse_agr,
    parse_vertical_ticks,
    parse_world_max,
    summary_dataset_id,
)


PHASE_LABELS = ("PhaseA", "PhaseB", "PhaseG", "PhaseT", "PhaseH")
TWO_PI = 2.0 * math.pi
SPECIAL_KPOINT_OVERRIDES = {
    "K3C60": {
        "x": np.array([0.0, 0.5, 0.5]),
        "l": np.array([0.5, 0.5, 0.5]),
        "w": np.array([0.25, 0.5, 0.75]),
    }
}


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


def load_agr_path(
    material: str,
    agr_path: Path,
    summary_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    summary = load_summary(summary_path)
    datasets = parse_agr(agr_path)
    grid_label = f"G0.S{summary_dataset_id(summary, 'dataset_minus')}"
    if grid_label not in datasets:
        raise ValueError(f"{agr_path} does not contain expected dataset {grid_label}")
    k_coords = np.array(datasets[grid_label].x, dtype=float)
    x_max = parse_world_max(agr_path, float(np.max(k_coords)))
    ticks = parse_vertical_ticks(agr_path, x_max)
    k_frac = interpolate_fractional_path(
        k_coords,
        ticks,
        summary["path_labels"],
        special_kpoints=SPECIAL_KPOINT_OVERRIDES.get(material),
    )
    return k_coords, k_frac, summary, datasets


def parallel_transport_phases(
    evecs: np.ndarray,
    overlap_tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fix phases by making adjacent same-band overlaps real-positive.

    Returns the gauge-fixed eigenvectors, the phase increments applied at each
    link, the post-fix residual link phases, and the same-band overlap
    magnitudes.  The phase increment equals ``-angle(<u_i|u_{i+1}>)`` using the
    already transported ``u_i`` and the next raw vector.
    """

    transported = np.array(evecs, copy=True)
    n_k, _num_orbitals, n_bands = transported.shape
    increments = np.full((max(n_k - 1, 0), n_bands), np.nan, dtype=float)
    residuals = np.full_like(increments, np.nan)
    overlap_abs = np.full_like(increments, np.nan)
    for band in range(n_bands):
        for idx in range(n_k - 1):
            overlap = np.vdot(transported[idx, :, band], transported[idx + 1, :, band])
            mag = abs(overlap)
            overlap_abs[idx, band] = mag
            if mag <= overlap_tol:
                continue
            increment = -float(np.angle(overlap))
            increments[idx, band] = increment
            transported[idx + 1, :, band] *= np.exp(1j * increment)
            residual = np.vdot(transported[idx, :, band], transported[idx + 1, :, band])
            residuals[idx, band] = -float(np.angle(residual))
    return transported, increments, residuals, overlap_abs


def cumulative_from_increments(increments: np.ndarray) -> np.ndarray:
    cumulative = np.zeros((increments.shape[0] + 1, increments.shape[1]), dtype=float)
    clean = np.where(np.isfinite(increments), increments, 0.0)
    cumulative[1:, :] = np.cumsum(clean, axis=0)
    return cumulative


def window_link_masks(k_coords: np.ndarray, k_left: float, k_right: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    midpoints = 0.5 * (k_coords[:-1] + k_coords[1:])
    inside = (midpoints >= k_left) & (midpoints <= k_right)
    return midpoints, inside, ~inside


def boundary_link_index(k_coords: np.ndarray, boundary: float) -> int:
    for idx in range(len(k_coords) - 1):
        if k_coords[idx] <= boundary <= k_coords[idx + 1]:
            return idx
    midpoints = 0.5 * (k_coords[:-1] + k_coords[1:])
    return int(np.argmin(np.abs(midpoints - boundary)))


def winding_classification(winding: float | None, tolerance: float = 0.1) -> str:
    if winding is None or not math.isfinite(winding):
        return "unavailable"
    distance_integer = abs(winding - round(winding))
    nearest_half = math.floor(winding) + 0.5
    distance_half = abs(winding - nearest_half)
    if distance_integer <= tolerance:
        return "near_integer"
    if distance_half <= tolerance:
        return "near_half_integer"
    return "fractional"


def finite_sum(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.sum(finite)) if finite.size else None


def phase_stats(
    increments: np.ndarray,
    inside_links: np.ndarray,
    left_boundary_idx: int,
    right_boundary_idx: int,
) -> dict[str, Any]:
    outside_links = ~inside_links
    inside_total = finite_sum(increments[inside_links])
    outside_total = finite_sum(increments[outside_links])
    total = finite_sum(increments)
    inside_abs_total = finite_sum(np.abs(increments[inside_links]))
    outside_abs_total = finite_sum(np.abs(increments[outside_links]))
    winding = inside_total / TWO_PI if inside_total is not None else None
    return {
        "inside_total_phase_rad": inside_total,
        "outside_total_phase_rad": outside_total,
        "total_phase_rad": total,
        "inside_abs_phase_sum_rad": inside_abs_total,
        "outside_abs_phase_sum_rad": outside_abs_total,
        "inside_winding_cycles": winding,
        "inside_winding_classification": winding_classification(winding),
        "left_boundary_phase_jump_rad": (
            float(increments[left_boundary_idx]) if math.isfinite(float(increments[left_boundary_idx])) else None
        ),
        "right_boundary_phase_jump_rad": (
            float(increments[right_boundary_idx]) if math.isfinite(float(increments[right_boundary_idx])) else None
        ),
        "left_boundary_phase_jump_cycles": (
            float(increments[left_boundary_idx] / TWO_PI)
            if math.isfinite(float(increments[left_boundary_idx]))
            else None
        ),
        "right_boundary_phase_jump_cycles": (
            float(increments[right_boundary_idx] / TWO_PI)
            if math.isfinite(float(increments[right_boundary_idx]))
            else None
        ),
    }


def select_bands(increments: np.ndarray, inside_links: np.ndarray, requested: list[int] | None, count: int) -> list[int]:
    n_bands = increments.shape[1]
    if requested:
        return sorted({band for band in requested if 1 <= band <= n_bands})
    inside_totals = []
    for band in range(n_bands):
        total = finite_sum(increments[inside_links, band])
        inside_totals.append((abs(total or 0.0), band + 1))
    return [band for _score, band in sorted(inside_totals, reverse=True)[:count]]


def write_phase_csv(
    path: Path,
    material: str,
    k_coords: np.ndarray,
    evals: np.ndarray,
    increments: np.ndarray,
    cumulative: np.ndarray,
    residuals: np.ndarray,
    overlap_abs: np.ndarray,
    k_left: float,
    k_right: float,
) -> None:
    midpoints, link_inside, _outside = window_link_masks(k_coords, k_left, k_right)
    point_inside = (k_coords >= k_left) & (k_coords <= k_right)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "material",
            "k_index",
            "k_coord",
            "k_mid_to_next",
            "band",
            "energy",
            "overlap_abs_to_next",
            "phase_increment",
            "phase_increment_over_2pi",
            "phase_gradient",
            "cumulative_phase",
            "cumulative_phase_over_2pi",
            "pt_residual_phase_increment",
            "inside_edt_window",
            "link_inside_edt_window",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, k_coord in enumerate(k_coords):
            for band in range(evals.shape[1]):
                has_link = idx < increments.shape[0]
                dk = float(k_coords[idx + 1] - k_coord) if has_link else math.nan
                inc = float(increments[idx, band]) if has_link else math.nan
                writer.writerow(
                    {
                        "material": material,
                        "k_index": idx,
                        "k_coord": csv_value(k_coord),
                        "k_mid_to_next": csv_value(midpoints[idx] if has_link else None),
                        "band": band + 1,
                        "energy": csv_value(evals[idx, band]),
                        "overlap_abs_to_next": csv_value(overlap_abs[idx, band] if has_link else None),
                        "phase_increment": csv_value(inc if has_link else None),
                        "phase_increment_over_2pi": csv_value(inc / TWO_PI if has_link and math.isfinite(inc) else None),
                        "phase_gradient": csv_value(inc / dk if has_link and math.isfinite(inc) and abs(dk) > 1e-14 else None),
                        "cumulative_phase": csv_value(cumulative[idx, band]),
                        "cumulative_phase_over_2pi": csv_value(cumulative[idx, band] / TWO_PI),
                        "pt_residual_phase_increment": csv_value(residuals[idx, band] if has_link else None),
                        "inside_edt_window": bool(point_inside[idx]),
                        "link_inside_edt_window": bool(link_inside[idx]) if has_link else "",
                    }
                )


def phase_files_status(root: Path) -> dict[str, Any]:
    search_roots = [root / "data", root / "epw_analysis", root / "epw_runs"]
    out: dict[str, Any] = {}
    for label in PHASE_LABELS:
        matches: list[str] = []
        for search_root in search_roots:
            if not search_root.exists():
                continue
            for path in search_root.rglob("*"):
                if path.is_file() and label.lower() in path.name.lower():
                    matches.append(str(path))
        out[label] = {
            "available": bool(matches),
            "files": matches,
            "plot_status": "not_generated_no_phase_file" if not matches else "available_not_parsed_by_na_wannier_phase_script",
        }
    return out


def plot_berry_phase_comparison(
    output: Path,
    k_coords: np.ndarray,
    curvature_minus: np.ndarray,
    curvature_plus: np.ndarray,
    increments: np.ndarray,
    cumulative: np.ndarray,
    selected_bands: list[int],
    k_left: float,
    k0: float,
    k_right: float,
) -> None:
    width = k_right - k_left
    x_min = max(float(np.min(k_coords)), k0 - width)
    x_max = min(float(np.max(k_coords)), k0 + width)
    point_mask = (k_coords >= x_min) & (k_coords <= x_max)
    midpoints = 0.5 * (k_coords[:-1] + k_coords[1:])
    link_mask = (midpoints >= x_min) & (midpoints <= x_max)
    gradients = np.full_like(increments, np.nan)
    dk = np.diff(k_coords)
    good_dk = np.abs(dk) > 1e-14
    gradients[good_dk, :] = increments[good_dk, :] / dk[good_dk, None]
    mean_cumulative = np.nanmean(cumulative, axis=1)
    mean_gradient = np.nanmean(gradients, axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(8.4, 7.6), sharex=True)
    axes[0].plot(k_coords[point_mask], curvature_minus[point_mask], label="kappa -", lw=1.3)
    axes[0].plot(k_coords[point_mask], curvature_plus[point_mask], label="kappa +", lw=1.3)
    axes[0].set_ylabel("kappa")
    axes[0].legend(frameon=False, ncols=2)

    axes[1].plot(k_coords[point_mask], mean_cumulative[point_mask] / TWO_PI, color="black", lw=1.6, label="band mean")
    for band in selected_bands:
        axes[1].plot(
            k_coords[point_mask],
            cumulative[point_mask, band - 1] / TWO_PI,
            lw=1.0,
            alpha=0.85,
            label=f"band {band}",
        )
    axes[1].set_ylabel("cumulative phase / 2pi")
    axes[1].legend(frameon=False, ncols=4, fontsize=8)

    axes[2].plot(midpoints[link_mask], mean_gradient[link_mask], color="black", lw=1.5, label="band mean")
    for band in selected_bands:
        axes[2].plot(midpoints[link_mask], gradients[link_mask, band - 1], lw=0.95, alpha=0.8, label=f"band {band}")
    axes[2].set_ylabel("phase gradient")
    axes[2].set_xlabel("AGR path coordinate")
    axes[2].legend(frameon=False, ncols=4, fontsize=8)

    for ax in axes:
        ax.axvspan(k_left, k_right, color="tab:orange", alpha=0.16)
        ax.axvline(k0, color="tab:orange", lw=1.1)
    fig.suptitle("k0-centered Berry-phase / EDT-window comparison")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def build_summary(
    material: str,
    k_coords: np.ndarray,
    increments: np.ndarray,
    residuals: np.ndarray,
    overlap_abs: np.ndarray,
    selected_bands: list[int],
    k_left: float,
    k0: float,
    k_right: float,
    phase_csv: Path,
    plot_path: Path,
    phase_file_status: dict[str, Any],
) -> dict[str, Any]:
    midpoints, inside_links, outside_links = window_link_masks(k_coords, k_left, k_right)
    left_boundary_idx = boundary_link_index(k_coords, k_left)
    right_boundary_idx = boundary_link_index(k_coords, k_right)
    band_stats = {}
    for band in range(increments.shape[1]):
        stats = phase_stats(increments[:, band], inside_links, left_boundary_idx, right_boundary_idx)
        stats["min_overlap_abs"] = float(np.nanmin(overlap_abs[:, band]))
        stats["mean_overlap_abs"] = float(np.nanmean(overlap_abs[:, band]))
        band_stats[str(band + 1)] = stats

    mean_increment = np.nanmean(increments, axis=1)
    sum_increment = np.nansum(increments, axis=1)
    aggregate_stats = {
        "band_mean": phase_stats(mean_increment, inside_links, left_boundary_idx, right_boundary_idx),
        "band_sum": phase_stats(sum_increment, inside_links, left_boundary_idx, right_boundary_idx),
    }
    finite_residuals = residuals[np.isfinite(residuals)]
    return {
        "material": material,
        "status": "completed",
        "phase_definition": (
            "phase_increment is the parallel-transport link correction -angle(<u_i^PT|u_{i+1}>); "
            "after applying it, the same-band overlap is real-positive when overlap_abs is above tolerance."
        ),
        "open_path_note": (
            "These are open-path, band-tracked, parallel-transport phase diagnostics. "
            "Individual-band phases near degeneracies should be interpreted with the overlap and residual gauges."
        ),
        "edt_window": {
            "k_left": k_left,
            "k0": k0,
            "k_right": k_right,
            "edt_width": k_right - k_left,
        },
        "counts": {
            "num_kpoints": int(len(k_coords)),
            "num_links": int(len(midpoints)),
            "inside_links": int(np.sum(inside_links)),
            "outside_links": int(np.sum(outside_links)),
        },
        "selected_bands_for_plot": selected_bands,
        "boundary_links": {
            "left": {
                "index": int(left_boundary_idx),
                "k_left": float(k_coords[left_boundary_idx]),
                "k_right": float(k_coords[left_boundary_idx + 1]),
                "k_mid": float(midpoints[left_boundary_idx]),
            },
            "right": {
                "index": int(right_boundary_idx),
                "k_left": float(k_coords[right_boundary_idx]),
                "k_right": float(k_coords[right_boundary_idx + 1]),
                "k_mid": float(midpoints[right_boundary_idx]),
            },
        },
        "aggregates": aggregate_stats,
        "band_resolved": band_stats,
        "parallel_transport_qc": {
            "max_abs_residual_phase_increment": float(np.max(np.abs(finite_residuals))) if finite_residuals.size else None,
            "mean_abs_residual_phase_increment": float(np.mean(np.abs(finite_residuals))) if finite_residuals.size else None,
            "min_overlap_abs": float(np.nanmin(overlap_abs)),
            "low_overlap_count_lt_1e_minus_6": int(np.sum(overlap_abs < 1e-6)),
        },
        "edt_phase_files": phase_file_status,
        "outputs": {
            "berry_phase_csv": str(phase_csv),
            "k0_centered_plot": str(plot_path),
        },
    }


def run(
    material: str,
    epw_root: Path,
    output_root: Path,
    agr_path: Path,
    summary_path: Path,
    selected_bands: list[int] | None = None,
    auto_band_count: int = 3,
) -> dict[str, Any]:
    output_dir = output_root / material
    output_dir.mkdir(parents=True, exist_ok=True)
    k_coords, k_frac, summary, datasets = load_agr_path(material, agr_path, summary_path)
    window = summary["nominal_result"]
    k_left = float(window["k_left"])
    k0 = float(window["k0"])
    k_right = float(window["k_right"])

    hr = load_hr(epw_root / material / "pwscf_hr.dat")
    evals, evecs = diagonalize_path(hr, k_frac)
    _transported, increments, residuals, overlap_abs = parallel_transport_phases(evecs)
    cumulative = cumulative_from_increments(increments)
    midpoints, inside_links, _outside_links = window_link_masks(k_coords, k_left, k_right)
    selected = select_bands(increments, inside_links, selected_bands, auto_band_count)

    phase_csv = output_dir / "berry_phase_on_agr_path.csv"
    write_phase_csv(phase_csv, material, k_coords, evals, increments, cumulative, residuals, overlap_abs, k_left, k_right)

    minus_label = f"G0.S{summary_dataset_id(summary, 'dataset_minus')}"
    plus_label = f"G0.S{summary_dataset_id(summary, 'dataset_plus')}"
    curvature_minus = np.array(datasets[minus_label].y, dtype=float)
    curvature_plus = np.array(datasets[plus_label].y, dtype=float)
    plot_path = output_dir / f"fig_{material.lower()}_k0_centered_berry_phase_comparison.png"
    plot_berry_phase_comparison(
        plot_path,
        k_coords,
        curvature_minus,
        curvature_plus,
        increments,
        cumulative,
        selected,
        k_left,
        k0,
        k_right,
    )

    phase_file_status = phase_files_status(Path("."))
    result = build_summary(
        material,
        k_coords,
        increments,
        residuals,
        overlap_abs,
        selected,
        k_left,
        k0,
        k_right,
        phase_csv,
        plot_path,
        phase_file_status,
    )
    write_json(output_dir / "berry_phase_edt_summary.json", result)
    return result


def parse_band_list(text: str | None) -> list[int] | None:
    if not text:
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", default="Na")
    parser.add_argument("--epw-root", type=Path, default=Path("epw_runs"))
    parser.add_argument("--output-root", type=Path, default=Path("epw_analysis"))
    parser.add_argument("--agr", type=Path, default=Path("data/na/Na-m5.agr"))
    parser.add_argument("--summary", type=Path, default=Path("data/processed/S1_Na_summary.json"))
    parser.add_argument("--selected-bands", default=None, help="Comma-separated 1-based bands to plot; default selects top EDT phase bands.")
    parser.add_argument("--auto-band-count", type=int, default=3)
    args = parser.parse_args(argv)

    result = run(
        args.material,
        args.epw_root,
        args.output_root,
        args.agr,
        args.summary,
        selected_bands=parse_band_list(args.selected_bands),
        auto_band_count=args.auto_band_count,
    )
    mean_stats = result["aggregates"]["band_mean"]
    print(f"{args.material}: wrote {result['outputs']['berry_phase_csv']}")
    print(f"{args.material}: wrote {result['outputs']['k0_centered_plot']}")
    print(f"{args.material}: inside mean-band phase = {mean_stats['inside_total_phase_rad']:.8g} rad")
    print(f"{args.material}: inside mean-band winding = {mean_stats['inside_winding_cycles']:.8g} cycles")
    print(f"{args.material}: PT residual max = {result['parallel_transport_qc']['max_abs_residual_phase_increment']:.3g} rad")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
