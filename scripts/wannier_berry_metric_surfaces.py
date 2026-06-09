#!/usr/bin/env python3
"""Compute Berry-curvature and quantum-metric surfaces from a Wannier Hamiltonian."""

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
    finite_mean,
    hamiltonian_at_k,
    interpolate_fractional_path,
    load_hr,
    load_summary,
    parse_agr,
    parse_vertical_ticks,
    parse_world_max,
    summary_dataset_id,
    write_json,
)

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - scipy is a project dependency.
    linear_sum_assignment = None


DEFAULT_INPUTS = {
    "Na": {
        "agr": Path("data/na/Na-m5.agr"),
        "summary": Path("data/processed/S1_Na_summary.json"),
    },
    "K3C60": {
        "agr": Path("data/k3c60/tmp.agr"),
        "summary": Path("data/k3c60/k3c60_edt_window_summary.json"),
        "special_kpoints": {
            "x": np.array([0.0, 0.5, 0.5]),
            "l": np.array([0.5, 0.5, 0.5]),
            "w": np.array([0.25, 0.5, 0.75]),
        },
    },
}


def assignment_order(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    overlap = np.abs(previous.conj().T @ current)
    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(-overlap)
        order = np.zeros(overlap.shape[0], dtype=int)
        order[rows] = cols
        return order
    remaining_rows = set(range(overlap.shape[0]))
    remaining_cols = set(range(overlap.shape[1]))
    order = np.zeros(overlap.shape[0], dtype=int)
    for _ in range(overlap.shape[0]):
        row, col = max(
            ((r, c) for r in remaining_rows for c in remaining_cols),
            key=lambda item: overlap[item[0], item[1]],
        )
        order[row] = col
        remaining_rows.remove(row)
        remaining_cols.remove(col)
    return order


def diagonalize_grid(hr, kx: np.ndarray, ky: np.ndarray, kz: float) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = len(kx), len(ky)
    evals = np.zeros((nx, ny, hr.num_wann), dtype=float)
    evecs = np.zeros((nx, ny, hr.num_wann, hr.num_wann), dtype=complex)
    for ix, x in enumerate(kx):
        for iy, y in enumerate(ky):
            w, v = np.linalg.eigh(hamiltonian_at_k(hr, np.array([x, y, kz], dtype=float)))
            if ix == 0 and iy > 0:
                order = assignment_order(evecs[ix, iy - 1], v)
                w = w[order]
                v = v[:, order]
            elif ix > 0:
                order = assignment_order(evecs[ix - 1, iy], v)
                w = w[order]
                v = v[:, order]
            evals[ix, iy] = w
            evecs[ix, iy] = v
    return evals, evecs


def overlap_band(evecs: np.ndarray, a: tuple[int, int], b: tuple[int, int], band: int) -> complex:
    ua = evecs[a[0], a[1], :, band]
    ub = evecs[b[0], b[1], :, band]
    return np.vdot(ua, ub)


def compute_berry_metric(evecs: np.ndarray, dkx: float, dky: float) -> tuple[np.ndarray, np.ndarray]:
    nx, ny, _num_wann, nbands = evecs.shape
    berry = np.full((nx, ny, nbands), np.nan, dtype=float)
    metric_trace = np.full((nx, ny, nbands), np.nan, dtype=float)
    area = dkx * dky
    for ix in range(nx):
        for iy in range(ny):
            for band in range(nbands):
                gx_values = []
                gy_values = []
                if ix + 1 < nx:
                    ox = abs(overlap_band(evecs, (ix, iy), (ix + 1, iy), band))
                    gx_values.append(max(0.0, 1.0 - ox * ox) / (dkx * dkx))
                if ix > 0:
                    ox = abs(overlap_band(evecs, (ix, iy), (ix - 1, iy), band))
                    gx_values.append(max(0.0, 1.0 - ox * ox) / (dkx * dkx))
                if iy + 1 < ny:
                    oy = abs(overlap_band(evecs, (ix, iy), (ix, iy + 1), band))
                    gy_values.append(max(0.0, 1.0 - oy * oy) / (dky * dky))
                if iy > 0:
                    oy = abs(overlap_band(evecs, (ix, iy), (ix, iy - 1), band))
                    gy_values.append(max(0.0, 1.0 - oy * oy) / (dky * dky))
                gx = float(np.mean(gx_values)) if gx_values else math.nan
                gy = float(np.mean(gy_values)) if gy_values else math.nan
                metric_trace[ix, iy, band] = gx + gy if math.isfinite(gx) and math.isfinite(gy) else math.nan
                if ix + 1 < nx and iy + 1 < ny:
                    loop = (
                        overlap_band(evecs, (ix, iy), (ix + 1, iy), band)
                        * overlap_band(evecs, (ix + 1, iy), (ix + 1, iy + 1), band)
                        * overlap_band(evecs, (ix + 1, iy + 1), (ix, iy + 1), band)
                        * overlap_band(evecs, (ix, iy + 1), (ix, iy), band)
                    )
                    berry[ix, iy, band] = -float(np.angle(loop)) / area
    if nx > 1:
        berry[-1, :, :] = berry[-2, :, :]
    if ny > 1:
        berry[:, -1, :] = berry[:, -2, :]
    return berry, metric_trace


def load_agr_path(
    material: str,
    agr_path: Path,
    summary_path: Path,
    special_kpoints: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    summary = load_summary(summary_path)
    datasets = parse_agr(agr_path)
    grid_label = f"G0.S{summary_dataset_id(summary, 'dataset_minus')}"
    if grid_label not in datasets:
        raise ValueError(f"{agr_path} does not contain expected dataset {grid_label}")
    k_coords = np.array(datasets[grid_label].x, dtype=float)
    x_max = parse_world_max(agr_path, float(np.max(k_coords)))
    ticks = parse_vertical_ticks(agr_path, x_max)
    k_frac = interpolate_fractional_path(k_coords, ticks, summary["path_labels"], special_kpoints=special_kpoints)
    return k_coords, k_frac, summary


def nearest_surface_values(
    k_frac: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    berry_abs_sum: np.ndarray,
    metric_sum: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    berry_values = np.zeros(len(k_frac), dtype=float)
    metric_values = np.zeros(len(k_frac), dtype=float)
    for idx, kpt in enumerate(k_frac):
        ix = int(np.argmin(np.abs(kx - kpt[0])))
        iy = int(np.argmin(np.abs(ky - kpt[1])))
        berry_values[idx] = berry_abs_sum[ix, iy]
        metric_values[idx] = metric_sum[ix, iy]
    return berry_values, metric_values


def hotspot_stats(values: np.ndarray, inside: np.ndarray, threshold: float) -> dict[str, Any]:
    outside = ~inside
    inside_values = values[inside]
    outside_values = values[outside]
    inside_hot = inside_values >= threshold if inside_values.size else np.array([], dtype=bool)
    outside_hot = outside_values >= threshold if outside_values.size else np.array([], dtype=bool)
    return {
        "inside_mean": finite_mean(inside_values),
        "outside_mean": finite_mean(outside_values),
        "inside_max": float(np.nanmax(inside_values)) if inside_values.size else None,
        "outside_max": float(np.nanmax(outside_values)) if outside_values.size else None,
        "inside_hotspot_fraction": float(np.mean(inside_hot)) if inside_hot.size else None,
        "outside_hotspot_fraction": float(np.mean(outside_hot)) if outside_hot.size else None,
        "hotspot_threshold": float(threshold),
    }


def write_surface_csv(
    path: Path,
    material: str,
    kx: np.ndarray,
    ky: np.ndarray,
    kz: float,
    berry: np.ndarray,
    metric_trace: np.ndarray,
    evals: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "material",
            "kx_frac",
            "ky_frac",
            "kz_frac",
            "band",
            "energy",
            "berry_curvature_xy",
            "quantum_metric_trace",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for ix, x in enumerate(kx):
            for iy, y in enumerate(ky):
                for band in range(berry.shape[2]):
                    writer.writerow(
                        {
                            "material": material,
                            "kx_frac": f"{x:.12g}",
                            "ky_frac": f"{y:.12g}",
                            "kz_frac": f"{kz:.12g}",
                            "band": band + 1,
                            "energy": f"{evals[ix, iy, band]:.12g}",
                            "berry_curvature_xy": f"{berry[ix, iy, band]:.12g}",
                            "quantum_metric_trace": f"{metric_trace[ix, iy, band]:.12g}",
                        }
                    )


def plot_surface(
    output: Path,
    title: str,
    values: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    k_frac: np.ndarray,
    inside: np.ndarray,
    cmap: str,
    label: str,
    log_norm: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    plot_values = np.asarray(values.T, dtype=float)
    if log_norm:
        plot_values = np.log10(np.maximum(plot_values, 1e-12))
        label = f"log10 {label}"
    mesh = ax.pcolormesh(kx, ky, plot_values, shading="auto", cmap=cmap)
    fig.colorbar(mesh, ax=ax, label=label)
    ax.plot(k_frac[:, 0], k_frac[:, 1], color="white", lw=1.3, alpha=0.9, label="AGR path projection")
    ax.plot(k_frac[inside, 0], k_frac[inside, 1], color="black", lw=2.4, label="EDT window segment")
    ax.scatter(k_frac[inside, 0], k_frac[inside, 1], s=8, color="black", alpha=0.65)
    ax.set_xlabel("kx fractional")
    ax.set_ylabel("ky fractional")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def run(
    material: str,
    epw_root: Path,
    output_root: Path,
    agr_path: Path,
    summary_path: Path,
    grid_size: int,
    kz: float,
    special_kpoints: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    output_dir = output_root / material
    output_dir.mkdir(parents=True, exist_ok=True)
    hr = load_hr(epw_root / material / "pwscf_hr.dat")
    kx = np.linspace(-0.5, 0.5, grid_size)
    ky = np.linspace(-0.5, 0.5, grid_size)
    evals, evecs = diagonalize_grid(hr, kx, ky, kz)
    dkx = float(kx[1] - kx[0])
    dky = float(ky[1] - ky[0])
    berry, metric_trace = compute_berry_metric(evecs, dkx, dky)
    berry_abs_sum = np.nansum(np.abs(berry), axis=2)
    metric_sum = np.nansum(metric_trace, axis=2)

    k_coords, k_frac, summary = load_agr_path(material, agr_path, summary_path, special_kpoints=special_kpoints)
    window = summary["nominal_result"]
    inside = (k_coords >= float(window["k_left"])) & (k_coords <= float(window["k_right"]))
    path_berry, path_metric = nearest_surface_values(k_frac, kx, ky, berry_abs_sum, metric_sum)

    berry_threshold = float(np.nanpercentile(berry_abs_sum, 90))
    metric_threshold = float(np.nanpercentile(metric_sum, 90))
    surface_csv = output_dir / "berry_metric_surface_xy_kz0.csv"
    write_surface_csv(surface_csv, material, kx, ky, kz, berry, metric_trace, evals)

    stem = material.lower()
    berry_plot = output_dir / f"fig_{stem}_berry_curvature_surface.png"
    metric_plot = output_dir / f"fig_{stem}_quantum_metric_trace_surface.png"
    plot_surface(
        berry_plot,
        f"{material} | Berry curvature surface | kz={kz:g}",
        berry_abs_sum,
        kx,
        ky,
        k_frac,
        inside,
        cmap="magma",
        label="sum_bands |Omega_xy|",
        log_norm=True,
    )
    plot_surface(
        metric_plot,
        f"{material} | Quantum metric trace surface | kz={kz:g}",
        metric_sum,
        kx,
        ky,
        k_frac,
        inside,
        cmap="viridis",
        label="sum_bands g_tr",
        log_norm=True,
    )

    berry_stats = hotspot_stats(path_berry, inside, berry_threshold)
    metric_stats = hotspot_stats(path_metric, inside, metric_threshold)
    summary_out = {
        "material": material,
        "status": "completed",
        "plane": "kx-ky",
        "kz_frac": kz,
        "grid_size": grid_size,
        "surface_csv": str(surface_csv),
        "plots": {
            "berry_abs_sum": str(berry_plot),
            "metric_trace_sum": str(metric_plot),
        },
        "edt_window": {
            "k0": window["k0"],
            "k_left": window["k_left"],
            "k_right": window["k_right"],
            "edt_width": window["edt_width"],
        },
        "projection_note": "AGR path is projected onto the kx-ky plane; kz along the AGR path is not represented in this 2D slice.",
        "berry_abs_sum_path_projected": berry_stats,
        "metric_trace_sum_path_projected": metric_stats,
        "interpretation": {
            "berry_hotspot_overlap": classify_overlap(berry_stats),
            "metric_hotspot_overlap": classify_overlap(metric_stats),
        },
        "epc_claim": "No EPC comparison is made here; resolved EPW EPC/lambda/linewidth outputs are required for EPC claims.",
    }
    write_json(output_dir / "berry_metric_surface_summary.json", summary_out)
    return summary_out


def classify_overlap(stats: dict[str, Any]) -> str:
    inside = stats.get("inside_hotspot_fraction")
    outside = stats.get("outside_hotspot_fraction")
    if inside is None or outside is None:
        return "insufficient path samples"
    if inside > outside:
        return "EDT projected segment has higher hotspot overlap than outside path"
    if inside < outside:
        return "EDT projected segment has lower hotspot overlap than outside path"
    return "EDT projected segment has similar hotspot overlap to outside path"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", default="Na")
    parser.add_argument("--epw-root", type=Path, default=Path("epw_runs"))
    parser.add_argument("--output-root", type=Path, default=Path("epw_analysis"))
    parser.add_argument("--agr", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--kz", type=float, default=0.0)
    args = parser.parse_args(argv)

    defaults = DEFAULT_INPUTS.get(args.material, {})
    agr_path = args.agr or defaults.get("agr")
    summary_path = args.summary or defaults.get("summary")
    if agr_path is None or summary_path is None:
        raise SystemExit(f"No default AGR/summary configured for {args.material}; pass --agr and --summary.")
    result = run(
        args.material,
        args.epw_root,
        args.output_root,
        agr_path,
        summary_path,
        args.grid_size,
        args.kz,
        special_kpoints=defaults.get("special_kpoints"),
    )
    print(f"{args.material}: wrote {result['surface_csv']}")
    print(f"{args.material}: wrote {result['plots']['berry_abs_sum']}")
    print(f"{args.material}: wrote {result['plots']['metric_trace_sum']}")
    print(f"{args.material}: Berry overlap: {result['interpretation']['berry_hotspot_overlap']}")
    print(f"{args.material}: Metric overlap: {result['interpretation']['metric_hotspot_overlap']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
