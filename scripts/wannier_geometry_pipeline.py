#!/usr/bin/env python3
"""Postprocess Wannier90 Hamiltonians onto AGR EDT paths.

The pipeline is intentionally conservative: it reads the Wannier90 QC files,
interpolates bands from ``pwscf_hr.dat`` on the AGR path coordinates, computes a
band-tracked overlap metric, and compares the resulting geometry proxy inside
and outside the stored EDT window.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - scipy is a project dependency.
    linear_sum_assignment = None


MATERIAL_INPUTS = {
    "Na": {
        "agr": Path("data/na/Na-m5.agr"),
        "summary": Path("data/processed/S1_Na_summary.json"),
    },
    "V": {
        "agr": Path("data/v/V-EP-F1.agr"),
        "summary": Path("data/processed/S2_V_summary.json"),
    },
    "Nb": {
        "agr": Path("data/nb/bands-Nb.agr"),
        "summary": None,
    },
    "Nb_m4": {
        "epw_material": "Nb",
        "agr": Path("data/nb/Nb-second-m4.agr"),
        "summary": Path("data/processed/Nb_m4_summary.json"),
    },
    "Nb_m5": {
        "epw_material": "Nb",
        "agr": Path("data/nb/Nb-seconddri_m5.agr"),
        "summary": Path("data/processed/Nb_m5_summary.json"),
    },
    "Nb_m6": {
        "epw_material": "Nb",
        "agr": Path("data/nb/Nb-seconddri_m6.agr"),
        "summary": Path("data/processed/Nb_m6_summary.json"),
    },
    "YBaCuO": {
        "agr": Path("data/ybacu3o7/YBa2Cu3O7_m33.agr"),
        "summary": Path("data/processed/S4_YBa2Cu3O7_summary.json"),
    },
    "K3C60": {
        "epw_material": "K3C60",
        "agr": Path("data/k3c60/tmp.agr"),
        "summary": Path("data/k3c60/k3c60_edt_window_summary.json"),
        "special_kpoints": {
            "x": np.array([0.0, 0.5, 0.5]),
            "l": np.array([0.5, 0.5, 0.5]),
            "w": np.array([0.25, 0.5, 0.75]),
        },
    },
}

SPECIAL_KPOINTS = {
    "g": np.array([0.0, 0.0, 0.0]),
    "gamma": np.array([0.0, 0.0, 0.0]),
    "\\g": np.array([0.0, 0.0, 0.0]),
    "\\gamma": np.array([0.0, 0.0, 0.0]),
    "h": np.array([0.5, -0.5, 0.5]),
    "n": np.array([0.0, 0.0, 0.5]),
    "p": np.array([0.25, 0.25, 0.25]),
    "x": np.array([0.0, 0.5, 0.0]),
    "y": np.array([0.5, 0.0, 0.0]),
    "z": np.array([0.0, 0.0, 0.5]),
    "s": np.array([0.5, 0.5, 0.0]),
    "r": np.array([0.5, 0.0, 0.5]),
    "t": np.array([0.0, 0.5, 0.5]),
    "u": np.array([0.5, 0.5, 0.5]),
}


@dataclass
class AgrDataset:
    label: str
    x: list[float]
    y: list[float]


@dataclass
class HrData:
    num_wann: int
    r_vectors: np.ndarray
    degeneracies: np.ndarray
    h_r: np.ndarray


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def parse_scalar(text: str, key: str) -> float | int | None:
    match = re.search(rf"^\s*{re.escape(key)}\s*=\s*([-+0-9.Ee]+)", text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    value = float(match.group(1))
    return int(value) if key in {"num_wann", "num_iter"} else value


def parse_win(path: Path) -> dict[str, Any]:
    text = read_text(path)
    projections: list[str] = []
    in_block = False
    for raw in text.splitlines():
        line = raw.strip()
        lower = line.lower()
        if lower.startswith("begin projections"):
            in_block = True
            continue
        if lower.startswith("end projections"):
            in_block = False
            continue
        if in_block and line and not line.startswith("#"):
            projections.append(line)

    return {
        "path": str(path),
        "projections": projections,
        "num_wann": parse_scalar(text, "num_wann"),
        "dis_win_min_ev": parse_scalar(text, "dis_win_min"),
        "dis_win_max_ev": parse_scalar(text, "dis_win_max"),
        "dis_froz_min_ev": parse_scalar(text, "dis_froz_min"),
        "dis_froz_max_ev": parse_scalar(text, "dis_froz_max"),
        "num_iter": parse_scalar(text, "num_iter"),
    }


def parse_wout(path: Path) -> dict[str, Any]:
    text = read_text(path)
    lower = text.lower()
    final_start = lower.rfind("final state")
    final_text = text[final_start:] if final_start != -1 else text
    spreads = [
        float(match.group(1))
        for match in re.finditer(
            r"WF\s+centre\s+and\s+spread\s+\d+\s+\([^)]*\)\s+([0-9.Ee+-]+)",
            final_text,
            re.IGNORECASE,
        )
    ]
    omega_match = list(
        re.finditer(
            r"Final\s+Spread\s+\(Ang\^2\)\s+Omega\s+Total\s+=\s+([0-9.Ee+-]+)",
            text,
            re.IGNORECASE,
        )
    )
    cycles = [int(match.group(1)) for match in re.finditer(r"Cycle:\s+(\d+)", text)]
    explicit_convergence = "convergence criteria satisfied" in lower
    max_cycle = max(cycles) if cycles else None

    return {
        "path": str(path),
        "normal_termination": "all done: wannier90 exiting" in lower,
        "converged": explicit_convergence,
        "explicit_convergence_message": explicit_convergence,
        "max_cycle": max_cycle,
        "final_spreads_ang2": spreads,
        "mean_spread_ang2": float(np.mean(spreads)) if spreads else None,
        "max_spread_ang2": float(np.max(spreads)) if spreads else None,
        "omega_total_ang2": float(omega_match[-1].group(1)) if omega_match else None,
    }


def count_wsvec_records(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "available": False, "records": 0, "max_degeneracy": None}
    lines = read_text(path).splitlines()
    records = 0
    max_deg = 0
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) == 5:
            try:
                [int(part) for part in parts]
                ndeg = int(lines[i + 1].strip())
            except (ValueError, IndexError):
                i += 1
                continue
            records += 1
            max_deg = max(max_deg, ndeg)
            i += 2 + ndeg
        else:
            i += 1
    return {"path": str(path), "available": True, "records": records, "max_degeneracy": max_deg}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_wannier_qc(material: str, material_dir: Path, output_dir: Path) -> dict[str, Any]:
    win_path = material_dir / "pwscf.win"
    wout_path = material_dir / "pwscf.wout"
    win = parse_win(win_path) if win_path.exists() else {"path": str(win_path), "available": False}
    wout = parse_wout(wout_path) if wout_path.exists() else {"path": str(wout_path), "available": False}
    wsvec = count_wsvec_records(material_dir / "pwscf_wsvec.dat")
    required_files_available = {
        name: (material_dir / name).exists()
        for name in (
            "pwscf.win",
            "pwscf.wout",
            "pwscf_hr.dat",
            "pwscf_wsvec.dat",
            "pwscf.nnkp",
            "pwscf.ukk",
            "pwscf.kgmap",
        )
    }
    warnings = []
    for name, available in required_files_available.items():
        if not available:
            warnings.append(f"missing {name}")
    if wout.get("available") is not False and not wout.get("explicit_convergence_message"):
        warnings.append("explicit Wannier90 convergence message not found; inspect wout cycles/spreads")
    qc = {
        "material": material,
        "win": win,
        "wout": wout,
        "wsvec": wsvec,
        "required_files_available": required_files_available,
        "all_required_files_present": all(required_files_available.values()),
        "warnings": warnings,
    }
    write_json(output_dir / "wannier_qc.json", qc)
    return qc


def parse_agr(path: Path) -> dict[str, AgrDataset]:
    datasets: dict[str, AgrDataset] = {}
    current = "unknown"
    xs: list[float] = []
    ys: list[float] = []

    def flush() -> None:
        nonlocal xs, ys
        if xs and ys:
            datasets[current] = AgrDataset(current, xs, ys)
        xs = []
        ys = []

    for raw in read_text(path).splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("@target"):
            flush()
            current = line.split()[-1]
            continue
        if line == "&":
            flush()
            continue
        if line.startswith("@") or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
            except ValueError:
                pass
    flush()
    return datasets


def parse_world_max(path: Path, fallback: float) -> float:
    for raw in read_text(path).splitlines():
        match = re.search(r"@\s+world\s+[-+0-9.Ee]+,\s*[-+0-9.Ee]+,\s*([-+0-9.Ee]+),", raw)
        if match:
            return float(match.group(1))
    return fallback


def parse_vertical_ticks(path: Path, x_max: float) -> list[float]:
    ticks = [0.0]
    for raw in read_text(path).splitlines():
        match = re.search(r"@\s+line\s+([-+0-9.Ee]+),\s*[-+0-9.Ee]+,\s*\1,\s*[-+0-9.Ee]+", raw)
        if match:
            x = float(match.group(1))
            if x > 1e-10:
                ticks.append(x)
    ticks.append(x_max)
    out: list[float] = []
    for tick in sorted(ticks):
        if not out or abs(tick - out[-1]) > 1e-8:
            out.append(tick)
    return out


def parse_path_labels(path: Path) -> list[str]:
    labels = []
    for raw in read_text(path).splitlines():
        match = re.search(r'@\s+string\s+def\s+"([^"]+)"', raw)
        if match:
            labels.append(match.group(1))
    return labels


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(read_text(path))


def canonical_label(label: str) -> str:
    return label.strip().replace("$", "").replace("{", "").replace("}", "").lower()


def path_labels_to_points(labels: list[str], special_kpoints: dict[str, np.ndarray] | None = None) -> list[np.ndarray]:
    kpoint_map = dict(SPECIAL_KPOINTS)
    if special_kpoints:
        kpoint_map.update({canonical_label(label): np.asarray(point, dtype=float) for label, point in special_kpoints.items()})
    points = []
    for label in labels:
        key = canonical_label(label)
        if key not in kpoint_map:
            raise ValueError(f"Unsupported path label {label!r}; known labels: {sorted(kpoint_map)}")
        points.append(kpoint_map[key])
    return points


def interpolate_fractional_path(
    k_coords: np.ndarray,
    ticks: list[float],
    labels: list[str],
    special_kpoints: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    if len(ticks) != len(labels):
        raise ValueError(f"Tick count {len(ticks)} does not match label count {len(labels)}")
    points = path_labels_to_points(labels, special_kpoints=special_kpoints)
    k_frac = np.zeros((len(k_coords), 3), dtype=float)
    for idx, k_coord in enumerate(k_coords):
        if k_coord <= ticks[0]:
            segment = 0
        elif k_coord >= ticks[-1]:
            segment = len(ticks) - 2
        else:
            segment = max(0, min(len(ticks) - 2, np.searchsorted(ticks, k_coord, side="right") - 1))
        left = ticks[segment]
        right = ticks[segment + 1]
        t = 0.0 if right == left else (float(k_coord) - left) / (right - left)
        start = points[segment]
        delta = points[segment + 1] - start
        delta = delta - np.rint(delta)
        k_frac[idx] = start + t * delta
    return k_frac


def load_hr(path: Path) -> HrData:
    lines = read_text(path).splitlines()
    if len(lines) < 3:
        raise ValueError(f"{path} is too short to be a Wannier90 hr.dat file")
    num_wann = int(lines[1].strip())
    nrpts = int(lines[2].strip())
    tokens: list[int] = []
    idx = 3
    while len(tokens) < nrpts:
        tokens.extend(int(part) for part in lines[idx].split())
        idx += 1
    degeneracies = np.array(tokens[:nrpts], dtype=float)
    r_vectors = np.zeros((nrpts, 3), dtype=float)
    h_r = np.zeros((nrpts, num_wann, num_wann), dtype=complex)
    for ir in range(nrpts):
        for _ in range(num_wann * num_wann):
            parts = lines[idx].split()
            idx += 1
            r1, r2, r3 = (int(parts[0]), int(parts[1]), int(parts[2]))
            m, n = int(parts[3]) - 1, int(parts[4]) - 1
            value = float(parts[5]) + 1j * float(parts[6])
            r_vectors[ir] = (r1, r2, r3)
            h_r[ir, m, n] = value
    return HrData(num_wann=num_wann, r_vectors=r_vectors, degeneracies=degeneracies, h_r=h_r)


def hamiltonian_at_k(hr: HrData, k_frac: np.ndarray) -> np.ndarray:
    phase = np.exp(2j * np.pi * (hr.r_vectors @ k_frac)) / hr.degeneracies
    h_k = np.tensordot(phase, hr.h_r, axes=(0, 0))
    return 0.5 * (h_k + h_k.conj().T)


def diagonalize_path(hr: HrData, k_frac: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    evals = np.zeros((len(k_frac), hr.num_wann), dtype=float)
    evecs = np.zeros((len(k_frac), hr.num_wann, hr.num_wann), dtype=complex)
    for idx, kpt in enumerate(k_frac):
        w, v = np.linalg.eigh(hamiltonian_at_k(hr, kpt))
        if idx > 0:
            overlap = np.abs(evecs[idx - 1].conj().T @ v)
            if linear_sum_assignment is not None:
                rows, cols = linear_sum_assignment(-overlap)
                order = np.zeros(hr.num_wann, dtype=int)
                order[rows] = cols
            else:
                order = greedy_assignment(overlap)
            w = w[order]
            v = v[:, order]
        evals[idx] = w
        evecs[idx] = v
    return evals, evecs


def greedy_assignment(overlap: np.ndarray) -> np.ndarray:
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


def quantum_metric(k_coords: np.ndarray, evecs: np.ndarray) -> np.ndarray:
    n_k, _num_wann, n_band = evecs.shape
    metric = np.full((n_k, n_band), np.nan, dtype=float)
    for idx in range(n_k - 1):
        dk = float(k_coords[idx + 1] - k_coords[idx])
        if abs(dk) < 1e-14:
            continue
        overlaps = np.abs(np.sum(evecs[idx].conj() * evecs[idx + 1], axis=0))
        metric[idx] = np.maximum(0.0, 1.0 - overlaps**2) / (dk * dk)
    if n_k >= 2:
        metric[-1] = metric[-2]
    return metric


def finite_mean(values: list[float] | np.ndarray) -> float | None:
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else None


def summary_dataset_id(summary: dict[str, Any], key: str) -> int:
    if key in summary:
        return int(summary[key])
    selected_pair = summary.get("selected_pair") or {}
    if key in selected_pair:
        return int(selected_pair[key])
    raise KeyError(f"Summary is missing {key!r}")


def summary_selected_pair(summary: dict[str, Any]) -> dict[str, Any]:
    pair = dict(summary.get("selected_pair") or {})
    for key in ("dataset_minus", "dataset_plus"):
        if key not in pair and key in summary:
            pair[key] = summary[key]
    if "band_minus_label" not in pair and "dataset_minus" in pair:
        pair["band_minus_label"] = f"Band {int(pair['dataset_minus']) + 1}"
    if "band_plus_label" not in pair and "dataset_plus" in pair:
        pair["band_plus_label"] = f"Band {int(pair['dataset_plus']) + 1}"
    return pair


def write_band_metric_csv(
    path: Path,
    material: str,
    k_coords: np.ndarray,
    k_frac: np.ndarray,
    evals: np.ndarray,
    metric: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "material",
            "k_index",
            "k_coord",
            "kx_frac",
            "ky_frac",
            "kz_frac",
            "band",
            "energy",
            "quantum_metric_parallel",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for k_index, k_coord in enumerate(k_coords):
            for band in range(evals.shape[1]):
                writer.writerow(
                    {
                        "material": material,
                        "k_index": k_index,
                        "k_coord": f"{k_coord:.12g}",
                        "kx_frac": f"{k_frac[k_index, 0]:.12g}",
                        "ky_frac": f"{k_frac[k_index, 1]:.12g}",
                        "kz_frac": f"{k_frac[k_index, 2]:.12g}",
                        "band": band + 1,
                        "energy": f"{evals[k_index, band]:.12g}",
                        "quantum_metric_parallel": f"{metric[k_index, band]:.12g}",
                    }
                )


def write_edt_geometry(
    material: str,
    output_dir: Path,
    summary: dict[str, Any],
    datasets: dict[str, AgrDataset],
    k_coords: np.ndarray,
    evals: np.ndarray,
    metric: np.ndarray,
) -> None:
    nominal = summary["nominal_result"]
    k_left = float(nominal["k_left"])
    k_right = float(nominal["k_right"])
    inside = (k_coords >= k_left) & (k_coords <= k_right)
    outside = ~inside
    rows = []
    for band in range(metric.shape[1]):
        inside_mean = finite_mean(metric[inside, band])
        outside_mean = finite_mean(metric[outside, band])
        rows.append(
            {
                "material": material,
                "band": band + 1,
                "edt_k_left": k_left,
                "edt_k_right": k_right,
                "inside_count": int(np.sum(inside)),
                "outside_count": int(np.sum(outside)),
                "inside_quantum_metric_parallel_mean": inside_mean,
                "outside_quantum_metric_parallel_mean": outside_mean,
                "inside_outside_metric_ratio": (
                    inside_mean / outside_mean if inside_mean is not None and outside_mean not in (None, 0.0) else None
                ),
                "inside_energy_mean": finite_mean(evals[inside, band]),
                "outside_energy_mean": finite_mean(evals[outside, band]),
            }
        )

    comparison_path = output_dir / "edt_geometry_comparison.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: csv_value(v) for k, v in row.items()} for row in rows)

    dataset_minus = f"G0.S{summary_dataset_id(summary, 'dataset_minus')}"
    dataset_plus = f"G0.S{summary_dataset_id(summary, 'dataset_plus')}"
    curvature_stats = {}
    for label in (dataset_minus, dataset_plus):
        dataset = datasets.get(label)
        if dataset:
            x = np.array(dataset.x, dtype=float)
            y = np.array(dataset.y, dtype=float)
            mask = (x >= k_left) & (x <= k_right)
            curvature_stats[label] = {
                "inside_mean": finite_mean(y[mask]),
                "outside_mean": finite_mean(y[~mask]),
                "inside_count": int(np.sum(mask)),
                "outside_count": int(np.sum(~mask)),
            }

    inside_all = finite_mean(metric[inside, :].ravel())
    outside_all = finite_mean(metric[outside, :].ravel())
    write_json(
        output_dir / "edt_geometry_summary.json",
        {
            "material": material,
            "status": "completed",
            "mode_label": summary.get("mode_label"),
            "selected_pair": summary_selected_pair(summary),
            "edt_window": {
                "k0": nominal.get("k0"),
                "k_left": k_left,
                "k_right": k_right,
                "edt_width": nominal.get("edt_width"),
                "delta_k_EDT": nominal.get("edt_width"),
            },
            "path_labels": summary.get("path_labels", []),
            "num_kpoints": int(len(k_coords)),
            "num_bands": int(metric.shape[1]),
            "inside_quantum_metric_parallel_mean_all_bands": inside_all,
            "outside_quantum_metric_parallel_mean_all_bands": outside_all,
            "inside_outside_metric_ratio_all_bands": (
                inside_all / outside_all if inside_all is not None and outside_all not in (None, 0.0) else None
            ),
            "curvature_datasets": curvature_stats,
            "comparison_csv": str(comparison_path),
        },
    )


def write_edt_only_geometry(
    material: str,
    output_dir: Path,
    summary: dict[str, Any],
    datasets: dict[str, AgrDataset],
    k_coords: np.ndarray,
    reason: str,
    qc: dict[str, Any],
) -> None:
    nominal = summary["nominal_result"]
    k_left = float(nominal["k_left"])
    k_right = float(nominal["k_right"])
    k0 = float(nominal["k0"])
    inside = (k_coords >= k_left) & (k_coords <= k_right)
    dataset_minus = f"G0.S{summary_dataset_id(summary, 'dataset_minus')}"
    dataset_plus = f"G0.S{summary_dataset_id(summary, 'dataset_plus')}"
    minus = datasets.get(dataset_minus)
    plus = datasets.get(dataset_plus)
    if minus is None or plus is None:
        raise ValueError(f"Missing K3C60 curvature datasets {dataset_minus!r}/{dataset_plus!r}")
    plus_y = np.interp(k_coords, np.array(plus.x, dtype=float), np.array(plus.y, dtype=float))
    minus_y = np.interp(k_coords, np.array(minus.x, dtype=float), np.array(minus.y, dtype=float))

    comparison_path = output_dir / "edt_geometry_comparison.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "material",
            "mode_label",
            "k_index",
            "k_coord",
            "kappa_minus_dataset",
            "kappa_plus_dataset",
            "kappa_minus",
            "kappa_plus",
            "k_left",
            "k0",
            "k_right",
            "delta_k_EDT",
            "inside_edt_window",
            "quantum_metric_parallel",
            "notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, k_coord in enumerate(k_coords):
            writer.writerow(
                {
                    "material": material,
                    "mode_label": summary.get("mode_label"),
                    "k_index": idx,
                    "k_coord": csv_value(float(k_coord)),
                    "kappa_minus_dataset": dataset_minus,
                    "kappa_plus_dataset": dataset_plus,
                    "kappa_minus": csv_value(float(minus_y[idx])),
                    "kappa_plus": csv_value(float(plus_y[idx])),
                    "k_left": csv_value(k_left),
                    "k0": csv_value(k0),
                    "k_right": csv_value(k_right),
                    "delta_k_EDT": csv_value(float(nominal.get("edt_width") or (k_right - k_left))),
                    "inside_edt_window": bool(inside[idx]),
                    "quantum_metric_parallel": "",
                    "notes": "EDT-only row; Wannier metric awaits pwscf_hr.dat",
                }
            )

    curvature_stats = {}
    for label, values in ((dataset_minus, minus_y), (dataset_plus, plus_y)):
        curvature_stats[label] = {
            "inside_mean": finite_mean(values[inside]),
            "outside_mean": finite_mean(values[~inside]),
            "inside_abs_sum": float(np.sum(np.abs(values[inside]))),
            "outside_abs_sum": float(np.sum(np.abs(values[~inside]))),
            "inside_count": int(np.sum(inside)),
            "outside_count": int(np.sum(~inside)),
        }

    write_json(
        output_dir / "edt_geometry_summary.json",
        {
            "material": material,
            "status": "edt_only_missing_wannier",
            "reason": reason,
            "mode_label": summary.get("mode_label"),
            "mode_frequency_cm^-1": summary.get("mode_frequency_cm^-1"),
            "selected_pair": summary_selected_pair(summary),
            "edt_window": {
                "k0": k0,
                "k_left": k_left,
                "k_right": k_right,
                "edt_width": nominal.get("edt_width"),
                "delta_k_EDT": nominal.get("edt_width"),
            },
            "path_labels": summary.get("path_labels", []),
            "num_kpoints": int(len(k_coords)),
            "num_bands": None,
            "inside_quantum_metric_parallel_mean_all_bands": None,
            "outside_quantum_metric_parallel_mean_all_bands": None,
            "inside_outside_metric_ratio_all_bands": None,
            "curvature_datasets": curvature_stats,
            "comparison_csv": str(comparison_path),
            "wannier_qc_json": str(output_dir / "wannier_qc.json"),
            "required_wannier_files_available": qc.get("required_files_available", {}),
            "warnings": [
                "Wannier Hamiltonian outputs are not yet available locally; bands, path metric, Berry phase, and surfaces were not computed.",
                "No EPC comparison is made without resolved EPW EPC/lambda/linewidth output.",
            ],
        },
    )


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.12g}"
    return str(value)


def plot_wannier_bands(
    output: Path,
    material: str,
    k_coords: np.ndarray,
    evals: np.ndarray,
    summary: dict[str, Any],
) -> None:
    nominal = summary["nominal_result"]
    k_left = float(nominal["k_left"])
    k_right = float(nominal["k_right"])
    k0 = float(nominal["k0"])
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for band in range(evals.shape[1]):
        ax.plot(k_coords, evals[:, band], color="0.15", lw=0.55, alpha=0.55)
    ax.axvspan(k_left, k_right, color="tab:orange", alpha=0.16)
    ax.axvline(k0, color="tab:orange", lw=1.1)
    ax.set_xlabel("AGR path coordinate")
    ax.set_ylabel("Wannier energy (eV)")
    ax.set_title(f"{material} Wannier bands on AGR path")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_quantum_metric(
    output: Path,
    material: str,
    k_coords: np.ndarray,
    metric: np.ndarray,
    summary: dict[str, Any],
) -> None:
    nominal = summary["nominal_result"]
    k_left = float(nominal["k_left"])
    k_right = float(nominal["k_right"])
    k0 = float(nominal["k0"])
    metric_sum = np.nansum(metric, axis=1)
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for band in range(metric.shape[1]):
        ax.plot(k_coords, metric[:, band], lw=0.45, alpha=0.22)
    ax.plot(k_coords, metric_sum, color="black", lw=1.35, label="sum over bands")
    ax.axvspan(k_left, k_right, color="tab:orange", alpha=0.16)
    ax.axvline(k0, color="tab:orange", lw=1.1)
    ax.set_xlabel("AGR path coordinate")
    ax.set_ylabel("g_parallel")
    ax.set_title(f"{material} path-projected quantum metric")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_metric_ratios(
    output: Path,
    material: str,
    k_coords: np.ndarray,
    metric: np.ndarray,
    summary: dict[str, Any],
) -> None:
    nominal = summary["nominal_result"]
    k_left = float(nominal["k_left"])
    k_right = float(nominal["k_right"])
    inside = (k_coords >= k_left) & (k_coords <= k_right)
    ratios = []
    for band in range(metric.shape[1]):
        inside_mean = finite_mean(metric[inside, band])
        outside_mean = finite_mean(metric[~inside, band])
        ratios.append(inside_mean / outside_mean if inside_mean is not None and outside_mean not in (None, 0.0) else math.nan)
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.bar(np.arange(1, metric.shape[1] + 1), ratios, color="#4f8aa6", width=0.85)
    ax.axhline(1.0, color="0.25", lw=1.0, ls="--")
    ax.set_xlabel("Wannier band")
    ax.set_ylabel("inside/outside metric ratio")
    ax.set_title(f"{material} band-resolved EDT metric ratios")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def write_path_plots(
    output_dir: Path,
    material: str,
    k_coords: np.ndarray,
    evals: np.ndarray,
    metric: np.ndarray,
    summary: dict[str, Any],
) -> dict[str, str]:
    stem = material.lower()
    band_plot = output_dir / f"fig_{stem}_wannier_bands_on_agr_path.png"
    metric_plot = output_dir / f"fig_{stem}_quantum_metric_parallel.png"
    ratio_plot = output_dir / f"fig_{stem}_metric_inside_outside_ratios.png"
    plot_wannier_bands(band_plot, material, k_coords, evals, summary)
    plot_quantum_metric(metric_plot, material, k_coords, metric, summary)
    plot_metric_ratios(ratio_plot, material, k_coords, metric, summary)
    return {
        "wannier_bands": str(band_plot),
        "quantum_metric_parallel": str(metric_plot),
        "metric_inside_outside_ratios": str(ratio_plot),
    }


def run_material(material: str, epw_root: Path, output_root: Path) -> dict[str, Any]:
    inputs = MATERIAL_INPUTS.get(material, {})
    material_dir = epw_root / inputs.get("epw_material", material)
    output_dir = output_root / material
    output_dir.mkdir(parents=True, exist_ok=True)
    qc = write_wannier_qc(material, material_dir, output_dir)

    agr_path = inputs.get("agr")
    summary_path = inputs.get("summary")
    if agr_path is None or summary_path is None or not agr_path.exists() or not summary_path.exists():
        write_json(
            output_dir / "edt_geometry_summary.json",
            {
                "material": material,
                "status": "skipped_missing_agr_or_edt_summary",
                "reason": "No AGR curvature file and EDT summary are available in this checkout.",
                "wannier_qc_json": str(output_dir / "wannier_qc.json"),
            },
        )
        return {"material": material, "status": "qc_only_missing_agr", "qc": qc}

    summary = load_summary(summary_path)
    datasets = parse_agr(agr_path)
    grid_label = f"G0.S{summary_dataset_id(summary, 'dataset_minus')}"
    if grid_label not in datasets:
        raise ValueError(f"{agr_path} does not contain expected dataset {grid_label}")
    k_coords = np.array(datasets[grid_label].x, dtype=float)
    x_max = parse_world_max(agr_path, float(np.max(k_coords)))
    ticks = parse_vertical_ticks(agr_path, x_max)
    labels = summary.get("path_labels", [])
    agr_labels = parse_path_labels(agr_path)
    if len(agr_labels) == len(ticks) and len(labels) != len(ticks):
        labels = agr_labels
    special_kpoints = inputs.get("special_kpoints")
    k_frac = interpolate_fractional_path(k_coords, ticks, labels, special_kpoints=special_kpoints)

    hr_path = material_dir / "pwscf_hr.dat"
    if not hr_path.exists():
        reason = f"Missing Wannier Hamiltonian file: {hr_path}"
        write_edt_only_geometry(material, output_dir, summary, datasets, k_coords, reason, qc)
        return {
            "material": material,
            "status": "edt_only_missing_wannier",
            "num_kpoints": int(len(k_coords)),
            "qc": qc,
            "reason": reason,
        }

    hr = load_hr(hr_path)
    evals, evecs = diagonalize_path(hr, k_frac)
    metric = quantum_metric(k_coords, evecs)

    write_band_metric_csv(output_dir / "wannier_bands_on_agr_path.csv", material, k_coords, k_frac, evals, metric)
    write_band_metric_csv(output_dir / "quantum_metric_on_agr_path.csv", material, k_coords, k_frac, evals, metric)
    write_edt_geometry(material, output_dir, summary, datasets, k_coords, evals, metric)
    plots = write_path_plots(output_dir, material, k_coords, evals, metric, summary)
    return {
        "material": material,
        "status": "completed",
        "num_kpoints": int(len(k_coords)),
        "num_bands": int(hr.num_wann),
        "qc": qc,
        "plots": plots,
    }


def merge_pipeline_results(summary_path: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    existing_results: list[dict[str, Any]] = []
    if summary_path.exists():
        try:
            existing = json.loads(read_text(summary_path))
            existing_results = list(existing.get("results") or [])
        except (json.JSONDecodeError, TypeError):
            existing_results = []
    merged = {row.get("material"): row for row in existing_results if row.get("material")}
    for row in results:
        if row.get("material"):
            merged[row["material"]] = row
    return {"results": [merged[key] for key in sorted(merged)]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epw-root", type=Path, default=Path("epw_runs"))
    parser.add_argument("--output-root", type=Path, default=Path("epw_analysis"))
    parser.add_argument("--materials", nargs="*", default=["Na", "V", "Nb"])
    args = parser.parse_args(argv)

    results = []
    for material in args.materials:
        result = run_material(material, args.epw_root, args.output_root)
        results.append(result)
        print(
            f"{material}: {result['status']}"
            + (f" kpoints={result.get('num_kpoints')} bands={result.get('num_bands')}" if "num_kpoints" in result else "")
        )
    summary_path = args.output_root / "wannier_geometry_pipeline_summary.json"
    write_json(summary_path, merge_pipeline_results(summary_path, results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
