from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


@dataclass(frozen=True)
class QuadraticPairFit:
    kappa0: float
    a: float
    k0: float
    kappa_plus_fit: np.ndarray
    kappa_minus_fit: np.ndarray
    success: bool
    cost: float


def load_band_data(path: str | Path) -> pd.DataFrame:
    """Load band or curvature data from a common tabular format.

    Supported formats are CSV/TSV/TXT, JSON records, NPY, and NPZ.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".csv", ".tsv", ".txt", ".dat"}:
        sep = "\t" if suffix == ".tsv" else None
        return pd.read_csv(path, sep=sep, engine="python")
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".npy":
        data = np.load(path, allow_pickle=False)
        return pd.DataFrame(np.asarray(data))
    if suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        return pd.DataFrame({key: data[key] for key in data.files})
    raise ValueError(f"Unsupported band-data format: {path.suffix}")


def compute_curvature_from_displacements(
    E_minus: np.ndarray | list[float],
    E_zero: np.ndarray | list[float],
    E_plus: np.ndarray | list[float],
    dQ: float,
) -> np.ndarray:
    """Compute curvature from centered finite differences in normal-mode displacement."""
    if dQ == 0.0:
        raise ValueError("dQ must be nonzero.")

    e_minus = np.asarray(E_minus, dtype=float)
    e_zero = np.asarray(E_zero, dtype=float)
    e_plus = np.asarray(E_plus, dtype=float)
    if e_minus.shape != e_zero.shape or e_zero.shape != e_plus.shape:
        raise ValueError("Displacement energies must have matching shapes.")
    return (e_plus - 2.0 * e_zero + e_minus) / float(dQ) ** 2


def _quadratic_pair_model(k_values: np.ndarray, kappa0: float, a: float, k0: float) -> tuple[np.ndarray, np.ndarray]:
    delta = k_values - k0
    shape = a * delta**2
    return kappa0 + shape, kappa0 - shape


def fit_quadratic_pair(
    k_values: np.ndarray | list[float],
    kappa_plus: np.ndarray | list[float],
    kappa_minus: np.ndarray | list[float],
) -> QuadraticPairFit:
    """Fit paired curvature branches to kappa0 plus/minus a (k-k0)^2."""
    k = np.asarray(k_values, dtype=float)
    plus = np.asarray(kappa_plus, dtype=float)
    minus = np.asarray(kappa_minus, dtype=float)
    if k.shape != plus.shape or plus.shape != minus.shape:
        raise ValueError("k_values, kappa_plus, and kappa_minus must have matching shapes.")
    if k.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")
    if k.size < 3:
        raise ValueError("Need at least three samples to fit a quadratic pair.")

    midpoint = 0.5 * (plus + minus)
    diff = 0.5 * (plus - minus)
    k0_guess = float(k[np.argmin(np.abs(diff))])
    kappa0_guess = float(np.mean(midpoint))
    scale = np.max(np.abs(k - k0_guess))
    a_guess = float(np.max(np.abs(diff)) / scale**2) if scale > 0.0 else 0.0

    def residuals(params: np.ndarray) -> np.ndarray:
        kappa0, a, k0 = params
        fit_plus, fit_minus = _quadratic_pair_model(k, kappa0, a, k0)
        return np.concatenate([fit_plus - plus, fit_minus - minus])

    result = least_squares(
        residuals,
        x0=np.array([kappa0_guess, a_guess, k0_guess], dtype=float),
        method="trf",
    )
    fit_plus, fit_minus = _quadratic_pair_model(k, *result.x)
    return QuadraticPairFit(
        kappa0=float(result.x[0]),
        a=float(result.x[1]),
        k0=float(result.x[2]),
        kappa_plus_fit=fit_plus,
        kappa_minus_fit=fit_minus,
        success=bool(result.success),
        cost=float(result.cost),
    )


def compute_quadratic_residual(
    k_values: np.ndarray | list[float],
    kappa_plus: np.ndarray | list[float],
    kappa_minus: np.ndarray | list[float],
    fit_params: QuadraticPairFit,
) -> float:
    """Compute the normalized quadratic residual for the paired-branch model."""
    k = np.asarray(k_values, dtype=float)
    plus = np.asarray(kappa_plus, dtype=float)
    minus = np.asarray(kappa_minus, dtype=float)
    if k.shape != plus.shape or plus.shape != minus.shape:
        raise ValueError("k_values, kappa_plus, and kappa_minus must have matching shapes.")
    if plus.shape != fit_params.kappa_plus_fit.shape or minus.shape != fit_params.kappa_minus_fit.shape:
        raise ValueError("Fit arrays must match the supplied data shape.")

    eps_plus = plus - fit_params.kappa_plus_fit
    eps_minus = minus - fit_params.kappa_minus_fit
    numerator = np.sqrt(np.sum(np.abs(eps_plus) ** 2 + np.abs(eps_minus) ** 2))
    denominator = np.sqrt(np.sum(np.abs(plus) ** 2 + np.abs(minus) ** 2))
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def find_edt_window(
    k_values: np.ndarray | list[float],
    residuals: np.ndarray | list[float],
    threshold: float,
) -> dict[str, Any]:
    """Find the largest contiguous window for which the residual stays below threshold."""
    k = np.asarray(k_values, dtype=float)
    r = np.asarray(residuals, dtype=float)
    if k.shape != r.shape:
        raise ValueError("k_values and residuals must have matching shapes.")
    if k.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    valid = r < threshold
    if not np.any(valid):
        return {
            "k_min": float(k[0]),
            "k_max": float(k[0]),
            "width": 0.0,
            "mask": valid,
            "threshold": float(threshold),
        }

    best_start = best_end = 0
    start = None
    for idx, is_valid in enumerate(valid):
        if is_valid and start is None:
            start = idx
        if start is not None and (idx == len(valid) - 1 or not valid[idx + 1]):
            end = idx
            if (k[end] - k[start]) > (k[best_end] - k[best_start]):
                best_start, best_end = start, end
            start = None

    return {
        "k_min": float(k[best_start]),
        "k_max": float(k[best_end]),
        "width": float(k[best_end] - k[best_start]),
        "mask": valid,
        "threshold": float(threshold),
    }


def bootstrap_edt_window(
    k_values: np.ndarray | list[float],
    kappa_plus: np.ndarray | list[float],
    kappa_minus: np.ndarray | list[float],
    n_boot: int = 1000,
) -> dict[str, Any]:
    """Estimate EDT-window uncertainty by residual bootstrap around the fitted pair model.

    A data-driven threshold is set from the best full-data residual profile:
    threshold = 1.1 * min(residual_profile).
    """
    k = np.asarray(k_values, dtype=float)
    plus = np.asarray(kappa_plus, dtype=float)
    minus = np.asarray(kappa_minus, dtype=float)
    if k.shape != plus.shape or plus.shape != minus.shape:
        raise ValueError("k_values, kappa_plus, and kappa_minus must have matching shapes.")
    if k.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")
    if k.size < 5:
        raise ValueError("Need at least five samples for bootstrap EDT-window estimation.")

    base_fit = fit_quadratic_pair(k, plus, minus)
    resid_plus = plus - base_fit.kappa_plus_fit
    resid_minus = minus - base_fit.kappa_minus_fit

    center_idx = int(np.argmin(np.abs(k - base_fit.k0)))
    half_widths: list[int] = []
    residual_profile: list[float] = []
    for half in range(1, min(center_idx, k.size - center_idx - 1) + 1):
        sl = slice(center_idx - half, center_idx + half + 1)
        fit = fit_quadratic_pair(k[sl], plus[sl], minus[sl])
        residual_profile.append(compute_quadratic_residual(k[sl], plus[sl], minus[sl], fit))
        half_widths.append(half)

    candidate_widths = np.array([k[center_idx + half] - k[center_idx - half] for half in half_widths], dtype=float)
    residual_profile_arr = np.asarray(residual_profile, dtype=float)
    threshold = float(1.1 * np.min(residual_profile_arr))
    window_scan = find_edt_window(candidate_widths, residual_profile_arr, threshold)
    widths = np.empty(n_boot, dtype=float)

    rng = np.random.default_rng()
    for idx in range(n_boot):
        sample_idx_plus = rng.integers(0, resid_plus.size, size=resid_plus.size)
        sample_idx_minus = rng.integers(0, resid_minus.size, size=resid_minus.size)
        plus_boot = base_fit.kappa_plus_fit + resid_plus[sample_idx_plus]
        minus_boot = base_fit.kappa_minus_fit + resid_minus[sample_idx_minus]

        boot_profile: list[float] = []
        for half in half_widths:
            sl = slice(center_idx - half, center_idx + half + 1)
            fit = fit_quadratic_pair(k[sl], plus_boot[sl], minus_boot[sl])
            boot_profile.append(compute_quadratic_residual(k[sl], plus_boot[sl], minus_boot[sl], fit))
        boot_window = find_edt_window(candidate_widths, np.asarray(boot_profile), threshold)
        widths[idx] = boot_window["width"]

    return {
        "fit": base_fit,
        "threshold": threshold,
        "residual_profile": residual_profile_arr,
        "candidate_widths": candidate_widths,
        "half_width_indices": np.asarray(half_widths, dtype=int),
        "window": window_scan,
        "width_samples": widths,
        "width_mean": float(np.mean(widths)),
        "width_std": float(np.std(widths, ddof=1)) if widths.size > 1 else 0.0,
        "width_ci_95": tuple(np.percentile(widths, [2.5, 97.5])),
    }
