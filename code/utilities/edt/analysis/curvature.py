from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CurvatureBranch:
    k: np.ndarray
    curvature: np.ndarray
    label: str = "branch"

    def __post_init__(self) -> None:
        if self.k.shape != self.curvature.shape:
            raise ValueError("k and curvature must have the same shape.")
        if self.k.ndim != 1:
            raise ValueError("k and curvature must be one-dimensional arrays.")


@dataclass(frozen=True)
class QuadraticFitResult:
    coefficients: np.ndarray
    fitted: np.ndarray
    residual_rms: float
    center: float
    half_width: float


@dataclass(frozen=True)
class EDTWindowResult:
    k_min: float
    k_max: float
    k0: float
    residual_series: np.ndarray
    valid_mask: np.ndarray
    max_residual: float

    @property
    def width(self) -> float:
        return float(self.k_max - self.k_min)


def fit_local_quadratic(k: np.ndarray, y: np.ndarray, center: float, half_width: float) -> QuadraticFitResult:
    mask = np.abs(k - center) <= half_width
    if mask.sum() < 3:
        raise ValueError("Need at least 3 points inside the fitting window.")

    k_local = k[mask]
    y_local = y[mask]
    coeffs = np.polyfit(k_local - center, y_local, deg=2)
    fitted_local = np.polyval(coeffs, k_local - center)
    residual_rms = float(np.sqrt(np.mean((y_local - fitted_local) ** 2)))

    fitted = np.full_like(y, np.nan, dtype=float)
    fitted[mask] = fitted_local
    return QuadraticFitResult(
        coefficients=coeffs,
        fitted=fitted,
        residual_rms=residual_rms,
        center=float(center),
        half_width=float(half_width),
    )


def paired_residual(
    branch_a: CurvatureBranch,
    branch_b: CurvatureBranch,
    center: float,
    half_width: float,
) -> float:
    if not np.allclose(branch_a.k, branch_b.k):
        raise ValueError("Paired branches must share the same k-grid.")

    fit_a = fit_local_quadratic(branch_a.k, branch_a.curvature, center, half_width)
    fit_b = fit_local_quadratic(branch_b.k, branch_b.curvature, center, half_width)

    mask = np.abs(branch_a.k - center) <= half_width
    res_a = branch_a.curvature[mask] - fit_a.fitted[mask]
    res_b = branch_b.curvature[mask] - fit_b.fitted[mask]
    return float(np.sqrt(np.mean(np.concatenate([res_a, res_b]) ** 2)))


def scan_edt_window(
    branch_a: CurvatureBranch,
    branch_b: CurvatureBranch,
    k0: float,
    half_widths: Iterable[float],
    residual_threshold: float,
) -> EDTWindowResult:
    half_widths = np.asarray(list(half_widths), dtype=float)
    residuals = np.array(
        [paired_residual(branch_a, branch_b, center=k0, half_width=w) for w in half_widths],
        dtype=float,
    )
    valid = residuals <= residual_threshold

    if not np.any(valid):
        return EDTWindowResult(
            k_min=float(k0),
            k_max=float(k0),
            k0=float(k0),
            residual_series=residuals,
            valid_mask=valid,
            max_residual=float(residual_threshold),
        )

    max_valid_width = float(np.max(half_widths[valid]))
    return EDTWindowResult(
        k_min=float(k0 - max_valid_width),
        k_max=float(k0 + max_valid_width),
        k0=float(k0),
        residual_series=residuals,
        valid_mask=valid,
        max_residual=float(residual_threshold),
    )


def estimate_k0_from_gap_minimum(branch_a: CurvatureBranch, branch_b: CurvatureBranch) -> float:
    if not np.allclose(branch_a.k, branch_b.k):
        raise ValueError("Paired branches must share the same k-grid.")
    gap = np.abs(branch_a.curvature - branch_b.curvature)
    return float(branch_a.k[np.argmin(gap)])
