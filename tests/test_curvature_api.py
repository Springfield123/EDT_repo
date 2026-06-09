import numpy as np

from edt.curvature import (
    bootstrap_edt_window,
    compute_curvature_from_displacements,
    compute_quadratic_residual,
    find_edt_window,
    fit_quadratic_pair,
)


def test_centered_curvature_matches_quadratic_second_derivative():
    dq = 0.1
    e_minus = np.array([dq**2, 2.0 * dq**2])
    e_zero = np.array([0.0, 0.0])
    e_plus = np.array([dq**2, 2.0 * dq**2])
    curvature = compute_curvature_from_displacements(e_minus, e_zero, e_plus, dq)
    assert np.allclose(curvature, np.array([2.0, 4.0]))


def test_quadratic_pair_fit_has_small_residual_on_exact_data():
    k = np.linspace(-1.0, 1.0, 41)
    plus = 0.5 + 1.25 * (k - 0.2) ** 2
    minus = 0.5 - 1.25 * (k - 0.2) ** 2
    fit = fit_quadratic_pair(k, plus, minus)
    residual = compute_quadratic_residual(k, plus, minus, fit)
    assert fit.success
    assert residual < 1e-8


def test_find_edt_window_returns_largest_contiguous_segment():
    k = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    residuals = np.array([0.2, 0.05, 0.04, 0.3, 0.02])
    window = find_edt_window(k, residuals, threshold=0.1)
    assert np.isclose(window["k_min"], 0.1)
    assert np.isclose(window["k_max"], 0.2)
    assert np.isclose(window["width"], 0.1)


def test_bootstrap_edt_window_returns_summary():
    k = np.linspace(-0.5, 0.5, 21)
    plus = 0.2 + 0.8 * k**2 + 0.01 * np.sin(5.0 * k)
    minus = 0.2 - 0.8 * k**2 + 0.01 * np.cos(5.0 * k)
    result = bootstrap_edt_window(k, plus, minus, n_boot=32)
    assert result["width_mean"] >= 0.0
    assert len(result["width_ci_95"]) == 2
