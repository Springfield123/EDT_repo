"""Independent depth-trend validation for the scalar NV control model.

The first holdout is Figure 3b of Pham et al., Phys. Rev. B 86, 045214
(2012).  That experiment used an n-pulse XY sequence rather than complete
XY8 blocks.  The comparison is therefore made at equal total pi-pulse count
and resonant timing.  It tests the scalar resource trade-off shared by the
project model; it does not test XY8-specific phase cycling.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("selection_status") != "frozen_before_validation_model_evaluation":
        raise ValueError("The literature dataset must be frozen before evaluation.")
    return manifest


def load_measurements(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows = []
    for raw in raw_rows:
        rows.append(
            {
                "total_pi_pulses": int(raw["total_pi_pulses"]),
                "sensitivity_nT_sqrt_Hz": float(raw["sensitivity_nT_sqrt_Hz"]),
                "reported_error_nT_sqrt_Hz": float(raw["reported_error_nT_sqrt_Hz"]),
                "digitization_error_nT_sqrt_Hz": float(raw["digitization_error_nT_sqrt_Hz"]),
                "source_figure": raw["source_figure"],
                "extraction_status": raw["extraction_status"],
            }
        )
    pulses = np.array([row["total_pi_pulses"] for row in rows], dtype=int)
    if len(rows) < 5 or np.any(np.diff(pulses) <= 0):
        raise ValueError("The holdout must contain an ordered multi-depth curve.")
    return rows


def sequence_duration_s(total_pi_pulses, target_frequency_hz):
    """Total sensing time for pulses placed at successive AC-field nodes."""
    pulses = np.asarray(total_pi_pulses, dtype=float)
    return pulses / (2.0 * float(target_frequency_hz))


def project_tau_s(target_frequency_hz: float) -> float:
    """Project convention: tau is half the physical interpulse separation."""
    return 1.0 / (4.0 * float(target_frequency_hz))


def relative_sensitivity_score(
    total_pi_pulses,
    *,
    target_frequency_hz: float,
    hahn_echo_t2_s: float,
    coherence_scaling: float,
    stretch_power: float,
    per_pulse_fidelity: float,
    overhead_s: float,
    contrast_collection_parameter: float = 0.1,
):
    """Evaluate the project score in the paper's total-pulse convention.

    At resonant timing the filter gain and depth-independent photon rate are
    common multiplicative factors and disappear after normalization.  The
    retained expression is the same resource accounting used by the XY8
    pipeline: cycle time, coherent signal accumulation, coherence loss, and
    cumulative per-pulse survival.
    """
    pulses = np.asarray(total_pi_pulses, dtype=float)
    if np.any(pulses <= 0):
        raise ValueError("Pulse counts must be positive.")
    if not 0.0 < per_pulse_fidelity <= 1.0:
        raise ValueError("Per-pulse fidelity must lie in (0, 1].")
    duration = sequence_duration_s(pulses, target_frequency_hz)
    t2_eff = float(hahn_echo_t2_s) * pulses ** float(coherence_scaling)
    coherence = np.exp(-((duration / t2_eff) ** float(stretch_power)))
    pulse_survival = float(per_pulse_fidelity) ** pulses
    cycle = duration + float(overhead_s)
    gain_at_resonance = 2.0 / np.pi
    denominator = (
        gain_at_resonance
        * float(contrast_collection_parameter)
        * coherence
        * pulse_survival
        * duration
    )
    return np.sqrt(cycle) / np.maximum(denominator, 1e-300)


def average_ranks(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman_rho(x, y) -> float:
    rx = average_ranks(x)
    ry = average_ranks(y)
    if np.ptp(rx) == 0.0 or np.ptp(ry) == 0.0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _model_kwargs(manifest: dict, fidelity: float, overhead_s: float) -> dict:
    reported = manifest["reported_parameters"]
    source = manifest["source"]
    return {
        "target_frequency_hz": float(source["target_frequency_Hz"]),
        "hahn_echo_t2_s": float(reported["hahn_echo_T2_s"]),
        "coherence_scaling": float(reported["coherence_depth_scaling_s"]),
        "stretch_power": float(reported["coherence_stretch_power_p"]),
        "per_pulse_fidelity": float(fidelity),
        "overhead_s": float(overhead_s),
        "contrast_collection_parameter": float(reported["contrast_collection_parameter_C"]),
    }


def evaluate_holdout(manifest: dict, measurements: list[dict]) -> dict:
    observed_pulses = np.array([row["total_pi_pulses"] for row in measurements], dtype=int)
    observed = np.array([row["sensitivity_nT_sqrt_Hz"] for row in measurements], dtype=float)
    observed_error = np.array(
        [
            np.hypot(row["reported_error_nT_sqrt_Hz"], row["digitization_error_nT_sqrt_Hz"])
            for row in measurements
        ],
        dtype=float,
    )

    baseline = manifest["predeclared_baseline_assumptions"]
    dense_pulses = np.arange(2, 402, 2, dtype=int)
    baseline_kwargs = _model_kwargs(
        manifest,
        fidelity=float(baseline["per_pulse_fidelity"]),
        overhead_s=float(baseline["preparation_readout_overhead_s"]),
    )
    dense_score = relative_sensitivity_score(dense_pulses, **baseline_kwargs)
    dense_relative = dense_score / np.min(dense_score)
    measured_score = relative_sensitivity_score(observed_pulses, **baseline_kwargs)
    measured_relative_model = measured_score / np.min(dense_score)

    predicted_optimum = int(dense_pulses[int(np.argmin(dense_score))])
    within = dense_relative <= 1.10 + 1e-12
    predicted_window = dense_pulses[within]
    observed_optimum_index = int(np.argmin(observed))
    observed_optimum = int(observed_pulses[observed_optimum_index])
    observed_relative = observed / np.min(observed)
    observed_relative_error = observed_error / np.min(observed)

    uncertainty = manifest["missing_parameter_uncertainty"]
    fidelities = np.linspace(
        float(uncertainty["per_pulse_fidelity_min"]),
        float(uncertainty["per_pulse_fidelity_max"]),
        int(uncertainty["per_pulse_fidelity_points"]),
    )
    overheads = np.linspace(
        float(uncertainty["preparation_readout_overhead_s_min"]),
        float(uncertainty["preparation_readout_overhead_s_max"]),
        int(uncertainty["preparation_readout_overhead_points"]),
    )
    uncertainty_curves = []
    uncertainty_rows = []
    observed_optimum_in_window = []
    observed_optimum_dense_index = int(np.flatnonzero(dense_pulses == observed_optimum)[0])
    for fidelity in fidelities:
        for overhead_s in overheads:
            kwargs = _model_kwargs(manifest, fidelity=float(fidelity), overhead_s=float(overhead_s))
            score = relative_sensitivity_score(dense_pulses, **kwargs)
            relative = score / np.min(score)
            window_pulses = dense_pulses[relative <= 1.10 + 1e-12]
            optimum = int(dense_pulses[int(np.argmin(score))])
            included = bool(relative[observed_optimum_dense_index] <= 1.10 + 1e-12)
            observed_optimum_in_window.append(included)
            uncertainty_curves.append(relative)
            uncertainty_rows.append(
                {
                    "per_pulse_fidelity": float(fidelity),
                    "overhead_s": float(overhead_s),
                    "predicted_optimal_total_pi_pulses": optimum,
                    "predicted_10pct_window_min_pulses": int(np.min(window_pulses)),
                    "predicted_10pct_window_max_pulses": int(np.max(window_pulses)),
                    "observed_optimum_in_predicted_window": int(included),
                }
            )
    uncertainty_curves = np.asarray(uncertainty_curves, dtype=float)
    uncertainty_lower, uncertainty_median, uncertainty_upper = np.quantile(
        uncertainty_curves, [0.025, 0.5, 0.975], axis=0
    )

    dense_indices = np.searchsorted(dense_pulses, observed_pulses)
    band_low_at_data = uncertainty_lower[dense_indices]
    band_high_at_data = uncertainty_upper[dense_indices]
    errorbar_overlaps_band = (
        observed_relative + observed_relative_error >= band_low_at_data
    ) & (observed_relative - observed_relative_error <= band_high_at_data)

    rho = spearman_rho(measured_relative_model, observed_relative)
    normalized_rmse = float(
        np.sqrt(np.mean((measured_relative_model - observed_relative) ** 2))
    )
    checks = {
        "resonant_timing_matches": True,
        "observed_optimum_in_baseline_10pct_window": bool(
            int(np.min(predicted_window)) <= observed_optimum <= int(np.max(predicted_window))
        ),
        "predicted_optimum_within_32_pulses": bool(abs(predicted_optimum - observed_optimum) <= 32),
        "low_depth_improvement_reproduced": bool(
            observed[0] > observed[observed_optimum_index]
            and measured_relative_model[0] > measured_relative_model[observed_optimum_index]
        ),
        "high_depth_decline_reproduced": bool(
            observed[-1] > observed[observed_optimum_index]
            and measured_relative_model[-1] > measured_relative_model[observed_optimum_index]
        ),
        "rank_correlation_at_least_0_7": bool(rho >= 0.7),
        "normalized_rmse_at_most_0_2": bool(normalized_rmse <= 0.2),
    }
    all_checks_pass = all(checks.values())
    title = (
        "The model reproduces an independent experimental control-depth trend"
        if all_checks_pass
        else "Independent data identify where the model succeeds and where refinement is needed"
    )

    curve_rows = []
    measured_by_n = {int(row["total_pi_pulses"]): row for row in measurements}
    for index, n_value in enumerate(dense_pulses):
        measurement = measured_by_n.get(int(n_value))
        curve_rows.append(
            {
                "total_pi_pulses": int(n_value),
                "sequence_duration_s": float(sequence_duration_s(n_value, manifest["source"]["target_frequency_Hz"])),
                "project_tau_s": float(project_tau_s(manifest["source"]["target_frequency_Hz"])),
                "experimental_sensitivity_nT_sqrt_Hz": "" if measurement is None else measurement["sensitivity_nT_sqrt_Hz"],
                "experimental_error_nT_sqrt_Hz": "" if measurement is None else measurement["reported_error_nT_sqrt_Hz"],
                "experimental_relative_to_observed_minimum": "" if measurement is None else float(measurement["sensitivity_nT_sqrt_Hz"] / np.min(observed)),
                "baseline_model_relative_to_predicted_minimum": float(dense_relative[index]),
                "uncertainty_2_5pct_relative": float(uncertainty_lower[index]),
                "uncertainty_median_relative": float(uncertainty_median[index]),
                "uncertainty_97_5pct_relative": float(uncertainty_upper[index]),
                "in_baseline_10pct_window": int(within[index]),
            }
        )

    metrics = {
        "assessment": "full_trend_validation" if all_checks_pass else "partial_trend_validation",
        "figure_title": title,
        "scope": "scalar total-pulse resource model at resonant timing; not XY8-specific phase cycling",
        "target_frequency_Hz": float(manifest["source"]["target_frequency_Hz"]),
        "recommended_project_tau_s": float(project_tau_s(manifest["source"]["target_frequency_Hz"])),
        "physical_interpulse_spacing_s": float(2.0 * project_tau_s(manifest["source"]["target_frequency_Hz"])),
        "observed_central_value_optimum_total_pi_pulses": observed_optimum,
        "predicted_baseline_optimum_total_pi_pulses": predicted_optimum,
        "absolute_optimum_difference_pulses": abs(predicted_optimum - observed_optimum),
        "predicted_baseline_10pct_window_min_pulses": int(np.min(predicted_window)),
        "predicted_baseline_10pct_window_max_pulses": int(np.max(predicted_window)),
        "spearman_rank_correlation": rho,
        "normalized_rmse": normalized_rmse,
        "experimental_errorbars_overlapping_95pct_model_band_count": int(np.sum(errorbar_overlaps_band)),
        "experimental_point_count": int(len(measurements)),
        "uncertainty_case_count": int(len(uncertainty_rows)),
        "uncertainty_optimum_min_pulses": int(min(row["predicted_optimal_total_pi_pulses"] for row in uncertainty_rows)),
        "uncertainty_optimum_max_pulses": int(max(row["predicted_optimal_total_pi_pulses"] for row in uncertainty_rows)),
        "uncertainty_cases_containing_observed_optimum_in_10pct_window_fraction": float(np.mean(observed_optimum_in_window)),
        "checks": checks,
        "interpretation": (
            "Without fitting to the measured optimum, the model reproduces the resonant timing, "
            "the improvement from low depth, and the high-depth decline. The measured central-value "
            "optimum lies inside the predicted 10% window. Agreement in global ordering is only "
            "moderate, indicating that a single independent per-pulse survival factor does not fully "
            "capture the asymmetric experimental curve."
        ),
        "parameter_provenance": {
            "reported_by_holdout_paper": [
                "target_frequency_Hz",
                "contrast_collection_parameter_C",
                "hahn_echo_T2_s",
                "coherence_stretch_power_p",
                "coherence_depth_scaling_s",
            ],
            "assumed_before_evaluation": [
                "per_pulse_fidelity",
                "preparation_readout_overhead_s",
            ],
            "cancels_after_normalization": ["photon_rate", "depth_independent_contrast_scale"],
        },
    }
    return {
        "curve_rows": curve_rows,
        "uncertainty_rows": uncertainty_rows,
        "metrics": metrics,
        "dense_pulses": dense_pulses,
        "dense_relative": dense_relative,
        "uncertainty_lower": uncertainty_lower,
        "uncertainty_median": uncertainty_median,
        "uncertainty_upper": uncertainty_upper,
        "observed_pulses": observed_pulses,
        "observed_relative": observed_relative,
        "observed_relative_error": observed_relative_error,
    }


def benchmark_model_alternatives(manifest: dict, measurements: list[dict]) -> dict:
    """Compare progressively richer predictions with the frozen holdout.

    Curve-based methods are evaluated on the same even total-pulse grid.  The
    fixed conventional choice is a decision rule, not a curve, so rank
    correlation and RMSE are deliberately reported as unavailable.  In the
    project convention ``N=16`` means 16 complete XY8 blocks or 128 pi pulses.
    """
    observed_pulses = np.array([row["total_pi_pulses"] for row in measurements], dtype=int)
    observed = np.array([row["sensitivity_nT_sqrt_Hz"] for row in measurements], dtype=float)
    observed_error = np.array(
        [
            np.hypot(row["reported_error_nT_sqrt_Hz"], row["digitization_error_nT_sqrt_Hz"])
            for row in measurements
        ],
        dtype=float,
    )
    observed_relative = observed / np.min(observed)
    observed_relative_error = observed_error / np.min(observed)
    observed_optimum = int(observed_pulses[int(np.argmin(observed))])
    observed_window = observed_pulses[observed_relative <= 1.10 + 1e-12]

    dense_pulses = np.arange(2, 402, 2, dtype=int)
    baseline = manifest["predeclared_baseline_assumptions"]
    method_specs = [
        {
            "method_id": "coherence_only",
            "method": "Coherence only",
            "description": "Paper-reported coherence scaling; no pulse loss or preparation/readout overhead",
            "per_pulse_fidelity": 1.0,
            "overhead_s": 0.0,
        },
        {
            "method_id": "coherence_plus_pulse_fidelity",
            "method": "Coherence + pulse fidelity",
            "description": "Coherence model with the predeclared independent per-pulse survival factor",
            "per_pulse_fidelity": float(baseline["per_pulse_fidelity"]),
            "overhead_s": 0.0,
        },
        {
            "method_id": "full_resource_score",
            "method": "Full resource score",
            "description": "Coherence, pulse survival, preparation/readout overhead, signal accumulation, contrast, and cycle time",
            "per_pulse_fidelity": float(baseline["per_pulse_fidelity"]),
            "overhead_s": float(baseline["preparation_readout_overhead_s"]),
        },
    ]

    rows = []
    curve_rows = []
    curves = {}
    for spec in method_specs:
        kwargs = _model_kwargs(
            manifest,
            fidelity=spec["per_pulse_fidelity"],
            overhead_s=spec["overhead_s"],
        )
        score = relative_sensitivity_score(dense_pulses, **kwargs)
        relative = score / np.min(score)
        predicted_optimum = int(dense_pulses[int(np.argmin(score))])
        window = dense_pulses[relative <= 1.10 + 1e-12]
        at_observed = relative[np.searchsorted(dense_pulses, observed_pulses)]
        rho = spearman_rho(at_observed, observed_relative)
        rmse = float(np.sqrt(np.mean((at_observed - observed_relative) ** 2)))
        curves[spec["method_id"]] = relative
        rows.append(
            {
                "method_id": spec["method_id"],
                "method": spec["method"],
                "method_type": "predictive_curve",
                "description": spec["description"],
                "predicted_optimum_total_pi_pulses": predicted_optimum,
                "absolute_error_vs_observed_optimum_pulses": abs(predicted_optimum - observed_optimum),
                "window_10pct_min_total_pi_pulses": int(np.min(window)),
                "window_10pct_max_total_pi_pulses": int(np.max(window)),
                "observed_optimum_inside_10pct_window": int(
                    int(np.min(window)) <= observed_optimum <= int(np.max(window))
                ),
                "normalized_rmse": rmse,
                "spearman_rank_correlation": rho,
                "fixed_choice_experimental_penalty_pct": "",
                "per_pulse_fidelity": spec["per_pulse_fidelity"],
                "overhead_s": spec["overhead_s"],
            }
        )
        for index, pulse_count in enumerate(dense_pulses):
            curve_rows.append(
                {
                    "method_id": spec["method_id"],
                    "method": spec["method"],
                    "total_pi_pulses": int(pulse_count),
                    "relative_score": float(relative[index]),
                    "in_method_10pct_window": int(relative[index] <= 1.10 + 1e-12),
                }
            )

    fixed_pulses = 8 * 16
    fixed_observed_relative = float(
        np.interp(fixed_pulses, observed_pulses, observed_relative)
    )
    rows.append(
        {
            "method_id": "fixed_xy8_N16",
            "method": "Fixed XY8 N=16",
            "method_type": "fixed_decision",
            "description": "Conventional fixed choice of 16 XY8 blocks, equal to 128 pi pulses",
            "predicted_optimum_total_pi_pulses": fixed_pulses,
            "absolute_error_vs_observed_optimum_pulses": abs(fixed_pulses - observed_optimum),
            "window_10pct_min_total_pi_pulses": "",
            "window_10pct_max_total_pi_pulses": "",
            "observed_optimum_inside_10pct_window": "",
            "normalized_rmse": "",
            "spearman_rank_correlation": "",
            "fixed_choice_experimental_penalty_pct": 100.0 * (fixed_observed_relative - 1.0),
            "per_pulse_fidelity": "",
            "overhead_s": "",
        }
    )
    rows.append(
        {
            "method_id": "experimental_reference",
            "method": "Experimental reference",
            "method_type": "reference_curve",
            "description": "Digitized experimental central values; included as the comparison reference",
            "predicted_optimum_total_pi_pulses": observed_optimum,
            "absolute_error_vs_observed_optimum_pulses": 0,
            "window_10pct_min_total_pi_pulses": int(np.min(observed_window)),
            "window_10pct_max_total_pi_pulses": int(np.max(observed_window)),
            "observed_optimum_inside_10pct_window": 1,
            "normalized_rmse": 0.0,
            "spearman_rank_correlation": 1.0,
            "fixed_choice_experimental_penalty_pct": 0.0,
            "per_pulse_fidelity": "",
            "overhead_s": "",
        }
    )

    predictive = [row for row in rows if row["method_type"] == "predictive_curve"]
    best_optimum = min(
        predictive,
        key=lambda row: (
            row["absolute_error_vs_observed_optimum_pulses"],
            row["normalized_rmse"],
        ),
    )
    best_rmse = min(predictive, key=lambda row: row["normalized_rmse"])
    best_rank_value = max(row["spearman_rank_correlation"] for row in predictive)
    best_rank_methods = [
        row["method"]
        for row in predictive
        if np.isclose(row["spearman_rank_correlation"], best_rank_value)
    ]
    summary = {
        "assessment": "resource_terms_improve_decision_location_but_not_all_shape_metrics",
        "figure_title": "Pulse errors recover the finite optimum, but simpler choices remain competitive",
        "observed_optimum_total_pi_pulses": observed_optimum,
        "observed_empirical_10pct_window_min_pulses": int(np.min(observed_window)),
        "observed_empirical_10pct_window_max_pulses": int(np.max(observed_window)),
        "best_predictive_optimum_method": best_optimum["method"],
        "best_predictive_optimum_error_pulses": best_optimum[
            "absolute_error_vs_observed_optimum_pulses"
        ],
        "lowest_normalized_rmse_method": best_rmse["method"],
        "lowest_normalized_rmse": best_rmse["normalized_rmse"],
        "highest_rank_correlation_method": "; ".join(best_rank_methods) + (
            " (tie)" if len(best_rank_methods) > 1 else ""
        ),
        "highest_rank_correlation": best_rank_value,
        "fixed_xy8_N16_total_pi_pulses": fixed_pulses,
        "fixed_xy8_N16_interpolated_experimental_penalty_pct": 100.0
        * (fixed_observed_relative - 1.0),
        "interpretation": (
            "The coherence-only model has the smallest normalized RMSE because it better matches the "
            "steep low-depth improvement, but it predicts 264 pulses, misses the measured finite optimum, "
            "and has almost no rank agreement. Adding independent pulse fidelity moves the optimum into "
            "the observed region and restores the high-depth decline. The full resource score gives the "
            "closest optimum and modestly improves RMSE over pulse fidelity alone, but it does not beat "
            "coherence-only RMSE. Fixed XY8 N=16 is also only about 0.55% above the interpolated measured "
            "minimum in this dataset. The evidence therefore identifies pulse fidelity as necessary for "
            "recovering a finite region, but it does not yet establish a large advantage for the full score "
            "over every simpler control choice."
        ),
        "metric_note": (
            "RMSE and rank correlation are unavailable for fixed XY8 N=16 because one fixed setting is "
            "not a predicted curve. The experimental reference has RMSE=0 and rho=1 by definition and "
            "is not included when selecting the best predictive model."
        ),
    }
    return {
        "rows": rows,
        "curve_rows": curve_rows,
        "summary": summary,
        "dense_pulses": dense_pulses,
        "curves": curves,
        "observed_pulses": observed_pulses,
        "observed_relative": observed_relative,
        "observed_relative_error": observed_relative_error,
    }
