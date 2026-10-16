"""One-dimensional EDT-region analysis for the frozen Pham holdout.

The experimental curve is sparse and measured at one resonant timing, so this
module evaluates connected runs along the ordered total-pulse axis.  It does
not claim two-dimensional ``(tau, N)`` topology or XY8-specific phase cycling.
Measurement uncertainty is propagated with a seeded parametric bootstrap that
treats the reported and digitization errors as independent Gaussian standard
deviations.  No model parameter is fitted to the observations.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

import xy8_literature_validation as validation


DEFAULT_THRESHOLDS_PERCENT = tuple(range(1, 101))
DEFAULT_CHECKPOINTS_PERCENT = (1, 5, 10, 20, 30, 50, 75, 100)


def _validate_thresholds(values: Iterable[float]) -> tuple[float, ...]:
    thresholds = tuple(float(value) for value in values)
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    if not all(np.isfinite(value) and value >= 0.0 for value in thresholds):
        raise ValueError("Thresholds must be finite and non-negative.")
    if any(right <= left for left, right in zip(thresholds, thresholds[1:])):
        raise ValueError("Thresholds must be strictly increasing.")
    return thresholds


def _validate_curve(pulses, scores) -> tuple[np.ndarray, np.ndarray]:
    pulse_values = np.asarray(pulses, dtype=int)
    score_values = np.asarray(scores, dtype=float)
    if pulse_values.ndim != 1 or pulse_values.size == 0:
        raise ValueError("Pulse counts must be a non-empty one-dimensional array.")
    if score_values.shape != pulse_values.shape:
        raise ValueError("Scores must have the same shape as pulse counts.")
    if np.any(np.diff(pulse_values) <= 0) or np.any(pulse_values <= 0):
        raise ValueError("Pulse counts must be positive and strictly increasing.")
    if not np.all(np.isfinite(score_values)) or np.any(score_values <= 0.0):
        raise ValueError("Scores must be positive and finite.")
    return pulse_values, score_values


def summarize_masks(mask, pulses, anchor_indices) -> dict[str, np.ndarray]:
    """Summarize connected runs and the run containing each anchor index."""
    masks = np.asarray(mask, dtype=bool)
    if masks.ndim == 1:
        masks = masks[None, :]
    pulse_values = np.asarray(pulses, dtype=int)
    anchors = np.asarray(anchor_indices, dtype=int)
    if masks.ndim != 2 or masks.shape[1] != pulse_values.size:
        raise ValueError("Masks must have one column per pulse count.")
    if anchors.ndim == 0:
        anchors = np.full(masks.shape[0], int(anchors), dtype=int)
    if anchors.shape != (masks.shape[0],):
        raise ValueError("One anchor index is required per mask row.")
    if np.any(anchors < 0) or np.any(anchors >= pulse_values.size):
        raise ValueError("Anchor indices are outside the sampled curve.")
    if not np.all(masks[np.arange(masks.shape[0]), anchors]):
        raise ValueError("Every anchor must lie inside its selected set.")

    starts = masks & np.concatenate(
        [np.ones((masks.shape[0], 1), dtype=bool), ~masks[:, :-1]], axis=1
    )
    labels = np.cumsum(starts, axis=1)
    labels[~masks] = 0
    anchor_labels = labels[np.arange(masks.shape[0]), anchors]
    anchor_component = masks & (labels == anchor_labels[:, None])
    lower = np.min(
        np.where(anchor_component, pulse_values[None, :], np.iinfo(np.int64).max),
        axis=1,
    )
    upper = np.max(np.where(anchor_component, pulse_values[None, :], -1), axis=1)
    return {
        "beta_0": np.sum(starts, axis=1).astype(int),
        "lower_boundary_pulses": lower.astype(int),
        "upper_boundary_pulses": upper.astype(int),
        "window_width_pulses": (upper - lower).astype(int),
        "anchor_component_point_count": np.sum(anchor_component, axis=1).astype(int),
        "labels": labels.astype(int),
    }


def summarize_curve(pulses, scores, threshold_percent: float) -> dict:
    pulse_values, score_values = _validate_curve(pulses, scores)
    relative = score_values / np.min(score_values)
    mask = relative <= 1.0 + float(threshold_percent) / 100.0 + 1e-12
    optimum_index = int(np.argmin(score_values))
    result = summarize_masks(mask, pulse_values, optimum_index)
    return {
        "beta_0": int(result["beta_0"][0]),
        "lower_boundary_pulses": int(result["lower_boundary_pulses"][0]),
        "upper_boundary_pulses": int(result["upper_boundary_pulses"][0]),
        "window_width_pulses": int(result["window_width_pulses"][0]),
        "component_point_count": int(result["anchor_component_point_count"][0]),
        "optimum_pulses": int(pulse_values[optimum_index]),
        "mask": mask,
        "relative_score": relative,
    }


def _positive_gaussian_draws(
    means: np.ndarray,
    standard_deviations: np.ndarray,
    trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    samples = rng.normal(means, standard_deviations, size=(int(trials), means.size))
    invalid = samples <= 0.0
    while np.any(invalid):
        samples[invalid] = rng.normal(
            np.broadcast_to(means, samples.shape)[invalid],
            np.broadcast_to(standard_deviations, samples.shape)[invalid],
        )
        invalid = samples <= 0.0
    return samples


def _quantiles(values) -> dict[str, float]:
    low, median, high = np.quantile(np.asarray(values, dtype=float), [0.025, 0.5, 0.975])
    return {
        "q2_5": float(low),
        "median": float(median),
        "q97_5": float(high),
    }


def _histogram_rows(threshold_percent: float, metric: str, values) -> list[dict]:
    unique, counts = np.unique(np.asarray(values), return_counts=True)
    total = int(np.sum(counts))
    return [
        {
            "threshold_percent": float(threshold_percent),
            "metric": metric,
            "value": float(value),
            "count": int(count),
            "probability": float(count / total),
        }
        for value, count in zip(unique, counts)
    ]


def _model_curves(manifest: dict, observed_pulses: np.ndarray) -> dict:
    baseline = manifest["predeclared_baseline_assumptions"]
    reported = manifest["reported_parameters"]
    kwargs = {
        "target_frequency_hz": float(manifest["source"]["target_frequency_Hz"]),
        "hahn_echo_t2_s": float(reported["hahn_echo_T2_s"]),
        "coherence_scaling": float(reported["coherence_depth_scaling_s"]),
        "stretch_power": float(reported["coherence_stretch_power_p"]),
        "per_pulse_fidelity": float(baseline["per_pulse_fidelity"]),
        "overhead_s": float(baseline["preparation_readout_overhead_s"]),
        "contrast_collection_parameter": float(
            reported["contrast_collection_parameter_C"]
        ),
    }
    dense_pulses = np.arange(2, 402, 2, dtype=int)
    dense_score = validation.relative_sensitivity_score(dense_pulses, **kwargs)
    observed_grid_score = validation.relative_sensitivity_score(observed_pulses, **kwargs)
    return {
        "dense_pulses": dense_pulses,
        "dense_score": dense_score,
        "observed_grid_score": observed_grid_score,
        "kwargs": kwargs,
    }


def analyze_pham_topology(
    manifest: dict,
    measurements: list[dict],
    *,
    thresholds_percent=DEFAULT_THRESHOLDS_PERCENT,
    checkpoints_percent=DEFAULT_CHECKPOINTS_PERCENT,
    trials: int = 100_000,
    seed: int = 12015686,
) -> dict:
    """Compare measured and frozen-model depth topology without fitting."""
    thresholds = _validate_thresholds(thresholds_percent)
    checkpoints = tuple(float(value) for value in checkpoints_percent)
    if int(trials) < 1000:
        raise ValueError("At least 1000 bootstrap trials are required.")
    if not set(checkpoints).issubset(set(thresholds)):
        raise ValueError("Every checkpoint must be present in the threshold grid.")

    pulses = np.asarray([row["total_pi_pulses"] for row in measurements], dtype=int)
    observed = np.asarray(
        [row["sensitivity_nT_sqrt_Hz"] for row in measurements], dtype=float
    )
    standard_deviation = np.asarray(
        [
            np.hypot(
                row["reported_error_nT_sqrt_Hz"],
                row["digitization_error_nT_sqrt_Hz"],
            )
            for row in measurements
        ],
        dtype=float,
    )
    pulses, observed = _validate_curve(pulses, observed)
    model_curves = _model_curves(manifest, pulses)
    rng = np.random.default_rng(int(seed))
    bootstrap_scores = _positive_gaussian_draws(
        observed, standard_deviation, int(trials), rng
    )
    bootstrap_relative = bootstrap_scores / np.min(
        bootstrap_scores, axis=1, keepdims=True
    )
    bootstrap_optimum_indices = np.argmin(bootstrap_scores, axis=1)

    threshold_rows = []
    distribution_rows = []
    beta0_matrix = np.empty((int(trials), len(thresholds)), dtype=np.uint8)
    central_relative = observed / np.min(observed)
    model_observed_relative = model_curves["observed_grid_score"] / np.min(
        model_curves["observed_grid_score"]
    )
    model_dense_relative = model_curves["dense_score"] / np.min(
        model_curves["dense_score"]
    )

    for threshold_index, threshold in enumerate(thresholds):
        limit = 1.0 + threshold / 100.0
        bootstrap_mask = bootstrap_relative <= limit + 1e-12
        bootstrap_summary = summarize_masks(
            bootstrap_mask, pulses, bootstrap_optimum_indices
        )
        beta0 = bootstrap_summary["beta_0"]
        beta0_matrix[:, threshold_index] = beta0

        central = summarize_curve(pulses, observed, threshold)
        model_observed = summarize_curve(
            pulses, model_curves["observed_grid_score"], threshold
        )
        model_dense = summarize_curve(
            model_curves["dense_pulses"], model_curves["dense_score"], threshold
        )
        intersection = np.sum(central["mask"] & model_observed["mask"])
        union = np.sum(central["mask"] | model_observed["mask"])
        lower_quantiles = _quantiles(bootstrap_summary["lower_boundary_pulses"])
        upper_quantiles = _quantiles(bootstrap_summary["upper_boundary_pulses"])
        width_quantiles = _quantiles(bootstrap_summary["window_width_pulses"])
        unique_beta, beta_counts = np.unique(beta0, return_counts=True)
        modal_beta = int(unique_beta[int(np.argmax(beta_counts))])
        threshold_rows.append(
            {
                "threshold_percent": float(threshold),
                "relative_limit": float(limit),
                "central_beta_0": central["beta_0"],
                "bootstrap_probability_beta_0_equals_1": float(np.mean(beta0 == 1)),
                "bootstrap_probability_beta_0_greater_than_1": float(np.mean(beta0 > 1)),
                "bootstrap_modal_beta_0": modal_beta,
                "central_lower_boundary_pulses": central["lower_boundary_pulses"],
                "bootstrap_lower_boundary_q2_5_pulses": lower_quantiles["q2_5"],
                "bootstrap_lower_boundary_median_pulses": lower_quantiles["median"],
                "bootstrap_lower_boundary_q97_5_pulses": lower_quantiles["q97_5"],
                "central_upper_boundary_pulses": central["upper_boundary_pulses"],
                "bootstrap_upper_boundary_q2_5_pulses": upper_quantiles["q2_5"],
                "bootstrap_upper_boundary_median_pulses": upper_quantiles["median"],
                "bootstrap_upper_boundary_q97_5_pulses": upper_quantiles["q97_5"],
                "central_window_width_pulses": central["window_width_pulses"],
                "bootstrap_window_width_q2_5_pulses": width_quantiles["q2_5"],
                "bootstrap_window_width_median_pulses": width_quantiles["median"],
                "bootstrap_window_width_q97_5_pulses": width_quantiles["q97_5"],
                "model_observed_grid_beta_0": model_observed["beta_0"],
                "model_observed_grid_lower_boundary_pulses": model_observed[
                    "lower_boundary_pulses"
                ],
                "model_observed_grid_upper_boundary_pulses": model_observed[
                    "upper_boundary_pulses"
                ],
                "model_observed_grid_window_width_pulses": model_observed[
                    "window_width_pulses"
                ],
                "model_dense_beta_0": model_dense["beta_0"],
                "model_dense_lower_boundary_pulses": model_dense[
                    "lower_boundary_pulses"
                ],
                "model_dense_upper_boundary_pulses": model_dense[
                    "upper_boundary_pulses"
                ],
                "model_dense_window_width_pulses": model_dense[
                    "window_width_pulses"
                ],
                "measured_model_sampled_mask_jaccard": float(intersection / union),
                "bootstrap_probability_beta_0_matches_sampled_model": float(
                    np.mean(beta0 == model_observed["beta_0"])
                ),
            }
        )
        if threshold in checkpoints:
            for metric in (
                "beta_0",
                "lower_boundary_pulses",
                "upper_boundary_pulses",
                "window_width_pulses",
            ):
                distribution_rows.extend(
                    _histogram_rows(threshold, metric, bootstrap_summary[metric])
                )
            distribution_rows.extend(
                _histogram_rows(
                    threshold,
                    "optimum_pulses",
                    pulses[bootstrap_optimum_indices],
                )
            )

    beta0_is_one_all_thresholds = np.all(beta0_matrix == 1, axis=1)
    beta0_transition_count = np.sum(np.diff(beta0_matrix, axis=1) != 0, axis=1)
    persistence_distribution = _histogram_rows(
        -1.0, "beta_0_transition_count_1_to_100pct", beta0_transition_count
    )
    distribution_rows.extend(persistence_distribution)
    central_beta0 = np.asarray([row["central_beta_0"] for row in threshold_rows])
    model_sampled_beta0 = np.asarray(
        [row["model_observed_grid_beta_0"] for row in threshold_rows]
    )
    model_dense_beta0 = np.asarray([row["model_dense_beta_0"] for row in threshold_rows])
    ten_percent_row = next(
        row for row in threshold_rows if row["threshold_percent"] == 10.0
    )
    summary = {
        "status": "frozen_pham_depth_topology_evaluated",
        "scope": (
            "one-dimensional connected runs on the ten sampled total-pulse settings "
            "at one resonant timing; not two-dimensional tau-depth topology and not "
            "XY8-specific phase cycling"
        ),
        "sequence": manifest["source"]["sequence"],
        "target_frequency_hz": float(manifest["source"]["target_frequency_Hz"]),
        "project_tau_s": float(
            validation.project_tau_s(manifest["source"]["target_frequency_Hz"])
        ),
        "experimental_point_count": int(pulses.size),
        "bootstrap": {
            "method": "independent positive Gaussian parametric bootstrap",
            "trials": int(trials),
            "seed": int(seed),
            "reported_error_assumption": (
                "reported error-bar half widths are treated as one standard deviation"
            ),
            "combined_error": "sqrt(reported_error^2 + digitization_error^2)",
            "normalization": "each realization divided by its own minimum",
            "correlation_assumption": "errors at different depths are independent",
        },
        "no_fit_statement": manifest["predeclared_baseline_assumptions"][
            "no_fit_statement"
        ],
        "central_measured_optimum_pulses": int(pulses[np.argmin(observed)]),
        "frozen_model_optimum_dense_pulses": int(
            model_curves["dense_pulses"][np.argmin(model_curves["dense_score"])]
        ),
        "central_beta_0_equals_1_from_1_to_100pct": bool(
            np.all(central_beta0 == 1)
        ),
        "sampled_model_beta_0_equals_1_from_1_to_100pct": bool(
            np.all(model_sampled_beta0 == 1)
        ),
        "dense_model_beta_0_equals_1_from_1_to_100pct": bool(
            np.all(model_dense_beta0 == 1)
        ),
        "bootstrap_probability_beta_0_equals_1_at_every_threshold": float(
            np.mean(beta0_is_one_all_thresholds)
        ),
        "bootstrap_beta_0_transition_count": _quantiles(beta0_transition_count),
        "ten_percent_comparison": dict(ten_percent_row),
        "separation_statement": (
            "The Farfurnik conventional-XY8 holdout is not included in this topology "
            "analysis; it remains a separate short-free-evolution pulse-error mechanism check."
        ),
        "interpretation_limit": (
            "Adjacency connects consecutive published sample locations despite unequal "
            "pulse-count gaps. The bootstrap propagates reported vertical uncertainty but "
            "cannot reveal an unmeasured performance excursion between sampled depths."
        ),
    }
    plot_data = {
        "observed_pulses": pulses,
        "observed": observed,
        "observed_error": standard_deviation,
        "observed_relative": central_relative,
        "model_observed_relative": model_observed_relative,
        "dense_pulses": model_curves["dense_pulses"],
        "model_dense_relative": model_dense_relative,
    }
    return {
        "threshold_rows": threshold_rows,
        "distribution_rows": distribution_rows,
        "summary": summary,
        "plot_data": plot_data,
    }
