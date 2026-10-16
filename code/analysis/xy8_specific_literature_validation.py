"""XY8-specific frozen-holdout validation of scalar and coherent models.

The frozen dataset is the conventional-XY8 series in Figure 3b of
Farfurnik et al., Phys. Rev. B 92, 060301(R) (2015).  The experiment is in
the short-free-evolution limit, so this comparison isolates pulse-error
accumulation.  It is not a validation of the complete magnetometry score.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import xy8_coherent_control as coherent


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
        raise ValueError("The XY8 holdout must be frozen before model evaluation.")
    if manifest["source"].get("sequence") != "conventional XY8":
        raise ValueError("This evaluator requires a conventional-XY8 holdout.")
    if manifest["predeclared_comparison"].get("fitting") != "none":
        raise ValueError("Holdout parameters must not be fitted to the measured curve.")
    coherent_comparison = manifest.get("predeclared_coherent_comparison", {})
    if coherent_comparison.get("freeze_status") != "frozen_before_coherent_model_evaluation":
        raise ValueError("Coherent holdout parameters must be frozen before evaluation.")
    if coherent_comparison.get("fitting") != "none":
        raise ValueError("Coherent holdout parameters must not be fitted.")
    return manifest


def load_measurements(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows: list[dict] = []
    for raw in raw_rows:
        row = {
            "xy8_blocks": int(raw["xy8_blocks"]),
            "total_pi_pulses": int(raw["total_pi_pulses"]),
            "relative_contrast": float(raw["relative_contrast"]),
            "reported_error_lower": float(raw["reported_error_lower"]),
            "reported_error_upper": float(raw["reported_error_upper"]),
            "digitization_error": float(raw["digitization_error"]),
            "source_figure": raw["source_figure"],
            "extraction_status": raw["extraction_status"],
        }
        if row["total_pi_pulses"] != 8 * row["xy8_blocks"]:
            raise ValueError("Every holdout depth must contain complete XY8 blocks (8N pulses).")
        rows.append(row)
    pulses = np.asarray([row["total_pi_pulses"] for row in rows], dtype=int)
    if len(rows) < 3 or np.any(np.diff(pulses) <= 0):
        raise ValueError("The XY8 holdout must contain at least three ordered depths.")
    return rows


def scalar_pulse_survival(total_pi_pulses, per_pulse_fidelity: float):
    """Independent identical-error approximation used by the project model."""
    pulses = np.asarray(total_pi_pulses, dtype=float)
    fidelity = float(per_pulse_fidelity)
    if np.any(pulses < 0):
        raise ValueError("Pulse counts cannot be negative.")
    if not 0.0 < fidelity <= 1.0:
        raise ValueError("Per-pulse fidelity must lie in (0, 1].")
    return fidelity**pulses


def _batched_rotation_unitary(angle_errors, axis_z_values, phase_errors, base_phase):
    """Vectorized equivalent of ``coherent.rotation_unitary`` for holdout draws."""
    angles = np.pi * (1.0 + np.asarray(angle_errors, dtype=float))
    nz = np.asarray(axis_z_values, dtype=float)
    phases = float(base_phase) + np.asarray(phase_errors, dtype=float)
    transverse = np.sqrt(np.maximum(0.0, 1.0 - nz * nz))
    nx = transverse * np.cos(phases)
    ny = transverse * np.sin(phases)
    cosine = np.cos(angles / 2.0)
    sine = np.sin(angles / 2.0)
    matrices = np.empty((angles.size, 2, 2), dtype=complex)
    matrices[:, 0, 0] = cosine - 1.0j * sine * nz
    matrices[:, 0, 1] = -1.0j * sine * (nx - 1.0j * ny)
    matrices[:, 1, 0] = -1.0j * sine * (nx + 1.0j * ny)
    matrices[:, 1, 1] = cosine + 1.0j * sine * nz
    return matrices


def coherent_xy8_contrast(
    xy8_blocks,
    coherent_comparison: dict,
    *,
    angle_std_multiplier: float = 1.0,
    axis_z_std_multiplier: float = 1.0,
    axis_in_plane_std_multiplier: float = 1.0,
) -> np.ndarray:
    """Predict signed Sx preservation using the paper's frozen error model.

    Each seeded sample represents one NV centre whose angle and axis errors
    remain fixed for the complete sequence.  Repeating the one-block XY8
    unitary with a matrix power is exactly equivalent to composing all 8N
    pulses and makes the dense frozen-holdout calculation inexpensive.
    """
    blocks = np.asarray(xy8_blocks, dtype=int)
    if blocks.ndim != 1 or blocks.size == 0 or np.any(blocks < 1):
        raise ValueError("xy8_blocks must contain positive integers.")
    if any(
        not np.isfinite(multiplier) or multiplier <= 0.0
        for multiplier in [
            angle_std_multiplier,
            axis_z_std_multiplier,
            axis_in_plane_std_multiplier,
        ]
    ):
        raise ValueError("Coherent uncertainty multipliers must be positive and finite.")

    sample_count = int(coherent_comparison["sample_count"])
    seed = int(coherent_comparison["random_seed"])
    if sample_count < 1:
        raise ValueError("The coherent holdout sample count must be positive.")
    rng = np.random.default_rng(seed)
    standard_samples = rng.normal(size=(sample_count, 3))
    angle_errors = standard_samples[:, 0] * float(
        coherent_comparison["rotation_angle_fractional_error_std"]
    ) * float(angle_std_multiplier)
    axis_z_values = standard_samples[:, 1] * float(
        coherent_comparison["rotation_axis_z_component_std"]
    ) * float(axis_z_std_multiplier)
    in_plane_values = standard_samples[:, 2] * float(
        coherent_comparison["rotation_axis_in_plane_component_std"]
    ) * float(axis_in_plane_std_multiplier)
    transverse = np.sqrt(np.maximum(0.0, 1.0 - axis_z_values * axis_z_values))
    if np.any(np.abs(in_plane_values) >= transverse):
        raise ValueError("A sampled in-plane axis error is not physical.")
    phase_errors = np.arcsin(in_plane_values / transverse)

    initial_state = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    requested_depths = {
        int(depth): np.flatnonzero(blocks == depth) for depth in np.unique(blocks)
    }
    numerator = np.zeros(blocks.size, dtype=float)
    hahn_reference = 0.0
    block_unitary = np.broadcast_to(np.eye(2, dtype=complex), (sample_count, 2, 2)).copy()
    for phase in coherent.XY8_BLOCK_PHASES_RAD:
        pulse = _batched_rotation_unitary(
            angle_errors, axis_z_values, phase_errors, phase
        )
        block_unitary = np.einsum("sij,sjk->sik", pulse, block_unitary)

    state = np.broadcast_to(initial_state, (sample_count, 2)).copy()
    for depth in range(1, int(np.max(blocks)) + 1):
        state = np.einsum("sij,sj->si", block_unitary, state)
        if depth in requested_depths:
            sx_expectation = 2.0 * np.real(np.conj(state[:, 0]) * state[:, 1])
            numerator[requested_depths[depth]] = float(np.mean(sx_expectation))

    hahn = _batched_rotation_unitary(
        angle_errors, axis_z_values, phase_errors, 0.0
    )
    hahn_state = np.einsum("sij,j->si", hahn, initial_state)
    hahn_reference = float(
        np.mean(2.0 * np.real(np.conj(hahn_state[:, 0]) * hahn_state[:, 1]))
    )
    if hahn_reference <= 0.0:
        raise ValueError("The predeclared Hahn-reference contrast is non-positive.")
    return numerator / hahn_reference


def coherent_uncertainty_envelope(xy8_blocks, coherent_comparison: dict) -> dict:
    """Evaluate the frozen 3x3x3 error-scale sensitivity envelope."""
    multipliers = [0.75, 1.0, 1.25]
    curves = []
    labels = []
    for angle_multiplier in multipliers:
        for axis_z_multiplier in multipliers:
            for in_plane_multiplier in multipliers:
                curves.append(
                    coherent_xy8_contrast(
                        xy8_blocks,
                        coherent_comparison,
                        angle_std_multiplier=angle_multiplier,
                        axis_z_std_multiplier=axis_z_multiplier,
                        axis_in_plane_std_multiplier=in_plane_multiplier,
                    )
                )
                labels.append(
                    [angle_multiplier, axis_z_multiplier, in_plane_multiplier]
                )
    curves = np.asarray(curves, dtype=float)
    nominal_index = labels.index([1.0, 1.0, 1.0])
    return {
        "nominal": curves[nominal_index],
        "low": np.min(curves, axis=0),
        "high": np.max(curves, axis=0),
        "curves": curves,
        "multipliers": labels,
    }


def _average_ranks(values) -> np.ndarray:
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
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    if np.ptp(rx) == 0.0 or np.ptp(ry) == 0.0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def evaluate_holdout(manifest: dict, measurements: list[dict]) -> dict:
    comparison = manifest["predeclared_comparison"]
    fidelity = float(comparison["baseline_per_pulse_fidelity"])
    fidelity_min, fidelity_max = map(float, comparison["uncertainty_range"])
    reference_pulses = int(comparison["normalization_reference_pi_pulses"])

    blocks = np.asarray([row["xy8_blocks"] for row in measurements], dtype=int)
    pulses = np.asarray([row["total_pi_pulses"] for row in measurements], dtype=int)
    observed = np.asarray([row["relative_contrast"] for row in measurements], dtype=float)
    error_lower = np.asarray(
        [np.hypot(row["reported_error_lower"], row["digitization_error"]) for row in measurements]
    )
    error_upper = np.asarray(
        [np.hypot(row["reported_error_upper"], row["digitization_error"]) for row in measurements]
    )

    effective_pulses = pulses - reference_pulses
    predicted = scalar_pulse_survival(effective_pulses, fidelity)
    predicted_low = scalar_pulse_survival(effective_pulses, fidelity_min)
    predicted_high = scalar_pulse_survival(effective_pulses, fidelity_max)
    overlap = ((observed + error_upper) >= predicted_low) & (
        (observed - error_lower) <= predicted_high
    )

    dense_pulses = np.arange(8, 1025, 8, dtype=int)
    dense_effective_pulses = dense_pulses - reference_pulses
    dense_baseline = scalar_pulse_survival(dense_effective_pulses, fidelity)
    dense_low = scalar_pulse_survival(dense_effective_pulses, fidelity_min)
    dense_high = scalar_pulse_survival(dense_effective_pulses, fidelity_max)

    coherent_comparison = manifest["predeclared_coherent_comparison"]
    coherent_measured = coherent_uncertainty_envelope(blocks, coherent_comparison)
    dense_blocks = dense_pulses // 8
    coherent_dense = coherent_uncertainty_envelope(dense_blocks, coherent_comparison)
    coherent_predicted = coherent_measured["nominal"]
    coherent_low = coherent_measured["low"]
    coherent_high = coherent_measured["high"]
    coherent_overlap = ((observed + error_upper) >= coherent_low) & (
        (observed - error_lower) <= coherent_high
    )

    rmse = float(np.sqrt(np.mean((predicted - observed) ** 2)))
    observed_span = float(np.ptp(observed))
    normalized_rmse = rmse / observed_span if observed_span > 0.0 else float("nan")
    rho = spearman_rho(predicted, observed)
    coherent_rmse = float(np.sqrt(np.mean((coherent_predicted - observed) ** 2)))
    coherent_normalized_rmse = (
        coherent_rmse / observed_span if observed_span > 0.0 else float("nan")
    )
    coherent_rho = spearman_rho(coherent_predicted, observed)

    implied_fidelity = np.ones_like(observed)
    positive = effective_pulses > 0
    implied_fidelity[positive] = np.clip(observed[positive], 1e-12, None) ** (
        1.0 / effective_pulses[positive]
    )

    threshold = 0.5
    predicted_below = dense_pulses[dense_baseline < threshold]
    predicted_first_below = int(predicted_below[0]) if predicted_below.size else None
    coherent_predicted_below = dense_pulses[coherent_dense["nominal"] < threshold]
    coherent_predicted_first_below = (
        int(coherent_predicted_below[0]) if coherent_predicted_below.size else None
    )
    observed_below_indices = np.flatnonzero(observed < threshold)
    observed_first_below = (
        int(pulses[observed_below_indices[0]]) if observed_below_indices.size else None
    )
    if observed_below_indices.size:
        crossing_index = int(observed_below_indices[0])
        observed_crossing_bracket = [
            int(pulses[max(0, crossing_index - 1)]),
            int(pulses[crossing_index]),
        ]
    else:
        observed_crossing_bracket = None

    curve_rows = []
    for index, row in enumerate(measurements):
        curve_rows.append(
            {
                "xy8_blocks": int(blocks[index]),
                "total_pi_pulses": int(pulses[index]),
                "measured_relative_contrast": float(observed[index]),
                "measured_error_lower_combined": float(error_lower[index]),
                "measured_error_upper_combined": float(error_upper[index]),
                "predicted_scalar_contrast_nominal": float(predicted[index]),
                "predicted_scalar_contrast_fidelity_0p995": float(predicted_low[index]),
                "predicted_scalar_contrast_fidelity_0p999": float(predicted_high[index]),
                "measured_interval_overlaps_predeclared_model_range": int(overlap[index]),
                "predicted_coherent_contrast_nominal": float(coherent_predicted[index]),
                "predicted_coherent_contrast_low": float(coherent_low[index]),
                "predicted_coherent_contrast_high": float(coherent_high[index]),
                "measured_interval_overlaps_coherent_uncertainty_range": int(
                    coherent_overlap[index]
                ),
                "implied_independent_per_pulse_fidelity": float(implied_fidelity[index]),
                "source_figure": row["source_figure"],
            }
        )

    metrics = {
        "validation_status": "partial_agreement_model_refinement_needed",
        "coherent_validation_status": "coherent_model_does_not_improve_frozen_holdout",
        "figure_title": "Independent XY8 data identify where the model needs refinement",
        "dataset_selected_before_model_evaluation": True,
        "sequence": "conventional XY8",
        "observable": manifest["observable_and_scope"]["measured_observable"],
        "number_of_depths": int(len(measurements)),
        "measured_xy8_blocks": blocks.tolist(),
        "measured_total_pi_pulses": pulses.tolist(),
        "baseline_per_pulse_fidelity": fidelity,
        "normalization_reference_pi_pulses": reference_pulses,
        "predeclared_fidelity_range": [fidelity_min, fidelity_max],
        "spearman_rank_correlation": rho,
        "rmse_relative_contrast": rmse,
        "normalized_rmse_by_measured_span": normalized_rmse,
        "points_overlapping_predeclared_model_range": int(np.sum(overlap)),
        "fraction_points_overlapping_predeclared_model_range": float(np.mean(overlap)),
        "predicted_first_depth_below_half_contrast_total_pi_pulses": predicted_first_below,
        "measured_first_sample_below_half_contrast_total_pi_pulses": observed_first_below,
        "measured_half_contrast_crossing_bracket_total_pi_pulses": observed_crossing_bracket,
        "implied_per_pulse_fidelity_range_all_depths": [
            float(np.min(implied_fidelity)),
            float(np.max(implied_fidelity)),
        ],
        "implied_per_pulse_fidelity_range_nontrivial_depths": [
            float(np.min(implied_fidelity[1:])),
            float(np.max(implied_fidelity[1:])),
        ],
        "checks": {
            "correct_monotonic_ordering": bool(rho > 0.99),
            "all_points_covered_by_predeclared_fidelity_range": bool(np.all(overlap)),
            "nominal_half_contrast_crossing_inside_measured_bracket": bool(
                observed_crossing_bracket is not None
                and predicted_first_below is not None
                and observed_crossing_bracket[0] <= predicted_first_below <= observed_crossing_bracket[1]
            ),
        },
        "coherent_model": {
            "freeze_status": coherent_comparison["freeze_status"],
            "fitting": coherent_comparison["fitting"],
            "sampling_method": coherent_comparison["sampling_method"],
            "sample_count": int(coherent_comparison["sample_count"]),
            "random_seed": int(coherent_comparison["random_seed"]),
            "uncertainty_combinations": int(
                coherent_measured["curves"].shape[0]
            ),
            "spearman_rank_correlation": coherent_rho,
            "rmse_relative_contrast": coherent_rmse,
            "normalized_rmse_by_measured_span": coherent_normalized_rmse,
            "points_overlapping_predeclared_model_range": int(
                np.sum(coherent_overlap)
            ),
            "fraction_points_overlapping_predeclared_model_range": float(
                np.mean(coherent_overlap)
            ),
            "predicted_first_depth_below_half_contrast_total_pi_pulses": (
                coherent_predicted_first_below
            ),
            "nominal_predictions_at_measured_depths": coherent_predicted.tolist(),
            "uncertainty_low_at_measured_depths": coherent_low.tolist(),
            "uncertainty_high_at_measured_depths": coherent_high.tolist(),
            "checks": {
                "correct_monotonic_ordering": bool(coherent_rho > 0.99),
                "all_points_covered_by_predeclared_range": bool(
                    np.all(coherent_overlap)
                ),
                "nominal_half_contrast_crossing_inside_measured_bracket": bool(
                    observed_crossing_bracket is not None
                    and coherent_predicted_first_below is not None
                    and observed_crossing_bracket[0]
                    <= coherent_predicted_first_below
                    <= observed_crossing_bracket[1]
                ),
            },
            "parameters": coherent_comparison,
        },
        "model_comparison": {
            "lower_normalized_rmse_model": (
                "coherent_xy8"
                if coherent_normalized_rmse < normalized_rmse
                else "scalar_pulse_survival"
            ),
            "normalized_rmse_improvement_fraction": float(
                (normalized_rmse - coherent_normalized_rmse) / normalized_rmse
            ),
            "coherent_has_better_normalized_rmse": bool(
                coherent_normalized_rmse < normalized_rmse
            ),
            "coherent_has_equal_or_better_pointwise_coverage": bool(
                np.sum(coherent_overlap) >= np.sum(overlap)
            ),
        },
        "interpretation": (
            "Both frozen predictions reproduce the measured direction of change, but neither reproduces "
            "the three-point curve quantitatively. The sequence-specific coherent model built from the "
            "paper's reported Gaussian angle and axis errors predicts substantially too much retained "
            "contrast at 72 and 584 pulses. Its normalized RMSE is "
            f"{coherent_normalized_rmse:.3f}, compared with {normalized_rmse:.3f} for the scalar "
            "baseline, and its uncertainty envelope overlaps fewer measurements. It therefore does not "
            "pass the predeclared improvement test. The outcome is consistent with the source paper's "
            "own statement that its pulse-error simulation overestimated experimental contrast. The "
            "remaining discrepancy points to missing loss, error correlations, preparation/readout "
            "imperfections, or other ensemble physics; these must be introduced from independent "
            "measurements rather than fitted to this three-point holdout."
        ),
        "scope_guardrail": (
            "This holdout isolates XY8 pulse-error accumulation in a short-free-evolution state-"
            "preservation experiment. It does not test target-frequency selection, absolute sensitivity, "
            "coherence scaling, photon collection, or preparation/readout overhead."
        ),
        "no_fit_statement": comparison["no_fit_statement"],
        "source": manifest["source"],
    }

    return {
        "blocks": blocks,
        "pulses": pulses,
        "observed": observed,
        "error_lower": error_lower,
        "error_upper": error_upper,
        "predicted": predicted,
        "predicted_low": predicted_low,
        "predicted_high": predicted_high,
        "implied_fidelity": implied_fidelity,
        "dense_pulses": dense_pulses,
        "dense_baseline": dense_baseline,
        "dense_low": dense_low,
        "dense_high": dense_high,
        "coherent_predicted": coherent_predicted,
        "coherent_low": coherent_low,
        "coherent_high": coherent_high,
        "coherent_overlap": coherent_overlap,
        "coherent_dense_baseline": coherent_dense["nominal"],
        "coherent_dense_low": coherent_dense["low"],
        "coherent_dense_high": coherent_dense["high"],
        "curve_rows": curve_rows,
        "metrics": metrics,
    }
