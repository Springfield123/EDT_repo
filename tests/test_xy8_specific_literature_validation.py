from __future__ import annotations

from copy import deepcopy
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "code" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import xy8_specific_literature_validation as validation
import xy8_coherent_control as coherent


MANIFEST_PATH = (
    ROOT
    / "data"
    / "raw"
    / "nv"
    / "literature_validation"
    / "farfurnik_2015_xy8_manifest.json"
)
DATA_PATH = (
    ROOT
    / "data"
    / "raw"
    / "nv"
    / "literature_validation"
    / "farfurnik_2015_fig3b_xy8.csv"
)


@pytest.fixture(scope="module")
def frozen_holdout():
    manifest = validation.load_manifest(MANIFEST_PATH)
    measurements = validation.load_measurements(DATA_PATH)
    return manifest, measurements


@pytest.fixture(scope="module")
def evaluated_holdout(frozen_holdout):
    manifest, measurements = frozen_holdout
    return validation.evaluate_holdout(manifest, measurements)


def test_dataset_was_frozen_before_model_evaluation(frozen_holdout):
    manifest, _ = frozen_holdout
    assert manifest["selection_status"] == "frozen_before_validation_model_evaluation"
    assert manifest["source"]["sequence"] == "conventional XY8"
    assert manifest["predeclared_comparison"]["fitting"] == "none"
    coherent_comparison = manifest["predeclared_coherent_comparison"]
    assert coherent_comparison["freeze_status"] == "frozen_before_coherent_model_evaluation"
    assert coherent_comparison["fitting"] == "none"
    assert coherent_comparison["sample_count"] == 10000
    assert coherent_comparison["random_seed"] == 150500636
    assert coherent_comparison["rotation_angle_fractional_error_std"] == 0.06
    assert coherent_comparison["rotation_axis_z_component_std"] == 0.06
    assert coherent_comparison["rotation_axis_in_plane_component_std"] == 0.05


def test_every_depth_is_a_complete_xy8_block(frozen_holdout):
    _, measurements = frozen_holdout
    assert [row["xy8_blocks"] for row in measurements] == [1, 9, 73]
    assert [row["total_pi_pulses"] for row in measurements] == [8, 72, 584]
    assert all(row["total_pi_pulses"] == 8 * row["xy8_blocks"] for row in measurements)


def test_digitized_curve_is_depth_dependent_with_reported_errors(frozen_holdout):
    _, measurements = frozen_holdout
    contrast = np.asarray([row["relative_contrast"] for row in measurements])
    assert np.all(np.diff(contrast) < 0.0)
    assert all(row["reported_error_lower"] > 0 for row in measurements)
    assert all(row["reported_error_upper"] > 0 for row in measurements)


def test_scalar_survival_matches_project_convention():
    pulses = np.asarray([8, 72, 584])
    predicted = validation.scalar_pulse_survival(pulses, 0.997)
    assert np.allclose(predicted, 0.997**pulses)


def test_holdout_exposes_partial_not_complete_agreement(evaluated_holdout):
    result = evaluated_holdout
    metrics = result["metrics"]
    assert metrics["validation_status"] == "partial_agreement_model_refinement_needed"
    assert metrics["spearman_rank_correlation"] == pytest.approx(1.0)
    assert metrics["points_overlapping_predeclared_model_range"] == 2
    assert not metrics["checks"]["all_points_covered_by_predeclared_fidelity_range"]
    assert metrics["checks"]["nominal_half_contrast_crossing_inside_measured_bracket"]
    assert metrics["normalized_rmse_by_measured_span"] == pytest.approx(0.286, abs=0.003)


def test_vectorized_holdout_rotation_matches_core_propagator():
    angle_errors = np.array([-0.08, 0.0, 0.04])
    axis_z = np.array([0.03, -0.06, 0.1])
    phase_errors = np.array([-0.02, 0.01, 0.05])
    phase = np.pi / 2.0
    batched = validation._batched_rotation_unitary(
        angle_errors, axis_z, phase_errors, phase
    )
    for index in range(angle_errors.size):
        expected = coherent.rotation_unitary(
            np.pi * (1.0 + angle_errors[index]),
            phase + phase_errors[index],
            axis_z=axis_z[index],
        )
        assert batched[index] == pytest.approx(expected, abs=2e-15)


def test_zero_error_coherent_holdout_preserves_sx_exactly(frozen_holdout):
    manifest, _ = frozen_holdout
    comparison = deepcopy(manifest["predeclared_coherent_comparison"])
    comparison["rotation_angle_fractional_error_std"] = 0.0
    comparison["rotation_axis_z_component_std"] = 0.0
    comparison["rotation_axis_in_plane_component_std"] = 0.0
    predicted = validation.coherent_xy8_contrast([1, 9, 73], comparison)
    assert predicted == pytest.approx([1.0, 1.0, 1.0], abs=2e-14)


def test_coherent_holdout_is_seeded_and_uncertainty_contains_nominal(frozen_holdout):
    manifest, _ = frozen_holdout
    comparison = manifest["predeclared_coherent_comparison"]
    first = validation.coherent_uncertainty_envelope([1, 9, 73], comparison)
    repeated = validation.coherent_uncertainty_envelope([1, 9, 73], comparison)
    assert np.array_equal(first["nominal"], repeated["nominal"])
    assert first["curves"].shape == (27, 3)
    assert np.all(first["low"] <= first["nominal"])
    assert np.all(first["nominal"] <= first["high"])


def test_frozen_coherent_model_does_not_pass_improvement_gate(evaluated_holdout):
    metrics = evaluated_holdout["metrics"]
    coherent_metrics = metrics["coherent_model"]
    comparison = metrics["model_comparison"]
    assert metrics["coherent_validation_status"] == (
        "coherent_model_does_not_improve_frozen_holdout"
    )
    assert coherent_metrics["nominal_predictions_at_measured_depths"] == pytest.approx(
        [1.0121398621, 0.9981798095, 0.8717147571], rel=2e-9
    )
    assert coherent_metrics["spearman_rank_correlation"] == pytest.approx(1.0)
    assert coherent_metrics["normalized_rmse_by_measured_span"] == pytest.approx(
        0.6137427009, rel=2e-9
    )
    assert coherent_metrics["points_overlapping_predeclared_model_range"] == 1
    assert coherent_metrics[
        "predicted_first_depth_below_half_contrast_total_pi_pulses"
    ] is None
    assert comparison["lower_normalized_rmse_model"] == "scalar_pulse_survival"
    assert not comparison["coherent_has_better_normalized_rmse"]
    assert not comparison["coherent_has_equal_or_better_pointwise_coverage"]


def test_curve_rows_report_both_frozen_models(evaluated_holdout):
    rows = evaluated_holdout["curve_rows"]
    assert len(rows) == 3
    assert all("predicted_scalar_contrast_nominal" in row for row in rows)
    assert all("predicted_coherent_contrast_nominal" in row for row in rows)
    assert [row["measured_interval_overlaps_coherent_uncertainty_range"] for row in rows] == [
        1,
        0,
        0,
    ]


def test_manifest_source_digests_are_recorded(frozen_holdout):
    manifest, _ = frozen_holdout
    assert len(manifest["source"]["source_archive_sha256"]) == 64
    assert len(manifest["source"]["source_figure_sha256"]) == 64
    json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
