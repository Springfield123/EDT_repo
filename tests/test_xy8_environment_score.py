from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
ANALYSIS_DIR = CODE_DIR / "analysis"
for path in (CODE_DIR, ANALYSIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import analyze_xy8_environment_scores as exporter
import nvratemodel_api as model
import xy8_environment_coherence as coherence
import xy8_environment_score as score


def scalar_gain(tau, depth, detuning_Hz, *, pulse_substeps=8):
    config = {
        "pi_pulse_duration_s": 40e-9,
        "rotation_angle_error_mean": 0.0,
        "rotation_angle_error_std": 0.0,
        "detuning_mean_angular_frequency_rad_s": 2.0 * math.pi * detuning_Hz,
        "detuning_std_angular_frequency_rad_s": 0.0,
        "quadrature_order": 1,
        "pulse_substeps": pulse_substeps,
    }
    return model.coherent_xy8_control_response(200e3, tau, depth, config).normalized_gain


@pytest.mark.parametrize(
    "tau,depth,detuning_Hz",
    [(1.25e-6, 1, 0.0), (1.25e-6, 20, 1e6), (3.75e-6, 8, -2e6)],
)
def test_batched_coherent_gain_matches_existing_scalar_propagator(tau, depth, detuning_Hz):
    batched = score.batched_deterministic_xy8_gains(
        [tau], [depth], [detuning_Hz],
        signal_frequency_hz=200e3,
        pi_pulse_duration_s=40e-9,
        pulse_substeps=8,
    )[0, 0, 0]
    assert batched == pytest.approx(scalar_gain(tau, depth, detuning_Hz), rel=2e-7, abs=1e-10)


def test_environment_point_grid_matches_frozen_operational_design():
    points = score.build_operational_environment_points()
    assert len(points) == 11 + 3 * 11 * 11 == 374
    assert len({point.condition_id for point in points}) == len(points)
    assert all(not point.transition.spectrum.ambiguous for point in points)
    assert all(not point.transition.warnings for point in points)


@pytest.mark.parametrize("variant", coherence.SUPPORTED_VARIANTS)
def test_reference_coherence_surface_exactly_matches_canonical(variant):
    canonical = score.load_canonical_surface()
    nominal = {"T2_1": 300e-6, "alpha": 0.6, "beta": 0.0, "tau0": 1e-6, "p": 2.0}
    conditioned, result = score.coherence_conditioned_surface(
        canonical, 300.0, variant=variant, nominal_t2_params=nominal
    )
    assert result.effective_T2_s == pytest.approx(300e-6)
    assert np.array_equal(conditioned, canonical.eta)


def test_surface_summary_recovers_canonical_optimum_and_window():
    canonical = score.load_canonical_surface()
    summary = score.summarize_surface(canonical.eta, canonical, relative_limit=1.10)
    assert summary["best_tau_s"] == pytest.approx(1.25e-6)
    assert summary["best_N"] == 20
    assert (summary["window_N_min"], summary["window_N_max"]) == (10, 34)
    assert not summary["best_touches_boundary"]
    assert not summary["window_touches_N_boundary"]


def test_committed_stage4_outputs_are_consistent_and_reference_is_exact():
    summary_path = ROOT / "data" / "processed" / exporter.SUMMARY_CSV
    surfaces_path = ROOT / "data" / "processed" / exporter.SURFACES_NPZ
    metadata_path = ROOT / "data" / "processed" / exporter.METADATA_JSON
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    arrays = np.load(surfaces_path)
    assert metadata["dimensions"]["surface_count"] == len(rows) == 1496
    assert arrays["eta"].shape == (1496, 37, 32)
    assert arrays["taus_s"].shape == (37,)
    assert arrays["depths_N"].shape == (32,)
    assert metadata["reference_regression"]["maximum_absolute_eta_error"] < 1e-14
    assert metadata["reference_regression"]["row_count"] == 16
    assert metadata["reference_regression"]["all_reference_rows_match_expected_summary"]
    assert all(int(row["surface_index"]) == index for index, row in enumerate(rows))
    assert metadata["summary_csv_sha256"] == hashlib.sha256(summary_path.read_bytes()).hexdigest()
    assert metadata["surfaces_npz_sha256"] == hashlib.sha256(surfaces_path.read_bytes()).hexdigest()
    assert all(int(row["best_pi_pulses"]) == 8 * int(row["best_N"]) for row in rows)
    assert {row["boundary_status"] for row in rows} == {
        "resolved_interior_on_discovery_grid"
    }


def test_fixed_and_retracked_surfaces_differ_away_from_reference():
    summary_path = ROOT / "data" / "processed" / exporter.SUMMARY_CSV
    surfaces_path = ROOT / "data" / "processed" / exporter.SURFACES_NPZ
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    arrays = np.load(surfaces_path)
    selected = [
        row for row in rows
        if row["condition_id"] == "zero_strain_T330K"
        and row["coherence_variant"] == coherence.JARMOLA_VARIANT
    ]
    assert len(selected) == 2
    by_policy = {row["carrier_policy"]: arrays["eta"][int(row["surface_index"])] for row in selected}
    assert not np.allclose(
        by_policy["fixed_reference_carrier"], by_policy["ideal_retracked_carrier"]
    )
