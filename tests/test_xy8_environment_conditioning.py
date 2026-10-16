from __future__ import annotations

import csv
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

import analyze_xy8_environment_transitions as exporter
import xy8_environment_conditioning as conditioning
import xy8_environment_scan_config as scan_config
import xy8_nv_environment_literature as literature


def test_reference_hamiltonian_and_transition_match_axial_analytic_result():
    config = scan_config.load_config()
    condition = conditioning.reference_condition(config)
    hamiltonian = conditioning.spin_hamiltonian_MHz(condition, config=config)
    d_300 = float(literature.cambria_zfs_MHz(300.0))
    zeeman = conditioning.ELECTRON_GYROMAGNETIC_RATIO_MHZ_PER_T * 0.02
    assert np.diag(hamiltonian).real.tolist() == pytest.approx(
        [d_300 + zeeman, 0.0, d_300 - zeeman]
    )
    assert np.allclose(hamiltonian - np.diag(np.diag(hamiltonian)), 0.0)
    result = conditioning.evaluate_transition(condition, config=config)
    assert result.transition_frequency_MHz == pytest.approx(d_300 - zeeman)
    assert result.transition_shift_Hz == pytest.approx(0.0, abs=1e-6)
    assert result.spectrum.minimum_assigned_overlap == pytest.approx(1.0)
    assert not result.spectrum.ambiguous


def test_temperature_shift_at_zero_strain_equals_change_in_D():
    config = scan_config.load_config()
    condition = conditioning.EnvironmentCondition(330.0, np.zeros((3, 3)))
    result = conditioning.evaluate_transition(condition, config=config)
    expected_Hz = float(
        (literature.cambria_zfs_MHz(330.0) - literature.cambria_zfs_MHz(300.0)) * 1e6
    )
    assert result.transition_shift_Hz == pytest.approx(expected_Hz, abs=1e-5)


def test_small_axial_strain_shift_uses_frozen_h43_coefficient():
    config = scan_config.load_config()
    strain = conditioning.strain_tensor_from_path("axial_zz", 100.0, config=config)
    result = conditioning.evaluate_transition(
        conditioning.EnvironmentCondition(300.0, strain), config=config
    )
    h43 = literature.load_spin_strain_coefficients()["h43"]["value_MHz_per_strain"]
    assert result.transition_shift_Hz == pytest.approx(h43 * 100e-6 * 1e6, abs=1e-5)


def test_carrier_policies_report_fixed_shift_and_zero_retracked_detuning():
    result = conditioning.evaluate_transition(
        conditioning.EnvironmentCondition(320.0, np.zeros((3, 3)))
    )
    assert conditioning.carrier_detuning_Hz(result, "fixed_reference_carrier") == pytest.approx(
        result.transition_shift_Hz
    )
    assert conditioning.carrier_detuning_Hz(result, "ideal_retracked_carrier") == 0.0
    assert result.fixed_reference_carrier_detuning_rad_s == pytest.approx(
        2.0 * math.pi * result.transition_shift_Hz
    )
    with pytest.raises(ValueError, match="Unknown carrier"):
        conditioning.carrier_detuning_Hz(result, "other")


def test_strain_path_builder_enforces_tensor_symmetry_and_frozen_range():
    shear = conditioning.strain_tensor_from_path("shear_xz", -80.0)
    assert shear[0, 2] == shear[2, 0] == pytest.approx(-80e-6)
    assert np.array_equal(shear, shear.T)
    with pytest.raises(ValueError, match="outside"):
        conditioning.strain_tensor_from_path("axial_zz", 101.0)
    with pytest.raises(ValueError, match="zero amplitude"):
        conditioning.strain_tensor_from_path("zero_strain", 20.0)


def test_path_tracking_is_continuous_from_the_zero_strain_anchor():
    amplitudes = [-100.0, -50.0, 0.0, 50.0, 100.0]
    conditions = [
        conditioning.EnvironmentCondition(
            300.0, conditioning.strain_tensor_from_path("transverse_differential", amplitude)
        )
        for amplitude in amplitudes
    ]
    results = conditioning.track_environment_path(conditions, anchor_index=2)
    assert len(results) == len(conditions)
    assert all(result.spectrum.minimum_assigned_overlap > 0.999 for result in results)
    assert all(not result.spectrum.ambiguous for result in results)
    for left, right in zip(results, results[1:]):
        sequential_overlap = np.abs(left.spectrum.eigenvectors.conj().T @ right.spectrum.eigenvectors) ** 2
        assert np.min(np.diag(sequential_overlap)) > 0.999


def test_out_of_range_inputs_and_ambiguous_tracking_are_reported():
    with pytest.warns(RuntimeWarning, match="temperature_outside_D_law_range"):
        outside = conditioning.evaluate_transition(
            conditioning.EnvironmentCondition(600.0, np.zeros((3, 3))), emit_warnings=True
        )
    assert any("temperature_outside_D_law_range" in message for message in outside.warnings)

    # A deliberately large transverse perturbation is outside the frozen scan
    # and tests the ambiguity guard without changing the preregistered range.
    large = conditioning.strain_tensor_from_path(
        "transverse_differential", 200_000.0, enforce_preregistered_range=False
    )
    mixed = conditioning.evaluate_transition(
        conditioning.EnvironmentCondition(300.0, large)
    )
    assert any("strain_outside_preregistered" in message for message in mixed.warnings)
    assert mixed.spectrum.ambiguous
    assert any("ambiguous_state_tracking" in message for message in mixed.warnings)


def test_stage2_export_is_deterministic_and_contains_no_edt_scores(tmp_path):
    csv_path, json_path, summary = exporter.write_outputs(tmp_path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    stored = json.loads(json_path.read_text(encoding="utf-8"))
    assert stored == summary
    assert summary["edt_score_surfaces_evaluated"] is False
    assert summary["diagnostic_row_count"] == len(rows) == 44
    assert summary["ambiguous_state_count"] == 0
    assert summary["warning_row_count"] == 0
    assert {row["scan_family"] for row in rows} == {
        "operational_temperature",
        "strain_at_reference_temperature",
    }
    assert "eta" not in rows[0]


def test_committed_stage2_artifacts_match_the_frozen_configuration():
    csv_path = ROOT / "data" / "processed" / exporter.CSV_NAME
    json_path = ROOT / "data" / "processed" / exporter.JSON_NAME
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    freeze = scan_config.verify_freeze_record()
    assert summary["configuration_sha256"] == freeze["configuration_sha256"]
    assert summary["diagnostic_row_count"] == len(rows) == 44
    assert summary["edt_score_surfaces_evaluated"] is False
