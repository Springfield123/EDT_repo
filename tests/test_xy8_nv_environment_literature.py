from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "code" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import xy8_nv_environment_literature as literature


def test_directly_tabulated_spin_strain_coefficients_are_complete():
    coefficients = literature.load_spin_strain_coefficients()
    assert set(coefficients) == {"h43", "h41", "h25", "h26", "h15", "h16"}
    assert coefficients["h16"]["value_MHz_per_strain"] == pytest.approx(19660.0)
    assert coefficients["h41"]["value_MHz_per_strain"] == pytest.approx(-6420.0)
    assert all(record["uncertainty_MHz_per_strain"] > 0 for record in coefficients.values())


def test_spin_strain_hamiltonian_is_zero_at_zero_strain_and_hermitian():
    assert np.allclose(literature.spin_strain_hamiltonian_MHz(np.zeros((3, 3))), 0.0)
    strain = np.array([[1e-5, 2e-6, -1e-6], [2e-6, -3e-6, 4e-6], [-1e-6, 4e-6, 5e-6]])
    hamiltonian = literature.spin_strain_hamiltonian_MHz(strain)
    assert np.allclose(hamiltonian, hamiltonian.conj().T)


def test_axial_strain_uses_h41_and_h43_exactly():
    coefficients = literature.load_spin_strain_coefficients()
    strain = np.diag([1e-5, 1e-5, 2e-5])
    expected = (
        coefficients["h41"]["value_MHz_per_strain"] * 2e-5
        + coefficients["h43"]["value_MHz_per_strain"] * 2e-5
    )
    hamiltonian = literature.spin_strain_hamiltonian_MHz(strain)
    assert np.diag(hamiltonian).real.tolist() == pytest.approx([expected, 0.0, expected])
    assert np.allclose(hamiltonian - np.diag(np.diag(hamiltonian)), 0.0)


def test_cambria_law_is_monotonic_and_has_expected_room_temperature_slope():
    temperatures = np.linspace(15.0, 500.0, 200)
    values = literature.cambria_zfs_MHz(temperatures)
    assert np.all(np.isfinite(values))
    # The 15 K tail changes by less than binary64 resolution; the law is
    # non-increasing there and strictly decreasing once thermally activated.
    assert np.all(np.diff(values) <= 0.0)
    assert np.all(np.diff(values[temperatures >= 50.0]) < 0.0)
    assert float(literature.cambria_zfs_MHz(0.0)) == pytest.approx(2877.38)
    slope = literature.numerical_slope_kHz_per_K(literature.cambria_zfs_MHz, 300.0)
    assert slope == pytest.approx(-70.267, abs=0.01)


def test_jarmola_law_uses_main_text_coefficient_and_validates_sample_name():
    rate = float(literature.jarmola_t1_rate_s_inv(300.0, sample="S8"))
    assert rate == pytest.approx(186.09, abs=0.1)
    assert 1000.0 / rate == pytest.approx(5.374, abs=0.01)
    with pytest.raises(ValueError, match="Unknown sample"):
        literature.jarmola_t1_rate_s_inv(300.0, sample="missing")


def test_digitized_curve_records_uncertainty_and_is_ordered():
    curve = literature.load_mondal_t1_curve()
    assert len(curve) == 12
    temperatures = np.array([row["temperature_K"] for row in curve])
    rates = np.array([row["t1_rate_s_inv"] for row in curve])
    assert np.all(np.diff(temperatures) > 0.0)
    assert np.all(np.diff(rates) > 0.0)
    assert all(row["temperature_digitization_uncertainty_K"] > 0.0 for row in curve)
    assert all(row["t1_rate_digitization_uncertainty_s_inv"] > 0.0 for row in curve)
    assert {row["extraction_status"] for row in curve} == {"vector_centerline_digitized"}


def test_independent_validation_is_separated_and_within_two_sigma():
    result = literature.validate_tang_against_acosta()
    assert result["separation_status"] == "validation_not_used_for_parameter_estimation"
    assert result["absolute_error_kHz_per_K"] == pytest.approx(0.99)
    assert result["absolute_error_in_measurement_sigma"] == pytest.approx(1.65)
    assert result["within_two_measurement_sigma"] is True


def test_manifest_forbids_post_freeze_tuning_to_edt():
    manifest = literature.load_manifest()
    assert manifest["analysis_freeze_policy"]["edt_results_consulted_during_extraction"] is False
    assert manifest["analysis_freeze_policy"]["post_freeze_parameter_tuning_allowed"] is False
    validation_sources = [source for source in manifest["sources"] if source["role"] == "external_validation_only"]
    assert len(validation_sources) == 1
    assert validation_sources[0]["parameter_estimation_permitted"] is False


def test_committed_freeze_record_matches_all_source_files():
    result = literature.verify_frozen_bundle()
    assert result["verified"] is True
    assert len(result["files"]) == 5
    assert all(item["matches"] for item in result["files"])


def test_committed_validation_report_has_passing_gate():
    path = ROOT / "data" / "processed" / "nv_environment_literature_validation.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["analysis_gate"]["status"] == "ready_for_edt_environment_analysis"
    assert report["frozen_bundle"]["verified"] is True
    assert report["independent_validation"]["within_two_measurement_sigma"] is True
