from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "code" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import xy8_environment_scan_config as scan


def config():
    return scan.load_config()


def test_frozen_scan_configuration_and_dependencies_verify():
    result = scan.verify_freeze_record()
    assert result["verified"] is True
    assert all(result["checks"].values())


def test_reference_environment_is_in_every_axis():
    data = config()
    assert data["reference_environment"]["temperature_K"] == 300.0
    assert np.allclose(data["reference_environment"]["strain_tensor_nv_frame"], 0.0)
    assert all(300.0 in axis["values_K"] for axis in data["temperature_axes"].values())
    assert 0.0 in data["strain_axes"]["amplitudes_microstrain"]


def test_strain_paths_are_symmetric_and_one_parameter_at_a_time():
    data = config()
    paths = {path["id"]: np.asarray(path["tensor_per_unit_amplitude"]) for path in data["strain_axes"]["paths"]}
    assert set(paths) == {"zero_strain", "axial_zz", "transverse_differential", "shear_xz"}
    assert all(np.array_equal(tensor, tensor.T) for tensor in paths.values())
    assert np.array_equal(paths["axial_zz"], np.diag([0.0, 0.0, 1.0]))
    assert np.array_equal(paths["transverse_differential"], np.diag([1.0, -1.0, 0.0]))
    assert paths["shear_xz"][0, 2] == paths["shear_xz"][2, 0] == 1.0
    assert "one strain path at a time" in data["strain_axes"]["combination_policy"]


def test_resonance_depth_and_xy8_conventions_are_frozen():
    data = config()
    signal = data["signal_and_control"]
    assert signal["primary_resonant_tau_s"] == pytest.approx(1.0 / (4.0 * signal["target_frequency_Hz"]))
    assert "8N" in signal["depth_convention"]
    assert data["control_grids"]["depths_N"] == [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20, 24, 28,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 48, 56, 64, 80, 96, 128,
    ]


def test_both_carrier_policies_and_double_counting_guard_are_declared():
    data = config()
    assert {policy["id"] for policy in data["carrier_policies"]} == {
        "fixed_reference_carrier",
        "ideal_retracked_carrier",
    }
    integration = data["signal_and_control"]["environmental_detuning_integration"]
    assert "G(detuning)/G(reference detuning=0)" in integration["method"]
    assert "only the relative environmental detuning penalty" in integration["double_counting_guard"]
    assert "Do not multiply" in data["relaxation_policy"]["double_counting_guard"]


def test_refinement_rules_are_predeclared_and_topology_thresholds_are_complete():
    data = config()
    topology = data["topology_analysis"]
    assert topology["relative_limits"] == [1.01, 1.05, 1.1, 1.25, 1.5, 2.0]
    assert topology["connectivity_primary"] == 4
    assert topology["connectivity_sensitivity"] == 8
    rules = topology["refinement_selection_rules"]
    assert any("beta_0 differs" in rule for rule in rules)
    assert any("sweep boundary" in rule for rule in rules)
    assert any("recommended N category changes" in rule for rule in rules)


def test_validator_rejects_post_hoc_tuning_and_source_extrapolation():
    tuned = copy.deepcopy(config())
    tuned["preregistration"]["post_freeze_parameter_tuning_allowed"] = True
    with pytest.raises(ValueError, match="tuning"):
        scan.validate_config(tuned, verify_dependencies=False)

    extrapolated = copy.deepcopy(config())
    extrapolated["temperature_axes"]["extended_stress_test"]["values_K"][0] = 75.0
    with pytest.raises(ValueError, match="extrapolate"):
        scan.validate_config(extrapolated, verify_dependencies=False)


def test_validator_rejects_non_symmetric_strain_and_missing_exact_resonance():
    bad_strain = copy.deepcopy(config())
    bad_strain["strain_axes"]["paths"][3]["tensor_per_unit_amplitude"][2][0] = 0.0
    with pytest.raises(ValueError, match="symmetric"):
        scan.validate_config(bad_strain, verify_dependencies=False)

    missing_resonance = copy.deepcopy(config())
    missing_resonance["control_grids"]["discovery_timing_grid"]["insert_exact_odd_harmonic_resonances"] = False
    with pytest.raises(ValueError, match="exact resonances"):
        scan.validate_config(missing_resonance, verify_dependencies=False)
