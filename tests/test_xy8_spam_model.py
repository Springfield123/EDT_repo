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

import xy8_coherent_control as coherent
import xy8_spam_model as spam


PROTOCOL_PATH = (
    ROOT / "data" / "raw" / "nv" / "calibration" / "xy8_spam_calibration_protocol.json"
)
TEMPLATE_PATH = (
    ROOT / "data" / "raw" / "nv" / "calibration" / "xy8_spam_calibration_template.json"
)


def ideal_calibration() -> spam.SPAMCalibration:
    return spam.SPAMCalibration(
        calibration_id="ideal-test",
        source_digest="a" * 64,
        optical_initial_bloch_vector=(0.0, 0.0, 1.0),
        preparation_pulse=spam.PulseSpecification(np.pi / 2.0, np.pi / 2.0),
        readout_pulse=spam.PulseSpecification(-np.pi / 2.0, np.pi / 2.0),
        optical_readout=spam.OpticalReadoutCalibration(
            bright_mean_counts=120.0,
            dark_mean_counts=80.0,
            bright_sem_counts=1.0,
            dark_sem_counts=1.2,
            bright_sample_count=100,
            dark_sample_count=100,
        ),
    )


def test_ideal_zero_depth_preparation_and_readout_return_bright_reference():
    result = spam.predict_spam_signal(ideal_calibration())
    assert coherent.bloch_vector_from_density_matrix(
        result.prepared_density_matrix
    ) == pytest.approx([1.0, 0.0, 0.0], abs=2e-14)
    assert result.expected_fluorescence_counts == pytest.approx(120.0)
    assert result.normalized_spin_signal == pytest.approx(1.0)


def test_control_that_reverses_sx_returns_dark_reference():
    z_pi = coherent.rotation_unitary(np.pi, 0.0, axis_z=1.0)
    result = spam.predict_spam_signal(ideal_calibration(), z_pi)
    assert result.expected_fluorescence_counts == pytest.approx(80.0)
    assert result.normalized_spin_signal == pytest.approx(-1.0)


def test_mixed_state_maps_to_readout_midpoint_and_zero_signal():
    readout = spam.OpticalReadoutCalibration(120.0, 80.0)
    mixed = np.eye(2, dtype=complex) / 2.0
    counts = readout.expected_counts(mixed)
    assert counts == pytest.approx(100.0)
    assert readout.normalized_spin_signal(counts) == pytest.approx(0.0)


def test_imperfect_preparation_and_analysis_reduce_zero_depth_visibility():
    ideal = spam.predict_spam_signal(ideal_calibration())
    imperfect = spam.SPAMCalibration(
        calibration_id="imperfect-test",
        optical_initial_bloch_vector=(0.0, 0.0, 0.92),
        preparation_pulse=spam.PulseSpecification(
            np.pi / 2.0,
            np.pi / 2.0,
            rotation_angle_error=0.08,
            axis_z=0.04,
        ),
        readout_pulse=spam.PulseSpecification(
            -np.pi / 2.0,
            np.pi / 2.0,
            rotation_angle_error=-0.05,
            phase_error_rad=0.03,
        ),
        optical_readout=ideal_calibration().optical_readout,
    )
    result = spam.predict_spam_signal(imperfect)
    assert abs(result.normalized_spin_signal) < abs(ideal.normalized_spin_signal)
    assert -1.0 <= result.normalized_spin_signal <= 1.0


def test_reference_estimator_uses_independent_shots_and_reports_sem():
    bright = np.array([118.0, 121.0, 119.0, 122.0])
    dark = np.array([79.0, 82.0, 80.0, 81.0])
    calibration = spam.estimate_optical_readout(bright, dark)
    assert calibration.bright_mean_counts == pytest.approx(np.mean(bright))
    assert calibration.dark_mean_counts == pytest.approx(np.mean(dark))
    assert calibration.bright_sem_counts == pytest.approx(
        np.std(bright, ddof=1) / np.sqrt(bright.size)
    )
    assert calibration.dark_sem_counts == pytest.approx(
        np.std(dark, ddof=1) / np.sqrt(dark.size)
    )
    assert calibration.bright_sample_count == 4
    assert calibration.dark_sample_count == 4


def test_calibration_json_round_trip_is_deterministic():
    calibration = ideal_calibration()
    payload = calibration.to_dict()
    encoded = json.dumps(payload, sort_keys=True)
    restored = spam.SPAMCalibration.from_dict(json.loads(encoded))
    assert restored.to_dict() == payload
    assert spam.predict_spam_signal(restored).normalized_spin_signal == pytest.approx(1.0)


def test_invalid_spam_inputs_fail_clearly():
    with pytest.raises(ValueError, match="bright_mean_counts"):
        spam.OpticalReadoutCalibration(80.0, 100.0)
    with pytest.raises(ValueError, match="axis_z"):
        spam.PulseSpecification(np.pi / 2.0, 0.0, axis_z=1.1)
    with pytest.raises(ValueError, match="source_digest"):
        spam.SPAMCalibration(
            calibration_id="bad-digest",
            source_digest="not-a-digest",
            optical_initial_bloch_vector=(0.0, 0.0, 1.0),
            preparation_pulse=spam.PulseSpecification(np.pi / 2.0, 0.0),
            readout_pulse=spam.PulseSpecification(-np.pi / 2.0, 0.0),
            optical_readout=spam.OpticalReadoutCalibration(100.0, 80.0),
        )
    with pytest.raises(ValueError, match="status='calibrated'"):
        spam.SPAMCalibration.from_dict(
            json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
        )


def test_nonunitary_control_is_rejected():
    with pytest.raises(ValueError, match="must be unitary"):
        spam.predict_spam_signal(ideal_calibration(), np.diag([1.0, 0.8]))


def test_prospective_protocol_is_empty_and_excludes_holdout_depths():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    spam.validate_prospective_spam_protocol(protocol)
    assert protocol["holdout_exclusion"]["xy8_blocks"] == [1, 9, 73]
    assert protocol["holdout_exclusion"]["total_pi_pulses"] == [8, 72, 584]
    assert all(row["measured_value"] is None for row in protocol["measurement_set"])
    assert all(
        row["xy8_blocks"] not in {1, 9, 73}
        for row in protocol["measurement_set"]
    )


@pytest.mark.parametrize("leak_kind", ["measured_value", "holdout_depth", "status"])
def test_prospective_protocol_validator_detects_leakage(leak_kind):
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    leaked = deepcopy(protocol)
    if leak_kind == "measured_value":
        leaked["measurement_set"][0]["measured_value"] = 42.0
        message = "must not contain measured values"
    elif leak_kind == "holdout_depth":
        leaked["measurement_set"][0]["xy8_blocks"] = 9
        message = "holdout XY8 depth"
    else:
        leaked["status"] = "calibrated"
        message = "frozen before measurements"
    with pytest.raises(ValueError, match=message):
        spam.validate_prospective_spam_protocol(leaked)
