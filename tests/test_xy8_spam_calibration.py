from __future__ import annotations

from copy import deepcopy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "code" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import xy8_spam_calibration as calibration


PROTOCOL_PATH = (
    ROOT / "data" / "raw" / "nv" / "calibration" / "xy8_spam_calibration_protocol.json"
)


def small_protocol(minimum_repeats: int = 10) -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol = deepcopy(protocol)
    protocol["acquisition_rules"]["minimum_repeats_per_setting"] = minimum_repeats
    return protocol


def synthetic_rows(repeats: int = 20) -> list[dict]:
    means = {
        "bright_ms0_reference": 120.0,
        "dark_ms1_reference": 80.0,
        "zero_depth_plus_x": 118.0,
        "zero_depth_minus_x": 82.0,
        "zero_depth_plus_y": 116.0,
        "zero_depth_minus_y": 84.0,
        "zero_depth_mixed_reference": 100.0,
    }
    rows = []
    timestamp = 0.0
    for measurement_id, mean in means.items():
        for repeat_index in range(repeats):
            rows.append(
                {
                    "measurement_id": measurement_id,
                    "repeat_index": repeat_index,
                    "fluorescence_counts": mean + (-1.0 if repeat_index % 2 == 0 else 1.0),
                    "timestamp_s": timestamp,
                    "xy8_blocks": None if "reference" in measurement_id else 0,
                }
            )
            timestamp += 1.0
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "measurement_id",
        "repeat_index",
        "fluorescence_counts",
        "timestamp_s",
        "xy8_blocks",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            if encoded["xy8_blocks"] is None:
                encoded["xy8_blocks"] = ""
            writer.writerow(encoded)


def physical_characterization() -> dict:
    names = [
        "optical_initial_bloch_x",
        "optical_initial_bloch_y",
        "optical_initial_bloch_z",
        "preparation_rotation_angle_error",
        "preparation_axis_z",
        "preparation_phase_error_rad",
        "readout_rotation_angle_error",
        "readout_axis_z",
        "readout_phase_error_rad",
    ]
    return {
        "status": "independently_calibrated",
        "calibration_id": "synthetic-stage1",
        "optical_initial_bloch_vector": [0.0, 0.0, 0.95],
        "preparation_pulse": {
            "nominal_angle_rad": np.pi / 2.0,
            "phase_rad": np.pi / 2.0,
            "rotation_angle_error": 0.01,
            "axis_z": 0.02,
            "phase_error_rad": 0.005,
        },
        "readout_pulse": {
            "nominal_angle_rad": -np.pi / 2.0,
            "phase_rad": np.pi / 2.0,
            "rotation_angle_error": -0.01,
            "axis_z": 0.01,
            "phase_error_rad": -0.004,
        },
        "parameter_names": names,
        "parameter_covariance": (np.eye(len(names)) * 1e-6).tolist(),
        "calibration_sources": ["synthetic_zero_depth_fit.csv"],
    }


def test_stage1_estimates_effective_parameters_and_covariance():
    result = calibration.estimate_stage1(
        synthetic_rows(),
        small_protocol(),
        source_digest_value="a" * 64,
    )
    estimates = dict(zip(result.parameter_names, result.estimates))
    assert result.passed
    assert estimates["bright_mean_counts"] == pytest.approx(120.0)
    assert estimates["dark_mean_counts"] == pytest.approx(80.0)
    assert estimates["x_visibility"] == pytest.approx(0.9)
    assert estimates["x_offset"] == pytest.approx(0.0)
    assert estimates["y_visibility"] == pytest.approx(0.8)
    assert estimates["y_offset"] == pytest.approx(0.0)
    assert estimates["mixed_state_offset"] == pytest.approx(0.0)
    assert result.covariance.shape == (len(calibration.PARAMETER_NAMES),) * 2
    assert result.covariance == pytest.approx(result.covariance.T)
    assert np.all(np.linalg.eigvalsh(result.covariance) >= -1e-10)
    json.dumps(result.to_dict())


def test_minimum_repeat_gate_fails_without_marking_result_calibrated():
    result = calibration.estimate_stage1(
        synthetic_rows(repeats=4),
        small_protocol(minimum_repeats=10),
        source_digest_value="b" * 64,
    )
    assert not result.passed
    assert not result.gates["minimum_repeats"]["passed"]
    assert result.to_dict()["status"] == "stage1_failed"


def test_file_workflow_loads_data_and_produces_deterministic_digest(tmp_path):
    shots_path = tmp_path / "shots.csv"
    protocol_path = tmp_path / "protocol.json"
    write_rows(shots_path, synthetic_rows())
    protocol_path.write_text(json.dumps(small_protocol()), encoding="utf-8")
    first = calibration.analyze_stage1_files(protocol_path, shots_path)
    second = calibration.analyze_stage1_files(protocol_path, shots_path)
    assert first.passed
    assert first.source_digest == second.source_digest
    assert len(first.source_digest) == 64


def test_shot_loader_rejects_holdout_depth(tmp_path):
    shots_path = tmp_path / "shots.csv"
    rows = synthetic_rows()
    rows[0]["xy8_blocks"] = 9
    write_rows(shots_path, rows)
    with pytest.raises(ValueError, match="holdout depth"):
        calibration.load_shot_csv(shots_path, small_protocol())


def test_shot_loader_rejects_undeclared_measurement(tmp_path):
    shots_path = tmp_path / "shots.csv"
    rows = synthetic_rows()
    rows[0]["measurement_id"] = "secret_depth_curve"
    write_rows(shots_path, rows)
    with pytest.raises(ValueError, match="Undeclared measurement_id"):
        calibration.load_shot_csv(shots_path, small_protocol())


def test_missing_required_measurement_is_rejected():
    rows = [
        row
        for row in synthetic_rows()
        if row["measurement_id"] != "zero_depth_minus_y"
    ]
    with pytest.raises(ValueError, match="no observations"):
        calibration.estimate_stage1(
            rows,
            small_protocol(),
            source_digest_value="c" * 64,
        )


def test_passing_result_populates_loadable_calibration_payload():
    result = calibration.estimate_stage1(
        synthetic_rows(),
        small_protocol(),
        source_digest_value="d" * 64,
    )
    payload = calibration.build_calibrated_payload(
        result, physical_characterization()
    )
    assert payload["status"] == "calibrated"
    assert payload["source_digest"] == "d" * 64
    assert payload["optical_readout"]["bright_mean_counts"] == pytest.approx(120.0)
    assert payload["optical_readout"]["dark_mean_counts"] == pytest.approx(80.0)
    assert payload["holdout_used_for_estimation"] is False
    assert payload["effective_parameter_estimates"]["x_visibility"] == pytest.approx(0.9)


def test_failed_result_cannot_populate_calibrated_payload():
    result = calibration.estimate_stage1(
        synthetic_rows(repeats=4),
        small_protocol(minimum_repeats=10),
        source_digest_value="e" * 64,
    )
    with pytest.raises(ValueError, match="failed Stage 1"):
        calibration.build_calibrated_payload(result, physical_characterization())


def test_physical_covariance_must_be_valid():
    result = calibration.estimate_stage1(
        synthetic_rows(),
        small_protocol(),
        source_digest_value="f" * 64,
    )
    physical = physical_characterization()
    physical["parameter_covariance"] = [[1.0]]
    with pytest.raises(ValueError, match="wrong shape"):
        calibration.build_calibrated_payload(result, physical)
