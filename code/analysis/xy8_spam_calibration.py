"""Prospective Stage 1 analysis for independent NV SPAM measurements.

The input is a shot-level CSV acquired from the frozen SPAM protocol.  This
module estimates effective readout and zero-depth parameters, propagates their
sampling covariance, evaluates the predeclared gates, and hashes the source
files.  It never reads the XY8 depth holdout.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from . import xy8_spam_model as spam
except ImportError:
    import xy8_spam_model as spam


REQUIRED_MEASUREMENTS = (
    "bright_ms0_reference",
    "dark_ms1_reference",
    "zero_depth_plus_x",
    "zero_depth_minus_x",
    "zero_depth_plus_y",
    "zero_depth_minus_y",
    "zero_depth_mixed_reference",
)

PARAMETER_NAMES = (
    "bright_mean_counts",
    "dark_mean_counts",
    "readout_midpoint_counts",
    "readout_half_span_counts",
    "x_visibility",
    "x_offset",
    "y_visibility",
    "y_offset",
    "mixed_state_offset",
)


@dataclass(frozen=True)
class SPAMStage1Result:
    """Effective Stage 1 estimates and prospective acceptance decisions."""

    parameter_names: tuple[str, ...]
    estimates: np.ndarray
    covariance: np.ndarray
    measurement_summary: dict
    gates: dict[str, dict]
    source_digest: str

    @property
    def passed(self) -> bool:
        return all(bool(gate["passed"]) for gate in self.gates.values())

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "status": "stage1_passed" if self.passed else "stage1_failed",
            "source_digest": self.source_digest,
            "parameter_names": list(self.parameter_names),
            "estimates": {
                name: float(value)
                for name, value in zip(self.parameter_names, self.estimates)
            },
            "covariance": self.covariance.tolist(),
            "measurement_summary": self.measurement_summary,
            "acceptance_gates": self.gates,
            "holdout_used_for_estimation": False,
        }


def load_protocol(path: str | Path) -> dict:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    spam.validate_prospective_spam_protocol(protocol)
    identifiers = {row["measurement_id"] for row in protocol["measurement_set"]}
    missing = sorted(set(REQUIRED_MEASUREMENTS) - identifiers)
    if missing:
        raise ValueError(f"SPAM protocol is missing required measurements: {missing}")
    return protocol


def load_shot_csv(path: str | Path, protocol: dict) -> list[dict]:
    """Load shot-level fluorescence while rejecting undeclared/holdout rows."""
    declared = {row["measurement_id"] for row in protocol["measurement_set"]}
    required_columns = {
        "measurement_id",
        "repeat_index",
        "fluorescence_counts",
        "timestamp_s",
        "xy8_blocks",
    }
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = sorted(required_columns - set(reader.fieldnames or []))
        if missing_columns:
            raise ValueError(f"Shot CSV is missing required columns: {missing_columns}")
        for line_number, raw in enumerate(reader, start=2):
            measurement_id = str(raw["measurement_id"]).strip()
            if measurement_id not in declared:
                raise ValueError(
                    f"Undeclared measurement_id {measurement_id!r} on line {line_number}."
                )
            try:
                repeat_index = int(raw["repeat_index"])
                counts = float(raw["fluorescence_counts"])
                timestamp_s = float(raw["timestamp_s"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid numeric value on shot CSV line {line_number}.") from error
            if repeat_index < 0:
                raise ValueError("repeat_index must be non-negative.")
            if not np.isfinite(counts) or counts < 0.0:
                raise ValueError("fluorescence_counts must be finite and non-negative.")
            if not np.isfinite(timestamp_s):
                raise ValueError("timestamp_s must be finite.")
            depth_text = str(raw["xy8_blocks"]).strip()
            depth = None if depth_text == "" else int(depth_text)
            if depth in {1, 9, 73}:
                raise ValueError("A frozen XY8 holdout depth appears in the SPAM shot data.")
            if depth not in {None, 0}:
                raise ValueError("Stage 1 SPAM data may contain only zero-depth controls.")
            rows.append(
                {
                    "measurement_id": measurement_id,
                    "repeat_index": repeat_index,
                    "fluorescence_counts": counts,
                    "timestamp_s": timestamp_s,
                    "xy8_blocks": depth,
                }
            )
    if not rows:
        raise ValueError("Shot CSV contains no measurements.")
    return rows


def source_digest(*paths: str | Path) -> str:
    """Hash source contents and stable labels in caller-supplied order."""
    digest = hashlib.sha256()
    for item in paths:
        path = Path(item)
        payload = path.read_bytes()
        label = path.name.encode("utf-8")
        digest.update(len(label).to_bytes(8, "big"))
        digest.update(label)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    groups = {name: [] for name in REQUIRED_MEASUREMENTS}
    for row in rows:
        if row["measurement_id"] in groups:
            groups[row["measurement_id"]].append(row)
    missing = [name for name, values in groups.items() if not values]
    if missing:
        raise ValueError(f"Shot CSV has no observations for: {missing}")
    return groups


def _effective_parameters(group_means: np.ndarray) -> np.ndarray:
    bright, dark, plus_x, minus_x, plus_y, minus_y, mixed = group_means
    span = bright - dark
    if span <= 0.0:
        raise ValueError("The measured bright mean must exceed the dark mean.")
    midpoint = 0.5 * (bright + dark)
    half_span = 0.5 * span
    return np.array(
        [
            bright,
            dark,
            midpoint,
            half_span,
            (plus_x - minus_x) / span,
            (plus_x + minus_x - bright - dark) / span,
            (plus_y - minus_y) / span,
            (plus_y + minus_y - bright - dark) / span,
            (mixed - midpoint) / half_span,
        ],
        dtype=float,
    )


def _numerical_jacobian(function, point: np.ndarray) -> np.ndarray:
    baseline = function(point)
    jacobian = np.empty((baseline.size, point.size), dtype=float)
    for column, value in enumerate(point):
        step = max(1e-7, abs(float(value)) * 1e-7)
        plus = point.copy()
        minus = point.copy()
        plus[column] += step
        minus[column] -= step
        jacobian[:, column] = (function(plus) - function(minus)) / (2.0 * step)
    return jacobian


def _reference_drift(group: list[dict], separation: float) -> float:
    ordered = sorted(group, key=lambda row: row["timestamp_s"])
    quartile = max(1, len(ordered) // 4)
    first = np.mean([row["fluorescence_counts"] for row in ordered[:quartile]])
    last = np.mean([row["fluorescence_counts"] for row in ordered[-quartile:]])
    return float(abs(last - first) / separation)


def estimate_stage1(
    rows: list[dict],
    protocol: dict,
    *,
    source_digest_value: str,
) -> SPAMStage1Result:
    """Estimate effective SPAM quantities and evaluate the frozen gates."""
    groups = _group_rows(rows)
    minimum_repeats = int(protocol["acquisition_rules"]["minimum_repeats_per_setting"])
    means = []
    variances_of_means = []
    summary = {}
    for name in REQUIRED_MEASUREMENTS:
        values = np.array([row["fluorescence_counts"] for row in groups[name]])
        sem = float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else float("inf")
        mean = float(np.mean(values))
        means.append(mean)
        variances_of_means.append(sem * sem)
        summary[name] = {
            "sample_count": int(values.size),
            "mean_counts": mean,
            "sem_counts": sem,
            "first_timestamp_s": float(min(row["timestamp_s"] for row in groups[name])),
            "last_timestamp_s": float(max(row["timestamp_s"] for row in groups[name])),
        }

    group_means = np.asarray(means, dtype=float)
    group_covariance = np.diag(variances_of_means)
    estimates = _effective_parameters(group_means)
    jacobian = _numerical_jacobian(_effective_parameters, group_means)
    covariance = jacobian @ group_covariance @ jacobian.T
    covariance = 0.5 * (covariance + covariance.T)

    bright_sem = summary["bright_ms0_reference"]["sem_counts"]
    dark_sem = summary["dark_ms1_reference"]["sem_counts"]
    separation = estimates[0] - estimates[1]
    separation_sem = float(np.hypot(bright_sem, dark_sem))
    visibility_z = float(separation / separation_sem) if separation_sem > 0.0 else float("inf")
    bright_drift = _reference_drift(groups["bright_ms0_reference"], separation)
    dark_drift = _reference_drift(groups["dark_ms1_reference"], separation)
    mixed_sem = float(np.sqrt(max(covariance[8, 8], 0.0)))
    mixed_z = abs(float(estimates[8])) / mixed_sem if mixed_sem > 0.0 else (
        0.0 if estimates[8] == 0.0 else float("inf")
    )
    finite_covariance = bool(
        np.all(np.isfinite(covariance))
        and np.all(np.linalg.eigvalsh(covariance) >= -1e-10)
    )

    gates = {
        "minimum_repeats": {
            "passed": all(len(values) >= minimum_repeats for values in groups.values()),
            "threshold_repeats_per_setting": minimum_repeats,
            "minimum_observed_repeats": min(len(values) for values in groups.values()),
        },
        "minimum_visibility": {
            "passed": visibility_z >= 1.96,
            "separation_standard_errors": visibility_z,
            "threshold": 1.96,
        },
        "reference_stability": {
            "passed": max(bright_drift, dark_drift) <= 0.20,
            "bright_drift_fraction_of_separation": bright_drift,
            "dark_drift_fraction_of_separation": dark_drift,
            "threshold": 0.20,
        },
        "mixed_state_midpoint": {
            "passed": mixed_z <= 1.96,
            "absolute_standardized_offset": mixed_z,
            "threshold": 1.96,
        },
        "finite_covariance": {
            "passed": finite_covariance,
            "method": "independent group-mean covariance with numerical delta propagation",
        },
    }
    estimates.setflags(write=False)
    covariance.setflags(write=False)
    return SPAMStage1Result(
        parameter_names=PARAMETER_NAMES,
        estimates=estimates,
        covariance=covariance,
        measurement_summary=summary,
        gates=gates,
        source_digest=source_digest_value,
    )


def analyze_stage1_files(
    protocol_path: str | Path,
    shot_csv_path: str | Path,
    *additional_source_paths: str | Path,
) -> SPAMStage1Result:
    protocol = load_protocol(protocol_path)
    rows = load_shot_csv(shot_csv_path, protocol)
    digest = source_digest(protocol_path, shot_csv_path, *additional_source_paths)
    return estimate_stage1(rows, protocol, source_digest_value=digest)


def build_calibrated_payload(
    result: SPAMStage1Result,
    physical_characterization: dict,
) -> dict:
    """Combine passing shot statistics with independent physical pulse estimates."""
    if not result.passed:
        raise ValueError("A failed Stage 1 result cannot populate a calibrated manifest.")
    if physical_characterization.get("status") != "independently_calibrated":
        raise ValueError(
            "Physical characterization must have status='independently_calibrated'."
        )
    names = physical_characterization.get("parameter_names")
    covariance = np.asarray(
        physical_characterization.get("parameter_covariance"), dtype=float
    )
    if not isinstance(names, list) or not names or len(set(names)) != len(names):
        raise ValueError("Physical parameter_names must be a non-empty unique list.")
    if covariance.shape != (len(names), len(names)):
        raise ValueError("Physical parameter_covariance has the wrong shape.")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("Physical parameter_covariance must be finite.")
    if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0.0):
        raise ValueError("Physical parameter_covariance must be symmetric.")
    if np.any(np.linalg.eigvalsh(covariance) < -1e-12):
        raise ValueError("Physical parameter_covariance must be positive semidefinite.")

    estimates = dict(zip(result.parameter_names, result.estimates))
    bright_sem = float(np.sqrt(max(result.covariance[0, 0], 0.0)))
    dark_sem = float(np.sqrt(max(result.covariance[1, 1], 0.0)))
    completed = spam.SPAMCalibration(
        calibration_id=physical_characterization["calibration_id"],
        source_digest=result.source_digest,
        optical_initial_bloch_vector=tuple(
            physical_characterization["optical_initial_bloch_vector"]
        ),
        preparation_pulse=spam.PulseSpecification.from_dict(
            physical_characterization["preparation_pulse"]
        ),
        readout_pulse=spam.PulseSpecification.from_dict(
            physical_characterization["readout_pulse"]
        ),
        optical_readout=spam.OpticalReadoutCalibration(
            bright_mean_counts=estimates["bright_mean_counts"],
            dark_mean_counts=estimates["dark_mean_counts"],
            bright_sem_counts=bright_sem,
            dark_sem_counts=dark_sem,
            bright_sample_count=result.measurement_summary["bright_ms0_reference"][
                "sample_count"
            ],
            dark_sample_count=result.measurement_summary["dark_ms1_reference"][
                "sample_count"
            ],
        ),
    ).to_dict()
    completed.update(
        {
            "calibration_sources": physical_characterization.get(
                "calibration_sources", []
            ),
            "excluded_validation_sources": [
                "farfurnik_2015_fig3b_xy8.csv: N=1,9,73"
            ],
            "physical_parameter_names": names,
            "physical_parameter_covariance": covariance.tolist(),
            "effective_parameter_names": list(result.parameter_names),
            "effective_parameter_estimates": {
                name: float(value)
                for name, value in zip(result.parameter_names, result.estimates)
            },
            "effective_parameter_covariance": result.covariance.tolist(),
            "stage1_acceptance_gates": result.gates,
            "holdout_used_for_estimation": False,
        }
    )
    return completed
