"""Preparation and readout (SPAM) model for calibrated NV XY8 predictions.

This module keeps SPAM calibration separate from the three-point literature
holdout.  It converts independently calibrated optical reference counts and
microwave preparation/readout pulses into a prepared density matrix and an
expected fluorescence signal.  It does not estimate any parameter from an
XY8 depth curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from . import xy8_coherent_control as coherent
except ImportError:
    import xy8_coherent_control as coherent


@dataclass(frozen=True)
class PulseSpecification:
    """Calibrated two-level microwave rotation used for preparation/readout."""

    nominal_angle_rad: float
    phase_rad: float
    rotation_angle_error: float = 0.0
    axis_z: float = 0.0
    phase_error_rad: float = 0.0

    def __post_init__(self) -> None:
        values = [
            self.nominal_angle_rad,
            self.phase_rad,
            self.rotation_angle_error,
            self.axis_z,
            self.phase_error_rad,
        ]
        if not all(np.isfinite(float(value)) for value in values):
            raise ValueError("Pulse specification values must be finite.")
        if abs(float(self.axis_z)) > 1.0:
            raise ValueError("Pulse axis_z must lie in [-1, 1].")

    def unitary(self) -> np.ndarray:
        return coherent.rotation_unitary(
            float(self.nominal_angle_rad) * (1.0 + float(self.rotation_angle_error)),
            float(self.phase_rad) + float(self.phase_error_rad),
            axis_z=float(self.axis_z),
        )

    def to_dict(self) -> dict:
        return {
            "nominal_angle_rad": float(self.nominal_angle_rad),
            "phase_rad": float(self.phase_rad),
            "rotation_angle_error": float(self.rotation_angle_error),
            "axis_z": float(self.axis_z),
            "phase_error_rad": float(self.phase_error_rad),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "PulseSpecification":
        if not isinstance(payload, dict):
            raise TypeError("Pulse specification must be a dictionary.")
        return cls(
            nominal_angle_rad=payload["nominal_angle_rad"],
            phase_rad=payload["phase_rad"],
            rotation_angle_error=payload.get("rotation_angle_error", 0.0),
            axis_z=payload.get("axis_z", 0.0),
            phase_error_rad=payload.get("phase_error_rad", 0.0),
        )


@dataclass(frozen=True)
class OpticalReadoutCalibration:
    """Independent bright/dark fluorescence reference calibration."""

    bright_mean_counts: float
    dark_mean_counts: float
    bright_sem_counts: float = 0.0
    dark_sem_counts: float = 0.0
    bright_sample_count: int | None = None
    dark_sample_count: int | None = None

    def __post_init__(self) -> None:
        values = [
            self.bright_mean_counts,
            self.dark_mean_counts,
            self.bright_sem_counts,
            self.dark_sem_counts,
        ]
        if not all(np.isfinite(float(value)) for value in values):
            raise ValueError("Optical readout calibration values must be finite.")
        if float(self.bright_mean_counts) <= float(self.dark_mean_counts):
            raise ValueError("bright_mean_counts must exceed dark_mean_counts.")
        if float(self.dark_mean_counts) < 0.0:
            raise ValueError("Reference fluorescence counts cannot be negative.")
        if float(self.bright_sem_counts) < 0.0 or float(self.dark_sem_counts) < 0.0:
            raise ValueError("Reference standard errors cannot be negative.")
        for name, count in [
            ("bright_sample_count", self.bright_sample_count),
            ("dark_sample_count", self.dark_sample_count),
        ]:
            if count is not None and (
                isinstance(count, bool) or int(count) != count or int(count) < 2
            ):
                raise ValueError(f"{name} must be at least two when supplied.")

    @property
    def midpoint_counts(self) -> float:
        return 0.5 * (float(self.bright_mean_counts) + float(self.dark_mean_counts))

    @property
    def half_span_counts(self) -> float:
        return 0.5 * (float(self.bright_mean_counts) - float(self.dark_mean_counts))

    def expected_counts(self, density_matrix) -> float:
        """Expected fluorescence for direct optical population readout."""
        z_expectation = coherent.expectation(density_matrix, coherent.SIGMA_Z)
        return float(self.midpoint_counts + self.half_span_counts * z_expectation)

    def normalized_spin_signal(self, counts: float) -> float:
        """Map bright/dark references to +1/-1 without clipping."""
        counts = float(counts)
        if not np.isfinite(counts):
            raise ValueError("Fluorescence counts must be finite.")
        return float((counts - self.midpoint_counts) / self.half_span_counts)

    def to_dict(self) -> dict:
        return {
            "bright_mean_counts": float(self.bright_mean_counts),
            "dark_mean_counts": float(self.dark_mean_counts),
            "bright_sem_counts": float(self.bright_sem_counts),
            "dark_sem_counts": float(self.dark_sem_counts),
            "bright_sample_count": self.bright_sample_count,
            "dark_sample_count": self.dark_sample_count,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "OpticalReadoutCalibration":
        if not isinstance(payload, dict):
            raise TypeError("Optical readout calibration must be a dictionary.")
        return cls(
            bright_mean_counts=payload["bright_mean_counts"],
            dark_mean_counts=payload["dark_mean_counts"],
            bright_sem_counts=payload.get("bright_sem_counts", 0.0),
            dark_sem_counts=payload.get("dark_sem_counts", 0.0),
            bright_sample_count=payload.get("bright_sample_count"),
            dark_sample_count=payload.get("dark_sample_count"),
        )


@dataclass(frozen=True)
class SPAMCalibration:
    """Calibrated state preparation, analysis pulse, and optical readout."""

    calibration_id: str
    optical_initial_bloch_vector: tuple[float, float, float]
    preparation_pulse: PulseSpecification
    readout_pulse: PulseSpecification
    optical_readout: OpticalReadoutCalibration
    source_digest: str | None = None

    def __post_init__(self) -> None:
        if not str(self.calibration_id).strip():
            raise ValueError("calibration_id must be non-empty.")
        vector = np.asarray(self.optical_initial_bloch_vector, dtype=float)
        coherent.density_matrix_from_bloch(vector)
        if self.source_digest is not None:
            digest = str(self.source_digest).lower()
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("source_digest must be a 64-character SHA-256 digest.")

    def initial_optical_state(self) -> np.ndarray:
        return coherent.density_matrix_from_bloch(self.optical_initial_bloch_vector)

    def prepared_state(self) -> np.ndarray:
        return apply_unitary(self.initial_optical_state(), self.preparation_pulse.unitary())

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "status": "calibrated",
            "calibration_id": self.calibration_id,
            "source_digest": self.source_digest,
            "optical_initial_bloch_vector": [
                float(value) for value in self.optical_initial_bloch_vector
            ],
            "preparation_pulse": self.preparation_pulse.to_dict(),
            "readout_pulse": self.readout_pulse.to_dict(),
            "optical_readout": self.optical_readout.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SPAMCalibration":
        if not isinstance(payload, dict):
            raise TypeError("SPAM calibration must be a dictionary.")
        if payload.get("status") != "calibrated":
            raise ValueError("Only a completed calibration with status='calibrated' can be loaded.")
        return cls(
            calibration_id=payload["calibration_id"],
            source_digest=payload.get("source_digest"),
            optical_initial_bloch_vector=tuple(payload["optical_initial_bloch_vector"]),
            preparation_pulse=PulseSpecification.from_dict(payload["preparation_pulse"]),
            readout_pulse=PulseSpecification.from_dict(payload["readout_pulse"]),
            optical_readout=OpticalReadoutCalibration.from_dict(payload["optical_readout"]),
        )


@dataclass(frozen=True)
class SPAMPrediction:
    """Prepared, controlled, and analyzed state with expected readout."""

    prepared_density_matrix: np.ndarray
    controlled_density_matrix: np.ndarray
    analyzed_density_matrix: np.ndarray
    expected_fluorescence_counts: float
    normalized_spin_signal: float


def apply_unitary(density_matrix, unitary) -> np.ndarray:
    """Apply a validated two-level unitary and return a Hermitian state."""
    rho = np.asarray(density_matrix, dtype=complex)
    # The public Bloch conversion performs the physical-state validation.
    coherent.bloch_vector_from_density_matrix(rho)
    unitary = np.asarray(unitary, dtype=complex)
    if unitary.shape != (2, 2) or not np.all(np.isfinite(unitary)):
        raise ValueError("Unitary must be a finite 2x2 matrix.")
    if not np.allclose(unitary.conj().T @ unitary, np.eye(2), atol=1e-12, rtol=0.0):
        raise ValueError("Control operation must be unitary.")
    result = unitary @ rho @ unitary.conj().T
    result = 0.5 * (result + result.conj().T)
    result /= np.trace(result)
    return result


def predict_spam_signal(
    calibration: SPAMCalibration,
    control_unitary=None,
) -> SPAMPrediction:
    """Predict fluorescence after preparation, optional control, and readout."""
    if not isinstance(calibration, SPAMCalibration):
        raise TypeError("calibration must be a SPAMCalibration.")
    prepared = calibration.prepared_state()
    if control_unitary is None:
        control_unitary = np.eye(2, dtype=complex)
    controlled = apply_unitary(prepared, control_unitary)
    analyzed = apply_unitary(controlled, calibration.readout_pulse.unitary())
    counts = calibration.optical_readout.expected_counts(analyzed)
    normalized = calibration.optical_readout.normalized_spin_signal(counts)
    for array in [prepared, controlled, analyzed]:
        array.setflags(write=False)
    return SPAMPrediction(
        prepared_density_matrix=prepared,
        controlled_density_matrix=controlled,
        analyzed_density_matrix=analyzed,
        expected_fluorescence_counts=counts,
        normalized_spin_signal=normalized,
    )


def estimate_optical_readout(bright_counts, dark_counts) -> OpticalReadoutCalibration:
    """Estimate reference means and standard errors from independent shots."""
    bright = np.asarray(bright_counts, dtype=float)
    dark = np.asarray(dark_counts, dtype=float)
    for name, values in [("bright_counts", bright), ("dark_counts", dark)]:
        if values.ndim != 1 or values.size < 2:
            raise ValueError(f"{name} must contain at least two measurements.")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(f"{name} must contain finite non-negative values.")
    return OpticalReadoutCalibration(
        bright_mean_counts=float(np.mean(bright)),
        dark_mean_counts=float(np.mean(dark)),
        bright_sem_counts=float(np.std(bright, ddof=1) / np.sqrt(bright.size)),
        dark_sem_counts=float(np.std(dark, ddof=1) / np.sqrt(dark.size)),
        bright_sample_count=int(bright.size),
        dark_sample_count=int(dark.size),
    )


def validate_prospective_spam_protocol(protocol: dict) -> None:
    """Reject protocols that leak holdout values into SPAM calibration."""
    if not isinstance(protocol, dict):
        raise TypeError("SPAM protocol must be a dictionary.")
    if protocol.get("status") != "prospective_protocol_no_measurements":
        raise ValueError("SPAM protocol must be frozen before measurements are entered.")
    holdout = protocol.get("holdout_exclusion", {})
    if holdout.get("xy8_blocks") != [1, 9, 73]:
        raise ValueError("The protocol must explicitly exclude holdout N=[1, 9, 73].")
    if holdout.get("permitted_for_parameter_estimation") is not False:
        raise ValueError("Holdout points must be forbidden from parameter estimation.")
    measurements = protocol.get("measurement_set", [])
    if not measurements:
        raise ValueError("The prospective SPAM measurement set cannot be empty.")
    identifiers = [row.get("measurement_id") for row in measurements]
    if len(set(identifiers)) != len(identifiers) or any(not value for value in identifiers):
        raise ValueError("Every SPAM measurement must have a unique identifier.")
    forbidden_depths = set(holdout["xy8_blocks"])
    for row in measurements:
        if row.get("xy8_blocks") in forbidden_depths:
            raise ValueError("A holdout XY8 depth appears in the calibration measurement set.")
        if row.get("measured_value") is not None:
            raise ValueError("Prospective protocol rows must not contain measured values.")
