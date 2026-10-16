"""Environment-to-transition adapter for the frozen NV XY8 scan.

Stage 2 combines the frozen temperature-dependent zero-field splitting, the
six-coefficient spin-strain Hamiltonian, and the preregistered magnetic bias.
It diagonalizes the complete spin-1 Hamiltonian and tracks reference-labelled
eigenstates by overlap.  This module does not evaluate an EDT score.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import math
from pathlib import Path
from typing import Any, Iterable
import warnings as python_warnings

import numpy as np

import xy8_environment_scan_config as scan_config
import xy8_nv_environment_literature as literature


STATE_LABELS = ("m_s_plus_1_like", "m_s_0_like", "m_s_minus_1_like")
MS_PLUS_INDEX = 0
MS_ZERO_INDEX = 1
MS_MINUS_INDEX = 2
ELECTRON_GYROMAGNETIC_RATIO_MHZ_PER_T = 28_024.95164


@dataclass(frozen=True)
class EnvironmentCondition:
    """Temperature and symmetric dimensionless strain tensor in the NV frame."""

    temperature_K: float
    strain_tensor: np.ndarray
    label: str = ""

    def __post_init__(self) -> None:
        temperature = float(self.temperature_K)
        strain = np.asarray(self.strain_tensor, dtype=float)
        if not math.isfinite(temperature) or temperature < 0.0:
            raise ValueError("temperature_K must be finite and non-negative.")
        if strain.shape != (3, 3) or not np.all(np.isfinite(strain)):
            raise ValueError("strain_tensor must be a finite 3-by-3 array.")
        if not np.allclose(strain, strain.T, rtol=0.0, atol=1e-14):
            raise ValueError("strain_tensor must be symmetric.")
        frozen_strain = strain.copy()
        frozen_strain.setflags(write=False)
        object.__setattr__(self, "temperature_K", temperature)
        object.__setattr__(self, "strain_tensor", frozen_strain)


@dataclass(frozen=True)
class LabelledSpectrum:
    """Spin-1 eigensystem ordered by reference-state identity, not energy."""

    energies_MHz: np.ndarray
    eigenvectors: np.ndarray
    assigned_overlaps: np.ndarray
    minimum_assigned_overlap: float
    ambiguous: bool

    def __post_init__(self) -> None:
        energies = np.asarray(self.energies_MHz, dtype=float).copy()
        eigenvectors = np.asarray(self.eigenvectors, dtype=complex).copy()
        overlaps = np.asarray(self.assigned_overlaps, dtype=float).copy()
        if energies.shape != (3,) or overlaps.shape != (3,) or eigenvectors.shape != (3, 3):
            raise ValueError("A labelled spin-1 spectrum must contain three states.")
        for array in (energies, eigenvectors, overlaps):
            array.setflags(write=False)
        object.__setattr__(self, "energies_MHz", energies)
        object.__setattr__(self, "eigenvectors", eigenvectors)
        object.__setattr__(self, "assigned_overlaps", overlaps)


@dataclass(frozen=True)
class EnvironmentTransition:
    """Tracked sensing transition and carrier detunings at one environment."""

    condition: EnvironmentCondition
    spectrum: LabelledSpectrum
    transition_frequency_MHz: float
    reference_transition_frequency_MHz: float
    transition_shift_Hz: float
    fixed_reference_carrier_detuning_Hz: float
    ideal_retracked_carrier_detuning_Hz: float
    warnings: tuple[str, ...]

    @property
    def fixed_reference_carrier_detuning_rad_s(self) -> float:
        return 2.0 * math.pi * self.fixed_reference_carrier_detuning_Hz

    @property
    def ideal_retracked_carrier_detuning_rad_s(self) -> float:
        return 0.0


def spin_one_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Sx, Sy, Sz in the |+1>, |0>, |-1> basis."""
    root_two = math.sqrt(2.0)
    sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex) / root_two
    sy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=complex) / root_two
    sz = np.diag([1.0, 0.0, -1.0]).astype(complex)
    return sx, sy, sz


def magnetic_field_vector_T(config: dict[str, Any]) -> np.ndarray:
    """Convert the frozen spherical bias-field declaration to Cartesian form."""
    bias = config["reference_environment"]["magnetic_bias"]
    magnitude = float(bias["magnitude_T"])
    theta = float(bias["theta_from_nv_axis_rad"])
    phi = float(bias["azimuth_rad"])
    if not all(math.isfinite(value) for value in (magnitude, theta, phi)) or magnitude < 0.0:
        raise ValueError("Magnetic-bias values must be finite and magnitude non-negative.")
    return magnitude * np.array(
        [math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)],
        dtype=float,
    )


def spin_hamiltonian_MHz(
    condition: EnvironmentCondition,
    *,
    config: dict[str, Any] | None = None,
    laws: dict[str, Any] | None = None,
    coefficients: dict[str, dict[str, float | str]] | None = None,
) -> np.ndarray:
    """Construct H/h in MHz from D(T), Zeeman, and spin-strain terms."""
    if config is None:
        config = scan_config.load_config()
    if laws is None:
        laws = literature.load_temperature_laws()
    sx, sy, sz = spin_one_matrices()
    zero_field_splitting = float(literature.cambria_zfs_MHz(condition.temperature_K, laws))
    field = magnetic_field_vector_T(config)
    zeeman = ELECTRON_GYROMAGNETIC_RATIO_MHZ_PER_T * (
        field[0] * sx + field[1] * sy + field[2] * sz
    )
    strain = literature.spin_strain_hamiltonian_MHz(condition.strain_tensor, coefficients)
    hamiltonian = zero_field_splitting * (sz @ sz) + zeeman + strain
    if not np.allclose(hamiltonian, hamiltonian.conj().T, rtol=0.0, atol=1e-11):
        raise RuntimeError("Environment-conditioned Hamiltonian is not Hermitian.")
    return hamiltonian


def _best_overlap_assignment(
    eigenvectors: np.ndarray,
    tracking_vectors: np.ndarray,
) -> tuple[tuple[int, int, int], np.ndarray]:
    """Find the one-to-one state assignment with maximum total squared overlap."""
    candidates = np.asarray(eigenvectors, dtype=complex)
    trackers = np.asarray(tracking_vectors, dtype=complex)
    if candidates.shape != (3, 3) or trackers.shape != (3, 3):
        raise ValueError("eigenvectors and tracking_vectors must both be 3-by-3.")
    overlap_matrix = np.abs(trackers.conj().T @ candidates) ** 2
    assignment = max(
        permutations(range(3)),
        key=lambda candidate: sum(overlap_matrix[label, raw] for label, raw in enumerate(candidate)),
    )
    assigned = np.array(
        [overlap_matrix[label, raw] for label, raw in enumerate(assignment)], dtype=float
    )
    return assignment, assigned


def diagonalize_and_label(
    hamiltonian_MHz: np.ndarray,
    *,
    tracking_vectors: np.ndarray | None = None,
    ambiguity_threshold: float = 0.9,
) -> LabelledSpectrum:
    """Diagonalize a spin-1 Hamiltonian and preserve reference-state labels."""
    hamiltonian = np.asarray(hamiltonian_MHz, dtype=complex)
    if hamiltonian.shape != (3, 3) or not np.all(np.isfinite(hamiltonian)):
        raise ValueError("hamiltonian_MHz must be a finite 3-by-3 array.")
    if not np.allclose(hamiltonian, hamiltonian.conj().T, rtol=0.0, atol=1e-11):
        raise ValueError("hamiltonian_MHz must be Hermitian.")
    threshold = float(ambiguity_threshold)
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("ambiguity_threshold must lie in (0,1].")
    if tracking_vectors is None:
        tracking_vectors = np.eye(3, dtype=complex)

    raw_energies, raw_vectors = np.linalg.eigh(hamiltonian)
    assignment, assigned_overlaps = _best_overlap_assignment(raw_vectors, tracking_vectors)
    ordered_energies = raw_energies[list(assignment)]
    ordered_vectors = raw_vectors[:, list(assignment)].copy()

    # Fix arbitrary eigenvector phases relative to their tracking vectors. This
    # keeps sequential overlap diagnostics deterministic and continuous.
    trackers = np.asarray(tracking_vectors, dtype=complex)
    for label in range(3):
        inner = np.vdot(trackers[:, label], ordered_vectors[:, label])
        if abs(inner) > 0.0:
            ordered_vectors[:, label] *= np.exp(-1j * np.angle(inner))

    minimum = float(np.min(assigned_overlaps))
    return LabelledSpectrum(
        energies_MHz=ordered_energies,
        eigenvectors=ordered_vectors,
        assigned_overlaps=assigned_overlaps,
        minimum_assigned_overlap=minimum,
        ambiguous=minimum < threshold,
    )


def source_range_warnings(
    condition: EnvironmentCondition,
    *,
    config: dict[str, Any],
    laws: dict[str, Any],
) -> list[str]:
    """Return warnings without changing or clipping the requested condition."""
    messages: list[str] = []
    d_min, d_max = laws["zfs_two_mode_law"]["temperature_range_K"]
    if not d_min <= condition.temperature_K <= d_max:
        messages.append(
            f"temperature_outside_D_law_range:{condition.temperature_K:g}K_not_in_{d_min:g}-{d_max:g}K"
        )
    scan_limit = max(abs(value) for value in config["strain_axes"]["amplitudes_microstrain"]) * 1e-6
    maximum_component = float(np.max(np.abs(condition.strain_tensor)))
    if maximum_component > scan_limit + 1e-15:
        messages.append(
            f"strain_outside_preregistered_component_range:{maximum_component / 1e-6:g}_microstrain"
        )
    return messages


def reference_condition(config: dict[str, Any] | None = None) -> EnvironmentCondition:
    if config is None:
        config = scan_config.load_config()
    reference = config["reference_environment"]
    return EnvironmentCondition(
        reference["temperature_K"],
        reference["strain_tensor_nv_frame"],
        label="reference_environment",
    )


def reference_spectrum(
    *,
    config: dict[str, Any] | None = None,
    laws: dict[str, Any] | None = None,
    coefficients: dict[str, dict[str, float | str]] | None = None,
) -> LabelledSpectrum:
    if config is None:
        config = scan_config.load_config()
    threshold = config["reference_environment"]["transition"][
        "ambiguity_threshold_minimum_overlap"
    ]
    return diagonalize_and_label(
        spin_hamiltonian_MHz(
            reference_condition(config), config=config, laws=laws, coefficients=coefficients
        ),
        tracking_vectors=np.eye(3, dtype=complex),
        ambiguity_threshold=threshold,
    )


def transition_frequency_MHz(spectrum: LabelledSpectrum) -> float:
    """Return the positive |0>-like to |-1>-like transition frequency."""
    frequency = abs(
        float(spectrum.energies_MHz[MS_MINUS_INDEX] - spectrum.energies_MHz[MS_ZERO_INDEX])
    )
    if frequency <= 0.0:
        raise RuntimeError("Tracked transition frequency is not positive.")
    return frequency


def evaluate_transition(
    condition: EnvironmentCondition,
    *,
    config: dict[str, Any] | None = None,
    laws: dict[str, Any] | None = None,
    coefficients: dict[str, dict[str, float | str]] | None = None,
    tracking_vectors: np.ndarray | None = None,
    emit_warnings: bool = False,
) -> EnvironmentTransition:
    """Evaluate one condition relative to the frozen reference carrier."""
    if config is None:
        config = scan_config.load_config()
    if laws is None:
        laws = literature.load_temperature_laws()
    reference = reference_spectrum(config=config, laws=laws, coefficients=coefficients)
    if tracking_vectors is None:
        tracking_vectors = reference.eigenvectors
    threshold = config["reference_environment"]["transition"][
        "ambiguity_threshold_minimum_overlap"
    ]
    spectrum = diagonalize_and_label(
        spin_hamiltonian_MHz(
            condition, config=config, laws=laws, coefficients=coefficients
        ),
        tracking_vectors=tracking_vectors,
        ambiguity_threshold=threshold,
    )
    transition = transition_frequency_MHz(spectrum)
    reference_transition = transition_frequency_MHz(reference)
    shift_Hz = (transition - reference_transition) * 1e6
    messages = source_range_warnings(condition, config=config, laws=laws)
    if spectrum.ambiguous:
        messages.append(
            f"ambiguous_state_tracking:min_overlap={spectrum.minimum_assigned_overlap:.6g}"
        )
    if emit_warnings:
        for message in messages:
            python_warnings.warn(message, RuntimeWarning, stacklevel=2)
    return EnvironmentTransition(
        condition=condition,
        spectrum=spectrum,
        transition_frequency_MHz=transition,
        reference_transition_frequency_MHz=reference_transition,
        transition_shift_Hz=shift_Hz,
        fixed_reference_carrier_detuning_Hz=shift_Hz,
        ideal_retracked_carrier_detuning_Hz=0.0,
        warnings=tuple(messages),
    )


def carrier_detuning_Hz(result: EnvironmentTransition, policy_id: str) -> float:
    if policy_id == "fixed_reference_carrier":
        return result.fixed_reference_carrier_detuning_Hz
    if policy_id == "ideal_retracked_carrier":
        return result.ideal_retracked_carrier_detuning_Hz
    raise ValueError(f"Unknown carrier policy: {policy_id!r}.")


def strain_tensor_from_path(
    path_id: str,
    amplitude_microstrain: float,
    *,
    config: dict[str, Any] | None = None,
    enforce_preregistered_range: bool = True,
) -> np.ndarray:
    """Construct a dimensionless strain tensor from one frozen scan path."""
    if config is None:
        config = scan_config.load_config()
    amplitude = float(amplitude_microstrain)
    if not math.isfinite(amplitude):
        raise ValueError("amplitude_microstrain must be finite.")
    allowed = config["strain_axes"]["amplitudes_microstrain"]
    if enforce_preregistered_range and not min(allowed) <= amplitude <= max(allowed):
        raise ValueError("Strain amplitude lies outside the preregistered range.")
    matches = [path for path in config["strain_axes"]["paths"] if path["id"] == path_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate strain path: {path_id!r}.")
    tensor = np.asarray(matches[0]["tensor_per_unit_amplitude"], dtype=float)
    if path_id == "zero_strain" and amplitude != 0.0:
        raise ValueError("zero_strain may only be evaluated at zero amplitude.")
    return tensor * amplitude * 1e-6


def track_environment_path(
    conditions: Iterable[EnvironmentCondition],
    *,
    config: dict[str, Any] | None = None,
    laws: dict[str, Any] | None = None,
    coefficients: dict[str, dict[str, float | str]] | None = None,
    anchor_index: int | None = None,
) -> list[EnvironmentTransition]:
    """Track states continuously in both directions from a reference-like anchor."""
    points = list(conditions)
    if not points:
        raise ValueError("At least one environment condition is required.")
    if config is None:
        config = scan_config.load_config()
    if laws is None:
        laws = literature.load_temperature_laws()
    if anchor_index is None:
        reference_temperature = float(config["reference_environment"]["temperature_K"])
        scale = max(abs(value) for value in config["strain_axes"]["amplitudes_microstrain"]) * 1e-6
        anchor_index = min(
            range(len(points)),
            key=lambda index: (
                abs(points[index].temperature_K - reference_temperature) / 50.0
                + np.linalg.norm(points[index].strain_tensor) / max(scale, 1e-30)
            ),
        )
    if isinstance(anchor_index, bool) or not 0 <= int(anchor_index) < len(points):
        raise ValueError("anchor_index is outside the condition list.")
    anchor_index = int(anchor_index)

    output: list[EnvironmentTransition | None] = [None] * len(points)
    output[anchor_index] = evaluate_transition(
        points[anchor_index], config=config, laws=laws, coefficients=coefficients
    )
    for index in range(anchor_index + 1, len(points)):
        previous = output[index - 1]
        assert previous is not None
        output[index] = evaluate_transition(
            points[index],
            config=config,
            laws=laws,
            coefficients=coefficients,
            tracking_vectors=previous.spectrum.eigenvectors,
        )
    for index in range(anchor_index - 1, -1, -1):
        following = output[index + 1]
        assert following is not None
        output[index] = evaluate_transition(
            points[index],
            config=config,
            laws=laws,
            coefficients=coefficients,
            tracking_vectors=following.spectrum.eigenvectors,
        )
    return [result for result in output if result is not None]
