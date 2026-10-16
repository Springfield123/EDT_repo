"""Two-level propagation for conventional XY8 control.

Stages 1--4 provide instantaneous and finite-duration propagation, weighted
quasi-static ensemble averaging, and a central-difference small-signal sensing
response.  This module does not yet change ``eta_ac``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


COMPLEX_DTYPE = np.complex128
IDENTITY = np.eye(2, dtype=COMPLEX_DTYPE)
SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=COMPLEX_DTYPE)
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=COMPLEX_DTYPE)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=COMPLEX_DTYPE)
PAULI = np.stack([SIGMA_X, SIGMA_Y, SIGMA_Z])
ELECTRON_GYROMAGNETIC_RATIO_RAD_S_T = 2.0 * np.pi * 28.02495164e9

# Conventional XY8 pulse phases: X-Y-X-Y-Y-X-Y-X.
XY8_BLOCK_PHASES_RAD = np.array(
    [0.0, np.pi / 2.0, 0.0, np.pi / 2.0, np.pi / 2.0, 0.0, np.pi / 2.0, 0.0],
    dtype=float,
)


@dataclass(frozen=True)
class XY8PropagationResult:
    """Final state and metadata from a deterministic XY8 propagation."""

    unitary: np.ndarray
    density_matrix: np.ndarray
    bloch_vector: np.ndarray
    blocks: int
    total_pi_pulses: int
    pulse_phases_rad: np.ndarray


@dataclass(frozen=True)
class XY8ScheduleSegment:
    """One free-evolution or driven-pulse interval in an XY8 schedule."""

    kind: str
    start_s: float
    duration_s: float
    pulse_index: int | None = None
    phase_rad: float | None = None

    @property
    def stop_s(self) -> float:
        return self.start_s + self.duration_s


@dataclass(frozen=True)
class XY8FinitePropagationResult(XY8PropagationResult):
    """Final state and timing metadata from finite-duration XY8."""

    tau_s: float
    sequence_duration_s: float
    pi_pulse_duration_s: float
    detuning_angular_frequency_rad_s: float
    schedule: tuple[XY8ScheduleSegment, ...]


@dataclass(frozen=True)
class QuasiStaticEnsemble:
    """Weighted amplitude-error and detuning samples held for a full sequence.

    ``rotation_angle_errors`` are fractional microwave-amplitude errors and
    ``detunings_angular_frequency_rad_s`` use angular-frequency units.  The
    constructor validates and normalizes the weights.  ``hyperfine_line_indices``
    identifies the parent hyperfine component when such a mixture is used; -1
    means that no named line was assigned.
    """

    rotation_angle_errors: np.ndarray
    detunings_angular_frequency_rad_s: np.ndarray
    weights: np.ndarray
    hyperfine_line_indices: np.ndarray | None = None

    def __post_init__(self) -> None:
        angle_errors = np.asarray(self.rotation_angle_errors, dtype=float)
        detunings = np.asarray(self.detunings_angular_frequency_rad_s, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        if self.hyperfine_line_indices is None:
            line_indices = np.full(angle_errors.size, -1, dtype=int)
        else:
            line_indices = np.asarray(self.hyperfine_line_indices, dtype=int)
        arrays = [angle_errors, detunings, weights, line_indices]
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("Quasi-static ensemble fields must be one-dimensional.")
        if angle_errors.size == 0:
            raise ValueError("A quasi-static ensemble must contain at least one sample.")
        if any(array.size != angle_errors.size for array in arrays[1:]):
            raise ValueError("All quasi-static ensemble fields must have the same length.")
        if not np.all(np.isfinite(angle_errors)) or not np.all(np.isfinite(detunings)):
            raise ValueError("Quasi-static errors and detunings must be finite.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("Quasi-static weights must be finite and non-negative.")
        total_weight = float(np.sum(weights))
        if total_weight <= 0.0:
            raise ValueError("At least one quasi-static weight must be positive.")

        normalized = weights / total_weight
        validated = [angle_errors.copy(), detunings.copy(), normalized, line_indices.copy()]
        for array in validated:
            array.setflags(write=False)
        object.__setattr__(self, "rotation_angle_errors", validated[0])
        object.__setattr__(self, "detunings_angular_frequency_rad_s", validated[1])
        object.__setattr__(self, "weights", validated[2])
        object.__setattr__(self, "hyperfine_line_indices", validated[3])

    @property
    def sample_count(self) -> int:
        return int(self.weights.size)


@dataclass(frozen=True)
class XY8EnsemblePropagationResult:
    """Weighted output state from quasi-static finite-duration propagation."""

    density_matrix: np.ndarray
    bloch_vector: np.ndarray
    purity: float
    sample_density_matrices: np.ndarray
    sample_bloch_vectors: np.ndarray
    sample_purities: np.ndarray
    ensemble: QuasiStaticEnsemble
    blocks: int
    total_pi_pulses: int
    tau_s: float
    sequence_duration_s: float
    pi_pulse_duration_s: float
    schedule: tuple[XY8ScheduleSegment, ...]


@dataclass(frozen=True)
class XY8SmallSignalResult:
    """Central-difference magnetic response and normalized coherent gain."""

    derivative_per_t: float
    normalized_signed_gain: float
    normalized_gain: float
    expectation_plus: float
    expectation_minus: float
    density_matrix_plus: np.ndarray
    density_matrix_minus: np.ndarray
    bloch_derivative_per_t: np.ndarray
    delta_field_t: float
    signal_frequency_hz: float
    signal_phase_rad: float
    readout_phase_rad: float
    phase_selection: str
    gyromagnetic_ratio_rad_s_t: float
    sequence_duration_s: float
    blocks: int
    total_pi_pulses: int
    tau_s: float
    pi_pulse_duration_s: float
    pulse_substeps: int
    ensemble: QuasiStaticEnsemble


def _validated_blocks(blocks: int) -> int:
    if isinstance(blocks, bool) or int(blocks) != blocks or int(blocks) < 1:
        raise ValueError("XY8 block count must be a positive integer.")
    return int(blocks)


def _expand_pulse_parameter(value, count: int, name: str) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    if values.ndim == 0:
        return np.full(count, float(values), dtype=float)
    if values.ndim != 1 or values.size != count:
        raise ValueError(f"{name} must be a scalar or contain exactly {count} values.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values.astype(float, copy=True)


def _validated_nonnegative_finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a non-negative finite value.")
    return value


def make_quasi_static_ensemble(
    rotation_angle_errors,
    detunings_angular_frequency_rad_s,
    *,
    weights=None,
    hyperfine_line_indices=None,
) -> QuasiStaticEnsemble:
    """Construct a validated ensemble from explicit weighted samples.

    Inputs describe absolute sample values, not offsets to another amplitude
    error or detuning.  Supplied weights need not already sum to one.
    """
    angle_errors = np.asarray(rotation_angle_errors, dtype=float)
    detunings = np.asarray(detunings_angular_frequency_rad_s, dtype=float)
    if angle_errors.ndim != 1 or detunings.ndim != 1:
        raise ValueError("Explicit quasi-static samples must be one-dimensional.")
    if angle_errors.size != detunings.size:
        raise ValueError("Amplitude-error and detuning samples must have the same length.")
    if weights is None:
        weights = np.ones(angle_errors.size, dtype=float)
    return QuasiStaticEnsemble(
        rotation_angle_errors=angle_errors,
        detunings_angular_frequency_rad_s=detunings,
        weights=np.asarray(weights, dtype=float),
        hyperfine_line_indices=hyperfine_line_indices,
    )


def _normal_quadrature(mean: float, standard_deviation: float, order: int):
    mean = float(mean)
    standard_deviation = _validated_nonnegative_finite(
        standard_deviation, "standard_deviation"
    )
    if not np.isfinite(mean):
        raise ValueError("Normal-distribution mean must be finite.")
    if isinstance(order, bool) or int(order) != order or int(order) < 1:
        raise ValueError("quadrature_order must be a positive integer.")
    if standard_deviation == 0.0:
        return np.array([mean]), np.array([1.0])
    nodes, weights = np.polynomial.hermite.hermgauss(int(order))
    return (
        mean + np.sqrt(2.0) * standard_deviation * nodes,
        weights / np.sqrt(np.pi),
    )


def _validated_hyperfine_lines(detunings, weights):
    if detunings is None:
        if weights is not None:
            raise ValueError("hyperfine_weights require hyperfine_detunings_rad_s.")
        return np.array([0.0]), np.array([1.0]), np.array([-1], dtype=int)
    offsets = np.asarray(detunings, dtype=float)
    if offsets.ndim != 1 or offsets.size == 0 or not np.all(np.isfinite(offsets)):
        raise ValueError("hyperfine_detunings_rad_s must contain finite line offsets.")
    if weights is None:
        populations = np.ones(offsets.size, dtype=float)
    else:
        populations = np.asarray(weights, dtype=float)
    if populations.ndim != 1 or populations.size != offsets.size:
        raise ValueError("hyperfine_weights must match the number of line offsets.")
    if not np.all(np.isfinite(populations)) or np.any(populations < 0.0):
        raise ValueError("hyperfine_weights must be finite and non-negative.")
    total = float(np.sum(populations))
    if total <= 0.0:
        raise ValueError("At least one hyperfine weight must be positive.")
    return offsets, populations / total, np.arange(offsets.size, dtype=int)


def gaussian_quadrature_ensemble(
    *,
    rotation_angle_error_mean: float = 0.0,
    rotation_angle_error_std: float = 0.0,
    detuning_mean_angular_frequency_rad_s: float = 0.0,
    detuning_std_angular_frequency_rad_s: float = 0.0,
    quadrature_order: int = 5,
    hyperfine_detunings_rad_s=None,
    hyperfine_weights=None,
) -> QuasiStaticEnsemble:
    """Create a deterministic Gaussian-Hermite product ensemble.

    Each hyperfine-line offset is combined with the common detuning
    distribution.  Its population multiplies the Gaussian quadrature weight.
    """
    amplitude_nodes, amplitude_weights = _normal_quadrature(
        rotation_angle_error_mean, rotation_angle_error_std, quadrature_order
    )
    detuning_nodes, detuning_weights = _normal_quadrature(
        detuning_mean_angular_frequency_rad_s,
        detuning_std_angular_frequency_rad_s,
        quadrature_order,
    )
    line_offsets, line_weights, line_indices = _validated_hyperfine_lines(
        hyperfine_detunings_rad_s, hyperfine_weights
    )

    samples = [
        (amplitude, detuning + line_offset, wa * wd * wl, line_index)
        for line_offset, wl, line_index in zip(line_offsets, line_weights, line_indices)
        for amplitude, wa in zip(amplitude_nodes, amplitude_weights)
        for detuning, wd in zip(detuning_nodes, detuning_weights)
    ]
    return make_quasi_static_ensemble(
        [sample[0] for sample in samples],
        [sample[1] for sample in samples],
        weights=[sample[2] for sample in samples],
        hyperfine_line_indices=[sample[3] for sample in samples],
    )


def monte_carlo_quasi_static_ensemble(
    sample_count: int,
    *,
    seed: int,
    rotation_angle_error_mean: float = 0.0,
    rotation_angle_error_std: float = 0.0,
    detuning_mean_angular_frequency_rad_s: float = 0.0,
    detuning_std_angular_frequency_rad_s: float = 0.0,
    hyperfine_detunings_rad_s=None,
    hyperfine_weights=None,
) -> QuasiStaticEnsemble:
    """Draw a reproducible, equally weighted quasi-static ensemble."""
    if isinstance(sample_count, bool) or int(sample_count) != sample_count or sample_count < 1:
        raise ValueError("sample_count must be a positive integer.")
    amplitude_std = _validated_nonnegative_finite(
        rotation_angle_error_std, "rotation_angle_error_std"
    )
    detuning_std = _validated_nonnegative_finite(
        detuning_std_angular_frequency_rad_s,
        "detuning_std_angular_frequency_rad_s",
    )
    amplitude_mean = float(rotation_angle_error_mean)
    detuning_mean = float(detuning_mean_angular_frequency_rad_s)
    if not np.isfinite(amplitude_mean) or not np.isfinite(detuning_mean):
        raise ValueError("Monte Carlo distribution means must be finite.")
    line_offsets, line_weights, line_indices = _validated_hyperfine_lines(
        hyperfine_detunings_rad_s, hyperfine_weights
    )
    rng = np.random.default_rng(seed)
    count = int(sample_count)
    sampled_line_positions = rng.choice(line_offsets.size, size=count, p=line_weights)
    amplitude_errors = rng.normal(amplitude_mean, amplitude_std, size=count)
    detunings = (
        rng.normal(detuning_mean, detuning_std, size=count)
        + line_offsets[sampled_line_positions]
    )
    return make_quasi_static_ensemble(
        amplitude_errors,
        detunings,
        weights=np.ones(count),
        hyperfine_line_indices=line_indices[sampled_line_positions],
    )


def xy8_pulse_phases(blocks: int) -> np.ndarray:
    """Return the intended phase of all ``8 * blocks`` pi pulses."""
    blocks = _validated_blocks(blocks)
    return np.tile(XY8_BLOCK_PHASES_RAD, blocks)


def rotation_unitary(
    angle_rad: float,
    phase_rad: float,
    *,
    axis_z: float = 0.0,
) -> np.ndarray:
    """Return a two-level rotation about an axis near the equatorial plane.

    ``phase_rad`` selects the transverse axis: zero is X and pi/2 is Y.
    ``axis_z`` is the normalized z component of the rotation axis.  It is a
    deterministic coherent error, not an independent loss probability.
    """
    angle = float(angle_rad)
    phase = float(phase_rad)
    nz = float(axis_z)
    if not np.isfinite(angle) or not np.isfinite(phase) or not np.isfinite(nz):
        raise ValueError("Rotation parameters must be finite.")
    if abs(nz) > 1.0:
        raise ValueError("axis_z must lie in [-1, 1].")

    transverse = np.sqrt(max(0.0, 1.0 - nz * nz))
    axis_operator = (
        transverse * np.cos(phase) * SIGMA_X
        + transverse * np.sin(phase) * SIGMA_Y
        + nz * SIGMA_Z
    )
    return (
        np.cos(angle / 2.0) * IDENTITY
        - 1.0j * np.sin(angle / 2.0) * axis_operator
    )


def constant_hamiltonian_unitary(
    angular_frequency_vector_rad_s,
    duration_s: float,
) -> np.ndarray:
    """Propagate ``H = (hx*sx + hy*sy + hz*sz)/2`` for one interval.

    The three Hamiltonian coefficients are angular frequencies in rad/s, so
    the resulting rotation angle is ``norm(h) * duration``.
    """
    vector = np.asarray(angular_frequency_vector_rad_s, dtype=float)
    duration = _validated_nonnegative_finite(duration_s, "duration_s")
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("Hamiltonian angular-frequency vector must contain three finite values.")
    magnitude = float(np.linalg.norm(vector))
    if magnitude == 0.0 or duration == 0.0:
        return IDENTITY.copy()
    axis = vector / magnitude
    operator = np.tensordot(axis, PAULI, axes=1)
    angle = magnitude * duration
    return np.cos(angle / 2.0) * IDENTITY - 1.0j * np.sin(angle / 2.0) * operator


def driven_pi_pulse_unitary(
    phase_rad: float,
    pi_pulse_duration_s: float,
    *,
    detuning_angular_frequency_rad_s: float = 0.0,
    rotation_angle_error: float = 0.0,
    axis_z: float = 0.0,
    phase_error_rad: float = 0.0,
) -> np.ndarray:
    """Return one finite-duration nominal pi-pulse propagator.

    The nominal on-resonance Rabi angular frequency is ``pi / t_pi``.
    ``rotation_angle_error`` scales the drive amplitude coherently.  ``axis_z``
    is an independently calibrated drive-axis tilt; static detuning is added
    separately and should not be used to describe that same tilt twice.
    """
    duration = _validated_nonnegative_finite(pi_pulse_duration_s, "pi_pulse_duration_s")
    detuning = float(detuning_angular_frequency_rad_s)
    angle_error = float(rotation_angle_error)
    nz = float(axis_z)
    phase = float(phase_rad) + float(phase_error_rad)
    if not all(np.isfinite(value) for value in [detuning, angle_error, nz, phase]):
        raise ValueError("Driven-pulse parameters must be finite.")
    if abs(nz) > 1.0:
        raise ValueError("axis_z must lie in [-1, 1].")
    if duration == 0.0:
        return rotation_unitary(np.pi * (1.0 + angle_error), phase, axis_z=nz)

    nominal_rabi = np.pi / duration
    actual_rabi = nominal_rabi * (1.0 + angle_error)
    transverse = np.sqrt(max(0.0, 1.0 - nz * nz))
    angular_frequency_vector = np.array(
        [
            actual_rabi * transverse * np.cos(phase),
            actual_rabi * transverse * np.sin(phase),
            actual_rabi * nz + detuning,
        ],
        dtype=float,
    )
    return constant_hamiltonian_unitary(angular_frequency_vector, duration)


def density_matrix_from_bloch(bloch_vector) -> np.ndarray:
    """Create a physical qubit density matrix from a Bloch vector."""
    vector = np.asarray(bloch_vector, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("A Bloch vector must contain three finite values.")
    if np.linalg.norm(vector) > 1.0 + 1e-12:
        raise ValueError("A physical Bloch vector cannot have norm greater than one.")
    return 0.5 * (IDENTITY + np.tensordot(vector, PAULI, axes=1))


def bloch_vector_from_density_matrix(density_matrix) -> np.ndarray:
    """Return the real Bloch vector of a validated qubit density matrix."""
    rho = _validated_density_matrix(density_matrix)
    return np.real(np.array([np.trace(rho @ sigma) for sigma in PAULI], dtype=complex))


def _validated_density_matrix(state_or_density) -> np.ndarray:
    value = np.asarray(state_or_density, dtype=COMPLEX_DTYPE)
    if value.shape == (2,):
        norm = float(np.linalg.norm(value))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("A state vector must have non-zero finite norm.")
        state = value / norm
        return np.outer(state, state.conj())
    if value.shape != (2, 2):
        raise ValueError("Initial state must be a length-two state vector or 2x2 density matrix.")
    if not np.all(np.isfinite(value)):
        raise ValueError("Density-matrix entries must be finite.")
    if not np.allclose(value, value.conj().T, atol=1e-12, rtol=0.0):
        raise ValueError("Density matrix must be Hermitian.")
    trace = np.trace(value)
    if not np.isclose(trace, 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("Density matrix must have unit trace.")
    eigenvalues = np.linalg.eigvalsh(value)
    if float(np.min(eigenvalues)) < -1e-12:
        raise ValueError("Density matrix must be positive semidefinite.")
    return value.copy()


def xy8_unitary(
    blocks: int,
    *,
    rotation_angle_error=0.0,
    axis_z=0.0,
    phase_error_rad=0.0,
) -> np.ndarray:
    """Compose a deterministic instantaneous-pulse XY8 unitary.

    Error arguments may be scalars (the same quasi-static error on every
    pulse) or arrays with one value per pulse.  The rotation angle of pulse
    ``k`` is ``pi * (1 + rotation_angle_error[k])``.
    """
    phases = xy8_pulse_phases(blocks)
    count = phases.size
    angle_errors = _expand_pulse_parameter(
        rotation_angle_error, count, "rotation_angle_error"
    )
    axis_z_values = _expand_pulse_parameter(axis_z, count, "axis_z")
    phase_errors = _expand_pulse_parameter(phase_error_rad, count, "phase_error_rad")
    if np.any(np.abs(axis_z_values) > 1.0):
        raise ValueError("Every axis_z value must lie in [-1, 1].")

    total = IDENTITY.copy()
    for phase, angle_error, nz, phase_error in zip(
        phases, angle_errors, axis_z_values, phase_errors
    ):
        pulse = rotation_unitary(
            np.pi * (1.0 + angle_error),
            phase + phase_error,
            axis_z=nz,
        )
        total = pulse @ total
    return total


def build_xy8_finite_schedule(
    tau_s: float,
    blocks: int,
    pi_pulse_duration_s: float,
) -> tuple[XY8ScheduleSegment, ...]:
    """Build finite pulses around the canonical XY8 pulse centres.

    Project convention: ``tau`` is half the centre-to-centre pulse spacing.
    Pulse centres are ``(2k-1)*tau`` and total duration remains ``16*N*tau``.
    """
    tau = _validated_nonnegative_finite(tau_s, "tau_s")
    if tau == 0.0:
        raise ValueError("tau_s must be greater than zero.")
    blocks = _validated_blocks(blocks)
    pulse_duration = _validated_nonnegative_finite(
        pi_pulse_duration_s, "pi_pulse_duration_s"
    )
    if pulse_duration >= 2.0 * tau and pulse_duration > 0.0:
        raise ValueError("pi_pulse_duration_s must be smaller than 2*tau_s.")

    phases = xy8_pulse_phases(blocks)
    pulse_centres = (2.0 * np.arange(1, phases.size + 1, dtype=float) - 1.0) * tau
    total_duration = 16.0 * blocks * tau
    segments: list[XY8ScheduleSegment] = []
    cursor = 0.0
    for pulse_index, (centre, phase) in enumerate(zip(pulse_centres, phases)):
        pulse_start = float(centre - pulse_duration / 2.0)
        free_duration = pulse_start - cursor
        if free_duration < -1e-15:
            raise ValueError("Finite pulses overlap under the requested timing convention.")
        segments.append(
            XY8ScheduleSegment(
                kind="free",
                start_s=float(cursor),
                duration_s=float(max(0.0, free_duration)),
            )
        )
        segments.append(
            XY8ScheduleSegment(
                kind="pulse",
                start_s=pulse_start,
                duration_s=pulse_duration,
                pulse_index=pulse_index,
                phase_rad=float(phase),
            )
        )
        cursor = pulse_start + pulse_duration

    final_free_duration = total_duration - cursor
    if final_free_duration < -1e-15:
        raise ValueError("Finite-pulse schedule extends beyond the sequence duration.")
    segments.append(
        XY8ScheduleSegment(
            kind="free",
            start_s=float(cursor),
            duration_s=float(max(0.0, final_free_duration)),
        )
    )
    return tuple(segments)


def xy8_finite_unitary(
    tau_s: float,
    blocks: int,
    pi_pulse_duration_s: float,
    *,
    detuning_angular_frequency_rad_s: float = 0.0,
    rotation_angle_error=0.0,
    axis_z=0.0,
    phase_error_rad=0.0,
) -> np.ndarray:
    """Compose finite driven pulses and free static-detuning evolution."""
    blocks = _validated_blocks(blocks)
    detuning = float(detuning_angular_frequency_rad_s)
    if not np.isfinite(detuning):
        raise ValueError("detuning_angular_frequency_rad_s must be finite.")
    schedule = build_xy8_finite_schedule(tau_s, blocks, pi_pulse_duration_s)
    count = 8 * blocks
    angle_errors = _expand_pulse_parameter(
        rotation_angle_error, count, "rotation_angle_error"
    )
    axis_z_values = _expand_pulse_parameter(axis_z, count, "axis_z")
    phase_errors = _expand_pulse_parameter(phase_error_rad, count, "phase_error_rad")
    if np.any(np.abs(axis_z_values) > 1.0):
        raise ValueError("Every axis_z value must lie in [-1, 1].")

    total = IDENTITY.copy()
    for segment in schedule:
        if segment.kind == "free":
            interval = constant_hamiltonian_unitary(
                [0.0, 0.0, detuning], segment.duration_s
            )
        else:
            pulse_index = int(segment.pulse_index)
            interval = driven_pi_pulse_unitary(
                float(segment.phase_rad),
                segment.duration_s,
                detuning_angular_frequency_rad_s=detuning,
                rotation_angle_error=angle_errors[pulse_index],
                axis_z=axis_z_values[pulse_index],
                phase_error_rad=phase_errors[pulse_index],
            )
        total = interval @ total
    return total


def ac_signal_integral(
    start_s: float,
    duration_s: float,
    frequency_hz: float,
    phase_rad: float,
) -> float:
    """Return the exact integral of ``cos(2*pi*f*t + phase)`` over an interval."""
    start = _validated_nonnegative_finite(start_s, "start_s")
    duration = _validated_nonnegative_finite(duration_s, "duration_s")
    frequency = _validated_nonnegative_finite(frequency_hz, "frequency_hz")
    phase = float(phase_rad)
    if not np.isfinite(phase):
        raise ValueError("phase_rad must be finite.")
    if duration == 0.0:
        return 0.0
    angular_frequency = 2.0 * np.pi * frequency
    if angular_frequency == 0.0:
        return float(duration * np.cos(phase))
    stop = start + duration
    return float(
        (np.sin(angular_frequency * stop + phase)
         - np.sin(angular_frequency * start + phase))
        / angular_frequency
    )


def xy8_sensing_unitary(
    tau_s: float,
    blocks: int,
    pi_pulse_duration_s: float,
    *,
    signal_frequency_hz: float,
    signal_field_t: float,
    signal_phase_rad: float,
    gyromagnetic_ratio_rad_s_t: float = ELECTRON_GYROMAGNETIC_RATIO_RAD_S_T,
    detuning_angular_frequency_rad_s: float = 0.0,
    rotation_angle_error=0.0,
    axis_z=0.0,
    phase_error_rad=0.0,
    pulse_substeps: int = 8,
) -> np.ndarray:
    """Compose XY8 evolution in a sinusoidal longitudinal sensing field.

    Free evolution uses the analytic signal integral because all longitudinal
    terms commute.  During each finite driven pulse, the non-commuting signal
    and drive Hamiltonians are integrated with midpoint substeps.  A zero-width
    pulse remains instantaneous and therefore accumulates no sensing phase.
    """
    blocks = _validated_blocks(blocks)
    frequency = _validated_nonnegative_finite(signal_frequency_hz, "signal_frequency_hz")
    field = float(signal_field_t)
    signal_phase = float(signal_phase_rad)
    gamma = float(gyromagnetic_ratio_rad_s_t)
    detuning = float(detuning_angular_frequency_rad_s)
    if not all(np.isfinite(value) for value in [field, signal_phase, gamma, detuning]):
        raise ValueError("Sensing-field, phase, gyromagnetic-ratio, and detuning inputs must be finite.")
    if gamma <= 0.0:
        raise ValueError("gyromagnetic_ratio_rad_s_t must be greater than zero.")
    if isinstance(pulse_substeps, bool) or int(pulse_substeps) != pulse_substeps or int(pulse_substeps) < 1:
        raise ValueError("pulse_substeps must be a positive integer.")
    pulse_substeps = int(pulse_substeps)

    schedule = build_xy8_finite_schedule(tau_s, blocks, pi_pulse_duration_s)
    count = 8 * blocks
    angle_errors = _expand_pulse_parameter(
        rotation_angle_error, count, "rotation_angle_error"
    )
    axis_z_values = _expand_pulse_parameter(axis_z, count, "axis_z")
    phase_errors = _expand_pulse_parameter(phase_error_rad, count, "phase_error_rad")
    if np.any(np.abs(axis_z_values) > 1.0):
        raise ValueError("Every axis_z value must lie in [-1, 1].")

    total = IDENTITY.copy()
    angular_signal_amplitude = gamma * field
    for segment in schedule:
        if segment.kind == "free":
            signal_angle = angular_signal_amplitude * ac_signal_integral(
                segment.start_s,
                segment.duration_s,
                frequency,
                signal_phase,
            )
            static_angle = detuning * segment.duration_s
            interval = rotation_unitary(
                static_angle + signal_angle,
                0.0,
                axis_z=1.0,
            )
            total = interval @ total
            continue

        pulse_index = int(segment.pulse_index)
        if segment.duration_s == 0.0:
            interval = driven_pi_pulse_unitary(
                float(segment.phase_rad),
                0.0,
                detuning_angular_frequency_rad_s=detuning,
                rotation_angle_error=angle_errors[pulse_index],
                axis_z=axis_z_values[pulse_index],
                phase_error_rad=phase_errors[pulse_index],
            )
            total = interval @ total
            continue

        substep_duration = segment.duration_s / pulse_substeps
        nominal_rabi = np.pi / segment.duration_s
        actual_rabi = nominal_rabi * (1.0 + angle_errors[pulse_index])
        nz = axis_z_values[pulse_index]
        transverse = np.sqrt(max(0.0, 1.0 - nz * nz))
        drive_phase = float(segment.phase_rad) + phase_errors[pulse_index]
        drive_x = actual_rabi * transverse * np.cos(drive_phase)
        drive_y = actual_rabi * transverse * np.sin(drive_phase)
        drive_z = actual_rabi * nz
        for substep in range(pulse_substeps):
            midpoint = segment.start_s + (substep + 0.5) * substep_duration
            signal_z = angular_signal_amplitude * np.cos(
                2.0 * np.pi * frequency * midpoint + signal_phase
            )
            interval = constant_hamiltonian_unitary(
                [drive_x, drive_y, drive_z + detuning + signal_z],
                substep_duration,
            )
            total = interval @ total
    return total


def propagate_xy8(
    initial_state,
    blocks: int,
    *,
    rotation_angle_error=0.0,
    axis_z=0.0,
    phase_error_rad=0.0,
) -> XY8PropagationResult:
    """Propagate a state vector or density matrix through instantaneous XY8."""
    rho_initial = _validated_density_matrix(initial_state)
    phases = xy8_pulse_phases(blocks)
    unitary = xy8_unitary(
        blocks,
        rotation_angle_error=rotation_angle_error,
        axis_z=axis_z,
        phase_error_rad=phase_error_rad,
    )
    rho_final = unitary @ rho_initial @ unitary.conj().T
    rho_final = 0.5 * (rho_final + rho_final.conj().T)
    rho_final /= np.trace(rho_final)
    return XY8PropagationResult(
        unitary=unitary,
        density_matrix=rho_final,
        bloch_vector=bloch_vector_from_density_matrix(rho_final),
        blocks=_validated_blocks(blocks),
        total_pi_pulses=int(phases.size),
        pulse_phases_rad=phases,
    )


def propagate_xy8_finite(
    initial_state,
    tau_s: float,
    blocks: int,
    pi_pulse_duration_s: float,
    *,
    detuning_angular_frequency_rad_s: float = 0.0,
    rotation_angle_error=0.0,
    axis_z=0.0,
    phase_error_rad=0.0,
) -> XY8FinitePropagationResult:
    """Propagate a state through the finite-duration, static-detuning schedule."""
    rho_initial = _validated_density_matrix(initial_state)
    blocks = _validated_blocks(blocks)
    tau = _validated_nonnegative_finite(tau_s, "tau_s")
    pulse_duration = _validated_nonnegative_finite(
        pi_pulse_duration_s, "pi_pulse_duration_s"
    )
    detuning = float(detuning_angular_frequency_rad_s)
    schedule = build_xy8_finite_schedule(tau, blocks, pulse_duration)
    phases = xy8_pulse_phases(blocks)
    unitary = xy8_finite_unitary(
        tau,
        blocks,
        pulse_duration,
        detuning_angular_frequency_rad_s=detuning,
        rotation_angle_error=rotation_angle_error,
        axis_z=axis_z,
        phase_error_rad=phase_error_rad,
    )
    rho_final = unitary @ rho_initial @ unitary.conj().T
    rho_final = 0.5 * (rho_final + rho_final.conj().T)
    rho_final /= np.trace(rho_final)
    return XY8FinitePropagationResult(
        unitary=unitary,
        density_matrix=rho_final,
        bloch_vector=bloch_vector_from_density_matrix(rho_final),
        blocks=blocks,
        total_pi_pulses=int(phases.size),
        pulse_phases_rad=phases,
        tau_s=tau,
        sequence_duration_s=16.0 * blocks * tau,
        pi_pulse_duration_s=pulse_duration,
        detuning_angular_frequency_rad_s=detuning,
        schedule=schedule,
    )


def density_matrix_purity(density_matrix) -> float:
    """Return ``Tr(rho**2)`` for a validated qubit density matrix."""
    rho = _validated_density_matrix(density_matrix)
    return float(np.real(np.trace(rho @ rho)))


def propagate_xy8_ensemble(
    initial_state,
    tau_s: float,
    blocks: int,
    pi_pulse_duration_s: float,
    ensemble: QuasiStaticEnsemble,
    *,
    axis_z=0.0,
    phase_error_rad=0.0,
) -> XY8EnsemblePropagationResult:
    """Propagate and average errors that remain fixed for each full sequence.

    A separate finite-duration propagation is performed for every ensemble
    member.  Density matrices are then averaged with the normalized weights;
    amplitudes or unsigned contrasts are never averaged in their place.
    """
    if not isinstance(ensemble, QuasiStaticEnsemble):
        raise TypeError("ensemble must be a QuasiStaticEnsemble.")
    rho_initial = _validated_density_matrix(initial_state)
    blocks = _validated_blocks(blocks)
    tau = _validated_nonnegative_finite(tau_s, "tau_s")
    pulse_duration = _validated_nonnegative_finite(
        pi_pulse_duration_s, "pi_pulse_duration_s"
    )
    schedule = build_xy8_finite_schedule(tau, blocks, pulse_duration)

    sample_density_matrices = np.empty(
        (ensemble.sample_count, 2, 2), dtype=COMPLEX_DTYPE
    )
    sample_bloch_vectors = np.empty((ensemble.sample_count, 3), dtype=float)
    sample_purities = np.empty(ensemble.sample_count, dtype=float)
    for index, (angle_error, detuning) in enumerate(
        zip(
            ensemble.rotation_angle_errors,
            ensemble.detunings_angular_frequency_rad_s,
        )
    ):
        propagated = propagate_xy8_finite(
            rho_initial,
            tau,
            blocks,
            pulse_duration,
            detuning_angular_frequency_rad_s=float(detuning),
            rotation_angle_error=float(angle_error),
            axis_z=axis_z,
            phase_error_rad=phase_error_rad,
        )
        sample_density_matrices[index] = propagated.density_matrix
        sample_bloch_vectors[index] = propagated.bloch_vector
        sample_purities[index] = density_matrix_purity(propagated.density_matrix)

    density_matrix = np.tensordot(
        ensemble.weights, sample_density_matrices, axes=(0, 0)
    )
    density_matrix = 0.5 * (density_matrix + density_matrix.conj().T)
    density_matrix /= np.trace(density_matrix)
    bloch_vector = bloch_vector_from_density_matrix(density_matrix)
    purity = density_matrix_purity(density_matrix)

    for array in [
        density_matrix,
        bloch_vector,
        sample_density_matrices,
        sample_bloch_vectors,
        sample_purities,
    ]:
        array.setflags(write=False)
    return XY8EnsemblePropagationResult(
        density_matrix=density_matrix,
        bloch_vector=bloch_vector,
        purity=purity,
        sample_density_matrices=sample_density_matrices,
        sample_bloch_vectors=sample_bloch_vectors,
        sample_purities=sample_purities,
        ensemble=ensemble,
        blocks=blocks,
        total_pi_pulses=8 * blocks,
        tau_s=tau,
        sequence_duration_s=16.0 * blocks * tau,
        pi_pulse_duration_s=pulse_duration,
        schedule=schedule,
    )


def propagate_xy8_sensing_ensemble(
    initial_state,
    tau_s: float,
    blocks: int,
    pi_pulse_duration_s: float,
    ensemble: QuasiStaticEnsemble,
    *,
    signal_frequency_hz: float,
    signal_field_t: float,
    signal_phase_rad: float,
    gyromagnetic_ratio_rad_s_t: float = ELECTRON_GYROMAGNETIC_RATIO_RAD_S_T,
    axis_z=0.0,
    phase_error_rad=0.0,
    pulse_substeps: int = 8,
) -> XY8EnsemblePropagationResult:
    """Propagate a weighted quasi-static ensemble in an AC sensing field."""
    if not isinstance(ensemble, QuasiStaticEnsemble):
        raise TypeError("ensemble must be a QuasiStaticEnsemble.")
    rho_initial = _validated_density_matrix(initial_state)
    blocks = _validated_blocks(blocks)
    tau = _validated_nonnegative_finite(tau_s, "tau_s")
    pulse_duration = _validated_nonnegative_finite(
        pi_pulse_duration_s, "pi_pulse_duration_s"
    )
    schedule = build_xy8_finite_schedule(tau, blocks, pulse_duration)
    sample_density_matrices = np.empty(
        (ensemble.sample_count, 2, 2), dtype=COMPLEX_DTYPE
    )
    sample_bloch_vectors = np.empty((ensemble.sample_count, 3), dtype=float)
    sample_purities = np.empty(ensemble.sample_count, dtype=float)
    for index, (angle_error, detuning) in enumerate(
        zip(
            ensemble.rotation_angle_errors,
            ensemble.detunings_angular_frequency_rad_s,
        )
    ):
        unitary = xy8_sensing_unitary(
            tau,
            blocks,
            pulse_duration,
            signal_frequency_hz=signal_frequency_hz,
            signal_field_t=signal_field_t,
            signal_phase_rad=signal_phase_rad,
            gyromagnetic_ratio_rad_s_t=gyromagnetic_ratio_rad_s_t,
            detuning_angular_frequency_rad_s=float(detuning),
            rotation_angle_error=float(angle_error),
            axis_z=axis_z,
            phase_error_rad=phase_error_rad,
            pulse_substeps=pulse_substeps,
        )
        rho_final = unitary @ rho_initial @ unitary.conj().T
        rho_final = 0.5 * (rho_final + rho_final.conj().T)
        rho_final /= np.trace(rho_final)
        sample_density_matrices[index] = rho_final
        sample_bloch_vectors[index] = bloch_vector_from_density_matrix(rho_final)
        sample_purities[index] = density_matrix_purity(rho_final)

    density_matrix = np.tensordot(
        ensemble.weights, sample_density_matrices, axes=(0, 0)
    )
    density_matrix = 0.5 * (density_matrix + density_matrix.conj().T)
    density_matrix /= np.trace(density_matrix)
    bloch_vector = bloch_vector_from_density_matrix(density_matrix)
    purity = density_matrix_purity(density_matrix)
    for array in [
        density_matrix,
        bloch_vector,
        sample_density_matrices,
        sample_bloch_vectors,
        sample_purities,
    ]:
        array.setflags(write=False)
    return XY8EnsemblePropagationResult(
        density_matrix=density_matrix,
        bloch_vector=bloch_vector,
        purity=purity,
        sample_density_matrices=sample_density_matrices,
        sample_bloch_vectors=sample_bloch_vectors,
        sample_purities=sample_purities,
        ensemble=ensemble,
        blocks=blocks,
        total_pi_pulses=8 * blocks,
        tau_s=tau,
        sequence_duration_s=16.0 * blocks * tau,
        pi_pulse_duration_s=pulse_duration,
        schedule=schedule,
    )


def equatorial_observable(phase_rad: float) -> np.ndarray:
    """Return the equatorial Pauli observable at the declared readout phase."""
    phase = float(phase_rad)
    if not np.isfinite(phase):
        raise ValueError("Readout phase must be finite.")
    return np.cos(phase) * SIGMA_X + np.sin(phase) * SIGMA_Y


def xy8_small_signal_response(
    tau_s: float,
    blocks: int,
    pi_pulse_duration_s: float,
    ensemble: QuasiStaticEnsemble | None = None,
    *,
    signal_frequency_hz: float,
    delta_field_t: float = 1e-12,
    gyromagnetic_ratio_rad_s_t: float = ELECTRON_GYROMAGNETIC_RATIO_RAD_S_T,
    signal_phase_rad: float | None = None,
    readout_phase_rad: float | None = None,
    initial_state=None,
    axis_z=0.0,
    phase_error_rad=0.0,
    pulse_substeps: int = 8,
) -> XY8SmallSignalResult:
    """Calculate the ensemble-averaged central magnetic derivative.

    When a phase is supplied it is used explicitly.  A missing readout phase
    selects the equatorial observable with the largest local slope.  A missing
    signal phase selects the AC quadrature with the largest local slope.  When
    both are missing, a two-quadrature singular-value decomposition performs
    the joint optimization without treating categorical control depth as a
    continuous variable.
    """
    delta_field = float(delta_field_t)
    gamma = float(gyromagnetic_ratio_rad_s_t)
    if not np.isfinite(delta_field) or delta_field <= 0.0:
        raise ValueError("delta_field_t must be a positive finite value.")
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gyromagnetic_ratio_rad_s_t must be greater than zero.")
    if signal_phase_rad is not None and not np.isfinite(float(signal_phase_rad)):
        raise ValueError("signal_phase_rad must be finite when supplied.")
    if readout_phase_rad is not None and not np.isfinite(float(readout_phase_rad)):
        raise ValueError("readout_phase_rad must be finite when supplied.")
    if ensemble is None:
        ensemble = make_quasi_static_ensemble([0.0], [0.0])
    if initial_state is None:
        initial_state = density_matrix_from_bloch([1.0, 0.0, 0.0])
    rho_initial = _validated_density_matrix(initial_state)

    common = {
        "signal_frequency_hz": signal_frequency_hz,
        "gyromagnetic_ratio_rad_s_t": gamma,
        "axis_z": axis_z,
        "phase_error_rad": phase_error_rad,
        "pulse_substeps": pulse_substeps,
    }

    def response_pair(signal_phase):
        plus = propagate_xy8_sensing_ensemble(
            rho_initial,
            tau_s,
            blocks,
            pi_pulse_duration_s,
            ensemble,
            signal_field_t=delta_field,
            signal_phase_rad=signal_phase,
            **common,
        )
        minus = propagate_xy8_sensing_ensemble(
            rho_initial,
            tau_s,
            blocks,
            pi_pulse_duration_s,
            ensemble,
            signal_field_t=-delta_field,
            signal_phase_rad=signal_phase,
            **common,
        )
        derivative = (plus.bloch_vector - minus.bloch_vector) / (2.0 * delta_field)
        return plus, minus, derivative

    if signal_phase_rad is None:
        _, _, derivative_cos = response_pair(0.0)
        _, _, derivative_quadrature = response_pair(np.pi / 2.0)
        if readout_phase_rad is None:
            response_matrix = np.column_stack(
                [derivative_cos[:2], derivative_quadrature[:2]]
            )
            left, _, right_adjoint = np.linalg.svd(response_matrix)
            signal_coefficients = right_adjoint[0]
            selected_signal_phase = float(
                np.arctan2(signal_coefficients[1], signal_coefficients[0])
            )
            selected_readout_phase = float(np.arctan2(left[1, 0], left[0, 0]))
            phase_selection = "optimized_signal_and_readout"
        else:
            selected_readout_phase = float(readout_phase_rad)
            readout_axis = np.array(
                [np.cos(selected_readout_phase), np.sin(selected_readout_phase)]
            )
            slopes = np.array(
                [
                    np.dot(readout_axis, derivative_cos[:2]),
                    np.dot(readout_axis, derivative_quadrature[:2]),
                ]
            )
            selected_signal_phase = float(np.arctan2(slopes[1], slopes[0]))
            phase_selection = "optimized_signal"
        plus, minus, bloch_derivative = response_pair(selected_signal_phase)
    else:
        selected_signal_phase = float(signal_phase_rad)
        plus, minus, bloch_derivative = response_pair(selected_signal_phase)
        if readout_phase_rad is None:
            selected_readout_phase = float(
                np.arctan2(bloch_derivative[1], bloch_derivative[0])
            )
            phase_selection = "optimized_readout"
        else:
            selected_readout_phase = float(readout_phase_rad)
            phase_selection = "explicit"

    observable = equatorial_observable(selected_readout_phase)
    expectation_plus = ensemble_average_observable(plus, observable)
    expectation_minus = ensemble_average_observable(minus, observable)
    derivative = (expectation_plus - expectation_minus) / (2.0 * delta_field)
    sequence_duration = 16.0 * _validated_blocks(blocks) * float(tau_s)
    normalized_signed_gain = derivative / (gamma * sequence_duration)
    normalized_gain = abs(normalized_signed_gain)
    bloch_derivative = np.asarray(bloch_derivative, dtype=float)
    bloch_derivative.setflags(write=False)
    return XY8SmallSignalResult(
        derivative_per_t=float(derivative),
        normalized_signed_gain=float(normalized_signed_gain),
        normalized_gain=float(normalized_gain),
        expectation_plus=float(expectation_plus),
        expectation_minus=float(expectation_minus),
        density_matrix_plus=plus.density_matrix,
        density_matrix_minus=minus.density_matrix,
        bloch_derivative_per_t=bloch_derivative,
        delta_field_t=delta_field,
        signal_frequency_hz=float(signal_frequency_hz),
        signal_phase_rad=selected_signal_phase,
        readout_phase_rad=selected_readout_phase,
        phase_selection=phase_selection,
        gyromagnetic_ratio_rad_s_t=gamma,
        sequence_duration_s=sequence_duration,
        blocks=_validated_blocks(blocks),
        total_pi_pulses=8 * _validated_blocks(blocks),
        tau_s=float(tau_s),
        pi_pulse_duration_s=float(pi_pulse_duration_s),
        pulse_substeps=int(pulse_substeps),
        ensemble=ensemble,
    )


def ensemble_average_observable(
    result: XY8EnsemblePropagationResult,
    observable,
) -> float:
    """Average signed sample expectations using the ensemble weights.

    This explicit sample-level calculation is intended for Stage 4's
    small-signal slopes.  It is equivalent to evaluating the observable on
    the weighted density matrix, while making the order of operations clear.
    """
    if not isinstance(result, XY8EnsemblePropagationResult):
        raise TypeError("result must be an XY8EnsemblePropagationResult.")
    sample_expectations = np.array(
        [expectation(rho, observable) for rho in result.sample_density_matrices]
    )
    return float(np.dot(result.ensemble.weights, sample_expectations))


def expectation(density_matrix, observable) -> float:
    """Return the real expectation value of a Hermitian 2x2 observable."""
    rho = _validated_density_matrix(density_matrix)
    operator = np.asarray(observable, dtype=COMPLEX_DTYPE)
    if operator.shape != (2, 2) or not np.allclose(
        operator, operator.conj().T, atol=1e-12, rtol=0.0
    ):
        raise ValueError("Observable must be a finite Hermitian 2x2 matrix.")
    return float(np.real(np.trace(rho @ operator)))
