"""Stage-4 environment-conditioned XY8 discovery surfaces.

The frozen canonical scalar score remains the reference resource model.  A
temperature-dependent coherence ratio and a deterministic finite-pulse
detuning-response ratio condition that score without refitting any parameter:

    eta_env = eta_canonical * C_ref/C_env * G(0)/G(detuning).

The coherent response is evaluated in batches for all environmental detunings.
This is algebraically the same two-quadrature small-signal derivative used by
``xy8_coherent_control.xy8_small_signal_response`` for a deterministic ensemble,
but it advances every detuning together and records all requested block depths
in one pass.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import nvratemodel_api as model
import xy8_environment_coherence as coherence
import xy8_environment_conditioning as conditioning
import xy8_environment_scan_config as scan_config
import xy8_nv_environment_literature as literature


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANONICAL_SURFACE = ROOT / "data" / "processed" / "xy8_qre_nv_results.csv"
XY8_PHASES_RAD = np.array(
    [0.0, np.pi / 2.0, 0.0, np.pi / 2.0, np.pi / 2.0, 0.0, np.pi / 2.0, 0.0]
)
ELECTRON_GYROMAGNETIC_RATIO_RAD_S_T = 2.0 * np.pi * 28.02495164e9


@dataclass(frozen=True)
class CanonicalSurface:
    taus_s: np.ndarray
    depths_N: np.ndarray
    eta: np.ndarray


@dataclass(frozen=True)
class EnvironmentPoint:
    condition_id: str
    scan_family: str
    strain_path: str
    temperature_K: float
    amplitude_microstrain: float
    strain_tensor: np.ndarray
    transition: conditioning.EnvironmentTransition


def load_canonical_surface(path: str | Path = DEFAULT_CANONICAL_SURFACE) -> CanonicalSurface:
    """Load the rectangular canonical score block after its metadata rows."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    try:
        header_index = lines.index("tau_s,N,eta_proxy")
    except ValueError as error:
        raise ValueError(f"Canonical surface header not found in {path}.") from error
    records = list(csv.DictReader(lines[header_index:]))
    if not records:
        raise ValueError("Canonical surface contains no score rows.")
    taus = np.array(sorted({float(row["tau_s"]) for row in records}), dtype=float)
    depths = np.array(sorted({int(row["N"]) for row in records}), dtype=int)
    if len(records) != taus.size * depths.size:
        raise ValueError("Canonical scores do not form a complete rectangular grid.")
    tau_index = {value: index for index, value in enumerate(taus)}
    depth_index = {value: index for index, value in enumerate(depths)}
    eta = np.full((taus.size, depths.size), np.nan, dtype=float)
    for row in records:
        tau = float(row["tau_s"])
        depth = int(row["N"])
        if np.isfinite(eta[tau_index[tau], depth_index[depth]]):
            raise ValueError(f"Duplicate canonical point tau={tau}, N={depth}.")
        eta[tau_index[tau], depth_index[depth]] = float(row["eta_proxy"])
    if not np.all(np.isfinite(eta)) or np.any(eta <= 0.0):
        raise ValueError("Canonical surface must contain positive finite scores.")
    return CanonicalSurface(taus_s=taus, depths_N=depths, eta=eta)


def _apply_constant_hamiltonian_batch(
    states: np.ndarray,
    omega_x: float,
    omega_y: float,
    omega_z: np.ndarray,
    duration_s: float,
) -> np.ndarray:
    """Apply exp[-i (omega.sigma) dt/2] to batched pure states."""
    norm = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
    half_angle = 0.5 * norm * duration_s
    cosine = np.cos(half_angle)
    sine_over_norm = np.empty_like(norm)
    nonzero = norm > 0.0
    sine_over_norm[nonzero] = np.sin(half_angle[nonzero]) / norm[nonzero]
    sine_over_norm[~nonzero] = 0.5 * duration_s
    factor = -1j * sine_over_norm
    first = states[:, 0].copy()
    second = states[:, 1].copy()
    states[:, 0] = (cosine + factor * omega_z) * first + factor * (
        omega_x - 1j * omega_y
    ) * second
    states[:, 1] = factor * (omega_x + 1j * omega_y) * first + (
        cosine - factor * omega_z
    ) * second
    return states


def _apply_free_batch(
    states: np.ndarray,
    detunings_rad_s: np.ndarray,
    signal_fields_T: np.ndarray,
    signal_phases_rad: np.ndarray,
    *,
    start_s: float,
    duration_s: float,
    signal_frequency_hz: float,
    gyromagnetic_ratio_rad_s_t: float,
) -> np.ndarray:
    if duration_s <= 0.0:
        return states
    omega = 2.0 * np.pi * signal_frequency_hz
    stop = start_s + duration_s
    integral = (
        np.sin(omega * stop + signal_phases_rad)
        - np.sin(omega * start_s + signal_phases_rad)
    ) / omega
    angle = detunings_rad_s * duration_s + gyromagnetic_ratio_rad_s_t * signal_fields_T * integral
    states[:, 0] *= np.exp(-0.5j * angle)
    states[:, 1] *= np.exp(0.5j * angle)
    return states


def batched_deterministic_xy8_gains(
    taus_s: Iterable[float],
    depths_N: Iterable[int],
    detunings_Hz: Iterable[float],
    *,
    signal_frequency_hz: float,
    pi_pulse_duration_s: float,
    pulse_substeps: int = 8,
    delta_field_T: float = 1e-12,
    gyromagnetic_ratio_rad_s_t: float = ELECTRON_GYROMAGNETIC_RATIO_RAD_S_T,
) -> np.ndarray:
    """Return optimized coherent gains with shape (detuning, tau, depth)."""
    taus = np.asarray(list(taus_s), dtype=float)
    depths = np.asarray(list(depths_N), dtype=int)
    detunings = np.asarray(list(detunings_Hz), dtype=float)
    if taus.ndim != 1 or taus.size == 0 or np.any(~np.isfinite(taus)) or np.any(taus <= 0.0):
        raise ValueError("taus_s must contain positive finite values.")
    if depths.ndim != 1 or depths.size == 0 or np.any(depths < 1) or np.any(np.diff(depths) <= 0):
        raise ValueError("depths_N must be strictly increasing positive integers.")
    if detunings.ndim != 1 or detunings.size == 0 or np.any(~np.isfinite(detunings)):
        raise ValueError("detunings_Hz must contain finite values.")
    if isinstance(pulse_substeps, bool) or int(pulse_substeps) < 1:
        raise ValueError("pulse_substeps must be a positive integer.")
    pulse_substeps = int(pulse_substeps)
    pulse_duration = float(pi_pulse_duration_s)
    if not math.isfinite(pulse_duration) or pulse_duration < 0.0:
        raise ValueError("pi_pulse_duration_s must be non-negative and finite.")
    if np.any(pulse_duration >= 2.0 * taus):
        raise ValueError("pi_pulse_duration_s must be smaller than 2*tau for every point.")
    field_delta = float(delta_field_T)
    if not math.isfinite(field_delta) or field_delta <= 0.0:
        raise ValueError("delta_field_T must be positive and finite.")
    frequency = float(signal_frequency_hz)
    gamma = float(gyromagnetic_ratio_rad_s_t)
    if not math.isfinite(frequency) or frequency <= 0.0 or not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("Signal frequency and gyromagnetic ratio must be positive and finite.")

    depth_lookup = {int(depth): index for index, depth in enumerate(depths)}
    detuning_scenarios = np.repeat(2.0 * np.pi * detunings, 4)
    signal_fields = np.tile(np.array([field_delta, -field_delta, field_delta, -field_delta]), detunings.size)
    signal_phases = np.tile(np.array([0.0, 0.0, np.pi / 2.0, np.pi / 2.0]), detunings.size)
    gains = np.empty((detunings.size, taus.size, depths.size), dtype=float)
    root_two = math.sqrt(2.0)
    rabi = np.pi / pulse_duration if pulse_duration > 0.0 else 0.0

    for tau_index, tau in enumerate(taus):
        states = np.full((detuning_scenarios.size, 2), 1.0 / root_two, dtype=complex)
        current_time = 0.0
        for block in range(1, int(depths[-1]) + 1):
            for within_block, phase in enumerate(XY8_PHASES_RAD):
                pulse_number = 8 * (block - 1) + within_block
                centre = (2.0 * pulse_number + 1.0) * tau
                pulse_start = centre - 0.5 * pulse_duration
                states = _apply_free_batch(
                    states,
                    detuning_scenarios,
                    signal_fields,
                    signal_phases,
                    start_s=current_time,
                    duration_s=pulse_start - current_time,
                    signal_frequency_hz=frequency,
                    gyromagnetic_ratio_rad_s_t=gamma,
                )
                if pulse_duration == 0.0:
                    omega_z = detuning_scenarios
                    states = _apply_constant_hamiltonian_batch(
                        states, np.pi * math.cos(phase), np.pi * math.sin(phase), omega_z * 0.0, 1.0
                    )
                else:
                    substep_duration = pulse_duration / pulse_substeps
                    for substep in range(pulse_substeps):
                        midpoint = pulse_start + (substep + 0.5) * substep_duration
                        signal_z = gamma * signal_fields * np.cos(
                            2.0 * np.pi * frequency * midpoint + signal_phases
                        )
                        states = _apply_constant_hamiltonian_batch(
                            states,
                            rabi * math.cos(phase),
                            rabi * math.sin(phase),
                            detuning_scenarios + signal_z,
                            substep_duration,
                        )
                current_time = pulse_start + pulse_duration

            block_end = 16.0 * block * tau
            states = _apply_free_batch(
                states,
                detuning_scenarios,
                signal_fields,
                signal_phases,
                start_s=current_time,
                duration_s=block_end - current_time,
                signal_frequency_hz=frequency,
                gyromagnetic_ratio_rad_s_t=gamma,
            )
            current_time = block_end
            if block not in depth_lookup:
                continue

            reshaped = states.reshape(detunings.size, 4, 2)
            coherence_xy = np.conj(reshaped[:, :, 0]) * reshaped[:, :, 1]
            bloch_x = 2.0 * coherence_xy.real
            bloch_y = 2.0 * coherence_xy.imag
            derivative_x_cos = (bloch_x[:, 0] - bloch_x[:, 1]) / (2.0 * field_delta)
            derivative_y_cos = (bloch_y[:, 0] - bloch_y[:, 1]) / (2.0 * field_delta)
            derivative_x_sin = (bloch_x[:, 2] - bloch_x[:, 3]) / (2.0 * field_delta)
            derivative_y_sin = (bloch_y[:, 2] - bloch_y[:, 3]) / (2.0 * field_delta)
            trace_gram = (
                derivative_x_cos**2
                + derivative_x_sin**2
                + derivative_y_cos**2
                + derivative_y_sin**2
            )
            determinant = (
                derivative_x_cos * derivative_y_sin
                - derivative_x_sin * derivative_y_cos
            )
            discriminant = np.maximum(trace_gram**2 - 4.0 * determinant**2, 0.0)
            largest_singular = np.sqrt(0.5 * (trace_gram + np.sqrt(discriminant)))
            duration = 16.0 * block * tau
            gains[:, tau_index, depth_lookup[block]] = largest_singular / (gamma * duration)

    if not np.all(np.isfinite(gains)) or np.any(gains < 0.0):
        raise RuntimeError("Batched coherent response produced invalid gains.")
    return gains


def build_operational_environment_points(
    config: dict[str, Any] | None = None,
) -> list[EnvironmentPoint]:
    """Construct the frozen operational T-by-strain paths with continuous tracking."""
    if config is None:
        config = scan_config.load_config()
    temperatures = config["temperature_axes"]["operational"]["values_K"]
    amplitudes = config["strain_axes"]["amplitudes_microstrain"]
    points: list[EnvironmentPoint] = []

    zero_conditions = [
        conditioning.EnvironmentCondition(
            temperature,
            np.zeros((3, 3)),
            label=f"zero_strain_T{temperature:g}K",
        )
        for temperature in temperatures
    ]
    zero_results = conditioning.track_environment_path(
        zero_conditions, config=config, anchor_index=temperatures.index(300.0)
    )
    for temperature, result in zip(temperatures, zero_results):
        points.append(
            EnvironmentPoint(
                condition_id=f"zero_strain_T{temperature:g}K",
                scan_family="operational_temperature",
                strain_path="zero_strain",
                temperature_K=float(temperature),
                amplitude_microstrain=0.0,
                strain_tensor=np.zeros((3, 3)),
                transition=result,
            )
        )

    for path in ("axial_zz", "transverse_differential", "shear_xz"):
        for temperature in temperatures:
            conditions = [
                conditioning.EnvironmentCondition(
                    temperature,
                    conditioning.strain_tensor_from_path(path, amplitude, config=config),
                    label=f"{path}_T{temperature:g}K_A{amplitude:+g}us",
                )
                for amplitude in amplitudes
            ]
            results = conditioning.track_environment_path(
                conditions, config=config, anchor_index=amplitudes.index(0.0)
            )
            for amplitude, result in zip(amplitudes, results):
                points.append(
                    EnvironmentPoint(
                        condition_id=f"{path}_T{temperature:g}K_A{amplitude:+g}us",
                        scan_family="operational_temperature_strain_map",
                        strain_path=path,
                        temperature_K=float(temperature),
                        amplitude_microstrain=float(amplitude),
                        strain_tensor=result.condition.strain_tensor,
                        transition=result,
                    )
                )
    return points


def coherence_conditioned_surface(
    canonical: CanonicalSurface,
    temperature_K: float,
    *,
    variant: str,
    nominal_t2_params: dict[str, Any],
) -> tuple[np.ndarray, coherence.AnchoredCoherence]:
    """Condition the canonical surface only by the anchored coherence ratio."""
    conditioned_params, result = coherence.temperature_conditioned_t2_params(
        nominal_t2_params, temperature_K, variant=variant
    )
    ratio = np.empty_like(canonical.eta)
    for tau_index, tau in enumerate(canonical.taus_s):
        for depth_index, depth in enumerate(canonical.depths_N):
            sequence_duration = model.sequence_duration(tau, depth)
            reference_t2 = model.t2_xy8_empirical(depth, tau, **nominal_t2_params)
            environment_t2 = model.t2_xy8_empirical(depth, tau, **conditioned_params)
            p = float(nominal_t2_params.get("p", 2.0))
            reference_factor = model.coherence_factor(sequence_duration, reference_t2, p=p)
            environment_factor = model.coherence_factor(sequence_duration, environment_t2, p=p)
            ratio[tau_index, depth_index] = reference_factor / max(environment_factor, 1e-300)
    return canonical.eta * ratio, result


def summarize_surface(
    eta: np.ndarray,
    canonical: CanonicalSurface,
    *,
    relative_limit: float,
) -> dict[str, Any]:
    scores = np.asarray(eta, dtype=float)
    if scores.shape != canonical.eta.shape or not np.all(np.isfinite(scores)) or np.any(scores <= 0.0):
        raise ValueError("Conditioned surface must match the canonical positive finite grid.")
    best_flat = int(np.argmin(scores))
    best_tau_index, best_depth_index = np.unravel_index(best_flat, scores.shape)
    best = float(scores[best_tau_index, best_depth_index])
    mask = scores <= float(relative_limit) * best
    tau_indices, depth_indices = np.nonzero(mask)
    accepted_taus = canonical.taus_s[tau_indices]
    accepted_depths = canonical.depths_N[depth_indices]

    def depth_penalty(depth: int) -> float:
        index = int(np.where(canonical.depths_N == depth)[0][0])
        return float(np.min(scores[:, index]) / best)

    return {
        "best_tau_s": float(canonical.taus_s[best_tau_index]),
        "best_N": int(canonical.depths_N[best_depth_index]),
        "best_eta": best,
        "best_sequence_duration_s": float(
            model.sequence_duration(
                canonical.taus_s[best_tau_index], canonical.depths_N[best_depth_index]
            )
        ),
        "best_pi_pulses": int(8 * canonical.depths_N[best_depth_index]),
        "window_point_count": int(np.count_nonzero(mask)),
        "window_tau_min_s": float(np.min(accepted_taus)),
        "window_tau_max_s": float(np.max(accepted_taus)),
        "window_N_min": int(np.min(accepted_depths)),
        "window_N_max": int(np.max(accepted_depths)),
        "window_touches_tau_boundary": bool(
            np.any(tau_indices == 0) or np.any(tau_indices == scores.shape[0] - 1)
        ),
        "window_touches_N_boundary": bool(
            np.any(depth_indices == 0) or np.any(depth_indices == scores.shape[1] - 1)
        ),
        "best_touches_boundary": bool(
            best_tau_index in (0, scores.shape[0] - 1)
            or best_depth_index in (0, scores.shape[1] - 1)
        ),
        "boundary_status": (
            "unresolved_best_on_boundary"
            if (
                best_tau_index in (0, scores.shape[0] - 1)
                or best_depth_index in (0, scores.shape[1] - 1)
            )
            else (
                "window_touches_boundary"
                if (
                    np.any(tau_indices == 0)
                    or np.any(tau_indices == scores.shape[0] - 1)
                    or np.any(depth_indices == 0)
                    or np.any(depth_indices == scores.shape[1] - 1)
                )
                else "resolved_interior_on_discovery_grid"
            )
        ),
        "N16_relative_penalty": depth_penalty(16),
        "N20_relative_penalty": depth_penalty(20),
    }
