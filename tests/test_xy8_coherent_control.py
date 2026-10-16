from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "code" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import xy8_coherent_control as coherent
import nvratemodel_api as nvpipe


def test_xy8_phase_order_and_pulse_count_are_exact():
    phases = coherent.xy8_pulse_phases(2)
    assert phases.shape == (16,)
    assert np.array_equal(phases[:8], coherent.XY8_BLOCK_PHASES_RAD)
    assert np.array_equal(phases[8:], coherent.XY8_BLOCK_PHASES_RAD)


@pytest.mark.parametrize("blocks", [1, 2, 7])
def test_ideal_xy8_preserves_an_arbitrary_state(blocks):
    initial_bloch = np.array([0.31, -0.47, 0.62])
    rho = coherent.density_matrix_from_bloch(initial_bloch)
    result = coherent.propagate_xy8(rho, blocks)

    assert result.total_pi_pulses == 8 * blocks
    assert np.allclose(result.density_matrix, rho, atol=2e-14, rtol=0.0)
    assert np.allclose(result.bloch_vector, initial_bloch, atol=2e-14, rtol=0.0)
    assert np.allclose(result.unitary.conj().T @ result.unitary, np.eye(2), atol=2e-14)


def test_x_and_y_pi_rotations_have_expected_action():
    plus_z = coherent.density_matrix_from_bloch([0.0, 0.0, 1.0])
    for phase in [0.0, np.pi / 2.0]:
        pulse = coherent.rotation_unitary(np.pi, phase)
        final = pulse @ plus_z @ pulse.conj().T
        assert coherent.bloch_vector_from_density_matrix(final) == pytest.approx(
            [0.0, 0.0, -1.0], abs=1e-14
        )


def test_coherent_errors_are_deterministic_and_not_scalar_loss():
    initial = coherent.density_matrix_from_bloch([1.0, 0.0, 0.0])
    phase_errors = np.zeros(8)
    phase_errors[3] = 0.13
    kwargs = {
        "rotation_angle_error": 0.08,
        "axis_z": 0.12,
        "phase_error_rad": phase_errors,
    }
    first = coherent.propagate_xy8(initial, 1, **kwargs)
    second = coherent.propagate_xy8(initial, 1, **kwargs)

    assert np.array_equal(first.density_matrix, second.density_matrix)
    assert not np.allclose(first.density_matrix, initial, atol=1e-6, rtol=0.0)
    assert np.trace(first.density_matrix) == pytest.approx(1.0)
    assert np.linalg.norm(first.bloch_vector) == pytest.approx(1.0)


def test_per_pulse_errors_must_match_eight_times_block_count():
    with pytest.raises(ValueError, match="exactly 16"):
        coherent.xy8_unitary(2, rotation_angle_error=np.zeros(8))


def test_invalid_axis_and_nonphysical_state_are_rejected():
    with pytest.raises(ValueError, match="axis_z"):
        coherent.rotation_unitary(np.pi, 0.0, axis_z=1.1)
    with pytest.raises(ValueError, match="norm greater than one"):
        coherent.density_matrix_from_bloch([1.1, 0.0, 0.0])


def test_state_vector_input_and_observable_expectation():
    plus_x = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    result = coherent.propagate_xy8(plus_x, 3)
    assert coherent.expectation(result.density_matrix, coherent.SIGMA_X) == pytest.approx(1.0)
    assert coherent.expectation(result.density_matrix, coherent.SIGMA_Y) == pytest.approx(0.0)
    assert coherent.expectation(result.density_matrix, coherent.SIGMA_Z) == pytest.approx(0.0)


def test_finite_schedule_preserves_pulse_centres_and_total_duration():
    tau = 1.25e-6
    blocks = 2
    pulse_duration = 40e-9
    schedule = coherent.build_xy8_finite_schedule(tau, blocks, pulse_duration)
    pulses = [segment for segment in schedule if segment.kind == "pulse"]
    free_segments = [segment for segment in schedule if segment.kind == "free"]

    assert len(pulses) == 8 * blocks
    assert len(free_segments) == 8 * blocks + 1
    assert sum(segment.duration_s for segment in schedule) == pytest.approx(
        16.0 * blocks * tau, abs=1e-18
    )
    expected_centres = np.asarray(
        [(2 * index - 1) * tau for index in range(1, 8 * blocks + 1)]
    )
    actual_centres = np.asarray(
        [segment.start_s + segment.duration_s / 2.0 for segment in pulses]
    )
    assert actual_centres == pytest.approx(expected_centres, abs=1e-18)
    assert all(segment.duration_s >= 0.0 for segment in schedule)


def test_zero_width_zero_detuning_limit_reproduces_stage_one():
    kwargs = {
        "rotation_angle_error": 0.04,
        "axis_z": 0.08,
        "phase_error_rad": np.linspace(-0.01, 0.01, 16),
    }
    instantaneous = coherent.xy8_unitary(2, **kwargs)
    finite_limit = coherent.xy8_finite_unitary(1.25e-6, 2, 0.0, **kwargs)
    assert np.allclose(finite_limit, instantaneous, atol=2e-13, rtol=0.0)


def test_finite_resonant_pulses_reproduce_instantaneous_xy8():
    instantaneous = coherent.xy8_unitary(3)
    finite = coherent.xy8_finite_unitary(
        1.25e-6,
        3,
        35e-9,
        detuning_angular_frequency_rad_s=0.0,
    )
    assert np.allclose(finite, instantaneous, atol=3e-13, rtol=0.0)


def test_single_finite_driven_pulse_matches_effective_axis_rotation():
    pulse_duration = 50e-9
    detuning = 2.0 * np.pi * 1.2e6
    actual = coherent.driven_pi_pulse_unitary(
        0.0,
        pulse_duration,
        detuning_angular_frequency_rad_s=detuning,
    )
    rabi = np.pi / pulse_duration
    magnitude = np.hypot(rabi, detuning)
    expected = coherent.rotation_unitary(
        magnitude * pulse_duration,
        0.0,
        axis_z=detuning / magnitude,
    )
    assert np.allclose(actual, expected, atol=2e-14, rtol=0.0)


def test_ideal_instantaneous_xy8_refocuses_static_detuning():
    detuning = 2.0 * np.pi * 0.8e6
    actual = coherent.xy8_finite_unitary(
        1.25e-6,
        2,
        0.0,
        detuning_angular_frequency_rad_s=detuning,
    )
    ideal = coherent.xy8_unitary(2)
    assert np.allclose(actual, ideal, atol=3e-13, rtol=0.0)


def test_detuning_during_finite_pulses_produces_coherent_residual():
    initial = coherent.density_matrix_from_bloch([1.0, 0.0, 0.0])
    resonant = coherent.propagate_xy8_finite(initial, 1.25e-6, 4, 40e-9)
    detuned = coherent.propagate_xy8_finite(
        initial,
        1.25e-6,
        4,
        40e-9,
        detuning_angular_frequency_rad_s=2.0 * np.pi * 2.2e6,
    )

    assert np.allclose(resonant.density_matrix, initial, atol=4e-13, rtol=0.0)
    assert not np.allclose(detuned.density_matrix, initial, atol=1e-6, rtol=0.0)
    assert np.trace(detuned.density_matrix) == pytest.approx(1.0)
    assert np.linalg.norm(detuned.bloch_vector) == pytest.approx(1.0)


def test_finite_propagation_reports_timing_metadata():
    result = coherent.propagate_xy8_finite(
        [1.0, 0.0],
        1.25e-6,
        5,
        30e-9,
        detuning_angular_frequency_rad_s=2.0 * np.pi * 0.25e6,
    )
    assert result.total_pi_pulses == 40
    assert result.sequence_duration_s == pytest.approx(100e-6)
    assert result.pi_pulse_duration_s == pytest.approx(30e-9)
    assert result.detuning_angular_frequency_rad_s == pytest.approx(2.0 * np.pi * 0.25e6)


@pytest.mark.parametrize("pulse_duration", [2.5e-6, 3.0e-6])
def test_overlapping_finite_pulses_are_rejected(pulse_duration):
    with pytest.raises(ValueError, match="smaller than 2\\*tau_s"):
        coherent.build_xy8_finite_schedule(1.25e-6, 1, pulse_duration)


def test_explicit_quasi_static_ensemble_normalizes_and_freezes_weights():
    ensemble = coherent.make_quasi_static_ensemble(
        [-0.02, 0.01],
        [-2.0e6, 3.0e6],
        weights=[1.0, 3.0],
    )
    assert ensemble.sample_count == 2
    assert ensemble.weights == pytest.approx([0.25, 0.75])
    assert np.sum(ensemble.weights) == pytest.approx(1.0)
    assert np.array_equal(ensemble.hyperfine_line_indices, [-1, -1])
    assert not ensemble.weights.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        ensemble.weights[0] = 1.0


@pytest.mark.parametrize(
    "angle_errors, detunings, weights, message",
    [
        ([0.0], [0.0, 1.0], None, "same length"),
        ([0.0], [0.0], [-1.0], "non-negative"),
        ([0.0], [0.0], [0.0], "positive"),
        ([np.nan], [0.0], [1.0], "finite"),
    ],
)
def test_invalid_quasi_static_ensembles_are_rejected(
    angle_errors, detunings, weights, message
):
    with pytest.raises(ValueError, match=message):
        coherent.make_quasi_static_ensemble(
            angle_errors,
            detunings,
            weights=weights,
        )


def test_gaussian_quadrature_has_declared_moments_and_is_deterministic():
    kwargs = {
        "rotation_angle_error_mean": 0.01,
        "rotation_angle_error_std": 0.03,
        "detuning_mean_angular_frequency_rad_s": 2.0e5,
        "detuning_std_angular_frequency_rad_s": 4.0e5,
        "quadrature_order": 5,
    }
    first = coherent.gaussian_quadrature_ensemble(**kwargs)
    second = coherent.gaussian_quadrature_ensemble(**kwargs)
    assert np.array_equal(first.rotation_angle_errors, second.rotation_angle_errors)
    assert np.array_equal(
        first.detunings_angular_frequency_rad_s,
        second.detunings_angular_frequency_rad_s,
    )
    assert np.dot(first.weights, first.rotation_angle_errors) == pytest.approx(0.01)
    assert np.dot(
        first.weights,
        (first.rotation_angle_errors - 0.01) ** 2,
    ) == pytest.approx(0.03**2)
    assert np.dot(
        first.weights,
        first.detunings_angular_frequency_rad_s,
    ) == pytest.approx(2.0e5)
    assert np.dot(
        first.weights,
        (first.detunings_angular_frequency_rad_s - 2.0e5) ** 2,
    ) == pytest.approx((4.0e5) ** 2)


def test_hyperfine_populations_are_preserved_by_product_quadrature():
    offsets = 2.0 * np.pi * np.array([-2.16e6, 0.0, 2.16e6])
    populations = np.array([0.2, 0.5, 0.3])
    ensemble = coherent.gaussian_quadrature_ensemble(
        detuning_std_angular_frequency_rad_s=2.0 * np.pi * 0.1e6,
        quadrature_order=3,
        hyperfine_detunings_rad_s=offsets,
        hyperfine_weights=populations,
    )
    assert ensemble.sample_count == 9
    for line_index, (offset, population) in enumerate(zip(offsets, populations)):
        selected = ensemble.hyperfine_line_indices == line_index
        assert np.sum(ensemble.weights[selected]) == pytest.approx(population)
        conditional_mean = np.dot(
            ensemble.weights[selected],
            ensemble.detunings_angular_frequency_rad_s[selected],
        ) / population
        assert conditional_mean == pytest.approx(offset)


def test_delta_ensemble_reproduces_stage_two_exactly():
    initial = coherent.density_matrix_from_bloch([0.3, -0.4, 0.5])
    angle_error = 0.025
    detuning = 2.0 * np.pi * 0.7e6
    deterministic = coherent.propagate_xy8_finite(
        initial,
        1.25e-6,
        3,
        40e-9,
        rotation_angle_error=angle_error,
        detuning_angular_frequency_rad_s=detuning,
    )
    ensemble = coherent.make_quasi_static_ensemble([angle_error], [detuning])
    averaged = coherent.propagate_xy8_ensemble(
        initial,
        1.25e-6,
        3,
        40e-9,
        ensemble,
    )
    assert np.array_equal(averaged.density_matrix, deterministic.density_matrix)
    assert np.array_equal(averaged.bloch_vector, deterministic.bloch_vector)
    assert averaged.total_pi_pulses == 24
    assert averaged.purity == pytest.approx(
        coherent.density_matrix_purity(deterministic.density_matrix)
    )


def test_ensemble_state_is_the_weighted_average_of_quasi_static_sequences():
    initial = coherent.density_matrix_from_bloch([1.0, 0.0, 0.0])
    ensemble = coherent.make_quasi_static_ensemble(
        [-0.08, 0.03, 0.1],
        [-1.0e7, 2.0e6, 1.4e7],
        weights=[0.2, 0.5, 0.3],
    )
    result = coherent.propagate_xy8_ensemble(
        initial, 1.25e-6, 2, 40e-9, ensemble
    )
    expected = np.tensordot(
        ensemble.weights, result.sample_density_matrices, axes=(0, 0)
    )
    assert result.density_matrix == pytest.approx(expected, abs=2e-15)
    assert np.trace(result.density_matrix) == pytest.approx(1.0)
    assert result.purity <= np.dot(ensemble.weights, result.sample_purities) + 1e-14


def test_signed_observables_are_averaged_without_taking_magnitudes():
    initial = coherent.density_matrix_from_bloch([1.0, 0.0, 0.0])
    ensemble = coherent.make_quasi_static_ensemble(
        [0.0, 0.0], [-2.0e7, 2.0e7]
    )
    result = coherent.propagate_xy8_ensemble(
        initial, 1.25e-6, 1, 40e-9, ensemble
    )
    signed = coherent.ensemble_average_observable(result, coherent.SIGMA_Y)
    sample_values = result.sample_bloch_vectors[:, 1]
    assert signed == pytest.approx(
        coherent.expectation(result.density_matrix, coherent.SIGMA_Y)
    )
    assert signed == pytest.approx(np.dot(ensemble.weights, sample_values))
    assert abs(signed) < np.dot(ensemble.weights, np.abs(sample_values))


def test_increasing_symmetric_amplitude_spread_reduces_ensemble_purity():
    initial = coherent.density_matrix_from_bloch([1.0, 0.0, 0.0])
    purities = []
    for spread in [0.0, 0.03, 0.06, 0.1]:
        ensemble = coherent.gaussian_quadrature_ensemble(
            rotation_angle_error_std=spread,
            quadrature_order=7,
        )
        result = coherent.propagate_xy8_ensemble(
            initial, 1.25e-6, 4, 40e-9, ensemble
        )
        purities.append(result.purity)
    assert np.all(np.diff(purities) <= 1e-12)
    assert purities[-1] < purities[0] - 0.02


def test_seeded_monte_carlo_ensembles_are_reproducible():
    kwargs = {
        "sample_count": 40,
        "rotation_angle_error_std": 0.03,
        "detuning_std_angular_frequency_rad_s": 5.0e5,
        "hyperfine_detunings_rad_s": [-1.0e6, 1.0e6],
        "hyperfine_weights": [0.25, 0.75],
    }
    first = coherent.monte_carlo_quasi_static_ensemble(seed=47, **kwargs)
    repeated = coherent.monte_carlo_quasi_static_ensemble(seed=47, **kwargs)
    different = coherent.monte_carlo_quasi_static_ensemble(seed=48, **kwargs)
    assert np.array_equal(first.rotation_angle_errors, repeated.rotation_angle_errors)
    assert np.array_equal(
        first.detunings_angular_frequency_rad_s,
        repeated.detunings_angular_frequency_rad_s,
    )
    assert np.array_equal(first.hyperfine_line_indices, repeated.hyperfine_line_indices)
    assert not np.array_equal(
        first.rotation_angle_errors, different.rotation_angle_errors
    )
    assert np.sum(first.weights) == pytest.approx(1.0)


def test_ac_signal_integral_matches_closed_form_limits():
    assert coherent.ac_signal_integral(2.0, 0.4, 0.0, 0.0) == pytest.approx(0.4)
    assert coherent.ac_signal_integral(0.0, 2.5e-6, 200e3, 0.0) == pytest.approx(
        0.0, abs=1e-20
    )
    start = 0.37e-6
    duration = 0.81e-6
    frequency = 310e3
    phase = 0.43
    grid = np.linspace(start, start + duration, 20001)
    numerical = np.trapezoid(np.cos(2.0 * np.pi * frequency * grid + phase), grid)
    assert coherent.ac_signal_integral(
        start, duration, frequency, phase
    ) == pytest.approx(numerical, rel=2e-9)


def test_zero_sensing_field_reproduces_stage_two_finite_unitary():
    kwargs = {
        "detuning_angular_frequency_rad_s": 2.0 * np.pi * 0.6e6,
        "rotation_angle_error": 0.03,
        "axis_z": 0.04,
        "phase_error_rad": 0.01,
    }
    stage_two = coherent.xy8_finite_unitary(1.25e-6, 2, 40e-9, **kwargs)
    sensing = coherent.xy8_sensing_unitary(
        1.25e-6,
        2,
        40e-9,
        signal_frequency_hz=200e3,
        signal_field_t=0.0,
        signal_phase_rad=0.2,
        pulse_substeps=8,
        **kwargs,
    )
    assert np.allclose(sensing, stage_two, atol=3e-13, rtol=0.0)


@pytest.mark.parametrize("frequency", [170e3, 200e3, 230e3])
def test_ideal_small_signal_gain_reproduces_bounded_filter_gain(frequency):
    result = coherent.xy8_small_signal_response(
        1.25e-6,
        2,
        0.0,
        signal_frequency_hz=frequency,
        delta_field_t=1e-12,
    )
    expected = nvpipe.xy8_filter_gain(frequency, 1.25e-6, 2)
    assert result.normalized_gain == pytest.approx(expected, rel=4e-10, abs=2e-13)
    assert 0.0 <= result.normalized_gain <= 1.0


def test_primary_resonance_gain_approaches_two_over_pi():
    result = coherent.xy8_small_signal_response(
        1.25e-6,
        4,
        0.0,
        signal_frequency_hz=200e3,
        delta_field_t=1e-12,
    )
    assert result.normalized_gain == pytest.approx(2.0 / np.pi, rel=2e-9)
    assert result.total_pi_pulses == 32
    assert result.sequence_duration_s == pytest.approx(80e-6)
    assert result.phase_selection == "optimized_signal_and_readout"


def test_small_signal_central_derivative_converges_with_delta_field():
    gains = []
    for delta_field in [1e-9, 1e-10, 1e-11, 1e-12]:
        result = coherent.xy8_small_signal_response(
            1.25e-6,
            2,
            0.0,
            signal_frequency_hz=200e3,
            delta_field_t=delta_field,
        )
        gains.append(result.normalized_gain)
    exact = 2.0 / np.pi
    errors = np.abs(np.asarray(gains) - exact)
    assert np.all(np.diff(errors) < 0.0)
    assert errors[-1] < 3e-12


def test_explicit_and_partly_optimized_phase_modes_are_reported():
    explicit = coherent.xy8_small_signal_response(
        1.25e-6,
        1,
        0.0,
        signal_frequency_hz=200e3,
        signal_phase_rad=0.0,
        readout_phase_rad=np.pi / 2.0,
    )
    readout_optimized = coherent.xy8_small_signal_response(
        1.25e-6,
        1,
        0.0,
        signal_frequency_hz=200e3,
        signal_phase_rad=0.0,
    )
    signal_optimized = coherent.xy8_small_signal_response(
        1.25e-6,
        1,
        0.0,
        signal_frequency_hz=200e3,
        readout_phase_rad=np.pi / 2.0,
    )
    assert explicit.phase_selection == "explicit"
    assert readout_optimized.phase_selection == "optimized_readout"
    assert signal_optimized.phase_selection == "optimized_signal"
    for result in [explicit, readout_optimized, signal_optimized]:
        assert result.normalized_gain == pytest.approx(2.0 / np.pi, rel=2e-9)


def test_quasi_static_ensemble_reduces_small_signal_gain():
    ideal = coherent.xy8_small_signal_response(
        1.25e-6,
        4,
        40e-9,
        signal_frequency_hz=200e3,
    )
    ensemble = coherent.gaussian_quadrature_ensemble(
        rotation_angle_error_std=0.1,
        detuning_std_angular_frequency_rad_s=2.0 * np.pi * 1.0e6,
        quadrature_order=5,
    )
    broadened = coherent.xy8_small_signal_response(
        1.25e-6,
        4,
        40e-9,
        ensemble,
        signal_frequency_hz=200e3,
    )
    assert broadened.normalized_gain < ideal.normalized_gain
    assert broadened.expectation_plus != pytest.approx(broadened.expectation_minus)
    assert broadened.ensemble is ensemble


def test_finite_pulse_signal_integration_converges_with_substeps():
    gains = []
    for substeps in [2, 4, 8, 16, 32]:
        result = coherent.xy8_small_signal_response(
            1.25e-6,
            1,
            500e-9,
            signal_frequency_hz=200e3,
            delta_field_t=1e-11,
            pulse_substeps=substeps,
        )
        gains.append(result.normalized_gain)
    successive_changes = np.abs(np.diff(gains))
    assert np.all(np.diff(successive_changes) < 0.0)
    assert successive_changes[-1] < 1e-4
