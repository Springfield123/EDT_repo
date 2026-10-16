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

import nvratemodel_api as pipe
import xy8_coherent_control as coherent


FREQUENCY_HZ = 200e3
TAU_S = 1.25e-6
BLOCKS = 4
READOUT_RATE_CPS = 2.0e6
CONTRAST = 0.12
NEFF = 1.0
T2_PARAMS = {"T2_1": 500e-6, "alpha": 0.4, "beta": 0.0, "tau0": 1e-6, "p": 2.0}
OVERHEADS = {"t_init": 5e-6, "t_read": 300e-9, "t_dead": 0.0}
IDEAL_COHERENT_CONFIG = {
    "pi_pulse_duration_s": 0.0,
    "signal_phase_rad": 0.0,
    "readout_phase_rad": np.pi / 2.0,
    "delta_field_t": 1e-12,
}


def eta(**updates):
    arguments = {
        "f": FREQUENCY_HZ,
        "tau": TAU_S,
        "N": BLOCKS,
        "Rph_1": READOUT_RATE_CPS,
        "C_1": CONTRAST,
        "Neff": NEFF,
        "T2_params": T2_PARAMS,
        "overheads": OVERHEADS,
        "piPulseFid": 0.997,
    }
    arguments.update(updates)
    return pipe.eta_ac(**arguments)


def test_scalar_default_is_exactly_the_legacy_formula_and_explicit_mode():
    default = eta()
    explicit = eta(control_model="scalar")
    sequence_time = pipe.sequence_duration(TAU_S, BLOCKS)
    t2_effective = pipe.t2_xy8_empirical(BLOCKS, TAU_S, **{
        key: value for key, value in T2_PARAMS.items() if key != "p"
    })
    envelope = pipe.coherence_factor(sequence_time, t2_effective, p=T2_PARAMS["p"])
    pulse_factor = 0.997 ** (8 * BLOCKS)
    gain = pipe.xy8_filter_gain(FREQUENCY_HZ, TAU_S, BLOCKS)
    cycle_time = sequence_time + sum(OVERHEADS.values())
    expected = np.sqrt(cycle_time / READOUT_RATE_CPS) / (
        gain * CONTRAST * envelope * pulse_factor * sequence_time
    )
    assert default == explicit
    assert default == pytest.approx(expected, rel=2e-15)


def test_ideal_coherent_backend_matches_scalar_without_pulse_loss():
    scalar = eta(control_model="scalar", piPulseFid=1.0)
    coherent_eta = eta(
        control_model="coherent_xy8",
        coherent_control=IDEAL_COHERENT_CONFIG,
        piPulseFid=1.0,
    )
    assert coherent_eta == pytest.approx(scalar, rel=2e-9)


def test_coherent_backend_does_not_apply_independent_pulse_fidelity_again():
    high_fidelity = eta(
        control_model="coherent_xy8",
        coherent_control=IDEAL_COHERENT_CONFIG,
        piPulseFid=1.0,
    )
    low_fidelity = eta(
        control_model="coherent_xy8",
        coherent_control=IDEAL_COHERENT_CONFIG,
        piPulseFid=0.5,
    )
    assert low_fidelity == high_fidelity
    assert eta(control_model="scalar", piPulseFid=0.5) > high_fidelity * 1e8


def test_coherent_ensemble_errors_degrade_the_control_gain_and_score():
    imperfect_config = {
        "pi_pulse_duration_s": 40e-9,
        "rotation_angle_error_std": 0.1,
        "detuning_std_angular_frequency_rad_s": 2.0 * np.pi * 1.0e6,
        "quadrature_order": 5,
        "signal_phase_rad": 0.0,
        "readout_phase_rad": np.pi / 2.0,
    }
    ideal = eta(
        control_model="coherent_xy8",
        coherent_control=IDEAL_COHERENT_CONFIG,
    )
    imperfect = eta(
        control_model="coherent_xy8",
        coherent_control=imperfect_config,
    )
    assert imperfect > ideal


def test_optical_bath_and_cycle_resources_remain_common_to_coherent_mode():
    base = eta(
        control_model="coherent_xy8",
        coherent_control=IDEAL_COHERENT_CONFIG,
    )
    assert eta(
        control_model="coherent_xy8",
        coherent_control=IDEAL_COHERENT_CONFIG,
        C_1=2.0 * CONTRAST,
    ) == pytest.approx(base / 2.0)
    assert eta(
        control_model="coherent_xy8",
        coherent_control=IDEAL_COHERENT_CONFIG,
        Rph_1=4.0 * READOUT_RATE_CPS,
    ) == pytest.approx(base / 2.0)
    worse_t2 = dict(T2_PARAMS, T2_1=T2_PARAMS["T2_1"] / 3.0)
    assert eta(
        control_model="coherent_xy8",
        coherent_control=IDEAL_COHERENT_CONFIG,
        T2_params=worse_t2,
    ) > base
    slower = dict(OVERHEADS, t_dead=100e-6)
    assert eta(
        control_model="coherent_xy8",
        coherent_control=IDEAL_COHERENT_CONFIG,
        overheads=slower,
    ) > base


def test_coherent_breakdown_is_json_safe_and_records_no_double_counting():
    ensemble = coherent.make_quasi_static_ensemble(
        [-0.03, 0.02],
        [-2.0e6, 1.0e6],
        weights=[0.4, 0.6],
        hyperfine_line_indices=[0, 1],
    )
    config = {
        "pi_pulse_duration_s": 40e-9,
        "ensemble": ensemble,
        "signal_phase_rad": 0.0,
        "readout_phase_rad": np.pi / 2.0,
        "axis_z": 0.01,
        "phase_error_rad": 0.02,
        "pulse_substeps": 6,
    }
    breakdown = pipe.best_point_breakdown(
        TAU_S,
        BLOCKS,
        FREQUENCY_HZ,
        READOUT_RATE_CPS,
        CONTRAST,
        NEFF,
        T2_PARAMS,
        OVERHEADS,
        0.2,
        control_model="coherent_xy8",
        coherent_control=config,
    )
    json.dumps(breakdown)
    assert breakdown["control_model"] == "coherent_xy8"
    assert breakdown["pulse_factor"] == 1.0
    assert breakdown["G"] == breakdown["coherent_control_gain"]
    assert breakdown["effective_control_gain"] == breakdown["G"]
    assert not breakdown["independent_pulse_factor_applied"]
    assert not breakdown["ideal_filter_gain_applied_separately"]
    assert breakdown["coherent_control"]["ensemble_source"] == "explicit_samples"
    assert breakdown["coherent_control"]["ensemble_sample_count"] == 2
    assert breakdown["coherent_control"]["pi_pulse_duration_s"] == pytest.approx(40e-9)
    assert breakdown["coherent_control"]["axis_z"] == pytest.approx(0.01)


def test_scalar_breakdown_preserves_legacy_factors_and_labels_backend():
    breakdown = pipe.best_point_breakdown(
        TAU_S,
        BLOCKS,
        FREQUENCY_HZ,
        READOUT_RATE_CPS,
        CONTRAST,
        NEFF,
        T2_PARAMS,
        OVERHEADS,
        0.997,
    )
    assert breakdown["control_model"] == "scalar"
    assert breakdown["pulse_factor"] == pytest.approx(0.997 ** (8 * BLOCKS))
    assert breakdown["G"] == pipe.xy8_filter_gain(FREQUENCY_HZ, TAU_S, BLOCKS)
    assert breakdown["effective_control_gain"] == pytest.approx(
        breakdown["G"] * breakdown["pulse_factor"]
    )
    assert breakdown["coherent_control_gain"] is None
    assert breakdown["coherent_control"] is None
    assert breakdown["independent_pulse_factor_applied"]
    assert breakdown["ideal_filter_gain_applied_separately"]


def test_invalid_backend_and_coherent_configuration_fail_clearly():
    with pytest.raises(ValueError, match="control_model"):
        eta(control_model="coherent")
    with pytest.raises(ValueError, match="coherent_control is required"):
        eta(control_model="coherent_xy8")
    with pytest.raises(ValueError, match="pi_pulse_duration_s"):
        eta(control_model="coherent_xy8", coherent_control={})
    with pytest.raises(ValueError, match="Unknown coherent_control keys"):
        eta(
            control_model="coherent_xy8",
            coherent_control={"pi_pulse_duration_s": 0.0, "typo": 1.0},
        )
    ensemble = coherent.make_quasi_static_ensemble([0.0], [0.0])
    with pytest.raises(ValueError, match="either ensemble or distribution"):
        eta(
            control_model="coherent_xy8",
            coherent_control={
                "pi_pulse_duration_s": 0.0,
                "ensemble": ensemble,
                "rotation_angle_error_std": 0.01,
            },
        )
