import numpy as np

from edt.nv.xy8 import NVParams, XY8Model


def test_nv_sweep_returns_positive_proxy():
    model = XY8Model(NVParams())
    result = model.sweep(np.linspace(1e-6, 5e-6, 4), np.array([8, 16, 24]))
    assert result.best_value > 0.0


def test_xy8_filter_gain_is_bounded():
    tau = 2.5e-6
    n_pulses = 32
    model = XY8Model(NVParams(signal_freq_hz=1.0 / (2.0 * tau)))
    gain = model.filter_gain(tau, n_pulses)
    assert 0.0 <= gain <= 1.0


def test_factor_breakdown_matches_proxy():
    params = NVParams(
        signal_freq_hz=200e3,
        photon_rate_hz=3e5,
        readout_contrast=0.04,
        pi_pulse_fidelity=0.997,
        t2_base_s=500e-6,
        t2_alpha=0.5,
        t2_beta=0.1,
    )
    model = XY8Model(params)
    tau = 1.0e-6
    n_pulses = 64
    breakdown = model.factor_breakdown(tau, n_pulses)

    assert breakdown["sequence_time_s"] > 0.0
    assert breakdown["cycle_time_s"] > breakdown["sequence_time_s"]
    assert breakdown["coherence_factor"] > 0.0
    assert breakdown["pulse_factor"] > 0.0
    assert np.isclose(breakdown["sensitivity_proxy"], model.sensitivity_proxy(tau, n_pulses))
