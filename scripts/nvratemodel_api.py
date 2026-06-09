import json
import os
from pathlib import Path

import numpy as np

# NumPy 2.0 compatibility shim for packages that still use np.float_
if not hasattr(np, "float_"):
    np.float_ = np.float64

try:
    import nvratemodel as nv  # per https://github.com/sernstETH/nvratemodel
except ModuleNotFoundError:
    nv = None
else:
    lut_path = os.path.join(os.path.dirname(nv.__file__), "PhononIntegralLUT", "PhononIntegralLUT_noApprox.json")
    if not os.path.exists(lut_path):
        nv.updatePhononIntegralFullLUT(0.16800)

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def plot_eta_vs_N(results):
    """
    results: list of (tau, N, eta, f_target)
    """
    taus = sorted(set(r[0] for r in results))
    for tau in taus:
        xs = sorted([r[1] for r in results if r[0] == tau])
        ys = [next(r[2] for r in results if r[0] == tau and r[1] == N) for N in xs]
        plt.plot(xs, ys, marker="o", label=f"tau={tau * 1e6:.1f} us")

    f_target = results[0][3]
    plt.xlabel("XY8 block count N  (pi pulses = 8N)")
    plt.ylabel("eta_proxy (arb.)")
    plt.title(f"AC magnetometry sensitivity proxy at f={f_target / 1e3:.1f} kHz")
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_eta_table(results):
    taus = sorted(set(r[0] for r in results))
    ns = sorted(set(r[1] for r in results))
    f_target = results[0][3]

    print(f"\neta_proxy table at f={f_target / 1e3:.1f} kHz (rows=tau, cols=N)\n")
    header = "tau(us) " + " ".join([f"{N:>10d}" for N in ns])
    print(header)
    for tau in taus:
        row = [tau * 1e6]
        for N in ns:
            eta = next(r[2] for r in results if r[0] == tau and r[1] == N)
            row.append(eta)
        print(f"{row[0]:7.2f} " + " ".join([f"{value:10.3e}" for value in row[1:]]))


def plot_filter_gain_best(best_tau, best_N):
    fgrid = np.linspace(1e3, 2e6, 600)
    gains = [xy8_filter_gain(f, best_tau, best_N) for f in fgrid]
    plt.plot(fgrid, gains)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized XY8 gain G(f)")
    plt.title(f"XY8 filter response (tau={best_tau * 1e6:.1f} us, N={best_N})")
    plt.tight_layout()
    plt.show()


def load_calibration(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_calibration_if_available(path):
    if path is None:
        return {}
    calibration_path = Path(path)
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")
    return load_calibration(calibration_path)


def fit_t2_power_law(points, tau0=None, p=2.0):
    """
    Fit T2eff(N, tau) = T2_1 * N^alpha * (tau / tau0)^beta
    from measured effective coherence times.

    Each point must provide: {"N": ..., "tau_s": ..., "T2eff_s": ...}
    """
    if len(points) < 2:
        raise ValueError("Need at least two coherence calibration points to fit the T2 power law.")

    ns = np.array([max(int(point["N"]), 1) for point in points], dtype=float)
    taus = np.array([max(float(point["tau_s"]), 1e-12) for point in points], dtype=float)
    t2_values = np.array([max(float(point["T2eff_s"]), 1e-12) for point in points], dtype=float)

    tau0 = float(tau0) if tau0 is not None else float(np.median(taus))
    tau0 = max(tau0, 1e-12)

    design = np.column_stack(
        [
            np.ones_like(ns),
            np.log(ns),
            np.log(taus / tau0),
        ]
    )
    coeffs, _, _, _ = np.linalg.lstsq(design, np.log(t2_values), rcond=None)
    return {
        "T2_1": float(np.exp(coeffs[0])),
        "alpha": float(coeffs[1]),
        "beta": float(coeffs[2]),
        "tau0": tau0,
        "p": float(p),
    }


def resolve_readout_resources(
    model_kwargs,
    calibration=None,
    integration_time=300e-9,
    contrast_mode="fixed",
    pl_model="MEmodel",
):
    calibration = calibration or {}
    readout = calibration.get("readout", {})
    pulse = calibration.get("pulse", {})

    if readout.get("mode", "nvratemodel").lower() == "explicit":
        if "effective_detected_rate_cps" in readout:
            Rph_1 = float(readout["effective_detected_rate_cps"])
        elif "Rph_total" in readout:
            Rph_1 = float(readout["Rph_total"])
        else:
            Rph_1 = float(readout["Rph_1"])
        C_1 = float(readout["C_1"])
        fid = float(readout.get("init_fidelity", 1.0))
        md = dict(model_kwargs)
        md["piPulseFid"] = float(pulse.get("piPulseFid", md.get("piPulseFid", 1.0)))
        return Rph_1, C_1, fid, {"modeldict": md, "contrast_raw": readout, "source": "explicit"}

    return readout_resources(
        model_kwargs,
        integration_time=integration_time,
        contrast_mode=contrast_mode,
        pl_model=pl_model,
    )


def resolve_t2_params(default_params, calibration=None):
    calibration = calibration or {}
    coherence = calibration.get("coherence", {})
    fit_points = calibration.get("coherence_fit_points", [])

    if fit_points:
        tau0 = coherence.get("tau0", default_params.get("tau0", 1e-6))
        p = coherence.get("p", default_params.get("p", 2.0))
        fitted = fit_t2_power_law(fit_points, tau0=tau0, p=p)
        fitted["fit_source"] = "coherence_fit_points"
        return fitted

    merged = dict(default_params)
    merged.update(coherence)
    merged["fit_source"] = "direct_parameters"
    return merged


def resolve_overheads(default_overheads, calibration=None):
    calibration = calibration or {}
    resolved = dict(default_overheads)
    resolved.update(calibration.get("overheads", {}))
    resolved.setdefault("t_dead", 0.0)
    return resolved


def resolve_neff(default_neff, calibration=None):
    calibration = calibration or {}
    ensemble = calibration.get("ensemble", {})
    readout = calibration.get("readout", {})
    if "effective_detected_rate_cps" in readout or "Rph_total" in readout:
        return 1.0
    return float(ensemble.get("Neff", default_neff))


def resolve_pi_pulse_fidelity(modeldict, calibration=None):
    calibration = calibration or {}
    pulse = calibration.get("pulse", {})
    if "piPulseFid" in pulse:
        return float(pulse["piPulseFid"])
    if "error_per_pulse" in pulse:
        return float(np.exp(-float(pulse["error_per_pulse"])))
    return float(modeldict.get("piPulseFid", 1.0))


def best_point_breakdown(best_tau, best_N, f_target, Rph_1, C_1, Neff, T2_params, overheads, piPulseFid):
    T_seq = sequence_duration(best_tau, best_N)
    p = float(T2_params.get("p", 2.0))
    t2_kwargs = dict(T2_params)
    t2_kwargs.pop("p", None)
    t2_kwargs.pop("fit_source", None)
    T2eff = t2_xy8_empirical(best_N, best_tau, **t2_kwargs)

    env_factor = coherence_factor(T_seq, T2eff, p=p)
    pulse_factor = piPulseFid ** (8 * best_N)
    gain = xy8_filter_gain(f_target, best_tau, best_N)

    breakdown = {
        "tau_s": float(best_tau),
        "N": int(best_N),
        "f_target_Hz": float(f_target),
        "T_seq_s": float(T_seq),
        "T2eff_s": float(T2eff),
        "env_factor": float(env_factor),
        "pulse_factor": float(pulse_factor),
        "G": float(gain),
        "effective_detected_rate_cps": float(Neff * Rph_1),
        "Rph_ens_cps": float(Neff * Rph_1),
        "C_raw": float(C_1),
        "piPulseFid": float(piPulseFid),
        "T_cycle_s": float(T_seq + overheads["t_init"] + overheads["t_read"] + overheads.get("t_dead", 0.0)),
    }
    return breakdown


def debug_best_point(best_tau, best_N, f_target, Rph_1, C_1, Neff, T2_params, overheads, piPulseFid):
    breakdown = best_point_breakdown(best_tau, best_N, f_target, Rph_1, C_1, Neff, T2_params, overheads, piPulseFid)
    print("\n[Best-point resource breakdown]")
    for key, value in breakdown.items():
        print(f"{key}: {value}")


# ----------------------------
# 1) Readout resource engine
# ----------------------------
def readout_resources(model_kwargs, integration_time=300e-9, contrast_mode="fixed", pl_model="MEmodel"):
    if nv is None:
        raise ModuleNotFoundError(
            "nvratemodel is not installed. Install it or use a calibration JSON with "
            '"readout": {"mode": "explicit", ...}.'
        )

    md = nv.makeModelDict(**model_kwargs)

    Model = nv.MEmodel if pl_model.lower().startswith("me") else nv.HighTmodel
    model = Model(**md)
    PL = float(model.PL())

    integration_selector = float(integration_time) if contrast_mode == "fixed" else str(contrast_mode)
    contrast_output = nv.getContrast(integration_selector, **md)
    contrast = float(contrast_output[0]) if isinstance(contrast_output, tuple) else float(contrast_output)
    init_fidelity = float(nv.getInitFidelity_ms0(**md))

    return PL, contrast, init_fidelity, {"modeldict": md, "contrast_raw": contrast_output, "source": "nvratemodel"}


# ----------------------------
# 2) XY8 coherence + filter engine
# ----------------------------
def sequence_duration(tau, N):
    return float(16.0 * max(int(N), 1) * max(float(tau), 1e-12))


def xy8_pulse_times(tau, N):
    N = max(int(N), 1)
    tau = max(float(tau), 1e-12)
    Npi = 8 * N
    pulse_index = np.arange(1, Npi + 1, dtype=float)
    return (2.0 * pulse_index - 1.0) * tau


def xy8_switching_response(f, tau, N):
    total_time = sequence_duration(tau, N)
    edges = np.concatenate(([0.0], xy8_pulse_times(tau, N), [total_time]))
    durations = np.diff(edges)
    signs = (-1.0) ** np.arange(durations.size, dtype=float)
    omega = 2.0 * np.pi * max(float(f), 0.0)
    if np.isclose(omega, 0.0):
        return complex(np.sum(signs * durations))
    phases = np.exp(1j * omega * edges)
    return complex(np.sum(signs * (phases[1:] - phases[:-1]) / (1j * omega)))


def xy8_sensitivity_kernel(f, tau, N):
    """
    Dimensionless AC sensing kernel bounded by 0 <= G <= 1.
    """
    total_time = sequence_duration(tau, N)
    response = abs(xy8_switching_response(f, tau, N))
    return float(response / max(total_time, 1e-30))


def xy8_filter_gain(f, tau, N):
    """
    Bounded XY8 filter gain derived from the toggling-function response.
    This keeps the gain dimensionless and avoids artificial blow-ups from
    normalizing by near-zero resonant values on coarse grids.
    """
    return xy8_sensitivity_kernel(f, tau, N)


def t2_xy8_empirical(N, tau, T2_1=400e-6, alpha=0.6, beta=0.0, tau0=1e-6, **_ignored):
    """
    Empirical XY8 coherence model with tau dependence.

    T2eff(N, tau) = T2_1 * N^alpha * (tau / tau0)^beta
    """
    N = max(int(N), 1)
    tau = max(float(tau), 1e-12)
    tau0 = max(float(tau0), 1e-12)
    return float(T2_1) * (N ** float(alpha)) * ((tau / tau0) ** float(beta))


def coherence_factor(T, T2eff, p=2.0):
    return float(np.exp(-((T / max(T2eff, 1e-30)) ** p)))


# ----------------------------
# 3) Ensemble QRE sensitivity functional
# ----------------------------
def eta_ac(f, tau, N, Rph_1, C_1, Neff, T2_params, overheads, piPulseFid=1.0, kappa_C=1.0, kappa_T2=1.0):
    Rph = Neff * Rph_1
    contrast = kappa_C * C_1

    T_seq = sequence_duration(tau, N)

    p = float(T2_params.get("p", 2.0))
    t2_kwargs = dict(T2_params)
    t2_kwargs.pop("p", None)
    t2_kwargs.pop("fit_source", None)

    T2eff = kappa_T2 * t2_xy8_empirical(N, tau, **t2_kwargs)
    env_factor = coherence_factor(T_seq, T2eff, p=p)

    Npi = 8 * max(int(N), 1)
    pulse_factor = piPulseFid ** Npi
    effective_contrast = contrast * env_factor * pulse_factor

    gain = xy8_filter_gain(f, tau, N)
    T_cycle = T_seq + overheads["t_init"] + overheads["t_read"] + overheads.get("t_dead", 0.0)

    return float(np.sqrt(T_cycle / max(Rph, 1e-30)) / max(gain * effective_contrast * T_seq, 1e-30))


# ----------------------------
# 4) Example sweep
# ----------------------------
def sweep_demo():
    model_kwargs = dict(
        T=300.0,
        B=0.02,
        thetaB=0.0,
        phiB=0.0,
        laserpower=2.2e-3,
        piPulseFid=0.995,
    )

    Rph_1, C_1, pol, extra = readout_resources(
        model_kwargs,
        integration_time=300e-9,
        contrast_mode="fixed",
        pl_model="MEmodel",
    )

    md = extra["modeldict"]
    piPulseFid = float(md.get("piPulseFid", 1.0))

    overheads = dict(t_init=5e-6, t_read=300e-9, t_dead=0.0)
    T2_params = dict(T2_1=400e-6, alpha=0.6, beta=0.0, tau0=1e-6, p=2.0)
    Neff = 1e9

    taus = [0.5e-6, 1e-6, 2e-6]
    Ns = [4, 8, 16, 32, 64]
    f_target = 200e3

    results = []
    for tau in taus:
        for N in Ns:
            eta = eta_ac(f_target, tau, N, Rph_1, C_1, Neff, T2_params, overheads, piPulseFid=piPulseFid)
            results.append((tau, N, eta, f_target))

    return results


if __name__ == "__main__":
    if nv is None:
        raise ModuleNotFoundError(
            "nvratemodel is not installed. The demo block in nvratemodel_api.py needs it."
        )

    md = nv.makeModelDict()
    model = nv.MEmodel(**md)

    print("Example PL:", float(model.PL()))

    contrast_output = nv.getContrast(300e-9, **md)
    contrast_value = float(contrast_output[0]) if isinstance(contrast_output, tuple) else float(contrast_output)
    print("Example contrast (300 ns):", contrast_output, " -> C =", contrast_value)
    print("Example init fidelity ms0:", float(nv.getInitFidelity_ms0(**md)))

    results = sweep_demo()

    plot_eta_vs_N(results)
    print_eta_table(results)

    results.sort(key=lambda row: row[2])
    tau_best, N_best, eta_best, f_best = results[0]

    plot_filter_gain_best(tau_best, N_best)

    print(f"Best: tau={tau_best:.3e} s, N={N_best}, eta_proxy={eta_best:.3e}, f={f_best:.3e} Hz")
