import numpy as np

# NumPy 2.0 compatibility shim for packages that still use np.float_
if not hasattr(np, "float_"):
    np.float_ = np.float64

import nvratemodel as nv  # per https://github.com/sernstETH/nvratemodel
#nv.updatePhononIntegralFullLUT(0.16800)

import os
lut_path = os.path.join(os.path.dirname(nv.__file__), "PhononIntegralLUT", "PhononIntegralLUT_noApprox.json")
if not os.path.exists(lut_path):
    nv.updatePhononIntegralFullLUT(0.16800)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def plot_eta_vs_N(results):
    """
    results: list of (tau, N, eta, f_target)
    """
    # group by tau
    taus = sorted(set(r[0] for r in results))
    for tau in taus:
        xs = sorted([r[1] for r in results if r[0] == tau])
        ys = [next(r[2] for r in results if r[0] == tau and r[1] == N) for N in xs]
        plt.plot(xs, ys, marker='o', label=f"tau={tau*1e6:.1f} µs")

    f_target = results[0][3]
    plt.xlabel("XY8 block count N  (π pulses = 8N)")
    plt.ylabel("eta_proxy (arb.)")
    plt.title(f"AC magnetometry sensitivity proxy at f={f_target/1e3:.1f} kHz")
    plt.legend()
    plt.tight_layout()
    plt.show()

def print_eta_table(results):
    taus = sorted(set(r[0] for r in results))
    Ns   = sorted(set(r[1] for r in results))
    f    = results[0][3]

    print(f"\neta_proxy table at f={f/1e3:.1f} kHz (rows=tau, cols=N)\n")
    header = "tau(µs) " + " ".join([f"{N:>10d}" for N in Ns])
    print(header)
    for tau in taus:
        row = [tau*1e6]
        for N in Ns:
            eta = next(r[2] for r in results if r[0] == tau and r[1] == N)
            row.append(eta)
        print(f"{row[0]:7.2f} " + " ".join([f"{v:10.3e}" for v in row[1:]]))

def plot_filter_gain_best(best_tau, best_N):
    fgrid = np.linspace(1e3, 2e6, 600)
    Gs = [xy8_filter_gain(f, best_tau, best_N) for f in fgrid]
    plt.plot(fgrid, Gs)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Filter gain proxy G(f)")
    plt.title(f"XY8 filter gain proxy (tau={best_tau*1e6:.1f} µs, N={best_N})")
    plt.tight_layout()
    plt.show()
    
def debug_best_point(best_tau, best_N, f_target, Rph_1, C_1, Neff, T2_params, overheads, piPulseFid):
    T_seq = 16.0 * best_N * best_tau
    p = float(T2_params.get("p", 2.0))
    t2_kwargs = dict(T2_params); t2_kwargs.pop("p", None)
    T2eff = t2_xy8_empirical(best_N, best_tau, **t2_kwargs)

    env_factor = coherence_factor(T_seq, T2eff, p=p)
    pulse_factor = (piPulseFid ** (8 * best_N))
    G = xy8_filter_gain(f_target, best_tau, best_N)

    Rph = Neff * Rph_1
    T_cycle = T_seq + overheads["t_init"] + overheads["t_read"]

    print("\n[Best-point resource breakdown]")
    print("T_seq (s):", T_seq)
    print("T2eff (s):", T2eff)
    print("env_factor:", env_factor)
    print("pulse_factor:", pulse_factor)
    print("G(f):", G)
    print("Rph_ens (cps):", Rph)
    print("C_raw:", C_1)
    print("piPulseFid:", piPulseFid)
    print("T_cycle (s):", T_cycle)

# ----------------------------
# 1) Readout resource engine
# ----------------------------
def readout_resources(model_kwargs,
                      integration_time=300e-9,
                      contrast_mode="fixed",
                      pl_model="MEmodel"):
    md = nv.makeModelDict(**model_kwargs)

    Model = nv.MEmodel if pl_model.lower().startswith("me") else nv.HighTmodel
    m = Model(**md)
    PL = float(m.PL())   # cps

    # integration time selector
    it = float(integration_time) if contrast_mode == "fixed" else str(contrast_mode)

    # getContrast returns (contrast, chosen_integrationTime, PLref/aux...)
    c_out = nv.getContrast(it, **md)
    C = float(c_out[0]) if isinstance(c_out, tuple) else float(c_out)

    fid = float(nv.getInitFidelity_ms0(**md))

    return PL, C, fid, {"modeldict": md, "contrast_raw": c_out}

# ----------------------------
# 2) XY8 coherence + filter engine
# ----------------------------
def xy8_filter_gain(f, tau, N):
    """
    Simple narrowband gain proxy:
    peak near f0 = 1/(2*tau); width ~ 1/(N*tau).
    Replace with a proper XY8 filter function later if you want.
    """
    f0 = 1.0 / (2.0 * tau)
    bw = 1.0 / (N * tau)
    return 1.0 / np.sqrt(1.0 + ((f - f0) / bw)**2)

def t2_xy8_empirical(N, tau, T2_1=400e-6, alpha=0.6, beta=0.0, tau0=1e-6, **_ignored):
    """
    Empirical XY8 coherence model with tau-dependence.

    T2eff(N,tau) = T2_1 * N^alpha * (tau/tau0)^beta

    beta=0.0 reproduces the previous model (no tau dependence).
    """
    N = max(int(N), 1)
    tau = max(float(tau), 1e-12)
    tau0 = max(float(tau0), 1e-12)
    return float(T2_1) * (N ** float(alpha)) * ((tau / tau0) ** float(beta))

def coherence_factor(T, T2eff, p=2.0):
    return np.exp(-(T / T2eff)**p)

# ----------------------------
# 3) Ensemble QRE sensitivity functional
# ----------------------------
def eta_ac(f, tau, N, Rph_1, C_1, Neff,
           T2_params, overheads,
           piPulseFid=1.0,
           kappa_C=1.0, kappa_T2=1.0):

    Rph = Neff * Rph_1
    C   = kappa_C * C_1

    T_seq = 16.0 * N * tau

    # --- split T2 scaling params vs decay exponent ---
    p = float(T2_params.get("p", 2.0))
    t2_kwargs = dict(T2_params)
    t2_kwargs.pop("p", None)

    T2eff = kappa_T2 * t2_xy8_empirical(N, tau, **t2_kwargs)
    env_factor = coherence_factor(T_seq, T2eff, p=p)

    Npi = 8 * N
    pulse_factor = (piPulseFid ** Npi)

    Ceff = C * env_factor * pulse_factor

    G = xy8_filter_gain(f, tau, N)

    T_cycle = T_seq + overheads["t_init"] + overheads["t_read"]

    return np.sqrt(T_cycle / max(Rph, 1e-30)) / max(G * Ceff * T_seq, 1e-30)

# ----------------------------
# 4) Example sweep
# ----------------------------
def sweep_demo():
    # Platform conditions for nvratemodel (use makeModelDict keys)
    model_kwargs = dict(
        T=300.0,
        B=0.02,                 # 20 mT bias
        thetaB=0.0,
        phiB=0.0,
        laserpower=2.2e-3,
        piPulseFid=0.995
        # NOTE: do NOT use Bdir; makeModelDict uses (B, thetaB, phiB)
    )

    # Readout resources from nvratemodel
    Rph_1, C_1, pol, extra = readout_resources(
        model_kwargs,
        integration_time=300e-9,      # 300 ns readout window
        contrast_mode="fixed",
        pl_model="MEmodel"
    )

    # Pull piPulseFid directly from the modeldict used
    md = extra["modeldict"]
    piPulseFid = float(md.get("piPulseFid", 1.0))

    overheads = dict(t_init=5e-6, t_read=300e-9)
    T2_params = dict(T2_1=400e-6, alpha=0.6, p=2.0)

    # Ensemble scaling (effective contributing NVs)
    Neff = 1e9

    # XY8 knobs
    taus = [0.5e-6, 1e-6, 2e-6]
    Ns   = [4, 8, 16, 32, 64]

    # Choose ONE of the following two objective styles:

    # ---- Style A (recommended): optimize at a target AC frequency ----
    f_target = 200e3  # 200 kHz example (set to your signal frequency)

    results = []
    for tau in taus:
        for N in Ns:
            eta = eta_ac(
                f_target, tau, N,
                Rph_1, C_1, Neff,
                T2_params, overheads,
                piPulseFid=piPulseFid
            )
            results.append((tau, N, eta, f_target))

    # ---- Style B (optional): minimize over frequency band ----
    # fgrid = np.linspace(1e3, 2e6, 400)
    # results = []
    # for tau in taus:
    #     for N in Ns:
    #         eta_min = np.inf
    #         f_best = None
    #         for f in fgrid:
    #             e = eta_ac(
    #                 f, tau, N,
    #                 Rph_1, C_1, Neff,
    #                 T2_params, overheads,
    #                 piPulseFid=piPulseFid
    #             )
    #             if e < eta_min:
    #                 eta_min = e
    #                 f_best = f
    #         results.append((tau, N, eta_min, f_best))

    return results

if __name__ == "__main__":
    # Baseline platform dict
    md = nv.makeModelDict()
    m = nv.MEmodel(**md)

    print("Example PL:", float(m.PL()))

    c_out = nv.getContrast(300e-9, **md)     # returns (C, t_int, PLref/aux...)
    C_ex = float(c_out[0]) if isinstance(c_out, tuple) else float(c_out)
    print("Example contrast (300 ns):", c_out, " -> C =", C_ex)

    print("Example init fidelity ms0:", float(nv.getInitFidelity_ms0(**md)))

    out = sweep_demo()

    plot_eta_vs_N(out)

    print_eta_table(out)

    tau_best, N_best, eta_best, f_best = out[0]
    
    plot_filter_gain_best(tau_best, N_best)

    
    # out elements are (tau, N, eta, f_target) [Style A]
    # or (tau, N, eta, f_best) [Style B]
    out.sort(key=lambda x: x[2])

    tau_best, N_best, eta_best, f_best = out[0]
    print(f"Best: tau={tau_best:.3e} s, N={N_best}, eta_proxy={eta_best:.3e}, f={f_best:.3e} Hz")

