#!/usr/bin/env python3
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# ---- Import from your pipeline file ----
# Ensure nvratemodel_api.py defines:
# - readout_resources
# - eta_ac
# - xy8_filter_gain
# - t2_xy8_empirical
# - coherence_factor
# - nv (nvratemodel module)
import nvratemodel_api as pipe


def run_sweep(model_kwargs, taus, Ns, f_target, Neff, overheads, T2_params,
              integration_time=300e-9, contrast_mode="fixed", pl_model="MEmodel"):
    Rph_1, C_1, pol, extra = pipe.readout_resources(
        model_kwargs,
        integration_time=integration_time,
        contrast_mode=contrast_mode,
        pl_model=pl_model
    )
    md = extra["modeldict"]
    piPulseFid = float(md.get("piPulseFid", 1.0))

    results = []
    for tau in taus:
        for N in Ns:
            eta = pipe.eta_ac(
                f_target, tau, N,
                Rph_1, C_1, Neff,
                T2_params, overheads,
                piPulseFid=piPulseFid
            )
            results.append((tau, N, float(eta)))

    # Find best
    results_sorted = sorted(results, key=lambda x: x[2])
    tau_best, N_best, eta_best = results_sorted[0]

    meta = {
        "Rph_1": float(Rph_1),
        "C_1": float(C_1),
        "pol": float(pol),
        "piPulseFid": piPulseFid,
        "tau_best": tau_best,
        "N_best": N_best,
        "eta_best": eta_best,
    }
    return results, meta, extra


def save_csv(results, meta, f_target, out_csv):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["f_target_Hz", f_target])
        for k, v in meta.items():
            w.writerow([k, v])
        w.writerow([])
        w.writerow(["tau_s", "N", "eta_proxy"])
        for tau, N, eta in results:
            w.writerow([tau, N, eta])


def plot_figure(results, meta, f_target, taus, Ns, out_png, out_pdf):
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    tau_best = meta["tau_best"]
    N_best = meta["N_best"]

    # ---------- Panel A: Filter gain for best setting ----------
    fgrid = np.linspace(1e3, 2e6, 800)
    G = np.array([pipe.xy8_filter_gain(f, tau_best, N_best) for f in fgrid])

    # ---------- Panel B: eta vs N for each tau ----------
    # reshape results into dict[tau][N] = eta
    eta_map = {(tau, N): eta for tau, N, eta in results}

    # ---------- Panel C: Resource breakdown at best point ----------
    # Compute factors used in eta_ac (as implemented in your pipeline)
    # This mirrors your eta_ac logic.
    T_seq = 16.0 * N_best * tau_best

    p = float(pipe.T2_params_get_p(meta=None) if hasattr(pipe, "T2_params_get_p") else 2.0)  # safe fallback
    # But we actually want p from the same params used in sweep; so we recompute from pipe defaults:
    # We'll infer p by asking eta_ac's internal logic? Not accessible. Use a constant =2 here, or edit below.
    # We'll just compute env_factor using p=2 by default; update if you want a different p.
    p = 2.0

    # Use your T2 model:
    # We'll call the same t2_xy8_empirical used in eta_ac by passing the same N,tau and reading its signature:
    # We don't have direct T2_params here; plot breakdown qualitatively.
    # For accuracy, set T2eff manually below to match your sweep inputs.
    # ---- Edit these to match your sweep ----
    T2_1 = 400e-6
    alpha = 0.6
    beta = 0.0
    tau0 = 1e-6
    T2eff = pipe.t2_xy8_empirical(N_best, tau_best, T2_1=T2_1, alpha=alpha, beta=beta, tau0=tau0)

    env_factor = pipe.coherence_factor(T_seq, T2eff, p=p)
    pulse_factor = (meta["piPulseFid"] ** (8 * N_best))
    G_target = pipe.xy8_filter_gain(f_target, tau_best, N_best)

    # ---------- Build figure ----------
    fig = plt.figure(figsize=(10.5, 3.6))

    # Panel A
    axA = fig.add_subplot(1, 3, 1)
    axA.plot(fgrid, G)
    axA.axvline(f_target, linestyle="--")
    axA.set_xlabel("Frequency (Hz)")
    axA.set_ylabel("Filter gain proxy G(f)")
    axA.text(-0.18, 1.05, "A", transform=axA.transAxes,
         fontsize=14, fontweight="bold", va="top", ha="left")

#    axA.set_title(f"A  Filter response (best)\nτ={tau_best*1e6:.1f} µs, N={N_best}")

    # Panel B
    axB = fig.add_subplot(1, 3, 2)
    for tau in sorted(taus):
        ys = [eta_map[(tau, N)] for N in Ns]
        axB.plot(Ns, ys, marker="o", label=f"τ={tau*1e6:.1f} µs")
    axB.set_xlabel("XY8 block count N  (π pulses = 8N)")
    axB.set_ylabel("η proxy (arb.)")
    axB.text(-0.18, 1.05, "B", transform=axB.transAxes,
         fontsize=14, fontweight="bold", va="top", ha="left")

#    axB.set_title(f"B  Sensitivity proxy at f={f_target/1e3:.1f} kHz")
    axB.legend(frameon=False, fontsize=8)

    # Panel C: resource factors (dimensionless)
    axC = fig.add_subplot(1, 3, 3)
    labels = ["G(f)", "env factor", "pulse factor", "C (readout)"]
    values = [G_target, env_factor, pulse_factor, meta["C_1"]]
    axC.bar(range(len(values)), values)
    axC.set_xticks(range(len(values)))
    axC.set_xticklabels(labels, rotation=30, ha="right")
    axC.set_ylabel("Factor (dimensionless)")
    axC.text(-0.18, 1.05, "C", transform=axC.transAxes,
         fontsize=14, fontweight="bold", va="top", ha="left")
#    axC.set_title("C  Resource factors at best point")

#    fig.suptitle(
#        f"NV–XY8 Quantum Resource Engineering (ensemble) | best η={meta['eta_best']:.3e}",
#        y=1.03, fontsize=11
#    )
    fig.tight_layout()

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    # --- Mirror your current sweep defaults ---
    model_kwargs = dict(
        T=300.0,
        B=0.02,
        thetaB=0.0,
        phiB=0.0,
        laserpower=2.2e-3,
        piPulseFid=0.995,
    )

    taus = [0.5e-6, 1e-6, 2e-6]
    Ns   = [4, 8, 16, 32, 64]
    f_target = 200e3

    Neff = 1e9
    overheads = dict(t_init=5e-6, t_read=300e-9)

    # Use the upgraded τ-dependent coherence parameters (see section 3 below)
    T2_params = dict(T2_1=400e-6, alpha=0.6, beta=0.0, tau0=1e-6, p=2.0)

    results, meta, extra = run_sweep(
        model_kwargs, taus, Ns, f_target, Neff, overheads, T2_params,
        integration_time=300e-9,
        contrast_mode="fixed",
        pl_model="MEmodel"
    )

    outdir = "figs"
    out_png = os.path.join(outdir, "fig_xy8_qre_nv.png")
    out_pdf = os.path.join(outdir, "fig_xy8_qre_nv.pdf")
    out_csv = os.path.join(outdir, "xy8_qre_nv_results.csv")

    save_csv(results, meta, f_target, out_csv)
    plot_figure(results, meta, f_target, taus, Ns, out_png, out_pdf)

    print("Wrote:", out_png)
    print("Wrote:", out_pdf)
    print("Wrote:", out_csv)
    print(f"Best: tau={meta['tau_best']:.3e} s, N={meta['N_best']}, eta={meta['eta_best']:.3e}, f={f_target:.3e} Hz")


if __name__ == "__main__":
    main()

