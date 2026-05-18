#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for extra in [ROOT / "code" / "analysis", ROOT / "code" / "utilities"]:
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import matplotlib.pyplot as plt
import numpy as np

import nvratemodel_api as pipe


def build_sweep_grid(calibration, f_target):
    sweep = (calibration or {}).get("sweep", {})
    heuristic_tau = 1.0 / (4.0 * f_target)

    tau_min = float(sweep.get("tau_min_s", max(0.25e-6, 0.25 * heuristic_tau)))
    tau_max = float(sweep.get("tau_max_s", max(5.0e-6, 2.5 * heuristic_tau)))
    tau_count = int(sweep.get("tau_count", 19))

    default_ns = [4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64]
    taus = np.linspace(tau_min, tau_max, tau_count)
    Ns = np.array(sweep.get("Ns", default_ns), dtype=int)
    return taus, Ns


def select_plot_taus(taus, f_target, best_tau):
    heuristic_tau = 1.0 / (4.0 * f_target)
    targets = [0.6 * heuristic_tau, heuristic_tau, best_tau]
    selected = []
    for target in targets:
        tau = float(min(taus, key=lambda value: abs(value - target)))
        if tau not in selected:
            selected.append(tau)
    return selected


def evaluate_point(tau, N, f_target, Rph_1, C_1, Neff, T2_params, overheads, piPulseFid):
    eta = pipe.eta_ac(
        f_target,
        tau,
        N,
        Rph_1,
        C_1,
        Neff,
        T2_params,
        overheads,
        piPulseFid=piPulseFid,
    )
    breakdown = pipe.best_point_breakdown(tau, N, f_target, Rph_1, C_1, Neff, T2_params, overheads, piPulseFid)
    return {"tau_s": float(tau), "N": int(N), "eta_proxy": float(eta), "breakdown": breakdown}


def choose_benchmarks(taus, Ns, f_target, meta, calibration):
    benchmark_config = (calibration or {}).get("benchmarks", {})
    heuristic_tau_target = float(benchmark_config.get("heuristic_tau_s", 1.0 / (4.0 * f_target)))
    heuristic_N_target = int(benchmark_config.get("heuristic_N", Ns[len(Ns) // 2]))
    overdriven_tau_target = float(benchmark_config.get("overdriven_tau_s", heuristic_tau_target))
    overdriven_N_target = int(benchmark_config.get("overdriven_N", Ns[-1]))

    heuristic_tau = min(taus, key=lambda tau: abs(tau - heuristic_tau_target))
    heuristic_N = min(Ns, key=lambda N: abs(N - heuristic_N_target))
    overdriven_tau = min(taus, key=lambda tau: abs(tau - overdriven_tau_target))
    overdriven_N = min(Ns, key=lambda N: abs(N - overdriven_N_target))

    evaluations = {
        "best": evaluate_point(
            meta["tau_best"],
            meta["N_best"],
            f_target,
            meta["Rph_1"],
            meta["C_1"],
            meta["Neff"],
            meta["T2_params"],
            meta["overheads"],
            meta["piPulseFid"],
        ),
        "heuristic": evaluate_point(
            heuristic_tau,
            heuristic_N,
            f_target,
            meta["Rph_1"],
            meta["C_1"],
            meta["Neff"],
            meta["T2_params"],
            meta["overheads"],
            meta["piPulseFid"],
        ),
        "overdriven": evaluate_point(
            overdriven_tau,
            overdriven_N,
            f_target,
            meta["Rph_1"],
            meta["C_1"],
            meta["Neff"],
            meta["T2_params"],
            meta["overheads"],
            meta["piPulseFid"],
        ),
    }

    best_eta = evaluations["best"]["eta_proxy"]
    for label, report in evaluations.items():
        report["relative_to_best"] = float(report["eta_proxy"] / max(best_eta, 1e-30))
        report["improvement_vs_point_percent"] = float((1.0 - best_eta / max(report["eta_proxy"], 1e-30)) * 100.0)
        report["label"] = label
    return evaluations


def run_sweep(
    model_kwargs,
    taus,
    Ns,
    f_target,
    Neff,
    overheads,
    T2_params,
    calibration=None,
    integration_time=300e-9,
    contrast_mode="fixed",
    pl_model="MEmodel",
):
    calibration = calibration or {}
    Rph_1, C_1, pol, extra = pipe.resolve_readout_resources(
        model_kwargs,
        calibration=calibration,
        integration_time=integration_time,
        contrast_mode=contrast_mode,
        pl_model=pl_model,
    )
    md = extra["modeldict"]
    piPulseFid = pipe.resolve_pi_pulse_fidelity(md, calibration=calibration)
    resolved_T2 = pipe.resolve_t2_params(T2_params, calibration=calibration)
    resolved_overheads = pipe.resolve_overheads(overheads, calibration=calibration)
    resolved_Neff = pipe.resolve_neff(Neff, calibration=calibration)

    results = []
    for tau in taus:
        for N in Ns:
            eta = pipe.eta_ac(
                f_target,
                tau,
                N,
                Rph_1,
                C_1,
                resolved_Neff,
                resolved_T2,
                resolved_overheads,
                piPulseFid=piPulseFid,
            )
            results.append((tau, N, float(eta)))

    tau_best, N_best, eta_best = sorted(results, key=lambda row: row[2])[0]
    breakdown = pipe.best_point_breakdown(
        tau_best,
        N_best,
        f_target,
        Rph_1,
        C_1,
        resolved_Neff,
        resolved_T2,
        resolved_overheads,
        piPulseFid,
    )

    meta = {
        "Rph_1": float(Rph_1),
        "C_1": float(C_1),
        "pol": float(pol),
        "piPulseFid": float(piPulseFid),
        "tau_best": float(tau_best),
        "N_best": int(N_best),
        "eta_best": float(eta_best),
        "Neff": float(resolved_Neff),
        "effective_detected_rate_cps": float(Rph_1 * resolved_Neff),
        "T2_params": resolved_T2,
        "overheads": resolved_overheads,
        "readout_source": extra.get("source", "nvratemodel"),
        "best_breakdown": breakdown,
    }
    benchmarks = choose_benchmarks(taus, Ns, f_target, meta, calibration)
    return results, meta, extra, benchmarks


def save_csv(results, meta, f_target, benchmarks, out_csv):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["f_target_Hz", f_target])
        writer.writerow(["readout_source", meta["readout_source"]])
        writer.writerow(["effective_detected_rate_cps", meta["effective_detected_rate_cps"]])
        writer.writerow(["Rph_1_cps", meta["Rph_1"]])
        writer.writerow(["C_1", meta["C_1"]])
        writer.writerow(["piPulseFid", meta["piPulseFid"]])
        writer.writerow(["Neff", meta["Neff"]])
        writer.writerow(["tau_best_s", meta["tau_best"]])
        writer.writerow(["N_best", meta["N_best"]])
        writer.writerow(["eta_best", meta["eta_best"]])
        for key, value in meta["T2_params"].items():
            writer.writerow([f"T2_params.{key}", value])
        for key, value in meta["overheads"].items():
            writer.writerow([f"overheads.{key}", value])

        writer.writerow([])
        writer.writerow(["benchmark", "tau_s", "N", "eta_proxy", "relative_to_best", "improvement_vs_point_percent"])
        for label, report in benchmarks.items():
            writer.writerow(
                [
                    label,
                    report["tau_s"],
                    report["N"],
                    report["eta_proxy"],
                    report["relative_to_best"],
                    report["improvement_vs_point_percent"],
                ]
            )

        writer.writerow([])
        writer.writerow(["tau_s", "N", "eta_proxy"])
        for tau, N, eta in results:
            writer.writerow([tau, N, eta])


def save_breakdown(meta, benchmarks, out_json):
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    payload = {
        "best": benchmarks["best"],
        "heuristic": benchmarks["heuristic"],
        "overdriven": benchmarks["overdriven"],
        "best_breakdown": meta["best_breakdown"],
    }
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def resolve_writable_output_path(path):
    candidate = Path(path)
    stem = candidate.stem
    suffix = candidate.suffix
    parent = candidate.parent

    for index in range(6):
        probe = candidate if index == 0 else parent / f"{stem}_retry{index}{suffix}"
        try:
            with open(probe, "wb"):
                pass
        except PermissionError:
            continue
        else:
            probe.unlink(missing_ok=True)
            return str(probe)

    raise PermissionError(f"Could not find a writable output path for {path}")


def plot_figure(results, meta, f_target, taus, Ns, out_png, out_pdf):
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    out_png = resolve_writable_output_path(out_png)
    out_pdf = resolve_writable_output_path(out_pdf)

    tau_best = meta["tau_best"]
    N_best = meta["N_best"]
    T2_params = dict(meta["T2_params"])

    fgrid = np.linspace(1e3, 2e6, 800)
    gains = np.array([pipe.xy8_filter_gain(f, tau_best, N_best) for f in fgrid])
    plot_taus = select_plot_taus(taus, f_target, tau_best)

    eta_map = {(tau, N): eta for tau, N, eta in results}

    T_seq = pipe.sequence_duration(tau_best, N_best)
    p = float(T2_params.get("p", 2.0))
    t2_kwargs = dict(T2_params)
    t2_kwargs.pop("p", None)
    t2_kwargs.pop("fit_source", None)
    T2eff = pipe.t2_xy8_empirical(N_best, tau_best, **t2_kwargs)
    env_factor = pipe.coherence_factor(T_seq, T2eff, p=p)
    pulse_factor = meta["piPulseFid"] ** (8 * N_best)
    G_target = pipe.xy8_filter_gain(f_target, tau_best, N_best)

    fig = plt.figure(figsize=(10.5, 3.6))

    axA = fig.add_subplot(1, 3, 1)
    axA.plot(fgrid, gains)
    axA.axvline(f_target, linestyle="--")
    axA.set_xlabel("Frequency (Hz)")
    axA.set_ylabel("Normalized XY8 gain G(f)")
    axA.set_ylim(0.0, 1.05)
    axA.text(-0.18, 1.05, "A", transform=axA.transAxes, fontsize=14, fontweight="bold", va="top", ha="left")

    axB = fig.add_subplot(1, 3, 2)
    for tau in plot_taus:
        ys = [eta_map[(tau, N)] for N in Ns]
        axB.plot(Ns, ys, marker="o", label=f"tau={tau * 1e6:.1f} us")
    axB.set_xlabel("XY8 block count N  (pi pulses = 8N)")
    axB.set_ylabel("eta proxy (arb.)")
    axB.text(-0.18, 1.05, "B", transform=axB.transAxes, fontsize=14, fontweight="bold", va="top", ha="left")
    axB.legend(frameon=False, fontsize=8)

    axC = fig.add_subplot(1, 3, 3)
    labels = ["G(f)", "env factor", "pulse factor", "C (readout)"]
    values = [G_target, env_factor, pulse_factor, meta["C_1"]]
    axC.bar(range(len(values)), values)
    axC.set_xticks(range(len(values)))
    axC.set_xticklabels(labels, rotation=30, ha="right")
    axC.set_ylabel("Factor (dimensionless)")
    axC.set_ylim(0.0, 1.05)
    axC.text(-0.18, 1.05, "C", transform=axC.transAxes, fontsize=14, fontweight="bold", va="top", ha="left")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png, out_pdf


def build_parser():
    parser = argparse.ArgumentParser(description="Run the calibrated NV XY8 EDT sweep.")
    parser.add_argument("--calibration", default=None, help="Path to a JSON calibration file.")
    parser.add_argument("--outdir", default="outputs/reproduced_figures", help="Directory for plots and result files.")
    parser.add_argument("--f-target", type=float, default=200e3, help="Target AC sensing frequency in Hz.")
    return parser


def main():
    args = build_parser().parse_args()
    calibration = pipe.load_calibration_if_available(args.calibration)

    model_kwargs = dict(
        T=300.0,
        B=0.02,
        thetaB=0.0,
        phiB=0.0,
        laserpower=2.2e-3,
        piPulseFid=0.995,
    )

    f_target = float(args.f_target)
    taus, Ns = build_sweep_grid(calibration, f_target)

    Neff = 1e9
    overheads = dict(t_init=5e-6, t_read=300e-9, t_dead=0.0)
    T2_params = dict(T2_1=400e-6, alpha=0.6, beta=0.0, tau0=1e-6, p=2.0)

    results, meta, extra, benchmarks = run_sweep(
        model_kwargs,
        taus,
        Ns,
        f_target,
        Neff,
        overheads,
        T2_params,
        calibration=calibration,
        integration_time=300e-9,
        contrast_mode="fixed",
        pl_model="MEmodel",
    )

    out_png = os.path.join(args.outdir, "fig_xy8_qre_nv.png")
    out_pdf = os.path.join(args.outdir, "fig_xy8_qre_nv.pdf")
    out_csv = os.path.join(args.outdir, "xy8_qre_nv_results.csv")
    out_json = os.path.join(args.outdir, "xy8_qre_nv_breakdown.json")

    save_csv(results, meta, f_target, benchmarks, out_csv)
    save_breakdown(meta, benchmarks, out_json)
    out_png, out_pdf = plot_figure(results, meta, f_target, taus, Ns, out_png, out_pdf)

    print("Wrote:", out_png)
    print("Wrote:", out_pdf)
    print("Wrote:", out_csv)
    print("Wrote:", out_json)
    print(
        f"Best: tau={meta['tau_best']:.3e} s, N={meta['N_best']}, eta={meta['eta_best']:.3e}, f={f_target:.3e} Hz"
    )
    print(
        "Benchmark improvement vs heuristic: "
        f"{benchmarks['heuristic']['improvement_vs_point_percent']:.2f}%"
    )
    print(
        "Benchmark improvement vs overdriven: "
        f"{benchmarks['overdriven']['improvement_vs_point_percent']:.2f}%"
    )


if __name__ == "__main__":
    main()
