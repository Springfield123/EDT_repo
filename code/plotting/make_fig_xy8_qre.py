#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for extra in [ROOT / "code" / "analysis", ROOT / "code" / "utilities"]:
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import numpy as np

import nvratemodel_api as pipe


DEFAULT_MODEL_KWARGS = {
    "T": 300.0,
    "B": 0.02,
    "thetaB": 0.0,
    "phiB": 0.0,
    "laserpower": 2.2e-3,
    "piPulseFid": 0.995,
}
DEFAULT_OVERHEADS = {"t_init": 5e-6, "t_read": 300e-9, "t_dead": 0.0}
DEFAULT_T2_PARAMS = {"T2_1": 400e-6, "alpha": 0.6, "beta": 0.0, "tau0": 1e-6, "p": 2.0}
DEFAULT_NEFF = 1e9
SCENARIO_PATHS = {
    "conservative": ROOT / "data" / "raw" / "nv" / "xy8_scenario_conservative.json",
    "nominal": ROOT / "data" / "raw" / "nv" / "xy8_scenario_nominal.json",
    "optimistic": ROOT / "data" / "raw" / "nv" / "xy8_scenario_optimistic.json",
}


def calibration_digest(calibration):
    payload = json.dumps(calibration or {}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def xy8_resonant_taus(f_target, tau_min, tau_max):
    """
    Pulse spacings where the XY8 filter is resonant with f_target.

    The toggling-frame response peaks at tau_m = (2m + 1) / (4 f), i.e. the
    fundamental and its odd harmonics. These peaks are narrow, so a grid that
    straddles them reports a sampling artifact rather than the true optimum.
    """
    taus = []
    m = 0
    while True:
        tau_m = (2.0 * m + 1.0) / (4.0 * float(f_target))
        if tau_m > tau_max:
            return taus
        if tau_m >= tau_min:
            taus.append(tau_m)
        m += 1


def xy8_resonant_frequencies(tau, f_min, f_max):
    """Frequencies where the XY8 filter is resonant for a fixed tau."""
    freqs = []
    m = 0
    while True:
        f_m = (2.0 * m + 1.0) / (4.0 * float(tau))
        if f_m > f_max:
            return freqs
        if f_m >= f_min:
            freqs.append(f_m)
        m += 1


def _resonance_resolved_fgrid(tau, f_min, f_max, background=800, halfwidth=0.02, brackets=401):
    """Background frequency grid with the exact resonances and their peaks resolved."""
    coarse = np.linspace(f_min, f_max, background)
    resonances = xy8_resonant_frequencies(tau, f_min, f_max)
    local = [
        np.linspace(f_m * (1.0 - halfwidth), f_m * (1.0 + halfwidth), brackets)
        for f_m in resonances
    ]
    merged = np.concatenate([coarse, np.array(resonances)] + local) if resonances else coarse
    merged = merged[(merged >= f_min) & (merged <= f_max)]
    return np.unique(merged)


def dedupe_taus(taus, rtol=1e-9):
    """
    Collapse spacings that differ only by floating-point noise.

    The resonance values and the centre of each bracket are the same physical
    point but need not compare equal, so np.unique alone would keep both and
    duplicate every row of the sweep at that spacing.
    """
    taus = np.sort(np.asarray(taus, dtype=float))
    if taus.size == 0:
        return taus
    kept = [taus[0]]
    for tau in taus[1:]:
        if tau - kept[-1] > rtol * max(abs(tau), 1e-30):
            kept.append(tau)
    return np.array(kept)


def build_sweep_grid(calibration, f_target):
    sweep = (calibration or {}).get("sweep", {})
    heuristic_tau = 1.0 / (4.0 * f_target)

    tau_min = float(sweep.get("tau_min_s", max(0.25e-6, 0.25 * heuristic_tau)))
    tau_max = float(sweep.get("tau_max_s", max(5.0e-6, 2.5 * heuristic_tau)))
    tau_count = int(sweep.get("tau_count", 19))

    if "taus_s" in sweep:
        taus = np.array(sweep["taus_s"], dtype=float)
    else:
        # Coarse background grid maps the landscape; the resonances and a narrow
        # bracket around each are inserted so the optimum is resolved rather
        # than aliased onto whichever background point happens to sit nearest.
        background = np.linspace(tau_min, tau_max, tau_count)
        resonances = xy8_resonant_taus(f_target, tau_min, tau_max)
        halfwidth = float(sweep.get("resonance_halfwidth_frac", 0.08))
        brackets = int(sweep.get("resonance_bracket_count", 9))
        local = [
            np.linspace(tau_m * (1.0 - halfwidth), tau_m * (1.0 + halfwidth), brackets)
            for tau_m in resonances
        ]
        merged = np.concatenate([background, np.array(resonances)] + local)
        taus = dedupe_taus(merged[(merged >= tau_min) & (merged <= tau_max)])

    # N is the XY8 block count (8N pi pulses). The sweep must start at N = 1 so
    # that a low-depth optimum can be shown to be interior rather than clamped
    # to the lower edge of the grid. N = 33-39 are sampled contiguously because
    # the earlier 32 -> 40 gap left the upper edge of the 10% window unresolved:
    # the edge has to fall on a sampled depth, not on the first grid point past
    # the threshold.
    default_ns = (
        list(range(1, 13))
        + [14, 16, 20, 24, 28]
        + list(range(32, 40))
        + [40, 48, 56, 64, 80, 96, 128]
    )
    Ns = np.array(sweep.get("Ns", default_ns), dtype=int)
    return taus, Ns


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


def extract_relative_window(results, eta_best, taus, Ns, relative_limit=1.10):
    """Return the finite set of controls within a relative sensitivity limit."""
    eta_limit = float(relative_limit) * max(float(eta_best), 1e-30)
    points = [
        {
            "tau_s": float(tau),
            "N": int(N),
            "eta_proxy": float(eta),
            "relative_to_best": float(eta / max(float(eta_best), 1e-30)),
        }
        for tau, N, eta in results
        if float(eta) <= eta_limit
    ]
    if not points:
        raise RuntimeError("The relative EDT window does not contain the best point.")
    accepted_taus = [point["tau_s"] for point in points]
    accepted_ns = [point["N"] for point in points]
    return {
        "relative_limit": float(relative_limit),
        "eta_limit": eta_limit,
        "point_count": len(points),
        "tau_min_s": min(accepted_taus),
        "tau_max_s": max(accepted_taus),
        "N_min": min(accepted_ns),
        "N_max": max(accepted_ns),
        "touches_tau_boundary": bool(min(accepted_taus) == float(taus[0]) or max(accepted_taus) == float(taus[-1])),
        "touches_N_boundary": bool(min(accepted_ns) == int(Ns[0]) or max(accepted_ns) == int(Ns[-1])),
        "points": points,
    }


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
        "scenario": str(calibration.get("label", "unspecified")),
        "scenario_status": str(calibration.get("status", "unspecified")),
        "calibration_digest": calibration_digest(calibration),
        "f_target_Hz": float(f_target),
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
    meta["edt_window_10pct"] = extract_relative_window(results, eta_best, taus, Ns, relative_limit=1.10)
    benchmarks = choose_benchmarks(taus, Ns, f_target, meta, calibration)
    return results, meta, extra, benchmarks


def evaluate_calibration(calibration, f_target=200e3):
    """Evaluate one calibration with the canonical paper sweep."""
    calibration = deepcopy(calibration or {})
    taus, Ns = build_sweep_grid(calibration, float(f_target))
    results, meta, extra, benchmarks = run_sweep(
        dict(DEFAULT_MODEL_KWARGS),
        taus,
        Ns,
        float(f_target),
        DEFAULT_NEFF,
        dict(DEFAULT_OVERHEADS),
        dict(DEFAULT_T2_PARAMS),
        calibration=calibration,
        integration_time=300e-9,
        contrast_mode="fixed",
        pl_model="MEmodel",
    )
    return results, meta, extra, benchmarks, taus, Ns


def evaluate_builtin_scenarios(f_target=200e3):
    summaries = []
    for label, path in SCENARIO_PATHS.items():
        calibration = pipe.load_calibration(path)
        _, meta, _, benchmarks, _, _ = evaluate_calibration(calibration, f_target=f_target)
        summaries.append(
            {
                "scenario": label,
                "tau_best_s": meta["tau_best"],
                "N_best": meta["N_best"],
                "eta_best": meta["eta_best"],
                "edt_window_N_min": meta["edt_window_10pct"]["N_min"],
                "edt_window_N_max": meta["edt_window_10pct"]["N_max"],
                "edt_window_tau_min_s": meta["edt_window_10pct"]["tau_min_s"],
                "edt_window_tau_max_s": meta["edt_window_10pct"]["tau_max_s"],
                "improvement_vs_N16_percent": benchmarks["heuristic"]["improvement_vs_point_percent"],
                "improvement_vs_N64_percent": benchmarks["overdriven"]["improvement_vs_point_percent"],
                "calibration_digest": meta["calibration_digest"],
            }
        )
    return summaries


def save_csv(results, meta, f_target, benchmarks, out_csv):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario", meta["scenario"]])
        writer.writerow(["scenario_status", meta["scenario_status"]])
        writer.writerow(["calibration_digest", meta["calibration_digest"]])
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
        writer.writerow(["edt_window_relative_limit", meta["edt_window_10pct"]["relative_limit"]])
        writer.writerow(["edt_window_N_min", meta["edt_window_10pct"]["N_min"]])
        writer.writerow(["edt_window_N_max", meta["edt_window_10pct"]["N_max"]])
        writer.writerow(["edt_window_tau_min_s", meta["edt_window_10pct"]["tau_min_s"]])
        writer.writerow(["edt_window_tau_max_s", meta["edt_window_10pct"]["tau_max_s"]])
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


def save_breakdown(meta, benchmarks, taus, Ns, out_json):
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    payload = {
        "configuration": {
            "scenario": meta["scenario"],
            "scenario_status": meta["scenario_status"],
            "calibration_digest": meta["calibration_digest"],
            "f_target_Hz": meta["f_target_Hz"],
            "tau_grid_s": [float(value) for value in taus],
            "N_blocks_grid": [int(value) for value in Ns],
            "pulse_count_convention": "N_pi = 8 * N_blocks",
        },
        "best": benchmarks["best"],
        "heuristic": benchmarks["heuristic"],
        "overdriven": benchmarks["overdriven"],
        "best_breakdown": meta["best_breakdown"],
        "edt_window_10pct": meta["edt_window_10pct"],
    }
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_scenario_summary(rows, out_csv):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_window_points(window, out_csv):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(window["points"][0].keys()))
        writer.writeheader()
        writer.writerows(window["points"])


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


def plot_figure(results, meta, f_target, taus, Ns, benchmarks, scenario_summaries, out_png, out_pdf):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    out_png = resolve_writable_output_path(out_png)
    out_pdf = resolve_writable_output_path(out_pdf)

    tau_best = meta["tau_best"]
    N_best = meta["N_best"]
    # The filter peaks are narrower than any affordable uniform frequency grid
    # (FWHM ~ f / (4 N) at the fundamental), so a plain linspace straddles them
    # and draws a peak that is both too low and displaced. Inject the exact
    # resonances f_m = (2m + 1) / (4 tau) with a dense local bracket around each,
    # mirroring what build_sweep_grid already does on the tau axis.
    f_plot_max = 2e6
    fgrid = _resonance_resolved_fgrid(tau_best, 1e3, f_plot_max)
    gains = np.array([pipe.xy8_filter_gain(f, tau_best, N_best) for f in fgrid])
    eta_map = {(float(tau), int(N)): float(eta) for tau, N, eta in results}
    surface = np.array([[eta_map[(float(tau), int(N))] for tau in taus] for N in Ns], dtype=float)
    log_penalty_raw = np.log10(np.maximum(surface / max(meta["eta_best"], 1e-30), 1.0))
    # The surface spans ~6 decades, so a full-range scale renders the window
    # this panel exists to show as a single column of pixels. Clip at one
    # decade and invert the map so the useful region is the bright region.
    penalty_clip = 1.0
    log_penalty = np.clip(log_penalty_raw, 0.0, penalty_clip)

    fig = plt.figure(figsize=(12.0, 3.9))

    axA = fig.add_subplot(1, 3, 1)
    axA.plot(fgrid, gains)
    axA.axvline(f_target, linestyle="--")
    axA.set_xlabel("Frequency (MHz)")
    axA.set_ylabel("Normalized XY8 gain, $G(f)$")
    axA.set_ylim(0.0, 1.05)
    axA.set_xlim(0.0, 2.0e6)
    axA.set_xticks([0.0, 0.5e6, 1.0e6, 1.5e6, 2.0e6])
    axA.set_xticklabels(["0", "0.5", "1.0", "1.5", "2.0"])
    axA.set_title(f"Filter response at N={N_best}")
    axA.text(-0.18, 1.05, "A", transform=axA.transAxes, fontsize=14, fontweight="bold", va="top", ha="left")

    axB = fig.add_subplot(1, 3, 2)
    mesh = axB.pcolormesh(
        taus * 1e6, Ns, log_penalty, shading="nearest", cmap="viridis_r", vmin=0.0, vmax=penalty_clip
    )
    axB.contour(
        taus * 1e6,
        Ns,
        log_penalty_raw,
        levels=[np.log10(meta["edt_window_10pct"]["relative_limit"])],
        colors="#000000",
        linewidths=1.8,
    )
    marker_specs = {
        "best": ("*", "#111111", "Best"),
        "heuristic": ("o", "#ff9f1c", "Heuristic N=16"),
        "overdriven": ("s", "#e63946", "High depth N=64"),
    }
    for key, (marker, color, label) in marker_specs.items():
        point = benchmarks[key]
        axB.scatter(
            point["tau_s"] * 1e6,
            point["N"],
            marker=marker,
            s=95 if key == "best" else 55,
            facecolor=color,
            edgecolor="black",
            linewidth=0.7,
            label=label,
            zorder=5,
        )
    axB.set_xlabel("Half-interpulse spacing, \u03c4 (\u00b5s)")
    axB.set_ylabel("XY8 block count, N")
    axB.set_yscale("log", base=2)
    axB.set_yticks([1, 2, 4, 8, 16, 32, 64, 128])
    axB.set_yticklabels(["1", "2", "4", "8", "16", "32", "64", "128"])
    axB.set_title("Sensitivity penalty (lower is better)")
    handles, labels = axB.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="#000000", linewidth=1.8))
    labels.append("Within 10% of best")
    axB.legend(handles, labels, frameon=True, framealpha=0.85, edgecolor="none", fontsize=7, loc="upper right")
    colorbar = fig.colorbar(mesh, ax=axB, pad=0.02)
    colorbar.set_label(f"log$_{{10}}$($\\eta$ / $\\eta_{{min}}$), clipped at {penalty_clip:.0f} (bright = better)")
    axB.text(-0.18, 1.05, "B", transform=axB.transAxes, fontsize=14, fontweight="bold", va="top", ha="left")

    axC = fig.add_subplot(1, 3, 3)
    labels = [row["scenario"].capitalize() for row in scenario_summaries]
    values = [row["N_best"] for row in scenario_summaries]
    bars = axC.bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b"])
    axC.set_ylabel("Best XY8 block count, N")
    axC.set_ylim(0.0, max(values) * 1.22)
    axC.set_title("Hardware quality shifts the optimum")
    for bar, row in zip(bars, scenario_summaries):
        axC.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.035,
            f"N={row['N_best']}\n\u03c4={row['tau_best_s'] * 1e6:.2f} \u00b5s",
            ha="center",
            va="bottom",
            fontsize=8,
        )
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
    f_target = float(args.f_target)
    results, meta, extra, benchmarks, taus, Ns = evaluate_calibration(calibration, f_target=f_target)
    scenario_summaries = evaluate_builtin_scenarios(f_target=f_target)

    out_png = os.path.join(args.outdir, "fig_xy8_qre_nv.png")
    out_pdf = os.path.join(args.outdir, "fig_xy8_qre_nv.pdf")
    out_csv = os.path.join(args.outdir, "xy8_qre_nv_results.csv")
    out_json = os.path.join(args.outdir, "xy8_qre_nv_breakdown.json")
    out_scenarios = os.path.join(args.outdir, "xy8_scenario_optima.csv")
    out_window = os.path.join(args.outdir, "xy8_edt_window_10pct.csv")

    save_csv(results, meta, f_target, benchmarks, out_csv)
    save_breakdown(meta, benchmarks, taus, Ns, out_json)
    save_scenario_summary(scenario_summaries, out_scenarios)
    save_window_points(meta["edt_window_10pct"], out_window)
    out_png, out_pdf = plot_figure(
        results,
        meta,
        f_target,
        taus,
        Ns,
        benchmarks,
        scenario_summaries,
        out_png,
        out_pdf,
    )

    print("Wrote:", out_png)
    print("Wrote:", out_pdf)
    print("Wrote:", out_csv)
    print("Wrote:", out_json)
    print("Wrote:", out_scenarios)
    print("Wrote:", out_window)
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
