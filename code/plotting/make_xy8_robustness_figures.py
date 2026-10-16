#!/usr/bin/env python3
"""Generate XY8 resource-ablation and hardware-uncertainty figures."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

import make_fig_xy8_qre as sweep


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "code" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import xy8_control_recommender as recommender


NOMINAL_CALIBRATION = ROOT / "data" / "raw" / "nv" / "xy8_scenario_nominal.json"
SCENARIO_PATHS = {
    "Conservative": ROOT / "data" / "raw" / "nv" / "xy8_scenario_conservative.json",
    "Nominal": NOMINAL_CALIBRATION,
    "Optimistic": ROOT / "data" / "raw" / "nv" / "xy8_scenario_optimistic.json",
}
REGION_CODES = {"Shallow": 0, "Moderate": 1, "Deep": 2, "Very deep": 3, "Unresolved": 4}


def ablation_calibrations(nominal):
    calibrations = [("Nominal", deepcopy(nominal))]

    no_pulse_errors = deepcopy(nominal)
    no_pulse_errors["label"] = "nominal_no_pulse_errors"
    no_pulse_errors["pulse"]["piPulseFid"] = 1.0
    calibrations.append(("No pulse errors", no_pulse_errors))

    no_decoherence = deepcopy(nominal)
    no_decoherence["label"] = "nominal_no_decoherence"
    no_decoherence["coherence"].update({"T2_1": 1.0e6, "alpha": 0.0, "beta": 0.0})
    calibrations.append(("No decoherence", no_decoherence))

    no_overhead = deepcopy(nominal)
    no_overhead["label"] = "nominal_no_timing_overhead"
    no_overhead["overheads"].update({"t_init": 0.0, "t_read": 0.0, "t_dead": 0.0})
    calibrations.append(("No timing overhead", no_overhead))
    return calibrations


def evaluate_ablation(nominal, f_target):
    rows = []
    curves = {}
    resonant_tau = 1.0 / (4.0 * f_target)
    for label, calibration in ablation_calibrations(nominal):
        results, meta, _, _, taus, Ns = sweep.evaluate_calibration(calibration, f_target=f_target)
        tau = float(min(taus, key=lambda value: abs(value - resonant_tau)))
        eta_by_n = {int(N): float(eta) for row_tau, N, eta in results if np.isclose(row_tau, tau, rtol=0.0, atol=1e-15)}
        curve = np.array([eta_by_n[int(N)] for N in Ns], dtype=float)
        curves[label] = (Ns.copy(), curve / np.min(curve))
        rows.append(
            {
                "ablation": label,
                "tau_best_s": meta["tau_best"],
                "N_best": meta["N_best"],
                "eta_best": meta["eta_best"],
                "best_at_N_boundary": int(meta["N_best"] in {int(Ns[0]), int(Ns[-1])}),
                "calibration_digest": meta["calibration_digest"],
            }
        )
    return rows, curves


def evaluate_phase_diagram(nominal, f_target, fidelities, t2_scales):
    best_n = np.zeros((len(t2_scales), len(fidelities)), dtype=int)
    region_codes = np.zeros((len(t2_scales), len(fidelities)), dtype=int)
    rows = []
    for i, t2_scale in enumerate(t2_scales):
        for j, fidelity in enumerate(fidelities):
            t2_s = float(nominal["coherence"]["T2_1"] * t2_scale)
            decision = recommender.recommend_control(
                nominal,
                frequency_hz=f_target,
                pulse_fidelity=float(fidelity),
                t2_s=t2_s,
            )
            recommendation = decision["recommendation"]
            window = recommendation["edt_window_10pct"]
            region = recommendation["control_region"]
            best_n[i, j] = recommendation["N"]
            region_codes[i, j] = REGION_CODES[region]
            rows.append(
                {
                    "piPulseFid": float(fidelity),
                    "T2_scale": float(t2_scale),
                    "T2_1_s": t2_s,
                    "tau_best_s": recommendation["tau_s"],
                    "N_best": recommendation["N"],
                    "total_pi_pulses": recommendation["total_pi_pulses"],
                    "eta_best": recommendation["predicted_eta_proxy"],
                    "edt_window_N_min": window["N_min"],
                    "edt_window_N_max": window["N_max"],
                    "pareto_alternative_N": recommendation["pareto_alternative"]["N"],
                    "control_region": region,
                    "control_region_code": REGION_CODES[region],
                    "best_at_N_boundary": int(recommendation["boundary_status"] != "resolved interior optimum"),
                    "window_at_N_boundary": int(window["touches_depth_boundary"]),
                    "recommendation_digest": decision["recommendation_digest"],
                }
            )
    return rows, best_n, region_codes


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_ablation(rows, curves, out_png, out_pdf):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [1.55, 1.0]})
    # "No decoherence" tracks "Nominal" closely enough to hide it completely, so
    # the reference curve is drawn last, thicker, and dashed.
    for (label, (Ns, penalty)), color in zip(curves.items(), colors):
        reference = label == "Nominal"
        ax_a.plot(
            Ns,
            penalty,
            marker="o",
            markersize=3,
            linewidth=2.6 if reference else 1.6,
            linestyle="--" if reference else "-",
            label=label,
            color=color,
            zorder=5 if reference else 3,
        )
    ax_a.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax_a.set_xscale("log", base=2)
    ax_a.set_yscale("log")
    ax_a.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
    ax_a.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128"])
    ax_a.set_xlabel("XY8 block count, N")
    ax_a.set_ylabel("Penalty relative to each ablation's best")
    ax_a.set_title("Pulse error sets depth in the nominal model")
    ax_a.legend(frameon=False, fontsize=8)
    ax_a.text(-0.12, 1.04, "A", transform=ax_a.transAxes, fontsize=14, fontweight="bold")

    bars = ax_b.bar([row["ablation"] for row in rows], [row["N_best"] for row in rows], color=colors)
    ax_b.set_ylabel("Best XY8 block count, N")
    ax_b.set_title("Optimal depth by ablation")
    ax_b.tick_params(axis="x", rotation=25)
    for label in ax_b.get_xticklabels():
        label.set_ha("right")
    maximum = max(row["N_best"] for row in rows)
    ax_b.set_ylim(0, maximum * 1.18)
    for bar, row in zip(bars, rows):
        boundary = "*" if row["best_at_N_boundary"] else ""
        ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + maximum * 0.025, f"N={row['N_best']}{boundary}", ha="center", fontsize=8)
    ax_b.text(-0.12, 1.04, "B", transform=ax_b.transAxes, fontsize=14, fontweight="bold")
    fig.text(0.99, 0.01, "* optimum reaches the tested sweep boundary", ha="right", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_phase(region_codes, fidelities, t2_scales, nominal, scenarios, selected_decision, out_png, out_pdf):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    colors = ["#9ecae1", "#4c78a8", "#f2cf5b", "#e45756", "#7f7f7f"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5, 1.0), cmap.N)
    fig, ax = plt.subplots(figsize=(8.3, 5.2))
    mesh = ax.pcolormesh(fidelities, t2_scales, region_codes, shading="nearest", cmap=cmap, norm=norm)
    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_ticks(range(len(REGION_CODES)))
    colorbar.set_ticklabels(
        ["Shallow\nN <= 10", "Moderate\nN = 11-20", "Deep\nN = 24-48", "Very deep\nN >= 56", "Unresolved\nboundary"]
    )
    colorbar.set_label("Recommended control region")

    markers = {"Conservative": "o", "Nominal": "*", "Optimistic": "^"}
    sizes = {"Conservative": 70, "Nominal": 170, "Optimistic": 85}
    for label, scenario in scenarios.items():
        t2_scale = scenario["coherence"]["T2_1"] / nominal["coherence"]["T2_1"]
        ax.scatter(
            [scenario["pulse"]["piPulseFid"]],
            [t2_scale],
            marker=markers[label],
            s=sizes[label],
            color="white",
            edgecolor="black",
            linewidth=1.1,
            label=label,
            zorder=5,
        )

    ax.annotate(
        "",
        xy=(0.9992, 1.87),
        xytext=(0.9940, 1.87),
        arrowprops={"arrowstyle": "->", "linewidth": 1.8, "color": "black"},
    )
    ax.text(0.9966, 1.91, "Improve pulse fidelity", ha="center", va="bottom", fontsize=9)

    selected = selected_decision["recommendation"]
    selected_window = selected["edt_window_10pct"]
    selected_alternative = selected["pareto_alternative"]
    callout = (
        f"Nominal recommendation\n"
        f"N={selected['N']} ({selected['total_pi_pulses']} pulses), tau={selected['tau_s'] * 1e6:.2f} us\n"
        f"10% window: N={selected_window['N_min']}-{selected_window['N_max']}; alternative: N={selected_alternative['N']}"
    )
    ax.annotate(
        callout,
        xy=(nominal["pulse"]["piPulseFid"], 1.0),
        xytext=(0.99225, 1.55),
        textcoords="data",
        ha="left",
        va="center",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#555555", "alpha": 0.94},
        arrowprops={"arrowstyle": "->", "color": "#333333", "linewidth": 1.1},
    )
    ax.set_xlabel("Per-pulse fidelity")
    ax.set_ylabel("T2 scale relative to nominal")
    ax.set_title("Measured hardware quality directly selects the control depth")
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
        fontsize=8,
    )
    fig.text(
        0.01,
        0.01,
        "Scenario markers project only F_pi and T2; contrast, overhead, photon rate, and coherence scaling remain nominal in the map.",
        ha="left",
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description="Generate XY8 robustness figures from the canonical sweep.")
    parser.add_argument("--calibration", default=str(NOMINAL_CALIBRATION))
    parser.add_argument("--outdir", default="outputs/reproduced_figures")
    parser.add_argument("--f-target", type=float, default=200e3)
    return parser


def main():
    args = build_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    nominal = json.loads(Path(args.calibration).read_text(encoding="utf-8"))

    ablation_rows, curves = evaluate_ablation(nominal, args.f_target)
    write_rows(outdir / "xy8_resource_ablation.csv", ablation_rows)
    plot_ablation(
        ablation_rows,
        curves,
        outdir / "fig_xy8_resource_ablation.png",
        outdir / "fig_xy8_resource_ablation.pdf",
    )

    fidelities = np.linspace(0.992, 1.0, 17)
    t2_scales = np.linspace(0.5, 2.0, 13)
    phase_rows, best_n, region_codes = evaluate_phase_diagram(nominal, args.f_target, fidelities, t2_scales)
    write_rows(outdir / "xy8_uncertainty_phase_diagram.csv", phase_rows)
    scenarios = {label: json.loads(path.read_text(encoding="utf-8")) for label, path in SCENARIO_PATHS.items()}
    selected_decision = recommender.recommend_control(
        nominal,
        frequency_hz=args.f_target,
        pulse_fidelity=nominal["pulse"]["piPulseFid"],
        t2_s=nominal["coherence"]["T2_1"],
    )
    plot_phase(
        region_codes,
        fidelities,
        t2_scales,
        nominal,
        scenarios,
        selected_decision,
        outdir / "fig_xy8_uncertainty_phase_diagram.png",
        outdir / "fig_xy8_uncertainty_phase_diagram.pdf",
    )

    summary = {
        "f_target_Hz": float(args.f_target),
        "nominal_calibration": str(Path(args.calibration)),
        "pulse_count_convention": "N_pi = 8 * N_blocks",
        "ablations": ablation_rows,
        "uncertainty_grid": {
            "piPulseFid": [float(value) for value in fidelities],
            "T2_scales": [float(value) for value in t2_scales],
        },
        "control_region_definitions": {
            "Shallow": "N <= 10",
            "Moderate": "N = 11-20",
            "Deep": "N = 24-48",
            "Very deep": "N >= 56",
            "Unresolved": "optimum reaches the sweep boundary",
        },
        "selected_recommendation": selected_decision,
        "figure_annotation": {
            "N": selected_decision["recommendation"]["N"],
            "tau_s": selected_decision["recommendation"]["tau_s"],
            "edt_window_N_min": selected_decision["recommendation"]["edt_window_10pct"]["N_min"],
            "edt_window_N_max": selected_decision["recommendation"]["edt_window_10pct"]["N_max"],
            "pareto_alternative_N": selected_decision["recommendation"]["pareto_alternative"]["N"],
            "recommendation_digest": selected_decision["recommendation_digest"],
        },
    }
    (outdir / "xy8_robustness_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote XY8 resource-ablation and uncertainty artifacts to", outdir)


if __name__ == "__main__":
    main()
