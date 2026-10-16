#!/usr/bin/env python3
"""Generate frequency, uncertainty, robust-control, and sequence-comparison analyses."""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

import make_fig_xy8_qre as sweep


ROOT = Path(__file__).resolve().parents[2]
NOMINAL_CALIBRATION = ROOT / "data" / "raw" / "nv" / "xy8_scenario_nominal.json"
DEFAULT_MC_SEED = 20260826
DEFAULT_MC_SAMPLES = 400
SEQUENCE_FAMILIES = {"XY8": 8, "XY4": 4, "CPMG": 1}


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def result_surface(results, taus, Ns):
    values = {(float(tau), int(N)): float(eta) for tau, N, eta in results}
    return np.array([[values[(float(tau), int(N))] for tau in taus] for N in Ns], dtype=float)


def run_frequency_sweep(nominal, frequencies):
    rows = []
    for frequency in frequencies:
        _, meta, _, _, _, _ = sweep.evaluate_calibration(nominal, f_target=float(frequency))
        window = meta["edt_window_10pct"]
        theoretical_tau = 1.0 / (4.0 * float(frequency))
        rows.append(
            {
                "f_target_Hz": float(frequency),
                "tau_theory_s": theoretical_tau,
                "tau_best_s": meta["tau_best"],
                "tau_ratio_to_primary_resonance": meta["tau_best"] / theoretical_tau,
                "N_best": meta["N_best"],
                "eta_best": meta["eta_best"],
                "edt_window_N_min": window["N_min"],
                "edt_window_N_max": window["N_max"],
                "edt_window_tau_min_s": window["tau_min_s"],
                "edt_window_tau_max_s": window["tau_max_s"],
                "edt_window_touches_boundary": int(window["touches_tau_boundary"] or window["touches_N_boundary"]),
            }
        )
    return rows


def run_pareto_analysis(nominal, f_target):
    """Return the best spacing at each depth and flag resource-efficient controls."""
    results, meta, _, _, _, Ns = sweep.evaluate_calibration(nominal, f_target=f_target)
    best_at_depth = {}
    for tau, N, eta in results:
        candidate = best_at_depth.get(int(N))
        if candidate is None or eta < candidate[1]:
            best_at_depth[int(N)] = (float(tau), float(eta))

    best_pulses = 8 * int(meta["N_best"])
    best_eta = float(meta["eta_best"])
    lowest_eta_at_lower_cost = np.inf
    rows = []
    for N in sorted(int(value) for value in Ns):
        tau, eta = best_at_depth[N]
        pareto_optimal = eta < lowest_eta_at_lower_cost - 1e-15
        if pareto_optimal:
            lowest_eta_at_lower_cost = eta
        relative = eta / best_eta
        rows.append(
            {
                "tau_best_at_depth_s": tau,
                "N": N,
                "total_pi_pulses": 8 * N,
                "eta_proxy": eta,
                "relative_to_best": relative,
                "sensitivity_penalty_pct": 100.0 * (relative - 1.0),
                "pulse_savings_vs_best_pct": 100.0 * (best_pulses - 8 * N) / best_pulses,
                "within_10pct": int(relative <= 1.10 + 1e-12),
                "pareto_optimal": int(pareto_optimal),
                "is_global_best": int(N == meta["N_best"] and abs(tau - meta["tau_best"]) <= 1e-15),
                "is_heuristic_N16": int(N == 16),
            }
        )
    return rows


def sample_calibrations(nominal, sample_count, seed=DEFAULT_MC_SEED):
    rng = np.random.default_rng(seed)
    for sample_id in range(sample_count):
        fidelity = float(np.clip(rng.normal(nominal["pulse"]["piPulseFid"], 0.0012), 0.990, 1.0))
        t2_scale = float(np.exp(rng.normal(0.0, 0.20)))
        alpha = float(np.clip(rng.normal(nominal["coherence"]["alpha"], 0.07), 0.30, 0.90))
        contrast_scale = float(np.exp(rng.normal(0.0, 0.18)))
        photon_rate_scale = float(np.exp(rng.normal(0.0, 0.25)))
        overhead_scale = float(np.exp(rng.normal(0.0, 0.20)))

        calibration = deepcopy(nominal)
        calibration["label"] = f"nominal_mc_{sample_id:04d}"
        calibration["pulse"]["piPulseFid"] = fidelity
        calibration["coherence"]["T2_1"] = float(nominal["coherence"]["T2_1"] * t2_scale)
        calibration["coherence"]["alpha"] = alpha
        calibration["readout"]["C_1"] = float(nominal["readout"]["C_1"] * contrast_scale)
        calibration["readout"]["effective_detected_rate_cps"] = float(
            nominal["readout"]["effective_detected_rate_cps"] * photon_rate_scale
        )
        for key in ["t_init", "t_read", "t_dead"]:
            calibration["overheads"][key] = float(nominal["overheads"][key] * overhead_scale)

        yield {
            "sample_id": sample_id,
            "piPulseFid": fidelity,
            "T2_scale": t2_scale,
            "T2_alpha": alpha,
            "contrast_scale": contrast_scale,
            "photon_rate_scale": photon_rate_scale,
            "overhead_scale": overhead_scale,
            "calibration": calibration,
        }


def run_monte_carlo(nominal, f_target, sample_count=DEFAULT_MC_SAMPLES, seed=DEFAULT_MC_SEED):
    sample_rows = []
    surfaces = []
    reference_taus = None
    reference_ns = None
    for sample in sample_calibrations(nominal, sample_count, seed=seed):
        results, meta, _, _, taus, Ns = sweep.evaluate_calibration(sample["calibration"], f_target=f_target)
        if reference_taus is None:
            reference_taus = taus.copy()
            reference_ns = Ns.copy()
        elif not np.array_equal(reference_taus, taus) or not np.array_equal(reference_ns, Ns):
            raise RuntimeError("Monte Carlo calibrations produced inconsistent control grids.")
        surfaces.append(result_surface(results, taus, Ns))
        window = meta["edt_window_10pct"]
        sample_rows.append(
            {
                "sample_id": sample["sample_id"],
                "piPulseFid": sample["piPulseFid"],
                "T2_scale": sample["T2_scale"],
                "T2_alpha": sample["T2_alpha"],
                "contrast_scale": sample["contrast_scale"],
                "photon_rate_scale": sample["photon_rate_scale"],
                "overhead_scale": sample["overhead_scale"],
                "tau_best_s": meta["tau_best"],
                "N_best": meta["N_best"],
                "eta_best": meta["eta_best"],
                "edt_window_N_min": window["N_min"],
                "edt_window_N_max": window["N_max"],
                "edt_window_touches_boundary": int(window["touches_tau_boundary"] or window["touches_N_boundary"]),
            }
        )
    return sample_rows, np.stack(surfaces), reference_taus, reference_ns


def optimum_distribution(sample_rows):
    counts = {}
    for row in sample_rows:
        counts[row["N_best"]] = counts.get(row["N_best"], 0) + 1
    total = len(sample_rows)
    return [
        {"N_best": int(N), "count": count, "probability": count / total}
        for N, count in sorted(counts.items())
    ]


def select_robust_controls(nominal, f_target, surfaces, taus, Ns):
    regret = surfaces / np.maximum(np.min(surfaces, axis=(1, 2))[:, None, None], 1e-30)
    q95 = np.quantile(regret, 0.95, axis=0)
    mean_regret = np.mean(regret, axis=0)

    nominal_results, nominal_meta, _, _, _, _ = sweep.evaluate_calibration(nominal, f_target=f_target)
    nominal_surface = result_surface(nominal_results, taus, Ns)
    nominal_index = np.unravel_index(np.argmin(nominal_surface), nominal_surface.shape)
    robust_index = np.unravel_index(np.argmin(q95), q95.shape)
    expected_index = np.unravel_index(np.argmin(mean_regret), mean_regret.shape)

    candidates = {
        "Nominal optimum": nominal_index,
        "Robust p95 optimum": robust_index,
        "Minimum-mean-regret": expected_index,
    }
    rows = []
    distributions = {}
    for label, (i, j) in candidates.items():
        penalties = regret[:, i, j]
        distributions[label] = penalties
        rows.append(
            {
                "control": label,
                "tau_s": float(taus[j]),
                "N": int(Ns[i]),
                "median_regret": float(np.median(penalties)),
                "mean_regret": float(np.mean(penalties)),
                "p90_regret": float(np.quantile(penalties, 0.90)),
                "p95_regret": float(np.quantile(penalties, 0.95)),
                "worst_regret": float(np.max(penalties)),
                "nominal_eta": float(nominal_surface[i, j]),
                "nominal_relative_to_best": float(nominal_surface[i, j] / nominal_meta["eta_best"]),
            }
        )
    return rows, distributions


def switching_gain_for_total_pulses(frequency, tau, total_pulses):
    total_pulses = max(int(total_pulses), 1)
    tau = max(float(tau), 1e-12)
    total_time = 2.0 * total_pulses * tau
    pulse_index = np.arange(1, total_pulses + 1, dtype=float)
    pulse_times = (2.0 * pulse_index - 1.0) * tau
    edges = np.concatenate(([0.0], pulse_times, [total_time]))
    durations = np.diff(edges)
    signs = (-1.0) ** np.arange(durations.size, dtype=float)
    omega = 2.0 * np.pi * float(frequency)
    phases = np.exp(1j * omega * edges)
    response = abs(np.sum(signs * (phases[1:] - phases[:-1]) / (1j * omega)))
    return float(response / total_time)


def eta_for_total_pulses(nominal, frequency, tau, total_pulses):
    total_pulses = max(int(total_pulses), 1)
    total_time = 2.0 * total_pulses * float(tau)
    equivalent_xy8_depth = max(total_pulses / 8.0, 1.0)
    coherence = nominal["coherence"]
    t2_eff = (
        coherence["T2_1"]
        * equivalent_xy8_depth ** coherence["alpha"]
        * (float(tau) / coherence["tau0"]) ** coherence.get("beta", 0.0)
    )
    env_factor = float(np.exp(-((total_time / t2_eff) ** coherence.get("p", 2.0))))
    pulse_factor = float(nominal["pulse"]["piPulseFid"] ** total_pulses)
    gain = switching_gain_for_total_pulses(frequency, tau, total_pulses)
    overhead = nominal["overheads"]
    cycle_time = total_time + overhead["t_init"] + overhead["t_read"] + overhead.get("t_dead", 0.0)
    rate = nominal["readout"]["effective_detected_rate_cps"]
    contrast = nominal["readout"]["C_1"]
    return float(np.sqrt(cycle_time / rate) / max(gain * contrast * env_factor * pulse_factor * total_time, 1e-30))


def run_sequence_comparison(nominal, f_target):
    tau = 1.0 / (4.0 * f_target)
    total_pulse_grid = np.array([8, 16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 1024])
    eta_by_pulses = {int(total): eta_for_total_pulses(nominal, f_target, tau, int(total)) for total in total_pulse_grid}
    best_total = min(eta_by_pulses, key=eta_by_pulses.get)
    rows = []
    for family, pulses_per_block in SEQUENCE_FAMILIES.items():
        for total_pulses in total_pulse_grid:
            rows.append(
                {
                    "sequence": family,
                    "model_scope": "instantaneous pulses; scalar per-pulse fidelity; phase cycling unresolved",
                    "tau_s": tau,
                    "pulses_per_block": pulses_per_block,
                    "block_count": int(total_pulses // pulses_per_block),
                    "total_pi_pulses": int(total_pulses),
                    "eta_proxy": eta_by_pulses[int(total_pulses)],
                    "relative_to_best": eta_by_pulses[int(total_pulses)] / eta_by_pulses[best_total],
                    "is_best_total_pulse_count": int(total_pulses == best_total),
                }
            )
    return rows


def plot_frequency(rows, out_png, out_pdf):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frequency_khz = np.array([row["f_target_Hz"] for row in rows]) / 1e3
    tau_best_us = np.array([row["tau_best_s"] for row in rows]) * 1e6
    tau_theory_us = np.array([row["tau_theory_s"] for row in rows]) * 1e6
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax_a.plot(frequency_khz, tau_best_us, marker="o", label="Modeled optimum")
    ax_a.plot(frequency_khz, tau_theory_us, linestyle="--", color="black", label="1 / (4f)")
    ax_a.set_xlabel("Target frequency (kHz)")
    ax_a.set_ylabel("Best pulse spacing (us)")
    ax_a.set_title("The optimum tracks the primary resonance")
    ax_a.legend(frameon=False)
    ax_a.text(-0.12, 1.04, "A", transform=ax_a.transAxes, fontsize=14, fontweight="bold")

    ax_b.plot(frequency_khz, [row["N_best"] for row in rows], marker="o", color="#f58518", label="Best N")
    ax_b.fill_between(
        frequency_khz,
        [row["edt_window_N_min"] for row in rows],
        [row["edt_window_N_max"] for row in rows],
        color="#f58518",
        alpha=0.22,
        label="10% EDT window",
    )
    ax_b.set_xlabel("Target frequency (kHz)")
    ax_b.set_ylabel("XY8 block count, N")
    ax_b.set_title("The acceptable depth window changes with band")
    ax_b.legend(frameon=False)
    ax_b.text(-0.12, 1.04, "B", transform=ax_b.transAxes, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(rows, out_png, out_pdf):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pareto = [row for row in rows if row["pareto_optimal"]]
    dominated = [row for row in rows if not row["pareto_optimal"]]
    within_window = [row for row in pareto if row["within_10pct"]]
    by_N = {row["N"]: row for row in rows}

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax_a.axhspan(1.0, 1.10, color="#54a24b", alpha=0.14, label="Within 10% of best")
    ax_a.plot(
        [row["total_pi_pulses"] for row in pareto],
        [row["relative_to_best"] for row in pareto],
        color="#4c78a8",
        linewidth=2.2,
        marker="o",
        markersize=4,
        label="Pareto frontier",
    )
    ax_a.scatter(
        [row["total_pi_pulses"] for row in dominated],
        [row["relative_to_best"] for row in dominated],
        color="#bab0ac",
        s=28,
        alpha=0.8,
        label="Dominated controls",
    )
    for N, color, marker, label in [
        (10, "#54a24b", "D", "Fewest pulses within 10%"),
        (16, "#f58518", "o", "Heuristic N=16"),
        (20, "#e45756", "*", "Best N=20"),
    ]:
        row = by_N[N]
        ax_a.scatter(
            row["total_pi_pulses"],
            row["relative_to_best"],
            color=color,
            marker=marker,
            s=105 if marker == "*" else 64,
            edgecolor="black",
            linewidth=0.7,
            zorder=5,
            label=label,
        )
    ax_a.set_xscale("log", base=2)
    ax_a.set_yscale("log")
    ax_a.set_xlabel("Total pi pulses, 8N")
    ax_a.set_ylabel("Sensitivity penalty, eta / eta_best")
    ax_a.set_title("More control helps only up to N=20")
    ax_a.legend(frameon=False, fontsize=8, loc="upper left")
    ax_a.text(-0.12, 1.04, "A", transform=ax_a.transAxes, fontsize=14, fontweight="bold")

    ax_b.plot(
        [row["pulse_savings_vs_best_pct"] for row in within_window],
        [row["sensitivity_penalty_pct"] for row in within_window],
        color="#4c78a8",
        linewidth=2.2,
        marker="o",
    )
    for N, color in [(10, "#54a24b"), (16, "#f58518"), (20, "#e45756")]:
        row = by_N[N]
        ax_b.scatter(
            row["pulse_savings_vs_best_pct"],
            row["sensitivity_penalty_pct"],
            color=color,
            edgecolor="black",
            linewidth=0.7,
            s=70,
            zorder=5,
        )
        offset = {10: (-8, -18), 16: (0, 9), 20: (7, 7)}[N]
        ax_b.annotate(
            f"N={N}",
            (row["pulse_savings_vs_best_pct"], row["sensitivity_penalty_pct"]),
            xytext=offset,
            textcoords="offset points",
            ha={10: "right", 16: "center", 20: "left"}[N],
            fontsize=9,
        )
    heuristic = by_N[16]
    ax_b.text(
        0.97,
        0.72,
        f"N=16 uses {heuristic['pulse_savings_vs_best_pct']:.0f}% fewer pulses\n"
        f"for a {heuristic['sensitivity_penalty_pct']:.2f}% penalty",
        transform=ax_b.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.92, "pad": 5},
    )
    ax_b.set_xlim(-2, max(row["pulse_savings_vs_best_pct"] for row in within_window) + 4)
    ax_b.set_ylim(-0.4, 10.8)
    ax_b.set_xlabel("Pulse reduction relative to N=20 (%)")
    ax_b.set_ylabel("Sensitivity penalty (%)")
    ax_b.set_title("Twenty percent fewer pulses costs about one percent")
    ax_b.text(-0.12, 1.04, "B", transform=ax_b.transAxes, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_monte_carlo(sample_rows, distribution, out_png, out_pdf):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax_a.bar([str(row["N_best"]) for row in distribution], [row["probability"] for row in distribution], color="#4c78a8")
    ax_a.set_xlabel("Optimal XY8 block count, N")
    ax_a.set_ylabel("Probability")
    ax_a.set_title("Calibration uncertainty broadens the optimum")
    ax_a.tick_params(axis="x", rotation=45)
    ax_a.text(-0.12, 1.04, "A", transform=ax_a.transAxes, fontsize=14, fontweight="bold")

    scatter = ax_b.scatter(
        [row["piPulseFid"] for row in sample_rows],
        [row["T2_scale"] for row in sample_rows],
        c=[row["N_best"] for row in sample_rows],
        cmap="viridis",
        s=18,
        alpha=0.75,
    )
    colorbar = fig.colorbar(scatter, ax=ax_b)
    colorbar.set_label("Best N")
    ax_b.set_xlabel("Per-pulse fidelity")
    ax_b.set_ylabel("T2 scale relative to nominal")
    ax_b.set_title("Calibration uncertainty maps to control depth")
    ax_b.text(-0.12, 1.04, "B", transform=ax_b.transAxes, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_robust(rows, distributions, out_png, out_pdf):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#4c78a8", "#e45756", "#54a24b"]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for (label, penalties), color in zip(distributions.items(), colors):
        values = np.sort(penalties)
        cumulative = np.arange(1, len(values) + 1) / len(values)
        ax_a.plot(values, cumulative, label=label, color=color)
    ax_a.set_xlabel("Regret relative to sample-specific optimum")
    ax_a.set_ylabel("Cumulative probability")
    ax_a.set_title("Robustness evaluated by sample-wise regret")
    if len({(row["tau_s"], row["N"]) for row in rows}) == 1:
        ax_a.text(
            0.53,
            0.40,
            "Curves overlap: all criteria select the same control",
            transform=ax_a.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 3},
        )
    ax_a.legend(frameon=False, fontsize=8)
    ax_a.text(-0.12, 1.04, "A", transform=ax_a.transAxes, fontsize=14, fontweight="bold")

    bars = ax_b.bar([row["control"] for row in rows], [row["p95_regret"] for row in rows], color=colors)
    ax_b.set_ylabel("95th-percentile regret")
    if len({(row["tau_s"], row["N"]) for row in rows}) == 1:
        ax_b.set_title("Nominal control is also the robust choice")
    else:
        ax_b.set_title("Robust versus nominal control")
    ax_b.tick_params(axis="x", rotation=24)
    for label in ax_b.get_xticklabels():
        label.set_ha("right")
    maximum = max(row["p95_regret"] for row in rows)
    ax_b.set_ylim(0.0, maximum * 1.22)
    for bar, row in zip(bars, rows):
        ax_b.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + maximum * 0.025,
            f"N={row['N']}\n{row['tau_s'] * 1e6:.2f} us",
            ha="center",
            fontsize=8,
        )
    ax_b.text(-0.12, 1.04, "B", transform=ax_b.transAxes, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_sequence(rows, out_png, out_pdf):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    styles = {"XY8": "-", "XY4": "--", "CPMG": ":"}
    best_rows = []
    for family in SEQUENCE_FAMILIES:
        family_rows = [row for row in rows if row["sequence"] == family]
        ax_a.plot(
            [row["total_pi_pulses"] for row in family_rows],
            [row["relative_to_best"] for row in family_rows],
            linestyle=styles[family],
            linewidth=2.0,
            label=family,
        )
        best_rows.append(next(row for row in family_rows if row["is_best_total_pulse_count"]))
    ax_a.set_xscale("log", base=2)
    ax_a.set_yscale("log")
    ax_a.set_xlabel("Total pi pulses")
    ax_a.set_ylabel("Sensitivity penalty")
    ax_a.set_title("Equal-pulse predictions coincide in this model")
    ax_a.legend(frameon=False)
    ax_a.text(-0.12, 1.04, "A", transform=ax_a.transAxes, fontsize=14, fontweight="bold")

    bars = ax_b.bar([row["sequence"] for row in best_rows], [row["block_count"] for row in best_rows], color=["#4c78a8", "#f58518", "#54a24b"])
    ax_b.set_ylabel("Blocks at the same best pulse count")
    ax_b.set_title("Block count is not a fair cross-sequence cost")
    maximum = max(row["block_count"] for row in best_rows)
    ax_b.set_ylim(0.0, maximum * 1.20)
    for bar, row in zip(bars, best_rows):
        ax_b.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + maximum * 0.03,
            f"{row['total_pi_pulses']} pulses\n{row['block_count']} blocks",
            ha="center",
            fontsize=8,
        )
    ax_b.text(-0.12, 1.04, "B", transform=ax_b.transAxes, fontsize=14, fontweight="bold")
    fig.text(0.5, 0.005, "A phase-resolved control-error model is required to rank XY8, XY4, and CPMG.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description="Generate extended NV XY8 robustness analyses.")
    parser.add_argument("--calibration", default=str(NOMINAL_CALIBRATION))
    parser.add_argument("--outdir", default="outputs/reproduced_figures")
    parser.add_argument("--f-target", type=float, default=200e3)
    parser.add_argument("--mc-samples", type=int, default=DEFAULT_MC_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_MC_SEED)
    return parser


def main():
    args = build_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    nominal = json.loads(Path(args.calibration).read_text(encoding="utf-8"))

    pareto_rows = run_pareto_analysis(nominal, args.f_target)
    write_rows(outdir / "xy8_pareto_frontier.csv", pareto_rows)
    plot_pareto(
        pareto_rows,
        outdir / "fig_xy8_pareto_frontier.png",
        outdir / "fig_xy8_pareto_frontier.pdf",
    )

    frequencies = np.linspace(50e3, 500e3, 19)
    frequency_rows = run_frequency_sweep(nominal, frequencies)
    write_rows(outdir / "xy8_frequency_sweep.csv", frequency_rows)
    plot_frequency(frequency_rows, outdir / "fig_xy8_frequency_sweep.png", outdir / "fig_xy8_frequency_sweep.pdf")

    sample_rows, surfaces, taus, Ns = run_monte_carlo(
        nominal,
        args.f_target,
        sample_count=args.mc_samples,
        seed=args.seed,
    )
    distribution = optimum_distribution(sample_rows)
    write_rows(outdir / "xy8_monte_carlo_samples.csv", sample_rows)
    write_rows(outdir / "xy8_monte_carlo_optimum_distribution.csv", distribution)
    plot_monte_carlo(
        sample_rows,
        distribution,
        outdir / "fig_xy8_monte_carlo_uncertainty.png",
        outdir / "fig_xy8_monte_carlo_uncertainty.pdf",
    )

    robust_rows, regret_distributions = select_robust_controls(nominal, args.f_target, surfaces, taus, Ns)
    write_rows(outdir / "xy8_robust_control_comparison.csv", robust_rows)
    plot_robust(
        robust_rows,
        regret_distributions,
        outdir / "fig_xy8_robust_vs_nominal.png",
        outdir / "fig_xy8_robust_vs_nominal.pdf",
    )

    sequence_rows = run_sequence_comparison(nominal, args.f_target)
    write_rows(outdir / "xy8_sequence_comparison.csv", sequence_rows)
    plot_sequence(
        sequence_rows,
        outdir / "fig_xy8_sequence_comparison.png",
        outdir / "fig_xy8_sequence_comparison.pdf",
    )

    summary = {
        "f_target_Hz": float(args.f_target),
        "mc_samples": int(args.mc_samples),
        "mc_seed": int(args.seed),
        "frequency_range_Hz": [float(frequencies[0]), float(frequencies[-1])],
        "pareto": {
            "definition": "Minimize sensitivity proxy and total pi-pulse count simultaneously.",
            "pareto_depths": [row["N"] for row in pareto_rows if row["pareto_optimal"]],
            "fewest_pulses_within_10pct": next(
                row for row in pareto_rows if row["pareto_optimal"] and row["within_10pct"]
            ),
            "heuristic_N16": next(row for row in pareto_rows if row["N"] == 16),
            "performance_optimum": next(row for row in pareto_rows if row["is_global_best"]),
        },
        "uncertainty_model": {
            "piPulseFid": "truncated normal, sigma=0.0012, range=[0.990, 1.000]",
            "T2_scale": "log-normal, log-sigma=0.20",
            "T2_alpha": "truncated normal, sigma=0.07, range=[0.30, 0.90]",
            "contrast_scale": "log-normal, log-sigma=0.18",
            "photon_rate_scale": "log-normal, log-sigma=0.25",
            "overhead_scale": "log-normal, log-sigma=0.20",
        },
        "optimal_depth_distribution": distribution,
        "robust_controls": robust_rows,
        "sequence_comparison_scope": "Instantaneous-pulse scalar-fidelity baseline; phase cycling is unresolved.",
    }
    (outdir / "xy8_extended_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote extended XY8 analyses to", outdir)


if __name__ == "__main__":
    main()
