#!/usr/bin/env python3
"""Generate a prospective seven-setting validation protocol for the NV XY8 model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import make_fig_xy8_qre as sweep
import make_xy8_extended_analyses as extended
import nvratemodel_api as pipe


ROOT = Path(__file__).resolve().parents[2]
NOMINAL_CALIBRATION = ROOT / "data" / "raw" / "nv" / "xy8_scenario_nominal.json"
MODEL_SOURCE = ROOT / "code" / "plotting" / "make_fig_xy8_qre.py"
MODEL_VERSION = "xy8-proxy-v1"
DEFAULT_UNCERTAINTY_SAMPLES = 400
DEFAULT_SEED = 20260826

# The two window-edge controls are resolved from the computed 10% window rather
# than hardcoded. A hardcoded edge silently stops being an edge as soon as the
# depth grid is refined, which would make the protocol test an interior depth
# while claiming to test a boundary.
DEPTH_CONTROLS = [
    ("low_depth", "Low depth", 8, "Show insufficient signal accumulation"),
    ("edt_lower_edge", "10% lower edge", "window_N_min", "Test the EDT-window boundary"),
    ("practical_choice", "Practical choice", 16, "Test the Pareto compromise"),
    ("predicted_optimum", "Predicted optimum", 20, "Primary prediction"),
    ("edt_upper_edge", "10% upper edge", "window_N_max", "Test the upper EDT boundary"),
    ("overdriven", "Overdriven", 64, "Demonstrate pulse-error accumulation"),
]


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def rows_digest(rows):
    predictive_rows = [
        {key: value for key, value in row.items() if not key.startswith("measured_") and key != "criterion_status"}
        for row in rows
    ]
    payload = json.dumps(predictive_rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def result_lookup(results):
    return {(float(tau), int(N)): float(eta) for tau, N, eta in results}


def select_off_resonance(results, meta, f_target, N=20, minimum_penalty=5.0):
    """Select the closest sampled timing control with at least the requested penalty."""
    primary_tau = 1.0 / (4.0 * float(f_target))
    candidates = []
    for tau, row_N, eta in results:
        if int(row_N) != int(N) or np.isclose(tau, primary_tau, rtol=0.0, atol=1e-15):
            continue
        relative = float(eta / meta["eta_best"])
        if relative >= float(minimum_penalty):
            candidates.append((abs(float(tau) - primary_tau), float(tau), float(eta)))
    if not candidates:
        raise RuntimeError("No sampled off-resonance control meets the requested minimum penalty.")
    _, tau, eta = min(candidates, key=lambda row: (row[0], row[1]))
    return tau, eta


def build_selected_controls(nominal, f_target):
    results, meta, _, _, taus, Ns = sweep.evaluate_calibration(nominal, f_target=f_target)
    values = result_lookup(results)
    primary_tau = 1.0 / (4.0 * float(f_target))
    if not any(np.isclose(primary_tau, tau, rtol=0.0, atol=1e-15) for tau in taus):
        raise RuntimeError("Canonical sweep does not contain the exact primary resonance.")

    window = meta["edt_window_10pct"]
    window_depths = {"window_N_min": int(window["N_min"]), "window_N_max": int(window["N_max"])}

    selected = []
    for control_id, label, depth_spec, purpose in DEPTH_CONTROLS:
        N = window_depths[depth_spec] if isinstance(depth_spec, str) else int(depth_spec)
        if N not in {int(value) for value in Ns}:
            raise RuntimeError(f"Canonical sweep does not contain required depth N={N}.")
        selected.append(
            {
                "control_id": control_id,
                "control": label,
                "tau_s": primary_tau,
                "N": N,
                "eta_proxy": values[(primary_tau, N)],
                "purpose": purpose,
            }
        )

    off_tau, off_eta = select_off_resonance(results, meta, f_target, N=20)
    selected.append(
        {
            "control_id": "off_resonance",
            "control": "Off resonance",
            "tau_s": off_tau,
            "N": 20,
            "eta_proxy": off_eta,
            "purpose": "Verify timing selectivity",
        }
    )
    return selected, results, meta


def uncertainty_intervals(nominal, selected, f_target, sample_count, seed):
    if sample_count < 1:
        raise ValueError("uncertainty sample count must be positive")
    sampled_values = {row["control_id"]: [] for row in selected}
    for sample in extended.sample_calibrations(nominal, sample_count, seed=seed):
        calibration = sample["calibration"]
        Rph_1, C_1, _, extra = pipe.resolve_readout_resources(
            dict(sweep.DEFAULT_MODEL_KWARGS),
            calibration=calibration,
            integration_time=300e-9,
            contrast_mode="fixed",
            pl_model="MEmodel",
        )
        pi_pulse_fidelity = pipe.resolve_pi_pulse_fidelity(extra["modeldict"], calibration=calibration)
        t2_params = pipe.resolve_t2_params(dict(sweep.DEFAULT_T2_PARAMS), calibration=calibration)
        overheads = pipe.resolve_overheads(dict(sweep.DEFAULT_OVERHEADS), calibration=calibration)
        neff = pipe.resolve_neff(sweep.DEFAULT_NEFF, calibration=calibration)
        for row in selected:
            eta = pipe.eta_ac(
                f_target,
                row["tau_s"],
                row["N"],
                Rph_1,
                C_1,
                neff,
                t2_params,
                overheads,
                piPulseFid=pi_pulse_fidelity,
            )
            sampled_values[row["control_id"]].append(float(eta))

    return {
        control_id: {
            "lower": float(np.quantile(values, 0.025)),
            "median": float(np.median(values)),
            "upper": float(np.quantile(values, 0.975)),
        }
        for control_id, values in sampled_values.items()
    }


def build_protocol_rows(nominal, f_target=200e3, uncertainty_samples=DEFAULT_UNCERTAINTY_SAMPLES, seed=DEFAULT_SEED):
    selected, canonical_results, meta = build_selected_controls(nominal, f_target)
    intervals = uncertainty_intervals(nominal, selected, f_target, uncertainty_samples, seed)
    expected_order = {
        row["control_id"]: rank
        for rank, row in enumerate(sorted(selected, key=lambda row: (row["eta_proxy"], row["control_id"])), start=1)
    }
    rows = []
    for row in selected:
        relative = float(row["eta_proxy"] / meta["eta_best"])
        interval = intervals[row["control_id"]]
        rows.append(
            {
                "control_id": row["control_id"],
                "control": row["control"],
                "purpose": row["purpose"],
                "f_target_Hz": float(f_target),
                "tau_s": row["tau_s"],
                "N": row["N"],
                "total_pi_pulses": 8 * row["N"],
                "sequence_duration_s": pipe.sequence_duration(row["tau_s"], row["N"]),
                "predicted_eta_proxy": row["eta_proxy"],
                "predicted_relative_to_best": relative,
                "predicted_rank": expected_order[row["control_id"]],
                "predicted_eta_95pct_lower": interval["lower"],
                "predicted_eta_median": interval["median"],
                "predicted_eta_95pct_upper": interval["upper"],
                "recommended_independent_repeats": 5 if row["control_id"] == "predicted_optimum" else 3,
                "calibration_digest": meta["calibration_digest"],
                "model_version": MODEL_VERSION,
                "model_source_digest": file_digest(MODEL_SOURCE),
                "measured_eta_proxy": "",
                "measured_standard_error": "",
                "measured_n_repeats": "",
                "measured_relative_to_best": "",
                "criterion_status": "",
                "measured_notes": "",
            }
        )
    return rows, canonical_results, meta


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_protocol(rows, meta, f_target, uncertainty_samples, seed):
    protocol_digest = rows_digest(rows)
    by_id = {row["control_id"]: row for row in rows}
    window_min = int(by_id["edt_lower_edge"]["N"])
    window_max = int(by_id["edt_upper_edge"]["N"])
    return {
        "protocol_id": f"xy8-prospective-{protocol_digest}",
        "status": "prospective_predictions_unmeasured",
        "slide_headline": "Seven control settings can test the predicted EDT window.",
        "f_target_Hz": float(f_target),
        "scenario": meta["scenario"],
        "scenario_status": meta["scenario_status"],
        "calibration_digest": meta["calibration_digest"],
        "model_version": MODEL_VERSION,
        "model_source": str(MODEL_SOURCE.relative_to(ROOT)),
        "model_source_digest": file_digest(MODEL_SOURCE),
        "protocol_digest": protocol_digest,
        "pulse_count_convention": "N_pi = 8 * N_blocks",
        "primary_resonance_tau_s": 1.0 / (4.0 * float(f_target)),
        "off_resonance_selection_rule": "Closest canonical-grid timing to the primary resonance with predicted eta/eta_best >= 5 at N=20; lower tau breaks ties.",
        "uncertainty": {
            "sample_count": int(uncertainty_samples),
            "seed": int(seed),
            "interval": "central 95% of the seeded calibration-uncertainty model",
            "sampling_model_source": "make_xy8_extended_analyses.sample_calibrations",
        },
        "validation_criteria": [
            {
                "criterion_id": "interior_pair",
                "statement": "N=16 and N=20 are statistically indistinguishable or ordered as predicted (eta_20 <= eta_16).",
                "status": "pending",
            },
            {
                "criterion_id": "depth_controls",
                "statement": "N=8 and N=64 perform worse than the N=16 and N=20 interior controls.",
                "status": "pending",
            },
            {
                "criterion_id": "window_contains_measured_best",
                "statement": (
                    f"The best measured resonant depth lies within N={window_min} "
                    f"through N={window_max}."
                ),
                "status": "pending",
            },
            {
                "criterion_id": "timing_selectivity",
                "statement": "The model-selected off-resonance N=20 control performs worse than resonant N=20.",
                "status": "pending",
            },
            {
                "criterion_id": "uncertainty_coverage",
                "statement": "Measured eta values lie within the model's prospective central 95% calibration intervals.",
                "status": "pending",
            },
        ],
        "measurement_columns": {
            "measured_eta_proxy": "blank until data collection",
            "measured_standard_error": "blank until data collection",
            "measured_n_repeats": "blank until data collection",
            "measured_relative_to_best": "blank until data collection",
            "criterion_status": "blank until the protocol is evaluated",
            "measured_notes": "blank until data collection",
        },
        "controls": rows,
    }


def plot_protocol(rows, out_png, out_pdf):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depth_rows = [row for row in rows if row["control_id"] != "off_resonance"]
    by_id = {row["control_id"]: row for row in rows}
    colors = [
        "#9d9d9d" if not row["control_id"].startswith("edt_") else "#54a24b"
        for row in depth_rows
    ]
    colors[2] = "#f58518"
    colors[3] = "#e45756"

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.0, 4.2), gridspec_kw={"width_ratios": [1.65, 0.9]})
    x = np.arange(len(depth_rows))
    relative = np.array([row["predicted_relative_to_best"] for row in depth_rows])
    bars = ax_a.bar(x, relative, color=colors, edgecolor="white", linewidth=0.7)
    ax_a.axhspan(1.0, 1.10, color="#54a24b", alpha=0.14, label="Within 10% of best")
    ax_a.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"N={row['N']}\n{row['total_pi_pulses']} pulses" for row in depth_rows])
    ax_a.set_ylabel("Predicted sensitivity penalty, eta / eta_best")
    ax_a.set_ylim(0.94, max(relative) * 1.13)
    ax_a.set_title("Six depths test the predicted interior optimum")
    ax_a.legend(frameon=False, loc="upper left", fontsize=8)
    for bar, value in zip(bars, relative):
        ax_a.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.2f}x", ha="center", fontsize=8)
    ax_a.text(-0.10, 1.04, "A", transform=ax_a.transAxes, fontsize=14, fontweight="bold")

    timing_rows = [by_id["predicted_optimum"], by_id["off_resonance"]]
    timing_relative = [row["predicted_relative_to_best"] for row in timing_rows]
    timing_bars = ax_b.bar(
        ["Resonant\n1.250 us", f"Off resonance\n{timing_rows[1]['tau_s'] * 1e6:.3f} us"],
        timing_relative,
        color=["#e45756", "#4c78a8"],
        edgecolor="white",
    )
    ax_b.set_ylabel("Predicted sensitivity penalty")
    ax_b.set_title("One timing control tests selectivity")
    ax_b.set_ylim(0.0, max(timing_relative) * 1.17)
    for bar, value in zip(timing_bars, timing_relative):
        ax_b.text(bar.get_x() + bar.get_width() / 2, value + max(timing_relative) * 0.035, f"{value:.2f}x", ha="center", fontsize=9)
    ax_b.text(-0.14, 1.04, "B", transform=ax_b.transAxes, fontsize=14, fontweight="bold")

    fig.suptitle("Seven control settings can test the predicted EDT window", fontsize=16, fontweight="bold", y=1.01)
    fig.text(
        0.5,
        0.005,
        "Prospective model predictions; measurement and pass/fail fields are intentionally blank.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description="Generate the prospective seven-setting NV XY8 validation protocol.")
    parser.add_argument("--calibration", default=str(NOMINAL_CALIBRATION))
    parser.add_argument("--outdir", default="outputs/reproduced_figures")
    parser.add_argument("--f-target", type=float, default=200e3)
    parser.add_argument("--uncertainty-samples", type=int, default=DEFAULT_UNCERTAINTY_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main():
    args = build_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    nominal = json.loads(Path(args.calibration).read_text(encoding="utf-8"))

    rows, _, meta = build_protocol_rows(
        nominal,
        f_target=args.f_target,
        uncertainty_samples=args.uncertainty_samples,
        seed=args.seed,
    )
    protocol = build_protocol(rows, meta, args.f_target, args.uncertainty_samples, args.seed)
    for row in rows:
        row["protocol_digest"] = protocol["protocol_digest"]

    write_rows(outdir / "xy8_prospective_validation.csv", rows)
    (outdir / "xy8_prospective_validation_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    plot_protocol(
        rows,
        outdir / "fig_xy8_prospective_validation.png",
        outdir / "fig_xy8_prospective_validation.pdf",
    )
    print("Wrote prospective XY8 validation protocol to", outdir)


if __name__ == "__main__":
    main()
