"""Hardware-to-control recommendations for the canonical NV XY8 sensing model."""

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PLOTTING_DIR = ROOT / "code" / "plotting"
if str(PLOTTING_DIR) not in sys.path:
    sys.path.insert(0, str(PLOTTING_DIR))

import make_fig_xy8_qre as sweep
import nvratemodel_api as pipe


MODEL_VERSION = "xy8-proxy-v1"
MODEL_SOURCE = ROOT / "code" / "plotting" / "make_fig_xy8_qre.py"
PHASE_FIDELITY_RANGE = (0.992, 1.0)
PHASE_T2_SCALE_RANGE = (0.5, 2.0)
DEFAULT_UNCERTAINTY_SEED = 20260826


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def payload_digest(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def control_region(N, boundary=False):
    if boundary:
        return "Unresolved"
    if int(N) <= 10:
        return "Shallow"
    if int(N) <= 20:
        return "Moderate"
    if int(N) <= 48:
        return "Deep"
    return "Very deep"


def calibration_with_hardware(base_calibration, pulse_fidelity, t2_s):
    if not 0.0 < float(pulse_fidelity) <= 1.0:
        raise ValueError("pulse fidelity must satisfy 0 < F_pi <= 1")
    if float(t2_s) <= 0.0:
        raise ValueError("T2 must be positive")
    calibration = deepcopy(base_calibration)
    calibration.setdefault("pulse", {})["piPulseFid"] = float(pulse_fidelity)
    calibration.setdefault("coherence", {})["T2_1"] = float(t2_s)
    return calibration


def resolved_context(calibration):
    Rph_1, C_1, _, extra = pipe.resolve_readout_resources(
        dict(sweep.DEFAULT_MODEL_KWARGS),
        calibration=calibration,
        integration_time=300e-9,
        contrast_mode="fixed",
        pl_model="MEmodel",
    )
    return {
        "Rph_1": Rph_1,
        "C_1": C_1,
        "piPulseFid": pipe.resolve_pi_pulse_fidelity(extra["modeldict"], calibration=calibration),
        "T2_params": pipe.resolve_t2_params(dict(sweep.DEFAULT_T2_PARAMS), calibration=calibration),
        "overheads": pipe.resolve_overheads(dict(sweep.DEFAULT_OVERHEADS), calibration=calibration),
        "Neff": pipe.resolve_neff(sweep.DEFAULT_NEFF, calibration=calibration),
    }


def evaluate_resonant_depths(calibration, frequency_hz):
    if float(frequency_hz) <= 0.0:
        raise ValueError("target frequency must be positive")
    tau = 1.0 / (4.0 * float(frequency_hz))
    _, Ns = sweep.build_sweep_grid(calibration, float(frequency_hz))
    context = resolved_context(calibration)
    eta = np.array(
        [
            pipe.eta_ac(
                frequency_hz,
                tau,
                int(N),
                context["Rph_1"],
                context["C_1"],
                context["Neff"],
                context["T2_params"],
                context["overheads"],
                piPulseFid=context["piPulseFid"],
            )
            for N in Ns
        ],
        dtype=float,
    )
    return tau, Ns, eta


def uncertainty_recommendation(
    base_calibration,
    frequency_hz,
    pulse_fidelity,
    t2_s,
    pulse_fidelity_sigma,
    t2_sigma_s,
    sample_count,
    seed,
):
    if int(sample_count) < 1:
        raise ValueError("uncertainty sample count must be positive")
    if float(pulse_fidelity_sigma) < 0.0 or float(t2_sigma_s) < 0.0:
        raise ValueError("uncertainty standard deviations cannot be negative")

    rng = np.random.default_rng(int(seed))
    fidelity_draws = np.clip(
        rng.normal(float(pulse_fidelity), float(pulse_fidelity_sigma), int(sample_count)),
        1e-9,
        1.0,
    )
    t2_draws = np.maximum(
        rng.normal(float(t2_s), float(t2_sigma_s), int(sample_count)),
        1e-12,
    )

    eta_surfaces = []
    best_depths = []
    reference_ns = None
    for fidelity, sampled_t2 in zip(fidelity_draws, t2_draws):
        calibration = calibration_with_hardware(base_calibration, fidelity, sampled_t2)
        _, Ns, eta = evaluate_resonant_depths(calibration, frequency_hz)
        if reference_ns is None:
            reference_ns = Ns.copy()
        elif not np.array_equal(reference_ns, Ns):
            raise RuntimeError("uncertainty samples produced inconsistent depth grids")
        eta_surfaces.append(eta)
        best_depths.append(int(Ns[int(np.argmin(eta))]))

    surfaces = np.stack(eta_surfaces)
    regret = surfaces / np.maximum(np.min(surfaces, axis=1)[:, None], 1e-30)
    robust_index = int(np.argmin(np.quantile(regret, 0.95, axis=0)))
    robust_N = int(reference_ns[robust_index])

    counts = {N: best_depths.count(N) for N in sorted(set(best_depths))}
    distribution = [
        {"N": int(N), "count": int(count), "probability": float(count / len(best_depths))}
        for N, count in counts.items()
    ]
    modal_N = min(counts, key=lambda N: (-counts[N], N))
    region_counts = {}
    for N, count in counts.items():
        region = control_region(N)
        region_counts[region] = region_counts.get(region, 0) + count
    region_distribution = [
        {"region": region, "count": count, "probability": float(count / len(best_depths))}
        for region, count in sorted(region_counts.items())
    ]

    probabilities = sorted((row["probability"] for row in distribution), reverse=True)
    multimodal = bool(len(probabilities) > 1 and probabilities[0] >= 0.15 and probabilities[1] >= 0.15)
    material_regions = [row for row in region_distribution if row["probability"] >= 0.05]
    spans_regions = len(material_regions) > 1
    modal_confidence = max(row["probability"] for row in distribution)
    additional_calibration = bool(multimodal or spans_regions or modal_confidence < 0.60)

    return {
        "sample_count": int(sample_count),
        "seed": int(seed),
        "pulse_fidelity_sigma": float(pulse_fidelity_sigma),
        "T2_sigma_s": float(t2_sigma_s),
        "modal_N": int(modal_N),
        "modal_probability": float(modal_confidence),
        "robust_p95_N": robust_N,
        "robust_p95_regret": float(np.quantile(regret[:, robust_index], 0.95)),
        "optimal_depth_distribution": distribution,
        "control_region_distribution": region_distribution,
        "multimodal_depth_recommendation": multimodal,
        "spans_multiple_control_regions": spans_regions,
        "additional_calibration_recommended": additional_calibration,
    }


def recommend_control(
    base_calibration,
    frequency_hz,
    pulse_fidelity,
    t2_s,
    *,
    pulse_fidelity_sigma=None,
    t2_sigma_s=None,
    uncertainty_samples=400,
    seed=DEFAULT_UNCERTAINTY_SEED,
):
    calibration = calibration_with_hardware(base_calibration, pulse_fidelity, t2_s)
    results, meta, _, _, taus, Ns = sweep.evaluate_calibration(calibration, f_target=float(frequency_hz))
    resonant_tau = 1.0 / (4.0 * float(frequency_hz))
    resonant_eta = {
        int(N): float(eta)
        for tau, N, eta in results
        if np.isclose(float(tau), resonant_tau, rtol=0.0, atol=1e-15)
    }
    if len(resonant_eta) != len(Ns):
        raise RuntimeError("canonical sweep does not contain every depth at the primary resonance")

    best_N = int(meta["N_best"])
    eta_best = float(meta["eta_best"])
    within_10pct = sorted(N for N, eta in resonant_eta.items() if eta <= 1.10 * eta_best + 1e-15)
    if best_N not in within_10pct:
        raise RuntimeError("reported 10% window does not contain the optimum")
    cheaper_candidates = [N for N in within_10pct if N < best_N]
    alternative_N = max(cheaper_candidates) if cheaper_candidates else best_N
    alternative_eta = resonant_eta[alternative_N]

    optimum_at_boundary = best_N in {int(Ns[0]), int(Ns[-1])}
    window_at_boundary = min(within_10pct) == int(Ns[0]) or max(within_10pct) == int(Ns[-1])
    base_t2 = float(base_calibration["coherence"]["T2_1"])
    t2_scale = float(t2_s) / base_t2
    outside_phase_range = not (
        PHASE_FIDELITY_RANGE[0] <= float(pulse_fidelity) <= PHASE_FIDELITY_RANGE[1]
        and PHASE_T2_SCALE_RANGE[0] <= t2_scale <= PHASE_T2_SCALE_RANGE[1]
    )

    warnings = []
    if outside_phase_range:
        warnings.append(
            {
                "code": "outside_phase_diagram_range",
                "severity": "warning",
                "message": "Inputs lie outside the validated phase-diagram slice; the model was evaluated directly.",
            }
        )
    if optimum_at_boundary:
        warnings.append(
            {
                "code": "optimum_at_sweep_boundary",
                "severity": "warning",
                "message": "The optimum reaches the tested depth boundary; expand the N sweep before treating it as resolved.",
            }
        )
    if window_at_boundary:
        warnings.append(
            {
                "code": "window_at_sweep_boundary",
                "severity": "warning",
                "message": "The 10% operating window reaches a tested depth boundary.",
            }
        )
    warnings.append(
        {
            "code": "nominal_resources_assumed",
            "severity": "info",
            "message": "Contrast, photon rate, overheads, coherence exponents, and other resources remain fixed to the base calibration.",
        }
    )

    uncertainty = None
    fidelity_sigma = 0.0 if pulse_fidelity_sigma is None else float(pulse_fidelity_sigma)
    coherence_sigma = 0.0 if t2_sigma_s is None else float(t2_sigma_s)
    if fidelity_sigma > 0.0 or coherence_sigma > 0.0:
        uncertainty = uncertainty_recommendation(
            base_calibration,
            frequency_hz,
            pulse_fidelity,
            t2_s,
            fidelity_sigma,
            coherence_sigma,
            uncertainty_samples,
            seed,
        )
        if uncertainty["multimodal_depth_recommendation"]:
            warnings.append(
                {
                    "code": "multimodal_depth_recommendation",
                    "severity": "warning",
                    "message": "Input uncertainty produces multiple materially probable optimal depths.",
                }
            )
        if uncertainty["spans_multiple_control_regions"]:
            warnings.append(
                {
                    "code": "uncertainty_spans_control_regions",
                    "severity": "warning",
                    "message": "Input uncertainty spans multiple control-depth regions.",
                }
            )
        if uncertainty["additional_calibration_recommended"]:
            warnings.append(
                {
                    "code": "additional_calibration_recommended",
                    "severity": "warning",
                    "message": "Additional hardware calibration is recommended before fixing the control depth.",
                }
            )

    payload = {
        "model_version": MODEL_VERSION,
        "model_source": str(MODEL_SOURCE.relative_to(ROOT)),
        "model_source_digest": file_digest(MODEL_SOURCE),
        "base_scenario": str(base_calibration.get("label", "unspecified")),
        "calibration_digest": meta["calibration_digest"],
        "inputs": {
            "frequency_Hz": float(frequency_hz),
            "pulse_fidelity": float(pulse_fidelity),
            "T2_s": float(t2_s),
            "T2_scale_relative_to_base": t2_scale,
        },
        "recommendation": {
            "tau_s": float(meta["tau_best"]),
            "resonant_tau_s": resonant_tau,
            "N": best_N,
            "total_pi_pulses": 8 * best_N,
            "predicted_eta_proxy": eta_best,
            "control_region": control_region(best_N, boundary=optimum_at_boundary),
            "edt_window_10pct": {
                "N_min": int(min(within_10pct)),
                "N_max": int(max(within_10pct)),
                "N_values": [int(N) for N in within_10pct],
                "touches_depth_boundary": bool(window_at_boundary),
            },
            "pareto_alternative": {
                "N": int(alternative_N),
                "total_pi_pulses": 8 * int(alternative_N),
                "predicted_eta_proxy": float(alternative_eta),
                "relative_to_best": float(alternative_eta / eta_best),
                "penalty_percent": float(100.0 * (alternative_eta / eta_best - 1.0)),
                "pulse_savings_percent": float(100.0 * (best_N - alternative_N) / best_N),
            },
            "boundary_status": "unresolved boundary optimum" if optimum_at_boundary else "resolved interior optimum",
        },
        "uncertainty": uncertainty,
        "warnings": warnings,
    }
    payload["recommendation_digest"] = payload_digest(payload)
    return payload


def format_recommendation(payload):
    recommendation = payload["recommendation"]
    window = recommendation["edt_window_10pct"]
    alternative = recommendation["pareto_alternative"]
    lines = [
        f"Recommended spacing: {recommendation['tau_s'] * 1e6:.3f} us",
        f"Performance optimum: N={recommendation['N']} ({recommendation['total_pi_pulses']} pi pulses)",
        f"10% operating window: N={window['N_min']}-{window['N_max']}",
        f"Pareto alternative: N={alternative['N']}",
        f"Alternative penalty: {alternative['penalty_percent']:.2f}%",
        f"Predicted sensitivity proxy: {recommendation['predicted_eta_proxy']:.6g}",
        f"Boundary status: {recommendation['boundary_status']}",
    ]
    if payload["uncertainty"] is not None:
        uncertainty = payload["uncertainty"]
        lines.extend(
            [
                f"Uncertainty modal choice: N={uncertainty['modal_N']} ({100 * uncertainty['modal_probability']:.1f}%)",
                f"Robust p95 choice: N={uncertainty['robust_p95_N']}",
                "Additional calibration recommended: "
                + ("yes" if uncertainty["additional_calibration_recommended"] else "no"),
            ]
        )
    warning_codes = [warning["code"] for warning in payload["warnings"] if warning["severity"] == "warning"]
    if warning_codes:
        lines.append("Warnings: " + ", ".join(warning_codes))
    return "\n".join(lines)
