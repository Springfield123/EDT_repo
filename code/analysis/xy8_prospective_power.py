"""Prospective power calculations for the frozen seven-setting XY8 protocol.

The measurement model treats each independent batch as one approximately
Gaussian field estimate.  The sensitivity proxy is estimated from the sample
standard deviation of those batches, so its sampling distribution is an exact
scaled chi distribution under that assumption.  This isolates measurement
precision from the existing calibration-uncertainty calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.special import gammaln
from scipy.stats import f as f_distribution


REQUIRED_CONTROL_IDS = (
    "low_depth",
    "edt_lower_edge",
    "practical_choice",
    "predicted_optimum",
    "edt_upper_edge",
    "overdriven",
    "off_resonance",
)

DEFAULT_BATCH_GRID = (
    3,
    5,
    8,
    10,
    16,
    25,
    40,
    63,
    100,
    160,
    250,
    400,
    630,
    1000,
    1600,
    2500,
    4000,
    6300,
    10000,
    16000,
    25000,
    40000,
)

METRIC_DESCRIPTIONS = {
    "exact_optimum": "N=20 is selected as the best of the six resonant controls",
    "operating_window": "the best measured resonant control lies inside the predicted 10% window",
    "depth_controls": "N=8 and N=64 both perform worse than N=16 and N=20",
    "timing_selectivity": "the off-resonance control performs worse than resonant N=20",
    "n20_beats_n8": "N=20 performs better than N=8",
    "n20_beats_n64": "N=20 performs better than N=64",
    "n20_beats_n16": "the measured ordering identifies the 0.97% advantage of N=20 over N=16",
    "n16_n20_equivalent_10pct": "a two-sided 90% confidence interval places N=16 and N=20 within 10%",
}

METRIC_CONTROL_COUNTS = {
    "exact_optimum": 6,
    "operating_window": 6,
    "depth_controls": 4,
    "timing_selectivity": 2,
    "n20_beats_n8": 2,
    "n20_beats_n64": 2,
    "n20_beats_n16": 2,
    "n16_n20_equivalent_10pct": 2,
}


@dataclass(frozen=True)
class ProtocolData:
    protocol_id: str
    protocol_digest: str
    calibration_digest: str
    controls: tuple[dict, ...]
    window_min: int
    window_max: int

    @property
    def by_id(self) -> dict[str, dict]:
        return {row["control_id"]: row for row in self.controls}


def load_protocol(path: str | Path) -> ProtocolData:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_protocol(payload)


def validate_protocol(payload: dict) -> ProtocolData:
    if payload.get("status") != "prospective_predictions_unmeasured":
        raise ValueError("Power analysis requires an unmeasured prospective protocol.")
    controls = payload.get("controls", [])
    by_id = {row.get("control_id"): row for row in controls}
    missing = sorted(set(REQUIRED_CONTROL_IDS) - set(by_id))
    if missing:
        raise ValueError(f"Prospective protocol is missing controls: {missing}")
    if len(by_id) != len(controls):
        raise ValueError("Prospective control identifiers must be unique.")
    for row in controls:
        if row.get("measured_eta_proxy") not in {"", None}:
            raise ValueError("Power analysis must be frozen before measurements are entered.")
        eta = float(row["predicted_eta_proxy"])
        if not np.isfinite(eta) or eta <= 0.0:
            raise ValueError("Every predicted sensitivity proxy must be positive and finite.")
        if int(row["total_pi_pulses"]) != 8 * int(row["N"]):
            raise ValueError("Prospective protocol violates N_pi = 8N.")
    window_min = int(by_id["edt_lower_edge"]["N"])
    window_max = int(by_id["edt_upper_edge"]["N"])
    if not window_min <= int(by_id["predicted_optimum"]["N"]) <= window_max:
        raise ValueError("The predicted optimum is outside the declared operating window.")
    return ProtocolData(
        protocol_id=str(payload["protocol_id"]),
        protocol_digest=str(payload["protocol_digest"]),
        calibration_digest=str(payload["calibration_digest"]),
        controls=tuple(dict(row) for row in controls),
        window_min=window_min,
        window_max=window_max,
    )


def _normalize_allocation(protocol: ProtocolData, allocation) -> dict[str, int]:
    if isinstance(allocation, (int, np.integer)):
        allocation = {control_id: int(allocation) for control_id in REQUIRED_CONTROL_IDS}
    else:
        allocation = {str(key): int(value) for key, value in allocation.items()}
    missing = sorted(set(REQUIRED_CONTROL_IDS) - set(allocation))
    if missing:
        raise ValueError(f"Batch allocation is missing controls: {missing}")
    if any(value < 3 for value in allocation.values()):
        raise ValueError("At least three independent batches are required per control.")
    return {row["control_id"]: allocation[row["control_id"]] for row in protocol.controls}


def _equivalence_decision(
    raw_eta_16: np.ndarray,
    raw_eta_20: np.ndarray,
    df_16: int,
    df_20: int,
    margin: float,
    confidence: float,
) -> np.ndarray:
    alpha = 1.0 - float(confidence)
    ratio_squared = np.square(raw_eta_16 / raw_eta_20)
    lower_variance = ratio_squared / f_distribution.ppf(1.0 - alpha / 2.0, df_16, df_20)
    upper_variance = ratio_squared / f_distribution.ppf(alpha / 2.0, df_16, df_20)
    lower = np.sqrt(lower_variance)
    upper = np.sqrt(upper_variance)
    return (lower >= 1.0 / margin) & (upper <= margin)


def _sample_sd_bias_factor(degrees_of_freedom: int) -> float:
    """Return E[s]/sigma for a Gaussian sample standard deviation."""
    df = int(degrees_of_freedom)
    return float(
        np.sqrt(2.0 / df)
        * np.exp(gammaln((df + 1.0) / 2.0) - gammaln(df / 2.0))
    )


def simulate_decision_power(
    protocol: ProtocolData,
    allocation,
    *,
    trials: int = 50000,
    seed: int = 20260915,
    equivalence_margin: float = 1.10,
    equivalence_confidence: float = 0.90,
) -> dict[str, dict]:
    if int(trials) < 100:
        raise ValueError("At least 100 Monte Carlo trials are required.")
    if float(equivalence_margin) <= 1.0:
        raise ValueError("Equivalence margin must exceed one.")
    if not 0.0 < float(equivalence_confidence) < 1.0:
        raise ValueError("Equivalence confidence must lie strictly between zero and one.")
    allocation = _normalize_allocation(protocol, allocation)
    rng = np.random.default_rng(seed)
    by_id = protocol.by_id
    raw_estimates = {}
    estimates = {}
    for control_id in REQUIRED_CONTROL_IDS:
        df = allocation[control_id] - 1
        eta = float(by_id[control_id]["predicted_eta_proxy"])
        raw = eta * np.sqrt(rng.chisquare(df, size=trials) / df)
        raw_estimates[control_id] = raw
        estimates[control_id] = raw / _sample_sd_bias_factor(df)

    resonant_ids = [
        control_id for control_id in REQUIRED_CONTROL_IDS if control_id != "off_resonance"
    ]
    resonant_matrix = np.column_stack([estimates[control_id] for control_id in resonant_ids])
    selected_ids = np.asarray(resonant_ids, dtype=object)[np.argmin(resonant_matrix, axis=1)]
    selected_depths = np.array([int(by_id[control_id]["N"]) for control_id in selected_ids])
    interior_max = np.maximum(estimates["practical_choice"], estimates["predicted_optimum"])

    decisions = {
        "exact_optimum": selected_ids == "predicted_optimum",
        "operating_window": (selected_depths >= protocol.window_min)
        & (selected_depths <= protocol.window_max),
        "depth_controls": (estimates["low_depth"] > interior_max)
        & (estimates["overdriven"] > interior_max),
        "timing_selectivity": estimates["off_resonance"]
        > estimates["predicted_optimum"],
        "n20_beats_n8": estimates["predicted_optimum"] < estimates["low_depth"],
        "n20_beats_n64": estimates["predicted_optimum"] < estimates["overdriven"],
        "n20_beats_n16": estimates["predicted_optimum"]
        < estimates["practical_choice"],
        "n16_n20_equivalent_10pct": _equivalence_decision(
            raw_estimates["practical_choice"],
            raw_estimates["predicted_optimum"],
            allocation["practical_choice"] - 1,
            allocation["predicted_optimum"] - 1,
            float(equivalence_margin),
            float(equivalence_confidence),
        ),
    }
    output = {}
    for metric_id, decision in decisions.items():
        power = float(np.mean(decision))
        output[metric_id] = {
            "description": METRIC_DESCRIPTIONS[metric_id],
            "power": power,
            "monte_carlo_standard_error": float(np.sqrt(power * (1.0 - power) / trials)),
        }
    return output


def build_power_curve(
    protocol: ProtocolData,
    *,
    batch_grid=DEFAULT_BATCH_GRID,
    trials: int = 50000,
    seed: int = 20260915,
    target_power: float = 0.90,
    batch_duration_s: float = 1.0,
    equivalence_margin: float = 1.10,
    equivalence_confidence: float = 0.90,
) -> tuple[list[dict], dict]:
    grid = tuple(sorted({int(value) for value in batch_grid}))
    if not grid or grid[0] < 3:
        raise ValueError("Batch grid must contain integers of at least three.")
    if not 0.0 < float(target_power) < 1.0:
        raise ValueError("Target power must lie strictly between zero and one.")
    if float(batch_duration_s) <= 0.0:
        raise ValueError("Batch duration must be positive.")

    rows = []
    for batch_count in grid:
        metric_results = simulate_decision_power(
            protocol,
            batch_count,
            trials=trials,
            seed=int(np.random.SeedSequence([seed, batch_count]).generate_state(1)[0]),
            equivalence_margin=equivalence_margin,
            equivalence_confidence=equivalence_confidence,
        )
        for metric_id, values in metric_results.items():
            rows.append(
                {
                    "metric_id": metric_id,
                    "description": values["description"],
                    "independent_batches_per_control": batch_count,
                    "batch_duration_s": float(batch_duration_s),
                    "full_protocol_acquisition_time_h": float(
                        batch_count * len(REQUIRED_CONTROL_IDS) * batch_duration_s / 3600.0
                    ),
                    "metric_specific_acquisition_time_h": float(
                        batch_count
                        * METRIC_CONTROL_COUNTS[metric_id]
                        * batch_duration_s
                        / 3600.0
                    ),
                    "power": values["power"],
                    "monte_carlo_standard_error": values["monte_carlo_standard_error"],
                    "target_power": float(target_power),
                }
            )

    required = {}
    for metric_id in METRIC_DESCRIPTIONS:
        candidates = [
            row
            for row in rows
            if row["metric_id"] == metric_id and row["power"] >= target_power
        ]
        first = min(candidates, key=lambda row: row["independent_batches_per_control"]) if candidates else None
        required[metric_id] = {
            "description": METRIC_DESCRIPTIONS[metric_id],
            "required_independent_batches_per_control": (
                None if first is None else first["independent_batches_per_control"]
            ),
            "full_protocol_acquisition_time_h": (
                None if first is None else first["full_protocol_acquisition_time_h"]
            ),
            "metric_specific_acquisition_time_h": (
                None if first is None else first["metric_specific_acquisition_time_h"]
            ),
            "achieved_power": None if first is None else first["power"],
            "resolved_within_grid": first is not None,
        }

    current_allocation = {
        row["control_id"]: int(row["recommended_independent_repeats"])
        for row in protocol.controls
    }
    current_power = simulate_decision_power(
        protocol,
        current_allocation,
        trials=trials,
        seed=int(np.random.SeedSequence([seed, 0]).generate_state(1)[0]),
        equivalence_margin=equivalence_margin,
        equivalence_confidence=equivalence_confidence,
    )
    core_allocation = {
        control_id: (3 if control_id == "off_resonance" else 160)
        for control_id in REQUIRED_CONTROL_IDS
    }
    equivalence_allocation = dict(core_allocation)
    equivalence_allocation["practical_choice"] = 1600
    equivalence_allocation["predicted_optimum"] = 1600

    def design_summary(allocation: dict[str, int]) -> dict:
        return {
            "allocation": allocation,
            "total_batch_time_h": float(
                sum(allocation.values()) * batch_duration_s / 3600.0
            ),
            "power": simulate_decision_power(
                protocol,
                allocation,
                trials=trials,
                seed=int(
                    np.random.SeedSequence(
                        [seed, *allocation.values()]
                    ).generate_state(1)[0]
                ),
                equivalence_margin=equivalence_margin,
                equivalence_confidence=equivalence_confidence,
            ),
        }

    sequence_execution_upper_bounds = {
        row["control_id"]: int(np.floor(batch_duration_s / float(row["sequence_duration_s"])))
        for row in protocol.controls
    }
    summary = {
        "schema_version": 1,
        "status": "prospective_power_analysis_no_measurements",
        "slide_headline": "The operating window is much easier to test than the exact optimum.",
        "protocol_id": protocol.protocol_id,
        "protocol_digest": protocol.protocol_digest,
        "calibration_digest": protocol.calibration_digest,
        "measurement_model": (
            "Each independent batch yields an approximately Gaussian field estimate; "
            "eta is estimated from the sample standard deviation, giving a scaled-chi sampling distribution."
        ),
        "interpretation_limit": (
            "This is conditional power under the nominal predicted eta values. It does not include model-form "
            "error, correlations between batches, drift, or uncertainty in the signal-calibration slope."
        ),
        "trials": int(trials),
        "seed": int(seed),
        "target_power": float(target_power),
        "batch_duration_s": float(batch_duration_s),
        "equivalence_margin": float(equivalence_margin),
        "equivalence_confidence": float(equivalence_confidence),
        "operating_window_N": [protocol.window_min, protocol.window_max],
        "candidate_batch_grid": list(grid),
        "maximum_sequence_executions_per_batch_excluding_overhead": sequence_execution_upper_bounds,
        "current_protocol_allocation": current_allocation,
        "current_protocol_total_batch_time_s": float(
            sum(current_allocation.values()) * batch_duration_s
        ),
        "current_protocol_power": current_power,
        "recommended_designs": {
            "core_window_and_boundary_test": {
                "purpose": (
                    "Prioritize the operating-window, shallow/overdriven, and timing claims; "
                    "do not claim that the exact N=20 optimum has been resolved."
                ),
                **design_summary(core_allocation),
            },
            "ten_percent_equivalence_add_on": {
                "purpose": (
                    "Add precision at N=16 and N=20 to establish their equivalence within 10%; "
                    "this still does not efficiently prove the 0.97% nominal ordering."
                ),
                **design_summary(equivalence_allocation),
            },
        },
        "required_for_target_power": required,
    }
    return rows, summary
