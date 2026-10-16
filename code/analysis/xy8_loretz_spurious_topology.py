"""One-dimensional connected-region persistence for Loretz Figure 4a."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def load(manifest_path: Path, data_path: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("selection_status") != "frozen_before_digitization_and_topology_evaluation":
        raise ValueError("The Loretz selection must be frozen before evaluation.")
    with data_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            {
                "segment_id": row["segment_id"],
                "total_pi_pulses": int(row["total_pi_pulses"]),
                "xy8_blocks": int(row["xy8_blocks"]),
                "frequency_MHz": float(row["detection_frequency_MHz"]),
                "transition_probability": float(row["transition_probability"]),
            }
            for row in csv.DictReader(handle)
        ]
    expected = {item["segment_id"]: item["expected_path_points"] for item in manifest["predeclared_segments"]}
    actual = {key: sum(row["segment_id"] == key for row in rows) for key in expected}
    if actual != expected:
        raise ValueError("Digitized Loretz segment counts do not match the frozen manifest.")
    return manifest, rows


def classify(frequency: float, reported_frequencies: list[float]) -> str:
    ratios = {
        "fundamental": 1.0,
        "second_harmonic": 2.0,
        "fourth_harmonic": 4.0,
    }
    candidates = []
    for label, multiplier in ratios.items():
        for reference in reported_frequencies:
            target = multiplier * reference
            candidates.append((abs(frequency - target) / target, label))
    mismatch, label = min(candidates)
    return label if mismatch <= 0.12 else "other_or_background"


def prepare_segments(manifest: dict, rows: list[dict]) -> list[dict]:
    segments = []
    for specification in manifest["predeclared_segments"]:
        selected = [row for row in rows if row["segment_id"] == specification["segment_id"]]
        frequency = np.asarray([row["frequency_MHz"] for row in selected])
        probability = np.asarray([row["transition_probability"] for row in selected])
        order = np.argsort(frequency)
        frequency, probability = frequency[order], probability[order]
        edge_count = max(3, int(round(0.2 * probability.size)))
        baseline = float(np.median(np.concatenate([probability[:edge_count], probability[-edge_count:]])))
        segments.append(
            {
                "segment_id": specification["segment_id"],
                "total_pi_pulses": specification["total_pi_pulses"],
                "xy8_blocks": specification["total_pi_pulses"] // 8,
                "frequency_MHz": frequency,
                "transition_probability": probability,
                "baseline_probability": baseline,
                "response": np.maximum(probability - baseline, 0.0),
            }
        )
    global_maximum = max(float(np.max(segment["response"])) for segment in segments)
    for segment in segments:
        segment["normalized_response"] = segment["response"] / global_maximum
    return segments


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(mask & np.concatenate([[True], ~mask[:-1]]))
    output = []
    for start in starts:
        stop = int(start)
        while stop + 1 < mask.size and mask[stop + 1]:
            stop += 1
        output.append((int(start), stop))
    return output


def analyze(manifest: dict, rows: list[dict], thresholds=tuple(range(1, 101))) -> dict:
    segments = prepare_segments(manifest, rows)
    threshold_rows = []
    component_rows = []
    events = []
    previous = None
    references = manifest["source"]["reported_nuclear_frequencies_MHz"]
    for threshold in thresholds:
        floor = 1.0 - threshold / 100.0
        component_count = 0
        identities = set()
        for segment in segments:
            mask = segment["normalized_response"] >= floor - 1e-12
            for start, stop in runs(mask):
                component_count += 1
                local = segment["normalized_response"][start : stop + 1]
                best = start + int(np.argmax(local))
                identity = classify(float(segment["frequency_MHz"][best]), references)
                identities.add(identity)
                component_rows.append(
                    {
                        "threshold_percent": int(threshold),
                        "component_index": component_count,
                        "segment_id": segment["segment_id"],
                        "total_pi_pulses": segment["total_pi_pulses"],
                        "frequency_min_MHz": float(segment["frequency_MHz"][start]),
                        "frequency_max_MHz": float(segment["frequency_MHz"][stop]),
                        "best_frequency_MHz": float(segment["frequency_MHz"][best]),
                        "normalized_response_max": float(segment["normalized_response"][best]),
                        "resonance_identity": identity,
                    }
                )
        threshold_rows.append(
            {
                "threshold_percent": int(threshold),
                "relative_response_floor": floor,
                "beta_0_segmented_spectrum": component_count,
                "fundamental_present": int("fundamental" in identities),
                "second_harmonic_present": int("second_harmonic" in identities),
                "fourth_harmonic_present": int("fourth_harmonic" in identities),
                "other_component_present": int("other_or_background" in identities),
            }
        )
        if component_count != previous:
            events.append({"threshold_percent": int(threshold), "beta_0": component_count})
            previous = component_count
    first_seen = {}
    for identity in ("fundamental", "second_harmonic", "fourth_harmonic"):
        matches = [row["threshold_percent"] for row in component_rows if row["resonance_identity"] == identity]
        first_seen[identity] = min(matches) if matches else None
    segment_rows = []
    for segment in segments:
        for index in range(segment["frequency_MHz"].size):
            segment_rows.append(
                {
                    "segment_id": segment["segment_id"],
                    "total_pi_pulses": segment["total_pi_pulses"],
                    "xy8_blocks": segment["xy8_blocks"],
                    "frequency_MHz": float(segment["frequency_MHz"][index]),
                    "transition_probability": float(segment["transition_probability"][index]),
                    "baseline_probability": segment["baseline_probability"],
                    "normalized_response_global": float(segment["normalized_response"][index]),
                }
            )
    summary = {
        "status": "frozen_loretz_spurious_topology_evaluated",
        "scope": "segmented one-dimensional experimental XY8 spectrum; separate pulse-error mechanism check",
        "first_threshold_percent_where_identity_is_selected": first_seen,
        "component_count_events": events,
        "no_fit_statement": "The published simulation was excluded. No frequency shift, curve fit, or model tuning was performed.",
        "interpretation": "The fundamental response dominates strict thresholds, while second- and fourth-harmonic regions enter only when a larger response loss is allowed. Connected-region persistence therefore detects experimentally observed spurious XY8 passbands.",
        "separation_guardrail": manifest["separation_guardrail"],
    }
    return {"segment_rows": segment_rows, "threshold_rows": threshold_rows, "component_rows": component_rows, "summary": summary}
