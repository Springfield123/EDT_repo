"""Threshold persistence for connected XY8 performance regions."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

import xy8_resonant_topology as topology


def _validate_thresholds(threshold_percents: Iterable[float]) -> tuple[float, ...]:
    thresholds = tuple(float(value) for value in threshold_percents)
    if not thresholds:
        raise ValueError("At least one performance threshold is required.")
    if not all(np.isfinite(value) and value >= 0.0 for value in thresholds):
        raise ValueError("Threshold percentages must be finite and non-negative.")
    if any(right <= left for left, right in zip(thresholds, thresholds[1:])):
        raise ValueError("Threshold percentages must be strictly increasing.")
    return thresholds


def timing_width_bounds(
    result: topology.ConnectedRegionResult,
    label: int,
) -> dict[str, float | None]:
    """Bound timing width using accepted edges and adjacent rejected rows."""
    row_indices = np.unique(np.nonzero(result.labels == int(label))[0])
    if row_indices.size == 0:
        raise ValueError(f"Component label {label} is absent from the result.")
    lower_index = int(row_indices[0])
    upper_index = int(row_indices[-1])
    if lower_index == 0 or upper_index == result.surface.taus_s.size - 1:
        return {
            "lower_bound_s": None,
            "upper_bound_s": None,
            "estimate_s": None,
            "half_interval_s": None,
        }
    accepted_lower = float(result.surface.taus_s[lower_index])
    accepted_upper = float(result.surface.taus_s[upper_index])
    rejected_lower = float(result.surface.taus_s[lower_index - 1])
    rejected_upper = float(result.surface.taus_s[upper_index + 1])
    lower_bound = accepted_upper - accepted_lower
    upper_bound = rejected_upper - rejected_lower
    return {
        "lower_bound_s": lower_bound,
        "upper_bound_s": upper_bound,
        "estimate_s": 0.5 * (lower_bound + upper_bound),
        "half_interval_s": 0.5 * (upper_bound - lower_bound),
    }


def _component_row(
    component_id: str,
    threshold_percent: float,
    region: topology.RegionCharacterization,
    width: dict[str, float | None],
) -> dict:
    def in_microseconds(value):
        return None if value is None else float(value * 1e6)

    return {
        "threshold_percent": float(threshold_percent),
        "relative_limit": float(1.0 + threshold_percent / 100.0),
        "component_id": component_id,
        "point_count": int(region.point_count),
        "sampled_tau_count": int(region.sampled_tau_count),
        "sampled_depth_count": int(region.sampled_depth_count),
        "relative_min": float(region.relative_min),
        "best_tau_us": float(region.best_tau_s * 1e6),
        "best_N": int(region.best_depth),
        "tau_min_us": float(region.tau_min_s * 1e6),
        "tau_max_us": float(region.tau_max_s * 1e6),
        "sampled_tau_width_us": float(region.tau_width_s * 1e6),
        "timing_width_lower_bound_us": in_microseconds(width["lower_bound_s"]),
        "timing_width_upper_bound_us": in_microseconds(width["upper_bound_s"]),
        "timing_width_estimate_us": in_microseconds(width["estimate_s"]),
        "timing_width_half_interval_us": in_microseconds(width["half_interval_s"]),
        "N_min": int(region.depth_min),
        "N_max": int(region.depth_max),
        "N_span_blocks": int(region.depth_span_blocks),
        "contains_global_optimum": bool(region.contains_global_optimum),
        "resonance_identity": region.resonance_identity,
        "resonance_orders": ";".join(str(value) for value in region.resonance_orders),
        "boundary_status": region.boundary_status,
        "sampling_status": region.sampling_status,
    }


def track_threshold_components(
    surface: topology.ScoreSurface,
    *,
    target_frequency_hz: float,
    threshold_percents: Iterable[float],
    connectivity: int = 4,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Track births, continuations, and mergers across nested sublevel sets.

    Component IDs are assigned in performance order at their first observed
    threshold.  When components merge, the oldest (then lowest-numbered)
    component survives and the others receive a merger event.
    """
    thresholds = _validate_thresholds(threshold_percents)
    stable_labels = np.zeros(surface.eta.shape, dtype=np.int32)
    next_component_number = 1
    tracks: dict[str, dict] = {}
    component_rows: list[dict] = []
    events: list[dict] = []

    for threshold_index, threshold_percent in enumerate(thresholds):
        result = topology.connected_regions(
            surface,
            relative_limit=1.0 + threshold_percent / 100.0,
            connectivity=connectivity,
        )
        regions = topology.characterize_regions(
            result,
            target_frequency_hz=target_frequency_hz,
        )
        region_by_label = {region.label: region for region in regions}
        next_stable_labels = np.zeros_like(stable_labels)

        for label in [region.label for region in regions]:
            cells = result.labels == label
            overlapping_numbers = sorted(
                int(value) for value in np.unique(stable_labels[cells]) if value > 0
            )
            if not overlapping_numbers:
                component_number = next_component_number
                next_component_number += 1
                component_id = f"R{component_number}"
                event_type = (
                    "present_at_first_threshold" if threshold_index == 0 else "birth"
                )
                tracks[component_id] = {
                    "component_id": component_id,
                    "first_observed_threshold_percent": float(threshold_percent),
                    "last_observed_separate_threshold_percent": float(threshold_percent),
                    "merged_at_threshold_percent": None,
                    "merged_into": None,
                    "survives_through_maximum_threshold": True,
                    "contains_global_optimum": bool(
                        region_by_label[label].contains_global_optimum
                    ),
                }
                events.append(
                    {
                        "event_type": event_type,
                        "threshold_percent": float(threshold_percent),
                        "component_id": component_id,
                        "related_component_ids": "",
                    }
                )
            else:
                component_number = min(
                    overlapping_numbers,
                    key=lambda number: (
                        tracks[f"R{number}"]["first_observed_threshold_percent"],
                        number,
                    ),
                )
                component_id = f"R{component_number}"
                if len(overlapping_numbers) > 1:
                    merged_ids = [
                        f"R{number}"
                        for number in overlapping_numbers
                        if number != component_number
                    ]
                    events.append(
                        {
                            "event_type": "merge",
                            "threshold_percent": float(threshold_percent),
                            "component_id": component_id,
                            "related_component_ids": ";".join(merged_ids),
                        }
                    )
                    for merged_id in merged_ids:
                        tracks[merged_id]["merged_at_threshold_percent"] = float(
                            threshold_percent
                        )
                        tracks[merged_id]["merged_into"] = component_id
                        tracks[merged_id]["survives_through_maximum_threshold"] = False

            next_stable_labels[cells] = component_number
            tracks[component_id]["last_observed_separate_threshold_percent"] = float(
                threshold_percent
            )
            region = region_by_label[label]
            component_rows.append(
                _component_row(
                    component_id,
                    threshold_percent,
                    region,
                    timing_width_bounds(result, label),
                )
            )
        stable_labels = next_stable_labels

    return component_rows, events, sorted(
        tracks.values(), key=lambda row: int(row["component_id"][1:])
    )
