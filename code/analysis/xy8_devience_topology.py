"""Sparse two-dimensional topology analysis for DeVience et al. Figure 2b.

The four published XY8 timing sweeps are interpolated onto a predeclared
common frequency grid.  The analysis uses the measured resonance response
only; it does not fit, shift, or rescale the project's sensitivity model.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


FOUR_NEIGHBOUR = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("selection_status") != "frozen_before_digitization_and_topology_evaluation":
        raise ValueError("The DeVience dataset must be frozen before evaluation.")
    return manifest


def load_measurements(path: Path, manifest: dict) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        raw = list(csv.DictReader(handle))
    rows = [
        {
            "xy8_blocks": int(row["xy8_blocks"]),
            "total_pi_pulses": int(row["total_pi_pulses"]),
            "point_index": int(row["point_index"]),
            "frequency_MHz": float(row["frequency_MHz"]),
            "project_tau_us": float(row["project_tau_us"]),
            "corrected_contrast": float(row["corrected_contrast"]),
            "resonant_response": float(row["resonant_response"]),
            "frequency_digitization_error_MHz": float(row["frequency_digitization_error_MHz"]),
            "contrast_digitization_error": float(row["contrast_digitization_error"]),
        }
        for row in raw
    ]
    depths = manifest["predeclared_depths"]["xy8_blocks"]
    counts = [sum(row["xy8_blocks"] == depth for row in rows) for depth in depths]
    if counts != manifest["predeclared_depths"]["expected_marker_counts"]:
        raise ValueError(f"Marker counts {counts} do not match the frozen manifest.")
    for depth in depths:
        frequencies = [row["frequency_MHz"] for row in rows if row["xy8_blocks"] == depth]
        if any(right <= left for left, right in zip(frequencies, frequencies[1:])):
            raise ValueError(f"XY8-{depth} frequencies must be strictly increasing.")
    return rows


def common_frequency_grid(manifest: dict) -> np.ndarray:
    specification = manifest["topology_policy"]["common_frequency_grid_MHz"]
    start = float(specification["start"])
    stop = float(specification["stop"])
    step = float(specification["step"])
    count = int(round((stop - start) / step)) + 1
    grid = start + step * np.arange(count, dtype=float)
    if not np.isclose(grid[-1], stop):
        raise ValueError("The frozen frequency grid does not terminate at its stop value.")
    return grid


def rectangular_surface(
    measurements: list[dict], manifest: dict, *, frequency_shift=None, response_shift=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depths = np.asarray(manifest["predeclared_depths"]["xy8_blocks"], dtype=int)
    grid = common_frequency_grid(manifest)
    surface = np.empty((depths.size, grid.size), dtype=float)
    for row_index, depth in enumerate(depths):
        selected = [row for row in measurements if row["xy8_blocks"] == depth]
        frequency = np.asarray([row["frequency_MHz"] for row in selected], dtype=float)
        response = np.asarray([row["resonant_response"] for row in selected], dtype=float)
        if frequency_shift is not None:
            frequency = frequency + np.asarray(frequency_shift[row_index], dtype=float)
        if response_shift is not None:
            response = response + np.asarray(response_shift[row_index], dtype=float)
        order = np.argsort(frequency)
        frequency = frequency[order]
        response = np.maximum(response[order], 0.0)
        if grid[0] < frequency[0] or grid[-1] > frequency[-1]:
            raise ValueError("Frozen common grid requires extrapolation for a sampled row.")
        surface[row_index] = np.interp(grid, frequency, response)
    maximum = float(np.max(surface))
    if maximum <= 0.0:
        raise ValueError("The measured surface contains no positive response.")
    return grid, depths, surface / maximum


def component_rows(mask: np.ndarray, normalized_response: np.ndarray, grid, depths, threshold: float) -> list[dict]:
    labels, count = ndimage.label(mask, structure=FOUR_NEIGHBOUR)
    rows = []
    for label in range(1, count + 1):
        depth_indices, frequency_indices = np.where(labels == label)
        values = normalized_response[depth_indices, frequency_indices]
        best = int(np.argmax(values))
        minimum_frequency = float(grid[np.min(frequency_indices)])
        maximum_frequency = float(grid[np.max(frequency_indices)])
        if maximum_frequency < 0.915:
            identity = "19F-like lower-frequency resonance"
        elif minimum_frequency > 0.915:
            identity = "1H-like higher-frequency resonance"
        else:
            identity = "merged or cross-species region"
        rows.append(
            {
                "threshold_percent": float(threshold),
                "component_label": label,
                "point_count": int(values.size),
                "frequency_min_MHz": minimum_frequency,
                "frequency_max_MHz": maximum_frequency,
                "frequency_width_MHz": maximum_frequency - minimum_frequency,
                "depth_min_blocks": int(depths[np.min(depth_indices)]),
                "depth_max_blocks": int(depths[np.max(depth_indices)]),
                "best_frequency_MHz": float(grid[frequency_indices[best]]),
                "best_depth_blocks": int(depths[depth_indices[best]]),
                "relative_response_max": float(values[best]),
                "resonance_identity": identity,
            }
        )
    return sorted(rows, key=lambda row: (-row["relative_response_max"], row["frequency_min_MHz"]))


def _bootstrap_beta0(
    measurements: list[dict], manifest: dict, thresholds: tuple[int, ...], trials: int, seed: int
) -> np.ndarray:
    if trials < 100:
        raise ValueError("At least 100 digitization-uncertainty trials are required.")
    depths = manifest["predeclared_depths"]["xy8_blocks"]
    rng = np.random.default_rng(seed)
    beta = np.empty((trials, len(thresholds)), dtype=np.uint8)
    row_measurements = [[row for row in measurements if row["xy8_blocks"] == depth] for depth in depths]
    for trial in range(trials):
        frequency_shift = []
        response_shift = []
        for selected in row_measurements:
            frequency_sigma = np.asarray([row["frequency_digitization_error_MHz"] for row in selected])
            contrast_sigma = np.asarray([row["contrast_digitization_error"] for row in selected])
            frequency_shift.append(rng.normal(0.0, frequency_sigma))
            response_shift.append(rng.normal(0.0, contrast_sigma))
        _, _, response = rectangular_surface(
            measurements,
            manifest,
            frequency_shift=frequency_shift,
            response_shift=response_shift,
        )
        for threshold_index, threshold in enumerate(thresholds):
            mask = response >= 1.0 - threshold / 100.0 - 1e-12
            _, beta[trial, threshold_index] = ndimage.label(mask, structure=FOUR_NEIGHBOUR)
    return beta


def analyze(
    manifest: dict,
    measurements: list[dict],
    *,
    thresholds=tuple(range(1, 101)),
    bootstrap_trials: int = 10_000,
    seed: int = 2014313,
) -> dict:
    threshold_values = tuple(int(value) for value in thresholds)
    if not threshold_values or any(b <= a for a, b in zip(threshold_values, threshold_values[1:])):
        raise ValueError("Thresholds must be a non-empty strictly increasing sequence.")
    grid, depths, response = rectangular_surface(measurements, manifest)
    bootstrap_beta = _bootstrap_beta0(
        measurements, manifest, threshold_values, bootstrap_trials, seed
    )
    threshold_rows = []
    components = []
    central_beta = []
    for threshold_index, threshold in enumerate(threshold_values):
        mask = response >= 1.0 - threshold / 100.0 - 1e-12
        labels, count = ndimage.label(mask, structure=FOUR_NEIGHBOUR)
        del labels
        central_beta.append(int(count))
        values, counts = np.unique(bootstrap_beta[:, threshold_index], return_counts=True)
        modal = int(values[int(np.argmax(counts))])
        threshold_rows.append(
            {
                "threshold_percent": threshold,
                "relative_response_floor": 1.0 - threshold / 100.0,
                "central_beta_0": int(count),
                "bootstrap_modal_beta_0_digitization_only": modal,
                "bootstrap_probability_matches_central_digitization_only": float(
                    np.mean(bootstrap_beta[:, threshold_index] == count)
                ),
                "bootstrap_probability_beta_0_equals_1_digitization_only": float(
                    np.mean(bootstrap_beta[:, threshold_index] == 1)
                ),
                "bootstrap_probability_beta_0_equals_2_digitization_only": float(
                    np.mean(bootstrap_beta[:, threshold_index] == 2)
                ),
                "accepted_grid_point_count": int(np.sum(mask)),
                "accepted_grid_fraction": float(np.mean(mask)),
            }
        )
        components.extend(component_rows(mask, response, grid, depths, threshold))

    events = []
    previous = None
    for threshold, count in zip(threshold_values, central_beta):
        if count != previous:
            events.append({"threshold_percent": threshold, "beta_0": count})
            previous = count
    optimum = np.unravel_index(int(np.argmax(response)), response.shape)
    primary_maximum = max(value for threshold, value in zip(threshold_values, central_beta) if threshold <= 60)
    summary = {
        "status": "frozen_devience_sparse_surface_evaluated",
        "scope": "digitized experimental frequency-by-XY8-depth resonance-response surface",
        "depths_xy8_blocks": depths.tolist(),
        "depths_total_pi_pulses": (8 * depths).tolist(),
        "frequency_grid_MHz": {
            "min": float(grid[0]),
            "max": float(grid[-1]),
            "step": float(grid[1] - grid[0]),
            "count": int(grid.size),
        },
        "normalization": manifest["normalization_policy"],
        "no_fit_statement": (
            "No timing shift, model fit, per-depth normalization, or absolute sensitivity comparison was performed."
        ),
        "strongest_measured_response": {
            "frequency_MHz": float(grid[optimum[1]]),
            "project_tau_us": float(0.25 / grid[optimum[1]]),
            "xy8_blocks": int(depths[optimum[0]]),
            "total_pi_pulses": int(8 * depths[optimum[0]]),
        },
        "primary_range_1_to_60pct": {
            "maximum_central_beta_0": int(primary_maximum),
            "central_events": [event for event in events if event["threshold_percent"] <= 60],
        },
        "all_threshold_events": events,
        "bootstrap": {
            "trials": int(bootstrap_trials),
            "seed": int(seed),
            "scope": "digitization uncertainty only; experimental error bars were not published for Figure 2b",
        },
        "interpretation": (
            "The data resolve distinct lower- and higher-frequency resonance regions over intermediate thresholds, "
            "followed by their merger as the response floor is relaxed. This is evidence that beta_0 captures "
            "spectral structure, not a validation of the project's 200 kHz optimum."
        ),
        "guardrail": manifest["interpretation_guardrail"],
    }
    return {
        "frequency_grid_MHz": grid,
        "depths": depths,
        "normalized_response": response,
        "threshold_rows": threshold_rows,
        "component_rows": components,
        "summary": summary,
    }
