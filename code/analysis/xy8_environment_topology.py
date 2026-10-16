"""Connected-region analysis and frozen refinement selection for Stage 5."""

from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import xy8_control_recommender as recommender
import nvratemodel_api as model
import xy8_environment_coherence as environment_coherence
import xy8_environment_score as environment_score
import xy8_resonant_topology as topology
import xy8_topology_persistence as persistence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = ROOT / "data" / "processed" / "xy8_environment_score_summary.csv"
DEFAULT_SURFACES = ROOT / "data" / "processed" / "xy8_environment_discovery_surfaces.npz"
DEFAULT_METADATA = ROOT / "data" / "processed" / "xy8_environment_score_metadata.json"
DEFAULT_SELECTION_FREEZE = (
    ROOT / "data" / "processed" / "xy8_environment_confirmation_selection_freeze.json"
)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_confirmation_selection_freeze(
    freeze_path: str | Path = DEFAULT_SELECTION_FREEZE,
) -> dict[str, Any]:
    record = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    paths = {
        "selection_csv_sha256": ROOT / record["selection_csv"],
        "source_discovery_thresholds_sha256": (
            ROOT / "data" / "processed" / "xy8_environment_topology_thresholds.csv"
        ),
        "source_stage4_summary_sha256": DEFAULT_SUMMARY,
        "source_stage4_surfaces_sha256": DEFAULT_SURFACES,
    }
    checks = {key: file_sha256(path) == record[key] for key, path in paths.items()}
    checks["confirmation_not_inspected_at_freeze"] = (
        record["confirmation_scores_inspected_before_freeze"] is False
    )
    checks["freeze_status"] = (
        record["freeze_status"] == "frozen_before_medium_or_fine_confirmation_scores"
    )
    with (ROOT / record["selection_csv"]).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    checks["selected_surface_count"] = len(rows) == int(record["selected_surface_count"])
    return {
        "verified": all(checks.values()),
        "checks": checks,
        "record": record,
    }


def load_stage4_artifacts(
    summary_path: str | Path = DEFAULT_SUMMARY,
    surfaces_path: str | Path = DEFAULT_SURFACES,
    metadata_path: str | Path = DEFAULT_METADATA,
) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    summary_file = Path(summary_path)
    surfaces_file = Path(surfaces_path)
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if file_sha256(summary_file) != metadata["summary_csv_sha256"]:
        raise ValueError("Stage-4 summary CSV digest does not match its metadata.")
    if file_sha256(surfaces_file) != metadata["surfaces_npz_sha256"]:
        raise ValueError("Stage-4 surface archive digest does not match its metadata.")
    with summary_file.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    archive = np.load(surfaces_file)
    eta = np.asarray(archive["eta"], dtype=float)
    taus = np.asarray(archive["taus_s"], dtype=float)
    depths = np.asarray(archive["depths_N"], dtype=int)
    if eta.shape != (len(rows), taus.size, depths.size):
        raise ValueError("Stage-4 surface archive dimensions do not match its summary.")
    if any(int(row["surface_index"]) != index for index, row in enumerate(rows)):
        raise ValueError("Stage-4 surface indices are not contiguous and ordered.")
    return rows, eta, taus, depths, metadata


def _region_record(region: topology.RegionCharacterization) -> dict[str, Any]:
    return {
        "component_id": region.component_id,
        "point_count": region.point_count,
        "sampled_tau_count": region.sampled_tau_count,
        "sampled_depth_count": region.sampled_depth_count,
        "relative_min": region.relative_min,
        "best_tau_s": region.best_tau_s,
        "best_N": region.best_depth,
        "tau_min_s": region.tau_min_s,
        "tau_max_s": region.tau_max_s,
        "tau_width_s": region.tau_width_s,
        "N_min": region.depth_min,
        "N_max": region.depth_max,
        "contains_global_optimum": region.contains_global_optimum,
        "resonance_identity": region.resonance_identity,
        "boundary_status": region.boundary_status,
        "sampling_status": region.sampling_status,
        "contains_N16": region.contains_depth_16,
        "contains_N20": region.contains_depth_20,
    }


def analyze_surface_thresholds(
    eta: np.ndarray,
    taus_s: np.ndarray,
    depths_N: np.ndarray,
    *,
    target_frequency_hz: float,
    relative_limits: list[float],
    stage: str,
    surface_index: int,
    source_row: dict[str, Any],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Return threshold, component, and persistence-track records."""
    surface = topology.validate_surface(taus_s, depths_N, eta)
    threshold_rows = []
    component_rows = []
    common = {
        "grid_stage": stage,
        "surface_index": int(surface_index),
        "condition_id": source_row["condition_id"],
        "strain_path": source_row["strain_path"],
        "temperature_K": float(source_row["temperature_K"]),
        "amplitude_microstrain": float(source_row["amplitude_microstrain"]),
        "carrier_policy": source_row["carrier_policy"],
        "coherence_variant": source_row["coherence_variant"],
    }
    for relative_limit in relative_limits:
        result4 = topology.connected_regions(
            surface, relative_limit=relative_limit, connectivity=4
        )
        result8 = topology.connected_regions(
            surface, relative_limit=relative_limit, connectivity=8
        )
        regions = topology.characterize_regions(
            result4, target_frequency_hz=target_frequency_hz
        )
        primary = next(region for region in regions if region.contains_global_optimum)
        width = persistence.timing_width_bounds(result4, primary.label)
        threshold_rows.append(
            {
                **common,
                "relative_limit": relative_limit,
                "threshold_percent": 100.0 * (relative_limit - 1.0),
                "beta_0_four": result4.beta_0,
                "beta_0_eight": result8.beta_0,
                "connectivity_agrees": result4.beta_0 == result8.beta_0,
                "best_tau_s": primary.best_tau_s,
                "best_N": primary.best_depth,
                "primary_tau_min_s": primary.tau_min_s,
                "primary_tau_max_s": primary.tau_max_s,
                "primary_tau_width_s": primary.tau_width_s,
                "primary_timing_width_estimate_s": width["estimate_s"],
                "primary_timing_width_half_interval_s": width["half_interval_s"],
                "primary_N_min": primary.depth_min,
                "primary_N_max": primary.depth_max,
                "primary_sampled_tau_count": primary.sampled_tau_count,
                "primary_sampled_depth_count": primary.sampled_depth_count,
                "primary_boundary_status": primary.boundary_status,
                "primary_sampling_status": primary.sampling_status,
            }
        )
        for region in regions:
            component_rows.append(
                {
                    **common,
                    "relative_limit": relative_limit,
                    "threshold_percent": 100.0 * (relative_limit - 1.0),
                    **_region_record(region),
                }
            )

    threshold_percents = [100.0 * (limit - 1.0) for limit in relative_limits]
    _, _, tracks = persistence.track_threshold_components(
        surface,
        target_frequency_hz=target_frequency_hz,
        threshold_percents=threshold_percents,
        connectivity=4,
    )
    track_rows = [{**common, **track} for track in tracks]
    return threshold_rows, component_rows, track_rows


def analyze_discovery_surfaces(
    rows: list[dict[str, str]],
    eta: np.ndarray,
    taus_s: np.ndarray,
    depths_N: np.ndarray,
    *,
    target_frequency_hz: float,
    relative_limits: list[float],
) -> tuple[list[dict], list[dict], list[dict]]:
    thresholds = []
    components = []
    tracks = []
    for index, row in enumerate(rows):
        surface_thresholds, surface_components, surface_tracks = analyze_surface_thresholds(
            eta[index],
            taus_s,
            depths_N,
            target_frequency_hz=target_frequency_hz,
            relative_limits=relative_limits,
            stage="discovery",
            surface_index=index,
            source_row=row,
        )
        thresholds.extend(surface_thresholds)
        components.extend(surface_components)
        tracks.extend(surface_tracks)
    return thresholds, components, tracks


def _primary_by_surface(
    threshold_rows: list[dict], primary_relative_limit: float
) -> dict[int, dict]:
    selected = {
        int(row["surface_index"]): row
        for row in threshold_rows
        if np.isclose(float(row["relative_limit"]), primary_relative_limit)
    }
    if len(selected) != len({int(row["surface_index"]) for row in threshold_rows}):
        raise ValueError("Every surface must have exactly one primary-threshold record.")
    return selected


def select_confirmation_surfaces(
    rows: list[dict[str, str]],
    threshold_rows: list[dict],
    *,
    primary_relative_limit: float,
) -> list[dict]:
    """Apply frozen refinement rules plus a labelled exact-N diagnostic rule."""
    primary = _primary_by_surface(threshold_rows, primary_relative_limit)
    reasons: dict[int, set[str]] = defaultdict(set)
    preregistered: dict[int, bool] = defaultdict(bool)

    def add(index: int, reason: str, *, frozen: bool = True) -> None:
        reasons[index].add(reason)
        preregistered[index] = preregistered[index] or frozen

    # Reference environment for both policies and both frozen T1 variants.
    for index, row in enumerate(rows):
        if row["condition_id"] == "zero_strain_T300K":
            add(index, "reference_environment")

    # One-dimensional zero-strain endpoints and all corners of each 2-D map.
    grouped: dict[tuple[str, str, str], list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[(row["strain_path"], row["carrier_policy"], row["coherence_variant"])].append(
            (index, row)
        )
        p = primary[index]
        if (
            p["primary_boundary_status"] != "resolved_interior"
            or str(row["best_touches_boundary"]).lower() == "true"
            or str(row["window_touches_tau_boundary"]).lower() == "true"
            or str(row["window_touches_N_boundary"]).lower() == "true"
        ):
            add(index, "discovery_boundary_contact")

    for (path, _, _), members in grouped.items():
        temperatures = sorted({float(row["temperature_K"]) for _, row in members})
        amplitudes = sorted({float(row["amplitude_microstrain"]) for _, row in members})
        by_coordinate = {
            (float(row["temperature_K"]), float(row["amplitude_microstrain"])): index
            for index, row in members
        }
        if path == "zero_strain":
            for temperature in (temperatures[0], temperatures[-1]):
                add(by_coordinate[(temperature, 0.0)], "operational_axis_endpoint")
        else:
            for temperature in (temperatures[0], temperatures[-1]):
                for amplitude in (amplitudes[0], amplitudes[-1]):
                    add(by_coordinate[(temperature, amplitude)], "operational_map_corner")

        neighbour_pairs = []
        for temperature_index, temperature in enumerate(temperatures):
            for amplitude_index, amplitude in enumerate(amplitudes):
                here = by_coordinate.get((temperature, amplitude))
                if here is None:
                    continue
                if temperature_index + 1 < len(temperatures):
                    other = by_coordinate.get((temperatures[temperature_index + 1], amplitude))
                    if other is not None:
                        neighbour_pairs.append((here, other))
                if path != "zero_strain" and amplitude_index + 1 < len(amplitudes):
                    other = by_coordinate.get((temperature, amplitudes[amplitude_index + 1]))
                    if other is not None:
                        neighbour_pairs.append((here, other))

        for left, right in neighbour_pairs:
            left_primary = primary[left]
            right_primary = primary[right]
            if left_primary["beta_0_four"] != right_primary["beta_0_four"]:
                add(left, "adjacent_beta0_transition")
                add(right, "adjacent_beta0_transition")
            left_category = recommender.control_region(int(rows[left]["best_N"]))
            right_category = recommender.control_region(int(rows[right]["best_N"]))
            if left_category != right_category:
                add(left, "adjacent_depth_category_transition")
                add(right, "adjacent_depth_category_transition")
            if int(rows[left]["best_N"]) != int(rows[right]["best_N"]):
                add(left, "supplementary_exact_N_transition", frozen=False)
                add(right, "supplementary_exact_N_transition", frozen=False)

    selection = []
    for index in sorted(reasons):
        row = rows[index]
        selection.append(
            {
                "discovery_surface_index": index,
                "condition_id": row["condition_id"],
                "strain_path": row["strain_path"],
                "temperature_K": float(row["temperature_K"]),
                "amplitude_microstrain": float(row["amplitude_microstrain"]),
                "carrier_policy": row["carrier_policy"],
                "coherence_variant": row["coherence_variant"],
                "applied_detuning_Hz": float(row["applied_detuning_Hz"]),
                "discovery_best_tau_s": float(row["best_tau_s"]),
                "discovery_best_N": int(row["best_N"]),
                "discovery_beta_0_four": int(primary[index]["beta_0_four"]),
                "discovery_beta_0_eight": int(primary[index]["beta_0_eight"]),
                "selection_reasons": ";".join(sorted(reasons[index])),
                "selected_by_preregistered_rule": preregistered[index],
                "selected_by_supplementary_exact_N_rule": (
                    "supplementary_exact_N_transition" in reasons[index]
                ),
            }
        )
    return selection


def build_confirmation_tau_grid(config: dict[str, Any], grid_stage: str) -> np.ndarray:
    if grid_stage == "fine":
        declared = config["control_grids"]["confirmation_timing_grid"]
    elif grid_stage == "medium":
        fine = config["control_grids"]["confirmation_timing_grid"]
        declared = {
            **fine,
            "background_count": 161,
            "resonance_bracket_count": 401,
            "expected_tau_count_at_200_kHz": 962,
        }
    else:
        raise ValueError("grid_stage must be 'medium' or 'fine'.")
    taus = topology.build_resonance_refined_tau_grid(
        config["signal_and_control"]["target_frequency_Hz"],
        declared["tau_min_s"],
        declared["tau_max_s"],
        background_count=declared["background_count"],
        resonance_halfwidth_fraction=declared["resonance_halfwidth_fraction"],
        resonance_point_count=declared["resonance_bracket_count"],
    )
    if taus.size != declared["expected_tau_count_at_200_kHz"]:
        raise RuntimeError(
            f"{grid_stage} timing grid has {taus.size} points; expected "
            f"{declared['expected_tau_count_at_200_kHz']}."
        )
    return taus


def evaluate_nominal_reference_surface(
    taus_s: np.ndarray,
    depths_N: np.ndarray,
    nominal_scenario: dict[str, Any],
    *,
    target_frequency_hz: float,
) -> environment_score.CanonicalSurface:
    readout = nominal_scenario["readout"]
    overheads = nominal_scenario["overheads"]
    t2_params = nominal_scenario["coherence"]
    pulse_fidelity = nominal_scenario["pulse"]["piPulseFid"]
    neff = float(nominal_scenario["ensemble"]["Neff"])
    eta = np.empty((taus_s.size, depths_N.size), dtype=float)
    for tau_index, tau in enumerate(taus_s):
        for depth_index, depth in enumerate(depths_N):
            eta[tau_index, depth_index] = model.eta_ac(
                target_frequency_hz,
                tau,
                int(depth),
                readout["effective_detected_rate_cps"],
                readout["C_1"],
                neff,
                t2_params,
                overheads,
                piPulseFid=pulse_fidelity,
            )
    return environment_score.CanonicalSurface(
        taus_s=np.asarray(taus_s, dtype=float),
        depths_N=np.asarray(depths_N, dtype=int),
        eta=eta,
    )


def build_confirmation_surfaces_for_grid(
    selection_rows: list[dict[str, str]],
    *,
    config: dict[str, Any],
    nominal_scenario: dict[str, Any],
    grid_stage: str,
) -> tuple[np.ndarray, environment_score.CanonicalSurface, dict[str, Any]]:
    """Evaluate every frozen confirmation row on one refined timing grid."""
    taus = build_confirmation_tau_grid(config, grid_stage)
    depths = np.asarray(config["control_grids"]["depths_N"], dtype=int)
    target = float(config["signal_and_control"]["target_frequency_Hz"])
    reference = evaluate_nominal_reference_surface(
        taus, depths, nominal_scenario, target_frequency_hz=target
    )
    fixed_detunings = sorted(
        {
            float(row["applied_detuning_Hz"])
            for row in selection_rows
            if row["carrier_policy"] == "fixed_reference_carrier"
        }
        | {0.0}
    )
    gains = environment_score.batched_deterministic_xy8_gains(
        taus,
        depths,
        fixed_detunings,
        signal_frequency_hz=target,
        pi_pulse_duration_s=config["signal_and_control"]["pi_pulse_duration_s"],
        pulse_substeps=8,
    )
    detuning_index = {value: index for index, value in enumerate(fixed_detunings)}
    zero_gain = gains[detuning_index[0.0]]
    gain_floor = 1e-12
    gain_ratios = np.maximum(gains, gain_floor) / np.maximum(
        zero_gain[None, :, :], gain_floor
    )
    nominal_t2 = dict(nominal_scenario["coherence"])
    base_by_temperature_variant = {}
    for row in selection_rows:
        key = (row["coherence_variant"], float(row["temperature_K"]))
        if key not in base_by_temperature_variant:
            base, _ = environment_score.coherence_conditioned_surface(
                reference,
                key[1],
                variant=key[0],
                nominal_t2_params=nominal_t2,
            )
            base_by_temperature_variant[key] = base

    surfaces = []
    for row in selection_rows:
        base = base_by_temperature_variant[
            (row["coherence_variant"], float(row["temperature_K"]))
        ]
        if row["carrier_policy"] == "fixed_reference_carrier":
            ratio = gain_ratios[detuning_index[float(row["applied_detuning_Hz"])]]
            surfaces.append(base / ratio)
        else:
            surfaces.append(base.copy())
    stacked = np.stack(surfaces)
    diagnostics = {
        "grid_stage": grid_stage,
        "tau_count": int(taus.size),
        "depth_count": int(depths.size),
        "surface_count": len(selection_rows),
        "unique_fixed_detuning_count": len(fixed_detunings),
        "gain_floor": gain_floor,
        "clipped_gain_count": int(np.count_nonzero(gains < gain_floor)),
    }
    return stacked, reference, diagnostics


def confirmation_stability_records(
    selection_rows: list[dict[str, str]],
    discovery_thresholds: list[dict],
    medium_thresholds: list[dict],
    fine_thresholds: list[dict],
    *,
    primary_relative_limit: float,
    width_relative_tolerance: float = 0.05,
    discovery_timing_resolution_s: float = 25e-9,
    medium_timing_resolution_s: float = 0.5e-9,
) -> list[dict]:
    def primary(rows):
        return {
            int(row["surface_index"]): row
            for row in rows
            if np.isclose(float(row["relative_limit"]), primary_relative_limit)
        }

    discovery = primary(discovery_thresholds)
    medium = primary(medium_thresholds)
    fine = primary(fine_thresholds)
    output = []
    for selection_index, selected in enumerate(selection_rows):
        source_index = int(selected["discovery_surface_index"])
        d = discovery[source_index]
        m = medium[source_index]
        f = fine[source_index]
        medium_width = m["primary_timing_width_estimate_s"]
        fine_width = f["primary_timing_width_estimate_s"]
        width_change = (
            abs(float(medium_width) - float(fine_width)) / float(fine_width)
            if medium_width is not None and fine_width is not None and float(fine_width) > 0.0
            else None
        )
        optimum_depth_stable = (
            int(d["best_N"]) == int(m["best_N"]) == int(f["best_N"])
        )
        timing_optimum_converged = (
            abs(float(d["best_tau_s"]) - float(f["best_tau_s"]))
            <= discovery_timing_resolution_s + 1e-15
            and abs(float(m["best_tau_s"]) - float(f["best_tau_s"]))
            <= medium_timing_resolution_s + 1e-15
        )
        checks = {
            "optimum_depth_stable_discovery_medium_fine": optimum_depth_stable,
            "timing_optimum_converged_within_grid_resolution": timing_optimum_converged,
            "global_optimum_stable_discovery_medium_fine": (
                optimum_depth_stable and timing_optimum_converged
            ),
            "beta0_stable_discovery_medium_fine": (
                int(d["beta_0_four"]) == int(m["beta_0_four"]) == int(f["beta_0_four"])
            ),
            "connectivity_agrees_medium_and_fine": (
                bool(m["connectivity_agrees"]) and bool(f["connectivity_agrees"])
            ),
            "fine_primary_region_resolved_on_both_axes": (
                int(f["primary_sampled_tau_count"]) > 1
                and int(f["primary_sampled_depth_count"]) > 1
            ),
            "fine_primary_region_avoids_boundaries": (
                f["primary_boundary_status"] == "resolved_interior"
            ),
            "medium_to_fine_timing_width_converged": (
                width_change is not None and width_change <= width_relative_tolerance
            ),
        }
        output.append(
            {
                "confirmation_surface_index": selection_index,
                **selected,
                "medium_best_tau_s": float(m["best_tau_s"]),
                "medium_best_N": int(m["best_N"]),
                "fine_best_tau_s": float(f["best_tau_s"]),
                "fine_best_N": int(f["best_N"]),
                "fine_beta_0_four": int(f["beta_0_four"]),
                "fine_beta_0_eight": int(f["beta_0_eight"]),
                "fine_primary_N_min": int(f["primary_N_min"]),
                "fine_primary_N_max": int(f["primary_N_max"]),
                "fine_primary_tau_min_s": float(f["primary_tau_min_s"]),
                "fine_primary_tau_max_s": float(f["primary_tau_max_s"]),
                "fine_timing_width_estimate_s": fine_width,
                "medium_to_fine_width_relative_change": width_change,
                "discovery_to_fine_best_tau_shift_s": abs(
                    float(d["best_tau_s"]) - float(f["best_tau_s"])
                ),
                "medium_to_fine_best_tau_shift_s": abs(
                    float(m["best_tau_s"]) - float(f["best_tau_s"])
                ),
                "discovery_timing_resolution_s": discovery_timing_resolution_s,
                "medium_timing_resolution_s": medium_timing_resolution_s,
                **checks,
                "confirmation_passed": all(checks.values()),
            }
        )
    return output
