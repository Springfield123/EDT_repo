#!/usr/bin/env python3
"""Test XY8 connected-region stability on resonance-refined timing grids."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "code", ROOT / "code" / "analysis", ROOT / "code" / "plotting"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import numpy as np

import analyze_xy8_resonant_topology as stage2
import make_fig_xy8_qre as qre
import nvratemodel_api as model
import xy8_resonant_topology as topology


DEFAULT_CALIBRATION = ROOT / "data" / "raw" / "nv" / "xy8_scenario_nominal.json"
DEFAULT_CANONICAL = ROOT / "data" / "processed" / "xy8_qre_nv_results.csv"
DEFAULT_OUTDIR = ROOT / "data" / "processed"
GRID_FILENAME = "xy8_topology_grid_refinement.csv"
RESONANCE_FILENAME = "xy8_topology_resonance_diagnostics.csv"
SUMMARY_FILENAME = "xy8_topology_refinement_summary.json"


def evaluate_surface(calibration: dict, taus_s, depths, target_frequency_hz: float):
    results, _, _, _ = qre.run_sweep(
        dict(qre.DEFAULT_MODEL_KWARGS),
        np.asarray(taus_s, dtype=float),
        np.asarray(depths, dtype=int),
        float(target_frequency_hz),
        qre.DEFAULT_NEFF,
        dict(qre.DEFAULT_OVERHEADS),
        dict(qre.DEFAULT_T2_PARAMS),
        calibration=calibration,
        integration_time=300e-9,
        contrast_mode="fixed",
        pl_model="MEmodel",
    )
    return topology.surface_from_points(results)


def local_resonance_resolution(surface: topology.ScoreSurface, resonance_tau_s: float) -> dict:
    index = int(np.argmin(np.abs(surface.taus_s - resonance_tau_s)))
    if not np.isclose(surface.taus_s[index], resonance_tau_s, rtol=1e-9, atol=1e-15):
        raise RuntimeError(f"Refined grid omitted exact resonance at {resonance_tau_s} s.")
    left = float(resonance_tau_s - surface.taus_s[index - 1]) if index > 0 else None
    right = (
        float(surface.taus_s[index + 1] - resonance_tau_s)
        if index + 1 < surface.taus_s.size
        else None
    )
    return {"left_step_s": left, "right_step_s": right}


def resonance_diagnostics(
    grid_id: str,
    surface: topology.ScoreSurface,
    target_frequency_hz: float,
) -> list[dict]:
    rows = []
    for order, tau in topology.resonant_taus(
        target_frequency_hz,
        float(surface.taus_s[0]),
        float(surface.taus_s[-1]),
    ):
        tau_index = int(np.argmin(np.abs(surface.taus_s - tau)))
        depth_index = int(np.argmin(surface.eta[tau_index]))
        resolution = local_resonance_resolution(surface, tau)
        rows.append(
            {
                "grid_id": grid_id,
                "resonance_order": int(order),
                "resonance_tau_s": float(tau),
                "resonance_tau_us": float(tau * 1e6),
                "best_N_at_resonance": int(surface.depths[depth_index]),
                "eta_at_resonance_minimum": float(surface.eta[tau_index, depth_index]),
                "relative_to_global_best": float(
                    surface.eta[tau_index, depth_index] / surface.eta_min
                ),
                "penalty_percent": float(
                    100.0 * (surface.eta[tau_index, depth_index] / surface.eta_min - 1.0)
                ),
                "left_step_s": resolution["left_step_s"],
                "right_step_s": resolution["right_step_s"],
                "left_step_ns": (
                    resolution["left_step_s"] * 1e9
                    if resolution["left_step_s"] is not None
                    else None
                ),
                "right_step_ns": (
                    resolution["right_step_s"] * 1e9
                    if resolution["right_step_s"] is not None
                    else None
                ),
            }
        )
    return rows


def timing_width_bracket(
    result: topology.ConnectedRegionResult,
    label: int,
) -> dict:
    """Bound a component's timing width using adjacent rejected grid rows."""
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


def characterize_grid(
    grid_id: str,
    surface: topology.ScoreSurface,
    target_frequency_hz: float,
    relative_limit: float,
) -> tuple[dict, list[dict]]:
    result4 = topology.connected_regions(
        surface, relative_limit=relative_limit, connectivity=4
    )
    result8 = topology.connected_regions(
        surface, relative_limit=relative_limit, connectivity=8
    )
    regions = topology.characterize_regions(
        result4, target_frequency_hz=target_frequency_hz
    )
    region_records = [stage2.region_record(region) for region in regions]
    best_row, best_column = np.unravel_index(np.argmin(surface.eta), surface.eta.shape)
    primary = next(region for region in regions if region.contains_global_optimum)
    width_bracket = timing_width_bracket(result4, primary.label)
    row = {
        "grid_id": grid_id,
        "tau_count": int(surface.taus_s.size),
        "N_count": int(surface.depths.size),
        "point_count": int(surface.eta.size),
        "beta_0_four_neighbour": int(result4.component_count),
        "beta_0_eight_neighbour": int(result8.component_count),
        "best_tau_s": float(surface.taus_s[best_row]),
        "best_tau_us": float(surface.taus_s[best_row] * 1e6),
        "best_N": int(surface.depths[best_column]),
        "eta_best": float(surface.eta[best_row, best_column]),
        "primary_tau_min_s": float(primary.tau_min_s),
        "primary_tau_max_s": float(primary.tau_max_s),
        "primary_tau_width_s": float(primary.tau_width_s),
        "primary_tau_min_us": float(primary.tau_min_s * 1e6),
        "primary_tau_max_us": float(primary.tau_max_s * 1e6),
        "primary_tau_width_us": float(primary.tau_width_s * 1e6),
        "primary_tau_width_lower_bound_us": (
            width_bracket["lower_bound_s"] * 1e6
            if width_bracket["lower_bound_s"] is not None
            else None
        ),
        "primary_tau_width_upper_bound_us": (
            width_bracket["upper_bound_s"] * 1e6
            if width_bracket["upper_bound_s"] is not None
            else None
        ),
        "primary_tau_width_estimate_us": (
            width_bracket["estimate_s"] * 1e6
            if width_bracket["estimate_s"] is not None
            else None
        ),
        "primary_tau_width_half_interval_us": (
            width_bracket["half_interval_s"] * 1e6
            if width_bracket["half_interval_s"] is not None
            else None
        ),
        "primary_N_min": int(primary.depth_min),
        "primary_N_max": int(primary.depth_max),
        "primary_sampled_tau_count": int(primary.sampled_tau_count),
        "primary_sampled_depth_count": int(primary.sampled_depth_count),
        "primary_boundary_status": primary.boundary_status,
        "primary_sampling_status": primary.sampling_status,
        "primary_resonance_identity": primary.resonance_identity,
    }
    detail = {
        "grid_id": grid_id,
        "summary": row,
        "regions": region_records,
    }
    return detail, resonance_diagnostics(grid_id, surface, target_frequency_hz)


def build_analysis(
    calibration_path: str | Path,
    canonical_path: str | Path,
    *,
    target_frequency_hz: float = 200e3,
    relative_limit: float = 1.10,
    medium_background_count: int = 161,
    medium_bracket_count: int = 401,
    fine_background_count: int = 321,
    fine_bracket_count: int = 801,
    resonance_halfwidth_fraction: float = 0.08,
    width_relative_tolerance: float = 0.05,
) -> tuple[list[dict], list[dict], dict]:
    calibration_source = Path(calibration_path)
    calibration = model.load_calibration(calibration_source)
    canonical_surface, metadata, canonical_digest = stage2.load_canonical_sweep(
        canonical_path
    )
    tau_min = float(canonical_surface.taus_s[0])
    tau_max = float(canonical_surface.taus_s[-1])
    target = float(target_frequency_hz)
    depths = canonical_surface.depths

    medium_taus = topology.build_resonance_refined_tau_grid(
        target,
        tau_min,
        tau_max,
        background_count=medium_background_count,
        resonance_halfwidth_fraction=resonance_halfwidth_fraction,
        resonance_point_count=medium_bracket_count,
    )
    fine_taus = topology.build_resonance_refined_tau_grid(
        target,
        tau_min,
        tau_max,
        background_count=fine_background_count,
        resonance_halfwidth_fraction=resonance_halfwidth_fraction,
        resonance_point_count=fine_bracket_count,
    )
    surfaces = {
        "canonical": canonical_surface,
        "medium": evaluate_surface(calibration, medium_taus, depths, target),
        "fine": evaluate_surface(calibration, fine_taus, depths, target),
    }

    grid_details = []
    resonance_rows = []
    for grid_id, surface in surfaces.items():
        detail, rows = characterize_grid(grid_id, surface, target, relative_limit)
        grid_details.append(detail)
        resonance_rows.extend(rows)

    by_id = {detail["grid_id"]: detail for detail in grid_details}
    medium = by_id["medium"]["summary"]
    fine = by_id["fine"]["summary"]
    fine_width = fine["primary_tau_width_estimate_us"]
    medium_width = medium["primary_tau_width_estimate_us"]
    width_relative_change = (
        abs(medium_width - fine_width) / fine_width
        if fine_width is not None
        and medium_width is not None
        and fine_width > 0.0
        else None
    )
    component_counts = [
        detail["summary"]["beta_0_four_neighbour"] for detail in grid_details
    ]
    topology_stable = len(set(component_counts)) == 1
    connectivity_stable = all(
        detail["summary"]["beta_0_four_neighbour"]
        == detail["summary"]["beta_0_eight_neighbour"]
        for detail in grid_details
    )
    optimum_stable = all(
        detail["summary"]["best_N"] == fine["best_N"]
        and np.isclose(
            detail["summary"]["best_tau_s"],
            fine["best_tau_s"],
            rtol=1e-9,
            atol=1e-15,
        )
        for detail in grid_details
    )
    width_converged = bool(
        width_relative_change is not None
        and width_relative_change <= float(width_relative_tolerance)
    )
    no_boundary_contact = all(
        region["boundary_status"] == "resolved_interior"
        for region in by_id["fine"]["regions"]
    )
    fine_timing_resolved = all(
        region["sampled_tau_count"] > 1 for region in by_id["fine"]["regions"]
    )
    checks = {
        "component_count_stable_across_grids": topology_stable,
        "four_and_eight_neighbour_counts_agree": connectivity_stable,
        "global_optimum_stable": optimum_stable,
        "medium_to_fine_timing_width_converged": width_converged,
        "fine_regions_avoid_sweep_boundaries": no_boundary_contact,
        "fine_regions_have_multiple_timing_samples": fine_timing_resolved,
    }
    config = {
        "canonical_source_sha256": canonical_digest,
        "calibration_source_sha256": hashlib.sha256(
            calibration_source.read_bytes()
        ).hexdigest(),
        "target_frequency_hz": target,
        "relative_limit": float(relative_limit),
        "medium_background_count": int(medium_background_count),
        "medium_bracket_count": int(medium_bracket_count),
        "fine_background_count": int(fine_background_count),
        "fine_bracket_count": int(fine_bracket_count),
        "resonance_halfwidth_fraction": float(resonance_halfwidth_fraction),
        "width_relative_tolerance": float(width_relative_tolerance),
    }
    summary = {
        "schema_version": "1.0",
        "status": "grid_refinement_passed" if all(checks.values()) else "grid_refinement_attention_required",
        "configuration_digest": stage2.configuration_digest(config),
        "calibration_digest": metadata.get("calibration_digest", ""),
        **config,
        "checks": checks,
        "medium_to_fine_width_relative_change": width_relative_change,
        "grid_results": grid_details,
        "interpretation_limit": (
            "This establishes numerical stability on the tested sampled grids. "
            "It does not establish stability under measurement or hardware uncertainty."
        ),
    }
    return [detail["summary"] for detail in grid_details], resonance_rows, summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path.name}.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(outdir: str | Path, grid_rows, resonance_rows, summary) -> tuple[Path, Path, Path]:
    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    grid_path = destination / GRID_FILENAME
    resonance_path = destination / RESONANCE_FILENAME
    summary_path = destination / SUMMARY_FILENAME
    _write_csv(grid_path, grid_rows)
    _write_csv(resonance_path, resonance_rows)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return grid_path, resonance_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--mirror-outdir", type=Path)
    parser.add_argument("--frequency-khz", type=float, default=200.0)
    parser.add_argument("--relative-limit", type=float, default=1.10)
    args = parser.parse_args()

    grid_rows, resonance_rows, summary = build_analysis(
        args.calibration,
        args.canonical,
        target_frequency_hz=args.frequency_khz * 1e3,
        relative_limit=args.relative_limit,
    )
    destinations = [args.outdir]
    if args.mirror_outdir is not None:
        destinations.append(args.mirror_outdir)
    for destination in destinations:
        paths = write_outputs(destination, grid_rows, resonance_rows, summary)
        print(f"Wrote {', '.join(str(path) for path in paths)}")
    fine = next(row for row in grid_rows if row["grid_id"] == "fine")
    print(
        f"Stage 3 status: {summary['status']}; fine beta_0="
        f"{fine['beta_0_four_neighbour']}, timing width estimate="
        f"{fine['primary_tau_width_estimate_us']:.6f} +/- "
        f"{fine['primary_tau_width_half_interval_us']:.6f} us"
    )


if __name__ == "__main__":
    main()
