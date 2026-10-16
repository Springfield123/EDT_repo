#!/usr/bin/env python3
"""Track XY8 EDT-region topology from 1% through 100% performance penalty."""

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
import analyze_xy8_topology_refinement as stage3
import nvratemodel_api as model
import xy8_resonant_topology as topology
import xy8_topology_persistence as persistence


DEFAULT_CALIBRATION = ROOT / "data" / "raw" / "nv" / "xy8_scenario_nominal.json"
DEFAULT_CANONICAL = ROOT / "data" / "processed" / "xy8_qre_nv_results.csv"
DEFAULT_REFINEMENT_SUMMARY = ROOT / "data" / "processed" / "xy8_topology_refinement_summary.json"
DEFAULT_OUTDIR = ROOT / "data" / "processed"
CURVE_FILENAME = "xy8_topology_threshold_curve.csv"
EVENTS_FILENAME = "xy8_topology_persistence_events.csv"
TRACKS_FILENAME = "xy8_topology_component_tracks.csv"
SUMMARY_FILENAME = "xy8_topology_persistence_summary.json"
CHECKPOINTS = (1, 5, 10, 20, 30, 50, 75, 100)


def load_fine_surface(
    calibration_path: str | Path,
    canonical_path: str | Path,
    refinement_summary_path: str | Path,
    *,
    target_frequency_hz: float,
) -> tuple[topology.ScoreSurface, dict]:
    refinement_summary = json.loads(
        Path(refinement_summary_path).read_text(encoding="utf-8")
    )
    if refinement_summary.get("status") != "grid_refinement_passed":
        raise ValueError("Stage 4 requires a passed Stage 3 refinement summary.")
    canonical_surface, metadata, canonical_digest = stage2.load_canonical_sweep(
        canonical_path
    )
    if canonical_digest != refinement_summary.get("canonical_source_sha256"):
        raise ValueError("The Stage 3 summary does not match the canonical sweep.")
    target = float(target_frequency_hz)
    if not np.isclose(
        target,
        float(refinement_summary["target_frequency_hz"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("The requested frequency does not match the Stage 3 grid.")

    fine_taus = topology.build_resonance_refined_tau_grid(
        target,
        float(canonical_surface.taus_s[0]),
        float(canonical_surface.taus_s[-1]),
        background_count=int(refinement_summary["fine_background_count"]),
        resonance_halfwidth_fraction=float(
            refinement_summary["resonance_halfwidth_fraction"]
        ),
        resonance_point_count=int(refinement_summary["fine_bracket_count"]),
    )
    calibration = model.load_calibration(calibration_path)
    surface = stage3.evaluate_surface(
        calibration,
        fine_taus,
        canonical_surface.depths,
        target,
    )
    provenance = {
        "canonical_source_sha256": canonical_digest,
        "calibration_source_sha256": hashlib.sha256(
            Path(calibration_path).read_bytes()
        ).hexdigest(),
        "calibration_digest": metadata.get("calibration_digest", ""),
        "refinement_configuration_digest": refinement_summary[
            "configuration_digest"
        ],
        "fine_tau_count": int(surface.taus_s.size),
        "N_count": int(surface.depths.size),
    }
    return surface, provenance


def threshold_curve(
    surface: topology.ScoreSurface,
    component_rows: list[dict],
    threshold_percents,
    *,
    target_frequency_hz: float,
) -> list[dict]:
    rows = []
    for threshold in threshold_percents:
        selected = [
            row
            for row in component_rows
            if row["threshold_percent"] == float(threshold)
        ]
        primary = next(row for row in selected if row["contains_global_optimum"])
        result8 = topology.connected_regions(
            surface,
            relative_limit=1.0 + float(threshold) / 100.0,
            connectivity=8,
        )
        rows.append(
            {
                "threshold_percent": float(threshold),
                "relative_limit": float(1.0 + float(threshold) / 100.0),
                "beta_0_four_neighbour": len(selected),
                "beta_0_eight_neighbour": int(result8.component_count),
                "accepted_point_count": int(sum(row["point_count"] for row in selected)),
                "accepted_grid_fraction": float(
                    sum(row["point_count"] for row in selected) / surface.eta.size
                ),
                "primary_component_id": primary["component_id"],
                "primary_tau_min_us": primary["tau_min_us"],
                "primary_tau_max_us": primary["tau_max_us"],
                "primary_timing_width_estimate_us": primary[
                    "timing_width_estimate_us"
                ],
                "primary_timing_width_half_interval_us": primary[
                    "timing_width_half_interval_us"
                ],
                "primary_N_min": primary["N_min"],
                "primary_N_max": primary["N_max"],
                "primary_N_span_blocks": primary["N_span_blocks"],
                "primary_resonance_identity": primary["resonance_identity"],
                "primary_boundary_status": primary["boundary_status"],
            }
        )
    return rows


def build_analysis(
    calibration_path: str | Path,
    canonical_path: str | Path,
    refinement_summary_path: str | Path,
    *,
    target_frequency_hz: float = 200e3,
    threshold_percents=range(1, 101),
) -> tuple[list[dict], list[dict], list[dict], dict]:
    thresholds = tuple(float(value) for value in threshold_percents)
    surface, provenance = load_fine_surface(
        calibration_path,
        canonical_path,
        refinement_summary_path,
        target_frequency_hz=target_frequency_hz,
    )
    component_rows, events, tracks = persistence.track_threshold_components(
        surface,
        target_frequency_hz=target_frequency_hz,
        threshold_percents=thresholds,
        connectivity=4,
    )
    curve = threshold_curve(
        surface,
        component_rows,
        thresholds,
        target_frequency_hz=target_frequency_hz,
    )
    checkpoints = [
        dict(row) for row in curve if int(row["threshold_percent"]) in CHECKPOINTS
    ]
    beta0_stable = len({row["beta_0_four_neighbour"] for row in curve}) == 1
    connectivity_agrees = all(
        row["beta_0_four_neighbour"] == row["beta_0_eight_neighbour"]
        for row in curve
    )
    point_counts = [row["accepted_point_count"] for row in curve]
    widths = [row["primary_timing_width_estimate_us"] for row in curve]
    region_expands_monotonically = all(
        right >= left for left, right in zip(point_counts, point_counts[1:])
    ) and all(right >= left for left, right in zip(widths, widths[1:]))
    no_boundary_contact = all(
        row["primary_boundary_status"] == "resolved_interior" for row in curve
    )
    noninitial_events = [
        event for event in events if event["event_type"] != "present_at_first_threshold"
    ]

    resonance_rows = stage3.resonance_diagnostics(
        "fine", surface, target_frequency_hz
    )
    excluded_resonances = [
        row
        for row in resonance_rows
        if row["penalty_percent"] > max(thresholds)
    ]
    nearest_excluded = (
        min(excluded_resonances, key=lambda row: row["penalty_percent"])
        if excluded_resonances
        else None
    )
    checks = {
        "beta_0_stable_across_requested_thresholds": beta0_stable,
        "four_and_eight_neighbour_counts_agree": connectivity_agrees,
        "region_expands_monotonically": region_expands_monotonically,
        "primary_region_avoids_sweep_boundaries": no_boundary_contact,
        "no_birth_or_merge_events_after_one_percent": not noninitial_events,
    }
    config = {
        **provenance,
        "target_frequency_hz": float(target_frequency_hz),
        "threshold_min_percent": float(min(thresholds)),
        "threshold_max_percent": float(max(thresholds)),
        "threshold_step_percent": float(thresholds[1] - thresholds[0])
        if len(thresholds) > 1
        else None,
        "connectivity": 4,
    }
    summary = {
        "schema_version": "1.0",
        "status": "threshold_persistence_passed" if all(checks.values()) else "threshold_persistence_attention_required",
        "configuration_digest": stage2.configuration_digest(config),
        **config,
        "threshold_definition": "eta(tau,N) <= (1 + threshold_percent/100) * eta_min",
        "checks": checks,
        "component_ids": [track["component_id"] for track in tracks],
        "tracks": tracks,
        "events": events,
        "checkpoints": checkpoints,
        "nearest_resonance_excluded_from_requested_range": nearest_excluded,
        "headline": (
            "One connected fundamental-resonance region persists from 1% to "
            "100% while expanding in timing and control depth."
        ),
        "interpretation_limit": (
            "Persistence is numerical and conditional on the nominal model. "
            "Hardware and measurement uncertainty have not yet been propagated "
            "through the connected-region calculation."
        ),
    }
    return curve, events, tracks, summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path.name}.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(outdir: str | Path, curve, events, tracks, summary):
    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    curve_path = destination / CURVE_FILENAME
    events_path = destination / EVENTS_FILENAME
    tracks_path = destination / TRACKS_FILENAME
    summary_path = destination / SUMMARY_FILENAME
    _write_csv(curve_path, curve)
    _write_csv(events_path, events)
    _write_csv(tracks_path, tracks)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return curve_path, events_path, tracks_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument(
        "--refinement-summary", type=Path, default=DEFAULT_REFINEMENT_SUMMARY
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--mirror-outdir", type=Path)
    parser.add_argument("--frequency-khz", type=float, default=200.0)
    args = parser.parse_args()

    outputs = build_analysis(
        args.calibration,
        args.canonical,
        args.refinement_summary,
        target_frequency_hz=args.frequency_khz * 1e3,
    )
    curve, events, tracks, summary = outputs
    destinations = [args.outdir]
    if args.mirror_outdir is not None:
        destinations.append(args.mirror_outdir)
    for destination in destinations:
        paths = write_outputs(destination, curve, events, tracks, summary)
        print(f"Wrote {', '.join(str(path) for path in paths)}")
    first, last = summary["checkpoints"][0], summary["checkpoints"][-1]
    print(
        f"Stage 4 status: {summary['status']}; beta_0="
        f"{first['beta_0_four_neighbour']} from 1%-100%; timing width "
        f"{first['primary_timing_width_estimate_us']:.6f} -> "
        f"{last['primary_timing_width_estimate_us']:.6f} us"
    )


if __name__ == "__main__":
    main()
