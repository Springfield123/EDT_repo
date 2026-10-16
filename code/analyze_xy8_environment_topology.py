#!/usr/bin/env python3
"""Analyze Stage-4 connected regions and predeclare Stage-5 fine-grid cells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "code" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import xy8_environment_scan_config as scan_config
import xy8_environment_topology as environment_topology
import xy8_nv_environment_literature as literature


DEFAULT_OUTDIR = ROOT / "data" / "processed"
THRESHOLDS_CSV = "xy8_environment_topology_thresholds.csv"
COMPONENTS_CSV = "xy8_environment_components.csv"
PERSISTENCE_CSV = "xy8_environment_persistence.csv"
SELECTION_CSV = "xy8_environment_confirmation_selection.csv"
SUMMARY_JSON = "xy8_environment_topology_summary.json"
CONFIRMATION_THRESHOLDS_CSV = "xy8_environment_confirmation_thresholds.csv"
CONFIRMATION_COMPONENTS_CSV = "xy8_environment_confirmation_components.csv"
CONFIRMATION_PERSISTENCE_CSV = "xy8_environment_confirmation_persistence.csv"
CONFIRMATION_RESULTS_CSV = "xy8_environment_confirmation_results.csv"
CONFIRMATION_SURFACES_NPZ = "xy8_environment_confirmation_surfaces.npz"


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_discovery():
    freeze = scan_config.verify_freeze_record()
    if not freeze["verified"]:
        raise RuntimeError("Frozen environment scan configuration did not verify.")
    config = scan_config.load_config()
    rows, eta, taus, depths, stage4_metadata = environment_topology.load_stage4_artifacts()
    relative_limits = [float(value) for value in config["topology_analysis"]["relative_limits"]]
    thresholds, components, tracks = environment_topology.analyze_discovery_surfaces(
        rows,
        eta,
        taus,
        depths,
        target_frequency_hz=config["signal_and_control"]["target_frequency_Hz"],
        relative_limits=relative_limits,
    )
    selection = environment_topology.select_confirmation_surfaces(
        rows,
        thresholds,
        primary_relative_limit=config["topology_analysis"]["primary_relative_limit"],
    )
    primary = [
        row for row in thresholds
        if abs(float(row["relative_limit"]) - config["topology_analysis"]["primary_relative_limit"]) < 1e-12
    ]
    summary = {
        "schema_version": "1.0",
        "stage": "discovery_topology_and_confirmation_selection",
        "configuration_id": config["configuration_id"],
        "configuration_sha256": freeze["configuration_sha256"],
        "stage4_surfaces_sha256": stage4_metadata["surfaces_npz_sha256"],
        "discovery": {
            "surface_count": len(rows),
            "threshold_count": len(relative_limits),
            "threshold_record_count": len(thresholds),
            "component_record_count": len(components),
            "persistence_track_count": len(tracks),
            "primary_beta_0_four_distribution": {
                str(value): sum(int(row["beta_0_four"]) == value for row in primary)
                for value in sorted({int(row["beta_0_four"]) for row in primary})
            },
            "primary_four_eight_disagreement_count": sum(
                not bool(row["connectivity_agrees"]) for row in primary
            ),
        },
        "confirmation_selection": {
            "selected_surface_count": len(selection),
            "preregistered_surface_count": sum(
                bool(row["selected_by_preregistered_rule"]) for row in selection
            ),
            "supplementary_exact_N_surface_count": sum(
                bool(row["selected_by_supplementary_exact_N_rule"]) for row in selection
            ),
            "selection_is_frozen_before_confirmation_scores": True,
        },
        "confirmation_status": "not_yet_run",
    }
    return thresholds, components, tracks, selection, summary


def write_discovery(outdir: str | Path = DEFAULT_OUTDIR):
    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    thresholds, components, tracks, selection, summary = run_discovery()
    paths = {
        "thresholds": destination / THRESHOLDS_CSV,
        "components": destination / COMPONENTS_CSV,
        "persistence": destination / PERSISTENCE_CSV,
        "selection": destination / SELECTION_CSV,
        "summary": destination / SUMMARY_JSON,
    }
    _write_csv(paths["thresholds"], thresholds)
    _write_csv(paths["components"], components)
    _write_csv(paths["persistence"], tracks)
    _write_csv(paths["selection"], selection)
    summary["artifacts"] = {
        key: str(path.relative_to(ROOT)) for key, path in paths.items() if key != "summary"
    }
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return paths, summary


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_confirmation(outdir: str | Path = DEFAULT_OUTDIR):
    destination = Path(outdir)
    freeze = scan_config.verify_freeze_record()
    selection_freeze = environment_topology.verify_confirmation_selection_freeze()
    literature_freeze = literature.verify_frozen_bundle()
    if not freeze["verified"] or not selection_freeze["verified"] or not literature_freeze["verified"]:
        raise RuntimeError("A frozen Stage-5 dependency failed verification.")
    config = scan_config.load_config()
    selection = _read_csv(destination / SELECTION_CSV)
    discovery_thresholds = _read_csv(destination / THRESHOLDS_CSV)
    scenario_path = ROOT / config["frozen_dependencies"]["nominal_hardware_scenario"]["path"]
    nominal_scenario = json.loads(scenario_path.read_text(encoding="utf-8"))

    grid_surfaces = {}
    grid_references = {}
    grid_diagnostics = {}
    all_thresholds = []
    all_components = []
    all_tracks = []
    relative_limits = [float(value) for value in config["topology_analysis"]["relative_limits"]]
    target = float(config["signal_and_control"]["target_frequency_Hz"])
    for grid_stage in ("medium", "fine"):
        surfaces, reference, diagnostics = environment_topology.build_confirmation_surfaces_for_grid(
            selection,
            config=config,
            nominal_scenario=nominal_scenario,
            grid_stage=grid_stage,
        )
        grid_surfaces[grid_stage] = surfaces
        grid_references[grid_stage] = reference
        grid_diagnostics[grid_stage] = diagnostics
        for selection_index, selected in enumerate(selection):
            source_index = int(selected["discovery_surface_index"])
            thresholds, components, tracks = environment_topology.analyze_surface_thresholds(
                surfaces[selection_index],
                reference.taus_s,
                reference.depths_N,
                target_frequency_hz=target,
                relative_limits=relative_limits,
                stage=grid_stage,
                surface_index=source_index,
                source_row=selected,
            )
            all_thresholds.extend(thresholds)
            all_components.extend(components)
            all_tracks.extend(tracks)

    medium_thresholds = [row for row in all_thresholds if row["grid_stage"] == "medium"]
    fine_thresholds = [row for row in all_thresholds if row["grid_stage"] == "fine"]
    results = environment_topology.confirmation_stability_records(
        selection,
        discovery_thresholds,
        medium_thresholds,
        fine_thresholds,
        primary_relative_limit=config["topology_analysis"]["primary_relative_limit"],
        width_relative_tolerance=0.05,
    )

    paths = {
        "thresholds": destination / CONFIRMATION_THRESHOLDS_CSV,
        "components": destination / CONFIRMATION_COMPONENTS_CSV,
        "persistence": destination / CONFIRMATION_PERSISTENCE_CSV,
        "results": destination / CONFIRMATION_RESULTS_CSV,
        "surfaces": destination / CONFIRMATION_SURFACES_NPZ,
        "summary": destination / SUMMARY_JSON,
    }
    _write_csv(paths["thresholds"], all_thresholds)
    _write_csv(paths["components"], all_components)
    _write_csv(paths["persistence"], all_tracks)
    _write_csv(paths["results"], results)
    np.savez_compressed(
        paths["surfaces"],
        eta_fine=grid_surfaces["fine"],
        taus_fine_s=grid_references["fine"].taus_s,
        depths_N=grid_references["fine"].depths_N,
        discovery_surface_indices=np.array(
            [int(row["discovery_surface_index"]) for row in selection], dtype=int
        ),
    )

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    fine_beta_distribution = {}
    for value in sorted({int(row["fine_beta_0_four"]) for row in results}):
        fine_beta_distribution[str(value)] = sum(
            int(row["fine_beta_0_four"]) == value for row in results
        )
    summary.update(
        {
            "stage": "environment_topology_confirmed",
            "confirmation_status": "complete",
            "confirmation": {
                "selection_freeze_sha256": hashlib.sha256(
                    environment_topology.DEFAULT_SELECTION_FREEZE.read_bytes()
                ).hexdigest(),
                "selected_surface_count": len(selection),
                "medium_grid": grid_diagnostics["medium"],
                "fine_grid": grid_diagnostics["fine"],
                "confirmation_pass_count": sum(
                    bool(row["confirmation_passed"]) for row in results
                ),
                "confirmation_failure_count": sum(
                    not bool(row["confirmation_passed"]) for row in results
                ),
                "exact_N_change_survives_count": sum(
                    int(row["discovery_best_N"]) == int(row["fine_best_N"])
                    for row in results
                    if str(row["selected_by_supplementary_exact_N_rule"]).lower() == "true"
                ),
                "supplementary_exact_N_selected_count": sum(
                    str(row["selected_by_supplementary_exact_N_rule"]).lower() == "true"
                    for row in results
                ),
                "fine_primary_beta_0_four_distribution": fine_beta_distribution,
                "fine_four_eight_disagreement_count": sum(
                    int(row["fine_beta_0_four"]) != int(row["fine_beta_0_eight"])
                    for row in results
                ),
                "fine_best_N_distribution": {
                    str(value): sum(int(row["fine_best_N"]) == value for row in results)
                    for value in sorted({int(row["fine_best_N"]) for row in results})
                },
                "artifacts": {
                    key: {
                        "path": str(path.relative_to(ROOT)),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for key, path in paths.items()
                    if key != "summary"
                },
            },
        }
    )
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return paths, summary


def finalize_confirmation(outdir: str | Path = DEFAULT_OUTDIR):
    """Recompute stability diagnostics from frozen, already-evaluated grids."""
    destination = Path(outdir)
    selection_freeze = environment_topology.verify_confirmation_selection_freeze()
    if not selection_freeze["verified"]:
        raise RuntimeError("Frozen confirmation selection failed verification.")
    config = scan_config.load_config()
    selection = _read_csv(destination / SELECTION_CSV)
    discovery_thresholds = _read_csv(destination / THRESHOLDS_CSV)
    confirmation_thresholds = _read_csv(destination / CONFIRMATION_THRESHOLDS_CSV)
    results = environment_topology.confirmation_stability_records(
        selection,
        discovery_thresholds,
        [row for row in confirmation_thresholds if row["grid_stage"] == "medium"],
        [row for row in confirmation_thresholds if row["grid_stage"] == "fine"],
        primary_relative_limit=config["topology_analysis"]["primary_relative_limit"],
        width_relative_tolerance=0.05,
    )
    results_path = destination / CONFIRMATION_RESULTS_CSV
    _write_csv(results_path, results)
    summary_path = destination / SUMMARY_JSON
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    confirmation = summary["confirmation"]
    confirmation["confirmation_pass_count"] = sum(
        bool(row["confirmation_passed"]) for row in results
    )
    confirmation["confirmation_failure_count"] = sum(
        not bool(row["confirmation_passed"]) for row in results
    )
    confirmation["exact_N_change_survives_count"] = sum(
        int(row["discovery_best_N"]) == int(row["fine_best_N"])
        for row in results
        if str(row["selected_by_supplementary_exact_N_rule"]).lower() == "true"
    )
    confirmation["maximum_discovery_to_fine_best_tau_shift_ns"] = 1e9 * max(
        float(row["discovery_to_fine_best_tau_shift_s"]) for row in results
    )
    confirmation["maximum_medium_to_fine_best_tau_shift_ns"] = 1e9 * max(
        float(row["medium_to_fine_best_tau_shift_s"]) for row in results
    )
    confirmation["maximum_medium_to_fine_width_relative_change"] = max(
        float(row["medium_to_fine_width_relative_change"])
        for row in results
        if row["medium_to_fine_width_relative_change"] not in (None, "")
    )
    confirmation["artifacts"]["results"]["sha256"] = hashlib.sha256(
        results_path.read_bytes()
    ).hexdigest()
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results_path, summary_path, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Run the frozen medium/fine confirmation instead of rewriting discovery selection.",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Recompute stability diagnostics from existing medium/fine artifacts.",
    )
    args = parser.parse_args()
    if args.finalize_only:
        results_path, summary_path, summary = finalize_confirmation(args.outdir)
        print(f"Updated {results_path}")
        print(f"Updated {summary_path}")
        confirmation = summary["confirmation"]
        print(
            f"passed={confirmation['confirmation_pass_count']}, "
            f"failed={confirmation['confirmation_failure_count']}"
        )
        return
    if args.confirm:
        paths, summary = run_confirmation(args.outdir)
        print(f"Wrote {paths['thresholds']}")
        print(f"Wrote {paths['components']}")
        print(f"Wrote {paths['persistence']}")
        print(f"Wrote {paths['results']}")
        print(f"Wrote {paths['surfaces']}")
        print(f"Updated {paths['summary']}")
        confirmation = summary["confirmation"]
        print(
            f"passed={confirmation['confirmation_pass_count']}, "
            f"failed={confirmation['confirmation_failure_count']}"
        )
        return
    paths, summary = write_discovery(args.outdir)
    print(f"Wrote {paths['thresholds']}")
    print(f"Wrote {paths['components']}")
    print(f"Wrote {paths['persistence']}")
    print(f"Wrote {paths['selection']}")
    print(f"Wrote {paths['summary']}")
    print(
        f"selected={summary['confirmation_selection']['selected_surface_count']}, "
        f"preregistered={summary['confirmation_selection']['preregistered_surface_count']}"
    )


if __name__ == "__main__":
    main()
