#!/usr/bin/env python3
"""Export Stage-2 NV transition diagnostics without evaluating EDT scores."""

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

import xy8_environment_conditioning as conditioning
import xy8_environment_scan_config as scan_config
import xy8_nv_environment_literature as literature


DEFAULT_OUTDIR = ROOT / "data" / "processed"
CSV_NAME = "xy8_environment_transition_diagnostics.csv"
JSON_NAME = "xy8_environment_transition_summary.json"


def _record(
    result: conditioning.EnvironmentTransition,
    *,
    family: str,
    strain_path: str,
    amplitude_microstrain: float,
) -> dict:
    strain = result.condition.strain_tensor
    return {
        "scan_family": family,
        "strain_path": strain_path,
        "temperature_K": result.condition.temperature_K,
        "amplitude_microstrain": amplitude_microstrain,
        "epsilon_xx": strain[0, 0],
        "epsilon_yy": strain[1, 1],
        "epsilon_zz": strain[2, 2],
        "epsilon_xy": strain[0, 1],
        "epsilon_xz": strain[0, 2],
        "epsilon_yz": strain[1, 2],
        "transition_frequency_MHz": result.transition_frequency_MHz,
        "transition_shift_kHz": result.transition_shift_Hz / 1e3,
        "fixed_reference_carrier_detuning_Hz": result.fixed_reference_carrier_detuning_Hz,
        "ideal_retracked_carrier_detuning_Hz": result.ideal_retracked_carrier_detuning_Hz,
        "overlap_m_s_plus_1_like": result.spectrum.assigned_overlaps[conditioning.MS_PLUS_INDEX],
        "overlap_m_s_0_like": result.spectrum.assigned_overlaps[conditioning.MS_ZERO_INDEX],
        "overlap_m_s_minus_1_like": result.spectrum.assigned_overlaps[conditioning.MS_MINUS_INDEX],
        "minimum_assigned_overlap": result.spectrum.minimum_assigned_overlap,
        "ambiguous_state_tracking": result.spectrum.ambiguous,
        "warnings": ";".join(result.warnings),
    }


def build_diagnostics(config_path: str | Path = scan_config.DEFAULT_CONFIG) -> tuple[list[dict], dict]:
    config_path = Path(config_path)
    freeze = scan_config.verify_freeze_record(config_path)
    if not freeze["verified"]:
        raise RuntimeError("Environment scan configuration freeze verification failed.")
    literature_freeze = literature.verify_frozen_bundle()
    if not literature_freeze["verified"]:
        raise RuntimeError("Environment literature bundle verification failed.")
    config = scan_config.load_config(config_path)

    rows: list[dict] = []
    temperatures = config["temperature_axes"]["operational"]["values_K"]
    zero_conditions = [
        conditioning.EnvironmentCondition(
            temperature,
            conditioning.strain_tensor_from_path("zero_strain", 0.0, config=config),
            label=f"zero_strain_{temperature:g}K",
        )
        for temperature in temperatures
    ]
    zero_results = conditioning.track_environment_path(
        zero_conditions, config=config, anchor_index=temperatures.index(300.0)
    )
    rows.extend(
        _record(result, family="operational_temperature", strain_path="zero_strain", amplitude_microstrain=0.0)
        for result in zero_results
    )

    amplitudes = config["strain_axes"]["amplitudes_microstrain"]
    for path in ("axial_zz", "transverse_differential", "shear_xz"):
        conditions = [
            conditioning.EnvironmentCondition(
                300.0,
                conditioning.strain_tensor_from_path(path, amplitude, config=config),
                label=f"{path}_{amplitude:g}_microstrain",
            )
            for amplitude in amplitudes
        ]
        results = conditioning.track_environment_path(
            conditions, config=config, anchor_index=amplitudes.index(0.0)
        )
        rows.extend(
            _record(
                result,
                family="strain_at_reference_temperature",
                strain_path=path,
                amplitude_microstrain=amplitude,
            )
            for amplitude, result in zip(amplitudes, results)
        )

    shifts = np.array([abs(float(row["transition_shift_kHz"])) for row in rows])
    overlaps = np.array([float(row["minimum_assigned_overlap"]) for row in rows])
    summary = {
        "schema_version": "1.0",
        "stage": "environment_to_spin_hamiltonian_adapter",
        "edt_score_surfaces_evaluated": False,
        "configuration_id": config["configuration_id"],
        "configuration_sha256": freeze["configuration_sha256"],
        "literature_bundle_sha256": literature_freeze["bundle_sha256"],
        "reference": {
            "temperature_K": 300.0,
            "strain": "zero",
            "transition": config["reference_environment"]["transition"]["name"],
            "transition_frequency_MHz": conditioning.evaluate_transition(
                conditioning.reference_condition(config), config=config
            ).transition_frequency_MHz,
        },
        "diagnostic_row_count": len(rows),
        "scan_families": {
            "operational_temperature_zero_strain": len(temperatures),
            "strain_paths_at_300K": 3 * len(amplitudes),
        },
        "carrier_policies": [policy["id"] for policy in config["carrier_policies"]],
        "maximum_absolute_transition_shift_kHz": float(np.max(shifts)),
        "minimum_assigned_state_overlap": float(np.min(overlaps)),
        "ambiguous_state_count": sum(bool(row["ambiguous_state_tracking"]) for row in rows),
        "warning_row_count": sum(bool(row["warnings"]) for row in rows),
        "interpretation": (
            "These diagnostics validate frequency shifts and state tracking only. "
            "They are not EDT scores, connected-region results, or experimental predictions."
        ),
    }
    return rows, summary


def write_outputs(outdir: str | Path = DEFAULT_OUTDIR) -> tuple[Path, Path, dict]:
    rows, summary = build_diagnostics()
    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / CSV_NAME
    json_path = destination / JSON_NAME
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary["diagnostic_csv_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return csv_path, json_path, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    csv_path, json_path, summary = write_outputs(args.outdir)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(
        f"reference={summary['reference']['transition_frequency_MHz']:.6f} MHz, "
        f"rows={summary['diagnostic_row_count']}, "
        f"ambiguous={summary['ambiguous_state_count']}"
    )


if __name__ == "__main__":
    main()
