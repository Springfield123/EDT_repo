#!/usr/bin/env python3
"""Generate Stage-4 environment-conditioned XY8 discovery score surfaces."""

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

import xy8_environment_coherence as coherence
import xy8_environment_score as score
import xy8_environment_scan_config as scan_config
import xy8_nv_environment_literature as literature


DEFAULT_OUTDIR = ROOT / "data" / "processed"
SUMMARY_CSV = "xy8_environment_score_summary.csv"
SURFACES_NPZ = "xy8_environment_discovery_surfaces.npz"
METADATA_JSON = "xy8_environment_score_metadata.json"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_analysis():
    scan_freeze = scan_config.verify_freeze_record()
    literature_freeze = literature.verify_frozen_bundle()
    if not scan_freeze["verified"] or not literature_freeze["verified"]:
        raise RuntimeError("Frozen scan configuration or literature bundle did not verify.")
    config = scan_config.load_config()
    canonical = score.load_canonical_surface()
    expected_depths = np.asarray(config["control_grids"]["depths_N"], dtype=int)
    if not np.array_equal(canonical.depths_N, expected_depths):
        raise RuntimeError("Canonical depths no longer match the frozen scan configuration.")
    expected_tau_count = config["control_grids"]["discovery_timing_grid"][
        "expected_tau_count_at_200_kHz"
    ]
    if canonical.taus_s.size != expected_tau_count:
        raise RuntimeError("Canonical timing grid no longer matches the frozen discovery grid.")

    scenario_path = ROOT / config["frozen_dependencies"]["nominal_hardware_scenario"]["path"]
    nominal_scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    nominal_t2 = dict(nominal_scenario["coherence"])
    points = score.build_operational_environment_points(config)
    detunings = np.array(
        [point.transition.fixed_reference_carrier_detuning_Hz for point in points], dtype=float
    )
    unique_detunings, inverse_detuning = np.unique(detunings, return_inverse=True)
    if not np.any(np.isclose(unique_detunings, 0.0, rtol=0.0, atol=1e-6)):
        unique_detunings = np.concatenate([unique_detunings, [0.0]])

    gains = score.batched_deterministic_xy8_gains(
        canonical.taus_s,
        canonical.depths_N,
        unique_detunings,
        signal_frequency_hz=config["signal_and_control"]["target_frequency_Hz"],
        pi_pulse_duration_s=config["signal_and_control"]["pi_pulse_duration_s"],
        pulse_substeps=8,
    )
    zero_index = int(np.argmin(np.abs(unique_detunings)))
    zero_gain = gains[zero_index]
    gain_floor = 1e-12
    ratios = np.maximum(gains, gain_floor) / np.maximum(zero_gain[None, :, :], gain_floor)
    clipped_gain_count = int(np.count_nonzero(gains < gain_floor))

    temperature_surfaces = {}
    coherence_results = {}
    for variant in coherence.SUPPORTED_VARIANTS:
        for temperature in config["temperature_axes"]["operational"]["values_K"]:
            conditioned, result = score.coherence_conditioned_surface(
                canonical,
                temperature,
                variant=variant,
                nominal_t2_params=nominal_t2,
            )
            temperature_surfaces[(variant, float(temperature))] = conditioned
            coherence_results[(variant, float(temperature))] = result

    rows = []
    surfaces = []
    relative_limit = config["topology_analysis"]["primary_relative_limit"]
    for point_index, point in enumerate(points):
        detuning_index = int(inverse_detuning[point_index])
        for variant in coherence.SUPPORTED_VARIANTS:
            base = temperature_surfaces[(variant, point.temperature_K)]
            coherence_result = coherence_results[(variant, point.temperature_K)]
            for carrier_policy in ("fixed_reference_carrier", "ideal_retracked_carrier"):
                if carrier_policy == "fixed_reference_carrier":
                    environment_surface = base / ratios[detuning_index]
                    applied_detuning = point.transition.fixed_reference_carrier_detuning_Hz
                else:
                    environment_surface = base.copy()
                    applied_detuning = 0.0
                summary = score.summarize_surface(
                    environment_surface, canonical, relative_limit=relative_limit
                )
                surface_index = len(surfaces)
                surfaces.append(environment_surface)
                rows.append(
                    {
                        "surface_index": surface_index,
                        "condition_id": point.condition_id,
                        "scan_family": point.scan_family,
                        "strain_path": point.strain_path,
                        "temperature_K": point.temperature_K,
                        "amplitude_microstrain": point.amplitude_microstrain,
                        "transition_frequency_MHz": point.transition.transition_frequency_MHz,
                        "transition_shift_kHz": point.transition.transition_shift_Hz / 1e3,
                        "state_tracking_minimum_overlap": point.transition.spectrum.minimum_assigned_overlap,
                        "state_tracking_ambiguous": point.transition.spectrum.ambiguous,
                        "transition_warnings": ";".join(point.transition.warnings),
                        "carrier_policy": carrier_policy,
                        "applied_detuning_Hz": applied_detuning,
                        "coherence_variant": variant,
                        "T1_ms": coherence_result.t1_s * 1e3,
                        "T2_1_effective_us": coherence_result.effective_T2_s * 1e6,
                        "relative_limit": relative_limit,
                        **summary,
                    }
                )
    eta_surfaces = np.stack(surfaces)
    reference_rows = [
        row
        for row in rows
        if row["temperature_K"] == 300.0
        and row["amplitude_microstrain"] == 0.0
    ]
    reference_max_error = max(
        float(np.max(np.abs(eta_surfaces[row["surface_index"]] - canonical.eta)))
        for row in reference_rows
    )
    metadata = {
        "schema_version": "1.0",
        "stage": "environment_conditioned_discovery_score_surfaces",
        "configuration_id": config["configuration_id"],
        "configuration_sha256": scan_freeze["configuration_sha256"],
        "literature_bundle_sha256": literature_freeze["bundle_sha256"],
        "canonical_surface_sha256": file_sha256(score.DEFAULT_CANONICAL_SURFACE),
        "score_rule": "eta_env = eta_canonical * coherence_reference/coherence_environment * G(0)/G(detuning)",
        "coherent_response": {
            "method": "batched deterministic finite-pulse two-quadrature small-signal derivative",
            "pi_pulse_duration_s": config["signal_and_control"]["pi_pulse_duration_s"],
            "pulse_substeps": 8,
            "gain_floor": gain_floor,
            "clipped_gain_count": clipped_gain_count,
            "unique_detuning_count": int(unique_detunings.size),
            "detuning_min_Hz": float(np.min(unique_detunings)),
            "detuning_max_Hz": float(np.max(unique_detunings)),
        },
        "dimensions": {
            "environment_condition_count": len(points),
            "coherence_variant_count": len(coherence.SUPPORTED_VARIANTS),
            "carrier_policy_count": 2,
            "surface_count": len(rows),
            "tau_count": canonical.taus_s.size,
            "depth_count": canonical.depths_N.size,
            "point_count": int(eta_surfaces.size),
        },
        "reference_regression": {
            "row_count": len(reference_rows),
            "maximum_absolute_eta_error": reference_max_error,
            "expected_best_tau_s": 1.25e-6,
            "expected_best_N": 20,
            "expected_window_N_min": 10,
            "expected_window_N_max": 34,
            "all_reference_rows_match_expected_summary": all(
                row["best_tau_s"] == 1.25e-6
                and row["best_N"] == 20
                and row["window_N_min"] == 10
                and row["window_N_max"] == 34
                for row in reference_rows
            ),
        },
        "interpretation_limit": (
            "These are literature-conditioned model surfaces, not measured sensitivities. "
            "Topology stability requires the preregistered confirmation-grid stage."
        ),
    }
    return rows, eta_surfaces, canonical, unique_detunings, gains, metadata


def write_outputs(outdir: str | Path = DEFAULT_OUTDIR):
    rows, surfaces, canonical, detunings, gains, metadata = build_analysis()
    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / SUMMARY_CSV
    surfaces_path = destination / SURFACES_NPZ
    metadata_path = destination / METADATA_JSON
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        surfaces_path,
        eta=surfaces,
        taus_s=canonical.taus_s,
        depths_N=canonical.depths_N,
        coherent_detunings_Hz=detunings,
        coherent_gains=gains,
    )
    metadata["summary_csv_sha256"] = file_sha256(summary_path)
    metadata["surfaces_npz_sha256"] = file_sha256(surfaces_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_path, surfaces_path, metadata_path, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    summary, surfaces, metadata_path, metadata = write_outputs(args.outdir)
    print(f"Wrote {summary}")
    print(f"Wrote {surfaces}")
    print(f"Wrote {metadata_path}")
    dimensions = metadata["dimensions"]
    print(
        f"surfaces={dimensions['surface_count']}, points={dimensions['point_count']}, "
        f"reference_error={metadata['reference_regression']['maximum_absolute_eta_error']:.3e}"
    )


if __name__ == "__main__":
    main()
