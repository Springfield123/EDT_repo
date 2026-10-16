#!/usr/bin/env python3
"""Export Stage-3 anchored T1/T2 diagnostics without evaluating EDT scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "code" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import xy8_environment_coherence as coherence
import xy8_environment_scan_config as scan_config
import xy8_nv_environment_literature as literature


DEFAULT_OUTDIR = ROOT / "data" / "processed"
CSV_NAME = "xy8_environment_coherence_diagnostics.csv"
JSON_NAME = "xy8_environment_coherence_summary.json"


def build_diagnostics() -> tuple[list[dict], dict]:
    scan_freeze = scan_config.verify_freeze_record()
    literature_freeze = literature.verify_frozen_bundle()
    if not scan_freeze["verified"] or not literature_freeze["verified"]:
        raise RuntimeError("Frozen scan configuration or literature bundle did not verify.")
    config = scan_config.load_config()
    operational = set(config["temperature_axes"]["operational"]["values_K"])
    extended = set(config["temperature_axes"]["extended_stress_test"]["values_K"])
    temperatures = sorted(operational | extended)
    scenario_path = ROOT / config["frozen_dependencies"]["nominal_hardware_scenario"]["path"]
    nominal_scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    nominal_t2_params = dict(nominal_scenario["coherence"])

    rows = []
    for variant in coherence.SUPPORTED_VARIANTS:
        for temperature in temperatures:
            result = coherence.anchored_coherence(
                temperature, variant=variant, config=config
            )
            depth_t2, depth_limited, _ = coherence.sequence_coherence_time_s(
                128,
                config["signal_and_control"]["primary_resonant_tau_s"],
                nominal_t2_params,
                temperature,
                variant=variant,
                config=config,
            )
            rows.append(
                {
                    "variant": variant,
                    "temperature_K": temperature,
                    "in_operational_axis": temperature in operational,
                    "in_extended_axis": temperature in extended,
                    "T1_rate_s_inv": result.t1_rate_s_inv,
                    "T1_rate_reference_300K_s_inv": result.reference_t1_rate_s_inv,
                    "T1_ms": result.t1_s * 1e3,
                    "T1_uncertainty_1sigma_ms": result.t1_uncertainty_1sigma_s * 1e3,
                    "residual_dephasing_rate_s_inv": result.residual_dephasing_rate_s_inv,
                    "T2_1_effective_us": result.effective_T2_s * 1e6,
                    "T2_1_uncertainty_1sigma_us": result.effective_T2_uncertainty_1sigma_s * 1e6,
                    "T2_1_relative_to_300K": result.effective_T2_s / result.reference_T2_s,
                    "T2_le_2T1": result.effective_T2_s <= 2.0 * result.t1_s,
                    "T2_at_N128_primary_tau_us": depth_t2 * 1e6,
                    "T2_at_N128_limited_by_2T1": depth_limited,
                    "warnings": ";".join(result.warnings),
                    "uncertainty_scope": result.uncertainty_scope,
                }
            )

    by_variant = {}
    for variant in coherence.SUPPORTED_VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        reference = next(row for row in selected if row["temperature_K"] == 300.0)
        by_variant[variant] = {
            "row_count": len(selected),
            "reference_T2_1_us": reference["T2_1_effective_us"],
            "minimum_T2_1_us": min(row["T2_1_effective_us"] for row in selected),
            "maximum_T2_1_us": max(row["T2_1_effective_us"] for row in selected),
            "minimum_T1_ms": min(row["T1_ms"] for row in selected),
            "maximum_T1_ms": max(row["T1_ms"] for row in selected),
            "warning_row_count": sum(bool(row["warnings"]) for row in selected),
            "all_T2_le_2T1": all(row["T2_le_2T1"] for row in selected),
            "all_N128_T2_le_2T1": all(
                row["T2_at_N128_primary_tau_us"] <= 2.0 * row["T1_ms"] * 1e3 + 1e-9
                for row in selected
            ),
            "N128_T1_limited_temperature_count": sum(
                bool(row["T2_at_N128_limited_by_2T1"]) for row in selected
            ),
        }
    summary = {
        "schema_version": "1.0",
        "stage": "anchored_temperature_dependent_coherence",
        "edt_score_surfaces_evaluated": False,
        "configuration_id": config["configuration_id"],
        "configuration_sha256": scan_freeze["configuration_sha256"],
        "literature_bundle_sha256": literature_freeze["bundle_sha256"],
        "anchoring_rule": "1/T2_1(T)=1/T2_1(300K)+[R1(T)-R1(300K)]/2",
        "reference_temperature_K": 300.0,
        "reference_T2_1_us": config["relaxation_policy"]["reference_T2_s"] * 1e6,
        "temperature_count": len(temperatures),
        "diagnostic_row_count": len(rows),
        "variants": by_variant,
        "separation_rule": (
            "Jarmola experimental and Mondal first-principles T1 sources are separate "
            "sensitivity variants; they are not averaged, combined, or fitted."
        ),
        "double_counting_guard": config["relaxation_policy"]["double_counting_guard"],
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
    for variant, values in summary["variants"].items():
        print(
            f"{variant}: T2_1={values['minimum_T2_1_us']:.3f}-"
            f"{values['maximum_T2_1_us']:.3f} us"
        )


if __name__ == "__main__":
    main()
