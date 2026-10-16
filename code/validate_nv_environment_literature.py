#!/usr/bin/env python3
"""Validate the frozen NV temperature inputs before any EDT calculation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = CODE_DIR / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import xy8_nv_environment_literature as literature


ROOT = CODE_DIR.parent
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "nv_environment_literature_validation.json"


def build_report() -> dict:
    laws = literature.load_temperature_laws()
    freeze = literature.verify_frozen_bundle()
    held_out = literature.validate_tang_against_acosta(laws)
    cambria_local = literature.numerical_slope_kHz_per_K(
        lambda temperature: literature.cambria_zfs_MHz(temperature, laws), 300.0
    )
    cambria_interval = literature.interval_slope_kHz_per_K(
        lambda temperature: literature.cambria_zfs_MHz(temperature, laws), 280.0, 330.0
    )
    curve = literature.load_mondal_t1_curve()
    ready = bool(freeze["verified"] and held_out["within_two_measurement_sigma"])
    return {
        "schema_version": 1,
        "analysis_gate": {
            "status": "ready_for_edt_environment_analysis" if ready else "not_ready",
            "criteria": {
                "frozen_bundle_verified": freeze["verified"],
                "independent_D_temperature_validation_within_2sigma": held_out[
                    "within_two_measurement_sigma"
                ],
            },
            "important_scope_note": (
                "Passing this gate checks provenance and one independent D(T) trend. "
                "It does not validate an environment-conditioned EDT prediction."
            ),
        },
        "frozen_bundle": freeze,
        "independent_validation": held_out,
        "analytical_law_diagnostics": {
            "cambria_local_slope_at_300K_kHz_per_K": cambria_local,
            "cambria_secant_slope_280_to_330K_kHz_per_K": cambria_interval,
            "diagnostic_role": "reported for transparency; not used as the held-out test",
        },
        "digitized_input_diagnostics": {
            "source_id": "mondal_2023_spin_phonon",
            "point_count": len(curve),
            "temperature_range_K": [curve[0]["temperature_K"], curve[-1]["temperature_K"]],
            "uncertainty_scope": "vector-coordinate extraction only",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["analysis_gate"]["status"] != "ready_for_edt_environment_analysis":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
