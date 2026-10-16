#!/usr/bin/env python3
"""Analyze independent Stage 1 NV SPAM measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "code" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import xy8_spam_calibration as calibration


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate effective preparation/readout quantities from the frozen "
            "shot-level protocol without reading the XY8 depth holdout."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "data/raw/nv/calibration/xy8_spam_calibration_protocol.json",
    )
    parser.add_argument("--shots", type=Path, required=True)
    parser.add_argument(
        "--physical-characterization",
        type=Path,
        help=(
            "Independent initialization/preparation/analysis estimates and covariance. "
            "When omitted, only the effective Stage 1 summary is written."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/xy8_spam_stage1_summary.json",
    )
    parser.add_argument(
        "--calibration-output",
        type=Path,
        default=ROOT / "data/raw/nv/calibration/xy8_spam_calibration.json",
    )
    arguments = parser.parse_args()

    additional_sources = (
        [arguments.physical_characterization]
        if arguments.physical_characterization is not None
        else []
    )
    result = calibration.analyze_stage1_files(
        arguments.protocol, arguments.shots, *additional_sources
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Stage 1 status: {'PASS' if result.passed else 'FAIL'}")
    print(f"Source digest: {result.source_digest}")
    print(f"Summary: {arguments.output}")
    if arguments.physical_characterization is not None:
        if not result.passed:
            print("Calibration manifest not written because a Stage 1 gate failed.")
        else:
            physical = json.loads(
                arguments.physical_characterization.read_text(encoding="utf-8")
            )
            payload = calibration.build_calibrated_payload(result, physical)
            arguments.calibration_output.parent.mkdir(parents=True, exist_ok=True)
            arguments.calibration_output.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"Calibrated manifest: {arguments.calibration_output}")
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
