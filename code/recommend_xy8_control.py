#!/usr/bin/env python3
"""Command-line hardware-to-control decision rule for the NV XY8 model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "code" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import xy8_control_recommender as recommender


NOMINAL_CALIBRATION = ROOT / "data" / "raw" / "nv" / "xy8_scenario_nominal.json"


def build_parser():
    parser = argparse.ArgumentParser(description="Recommend an XY8 control from measured hardware inputs.")
    parser.add_argument("--frequency-khz", type=float, required=True)
    parser.add_argument("--pulse-fidelity", type=float, required=True)
    parser.add_argument("--t2-us", type=float, required=True)
    parser.add_argument("--pulse-fidelity-sigma", type=float)
    parser.add_argument("--t2-sigma-us", type=float)
    parser.add_argument("--uncertainty-samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=recommender.DEFAULT_UNCERTAINTY_SEED)
    parser.add_argument("--calibration", default=str(NOMINAL_CALIBRATION))
    parser.add_argument("--json-out")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    payload = recommender.recommend_control(
        calibration,
        frequency_hz=args.frequency_khz * 1e3,
        pulse_fidelity=args.pulse_fidelity,
        t2_s=args.t2_us * 1e-6,
        pulse_fidelity_sigma=args.pulse_fidelity_sigma,
        t2_sigma_s=None if args.t2_sigma_us is None else args.t2_sigma_us * 1e-6,
        uncertainty_samples=args.uncertainty_samples,
        seed=args.seed,
    )
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not args.quiet:
        print(recommender.format_recommendation(payload))


if __name__ == "__main__":
    main()
