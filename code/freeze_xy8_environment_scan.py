#!/usr/bin/env python3
"""Freeze or verify the preregistered XY8 environment-scan configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ANALYSIS_DIR = Path(__file__).resolve().parent / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import xy8_environment_scan_config as scan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=scan.DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=scan.DEFAULT_FREEZE_RECORD)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if args.verify:
        result = scan.verify_freeze_record(args.config, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["verified"]:
            raise SystemExit(1)
        return

    if args.output.exists():
        raise SystemExit(
            f"Refusing to replace existing freeze record: {args.output}. "
            "Create a new configuration version instead."
        )
    record = scan.build_freeze_record(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Frozen {record['configuration_id']} as "
        f"{record['configuration_sha256']}"
    )


if __name__ == "__main__":
    main()
