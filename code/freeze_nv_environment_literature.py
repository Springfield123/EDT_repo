#!/usr/bin/env python3
"""Freeze or verify the provenance-preserving NV environment data bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ANALYSIS_DIR = Path(__file__).resolve().parent / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import xy8_nv_environment_literature as literature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=literature.DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=literature.DEFAULT_FREEZE_RECORD)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the existing record instead of replacing it.",
    )
    args = parser.parse_args()

    if args.verify:
        result = literature.verify_frozen_bundle(args.manifest, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["verified"]:
            raise SystemExit(1)
        return

    if args.output.exists():
        raise SystemExit(
            f"Refusing to replace existing freeze record: {args.output}. "
            "Create a new bundle version instead."
        )
    record = literature.build_freeze_record(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Frozen {len(record['files'])} files as {record['bundle_sha256']}")


if __name__ == "__main__":
    main()
