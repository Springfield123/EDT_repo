#!/usr/bin/env python3
"""Characterize connected regions in the canonical XY8 score surface."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "code" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import numpy as np

import xy8_resonant_topology as topology


DEFAULT_INPUT = ROOT / "data" / "processed" / "xy8_qre_nv_results.csv"
DEFAULT_OUTDIR = ROOT / "data" / "processed"
REGIONS_FILENAME = "xy8_resonant_regions.csv"
SUMMARY_FILENAME = "xy8_resonant_topology_summary.json"


def load_canonical_sweep(path: str | Path):
    source = Path(path)
    source_bytes = source.read_bytes()
    text = source_bytes.decode("utf-8")
    lines = text.splitlines()
    try:
        surface_header = lines.index("tau_s,N,eta_proxy")
    except ValueError as error:
        raise ValueError(f"No canonical XY8 surface found in {source}.") from error

    metadata = {}
    for row in csv.reader(lines[:surface_header]):
        if len(row) == 2 and row[0]:
            metadata[row[0]] = row[1]
    points = [
        (float(row["tau_s"]), int(row["N"]), float(row["eta_proxy"]))
        for row in csv.DictReader(lines[surface_header:])
    ]
    return (
        topology.surface_from_points(points),
        metadata,
        hashlib.sha256(source_bytes).hexdigest(),
    )


def region_record(region: topology.RegionCharacterization) -> dict:
    record = asdict(region)
    record["resonance_orders"] = list(region.resonance_orders)
    record["resonance_taus_s"] = list(region.resonance_taus_s)
    record.update(
        {
            "threshold_percent": round(
                100.0 * (region.relative_limit - 1.0), 12
            ),
            "best_tau_us": region.best_tau_s * 1e6,
            "tau_min_us": region.tau_min_s * 1e6,
            "tau_max_us": region.tau_max_s * 1e6,
            "tau_width_us": region.tau_width_s * 1e6,
            "best_N": region.best_depth,
            "N_min": region.depth_min,
            "N_max": region.depth_max,
            "N_span_blocks": region.depth_span_blocks,
            "resonance_identity": region.resonance_identity,
            "resonance_taus_us": [tau * 1e6 for tau in region.resonance_taus_s],
            "touches_tau_boundary": region.touches_tau_boundary,
            "touches_N_boundary": region.touches_depth_boundary,
            "boundary_status": region.boundary_status,
            "sampling_status": region.sampling_status,
        }
    )
    return record


def csv_record(record: dict) -> dict:
    output = dict(record)
    output["resonance_orders"] = ";".join(str(value) for value in record["resonance_orders"])
    output["resonance_taus_s"] = ";".join(
        f"{value:.12g}" for value in record["resonance_taus_s"]
    )
    output["resonance_taus_us"] = ";".join(
        f"{value:.12g}" for value in record["resonance_taus_us"]
    )
    return output


def configuration_digest(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def write_outputs(
    input_path: str | Path,
    outdir: str | Path,
    *,
    target_frequency_hz: float,
    relative_limit: float,
    connectivity: int,
) -> tuple[Path, Path, dict]:
    surface, metadata, source_digest = load_canonical_sweep(input_path)
    result = topology.connected_regions(
        surface,
        relative_limit=relative_limit,
        connectivity=connectivity,
    )
    regions = topology.characterize_regions(
        result,
        target_frequency_hz=target_frequency_hz,
    )
    records = [region_record(region) for region in regions]
    best_row, best_column = np.unravel_index(np.argmin(surface.eta), surface.eta.shape)
    config = {
        "source_data_sha256": source_digest,
        "target_frequency_hz": float(target_frequency_hz),
        "relative_limit": float(relative_limit),
        "connectivity": int(connectivity),
    }
    resolved_source = Path(input_path).resolve()
    try:
        displayed_source = str(resolved_source.relative_to(ROOT))
    except ValueError:
        displayed_source = str(resolved_source)
    summary = {
        "schema_version": "1.0",
        "status": "sampled_grid_region_characterization",
        "configuration_digest": configuration_digest(config),
        "source_csv": displayed_source,
        "source_data_sha256": source_digest,
        "calibration_digest": metadata.get("calibration_digest", ""),
        "scenario": metadata.get("scenario", ""),
        "target_frequency_hz": float(target_frequency_hz),
        "relative_limit": float(relative_limit),
        "threshold_percent": round(100.0 * (float(relative_limit) - 1.0), 12),
        "connectivity": int(connectivity),
        "beta_0": result.component_count,
        "surface_shape": [int(value) for value in surface.eta.shape],
        "global_minimum": {
            "tau_s": float(surface.taus_s[best_row]),
            "tau_us": float(surface.taus_s[best_row] * 1e6),
            "N": int(surface.depths[best_column]),
            "eta_proxy": float(surface.eta[best_row, best_column]),
        },
        "regions": records,
        "interpretation_limit": (
            "Connectedness is defined on the sampled grid. Grid refinement and "
            "threshold-persistence checks are required before interpreting beta_0 "
            "as a stable property of the underlying continuous landscape."
        ),
    }

    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    regions_path = destination / REGIONS_FILENAME
    summary_path = destination / SUMMARY_FILENAME
    if records:
        fieldnames = list(csv_record(records[0]).keys())
        with regions_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_record(record) for record in records)
    else:
        raise RuntimeError("A relative limit at or above one must contain the global minimum.")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return regions_path, summary_path, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--frequency-khz", type=float, default=200.0)
    parser.add_argument("--relative-limit", type=float, default=1.10)
    parser.add_argument("--connectivity", type=int, choices=(4, 8), default=4)
    args = parser.parse_args()

    regions_path, summary_path, summary = write_outputs(
        args.input,
        args.outdir,
        target_frequency_hz=args.frequency_khz * 1e3,
        relative_limit=args.relative_limit,
        connectivity=args.connectivity,
    )
    print(
        f"Characterized {summary['beta_0']} connected region(s) at "
        f"{summary['threshold_percent']:.1f}%: {regions_path}"
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
