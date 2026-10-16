#!/usr/bin/env python3
"""Extract the five experimental XY8 polylines in Loretz et al. Figure 4a."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "data" / "raw" / "nv" / "literature_validation" / "loretz_2015_fig4a_topology_manifest.json"
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "nv" / "literature_validation" / "loretz_2015_fig4a_xy8_spectrum.csv"
COLORS = {(0.93333, 0.2, 0.17647), (0.13333, 0.33725, 0.65098)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def map_axis(value: float, point_1: dict, point_2: dict, input_key: str, output_key: str) -> float:
    return float(point_1[output_key]) + (value - float(point_1[input_key])) * (
        float(point_2[output_key]) - float(point_1[output_key])
    ) / (float(point_2[input_key]) - float(point_1[input_key]))


def extract(vector_pdf: Path, manifest_path: Path) -> tuple[list[dict], dict]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("pdfplumber is required for vector extraction.") from exc
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["selection_status"] != "frozen_before_digitization_and_topology_evaluation":
        raise ValueError("The Loretz selection must be frozen before extraction.")
    with pdfplumber.open(vector_pdf) as document:
        page = document.pages[0]
        curves = [
            curve
            for curve in page.curves
            if curve["x1"] - curve["x0"] > 20
            and 100 < curve["top"] < 205
            and curve.get("stroking_color") in COLORS
        ]
    curves.sort(key=lambda curve: curve["x0"])
    segments = manifest["predeclared_segments"]
    if len(curves) != len(segments):
        raise ValueError(f"Expected {len(segments)} experimental paths, found {len(curves)}.")
    frequency_axis = manifest["axis_calibration_pdf_points"]["frequency"]
    probability_axis = manifest["axis_calibration_pdf_points"]["transition_probability"]
    rows = []
    for segment, curve in zip(segments, curves):
        points = sorted(curve["pts"])
        if len(points) != segment["expected_path_points"]:
            raise ValueError(f"{segment['segment_id']} point count changed.")
        for index, (x, top) in enumerate(points):
            frequency = map_axis(x, frequency_axis["point_1"], frequency_axis["point_2"], "x", "frequency_MHz")
            probability = map_axis(top, probability_axis["point_1"], probability_axis["point_2"], "top", "probability")
            rows.append(
                {
                    "segment_id": segment["segment_id"],
                    "total_pi_pulses": segment["total_pi_pulses"],
                    "xy8_blocks": segment["total_pi_pulses"] // 8,
                    "point_index": index,
                    "detection_frequency_MHz": round(frequency, 9),
                    "transition_probability": round(probability, 9),
                    "frequency_digitization_error_MHz": manifest["digitization_policy"]["frequency_error_MHz"],
                    "probability_digitization_error": manifest["digitization_policy"]["probability_error"],
                    "source_figure": "Loretz et al. Figure 4a",
                    "extraction_status": "vector_polyline_vertex_digitized"
                }
            )
    metadata = {
        "status": "experimental_vector_paths_extracted",
        "source_vector_pdf_sha256": sha256(vector_pdf),
        "manifest_sha256": sha256(manifest_path),
        "row_count": len(rows),
        "segment_point_counts": {segment["segment_id"]: segment["expected_path_points"] for segment in segments},
        "published_simulation_excluded": True
    }
    return rows, metadata


def run(vector_pdf: Path, manifest_path: Path, output: Path) -> dict:
    rows, metadata = extract(vector_pdf, manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata["output_csv_sha256"] = sha256(output)
    output.with_suffix(".extraction.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-pdf", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.vector_pdf, args.manifest, args.output)
    print(f"Extracted {result['row_count']} experimental vertices; sha256={result['output_csv_sha256']}")


if __name__ == "__main__":
    main()
