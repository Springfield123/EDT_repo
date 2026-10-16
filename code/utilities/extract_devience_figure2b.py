#!/usr/bin/env python3
"""Extract DeVience et al. Figure 2b marker centers from its vector PDF."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    ROOT
    / "data"
    / "raw"
    / "nv"
    / "literature_validation"
    / "devience_2015_fig2b_manifest.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "raw"
    / "nv"
    / "literature_validation"
    / "devience_2015_fig2b_xy8.csv"
)


SERIES = {
    10: {
        "object_type": "rects",
        "fill_cmyk": (0.884, 0.769, 0.0, 0.0),
        "top_max": 60.0,
        "offset": 0.6,
        "marker": "square",
    },
    20: {
        "object_type": "curves",
        "fill_cmyk": (0.874, 0.246, 1.0, 0.134),
        "top_max": 110.0,
        "offset": 0.4,
        "marker": "triangle",
    },
    25: {
        "object_type": "curves",
        "fill_cmyk": (0.691, 0.821, 0.0, 0.0),
        "top_max": 160.0,
        "offset": 0.2,
        "marker": "diamond",
    },
    40: {
        "object_type": "curves",
        "fill_cmyk": (0.815, 0.195, 0.425, 0.009),
        "top_max": 220.0,
        "offset": 0.0,
        "marker": "circle",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def linear_map(value: float, point_1: dict, point_2: dict, input_key: str, output_key: str) -> float:
    x1 = float(point_1[input_key])
    x2 = float(point_2[input_key])
    y1 = float(point_1[output_key])
    y2 = float(point_2[output_key])
    return y1 + (float(value) - x1) * (y2 - y1) / (x2 - x1)


def approximately_same(point_1: tuple[float, float], point_2: tuple[float, float]) -> bool:
    return abs(point_1[0] - point_2[0]) < 0.01 and abs(point_1[1] - point_2[1]) < 0.01


def extract(vector_pdf: Path, manifest_path: Path) -> tuple[list[dict], dict]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError(
            "pdfplumber is required. Use the bundled document runtime described in the repository instructions."
        ) from exc

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["selection_status"] != "frozen_before_digitization_and_topology_evaluation":
        raise ValueError("The DeVience selection must be frozen before extraction.")
    calibration = manifest["axis_calibration_pdf_points"]
    frequency_axis = calibration["frequency"]
    contrast_axis = calibration["plotted_contrast"]
    expected_counts = dict(
        zip(
            manifest["predeclared_depths"]["xy8_blocks"],
            manifest["predeclared_depths"]["expected_marker_counts"],
        )
    )

    with pdfplumber.open(vector_pdf) as document:
        if len(document.pages) != 1:
            raise ValueError("Expected the one-page vector Figure 2 asset.")
        page = document.pages[0]
        rows = []
        extracted_counts = {}
        for depth, specification in SERIES.items():
            centers: list[tuple[float, float]] = []
            for obj in getattr(page, specification["object_type"]):
                fill = obj.get("non_stroking_color")
                if not isinstance(fill, tuple) or tuple(fill) != specification["fill_cmyk"]:
                    continue
                if not (435.0 < obj["x0"] < 680.0 and obj["top"] < specification["top_max"]):
                    continue
                width = obj["x1"] - obj["x0"]
                height = obj["bottom"] - obj["top"]
                if not (4.0 < width < 9.0 and 4.0 < height < 9.0):
                    continue
                center = (
                    0.5 * (obj["x0"] + obj["x1"]),
                    0.5 * (obj["top"] + obj["bottom"]),
                )
                if not any(approximately_same(center, existing) for existing in centers):
                    centers.append(center)
            centers.sort()
            extracted_counts[str(depth)] = len(centers)
            if len(centers) != int(expected_counts[depth]):
                raise ValueError(
                    f"XY8-{depth}: expected {expected_counts[depth]} markers, extracted {len(centers)}."
                )
            for index, (center_x, center_top) in enumerate(centers):
                frequency_mhz = linear_map(
                    center_x,
                    frequency_axis["point_1"],
                    frequency_axis["point_2"],
                    "x",
                    "frequency_MHz",
                )
                plotted_contrast = linear_map(
                    center_top,
                    contrast_axis["point_1"],
                    contrast_axis["point_2"],
                    "top",
                    "contrast",
                )
                corrected_contrast = plotted_contrast - specification["offset"]
                rows.append(
                    {
                        "xy8_blocks": depth,
                        "total_pi_pulses": 8 * depth,
                        "point_index": index,
                        "frequency_MHz": round(frequency_mhz, 9),
                        "project_tau_us": round(0.25 / frequency_mhz, 9),
                        "plotted_contrast": round(plotted_contrast, 9),
                        "display_offset": specification["offset"],
                        "corrected_contrast": round(corrected_contrast, 9),
                        "resonant_response": round(1.0 - corrected_contrast, 9),
                        "frequency_digitization_error_MHz": manifest["digitization_policy"]["frequency_error_MHz"],
                        "contrast_digitization_error": manifest["digitization_policy"]["contrast_error"],
                        "marker": specification["marker"],
                        "source_figure": "DeVience et al. Figure 2b",
                        "extraction_status": "vector_marker_center_digitized",
                    }
                )
    metadata = {
        "status": "vector_markers_extracted",
        "source_vector_pdf_sha256": sha256(vector_pdf),
        "manifest_sha256": sha256(manifest_path),
        "extracted_marker_counts_by_xy8_depth": extracted_counts,
        "row_count": len(rows),
        "published_fit_curves_excluded": True,
    }
    return rows, metadata


def run(vector_pdf: Path, manifest_path: Path, output_csv: Path, metadata_json: Path) -> dict:
    rows, metadata = extract(vector_pdf, manifest_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata["output_csv_sha256"] = sha256(output_csv)
    metadata_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-pdf", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata_path = args.metadata or args.output.with_suffix(".extraction.json")
    result = run(args.vector_pdf, args.manifest, args.output, metadata_path)
    print(
        f"Extracted {result['row_count']} Figure 2b markers; "
        f"sha256={result['output_csv_sha256']}"
    )


if __name__ == "__main__":
    main()
