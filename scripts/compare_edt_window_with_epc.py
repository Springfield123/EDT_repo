#!/usr/bin/env python3
"""Compare extracted EPW EPC quantities with an EDT window."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


EPC_COLUMNS = ("epc_g2", "lambda_qnu", "linewidth")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}" if math.isfinite(value) else ""
    return str(value)


def load_k_grid(grid_csv: Path | None, agr_file: Path | None) -> list[float]:
    if grid_csv and grid_csv.exists():
        rows = read_csv(grid_csv)
        by_index: dict[int, float] = {}
        for row in rows:
            if "k_index" not in row or "k_coord" not in row:
                continue
            k_index = int(row["k_index"])
            k_coord = as_float(row["k_coord"])
            if k_coord is not None and k_index not in by_index:
                by_index[k_index] = k_coord
        if by_index:
            return [by_index[idx] for idx in sorted(by_index)]
    if agr_file and agr_file.exists():
        return parse_first_agr_grid(agr_file)
    return []


def parse_first_agr_grid(path: Path) -> list[float]:
    xs: list[float] = []
    in_dataset = False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("@target"):
            if in_dataset and xs:
                break
            in_dataset = True
            continue
        if not in_dataset:
            continue
        if line == "&":
            break
        if line.startswith("@") or line.startswith("#") or not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                xs.append(float(parts[0]))
            except ValueError:
                pass
    return xs


def map_k_coord(row: dict[str, str], k_grid: list[float]) -> tuple[float | None, str]:
    direct = as_float(row.get("k_coord"))
    if direct is not None:
        return direct, "direct k_coord"
    k_index_value = row.get("k_index", "")
    if k_index_value:
        try:
            k_index = int(k_index_value)
        except ValueError:
            return None, "unparseable k_index"
        for idx in (k_index, k_index - 1):
            if 0 <= idx < len(k_grid):
                return k_grid[idx], f"mapped from k_index={k_index}"
        return None, "k_index outside AGR grid"
    return None, "no k_coord or k_index"


def finite_values(rows: list[dict[str, Any]], column: str, inside: bool) -> list[float]:
    values = []
    for row in rows:
        if row.get("inside_edt_window") is not inside:
            continue
        value = as_float(row.get(column))
        if value is not None:
            values.append(value)
    return values


def stats_for_column(rows: list[dict[str, Any]], column: str) -> dict[str, Any]:
    inside = finite_values(rows, column, True)
    outside = finite_values(rows, column, False)
    inside_mean = sum(inside) / len(inside) if inside else None
    outside_mean = sum(outside) / len(outside) if outside else None
    return {
        "inside_count": len(inside),
        "outside_count": len(outside),
        "inside_mean": inside_mean,
        "outside_mean": outside_mean,
        "inside_median": median(inside) if inside else None,
        "outside_median": median(outside) if outside else None,
        "inside_max": max(inside) if inside else None,
        "outside_max": max(outside) if outside else None,
        "inside_outside_ratio": inside_mean / outside_mean if inside_mean is not None and outside_mean not in (None, 0.0) else None,
    }


def compare(
    material: str,
    summary_path: Path,
    epc_csv: Path,
    output_csv: Path,
    summary_output: Path,
    grid_csv: Path | None,
    agr_file: Path | None,
) -> dict[str, Any]:
    summary = read_json(summary_path)
    window = summary["edt_window"]
    k_left = float(window["k_left"])
    k_right = float(window["k_right"])
    epc_rows = read_csv(epc_csv) if epc_csv.exists() else []
    k_grid = load_k_grid(grid_csv, agr_file)
    mapped_rows: list[dict[str, Any]] = []
    for row in epc_rows:
        k_coord, mapping_note = map_k_coord(row, k_grid)
        inside = (k_left <= k_coord <= k_right) if k_coord is not None else None
        out = dict(row)
        out["mapped_k_coord"] = k_coord
        out["inside_edt_window"] = inside
        out["mapping_notes"] = mapping_note
        mapped_rows.append(out)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(epc_rows[0]) if epc_rows else [
        "material",
        "k_index",
        "k_coord",
        "q_index",
        "q_coord",
        "band_i",
        "band_j",
        "mode",
        "frequency",
        "epc_g2",
        "lambda_qnu",
        "linewidth",
        "source_file",
        "notes",
    ]
    fieldnames = fieldnames + [name for name in ("mapped_k_coord", "inside_edt_window", "mapping_notes") if name not in fieldnames]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in mapped_rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fieldnames})

    column_stats = {column: stats_for_column(mapped_rows, column) for column in EPC_COLUMNS}
    any_values = any(stats["inside_count"] or stats["outside_count"] for stats in column_stats.values())
    any_mappable = any(row.get("inside_edt_window") is not None for row in mapped_rows)
    status = "completed" if any_values and any_mappable else "insufficient_epc_data"
    result = {
        "material": material,
        "status": status,
        "edt_window": window,
        "input_epc_csv": str(epc_csv),
        "output_csv": str(output_csv),
        "k_grid_source": str(grid_csv if grid_csv and grid_csv.exists() else agr_file),
        "num_epc_rows": len(epc_rows),
        "num_rows_mapped_to_k": sum(1 for row in mapped_rows if row.get("inside_edt_window") is not None),
        "metrics": {
            "R_EPC_g2": column_stats["epc_g2"]["inside_outside_ratio"],
            "R_lambda": column_stats["lambda_qnu"]["inside_outside_ratio"],
            "R_linewidth": column_stats["linewidth"]["inside_outside_ratio"],
        },
        "details": column_stats,
        "notes": (
            "No EPC values could be mapped into the EDT window. Run or stage EPW EPC outputs first."
            if status != "completed"
            else "Computed EPC inside/outside EDT-window ratios."
        ),
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", default="Na")
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--epc-csv", type=Path, default=None)
    parser.add_argument("--grid-csv", type=Path, default=None)
    parser.add_argument("--agr-file", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-summary", type=Path, default=None)
    args = parser.parse_args(argv)

    analysis_dir = args.analysis_dir or Path("epw_analysis") / args.material
    summary = args.summary or analysis_dir / "edt_geometry_summary.json"
    epc_csv = args.epc_csv or analysis_dir / "epc_extracted.csv"
    grid_csv = args.grid_csv or analysis_dir / "quantum_metric_on_agr_path.csv"
    agr_file = args.agr_file or Path("epw_runs") / args.material / f"{args.material}-m5.agr"
    if not agr_file.exists() and args.material == "Na":
        agr_file = Path("data/na/Na-m5.agr")
    if not agr_file.exists() and args.material == "K3C60":
        agr_file = Path("data/k3c60/tmp.agr")
    output_csv = args.output_csv or analysis_dir / "epc_edt_window_comparison.csv"
    output_summary = args.output_summary or analysis_dir / "epc_edt_window_summary.json"

    result = compare(args.material, summary, epc_csv, output_csv, output_summary, grid_csv, agr_file)
    print(f"{args.material}: wrote {output_csv}")
    print(f"{args.material}: wrote {output_summary} ({result['status']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
