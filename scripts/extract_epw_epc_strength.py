#!/usr/bin/env python3
"""Extract the most resolved EPC quantities available from an EPW run folder."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OUTPUT_COLUMNS = [
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

TEXT_SUFFIXES = {
    ".in",
    ".out",
    ".txt",
    ".dat",
    ".gnu",
    ".csv",
    ".json",
    ".log",
}

EXTRACT_SUFFIXES = {
    ".out",
    ".dat",
    ".gnu",
    ".csv",
    ".log",
}

EPC_NAME_PATTERNS = (
    "epb",
    "epw",
    "epmat",
    "eph",
    "lambda",
    "linewidth",
    "gamma",
    "specfun",
    "selfen",
    "gkk",
    "g2",
)


@dataclass
class CandidateFile:
    path: Path
    size: int
    text_scanned: bool
    roles: list[str]
    matched_lines: int
    notes: str = ""


def is_probably_text(path: Path, max_text_bytes: int) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES and path.stat().st_size <= max_text_bytes


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def classify_by_name(path: Path) -> list[str]:
    name = path.name.lower()
    roles = []
    for pattern in EPC_NAME_PATTERNS:
        if pattern in name:
            roles.append(f"name:{pattern}")
    return roles


def classify_by_text(text: str) -> list[str]:
    lower = text.lower()
    roles = []
    checks = {
        "text:g2": ("|g", "g^2", "g2", "matrix element"),
        "text:lambda": ("lambda",),
        "text:linewidth": ("linewidth", "phonon linewidth", "gamma"),
        "text:epb": ("epb",),
        "text:specfun": ("specfun", "selfen"),
        "text:phonon": ("phonon",),
        "text:electron_phonon": ("electron-phonon", "electron phonon", "el-ph", "e-ph"),
    }
    for role, terms in checks.items():
        if any(term in lower for term in terms):
            roles.append(role)
    return roles


def should_extract_values(path: Path) -> bool:
    name = path.name.lower()
    if name.startswith("readme") or path.suffix.lower() in {".in", ".txt"}:
        return False
    return path.suffix.lower() in EXTRACT_SUFFIXES or any(pattern in name for pattern in EPC_NAME_PATTERNS)


def parse_float(value: str) -> float | None:
    try:
        parsed = float(value.replace("D", "E").replace("d", "e"))
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def format_optional(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}" if math.isfinite(value) else ""
    return str(value)


def first_int(patterns: tuple[str, ...], line: str) -> int | None:
    for pattern in patterns:
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def first_float(patterns: tuple[str, ...], line: str) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            return parse_float(match.group(1))
    return None


def extract_coord(label: str, line: str) -> str:
    match = re.search(rf"\b{label}\s*(?:\([^)]*\))?\s*=\s*\(([^)]*)\)", line, re.IGNORECASE)
    if not match:
        return ""
    return " ".join(match.group(1).replace(",", " ").split())


def last_numeric(line: str) -> float | None:
    values = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?", line)
    for value in reversed(values):
        parsed = parse_float(value)
        if parsed is not None:
            return parsed
    return None


def extract_rows_from_line(material: str, path: Path, line: str) -> list[dict[str, Any]]:
    lower = line.lower()
    gamma_like = bool(re.search(r"\bgamma(?:\s*[=(:]|\s+q|_)", lower)) and "gamma-only" not in lower and "x_gamma" not in lower
    if not any(term in lower for term in ("lambda", "linewidth", "|g", "g^2", "g2", "g_", "matrix element")) and not gamma_like:
        return []
    if "dipole matrix elements calculated" in lower:
        return []

    epc_g2 = first_float(
        (
            r"\|g[^=]*\|\s*\^?\s*2\s*=\s*([-+0-9.EeDd]+)",
            r"\bg\^?2\b\s*=\s*([-+0-9.EeDd]+)",
            r"\bg2\b\s*=\s*([-+0-9.EeDd]+)",
            r"\bg_?mn\w*\b\s*=\s*([-+0-9.EeDd]+)",
            r"matrix element[^=]*=\s*([-+0-9.EeDd]+)",
        ),
        line,
    )
    lambda_qnu = first_float((r"\blambda(?:_q?nu)?\b[^=]*=\s*([-+0-9.EeDd]+)",), line)
    linewidth = first_float((r"\blinewidth\b[^=]*=\s*([-+0-9.EeDd]+)",), line)
    if linewidth is None and gamma_like:
        linewidth = first_float((r"\bgamma\b[^=]*=\s*([-+0-9.EeDd]+)",), line)

    if epc_g2 is None and lambda_qnu is None and linewidth is None:
        if "lambda" in lower:
            lambda_qnu = last_numeric(line)
        elif "linewidth" in lower or gamma_like:
            linewidth = last_numeric(line)
        elif any(term in lower for term in ("|g", "g^2", "g2", "matrix element")):
            epc_g2 = last_numeric(line)

    if epc_g2 is None and lambda_qnu is None and linewidth is None:
        return []

    row = {
        "material": material,
        "k_index": first_int((r"\bk(?:_?index|point)?\s*[=:]\s*(\d+)", r"\bk\(\s*(\d+)\s*\)"), line),
        "k_coord": first_float((r"\bk_coord\s*[=:]\s*([-+0-9.EeDd]+)",), line),
        "q_index": first_int((r"\bq(?:_?index|point)?\s*[=:]\s*(\d+)", r"\bq\(\s*(\d+)\s*\)"), line),
        "q_coord": extract_coord("q", line),
        "band_i": first_int((r"\b(?:ibnd|band_i|band i|m)\s*[=:]\s*(\d+)",), line),
        "band_j": first_int((r"\b(?:jbnd|band_j|band j|n)\s*[=:]\s*(\d+)",), line),
        "mode": first_int((r"\b(?:mode|nu|imode)\s*[=:]\s*(\d+)",), line),
        "frequency": first_float((r"\b(?:freq|frequency|omega)\b[^=]*=\s*([-+0-9.EeDd]+)",), line),
        "epc_g2": epc_g2,
        "lambda_qnu": lambda_qnu,
        "linewidth": linewidth,
        "source_file": str(path),
        "notes": "parsed from text line",
    }
    return [row]


def extraction_resolution(rows: list[dict[str, Any]]) -> str:
    if any(row.get("epc_g2") is not None and row.get("k_index") is not None for row in rows):
        return "band/mode/k-resolved g2"
    if any(row.get("epc_g2") is not None for row in rows):
        return "mode/band g2 without k_coord"
    if any(row.get("lambda_qnu") is not None and row.get("q_index") is not None for row in rows):
        return "q/mode lambda"
    if any(row.get("linewidth") is not None and row.get("q_index") is not None for row in rows):
        return "q/mode linewidth"
    if any(row.get("lambda_qnu") is not None for row in rows):
        return "total or low-resolution lambda"
    if any(row.get("linewidth") is not None for row in rows):
        return "total or low-resolution linewidth"
    return "none"


def inspect_and_extract(material: str, epw_folder: Path, max_text_bytes: int) -> tuple[list[dict[str, Any]], list[CandidateFile]]:
    rows: list[dict[str, Any]] = []
    candidates: list[CandidateFile] = []
    for path in sorted(epw_folder.rglob("*")):
        if not path.is_file():
            continue
        size = path.stat().st_size
        roles = classify_by_name(path)
        matched = 0
        text_scanned = False
        notes = ""
        if is_probably_text(path, max_text_bytes):
            text_scanned = True
            text = read_text(path)
            roles.extend(classify_by_text(text))
            if should_extract_values(path):
                for line in text.splitlines():
                    parsed = extract_rows_from_line(material, path, line)
                    if parsed:
                        matched += len(parsed)
                        rows.extend(parsed)
        elif roles:
            notes = f"not text-scanned: size={size} bytes or suffix={path.suffix!r}"
        if roles or matched:
            candidates.append(
                CandidateFile(
                    path=path,
                    size=size,
                    text_scanned=text_scanned,
                    roles=sorted(set(roles)),
                    matched_lines=matched,
                    notes=notes,
                )
            )
    return rows, candidates


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: format_optional(row.get(column)) for column in OUTPUT_COLUMNS})


def write_status(
    path: Path,
    material: str,
    epw_folder: Path,
    output_csv: Path,
    rows: list[dict[str, Any]],
    candidates: list[CandidateFile],
) -> None:
    status = {
        "material": material,
        "epw_folder": str(epw_folder),
        "output_csv": str(output_csv),
        "num_extracted_rows": len(rows),
        "resolution": extraction_resolution(rows),
        "status": "extracted" if rows else "no_resolved_epc_quantities_found",
        "candidate_files": [
            {
                "path": str(candidate.path),
                "bytes": candidate.size,
                "text_scanned": candidate.text_scanned,
                "roles": candidate.roles,
                "matched_lines": candidate.matched_lines,
                "notes": candidate.notes,
            }
            for candidate in candidates
        ],
        "notes": (
            "No band/mode/k-resolved EPC, lambda, or linewidth quantities were parsed. "
            "The current EPW output appears to stop before phonon self-energy/EPC data are written."
            if not rows
            else "Parsed EPC-like quantities from available text outputs."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", default="Na")
    parser.add_argument("--epw-folder", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--status-json", type=Path, default=None)
    parser.add_argument("--max-text-bytes", type=int, default=5_000_000)
    args = parser.parse_args(argv)

    epw_folder = args.epw_folder or Path("epw_runs") / args.material
    output = args.output or Path("epw_analysis") / args.material / "epc_extracted.csv"
    status_json = args.status_json or Path("epw_analysis") / args.material / "epc_extraction_status.json"
    rows, candidates = inspect_and_extract(args.material, epw_folder, args.max_text_bytes)
    write_csv(output, rows)
    write_status(status_json, args.material, epw_folder, output, rows, candidates)
    print(f"{args.material}: wrote {output} with {len(rows)} extracted EPC rows")
    print(f"{args.material}: wrote {status_json} ({extraction_resolution(rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
