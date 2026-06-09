#!/usr/bin/env python3
"""Collect EPW/Wannier quality-control fields into epw_qc_summary.csv."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


FATAL_PATTERNS = (
    "error in routine",
    "stopping ...",
    "segmentation fault",
    "forrtl:",
    "application called mpi_abort",
    "fatal",
    "%%%%%%",
)


def read_text(path: Path, limit: int | None = None) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:] if limit else text


def parse_float(patterns: tuple[str, ...], text: str) -> str:
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            return matches[-1].group(1)
    return ""


def parse_final_spreads(epw_text: str) -> tuple[str, str, str]:
    final_state = epw_text.lower().rfind("final state")
    if final_state != -1:
        epw_text = epw_text[final_state:]
    values = []
    for match in re.finditer(r"WF\s+centre\s+and\s+spread\s+\d+\s+\([^)]*\)\s+([0-9.Ee+-]+)", epw_text, re.IGNORECASE):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            pass
    if not values:
        for pattern in (
            r"Final\s+Spread\s*[:=]\s*([0-9.Ee+-]+)",
            r"Omega\s+Total\s*[:=]\s*([0-9.Ee+-]+)",
        ):
            for match in re.finditer(pattern, epw_text, re.IGNORECASE):
                try:
                    values.append(float(match.group(1)))
                except ValueError:
                    pass
    if not values:
        return "", "", ""
    return (
        ";".join(f"{value:.8g}" for value in values[-40:]),
        f"{sum(values) / len(values):.8g}",
        f"{max(values):.8g}",
    )


def parse_projections(epw_in: str) -> str:
    projections = []
    for match in re.finditer(r"proj\(\d+\)\s*=\s*['\"]([^'\"]+)['\"]", epw_in, re.IGNORECASE):
        projections.append(match.group(1))
    return ";".join(projections)


def parse_nbndsub(epw_in: str) -> str:
    match = re.search(r"nbndsub\s*=\s*([0-9]+)", epw_in, re.IGNORECASE)
    return match.group(1) if match else ""


def has_wannier_artifacts(material_dir: Path) -> bool:
    return any((material_dir / f"pwscf{suffix}").exists() for suffix in (".chk", ".mmn", ".wout"))


def has_dfpt_save_artifacts(material_dir: Path) -> bool:
    save_dir = material_dir / "save"
    return save_dir.exists() and any(save_dir.iterdir())


def has_any(material_dir: Path, patterns: tuple[str, ...]) -> bool:
    return any(any(material_dir.glob(pattern)) for pattern in patterns)


def collect_material(material_dir: Path) -> dict[str, str]:
    epw_in = read_text(material_dir / "epw.in")
    scf_out = read_text(material_dir / "outputs" / "scf.out", limit=500000)
    nscf_out = read_text(material_dir / "outputs" / "nscf.out", limit=500000)
    epw_out = read_text(material_dir / "outputs" / "epw.out")
    wout = read_text(material_dir / "pwscf.wout")
    combined = "\n".join([scf_out, nscf_out, epw_out])
    wannier_text = epw_out or wout
    spreads, spread_mean, spread_max = parse_final_spreads(wannier_text)
    warnings = []
    warning_text = epw_out.lower()
    for pattern in ("warning", "not converged", "disentanglement", "wannier"):
        if pattern in warning_text:
            warnings.append(pattern)
    errors = [pattern for pattern in FATAL_PATTERNS if pattern in combined.lower()]

    lambda_available = bool(re.search(r"\blambda\b", epw_out, re.IGNORECASE))
    linewidth_available = bool(re.search(r"linewidth|gamma", epw_out, re.IGNORECASE))
    band_quality = parse_float(
        (
            r"band\s+interpolation\s+(?:error|rms)\s*[:=]\s*([0-9.Ee+-]+)",
            r"rms\s+band\s+error\s*[:=]\s*([0-9.Ee+-]+)",
        ),
        epw_out,
    )
    fermi = parse_float(
        (
            r"Fermi\s+energy\s+is\s+([0-9.Ee+-]+)",
            r"the\s+Fermi\s+energy\s+is\s+([0-9.Ee+-]+)",
            r"Fermi\s+level\s*[:=]\s*([0-9.Ee+-]+)",
        ),
        combined,
    )
    omega_total = parse_float(
        (
            r"Final\s+Spread\s+\(Ang\^2\)\s+Omega\s+Total\s+=\s+([0-9.Ee+-]+)",
            r"Omega\s+Total\s*[:=]\s*([0-9.Ee+-]+)",
        ),
        wannier_text,
    )
    wannier_completed = (
        "all done: wannier90 exiting" in wout.lower()
        or "wannier90: execution completed" in epw_out.lower()
        or "convergence criteria satisfied" in epw_out.lower()
    )

    return {
        "material": material_dir.name,
        "completed": str("job done" in epw_out.lower() or "epw workflow finished" in combined.lower()).lower(),
        "fermi_level_ev": fermi,
        "num_wann": parse_nbndsub(epw_in),
        "selected_projections": parse_projections(epw_in),
        "wannier_completed": str(wannier_completed).lower(),
        "wannier_artifacts_available": str(has_wannier_artifacts(material_dir)).lower(),
        "dfpt_save_artifacts_available": str(has_dfpt_save_artifacts(material_dir)).lower(),
        "final_wannier_spreads": spreads,
        "mean_wannier_spread": spread_mean,
        "max_wannier_spread": spread_max,
        "wannier_omega_total": omega_total,
        "band_interpolation_quality": band_quality,
        "epb_available": str(has_any(material_dir, ("**/*.epb*",))).lower(),
        "epw_available": str(has_any(material_dir, ("**/*.epw*", "**/*.epmatwp*"))).lower(),
        "lambda_available": str(lambda_available).lower(),
        "linewidth_available": str(linewidth_available).lower(),
        "warnings": ";".join(sorted(set(warnings))),
        "errors": ";".join(sorted(set(errors))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epw-root", type=Path, default=Path("epw_runs"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    epw_root = args.epw_root
    output = args.output or epw_root / "epw_qc_summary.csv"
    rows = [collect_material(path) for path in sorted(epw_root.iterdir()) if path.is_dir()]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0]) if rows else [
            "material",
            "completed",
            "fermi_level_ev",
            "num_wann",
            "selected_projections",
            "wannier_completed",
            "wannier_artifacts_available",
            "dfpt_save_artifacts_available",
            "final_wannier_spreads",
            "mean_wannier_spread",
            "max_wannier_spread",
            "wannier_omega_total",
            "band_interpolation_quality",
            "epb_available",
            "epw_available",
            "lambda_available",
            "linewidth_available",
            "warnings",
            "errors",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")
    for row in rows:
        print(
            f"{row['material']}: completed={row['completed']} num_wann={row['num_wann'] or '-'} "
            f"w90={row['wannier_artifacts_available']} save={row['dfpt_save_artifacts_available']} "
            f"epb={row['epb_available']} epw={row['epw_available']} errors={row['errors'] or '-'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
