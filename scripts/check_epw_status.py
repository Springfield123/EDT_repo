#!/usr/bin/env python3
"""Scan EPW run folders and report Slurm/EPW/Wannier status."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
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

WARNING_PATTERNS = (
    "warning",
    "not converged",
    "disentanglement",
    "wannier",
)


def read_text(path: Path, limit: int = 600000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:]


def load_manifest(epw_root: Path) -> dict[str, dict[str, str]]:
    manifest = epw_root / "epw_run_manifest.csv"
    if not manifest.exists():
        return {}
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        return {row["material"]: row for row in csv.DictReader(handle)}


def slurm_state(job_id: str) -> str:
    if not job_id:
        return "not submitted"
    if shutil.which("sacct"):
        result = subprocess.run(
            ["sacct", "-j", job_id, "--format=JobIDRaw,State", "--noheader", "--parsable2"],
            text=True,
            capture_output=True,
            check=False,
        )
        states = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split("|")]
            if len(parts) >= 2 and parts[0] == job_id:
                states.append(parts[1])
        if states:
            return ",".join(sorted(set(states)))
    if shutil.which("squeue"):
        result = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%T"],
            text=True,
            capture_output=True,
            check=False,
        )
        state = result.stdout.strip()
        if state:
            return state
    return "unknown-slurm-state"


def has_fatal(text: str) -> bool:
    lower = text.lower()
    return any(pattern in lower for pattern in FATAL_PATTERNS)


def find_final_spreads(epw_text: str) -> list[float]:
    final_state = epw_text.lower().rfind("final state")
    if final_state != -1:
        epw_text = epw_text[final_state:]
    spreads: list[float] = []
    for match in re.finditer(r"WF\s+centre\s+and\s+spread\s+\d+\s+\([^)]*\)\s+([0-9.Ee+-]+)", epw_text, re.IGNORECASE):
        try:
            spreads.append(float(match.group(1)))
        except ValueError:
            pass
    if not spreads:
        for pattern in (
            r"Final\s+Spread\s*[:=]\s*([0-9.Ee+-]+)",
            r"Omega\s+Total\s*[:=]\s*([0-9.Ee+-]+)",
        ):
            for match in re.finditer(pattern, epw_text, re.IGNORECASE):
                try:
                    spreads.append(float(match.group(1)))
                except ValueError:
                    pass
    return spreads[-20:]


def wannier_status(epw_text: str, wout_text: str = "") -> str:
    lower = epw_text.lower()
    wout_lower = wout_text.lower()
    if not epw_text and not wout_text:
        return "missing epw.out"
    if "wannier90 error" in lower or "error in wannier" in lower:
        return "wannier error"
    if "convergence criteria satisfied" in lower or "wannier90: execution completed" in lower:
        return "converged/completed"
    if "all done: wannier90 exiting" in wout_lower:
        return "standalone w90 completed"
    if "not converged" in lower:
        return "not converged"
    if "wannier" in lower or wout_text:
        return "started/unknown"
    return "not detected"


def file_availability(material_dir: Path) -> dict[str, bool]:
    save_dir = material_dir / "save"
    files = {
        "epb": any(material_dir.glob("**/*.epb*")),
        "epw": any(material_dir.glob("**/*.epw*")) or any(material_dir.glob("**/*.epmatwp*")),
        "bands": any(material_dir.glob("**/*band*")) or any(material_dir.glob("**/*.gnu")),
        "ukk": any(material_dir.glob("**/*.ukk")) or any(material_dir.glob("**/*.uHu")) or any(material_dir.glob("**/*.uIu")),
        "w90": any((material_dir / f"pwscf{suffix}").exists() for suffix in (".chk", ".mmn", ".wout")),
        "dfpt_save": save_dir.exists() and any(save_dir.iterdir()),
    }
    return files


def infer_state(material_dir: Path, manifest_row: dict[str, str], files: dict[str, bool]) -> tuple[str, list[str], str]:
    if manifest_row.get("status") == "skipped":
        return "skipped", [], "none until DFPT/ph.x/dynmat.x are complete"

    missing = []
    for rel in ("scf.in", "nscf.in", "epw.in", "submit_epw.sh"):
        if not (material_dir / rel).exists():
            missing.append(rel)
    scf_out = material_dir / "outputs" / "scf.out"
    nscf_out = material_dir / "outputs" / "nscf.out"
    epw_out = material_dir / "outputs" / "epw.out"
    epw_text = read_text(epw_out)
    texts = {
        "scf": read_text(scf_out),
        "nscf": read_text(nscf_out),
        "epw": epw_text,
    }
    log_out = ""
    log_files = sorted((material_dir / "logs").glob("*.out")) if (material_dir / "logs").exists() else []
    if log_files:
        log_out = read_text(log_files[-1])

    fatal_steps = [name for name, text in texts.items() if has_fatal(text)]
    if missing:
        state = "incomplete setup"
        action = "regenerate with scripts/setup_epw_runs.py"
    elif fatal_steps:
        state = "failed"
        action = f"inspect outputs/{fatal_steps[0]}.out and relax EPW/Wannier settings"
    elif "job done" in epw_text.lower() or "epw workflow finished" in log_out.lower():
        state = "completed"
        action = "run scripts/collect_epw_qc.py"
    elif files["w90"] and not files["dfpt_save"]:
        state = "wannier-ready/missing-dfpt-save"
        action = "copy/link DFPT dvscf/dyn/pattern files into save/, then submit_epw_only.sh"
    elif files["w90"]:
        state = "epw-only ready"
        action = "submit with sbatch submit_epw_only.sh"
    elif manifest_row.get("submitted") == "true":
        state = "submitted/running"
        action = "wait or inspect Slurm log"
    else:
        state = manifest_row.get("status") or "not submitted"
        action = "submit with sbatch submit_epw.sh if validation is clean"
    return state, missing, action


def scan(epw_root: Path) -> list[dict[str, str]]:
    manifest = load_manifest(epw_root)
    material_dirs = sorted([p for p in epw_root.iterdir() if p.is_dir()]) if epw_root.exists() else []
    names = sorted(set(manifest) | {p.name for p in material_dirs})
    rows = []
    for name in names:
        material_dir = epw_root / name
        mrow = manifest.get(name, {})
        epw_out = material_dir / "outputs" / "epw.out"
        epw_text = read_text(epw_out)
        wout_text = read_text(material_dir / "pwscf.wout")
        spreads = find_final_spreads(epw_text or wout_text)
        files = file_availability(material_dir)
        state, missing, action = infer_state(material_dir, mrow, files)
        fatal = "yes" if has_fatal(epw_text or wout_text) else "no"
        warnings = []
        lower = epw_text.lower()
        for pattern in WARNING_PATTERNS:
            if pattern in lower:
                warnings.append(pattern)
        rows.append(
            {
                "material": name,
                "job_id": mrow.get("job_id", ""),
                "slurm_state": slurm_state(mrow.get("job_id", "")),
                "status": state,
                "epw_fatal": fatal,
                "wannier_status": wannier_status(epw_text, wout_text),
                "final_spreads": ";".join(f"{value:.6g}" for value in spreads),
                "w90_artifacts": str(files["w90"]).lower(),
                "dfpt_save_artifacts": str(files["dfpt_save"]).lower(),
                "epb_written": str(files["epb"]).lower(),
                "epw_written": str(files["epw"]).lower(),
                "bands_generated": str(files["bands"]).lower(),
                "missing_files": ";".join(missing),
                "warnings": ";".join(sorted(set(warnings))),
                "suggested_next_action": action,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epw-root", type=Path, default=Path("epw_runs"))
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV report path.")
    args = parser.parse_args()

    rows = scan(args.epw_root)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)

    for row in rows:
        print(
            f"{row['material']}: {row['status']} | job={row['job_id'] or '-'} | "
            f"slurm={row['slurm_state']} | wannier={row['wannier_status']} | "
            f"w90={row['w90_artifacts']} save={row['dfpt_save_artifacts']} "
            f"epb={row['epb_written']} epw={row['epw_written']} bands={row['bands_generated']} | "
            f"next={row['suggested_next_action']}"
        )
        if row["epw_fatal"] == "yes" or row["missing_files"]:
            print(f"  warnings: fatal={row['epw_fatal']} missing={row['missing_files'] or '-'}")
        if row["final_spreads"]:
            print(f"  final_spreads: {row['final_spreads']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
