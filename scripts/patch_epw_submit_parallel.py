#!/usr/bin/env python3
"""Patch existing epw_runs/*/submit_epw.sh files for EPW pool/image layout.

EPW requires:

    MPI tasks = number of k pools * number of images

This patch keeps pw.x launches unchanged and changes only the epw.x launch to:

    epw.x -npool "$EPW_NPOOL" -in epw.in

By default EPW_NIMAGE=1 and EPW_NPOOL=SLURM_NTASKS.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
from pathlib import Path


ENV_BLOCK = """export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export EPW_NIMAGE="${EPW_NIMAGE:-1}"
export EPW_NPOOL="${EPW_NPOOL:-$SLURM_NTASKS}"

if (( EPW_NPOOL * EPW_NIMAGE != SLURM_NTASKS )); then
  echo "EPW parallel layout error: SLURM_NTASKS=$SLURM_NTASKS but EPW_NPOOL*EPW_NIMAGE=$((EPW_NPOOL * EPW_NIMAGE))" >&2
  echo "Set EPW_NPOOL and EPW_NIMAGE so their product equals SLURM_NTASKS." >&2
  exit 2
fi"""

RUN_EPW_BLOCK = """run_epw() {
  local input="$1"
  local output="$2"
  local args=(-npool "$EPW_NPOOL")
  if (( EPW_NIMAGE != 1 )); then
    args+=(-nimage "$EPW_NIMAGE")
  fi
  echo "[$(date '+%F %T')] Running epw.x on $SLURM_NTASKS MPI tasks with ${args[*]}"
  mpirun -np "$SLURM_NTASKS" epw.x "${args[@]}" -in "$input" > "$output"
}
"""


def patch_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    original = text

    text = re.sub(
        r'export OMP_NUM_THREADS="\$\{OMP_NUM_THREADS:-1\}"(?:\nexport EPW_NIMAGE=.*?\nfi)?',
        ENV_BLOCK,
        text,
        flags=re.DOTALL,
    )

    marker = re.search(r"run_epw\(\) \{.*?\n\}\n", text, flags=re.DOTALL)
    if marker:
        text = text[: marker.start()] + RUN_EPW_BLOCK + text[marker.end() :]
    else:
        marker = re.search(r"run_mpi\(\) \{.*?\n\}\n", text, flags=re.DOTALL)
        if marker:
            insert_at = marker.end()
            text = text[:insert_at] + "\n" + RUN_EPW_BLOCK + text[insert_at:]

    text = re.sub(
        r"run_mpi\s+epw\.x\s+epw\.in\s+outputs/epw\.out",
        "run_epw epw.in outputs/epw.out",
        text,
    )

    if text != original:
        path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")
        try:
            os.chmod(path, 0o755)
        except OSError:
            pass
        return True
    return False


def submit(path: Path, dry_run: bool) -> tuple[bool, str, str]:
    if dry_run:
        return True, "", f"DRY-RUN: cd {path.parent} && sbatch submit_epw.sh"
    result = subprocess.run(
        ["sbatch", "submit_epw.sh"],
        cwd=path.parent,
        text=True,
        capture_output=True,
        check=False,
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    match = re.search(r"Submitted batch job\s+(\d+)", output)
    job_id = match.group(1) if match else ""
    return result.returncode == 0, job_id, output


def update_manifest(epw_root: Path, material: str, job_id: str, note: str) -> None:
    manifest = epw_root / "epw_run_manifest.csv"
    if not manifest.exists():
        return
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0]) if rows else []
    if not rows or not fieldnames:
        return
    for row in rows:
        if row.get("material") == material:
            row["submitted"] = "true"
            row["job_id"] = job_id
            row["status"] = "resubmitted"
            old_notes = row.get("notes", "")
            row["notes"] = f"{old_notes} | {note}" if old_notes else note
            break
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epw-root", type=Path, default=Path("epw_runs"))
    parser.add_argument("--submit", action="store_true", help="Resubmit each patched EPW job.")
    parser.add_argument("--dry-run", action="store_true", help="Print sbatch commands without submitting.")
    parser.add_argument("--materials", nargs="*", default=None, help="Optional subset to patch/resubmit.")
    args = parser.parse_args()

    selected = set(args.materials or [])
    scripts = sorted(args.epw_root.glob("*/submit_epw.sh"))
    if selected:
        scripts = [path for path in scripts if path.parent.name in selected]

    if not scripts:
        print(f"No submit_epw.sh files found under {args.epw_root}")
        return 1

    for script in scripts:
        changed = patch_file(script)
        status = "patched" if changed else "already patched"
        print(f"{script.parent.name}: {status}")
        if args.submit or args.dry_run:
            ok, job_id, output = submit(script, args.dry_run)
            print(f"  {'submitted' if ok else 'submit failed'}: {output}")
            if ok and job_id:
                update_manifest(args.epw_root, script.parent.name, job_id, "resubmitted after EPW -nk/-ni launcher fix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
