#!/usr/bin/env python3
"""Remove invalid EPW namelist fields and optionally resubmit EPW-only jobs.

Use this when scf/nscf completed but epw.x stopped at input parsing, e.g.

    Bad line in namelist &inputepw: "  fildyn = 'dmat.V',"

Current EPW input documentation uses dvscf_dir for the directory containing
prefix.[dvscf|dyn]_q?? files; fildyn belongs to ph.x/q2r.x/dynmat.x workflows,
not &inputepw in this EPW build.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
from pathlib import Path


EPW_ONLY_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=epw_only_{material}
#SBATCH --account=def-hanghu
#SBATCH --ntasks={ntasks}
#SBATCH --time={time_limit}
#SBATCH --mem=0
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module --force purge
module load StdEnv/2023
module load gcc/12.3
module load openmpi/4.1.5
module load quantumespresso/7.5

export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-1}}"
export EPW_NIMAGE="${{EPW_NIMAGE:-1}}"
export EPW_NPOOL="${{EPW_NPOOL:-$SLURM_NTASKS}}"

if (( EPW_NPOOL * EPW_NIMAGE != SLURM_NTASKS )); then
  echo "EPW parallel layout error: SLURM_NTASKS=$SLURM_NTASKS but EPW_NPOOL*EPW_NIMAGE=$((EPW_NPOOL * EPW_NIMAGE))" >&2
  exit 2
fi

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs save

args=(-npool "$EPW_NPOOL")
if (( EPW_NIMAGE != 1 )); then
  args+=(-nimage "$EPW_NIMAGE")
fi

echo "[$(date '+%F %T')] Running epw.x only with $SLURM_NTASKS MPI tasks and ${{args[*]}}"
mpirun -np "$SLURM_NTASKS" epw.x "${{args[@]}}" -in epw.in > outputs/epw.out
echo "[$(date '+%F %T')] EPW-only retry finished for {material}"
"""


DEFAULT_TASKS = {
    "Na": 32,
    "V": 32,
    "Nb": 32,
    "YBaCuO": 48,
}

DEFAULT_TIME = {
    "Na": "24:00:00",
    "V": "36:00:00",
    "Nb": "36:00:00",
    "YBaCuO": "72:00:00",
}


UNSUPPORTED_KEYS = (
    "fildyn",
    "kmaps",
    "epwwrite",
    "epwread",
    "use_ws",
    "prtgkk",
    "iverbosity",
    "wdata",
)

MATERIAL_WANNIER = {
    "Na": {
        "nbndsub": 8,
        "projections": ("Na:s", "Na:p"),
    },
    "V": {
        "nbndsub": 18,
        "projections": ("V:s", "V:p", "V:d"),
    },
    "Nb": {
        "nbndsub": 18,
        "projections": ("Nb:s", "Nb:p", "Nb:d"),
    },
    "YBaCuO": {
        "nbndsub": 44,
        "projections": ("Cu:d", "O:p", "Y:d", "Y:s", "Ba:s"),
    },
}


def wannier_block(material: str) -> str:
    settings = MATERIAL_WANNIER.get(material)
    if not settings:
        return ""
    lines = [f"  nbndsub = {settings['nbndsub']},"]
    for idx, proj in enumerate(settings["projections"], start=1):
        lines.append(f"  proj({idx}) = '{proj}',")
    return "\n".join(lines) + "\n"


def patch_epw_input(epw_in: Path, material: str) -> bool:
    text = epw_in.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    original = text
    for key in UNSUPPORTED_KEYS:
        text = re.sub(rf"^\s*{re.escape(key)}(?:\(\d+\))?\s*=.*\n", "", text, flags=re.IGNORECASE | re.MULTILINE)
    if material in {"V", "Nb", "YBaCuO"}:
        for key in ("dis_win_min", "dis_win_max", "dis_froz_min", "dis_froz_max"):
            text = re.sub(rf"^\s*{key}\s*=.*\n", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"^\s*nbndsub\s*=.*\n", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"^\s*proj\(\d+\)\s*=.*\n", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(
        r"^\s*dvscf_dir\s*=.*$",
        "  dvscf_dir = './save',",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if "dvscf_dir" not in text.lower():
        text = text.replace("  outdir = './qe_out',", "  outdir = './qe_out',\n  dvscf_dir = './save',")
    block = wannier_block(material)
    if block:
        if re.search(r"^\s*num_iter\s*=", text, flags=re.IGNORECASE | re.MULTILINE):
            text = re.sub(r"^\s*num_iter\s*=.*\n", block + "  num_iter = 1000,\n", text, count=1, flags=re.IGNORECASE | re.MULTILINE)
        elif re.search(r"^\s*dis_win_min\s*=", text, flags=re.IGNORECASE | re.MULTILINE):
            text = re.sub(r"^\s*dis_win_min\s*=", block + "  dis_win_min = ", text, count=1, flags=re.IGNORECASE | re.MULTILINE)
        else:
            text = text.replace("  epbread = .false.,", "  epbread = .false.,\n\n" + block.rstrip())
    if text != original:
        epw_in.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")
        return True
    return False


def write_epw_only_submit(material_dir: Path, ntasks: int, time_limit: str) -> Path:
    material = material_dir.name
    path = material_dir / "submit_epw_only.sh"
    path.write_text(
        EPW_ONLY_TEMPLATE.format(material=material, ntasks=ntasks, time_limit=time_limit).rstrip() + "\n",
        encoding="utf-8",
        newline="\n",
    )
    try:
        os.chmod(path, 0o755)
    except OSError:
        pass
    return path


def update_manifest(epw_root: Path, material: str, job_id: str, status: str, note: str) -> None:
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
            row["submitted"] = "true" if job_id else row.get("submitted", "false")
            row["job_id"] = job_id or row.get("job_id", "")
            row["status"] = status
            old_notes = row.get("notes", "")
            row["notes"] = f"{old_notes} | {note}" if old_notes else note
            break
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def submit(material_dir: Path, dry_run: bool) -> tuple[bool, str, str]:
    if dry_run:
        return True, "", f"DRY-RUN: cd {material_dir} && sbatch submit_epw_only.sh"
    result = subprocess.run(
        ["sbatch", "submit_epw_only.sh"],
        cwd=material_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    match = re.search(r"Submitted batch job\s+(\d+)", output)
    return result.returncode == 0, match.group(1) if match else "", output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epw-root", type=Path, default=Path("epw_runs"))
    parser.add_argument("--materials", nargs="*", default=["Na", "V", "Nb", "YBaCuO"])
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for material in args.materials:
        material_dir = args.epw_root / material
        epw_in = material_dir / "epw.in"
        if not epw_in.exists():
            print(f"{material}: missing {epw_in}")
            update_manifest(args.epw_root, material, "", "failed_validation", "missing epw.in for namelist patch")
            continue

        changed = patch_epw_input(epw_in, material)
        submit_path = write_epw_only_submit(
            material_dir,
            ntasks=DEFAULT_TASKS.get(material, 32),
            time_limit=DEFAULT_TIME.get(material, "36:00:00"),
        )
        save_dir = material_dir / "save"
        save_has_files = save_dir.exists() and any(save_dir.iterdir())
        missing_save_note = "" if save_has_files else " save/ is empty; EPW may next need DFPT dvscf/dyn/pattern files."
        print(f"{material}: {'patched epw.in' if changed else 'epw.in already patched'}; wrote {submit_path.name}.{missing_save_note}")

        if args.submit or args.dry_run:
            ok, job_id, output = submit(material_dir, args.dry_run)
            print(f"  {'submitted' if ok else 'submit failed'}: {output}")
            if ok and job_id:
                update_manifest(
                    args.epw_root,
                    material,
                    job_id,
                    "epw_only_resubmitted",
                    "removed invalid fildyn from epw.in and resubmitted epw.x only",
                )
            elif not ok:
                update_manifest(args.epw_root, material, "", "submit_failed", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
