#!/usr/bin/env python3
"""Create second-stage EPW restart inputs and Narval submit scripts."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


DEFAULT_NTASKS = {
    "Na": 64,
    "V": 64,
    "Nb": 64,
    "YBaCuO": 96,
    "K3C60": 128,
}

DEFAULT_TIME = {
    "Na": "48:00:00",
    "V": "72:00:00",
    "Nb": "72:00:00",
    "YBaCuO": "96:00:00",
    "K3C60": "24:00:00",
}

RESTART_REPLACEMENTS = {
    "ep_coupling": ".true.",
    "elph": ".true.",
    "wannierize": ".false.",
    "epbread": ".true.",
    "epbwrite": ".false.",
    "epwread": ".true.",
    "epwwrite": ".false.",
    "phonselfen": ".true.",
    "a2f": ".true.",
    "specfun_ph": ".true.",
    "iverbosity": "3",
    "fsthick": "1.0",
    "degaussw": "0.02",
    "ngaussw": "1",
    "wmin_specfun": "0.00",
    "wmax_specfun": "0.25",
    "nw_specfun": "300",
    "temps(1)": "300.0",
    "nstemp": "1",
}


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def split_namelist(text: str) -> tuple[str, list[str], str]:
    lines = normalize_text(text).splitlines()
    start = next((idx for idx, line in enumerate(lines) if line.strip().lower().startswith("&inputepw")), None)
    if start is None:
        raise ValueError("Missing &inputepw namelist")
    end = next((idx for idx in range(start + 1, len(lines)) if lines[idx].strip().startswith("/")), None)
    if end is None:
        raise ValueError("Missing closing / for &inputepw")
    return "\n".join(lines[: start + 1]), lines[start + 1 : end], "\n".join(lines[end:])


def key_regex(key: str) -> re.Pattern[str]:
    return re.compile(rf"^\s*{re.escape(key)}\s*=", re.IGNORECASE)


def render_restart_input(base_text: str, nqf: tuple[int, int, int], nkf: tuple[int, int, int] | None) -> str:
    header, body, footer = split_namelist(base_text)
    replacements = dict(RESTART_REPLACEMENTS)
    replacements.update(
        {
            "nqf1": str(nqf[0]),
            "nqf2": str(nqf[1]),
            "nqf3": str(nqf[2]),
        }
    )
    if nkf:
        replacements.update({"nkf1": str(nkf[0]), "nkf2": str(nkf[1]), "nkf3": str(nkf[2])})

    seen: set[str] = set()
    out: list[str] = []
    for line in body:
        clean = line.strip()
        if not clean:
            out.append(line)
            continue
        replaced = False
        for key, value in replacements.items():
            if key_regex(key).match(clean):
                out.append(f"  {key} = {value},")
                seen.add(key)
                replaced = True
                break
        if not replaced:
            out.append(line)

    insert_idx = next((idx for idx, line in enumerate(out) if key_regex("nbndsub").match(line.strip())), len(out))
    missing_lines = [f"  {key} = {value}," for key, value in replacements.items() if key not in seen]
    out = out[:insert_idx] + missing_lines + out[insert_idx:]
    return (header + "\n" + "\n".join(out).rstrip() + "\n" + footer.strip() + "\n").replace("\n\n\n", "\n\n")


def render_submit(material: str, ntasks: int, time_limit: str, account: str, restart_name: str) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=epw_restart_{material}
#SBATCH --account={account}
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
export EPW_INPUT="${{EPW_INPUT:-{restart_name}}}"
export MPI_LAUNCHER="${{MPI_LAUNCHER:-srun}}"

if (( EPW_NPOOL * EPW_NIMAGE != SLURM_NTASKS )); then
  echo "EPW parallel layout error: SLURM_NTASKS=$SLURM_NTASKS but EPW_NPOOL*EPW_NIMAGE=$((EPW_NPOOL * EPW_NIMAGE))" >&2
  exit 2
fi

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs save

if ! find . -maxdepth 4 -type f \\( -name "*.epb*" -o -name "*.epmatwp*" -o -name "epwdata.fmt" \\) | grep -q .; then
  echo "No EPB/EPW restart payload found. Run the first EPW stage to completion or stage epb/epw restart files before submitting this Narval restart." >&2
  exit 3
fi

args=(-npool "$EPW_NPOOL")
if (( EPW_NIMAGE != 1 )); then
  args+=(-nimage "$EPW_NIMAGE")
fi

echo "[$(date '+%F %T')] Running second-stage EPW restart on Narval with $SLURM_NTASKS MPI tasks and ${{args[*]}}"
if [[ "$MPI_LAUNCHER" == "mpirun" ]]; then
  mpirun -np "$SLURM_NTASKS" epw.x "${{args[@]}}" -in "$EPW_INPUT" > outputs/epw_restart_narval.out
else
  "$MPI_LAUNCHER" epw.x "${{args[@]}}" -in "$EPW_INPUT" > outputs/epw_restart_narval.out
fi
echo "[$(date '+%F %T')] EPW Narval restart finished for {material}"
"""


def render_readme(material: str, restart_name: str, nqf: tuple[int, int, int], nkf: tuple[int, int, int] | None) -> str:
    return f"""# Narval second-stage EPW restart for {material}

Generated files:
- {restart_name}
- submit_epw_restart_narval.sh

Purpose:
- restart from completed first-stage EPW/Wannier/EPB data;
- enable epbread=.true. and epwread=.true.;
- enable phonselfen=.true., a2f=.true., and specfun_ph=.true.;
- use denser fine q-grid nqf = {nqf[0]} {nqf[1]} {nqf[2]}.

Prerequisites:
- The first EPW stage must have produced EPB/EPW restart payloads such as *.epb*, *.epmatwp*, or epwdata.fmt.
- DFPT save/dvscf/dyn/pattern data must be staged consistently with dvscf_dir.
- If this EPW build rejects epwread/epwwrite, set EPW_INPUT to a manually edited input with epwread/epwwrite removed while keeping epbread=.true.

Submit on Narval from this material directory:
  sbatch submit_epw_restart_narval.sh

Optional runtime overrides:
  EPW_NPOOL, EPW_NIMAGE, OMP_NUM_THREADS, MPI_LAUNCHER, EPW_INPUT

No EPC comparison should be claimed until the restart writes resolved EPC/lambda/linewidth outputs that can be parsed.
"""


def create_for_material(
    material: str,
    epw_root: Path,
    nqf: tuple[int, int, int],
    nkf: tuple[int, int, int] | None,
    account: str,
    ntasks: int | None,
    time_limit: str | None,
    restart_name: str = "epw_restart_narval.in",
) -> dict[str, str]:
    material_dir = epw_root / material
    base_input = material_dir / "epw.in"
    if not base_input.exists():
        raise FileNotFoundError(base_input)
    restart_input = material_dir / restart_name
    submit_path = material_dir / "submit_epw_restart_narval.sh"
    readme_path = material_dir / "README_epw_restart_narval.txt"
    restart_input.write_text(render_restart_input(base_input.read_text(encoding="utf-8"), nqf, nkf), encoding="utf-8", newline="\n")
    submit_path.write_text(
        render_submit(
            material,
            ntasks=ntasks or DEFAULT_NTASKS.get(material, 64),
            time_limit=time_limit or DEFAULT_TIME.get(material, "72:00:00"),
            account=account,
            restart_name=restart_name,
        ),
        encoding="utf-8",
        newline="\n",
    )
    readme_path.write_text(render_readme(material, restart_name, nqf, nkf), encoding="utf-8", newline="\n")
    try:
        os.chmod(submit_path, 0o755)
    except OSError:
        pass
    return {
        "material": material,
        "restart_input": str(restart_input),
        "submit_script": str(submit_path),
        "readme": str(readme_path),
    }


def parse_grid(text: str) -> tuple[int, int, int]:
    parts = [int(part) for part in text.lower().replace("x", " ").replace(",", " ").split()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("grid must have exactly three integers, e.g. 4 4 4 or 4x4x4")
    return parts[0], parts[1], parts[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epw-root", type=Path, default=Path("epw_runs"))
    parser.add_argument("--materials", nargs="*", default=["Na"])
    parser.add_argument("--nqf", type=parse_grid, default=(4, 4, 4), help="Fine q-grid, for example 4x4x4.")
    parser.add_argument("--nkf", type=parse_grid, default=None, help="Optional fine k-grid override.")
    parser.add_argument("--account", default="def-hanghu")
    parser.add_argument("--ntasks", type=int, default=None)
    parser.add_argument("--time", default=None)
    parser.add_argument("--restart-name", default="epw_restart_narval.in")
    args = parser.parse_args(argv)

    for material in args.materials:
        result = create_for_material(
            material,
            args.epw_root,
            args.nqf,
            args.nkf,
            args.account,
            args.ntasks,
            args.time,
            restart_name=args.restart_name,
        )
        print(f"{material}: wrote {result['restart_input']}")
        print(f"{material}: wrote {result['submit_script']}")
        print(f"{material}: wrote {result['readme']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
