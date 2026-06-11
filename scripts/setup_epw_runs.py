#!/usr/bin/env python3
"""Prepare and optionally submit EPW/Wannier first-batch jobs.

The script is designed to be run from the Rorqual EDT scratch directory:

    cd /home/hanghu/links/scratch/EDT
    python3 scripts/setup_epw_runs.py --submit

It can also be run locally with --assume-completed to generate reviewable input
folders when the completed ph.x/dynmat.x outputs are not mounted locally.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


MANIFEST_COLUMNS = [
    "material",
    "source_dir",
    "prefix",
    "ph_done",
    "dynamat_done",
    "nscf_ready",
    "epw_input_created",
    "submitted",
    "job_id",
    "status",
    "notes",
]

FATAL_PATTERNS = (
    "error in routine",
    "stopping ...",
    "segmentation fault",
    "forrtl:",
    "application called mpi_abort",
    "fatal",
    "convergence not achieved",
)

CARD_HEADERS = (
    "ATOMIC_SPECIES",
    "ATOMIC_POSITIONS",
    "CELL_PARAMETERS",
    "K_POINTS",
    "CONSTRAINTS",
    "OCCUPATIONS",
    "HUBBARD",
)


@dataclass(frozen=True)
class MaterialConfig:
    material: str
    source_subdir: str
    pw_input: str
    ph_input: str
    dm_input: str
    agr_summary: str
    agr_file: str
    projections: tuple[str, ...]
    projection_note: str
    scf_grid: tuple[int, int, int]
    nscf_grid: tuple[int, int, int]
    epw_kf_grid: tuple[int, int, int]
    epw_qf_grid: tuple[int, int, int]
    nscf_nbnd: int
    nbndsub: int
    dis_win_min: float
    dis_win_max: float
    dis_froz_min: float
    dis_froz_max: float
    slurm_tasks: int
    slurm_time: str
    stage1_self_energy: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)


MATERIALS: tuple[MaterialConfig, ...] = (
    MaterialConfig(
        material="Na",
        source_subdir="Na",
        pw_input="Na-bcc.in",
        ph_input="Na-bcc-ph.in",
        dm_input="Na-bcc-dm.in",
        agr_summary="data/processed/S1_Na_summary.json",
        agr_file="data/na/Na-m5.agr",
        projections=("Na:s", "Na:p"),
        projection_note="Alkali metal: Na s+p projections on two Na atoms, giving 8 Wannier trial functions.",
        scf_grid=(10, 10, 10),
        nscf_grid=(8, 8, 8),
        epw_kf_grid=(32, 32, 32),
        epw_qf_grid=(1, 1, 1),
        nscf_nbnd=48,
        nbndsub=8,
        dis_win_min=-12.0,
        dis_win_max=18.0,
        dis_froz_min=-5.0,
        dis_froz_max=6.0,
        slurm_tasks=32,
        slurm_time="48:00:00",
        notes=(
            "Gamma-only DFPT from completed workflow; suitable for the current mode-resolved EDT comparison, not full-BZ lambda.",
            "Stability retry: NSCF grid/nbnd reduced after dense first batch stressed pw.x.",
        ),
    ),
    MaterialConfig(
        material="V",
        source_subdir="V",
        pw_input="V-sp229.in",
        ph_input="V-sp229PH.in",
        dm_input="V-sp229DM.in",
        agr_summary="data/processed/S2_V_summary.json",
        agr_file="data/v/V-EP-F1.agr",
        projections=("V:s", "V:p", "V:d"),
        projection_note="Transition metal: V s+p+d projections on two V atoms, giving 18 Wannier trial functions.",
        scf_grid=(8, 8, 8),
        nscf_grid=(8, 8, 8),
        epw_kf_grid=(32, 32, 32),
        epw_qf_grid=(1, 1, 1),
        nscf_nbnd=60,
        nbndsub=18,
        dis_win_min=-12.0,
        dis_win_max=16.0,
        dis_froz_min=-6.0,
        dis_froz_max=6.0,
        slurm_tasks=32,
        slurm_time="72:00:00",
        notes=(
            "Metallic d manifold; first-batch windows intentionally include nearby s/p bands.",
            "V completed in the previous dense batch; these reduced settings are for reproducible retries only.",
        ),
    ),
    MaterialConfig(
        material="Nb",
        source_subdir="Nb",
        pw_input="Nb-cube.in",
        ph_input="Nb-cubePH.in",
        dm_input="Nb-cubeDM.in",
        agr_summary="",
        agr_file="",
        projections=("Nb:s", "Nb:p", "Nb:d"),
        projection_note="Transition metal: Nb s+p+d projections on two Nb atoms, giving 18 Wannier trial functions.",
        scf_grid=(8, 8, 8),
        nscf_grid=(8, 8, 8),
        epw_kf_grid=(32, 32, 32),
        epw_qf_grid=(1, 1, 1),
        nscf_nbnd=120,
        nbndsub=18,
        dis_win_min=-12.0,
        dis_win_max=14.0,
        dis_froz_min=-6.0,
        dis_froz_max=6.0,
        slurm_tasks=32,
        slurm_time="72:00:00",
        notes=(
            "No AGR summary found in this workspace; EPW output is still prepared for EPC/geometry comparison.",
            "Stability retry: reduced from 16^3/220 bands after Nb NSCF cdiaghg/cholesky failure.",
        ),
    ),
    MaterialConfig(
        material="YBaCuO",
        source_subdir="YBaCuO",
        pw_input="YBa2CuO7.in",
        ph_input="YBa2CuO7-PH.in",
        dm_input="YBa2CuO7-DM.in",
        agr_summary="data/processed/S4_YBa2Cu3O7_summary.json",
        agr_file="data/ybacu3o7/YBa2Cu3O7_m33.agr",
        projections=("Cu:d", "O:p", "Y:d", "Y:s", "Ba:s"),
        projection_note="Cuprate oxide: Cu d + O p manifold with Y d/s and Ba s, giving exactly 44 Wannier trial functions.",
        scf_grid=(8, 8, 4),
        nscf_grid=(6, 6, 3),
        epw_kf_grid=(18, 18, 9),
        epw_qf_grid=(1, 1, 1),
        nscf_nbnd=110,
        nbndsub=44,
        dis_win_min=-8.0,
        dis_win_max=10.0,
        dis_froz_min=-5.0,
        dis_froz_max=3.0,
        slurm_tasks=48,
        slurm_time="48:00:00",
        notes=(
            "Oxide/cuprate first batch; expect projection/window tuning may be needed after inspecting Wannier spreads.",
            "Stability retry: reduced from 12x12x6/220 bands after YBaCuO NSCF band non-convergence; second retry uses 6x6x3/110 bands.",
        ),
    ),
    MaterialConfig(
        material="K3C60",
        source_subdir="KC60",
        pw_input="KC60.in",
        ph_input="KC60PH.in",
        dm_input="KC60DM.in",
        agr_summary="data/k3c60/k3c60_edt_window_summary.json",
        agr_file="data/k3c60/tmp.agr",
        projections=("C:p", "K:s"),
        projection_note=(
            "Fulleride first pass: C p orbitals for the C60 t1u/t1g-like manifold plus K s states. "
            "This gives 183 Wannier trial functions for 60 C atoms and 3 K atoms; add C:s only if the "
            "post-run spreads/interpolated bands show the disentanglement is too narrow."
        ),
        scf_grid=(4, 4, 4),
        nscf_grid=(4, 4, 4),
        epw_kf_grid=(8, 8, 8),
        epw_qf_grid=(1, 1, 1),
        nscf_nbnd=320,
        nbndsub=183,
        dis_win_min=-30.0,
        dis_win_max=30.0,
        dis_froz_min=-2.0,
        dis_froz_max=2.0,
        slurm_tasks=96,
        slurm_time="24:00:00",
        stage1_self_energy=True,
        notes=(
            "K3C60 DFPT is now treated as completed; stage this from the KC60 source input folder used by the QE runner.",
            "Stage 1 mirrors the successful Na-style EPW input: Wannierize, write EPB data, and enable phonselfen/specfun_ph in the first run.",
            "The K3C60 outer disentanglement window is intentionally wide because [-8, 8] eV contained only 104 states at k-point 1 for 183 target WFs.",
            "The AGR EDT pair is Band 208/Band 209 for mode m120 along Gamma-X-L-W-Gamma.",
        ),
    ),
)

SKIPPED_MATERIALS: dict[str, str] = {}


def strip_comments(line: str) -> str:
    return line.split("!", 1)[0].strip()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def grid_text(grid: tuple[int, int, int]) -> str:
    return f"{grid[0]} {grid[1]} {grid[2]}"


def render_uniform_kpoints_crystal(grid: tuple[int, int, int]) -> list[str]:
    nk1, nk2, nk3 = grid
    total = nk1 * nk2 * nk3
    weight = 1.0 / total
    lines = ["K_POINTS crystal", str(total)]
    for i in range(nk1):
        for j in range(nk2):
            for k in range(nk3):
                lines.append(
                    f"  {i / nk1:12.8f}  {j / nk2:12.8f}  {k / nk3:12.8f}  {weight:.8e}"
                )
    return lines


def find_namelist(text: str, name: str) -> list[str]:
    pattern = re.compile(rf"^\s*&{re.escape(name)}\b", re.IGNORECASE)
    lines = text.splitlines()
    body: list[str] = []
    in_block = False
    for line in lines:
        if not in_block and pattern.search(line):
            in_block = True
            continue
        if in_block:
            if strip_comments(line).startswith("/"):
                break
            body.append(line)
    return body


def update_namelist_lines(body: list[str], replacements: dict[str, str], drop_keys: Iterable[str] = ()) -> list[str]:
    drop = {k.lower() for k in drop_keys}
    replaced: set[str] = set()
    out: list[str] = []
    key_pattern = re.compile(r"^\s*([A-Za-z0-9_()]+)\s*=")

    for line in body:
        clean = strip_comments(line)
        if not clean:
            continue
        segments = [segment.strip() for segment in clean.split(",") if segment.strip()]
        if not segments:
            continue
        wrote_segment = False
        for segment in segments:
            match = key_pattern.match(segment)
            if not match:
                out.append(f"    {segment},")
                wrote_segment = True
                continue
            key = match.group(1).lower()
            if key in drop:
                wrote_segment = True
                continue
            if key in replacements:
                out.append(f"    {match.group(1)} = {replacements[key]},")
                replaced.add(key)
            else:
                out.append(f"    {segment},")
            wrote_segment = True
        if not wrote_segment:
            out.append(line.rstrip())

    for key, value in replacements.items():
        if key not in replaced:
            out.append(f"    {key} = {value},")
    return out


def extract_card(text: str, card_name: str) -> tuple[str, list[str]]:
    lines = text.splitlines()
    header = ""
    body: list[str] = []
    in_card = False
    for line in lines:
        clean = strip_comments(line).upper()
        first = clean.split(maxsplit=1)[0] if clean else ""
        if not in_card and first == card_name:
            in_card = True
            header = line.rstrip()
            continue
        if in_card:
            if clean.startswith("&") or first in CARD_HEADERS:
                break
            if line.strip():
                body.append(line.rstrip())
    return header, body


def parse_atomic_species(text: str) -> list[tuple[str, str, str]]:
    _header, lines = extract_card(text, "ATOMIC_SPECIES")
    species: list[tuple[str, str, str]] = []
    for line in lines:
        parts = strip_comments(line).split()
        if len(parts) >= 3:
            species.append((parts[0], parts[1], parts[2]))
    return species


def parse_atomic_positions(text: str) -> tuple[str, list[str]]:
    return extract_card(text, "ATOMIC_POSITIONS")


def parse_cell_parameters(text: str) -> tuple[str, list[str]]:
    return extract_card(text, "CELL_PARAMETERS")


def parse_k_grid(text: str) -> tuple[int, int, int] | None:
    header, lines = extract_card(text, "K_POINTS")
    if "automatic" not in header.lower() or not lines:
        return None
    parts = strip_comments(lines[0]).split()
    if len(parts) < 3:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None


def parse_scalar_from_namelist(body: list[str], key: str) -> str | None:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*([^,!/]+)", re.IGNORECASE)
    for line in body:
        match = pattern.search(line)
        if match:
            return match.group(1).strip().strip(",")
    return None


def parse_int_scalar(value: str | None) -> int | None:
    if value is None:
        return None
    match = re.search(r"[-+]?\d+", value)
    return int(match.group(0)) if match else None


def used_species_from_positions(position_lines: list[str]) -> set[str]:
    used: set[str] = set()
    for line in position_lines:
        parts = strip_comments(line).split()
        if len(parts) >= 4:
            used.add(parts[0])
    return used


def extract_final_coordinates(pw_output: Path) -> tuple[str, list[str], str, list[str]] | None:
    if not pw_output.exists():
        return None
    text = read_text(pw_output)
    starts = [m.start() for m in re.finditer(r"Begin final coordinates", text, re.IGNORECASE)]
    if not starts:
        return None
    block = text[starts[-1] :]
    end_match = re.search(r"End final coordinates", block, re.IGNORECASE)
    if end_match:
        block = block[: end_match.start()]
    cell_header, cell_lines = parse_cell_parameters(block)
    pos_header, pos_lines = parse_atomic_positions(block)
    if cell_header and len(cell_lines) >= 3 and pos_header and pos_lines:
        return cell_header, cell_lines, pos_header, pos_lines
    return None


def find_latest_run_dir(source_dir: Path) -> Path | None:
    candidates = [p for p in source_dir.glob("run_*") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def sorted_run_dirs(source_dir: Path) -> list[Path]:
    return sorted([p for p in source_dir.glob("run_*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)


def find_first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def output_has_fatal_error(path: Path) -> bool:
    if not path.exists():
        return False
    tail = read_text(path)[-200000:].lower()
    return any(pattern in tail for pattern in FATAL_PATTERNS)


def output_is_done(path: Path, allow_no_job_done: bool = False) -> bool:
    if not path.exists() or output_has_fatal_error(path):
        return False
    text = read_text(path)[-200000:].lower()
    if "job done" in text:
        return True
    return allow_no_job_done and bool(text.strip())


def find_qe_outputs(run_dir: Path | None, material: str) -> dict[str, Path | None]:
    if run_dir is None:
        return {"pw": None, "ph": None, "dm": None}
    lower = material.lower()
    pw = find_first_existing(list(run_dir.glob("*.pw.out")) + list(run_dir.glob("*pw*.out")))
    ph = find_first_existing(list(run_dir.glob("*.ph.out")) + list(run_dir.glob("*ph*.out")))
    dm = find_first_existing(list(run_dir.glob("*.dm.out")) + list(run_dir.glob("*dm*.out")) + list(run_dir.glob("*dynmat*.out")))
    if pw is None:
        pw = find_first_existing(run_dir.glob(f"*{lower}*.out"))
    return {"pw": pw, "ph": ph, "dm": dm}


def find_best_run_dir(source_dir: Path, material: str) -> tuple[Path | None, dict[str, Path | None]]:
    latest_run_dir: Path | None = None
    latest_outputs: dict[str, Path | None] = {"pw": None, "ph": None, "dm": None}
    for idx, run_dir in enumerate(sorted_run_dirs(source_dir)):
        outputs = find_qe_outputs(run_dir, material)
        if idx == 0:
            latest_run_dir = run_dir
            latest_outputs = outputs
        ph_ok = output_is_done(outputs["ph"]) if outputs["ph"] else False
        dm_ok = output_is_done(outputs["dm"], allow_no_job_done=True) if outputs["dm"] else False
        if ph_ok and dm_ok:
            return run_dir, outputs
    return latest_run_dir, latest_outputs


def parse_fildyn(ph_text: str) -> str:
    match = re.search(r"fildyn\s*=\s*['\"]([^'\"]+)['\"]", ph_text, re.IGNORECASE)
    return match.group(1) if match else "dmat"


def parse_amass_lines(ph_text: str) -> list[str]:
    lines = []
    for line in ph_text.splitlines():
        clean = strip_comments(line)
        if re.match(r"amass\(\d+\)\s*=", clean, re.IGNORECASE):
            lines.append(clean.rstrip(",") + ",")
    return lines


def safe_link_or_copy(src: Path, dest: Path, mode: str) -> str:
    if not src.exists():
        return "missing"
    if dest.exists() or dest.is_symlink():
        return "exists"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            shutil.copy2(src, dest)
        return "copied"
    try:
        rel_src = os.path.relpath(src, dest.parent)
        dest.symlink_to(rel_src)
        return "symlinked"
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            shutil.copy2(src, dest)
        return "copied_after_symlink_failed"


def link_matching_files(sources: Iterable[Path], dest_dir: Path, mode: str) -> list[str]:
    actions = []
    seen: set[Path] = set()
    for src in sources:
        if not src.exists() or src in seen:
            continue
        seen.add(src)
        action = safe_link_or_copy(src, dest_dir / src.name, mode)
        actions.append(f"{action} {src.name}")
    return actions


def format_namelist(name: str, lines: list[str]) -> str:
    return f"&{name}\n" + "\n".join(lines) + "\n/"


def render_qe_input(
    source_text: str,
    calculation: str,
    cfg: MaterialConfig,
    pseudo_dir_text: str,
    outdir: str,
    k_grid: tuple[int, int, int],
    nbnd: int,
    final_coords: tuple[str, list[str], str, list[str]] | None,
    nscf: bool,
) -> str:
    control_lines = [
        f"    calculation = '{calculation}',",
        "    restart_mode = 'from_scratch',",
        "    prefix = 'pwscf',",
        f"    outdir = '{outdir}',",
        f"    pseudo_dir = '{pseudo_dir_text}',",
        "    verbosity = 'high',",
        "    wf_collect = .true.,",
    ]

    system_body = find_namelist(source_text, "system")
    _pos_header, pos_lines_original = parse_atomic_positions(source_text)
    species_original = parse_atomic_species(source_text)
    final_cell_header = ""
    final_cell_lines: list[str] = []
    final_pos_header = ""
    final_pos_lines: list[str] = []

    if final_coords:
        final_cell_header, final_cell_lines, final_pos_header, final_pos_lines = final_coords

    pos_header = final_pos_header or parse_atomic_positions(source_text)[0]
    pos_lines = final_pos_lines or pos_lines_original
    cell_header = final_cell_header or parse_cell_parameters(source_text)[0]
    cell_lines = final_cell_lines or parse_cell_parameters(source_text)[1]

    used_species = used_species_from_positions(pos_lines)
    species_lines = [sp for sp in species_original if sp[0] in used_species]
    nat = len(pos_lines)
    ntyp = len(species_lines)

    replacements = {
        "nat": str(nat),
        "ntyp": str(ntyp),
        "nbnd": str(nbnd),
    }
    if nscf:
        replacements["nosym"] = ".true."
        replacements["noinv"] = ".true."

    system_lines = update_namelist_lines(system_body, replacements)
    electron_replacements = {"conv_thr": "1.0d-8", "electron_maxstep": "200"}
    if nscf:
        electron_replacements["diagonalization"] = "'cg'"
    electrons_lines = update_namelist_lines(
        find_namelist(source_text, "electrons"),
        electron_replacements,
        drop_keys=("scf_must_converge",),
    )
    if not electrons_lines:
        electrons_lines = [f"    {key} = {value}," for key, value in electron_replacements.items()]

    text = [
        format_namelist("CONTROL", control_lines),
        format_namelist("SYSTEM", system_lines),
        format_namelist("ELECTRONS", electrons_lines),
        "ATOMIC_SPECIES",
    ]
    text.extend(f"  {sym:<4} {mass:<14} {pseudo}" for sym, mass, pseudo in species_lines)
    if cell_header and cell_lines:
        text.append(cell_header)
        text.extend(cell_lines)
    text.append(pos_header or "ATOMIC_POSITIONS crystal")
    text.extend(pos_lines)
    if nscf:
        text.extend(render_uniform_kpoints_crystal(k_grid))
    else:
        text.append("K_POINTS automatic")
        text.append(f"  {k_grid[0]} {k_grid[1]} {k_grid[2]}  0 0 0")
    return "\n".join(text)


def render_epw_input(
    cfg: MaterialConfig,
    source_dir: Path,
    source_run_dir: Path | None,
    qe_outdir: str,
    amass_lines: list[str],
    coarse_k: tuple[int, int, int],
    coarse_q: tuple[int, int, int],
) -> str:
    stage1_self_energy = ".true." if cfg.stage1_self_energy else ".false."
    lines = [
        "&inputepw",
        "  prefix = 'pwscf',",
        f"  outdir = '{qe_outdir}',",
        "  dvscf_dir = './save',",
        "",
        "  ep_coupling = .true.,",
        "  elph = .true.,",
        "  wannierize = .true.,",
        "  epbwrite = .true.,",
        "  epbread = .false.,",
        "",
        f"  phonselfen = {stage1_self_energy},",
        f"  specfun_ph = {stage1_self_energy},",
        "  wmin_specfun = 0.00,",
        "  wmax_specfun = 0.25,",
        "  nw_specfun = 300,",
        "  temps(1) = 300.0,",
        "  nstemp = 1,",
        "",
        f"  nbndsub = {cfg.nbndsub},",
        "  num_iter = 1000,",
    ]
    if cfg.material not in {"V", "Nb", "YBaCuO"}:
        lines.extend(
            [
                f"  dis_win_min = {cfg.dis_win_min:.6f},",
                f"  dis_win_max = {cfg.dis_win_max:.6f},",
                f"  dis_froz_min = {cfg.dis_froz_min:.6f},",
                f"  dis_froz_max = {cfg.dis_froz_max:.6f},",
            ]
        )
    else:
        lines.append("  ! Disentanglement windows omitted for this retry so Wannier90 can use all NSCF bands.")
    for idx, proj in enumerate(cfg.projections, start=1):
        lines.append(f"  proj({idx}) = '{proj}',")
    for line in amass_lines:
        lines.append(f"  {line}")
    lines.extend(
        [
            "",
            f"  nk1 = {coarse_k[0]}, nk2 = {coarse_k[1]}, nk3 = {coarse_k[2]},",
            f"  nq1 = {coarse_q[0]}, nq2 = {coarse_q[1]}, nq3 = {coarse_q[2]},",
            f"  nkf1 = {cfg.epw_kf_grid[0]}, nkf2 = {cfg.epw_kf_grid[1]}, nkf3 = {cfg.epw_kf_grid[2]},",
            f"  nqf1 = {cfg.epw_qf_grid[0]}, nqf2 = {cfg.epw_qf_grid[1]}, nqf3 = {cfg.epw_qf_grid[2]},",
            "/",
        ]
    )
    return "\n".join(lines)


def render_submit_script(cfg: MaterialConfig) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=epw_{cfg.material}
#SBATCH --account=def-hanghu
#SBATCH --ntasks={cfg.slurm_tasks}
#SBATCH --time={cfg.slurm_time}
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
export CLEAN_QE_OUT="${{CLEAN_QE_OUT:-1}}"

if (( EPW_NPOOL * EPW_NIMAGE != SLURM_NTASKS )); then
  echo "EPW parallel layout error: SLURM_NTASKS=$SLURM_NTASKS but EPW_NPOOL*EPW_NIMAGE=$((EPW_NPOOL * EPW_NIMAGE))" >&2
  echo "Set EPW_NPOOL and EPW_NIMAGE so their product equals SLURM_NTASKS." >&2
  exit 2
fi

cd "$SLURM_SUBMIT_DIR"
if [[ "$CLEAN_QE_OUT" == "1" ]]; then
  echo "[$(date '+%F %T')] Cleaning local retry artifacts in $SLURM_SUBMIT_DIR"
  rm -rf qe_out tmp
  rm -f outputs/scf.out outputs/nscf.out outputs/epw.out
fi
mkdir -p logs outputs qe_out tmp

run_mpi() {{
  local exe="$1"
  local input="$2"
  local output="$3"
  echo "[$(date '+%F %T')] Running $exe on $SLURM_NTASKS MPI tasks"
  mpirun -np "$SLURM_NTASKS" "$exe" -in "$input" > "$output"
}}

run_epw() {{
  local input="$1"
  local output="$2"
  local args=(-npool "$EPW_NPOOL")
  if (( EPW_NIMAGE != 1 )); then
    args+=(-nimage "$EPW_NIMAGE")
  fi
  echo "[$(date '+%F %T')] Running epw.x on $SLURM_NTASKS MPI tasks with ${{args[*]}}"
  mpirun -np "$SLURM_NTASKS" epw.x "${{args[@]}}" -in "$input" > "$output"
}}

run_mpi pw.x scf.in outputs/scf.out
run_mpi pw.x nscf.in outputs/nscf.out
run_epw epw.in outputs/epw.out

echo "[$(date '+%F %T')] EPW workflow finished for {cfg.material}"
"""


def render_readme(
    cfg: MaterialConfig,
    source_dir: Path,
    source_run_dir: Path | None,
    local_pseudo_dir: Path,
    runtime_pseudo_dir: str,
    fildyn: str,
    original_grid: tuple[int, int, int] | None,
    used_final_geometry: bool,
    notes: list[str],
) -> str:
    lines = [
        f"# EPW assumptions for {cfg.material}",
        "",
        f"Source directory: {source_dir}",
        f"Completed QE run directory: {source_run_dir if source_run_dir else 'not found during setup'}",
        "Prefix: pwscf",
        f"Local pseudopotential directory checked during setup: {local_pseudo_dir}",
        f"Runtime pseudopotential directory written to QE inputs: {runtime_pseudo_dir}",
        f"Dynamic matrix file: {fildyn}",
        f"Original SCF/relax k-grid: {grid_text(original_grid) if original_grid else 'not parsed'}",
        f"Generated SCF k-grid: {grid_text(cfg.scf_grid)}",
        f"Generated NSCF k-grid: {grid_text(cfg.nscf_grid)}",
        "Generated NSCF k-point style: explicit K_POINTS crystal list with equal weights, matching the previous BN EPW workflow pattern.",
        f"EPW fine k-grid: {grid_text(cfg.epw_kf_grid)}",
        f"EPW fine q-grid: {grid_text(cfg.epw_qf_grid)}",
        f"NSCF nbnd: {cfg.nscf_nbnd}",
        f"EPW nbndsub/num_wann: {cfg.nbndsub}",
        f"Disentanglement window: [{cfg.dis_win_min}, {cfg.dis_win_max}] eV",
        f"Frozen window: [{cfg.dis_froz_min}, {cfg.dis_froz_max}] eV",
        f"Used final relaxed geometry from pw.x output: {used_final_geometry}",
        "",
        "Projection assumptions:",
        f"- {cfg.projection_note}",
    ]
    lines.extend(f"- proj: {proj}" for proj in cfg.projections)
    if cfg.agr_file:
        lines.append(f"- AGR curvature file planned for comparison: {cfg.agr_file}")
    if cfg.agr_summary:
        lines.append(f"- EDT summary file planned for comparison: {cfg.agr_summary}")
    lines.append("")
    lines.append("Validation notes:")
    for note in (*cfg.notes, *notes):
        lines.append(f"- {note}")
    lines.append("")
    lines.append("EPW spectral-function assumptions:")
    if cfg.stage1_self_energy:
        lines.append("- Enabled phonselfen=.true. and specfun_ph=.true. with wmin/wmax/nw = 0.00/0.25/300 and T=300 K, following the previous BN workflow.")
    else:
        lines.append("- Stage 1 keeps phonselfen=.false., specfun_ph=.false., and a2f=.false. so the first run only writes stable Wannier/EPB/EPW restart payloads.")
        lines.append("- Stage 2 uses epw_epc_restart.in to enable phonselfen=.true., specfun_ph=.true., and a2f=.true. after Stage 1 completes.")
    lines.append("- lpolar and shortrange are not enabled by default because the completed ph.x inputs used epsil=.false.; enable them only after a DFPT run with the required dielectric/Born-charge data.")
    lines.append("- filqf is not set by default because the completed DFPT workflow is Gamma/q=(0,0,0); add a material-specific q-point file only after generating matching DFPT q data.")
    lines.append("")
    lines.append("Restart notes:")
    lines.append("- First run writes epb/epw data with epbread=.false. and epwread=.false.")
    lines.append("- After a successful first run, later restarts can use epbread=.true. and/or epwread=.true. if appropriate.")
    if cfg.material == "K3C60":
        lines.append("- Submit the second-stage EPC restart only after pwscf.wout, pwscf_hr.dat, EPB files, and epwdata.fmt are present.")
    else:
        lines.append("- K3C60 is now handled as its own EPW material; this file records only the assumptions for the current material.")
    return "\n".join(lines)


def manifest_row(
    cfg: MaterialConfig | None,
    source_dir: Path,
    ph_done: bool,
    dynamat_done: bool,
    nscf_ready: bool,
    epw_input_created: bool,
    submitted: bool,
    job_id: str,
    status: str,
    notes: list[str],
    material_name: str | None = None,
) -> dict[str, str]:
    return {
        "material": cfg.material if cfg else (material_name or ""),
        "source_dir": str(source_dir),
        "prefix": "pwscf",
        "ph_done": bool_text(ph_done),
        "dynamat_done": bool_text(dynamat_done),
        "nscf_ready": bool_text(nscf_ready),
        "epw_input_created": bool_text(epw_input_created),
        "submitted": bool_text(submitted),
        "job_id": job_id,
        "status": status,
        "notes": " | ".join(dict.fromkeys(note for note in notes if note)),
    }


def submit_job(material_dir: Path) -> tuple[bool, str, str]:
    cmd = ["sbatch", "submit_epw.sh"]
    try:
        result = subprocess.run(cmd, cwd=material_dir, check=False, text=True, capture_output=True)
    except FileNotFoundError:
        return False, "", "sbatch not found; run on Rorqual login node"
    output = (result.stdout + "\n" + result.stderr).strip()
    match = re.search(r"Submitted batch job\s+(\d+)", output)
    if result.returncode == 0 and match:
        return True, match.group(1), output
    return False, "", output or f"sbatch exited with code {result.returncode}"


def setup_material(
    cfg: MaterialConfig,
    qe_base: Path,
    pseudo_dir: Path,
    runtime_pseudo_dir: str,
    epw_root: Path,
    submit: bool,
    assume_completed: bool,
    link_mode: str,
) -> dict[str, str]:
    source_dir = qe_base / cfg.source_subdir
    material_dir = epw_root / cfg.material
    notes: list[str] = []

    if not source_dir.exists():
        return manifest_row(
            cfg,
            source_dir,
            False,
            False,
            False,
            False,
            False,
            "",
            "failed_validation",
            ["source directory missing"],
        )

    pw_path = source_dir / cfg.pw_input
    ph_path = source_dir / cfg.ph_input
    dm_path = source_dir / cfg.dm_input
    missing_inputs = [str(p) for p in (pw_path, ph_path, dm_path) if not p.exists()]
    if missing_inputs:
        return manifest_row(
            cfg,
            source_dir,
            False,
            False,
            False,
            False,
            False,
            "",
            "failed_validation",
            [f"missing source inputs: {', '.join(missing_inputs)}"],
        )

    source_text = read_text(pw_path)
    ph_text = read_text(ph_path)
    fildyn = parse_fildyn(ph_text)
    amass_lines = parse_amass_lines(ph_text)
    source_system_body = find_namelist(source_text, "system")
    source_nbnd = parse_int_scalar(parse_scalar_from_namelist(source_system_body, "nbnd")) or cfg.nscf_nbnd
    original_grid = parse_k_grid(source_text)
    source_run_dir, outputs = find_best_run_dir(source_dir, cfg.material)

    ph_done = output_is_done(outputs["ph"]) if outputs["ph"] else False
    dynamat_done = output_is_done(outputs["dm"], allow_no_job_done=True) if outputs["dm"] else False

    if assume_completed:
        if not ph_done:
            notes.append("ph.x output not found locally; assume-completed enabled for setup only")
        if not dynamat_done:
            notes.append("dynmat.x output not found locally; assume-completed enabled for setup only")
        ph_done = True
        dynamat_done = True

    if outputs["ph"] and output_has_fatal_error(outputs["ph"]):
        notes.append(f"fatal pattern found in ph output: {outputs['ph']}")
    if outputs["dm"] and output_has_fatal_error(outputs["dm"]):
        notes.append(f"fatal pattern found in dynmat output: {outputs['dm']}")

    if not ph_done or not dynamat_done:
        return manifest_row(
            cfg,
            source_dir,
            ph_done,
            dynamat_done,
            False,
            False,
            False,
            "",
            "failed_validation",
            notes + ["completed ph.x/dynmat.x outputs not found; not submitting EPW"],
        )

    species = parse_atomic_species(source_text)
    missing_pseudos = [pseudo for _sym, _mass, pseudo in species if not (pseudo_dir / pseudo).exists()]
    if missing_pseudos:
        if submit or not assume_completed:
            return manifest_row(
                cfg,
                source_dir,
                ph_done,
                dynamat_done,
                False,
                False,
                False,
                "",
                "failed_validation",
                notes + [f"missing pseudopotentials in {pseudo_dir}: {', '.join(missing_pseudos)}"],
            )
        notes.append(f"missing pseudopotentials in local preview path {pseudo_dir}: {', '.join(missing_pseudos)}")

    material_dir.mkdir(parents=True, exist_ok=True)
    (material_dir / "logs").mkdir(exist_ok=True)
    (material_dir / "outputs").mkdir(exist_ok=True)
    save_dir = material_dir / "save"
    save_dir.mkdir(exist_ok=True)

    final_coords = extract_final_coordinates(outputs["pw"]) if outputs["pw"] else None
    if final_coords:
        notes.append("using final relaxed coordinates from completed pw.x output")
    else:
        notes.append("final relaxed coordinates not parsed; generated inputs use source QE geometry")

    qe_outdir = "./qe_out"
    scf_text = render_qe_input(
        source_text=source_text,
        calculation="scf",
        cfg=cfg,
        pseudo_dir_text=runtime_pseudo_dir,
        outdir=qe_outdir,
        k_grid=cfg.scf_grid,
        nbnd=max(source_nbnd, cfg.nbndsub),
        final_coords=final_coords,
        nscf=False,
    )
    nscf_text = render_qe_input(
        source_text=source_text,
        calculation="nscf",
        cfg=cfg,
        pseudo_dir_text=runtime_pseudo_dir,
        outdir=qe_outdir,
        k_grid=cfg.nscf_grid,
        nbnd=cfg.nscf_nbnd,
        final_coords=final_coords,
        nscf=True,
    )

    write_text(material_dir / "scf.in", scf_text)
    write_text(material_dir / "nscf.in", nscf_text)
    coarse_k = cfg.nscf_grid
    coarse_q = (1, 1, 1)
    epw_text = render_epw_input(
        cfg=cfg,
        source_dir=source_dir,
        source_run_dir=source_run_dir,
        qe_outdir=qe_outdir,
        amass_lines=amass_lines,
        coarse_k=coarse_k,
        coarse_q=coarse_q,
    )
    write_text(material_dir / "epw.in", epw_text)
    write_text(material_dir / "submit_epw.sh", render_submit_script(cfg))
    try:
        os.chmod(material_dir / "submit_epw.sh", 0o755)
    except OSError:
        pass

    source_dmat_candidates = []
    if source_run_dir:
        source_dmat_candidates.extend([source_run_dir / fildyn, *source_run_dir.glob(f"{fildyn}*")])
    source_dmat_candidates.append(source_dir / fildyn)
    dmat_src = find_first_existing(source_dmat_candidates)
    if dmat_src:
        action = safe_link_or_copy(dmat_src, save_dir / Path(dmat_src).name, link_mode)
        notes.append(f"{action} dynamic-matrix file into save/: {dmat_src.name}")
    else:
        notes.append(f"dynamic-matrix file {fildyn} not found; EPW may fail until linked manually")

    artifact_sources: list[Path] = []
    if source_run_dir:
        artifact_sources.extend(source_run_dir.glob("*.dyn*"))
        artifact_sources.extend(source_run_dir.glob("*dvscf*"))
        artifact_sources.extend(source_run_dir.glob("*patterns*"))
        qe_tmp = source_run_dir / "qe_tmp"
        if qe_tmp.exists():
            artifact_sources.extend(qe_tmp.glob("*.dyn*"))
            artifact_sources.extend(qe_tmp.glob("*dvscf*"))
            artifact_sources.extend(qe_tmp.glob("*patterns*"))
            artifact_sources.extend(qe_tmp.glob("_ph0"))
            artifact_sources.extend(qe_tmp.glob("*.phsave"))
            artifact_sources.extend(qe_tmp.glob("**/*.dvscf*"))
            artifact_sources.extend(qe_tmp.glob("**/*patterns*"))
            artifact_sources.extend(qe_tmp.glob("**/*.dyn*"))
    artifact_sources.extend(source_dir.glob("*.dyn*"))
    artifact_sources.extend(source_dir.glob("*dvscf*"))
    artifact_sources.extend(source_dir.glob("*patterns*"))
    artifact_actions = link_matching_files(artifact_sources, save_dir, link_mode)
    if artifact_actions:
        notes.append("save/ artifacts: " + "; ".join(artifact_actions[:12]))
        if len(artifact_actions) > 12:
            notes.append(f"save/ artifacts: plus {len(artifact_actions) - 12} more")
    else:
        notes.append("no dvscf/pattern artifacts found for save/; if EPW fails after namelist parsing, run EPW pp.py or copy DFPT save files into epw_runs/<material>/save")

    write_text(
        material_dir / "README_assumptions.txt",
        render_readme(
            cfg=cfg,
            source_dir=source_dir,
            source_run_dir=source_run_dir,
            local_pseudo_dir=pseudo_dir,
            runtime_pseudo_dir=runtime_pseudo_dir,
            fildyn=fildyn,
            original_grid=original_grid,
            used_final_geometry=final_coords is not None,
            notes=notes,
        ),
    )

    submitted = False
    job_id = ""
    status = "ready"
    if submit:
        submitted, job_id, submit_note = submit_job(material_dir)
        notes.append(f"sbatch: {submit_note}")
        status = "submitted" if submitted else "submit_failed"

    return manifest_row(
        cfg,
        source_dir,
        ph_done,
        dynamat_done,
        True,
        True,
        submitted,
        job_id,
        status,
        notes,
    )


def existing_skipped_rows(qe_base: Path) -> list[dict[str, str]]:
    rows = []
    for material, note in SKIPPED_MATERIALS.items():
        source_dir = qe_base / ("KC60" if material == "K3C60" else material)
        if source_dir.exists() or material == "KC60":
            rows.append(
                manifest_row(
                    None,
                    source_dir,
                    False,
                    False,
                    False,
                    False,
                    False,
                    "",
                    "skipped",
                    [note],
                    material_name=material,
                )
            )
    # Avoid both aliases unless both folders truly exist.
    deduped: list[dict[str, str]] = []
    seen_dirs: set[str] = set()
    for row in rows:
        key = row["source_dir"]
        if key not in seen_dirs:
            deduped.append(row)
            seen_dirs.add(key)
    return deduped


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def merge_manifest_rows(existing_rows: list[dict[str, str]], new_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if any(row.get("material") == "K3C60" for row in new_rows):
        existing_rows = [row for row in existing_rows if row.get("material") != "KC60"]
    merged = {row.get("material", ""): row for row in existing_rows if row.get("material")}
    for row in new_rows:
        if row.get("material"):
            merged[row["material"]] = row
    preferred_order = [cfg.material for cfg in MATERIALS] + list(SKIPPED_MATERIALS)
    ordered = [merged.pop(material) for material in preferred_order if material in merged]
    ordered.extend(row for _material, row in sorted(merged.items()))
    return ordered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qe-base", type=Path, default=Path("/home/hanghu/links/scratch/EDT"), help="Directory containing QE material folders.")
    parser.add_argument("--pseudo-dir", type=Path, default=None, help="Pseudopotential directory. Defaults to QE_BASE/pseudo.")
    parser.add_argument(
        "--runtime-pseudo-dir",
        default=None,
        help="Pseudopotential directory string written into generated QE inputs. Defaults to --pseudo-dir.",
    )
    parser.add_argument("--epw-root", type=Path, default=Path("epw_runs"), help="EPW output root.")
    parser.add_argument("--submit", action="store_true", help="Submit valid material jobs with sbatch.")
    parser.add_argument("--assume-completed", action="store_true", help="Generate inputs even if ph/dynmat outputs are not locally visible.")
    parser.add_argument("--materials", nargs="*", default=None, help="Subset of materials to prepare.")
    parser.add_argument("--link-mode", choices=("symlink", "copy"), default="symlink", help="How to bring dynamic matrix files into EPW folders.")
    args = parser.parse_args(argv)

    qe_base = args.qe_base.resolve()
    pseudo_dir = (args.pseudo_dir or (qe_base / "pseudo")).resolve()
    runtime_pseudo_dir = args.runtime_pseudo_dir or str(pseudo_dir)
    epw_root = args.epw_root.resolve()
    selected = set(args.materials) if args.materials else {cfg.material for cfg in MATERIALS}

    rows: list[dict[str, str]] = []
    for cfg in MATERIALS:
        if cfg.material not in selected:
            continue
        try:
            row = setup_material(
                cfg=cfg,
                qe_base=qe_base,
                pseudo_dir=pseudo_dir,
                runtime_pseudo_dir=runtime_pseudo_dir,
                epw_root=epw_root,
                submit=args.submit,
                assume_completed=args.assume_completed,
                link_mode=args.link_mode,
            )
        except Exception as exc:  # Keep the batch moving.
            row = manifest_row(
                cfg,
                qe_base / cfg.source_subdir,
                False,
                False,
                False,
                False,
                False,
                "",
                "setup_exception",
                [f"{type(exc).__name__}: {exc}"],
            )
        rows.append(row)

    rows.extend(existing_skipped_rows(qe_base))
    manifest_path = epw_root / "epw_run_manifest.csv"
    if args.materials:
        rows = merge_manifest_rows(read_manifest(manifest_path), rows)
    write_manifest(manifest_path, rows)

    print(f"Wrote manifest: {manifest_path}")
    print("Material summary:")
    for row in rows:
        job = f" job_id={row['job_id']}" if row["job_id"] else ""
        print(f"  {row['material']}: {row['status']}{job} - {row['notes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
