from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

PAPER_ASSETS = {
    "main_paper": ROOT / "docs" / "main_paper_draft.tex",
    "supplementary_information": ROOT / "docs" / "supplementary_information.tex",
    "k3c60_agr": ROOT / "data" / "k3c60" / "tmp.agr",
    "fig1_script": ROOT / "scripts" / "fig1_edt_framework_camera_ready.py",
    "k3c60_localfit_script": ROOT / "scripts" / "superfinger_k3c60_edt_localfit_v6.py",
    "plaquette_script": ROOT / "scripts" / "demo_edt_gyro_trotter_4q_fig34_clean.py",
    "nv_script": ROOT / "scripts" / "make_fig_xy8_qre.py",
    "nv_api": ROOT / "scripts" / "nvratemodel_api.py",
}


def asset_path(name: str) -> Path:
    return PAPER_ASSETS[name]
