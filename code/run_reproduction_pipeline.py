from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLOTTING = ROOT / "code" / "plotting"
UTILITIES = ROOT / "code" / "utilities"


def run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(PLOTTING / script), *args]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def has_qiskit() -> bool:
    return importlib.util.find_spec("qiskit") is not None


def main() -> None:
    run("make_fig2_k3c60_edt.py")
    run("make_material_figure2_panels.py")
    run("make_fig1_material_summary.py")
    run("make_figS4_material_controls.py")
    run("fig4_plaquette_dynamics.py")
    run("make_fig_xy8_qre.py", "--calibration", "data/raw/nv/xy8_scenario_nominal.json", "--outdir", "outputs/reproduced_figures")
    shutil.copy2(ROOT / "outputs" / "reproduced_figures" / "xy8_qre_nv_results.csv", ROOT / "data" / "processed" / "xy8_qre_nv_results.csv")
    shutil.copy2(ROOT / "outputs" / "reproduced_figures" / "xy8_qre_nv_breakdown.json", ROOT / "data" / "processed" / "xy8_qre_nv_breakdown.json")
    subprocess.run([sys.executable, str(UTILITIES / "export_figure_source_data.py")], check=True, cwd=ROOT)
    if has_qiskit():
        run("demo_edt_gyro_trotter_4q_fig34_clean.py")
    else:
        print("Skipping demo_edt_gyro_trotter_4q_fig34_clean.py because qiskit is not installed.")


if __name__ == "__main__":
    main()
