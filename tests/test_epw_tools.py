import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_collect_final_spreads_uses_final_wf_values():
    collect_epw_qc = load_script("collect_epw_qc")
    wout = """
 Final State
  WF centre and spread    1  (  0.0,  0.0,  0.0 )     0.12300000
  WF centre and spread    2  (  1.0,  1.0,  1.0 )     0.45600000
    Final Spread (Ang^2)       Omega Total  =     0.57900000
 All done: wannier90 exiting
"""

    spreads, spread_mean, spread_max = collect_epw_qc.parse_final_spreads(wout)

    assert spreads == "0.123;0.456"
    assert spread_mean == "0.2895"
    assert spread_max == "0.456"


def test_status_reads_standalone_wannier_completion():
    check_epw_status = load_script("check_epw_status")

    status = check_epw_status.wannier_status("", "All done: wannier90 exiting\n")

    assert status == "standalone w90 completed"
