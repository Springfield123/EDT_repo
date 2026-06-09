import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extractor_ignores_gamma_only_prose():
    extractor = load_script("extract_epw_epc_strength")
    rows = extractor.extract_rows_from_line(
        "Na",
        Path("README_assumptions.txt"),
        "Gamma-only DFPT from completed workflow; q=(0,0,0)",
    )

    assert rows == []


def test_extractor_parses_lambda_line():
    extractor = load_script("extract_epw_epc_strength")
    rows = extractor.extract_rows_from_line(
        "Na",
        Path("epw.out"),
        "q_index = 3 mode = 2 lambda = 0.125 linewidth = 0.004",
    )

    assert len(rows) == 1
    assert rows[0]["q_index"] == 3
    assert rows[0]["mode"] == 2
    assert rows[0]["lambda_qnu"] == 0.125
    assert rows[0]["linewidth"] == 0.004


def test_epc_stats_ratio():
    compare = load_script("compare_edt_window_with_epc")
    rows = [
        {"inside_edt_window": True, "epc_g2": "2.0"},
        {"inside_edt_window": True, "epc_g2": "4.0"},
        {"inside_edt_window": False, "epc_g2": "1.0"},
    ]

    stats = compare.stats_for_column(rows, "epc_g2")

    assert stats["inside_mean"] == 3.0
    assert stats["outside_mean"] == 1.0
    assert stats["inside_outside_ratio"] == 3.0
