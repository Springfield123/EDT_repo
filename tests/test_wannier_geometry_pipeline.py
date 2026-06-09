import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_pipeline():
    path = ROOT / "scripts" / "wannier_geometry_pipeline.py"
    spec = importlib.util.spec_from_file_location("wannier_geometry_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_na_win_expected_fields():
    pipeline = load_pipeline()
    parsed = pipeline.parse_win(ROOT / "epw_runs" / "Na" / "pwscf.win")

    assert parsed["projections"] == ["Na:s", "Na:p"]
    assert parsed["num_wann"] == 8
    assert parsed["dis_win_min_ev"] == -12.0
    assert parsed["dis_win_max_ev"] == 18.0
    assert parsed["dis_froz_min_ev"] == -5.0
    assert parsed["dis_froz_max_ev"] == 6.0
    assert parsed["num_iter"] == 1000


def test_parse_na_wout_final_spread_and_termination():
    pipeline = load_pipeline()
    parsed = pipeline.parse_wout(ROOT / "epw_runs" / "Na" / "pwscf.wout")

    assert parsed["normal_termination"] is True
    assert parsed["omega_total_ang2"] == 37.95043629
    assert len(parsed["final_spreads_ang2"]) == 8


def test_k3c60_summary_selected_pair_format():
    pipeline = load_pipeline()
    summary = pipeline.load_summary(ROOT / "data" / "k3c60" / "k3c60_edt_window_summary.json")

    assert pipeline.summary_dataset_id(summary, "dataset_minus") == 207
    assert pipeline.summary_dataset_id(summary, "dataset_plus") == 208
    assert pipeline.summary_selected_pair(summary)["band_minus_label"] == "Band 208"
