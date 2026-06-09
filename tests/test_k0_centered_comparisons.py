import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    path = scripts / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_detect_curvature_labels_orders_by_inside_mean():
    compare = load_script("plot_k0_centered_comparisons")
    summary = {
        "curvature_datasets": {
            "G0.S23": {"inside_mean": 15.0},
            "G0.S22": {"inside_mean": -16.0},
        }
    }

    assert compare.detect_curvature_labels(summary) == ("G0.S22", "G0.S23")


def test_local_significance_exchange_and_metric_boundary_stats():
    compare = load_script("plot_k0_centered_comparisons")
    window = compare.WindowInfo(
        material="X",
        mode_label="m",
        k_left=1.0,
        k0=2.0,
        k_right=3.0,
        delta_k_edt=2.0,
        path_labels=[],
        kappa_minus_label="minus",
        kappa_plus_label="plus",
    )
    k = np.array([0, 1, 2, 3, 4], dtype=float)
    kappa_minus = np.array([1, -4, -4, -4, 1], dtype=float)
    kappa_plus = np.array([1, 4, 4, 4, 1], dtype=float)
    metric = np.array([2, 5, 6, 50, 3], dtype=float)

    stats = compare.local_significance(window, k, kappa_minus, kappa_plus, k, metric)

    assert stats["mean_inside_abs_kappa_minus"] == 4.0
    assert stats["mean_inside_abs_kappa_plus"] == 4.0
    assert stats["exchange_score"] == 0.0
    assert stats["metric_inside_mean"] == pytest.approx((5 + 6 + 50) / 3)
    assert stats["metric_outside_mean"] == pytest.approx((2 + 3) / 2)
    assert stats["max_metric_inside_window"] == 50
    assert stats["max_metric_left_boundary"] == 5
    assert stats["max_metric_right_boundary"] == 50


def test_missing_result_writes_status_json():
    compare = load_script("plot_k0_centered_comparisons")
    workdir = ROOT / "epw_analysis" / "_test_k0_missing_result"
    workdir.mkdir(parents=True, exist_ok=True)
    summary_path = workdir / "edt_local_significance_summary.json"
    if summary_path.exists():
        summary_path.unlink()
    paths = {
        "edt_geometry_summary": workdir / "edt_geometry_summary.json",
        "edt_geometry_comparison": workdir / "edt_geometry_comparison.csv",
        "quantum_metric": workdir / "quantum_metric_on_agr_path.csv",
        "wannier_bands": workdir / "wannier_bands_on_agr_path.csv",
        "agr_curvature": None,
    }

    result = compare.build_missing_result("Nb", workdir, paths, ["agr_curvature"])

    assert result.status == "missing_required_inputs"
    assert summary_path.exists()
    assert result.row["material"] == "Nb"
    summary_path.unlink()
    try:
        workdir.rmdir()
    except OSError:
        pass


def test_nb_mode_selector_prefers_finite_window_pair():
    generator = load_script("generate_nb_mode_summaries")
    k = np.linspace(-1.0, 1.0, 41)
    datasets = [
        np.column_stack([k, 0.1 * k**2]),
        np.column_stack([k, -10.0 + 5.0 * k**2]),
        np.column_stack([k, 10.0 - 5.0 * k**2]),
    ]
    mode = generator.NbMode("Nb_test", "test", Path("synthetic.agr"), Path("synthetic.json"))

    minus, plus, metadata = generator.auto_select_adjacent_pair(mode, datasets, fit_half_window=5, threshold=0.08)

    assert (minus, plus) == (1, 2)
    assert metadata["edt_width"] > 0.0


def test_ybacuo_s4_config_uses_corrected_band_pair_and_window():
    panels = load_script("make_material_figure2_panels")
    config = next(item for item in panels.CONFIGS if item.key == "YBa2Cu3O7")

    assert config.dataset_minus == 146
    assert config.dataset_plus == 147
    assert config.manual_k_left == 1.062522
    assert config.manual_k_right == 1.339270
