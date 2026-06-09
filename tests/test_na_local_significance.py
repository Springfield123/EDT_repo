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


def test_window_masks_use_same_width_neighbors():
    local = load_script("analyze_na_local_significance")
    k = np.array([0, 1, 2, 3, 4, 5], dtype=float)

    masks = local.window_masks(k, 2.0, 3.0)

    assert masks["inside"].tolist() == [False, False, True, True, False, False]
    assert masks["outside"].tolist() == [True, True, False, False, True, True]
    assert masks["left_neighbor"].tolist() == [False, True, False, False, False, False]
    assert masks["right_neighbor"].tolist() == [False, False, False, False, True, False]


def test_contribution_fraction_and_k0_interpolation():
    local = load_script("analyze_na_local_significance")
    values = np.array([1, 2, 3, 4, 5], dtype=float)
    mask = np.array([False, False, True, True, False])

    assert local.contribution_fraction(values, mask) == pytest.approx(7.0 / 15.0)
    assert local.interp_at(np.array([0, 1, 2], dtype=float), np.array([0, 10, 20], dtype=float), 1.5) == 15.0


def test_summarize_channel_reports_local_contrast():
    local = load_script("analyze_na_local_significance")
    k = np.array([0, 1, 2, 3, 4], dtype=float)
    values = np.array([1, 2, 4, 8, 16], dtype=float)
    masks = local.window_masks(k, 2.0, 3.0)

    stats = local.summarize_channel(k, values, masks, 2.5)

    assert stats["count_inside"] == 2
    assert stats["count_left_neighbor"] == 1
    assert stats["count_right_neighbor"] == 1
    assert stats["contribution_fraction_abs"] == pytest.approx(12.0 / 31.0)
    assert stats["inside_mean_abs"] == 6.0
    assert stats["neighbor_mean_abs"] == 9.0
    assert stats["local_background_contrast_abs"] == pytest.approx(2.0 / 3.0)
    assert stats["k0_value"] == 6.0
