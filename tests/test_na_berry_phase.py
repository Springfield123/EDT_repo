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


def test_parallel_transport_makes_overlaps_real_positive():
    berry_phase = load_script("analyze_na_berry_phase")
    evecs = np.zeros((3, 2, 2), dtype=complex)
    evecs[0] = np.eye(2)
    evecs[1] = np.eye(2) @ np.diag([np.exp(0.4j), np.exp(-0.7j)])
    evecs[2] = np.eye(2) @ np.diag([np.exp(1.1j), np.exp(0.2j)])

    transported, increments, residuals, overlap_abs = berry_phase.parallel_transport_phases(evecs)

    for idx in range(2):
        for band in range(2):
            overlap = np.vdot(transported[idx, :, band], transported[idx + 1, :, band])
            assert overlap.real == pytest.approx(1.0)
            assert overlap.imag == pytest.approx(0.0, abs=1e-14)
    assert np.allclose(overlap_abs, 1.0)
    assert np.nanmax(np.abs(residuals)) < 1e-14
    assert increments[0, 0] == pytest.approx(-0.4)
    assert increments[0, 1] == pytest.approx(0.7)


def test_phase_stats_classifies_integer_and_fractional_winding():
    berry_phase = load_script("analyze_na_berry_phase")
    inside = np.array([True, True, False])
    increments = np.array([np.pi, np.pi, 0.25 * np.pi])

    stats = berry_phase.phase_stats(increments, inside, left_boundary_idx=0, right_boundary_idx=1)

    assert stats["inside_total_phase_rad"] == pytest.approx(2 * np.pi)
    assert stats["inside_winding_cycles"] == pytest.approx(1.0)
    assert stats["inside_winding_classification"] == "near_integer"
    assert berry_phase.winding_classification(0.33) == "fractional"
