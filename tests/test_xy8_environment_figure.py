from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "plotting" / "make_xy8_environment_topology.py"
SPEC = importlib.util.spec_from_file_location("make_xy8_environment_topology", SCRIPT)
figure = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(figure)


def read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / "data" / "processed" / name).open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def test_stage6_depth_maps_use_complete_frozen_axial_slice():
    rows = figure.build_depth_map_rows(read_csv(figure.SCORE_SUMMARY))
    assert len(rows) == 2 * 11 * 11
    for policy in ("fixed_reference_carrier", "ideal_retracked_carrier"):
        selected = [row for row in rows if row["carrier_policy"] == policy]
        assert len({row["temperature_K"] for row in selected}) == 11
        assert len({row["amplitude_microstrain"] for row in selected}) == 11
        assert all(row["boundary_status"] == "resolved_interior_on_discovery_grid" for row in selected)


def test_stage6_carrier_policy_comparison_preserves_expected_discrete_depths():
    rows = figure.build_depth_map_rows(read_csv(figure.SCORE_SUMMARY))
    fixed = {row["best_N"] for row in rows if row["carrier_policy"] == "fixed_reference_carrier"}
    retracked = {row["best_N"] for row in rows if row["carrier_policy"] == "ideal_retracked_carrier"}
    assert fixed == {12, 14, 16, 20}
    assert retracked == {20}


def test_stage6_contours_are_extracted_from_frozen_fine_grid_without_refitting():
    selection = read_csv(figure.CONFIRMATION_SELECTION)
    results = read_csv(figure.CONFIRMATION_RESULTS)
    with np.load(ROOT / "data" / "processed" / figure.CONFIRMATION_SURFACES) as archive:
        rows, examples = figure.build_contour_rows(selection, results, archive)
    assert rows
    assert [row["condition_id"] for row in examples] == [
        figure.REFERENCE_CONDITION,
        figure.SHIFTED_CONDITION,
    ]
    reference, shifted = examples
    assert (reference["best_N"], reference["window_N_min"], reference["window_N_max"]) == (20, 10, 34)
    assert (shifted["best_N"], shifted["window_N_min"], shifted["window_N_max"]) == (12, 7, 16)
    assert all(row["beta_0_four"] == row["beta_0_eight"] == 1 for row in examples)
    assert all(row["confirmation_passed"] is True for row in examples)
    assert {row["example"] for row in rows} == {"Nominal reference", "Environment-shifted"}


def test_committed_stage6_summary_reports_only_confirmed_claims():
    path = ROOT / "data" / "processed" / figure.FIGURE_SUMMARY
    summary = json.loads(path.read_text(encoding="utf-8"))
    result = summary["confirmed_result"]
    assert summary["stage"] == "stage6_environment_topology_communication"
    assert result["discovery_surface_count"] == 1496
    assert result["selected_fine_grid_surface_count"] == 226
    assert result["confirmation_pass_count"] == 226
    assert result["confirmation_failure_count"] == 0
    assert result["fine_beta_0_distribution"] == {"1": 226}
    assert result["four_eight_disagreement_count"] == 0
    assert summary["frozen_slice"]["fixed_carrier_best_depths"] == [12, 14, 16, 20]
    assert summary["frozen_slice"]["retracked_carrier_best_depths"] == [20]
