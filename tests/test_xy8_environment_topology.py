from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
ANALYSIS_DIR = CODE_DIR / "analysis"
for path in (CODE_DIR, ANALYSIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import analyze_xy8_environment_topology as exporter
import xy8_environment_scan_config as scan_config
import xy8_environment_topology as environment_topology


def read_csv(name):
    path = ROOT / "data" / "processed" / name
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_confirmation_selection_was_frozen_before_refined_scores():
    result = environment_topology.verify_confirmation_selection_freeze()
    assert result["verified"] is True
    assert all(result["checks"].values())
    assert result["record"]["selected_surface_count"] == 226
    assert result["record"]["preregistered_surface_count"] == 60
    assert result["record"]["confirmation_scores_inspected_before_freeze"] is False


def test_discovery_topology_has_one_region_and_connectivity_agrees_everywhere():
    rows = read_csv(exporter.THRESHOLDS_CSV)
    primary = [row for row in rows if float(row["relative_limit"]) == pytest.approx(1.10)]
    assert len(rows) == 1496 * 6
    assert len(primary) == 1496
    assert {int(row["beta_0_four"]) for row in primary} == {1}
    assert {int(row["beta_0_eight"]) for row in primary} == {1}
    assert all(row["connectivity_agrees"] == "True" for row in primary)


def test_confirmation_selection_distinguishes_frozen_and_supplementary_rules():
    rows = read_csv(exporter.SELECTION_CSV)
    assert len(rows) == 226
    assert sum(row["selected_by_preregistered_rule"] == "True" for row in rows) == 60
    assert sum(row["selected_by_supplementary_exact_N_rule"] == "True" for row in rows) == 180
    reasons = {reason for row in rows for reason in row["selection_reasons"].split(";")}
    assert "reference_environment" in reasons
    assert "operational_map_corner" in reasons
    assert "supplementary_exact_N_transition" in reasons


def test_medium_and_fine_grids_match_frozen_resolution_and_exact_resonances():
    config = scan_config.load_config()
    medium = environment_topology.build_confirmation_tau_grid(config, "medium")
    fine = environment_topology.build_confirmation_tau_grid(config, "fine")
    assert medium.size == 962
    assert fine.size == 1922
    for resonance in (1.25e-6, 3.75e-6):
        assert np.any(np.isclose(medium, resonance, rtol=0.0, atol=1e-15))
        assert np.any(np.isclose(fine, resonance, rtol=0.0, atol=1e-15))


def test_all_selected_topologies_pass_refinement_and_exact_depth_changes_survive():
    rows = read_csv(exporter.CONFIRMATION_RESULTS_CSV)
    assert len(rows) == 226
    assert all(row["confirmation_passed"] == "True" for row in rows)
    assert all(row["beta0_stable_discovery_medium_fine"] == "True" for row in rows)
    assert all(row["connectivity_agrees_medium_and_fine"] == "True" for row in rows)
    assert all(row["fine_primary_region_resolved_on_both_axes"] == "True" for row in rows)
    assert all(row["fine_primary_region_avoids_boundaries"] == "True" for row in rows)
    assert all(row["medium_to_fine_timing_width_converged"] == "True" for row in rows)
    assert all(int(row["discovery_best_N"]) == int(row["fine_best_N"]) for row in rows)
    assert max(float(row["medium_to_fine_width_relative_change"]) for row in rows) < 0.05


def test_reference_fine_region_reproduces_canonical_width_and_depth_window():
    rows = [
        row for row in read_csv(exporter.CONFIRMATION_RESULTS_CSV)
        if row["condition_id"] == "zero_strain_T300K"
    ]
    assert len(rows) == 4
    for row in rows:
        assert float(row["fine_best_tau_s"]) == pytest.approx(1.25e-6)
        assert int(row["fine_best_N"]) == 20
        assert (int(row["fine_primary_N_min"]), int(row["fine_primary_N_max"])) == (10, 34)
        assert float(row["fine_timing_width_estimate_s"]) * 1e6 == pytest.approx(0.009)


def test_committed_confirmation_artifacts_match_summary_hashes_and_shapes():
    summary_path = ROOT / "data" / "processed" / exporter.SUMMARY_JSON
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    confirmation = summary["confirmation"]
    assert summary["confirmation_status"] == "complete"
    assert confirmation["confirmation_pass_count"] == 226
    assert confirmation["confirmation_failure_count"] == 0
    assert confirmation["exact_N_change_survives_count"] == 180
    assert confirmation["fine_primary_beta_0_four_distribution"] == {"1": 226}
    assert confirmation["fine_four_eight_disagreement_count"] == 0
    assert confirmation["maximum_medium_to_fine_width_relative_change"] < 0.05
    for record in confirmation["artifacts"].values():
        path = ROOT / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    archive = np.load(ROOT / confirmation["artifacts"]["surfaces"]["path"])
    assert archive["eta_fine"].shape == (226, 1922, 32)
    assert archive["discovery_surface_indices"].shape == (226,)
