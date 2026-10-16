from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "code" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import analyze_xy8_resonant_topology as exporter
import analyze_xy8_topology_refinement as refinement
import make_xy8_topology_figure as topology_figure
import xy8_resonant_topology as topology
import xy8_topology_persistence as persistence


CANONICAL_RESULTS = ROOT / "data" / "processed" / "xy8_qre_nv_results.csv"
REGIONS_ARTIFACT = ROOT / "data" / "processed" / "xy8_resonant_regions.csv"
SUMMARY_ARTIFACT = ROOT / "data" / "processed" / "xy8_resonant_topology_summary.json"
REFINEMENT_GRID_ARTIFACT = ROOT / "data" / "processed" / "xy8_topology_grid_refinement.csv"
RESONANCE_ARTIFACT = ROOT / "data" / "processed" / "xy8_topology_resonance_diagnostics.csv"
REFINEMENT_SUMMARY_ARTIFACT = ROOT / "data" / "processed" / "xy8_topology_refinement_summary.json"
THRESHOLD_CURVE_ARTIFACT = ROOT / "data" / "processed" / "xy8_topology_threshold_curve.csv"
PERSISTENCE_EVENTS_ARTIFACT = ROOT / "data" / "processed" / "xy8_topology_persistence_events.csv"
COMPONENT_TRACKS_ARTIFACT = ROOT / "data" / "processed" / "xy8_topology_component_tracks.csv"
PERSISTENCE_SUMMARY_ARTIFACT = ROOT / "data" / "processed" / "xy8_topology_persistence_summary.json"
TOPOLOGY_FIGURE_PNG = ROOT / "outputs" / "reproduced_figures" / "fig_xy8_topology_persistence.png"
TOPOLOGY_FIGURE_PDF = ROOT / "outputs" / "reproduced_figures" / "fig_xy8_topology_persistence.pdf"
NESTED_REGION_SOURCE = ROOT / "outputs" / "reproduced_figures" / "xy8_topology_nested_region_source.csv"
TOPOLOGY_FIGURE_SUMMARY = ROOT / "outputs" / "reproduced_figures" / "xy8_topology_figure_summary.json"


def make_surface(relative_scores) -> topology.ScoreSurface:
    values = np.asarray(relative_scores, dtype=float)
    return topology.validate_surface(
        np.arange(1, values.shape[0] + 1, dtype=float) * 1e-6,
        np.arange(1, values.shape[1] + 1, dtype=int),
        values,
    )


def canonical_points() -> list[tuple[float, int, float]]:
    lines = CANONICAL_RESULTS.read_text(encoding="utf-8").splitlines()
    header_index = lines.index("tau_s,N,eta_proxy")
    rows = csv.DictReader(lines[header_index:])
    return [
        (float(row["tau_s"]), int(row["N"]), float(row["eta_proxy"]))
        for row in rows
    ]


def test_two_separated_low_score_islands_have_two_components():
    surface = make_surface(
        [
            [1.00, 1.05, 3.00, 3.00],
            [1.03, 1.08, 3.00, 3.00],
            [3.00, 3.00, 1.04, 1.06],
            [3.00, 3.00, 1.02, 1.07],
        ]
    )
    result = topology.connected_regions(surface, relative_limit=1.10)
    assert result.beta_0 == 2
    assert result.component_count == 2
    assert result.threshold_percent == pytest.approx(10.0)
    assert np.count_nonzero(result.mask) == 8


def test_diagonal_contact_is_separate_with_four_neighbours():
    surface = make_surface([[1.0, 3.0], [3.0, 1.0]])
    assert topology.count_connected_regions(surface, connectivity=4) == 2
    assert topology.count_connected_regions(surface, connectivity=8) == 1


def test_uniform_near_optimal_surface_is_one_component():
    surface = make_surface(np.ones((3, 5)))
    result = topology.connected_regions(surface, relative_limit=1.0)
    assert result.component_count == 1
    assert np.all(result.mask)
    assert set(np.unique(result.labels)) == {1}


def test_surface_from_points_sorts_axes_and_places_scores_correctly():
    surface = topology.surface_from_points(
        [
            (2e-6, 16, 4.0),
            (1e-6, 16, 2.0),
            (2e-6, 8, 3.0),
            (1e-6, 8, 1.0),
        ]
    )
    assert surface.taus_s.tolist() == [1e-6, 2e-6]
    assert surface.depths.tolist() == [8, 16]
    assert surface.eta.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_incomplete_duplicate_and_invalid_surfaces_are_rejected():
    with pytest.raises(ValueError, match="complete rectangular"):
        topology.surface_from_points([(1e-6, 8, 1.0), (2e-6, 8, 2.0), (2e-6, 16, 3.0)])
    with pytest.raises(ValueError, match="Duplicate"):
        topology.surface_from_points(
            [(1e-6, 8, 1.0), (1e-6, 8, 2.0), (2e-6, 16, 3.0), (2e-6, 16, 4.0)]
        )
    with pytest.raises(ValueError, match="positive finite"):
        topology.validate_surface([1e-6], [8], [[0.0]])
    with pytest.raises(ValueError, match="at least 1.0"):
        topology.connected_regions(make_surface([[1.0]]), relative_limit=0.99)
    with pytest.raises(ValueError, match="either 4 or 8"):
        topology.connected_regions(make_surface([[1.0]]), connectivity=6)


def test_canonical_surface_is_complete_and_contains_global_minimum():
    surface = topology.surface_from_points(canonical_points())
    result = topology.connected_regions(surface, relative_limit=1.10)
    best_row, best_column = np.unravel_index(np.argmin(surface.eta), surface.eta.shape)
    assert surface.taus_s[best_row] == pytest.approx(1.25e-6)
    assert surface.depths[best_column] == 20
    assert result.mask[best_row, best_column]
    assert result.labels[best_row, best_column] > 0
    assert result.component_count >= 1


def test_characterization_reports_resonance_width_and_boundaries():
    surface = make_surface(
        [
            [1.00, 1.04, 3.00],
            [3.00, 3.00, 3.00],
            [3.00, 1.08, 1.02],
        ]
    )
    result = topology.connected_regions(surface, relative_limit=1.10)
    regions = topology.characterize_regions(result, target_frequency_hz=250e3)

    assert [region.component_id for region in regions] == ["R1", "R2"]
    primary, secondary = regions
    assert primary.contains_global_optimum
    assert primary.best_tau_s == pytest.approx(1e-6)
    assert primary.best_depth == 1
    assert primary.resonance_orders == (0,)
    assert primary.resonance_taus_s == pytest.approx((1e-6,))
    assert primary.tau_width_s == pytest.approx(0.0)
    assert primary.depth_span_blocks == 1
    assert primary.touches_tau_lower_boundary
    assert primary.touches_depth_lower_boundary
    assert primary.boundary_status == "tau_lower;depth_lower"
    assert primary.sampling_status == "timing_single_grid_row"

    assert not secondary.contains_global_optimum
    assert secondary.best_tau_s == pytest.approx(3e-6)
    assert secondary.resonance_orders == (1,)
    assert secondary.touches_tau_upper_boundary
    assert secondary.touches_depth_upper_boundary


def test_canonical_region_characterization_matches_nominal_window():
    surface = topology.surface_from_points(canonical_points())
    result = topology.connected_regions(surface, relative_limit=1.10)
    regions = topology.characterize_regions(result, target_frequency_hz=200e3)
    assert len(regions) == 1
    region = regions[0]
    assert region.component_id == "R1"
    assert region.contains_global_optimum
    assert region.best_tau_s == pytest.approx(1.25e-6)
    assert region.best_depth == 20
    assert (region.depth_min, region.depth_max) == (10, 34)
    assert region.resonance_orders == (0,)
    assert region.resonance_taus_s == pytest.approx((1.25e-6,))
    assert region.contains_depth_16
    assert region.contains_depth_20
    assert region.boundary_status == "resolved_interior"
    assert region.sampling_status == "timing_single_grid_row"


def test_exported_csv_and_json_describe_the_same_regions(tmp_path):
    regions_path, summary_path, summary = exporter.write_outputs(
        CANONICAL_RESULTS,
        tmp_path,
        target_frequency_hz=200e3,
        relative_limit=1.10,
        connectivity=4,
    )
    with regions_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    stored_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert stored_summary == summary
    assert stored_summary["beta_0"] == len(rows) == 1
    assert rows[0]["component_id"] == stored_summary["regions"][0]["component_id"]
    assert int(rows[0]["best_N"]) == stored_summary["global_minimum"]["N"] == 20
    assert float(rows[0]["best_tau_us"]) == pytest.approx(1.25)
    assert rows[0]["resonance_identity"] == "m=0"
    assert rows[0]["boundary_status"] == "resolved_interior"
    assert rows[0]["sampling_status"] == "timing_single_grid_row"


def test_committed_stage2_artifacts_match_canonical_source():
    summary = json.loads(SUMMARY_ARTIFACT.read_text(encoding="utf-8"))
    with REGIONS_ARTIFACT.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    source_digest = hashlib.sha256(CANONICAL_RESULTS.read_bytes()).hexdigest()

    assert summary["source_data_sha256"] == source_digest
    assert summary["source_csv"] == "data/processed/xy8_qre_nv_results.csv"
    assert summary["beta_0"] == len(rows) == 1
    assert summary["calibration_digest"] == "fe2b8d2c6faee1d4"
    assert rows[0]["component_id"] == "R1"
    assert int(rows[0]["N_min"]) == 10
    assert int(rows[0]["N_max"]) == 34


def test_refined_grid_contains_exact_resonances_and_halves_local_step():
    medium = topology.build_resonance_refined_tau_grid(
        200e3,
        0.3125e-6,
        5e-6,
        background_count=21,
        resonance_point_count=101,
    )
    fine = topology.build_resonance_refined_tau_grid(
        200e3,
        0.3125e-6,
        5e-6,
        background_count=41,
        resonance_point_count=201,
    )
    for resonance in (1.25e-6, 3.75e-6):
        assert np.any(np.isclose(medium, resonance, rtol=1e-12, atol=1e-18))
        assert np.any(np.isclose(fine, resonance, rtol=1e-12, atol=1e-18))
        medium_resolution = refinement.local_resonance_resolution(
            topology.validate_surface(medium, [1], np.ones((medium.size, 1))),
            resonance,
        )
        fine_resolution = refinement.local_resonance_resolution(
            topology.validate_surface(fine, [1], np.ones((fine.size, 1))),
            resonance,
        )
        assert fine_resolution["right_step_s"] == pytest.approx(
            medium_resolution["right_step_s"] / 2.0
        )


def test_refined_grid_rejects_even_resonance_brackets():
    with pytest.raises(ValueError, match="odd integer"):
        topology.build_resonance_refined_tau_grid(
            200e3,
            0.3125e-6,
            5e-6,
            resonance_point_count=100,
        )


def test_timing_width_bracket_uses_adjacent_rejected_rows():
    surface = topology.validate_surface(
        np.arange(1, 6, dtype=float) * 1e-6,
        [1],
        np.array([[2.0], [1.05], [1.0], [1.05], [2.0]]),
    )
    result = topology.connected_regions(surface, relative_limit=1.10)
    bracket = refinement.timing_width_bracket(result, 1)
    assert bracket["lower_bound_s"] == pytest.approx(2e-6)
    assert bracket["upper_bound_s"] == pytest.approx(4e-6)
    assert bracket["estimate_s"] == pytest.approx(3e-6)
    assert bracket["half_interval_s"] == pytest.approx(1e-6)


def test_stage3_artifacts_show_stable_resolved_primary_region():
    summary = json.loads(REFINEMENT_SUMMARY_ARTIFACT.read_text(encoding="utf-8"))
    with REFINEMENT_GRID_ARTIFACT.open("r", encoding="utf-8", newline="") as handle:
        grid_rows = {row["grid_id"]: row for row in csv.DictReader(handle)}
    with RESONANCE_ARTIFACT.open("r", encoding="utf-8", newline="") as handle:
        resonance_rows = list(csv.DictReader(handle))

    assert summary["status"] == "grid_refinement_passed"
    assert all(summary["checks"].values())
    assert summary["medium_to_fine_width_relative_change"] < 0.05
    assert set(grid_rows) == {"canonical", "medium", "fine"}
    assert {int(row["beta_0_four_neighbour"]) for row in grid_rows.values()} == {1}
    assert {int(row["beta_0_eight_neighbour"]) for row in grid_rows.values()} == {1}
    assert float(grid_rows["fine"]["primary_tau_width_us"]) == pytest.approx(0.00875)
    assert float(grid_rows["fine"]["primary_tau_width_estimate_us"]) == pytest.approx(0.009)
    assert float(grid_rows["fine"]["primary_tau_width_half_interval_us"]) == pytest.approx(0.00025)
    assert int(grid_rows["fine"]["primary_sampled_tau_count"]) == 36
    assert grid_rows["fine"]["primary_boundary_status"] == "resolved_interior"
    assert grid_rows["fine"]["primary_sampling_status"] == "multiple_grid_points_on_both_axes"

    fine_secondary = next(
        row
        for row in resonance_rows
        if row["grid_id"] == "fine" and int(row["resonance_order"]) == 1
    )
    assert float(fine_secondary["resonance_tau_us"]) == pytest.approx(3.75)
    assert int(fine_secondary["best_N_at_resonance"]) == 12
    assert float(fine_secondary["penalty_percent"]) == pytest.approx(134.34437459237273)


def test_component_tracking_records_birth_and_merge_events():
    surface = topology.validate_surface(
        [1e-6],
        [1, 2, 3, 4, 5],
        [[1.0, 3.0, 3.0, 1.2, 1.2]],
    )
    rows, events, tracks = persistence.track_threshold_components(
        surface,
        target_frequency_hz=250e3,
        threshold_percents=[0, 20, 200],
    )

    assert [event["event_type"] for event in events] == [
        "present_at_first_threshold",
        "birth",
        "merge",
    ]
    assert events[1]["component_id"] == "R2"
    assert events[2]["component_id"] == "R1"
    assert events[2]["related_component_ids"] == "R2"
    assert tracks[0]["survives_through_maximum_threshold"]
    assert tracks[1]["merged_at_threshold_percent"] == 200.0
    assert tracks[1]["merged_into"] == "R1"
    assert [sum(row["threshold_percent"] == value for row in rows) for value in (0, 20, 200)] == [1, 2, 1]


def test_component_tracking_requires_increasing_thresholds():
    surface = make_surface([[1.0]])
    with pytest.raises(ValueError, match="strictly increasing"):
        persistence.track_threshold_components(
            surface,
            target_frequency_hz=200e3,
            threshold_percents=[10, 5],
        )


def test_stage4_artifacts_show_one_expanding_persistent_region():
    summary = json.loads(PERSISTENCE_SUMMARY_ARTIFACT.read_text(encoding="utf-8"))
    with THRESHOLD_CURVE_ARTIFACT.open("r", encoding="utf-8", newline="") as handle:
        curve = list(csv.DictReader(handle))
    with PERSISTENCE_EVENTS_ARTIFACT.open("r", encoding="utf-8", newline="") as handle:
        events = list(csv.DictReader(handle))
    with COMPONENT_TRACKS_ARTIFACT.open("r", encoding="utf-8", newline="") as handle:
        tracks = list(csv.DictReader(handle))

    assert summary["status"] == "threshold_persistence_passed"
    assert all(summary["checks"].values())
    assert len(curve) == 100
    assert {int(row["beta_0_four_neighbour"]) for row in curve} == {1}
    assert {int(row["beta_0_eight_neighbour"]) for row in curve} == {1}
    assert len(events) == len(tracks) == 1
    assert events[0]["event_type"] == "present_at_first_threshold"
    assert tracks[0]["component_id"] == "R1"

    by_threshold = {int(float(row["threshold_percent"])): row for row in curve}
    assert float(by_threshold[10]["primary_timing_width_estimate_us"]) == pytest.approx(0.009)
    assert (int(by_threshold[10]["primary_N_min"]), int(by_threshold[10]["primary_N_max"])) == (10, 34)
    assert float(by_threshold[100]["primary_timing_width_estimate_us"]) == pytest.approx(0.06275)
    assert (int(by_threshold[100]["primary_N_min"]), int(by_threshold[100]["primary_N_max"])) == (3, 64)
    excluded = summary["nearest_resonance_excluded_from_requested_range"]
    assert excluded["resonance_tau_us"] == pytest.approx(3.75)
    assert excluded["penalty_percent"] == pytest.approx(134.34437459237273)


def test_categorical_edges_do_not_interpolate_depth_values():
    edges = topology_figure.categorical_edges(np.array([1.0, 2.0, 4.0]))
    assert edges.tolist() == pytest.approx([0.5, 1.5, 3.0, 5.0])


def test_topology_figure_and_source_artifacts_are_consistent():
    summary = json.loads(TOPOLOGY_FIGURE_SUMMARY.read_text(encoding="utf-8"))
    persistence_summary = json.loads(PERSISTENCE_SUMMARY_ARTIFACT.read_text(encoding="utf-8"))
    with NESTED_REGION_SOURCE.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert summary["status"] == "topology_figure_generated"
    assert summary["persistence_configuration_digest"] == persistence_summary["configuration_digest"]
    assert summary["display_thresholds_percent"] == [1, 5, 10, 20, 50, 100]
    assert summary["nested_source_sha256"] == hashlib.sha256(NESTED_REGION_SOURCE.read_bytes()).hexdigest()
    assert summary["threshold_curve_sha256"] == hashlib.sha256(THRESHOLD_CURVE_ARTIFACT.read_bytes()).hexdigest()
    assert TOPOLOGY_FIGURE_PNG.stat().st_size > 100_000
    assert TOPOLOGY_FIGURE_PDF.stat().st_size > 10_000
    assert len(rows) == summary["crop"]["tau_count"] * summary["crop"]["N_count"]

    best = min(rows, key=lambda row: float(row["eta_proxy"]))
    assert float(best["tau_us"]) == pytest.approx(1.25)
    assert int(best["N"]) == 20
    assert float(best["relative_to_best"]) == pytest.approx(1.0)
    assert int(best["lowest_display_threshold_percent"]) == 1
