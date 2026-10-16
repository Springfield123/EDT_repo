from __future__ import annotations

from copy import deepcopy
import csv
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "code" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import xy8_prospective_power as power


PROTOCOL_PATH = ROOT / "outputs/reproduced_figures/xy8_prospective_validation_protocol.json"
SUMMARY_PATH = ROOT / "outputs/reproduced_figures/xy8_prospective_power_summary.json"
CURVE_PATH = ROOT / "outputs/reproduced_figures/xy8_prospective_power_curve.csv"


def protocol_payload() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_protocol_is_unmeasured_and_window_edges_are_canonical():
    protocol = power.validate_protocol(protocol_payload())
    assert (protocol.window_min, protocol.window_max) == (10, 34)
    assert protocol.by_id["predicted_optimum"]["N"] == 20
    assert all(
        row["measured_eta_proxy"] == "" for row in protocol.controls
    )


def test_power_simulation_is_seeded_and_timing_control_is_easy_to_detect():
    protocol = power.load_protocol(PROTOCOL_PATH)
    first = power.simulate_decision_power(protocol, 3, trials=5000, seed=17)
    second = power.simulate_decision_power(protocol, 3, trials=5000, seed=17)
    assert first == second
    assert first["timing_selectivity"]["power"] > 0.94


def test_tiny_n16_n20_difference_requires_more_data_than_window_membership():
    protocol = power.load_protocol(PROTOCOL_PATH)
    low_precision = power.simulate_decision_power(
        protocol, 25, trials=10000, seed=21
    )
    high_precision = power.simulate_decision_power(
        protocol, 25000, trials=10000, seed=22
    )
    assert low_precision["operating_window"]["power"] > 0.90
    assert low_precision["n20_beats_n16"]["power"] < 0.60
    assert high_precision["n20_beats_n16"]["power"] > 0.90


def test_equivalence_test_reaches_power_with_sufficient_batches():
    protocol = power.load_protocol(PROTOCOL_PATH)
    result = power.simulate_decision_power(
        protocol, 1600, trials=10000, seed=23
    )
    assert result["n16_n20_equivalent_10pct"]["power"] > 0.90


def test_power_curve_reports_required_counts_and_times():
    protocol = power.load_protocol(PROTOCOL_PATH)
    _, summary = power.build_power_curve(
        protocol,
        batch_grid=[3, 25, 160, 1600, 25000],
        trials=10000,
        seed=29,
    )
    required = summary["required_for_target_power"]
    assert required["operating_window"]["required_independent_batches_per_control"] == 25
    assert required["exact_optimum"]["required_independent_batches_per_control"] == 25000
    assert required["n16_n20_equivalent_10pct"]["required_independent_batches_per_control"] == 1600
    assert (
        required["operating_window"]["metric_specific_acquisition_time_h"]
        < required["exact_optimum"]["metric_specific_acquisition_time_h"]
    )
    designs = summary["recommended_designs"]
    assert (
        designs["core_window_and_boundary_test"]["power"]["depth_controls"]["power"]
        > 0.90
    )
    assert (
        designs["ten_percent_equivalence_add_on"]["power"]
        ["n16_n20_equivalent_10pct"]["power"]
        > 0.90
    )


def test_measured_protocol_and_invalid_allocations_are_rejected():
    payload = deepcopy(protocol_payload())
    payload["controls"][0]["measured_eta_proxy"] = 1.0
    with pytest.raises(ValueError, match="before measurements"):
        power.validate_protocol(payload)
    protocol = power.load_protocol(PROTOCOL_PATH)
    with pytest.raises(ValueError, match="At least three"):
        power.simulate_decision_power(protocol, 2)


def test_generated_power_artifacts_match_the_frozen_protocol():
    protocol = power.load_protocol(PROTOCOL_PATH)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    with CURVE_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert summary["status"] == "prospective_power_analysis_no_measurements"
    assert summary["protocol_digest"] == protocol.protocol_digest
    assert summary["calibration_digest"] == protocol.calibration_digest
    assert rows
    assert {row["metric_id"] for row in rows} == set(power.METRIC_DESCRIPTIONS)
    required = summary["required_for_target_power"]
    assert (
        required["operating_window"]["required_independent_batches_per_control"]
        < required["exact_optimum"]["required_independent_batches_per_control"]
    )
