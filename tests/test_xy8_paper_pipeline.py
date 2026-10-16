from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PLOTTING_SCRIPT = ROOT / "code" / "plotting" / "make_fig_xy8_qre.py"
EXTENDED_SCRIPT = ROOT / "code" / "plotting" / "make_xy8_extended_analyses.py"
PROSPECTIVE_SCRIPT = ROOT / "code" / "plotting" / "make_xy8_prospective_validation.py"
ANALYSIS_DIR = ROOT / "code" / "analysis"
PLOTTING_DIR = ROOT / "code" / "plotting"
for path in [ANALYSIS_DIR, PLOTTING_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

spec = importlib.util.spec_from_file_location("paper_xy8", PLOTTING_SCRIPT)
paper_xy8 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(paper_xy8)
pipe = sys.modules["nvratemodel_api"]

extended_spec = importlib.util.spec_from_file_location("extended_xy8", EXTENDED_SCRIPT)
extended_xy8 = importlib.util.module_from_spec(extended_spec)
assert extended_spec.loader is not None
extended_spec.loader.exec_module(extended_xy8)

prospective_spec = importlib.util.spec_from_file_location("prospective_xy8", PROSPECTIVE_SCRIPT)
prospective_xy8 = importlib.util.module_from_spec(prospective_spec)
assert prospective_spec.loader is not None
prospective_spec.loader.exec_module(prospective_xy8)

import xy8_control_recommender as recommender


def load_scenario(label: str) -> dict:
    path = ROOT / "data" / "raw" / "nv" / f"xy8_scenario_{label}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def read_result_csv(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    table_start = lines.index("tau_s,N,eta_proxy")
    metadata = {}
    for line in lines[:table_start]:
        if line and "," in line:
            key, value = line.split(",", 1)
            metadata[key] = value.rstrip("\r")
    rows = list(csv.DictReader(lines[table_start:]))
    return metadata, rows


def test_200khz_grid_contains_exact_primary_and_third_harmonic_resonances():
    taus, _ = paper_xy8.build_sweep_grid(load_scenario("nominal"), 200e3)
    assert np.any(np.isclose(taus, 1.25e-6, rtol=0.0, atol=1e-15))
    assert np.any(np.isclose(taus, 3.75e-6, rtol=0.0, atol=1e-15))


def test_tau_grid_refinement_does_not_move_the_nominal_optimum():
    coarse = load_scenario("nominal")
    coarse["sweep"] = {"tau_count": 11, "resonance_bracket_count": 5}
    refined = deepcopy(coarse)
    refined["sweep"] = {"tau_count": 71, "resonance_bracket_count": 21}

    _, coarse_meta, _, _, _, _ = paper_xy8.evaluate_calibration(coarse, f_target=200e3)
    _, refined_meta, _, _, _, _ = paper_xy8.evaluate_calibration(refined, f_target=200e3)

    assert coarse_meta["tau_best"] == pytest.approx(1.25e-6, abs=1e-15)
    assert refined_meta["tau_best"] == pytest.approx(coarse_meta["tau_best"], abs=1e-15)
    assert refined_meta["N_best"] == coarse_meta["N_best"]
    assert refined_meta["eta_best"] == pytest.approx(coarse_meta["eta_best"], rel=1e-12)


def test_nominal_ten_percent_edt_window_contains_the_optimum_and_is_interior():
    _, meta, _, _, _, _ = paper_xy8.evaluate_calibration(load_scenario("nominal"), f_target=200e3)
    window = meta["edt_window_10pct"]

    assert window["relative_limit"] == pytest.approx(1.10)
    assert window["eta_limit"] == pytest.approx(1.10 * meta["eta_best"])
    assert window["tau_min_s"] <= meta["tau_best"] <= window["tau_max_s"]
    assert window["N_min"] <= meta["N_best"] <= window["N_max"]
    assert window["N_min"] == 10
    assert window["N_max"] == 34
    assert not window["touches_tau_boundary"]
    assert not window["touches_N_boundary"]

    # An edge is only resolved if the grid samples the depth just outside it.
    # Without this the reported edge is wherever the grid happens to stop.
    _, Ns = paper_xy8.build_sweep_grid(load_scenario("nominal"), 200e3)
    sampled = sorted(int(value) for value in Ns)
    assert window["N_min"] - 1 in sampled
    assert window["N_max"] + 1 in sampled


def test_N_is_xy8_block_count_with_eight_pi_pulses_per_block():
    tau = 1.25e-6
    N = 8
    assert pipe.sequence_duration(tau, N) == pytest.approx(16.0 * N * tau)
    assert len(pipe.xy8_pulse_times(tau, N)) == 8 * N

    scenario = load_scenario("conservative")
    _, meta, _, _, _, _ = paper_xy8.evaluate_calibration(scenario, f_target=200e3)
    expected = scenario["pulse"]["piPulseFid"] ** (8 * meta["N_best"])
    assert meta["best_breakdown"]["pulse_factor"] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("label", "expected_N"),
    [("conservative", 8), ("nominal", 20), ("optimistic", 64)],
)
def test_scenario_fixtures_reproduce_refined_optimal_depths(label, expected_N):
    _, meta, _, _, _, _ = paper_xy8.evaluate_calibration(load_scenario(label), f_target=200e3)
    assert meta["tau_best"] == pytest.approx(1.25e-6, abs=1e-15)
    assert meta["N_best"] == expected_N


def test_frequency_sweep_tracks_the_primary_resonance():
    frequencies = np.array([100e3, 200e3, 400e3])
    rows = extended_xy8.run_frequency_sweep(load_scenario("nominal"), frequencies)

    assert [row["f_target_Hz"] for row in rows] == list(frequencies)
    for row in rows:
        assert row["tau_best_s"] == pytest.approx(row["tau_theory_s"], abs=1e-15)
        assert row["tau_ratio_to_primary_resonance"] == pytest.approx(1.0)
        assert not row["edt_window_touches_boundary"]


def test_pareto_frontier_identifies_efficient_and_dominated_depths():
    rows = extended_xy8.run_pareto_analysis(load_scenario("nominal"), 200e3)
    by_N = {row["N"]: row for row in rows}
    pareto_depths = [row["N"] for row in rows if row["pareto_optimal"]]

    assert max(pareto_depths) == 20
    assert by_N[10]["pareto_optimal"] and by_N[10]["within_10pct"]
    assert by_N[10]["pulse_savings_vs_best_pct"] == pytest.approx(50.0)
    assert by_N[16]["pareto_optimal"] and by_N[16]["within_10pct"]
    assert by_N[16]["pulse_savings_vs_best_pct"] == pytest.approx(20.0)
    assert by_N[16]["sensitivity_penalty_pct"] == pytest.approx(0.9727549237354669)
    assert by_N[20]["is_global_best"]
    assert not by_N[32]["pareto_optimal"]
    assert by_N[32]["eta_proxy"] > by_N[20]["eta_proxy"]


def test_nominal_hardware_recommendation_matches_paper_optimum_and_window():
    decision = recommender.recommend_control(
        load_scenario("nominal"),
        frequency_hz=200e3,
        pulse_fidelity=0.997,
        t2_s=300e-6,
    )
    recommendation = decision["recommendation"]

    assert recommendation["tau_s"] == pytest.approx(1.25e-6, abs=1e-15)
    assert recommendation["resonant_tau_s"] == pytest.approx(1.25e-6, abs=1e-15)
    assert recommendation["N"] == 20
    assert recommendation["total_pi_pulses"] == 160
    assert recommendation["edt_window_10pct"]["N_min"] == 10
    assert recommendation["edt_window_10pct"]["N_max"] == 34
    assert recommendation["pareto_alternative"]["N"] == 16
    assert recommendation["pareto_alternative"]["penalty_percent"] == pytest.approx(0.9727549237354749)
    assert recommendation["boundary_status"] == "resolved interior optimum"
    assert recommendation["N"] in recommendation["edt_window_10pct"]["N_values"]


def test_frequency_input_updates_recommended_resonant_spacing():
    nominal = load_scenario("nominal")
    for frequency in [100e3, 250e3, 400e3]:
        decision = recommender.recommend_control(
            nominal,
            frequency_hz=frequency,
            pulse_fidelity=0.997,
            t2_s=300e-6,
        )
        assert decision["recommendation"]["tau_s"] == pytest.approx(1.0 / (4.0 * frequency), abs=1e-15)


def test_boundary_hardware_recommendation_emits_guardrail_warnings():
    decision = recommender.recommend_control(
        load_scenario("nominal"),
        frequency_hz=200e3,
        pulse_fidelity=1.0,
        t2_s=187.5e-6,
    )
    warning_codes = {warning["code"] for warning in decision["warnings"]}

    assert decision["recommendation"]["N"] == 128
    assert decision["recommendation"]["boundary_status"] == "unresolved boundary optimum"
    assert "optimum_at_sweep_boundary" in warning_codes
    assert "window_at_sweep_boundary" in warning_codes

    extrapolated = recommender.recommend_control(
        load_scenario("nominal"),
        frequency_hz=200e3,
        pulse_fidelity=0.991,
        t2_s=120e-6,
    )
    extrapolated_codes = {warning["code"] for warning in extrapolated["warnings"]}
    assert "outside_phase_diagram_range" in extrapolated_codes
    assert "nominal_resources_assumed" in extrapolated_codes


def test_uncertain_hardware_recommendation_is_seeded_and_reports_robust_choice():
    kwargs = {
        "base_calibration": load_scenario("nominal"),
        "frequency_hz": 200e3,
        "pulse_fidelity": 0.997,
        "t2_s": 300e-6,
        "pulse_fidelity_sigma": 0.0005,
        "t2_sigma_s": 30e-6,
        "uncertainty_samples": 40,
        "seed": 1234,
    }
    first = recommender.recommend_control(**kwargs)
    second = recommender.recommend_control(**kwargs)

    assert first == second
    assert first["uncertainty"]["modal_N"] in {row["N"] for row in first["uncertainty"]["optimal_depth_distribution"]}
    assert first["uncertainty"]["robust_p95_N"] in first["recommendation"]["edt_window_10pct"]["N_values"]
    warning_codes = {warning["code"] for warning in first["warnings"]}
    assert "multimodal_depth_recommendation" in warning_codes
    assert "uncertainty_spans_control_regions" in warning_codes
    assert "additional_calibration_recommended" in warning_codes


def test_phase_grid_rows_reproduce_direct_recommender_depths():
    phase_path = ROOT / "outputs" / "reproduced_figures" / "xy8_uncertainty_phase_diagram.csv"
    nominal = load_scenario("nominal")
    with phase_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 17 * 13
    for row in rows:
        calibration = recommender.calibration_with_hardware(
            nominal, float(row["piPulseFid"]), float(row["T2_1_s"])
        )
        tau, Ns, eta = recommender.evaluate_resonant_depths(calibration, 200e3)
        index = int(np.argmin(eta))
        expected_boundary = int(int(Ns[index]) in {int(Ns[0]), int(Ns[-1])})
        assert float(row["tau_best_s"]) == pytest.approx(tau, abs=1e-15)
        assert int(row["N_best"]) == int(Ns[index])
        assert float(row["eta_best"]) == pytest.approx(float(eta[index]), rel=1e-12)
        assert int(row["best_at_N_boundary"]) == expected_boundary
        assert row["control_region"] == recommender.control_region(int(Ns[index]), bool(expected_boundary))


def test_cli_json_and_phase_figure_annotation_share_the_same_recommendation(tmp_path):
    json_path = tmp_path / "recommendation.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "code" / "recommend_xy8_control.py"),
            "--frequency-khz",
            "200",
            "--pulse-fidelity",
            "0.997",
            "--t2-us",
            "300",
            "--json-out",
            str(json_path),
            "--quiet",
        ],
        check=True,
        cwd=ROOT,
    )
    cli_payload = json.loads(json_path.read_text(encoding="utf-8"))
    generated_payload = json.loads(
        (ROOT / "outputs" / "reproduced_figures" / "xy8_control_recommendation.json").read_text(encoding="utf-8")
    )
    robustness = json.loads(
        (ROOT / "outputs" / "reproduced_figures" / "xy8_robustness_summary.json").read_text(encoding="utf-8")
    )
    annotation = robustness["figure_annotation"]

    assert cli_payload == generated_payload == robustness["selected_recommendation"]
    assert annotation["recommendation_digest"] == cli_payload["recommendation_digest"]
    assert annotation["N"] == cli_payload["recommendation"]["N"]
    assert annotation["tau_s"] == cli_payload["recommendation"]["tau_s"]
    assert annotation["edt_window_N_min"] == cli_payload["recommendation"]["edt_window_10pct"]["N_min"]
    assert annotation["edt_window_N_max"] == cli_payload["recommendation"]["edt_window_10pct"]["N_max"]
    assert annotation["pareto_alternative_N"] == cli_payload["recommendation"]["pareto_alternative"]["N"]


def test_monte_carlo_sampling_is_reproducible_and_robust_control_is_not_worse():
    nominal = load_scenario("nominal")
    first_rows, first_surfaces, taus, Ns = extended_xy8.run_monte_carlo(
        nominal, 200e3, sample_count=12, seed=1234
    )
    second_rows, second_surfaces, _, _ = extended_xy8.run_monte_carlo(
        nominal, 200e3, sample_count=12, seed=1234
    )

    assert first_rows == second_rows
    assert np.array_equal(first_surfaces, second_surfaces)
    robust_rows, _ = extended_xy8.select_robust_controls(nominal, 200e3, first_surfaces, taus, Ns)
    by_label = {row["control"]: row for row in robust_rows}
    assert by_label["Robust p95 optimum"]["p95_regret"] <= (
        by_label["Nominal optimum"]["p95_regret"] + 1e-12
    )


def test_alternative_sequence_baseline_compares_equal_total_pulse_counts():
    rows = extended_xy8.run_sequence_comparison(load_scenario("nominal"), 200e3)
    by_pulses = {}
    for row in rows:
        by_pulses.setdefault(row["total_pi_pulses"], []).append(row)
        assert row["block_count"] * row["pulses_per_block"] == row["total_pi_pulses"]
        assert "phase cycling unresolved" in row["model_scope"]

    for family_rows in by_pulses.values():
        assert {row["sequence"] for row in family_rows} == {"XY8", "XY4", "CPMG"}
        assert np.ptp([row["eta_proxy"] for row in family_rows]) == pytest.approx(0.0)


def test_prospective_validation_rows_are_canonical_deterministic_and_unmeasured():
    nominal = load_scenario("nominal")
    rows, canonical_results, meta = prospective_xy8.build_protocol_rows(
        nominal, f_target=200e3, uncertainty_samples=8, seed=1234
    )
    values = {(float(tau), int(N)): float(eta) for tau, N, eta in canonical_results}
    by_id = {row["control_id"]: row for row in rows}

    assert len(rows) == 7
    assert by_id["predicted_optimum"]["N"] == 20
    assert by_id["predicted_optimum"]["predicted_rank"] == 1
    assert by_id["practical_choice"]["predicted_rank"] == 2
    assert by_id["off_resonance"]["predicted_rank"] == 7
    assert by_id["off_resonance"]["predicted_relative_to_best"] >= 5.0
    assert abs(by_id["off_resonance"]["tau_s"] - 1.25e-6) == pytest.approx(0.025e-6)

    for row in rows:
        assert row["predicted_eta_proxy"] == pytest.approx(values[(row["tau_s"], row["N"])])
        assert row["total_pi_pulses"] == 8 * row["N"]
        assert row["sequence_duration_s"] == pytest.approx(pipe.sequence_duration(row["tau_s"], row["N"]))
        assert row["calibration_digest"] == meta["calibration_digest"]
        assert row["predicted_eta_95pct_lower"] <= row["predicted_eta_median"]
        assert row["predicted_eta_median"] <= row["predicted_eta_95pct_upper"]
        assert row["measured_eta_proxy"] == ""
        assert row["criterion_status"] == ""


def test_generated_prospective_protocol_matches_main_figure_configuration():
    breakdown_path = ROOT / "outputs" / "reproduced_figures" / "xy8_qre_nv_breakdown.json"
    csv_path = ROOT / "outputs" / "reproduced_figures" / "xy8_prospective_validation.csv"
    protocol_path = ROOT / "outputs" / "reproduced_figures" / "xy8_prospective_validation_protocol.json"

    breakdown = json.loads(breakdown_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 7
    assert protocol["status"] == "prospective_predictions_unmeasured"
    assert protocol["calibration_digest"] == breakdown["configuration"]["calibration_digest"]
    assert protocol["f_target_Hz"] == breakdown["configuration"]["f_target_Hz"]
    assert all(row["calibration_digest"] == protocol["calibration_digest"] for row in rows)
    assert all(row["protocol_digest"] == protocol["protocol_digest"] for row in rows)
    assert all(row["measured_eta_proxy"] == "" and row["criterion_status"] == "" for row in rows)


def test_generated_csv_json_and_fig5_source_data_share_configuration_and_optimum():
    result_path = ROOT / "outputs" / "reproduced_figures" / "xy8_qre_nv_results.csv"
    breakdown_path = ROOT / "outputs" / "reproduced_figures" / "xy8_qre_nv_breakdown.json"
    source_path = ROOT / "data" / "figure_source_data" / "Fig5_source_data.csv"
    window_path = ROOT / "outputs" / "reproduced_figures" / "xy8_edt_window_10pct.csv"

    metadata, result_rows = read_result_csv(result_path)
    breakdown = json.loads(breakdown_path.read_text(encoding="utf-8"))
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    with window_path.open("r", encoding="utf-8", newline="") as handle:
        window_rows = list(csv.DictReader(handle))

    assert metadata["scenario"] == breakdown["configuration"]["scenario"] == "nominal"
    assert metadata["calibration_digest"] == breakdown["configuration"]["calibration_digest"]
    assert float(metadata["f_target_Hz"]) == breakdown["configuration"]["f_target_Hz"]
    assert float(metadata["tau_best_s"]) == pytest.approx(breakdown["best"]["tau_s"])
    assert int(metadata["N_best"]) == breakdown["best"]["N"]
    assert float(metadata["eta_best"]) == pytest.approx(breakdown["best"]["eta_proxy"])

    assert len(source_rows) == len(result_rows)
    assert source_rows[0]["scenario"] == metadata["scenario"]
    assert source_rows[0]["calibration_digest"] == metadata["calibration_digest"]
    assert float(source_rows[0]["tau_best_s"]) == pytest.approx(float(metadata["tau_best_s"]))
    assert int(source_rows[0]["N_best"]) == int(metadata["N_best"])
    assert breakdown["edt_window_10pct"]["point_count"] == len(window_rows)
    assert all(float(row["eta_proxy"]) <= breakdown["edt_window_10pct"]["eta_limit"] for row in window_rows)
    assert any(
        float(row["tau_s"]) == pytest.approx(breakdown["best"]["tau_s"])
        and int(row["N"]) == breakdown["best"]["N"]
        for row in window_rows
    )
