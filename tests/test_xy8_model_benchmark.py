from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "code" / "analysis"
PLOTTING_DIR = ROOT / "code" / "plotting"
for path in [ANALYSIS_DIR, PLOTTING_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import xy8_literature_validation as validation

SCRIPT = PLOTTING_DIR / "make_xy8_model_benchmark.py"
SPEC = importlib.util.spec_from_file_location("make_xy8_model_benchmark", SCRIPT)
benchmark_script = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark_script)

MANIFEST_PATH = (
    ROOT / "data" / "raw" / "nv" / "literature_validation" / "pham_2012_manifest.json"
)
DATA_PATH = ROOT / "data" / "raw" / "nv" / "literature_validation" / "pham_2012_fig3b.csv"


def evaluate():
    manifest = validation.load_manifest(MANIFEST_PATH)
    measurements = validation.load_measurements(DATA_PATH)
    return validation.benchmark_model_alternatives(manifest, measurements)


def test_nested_models_recover_different_optima_and_windows():
    result = evaluate()
    by_id = {row["method_id"]: row for row in result["rows"]}

    coherence = by_id["coherence_only"]
    pulse = by_id["coherence_plus_pulse_fidelity"]
    full = by_id["full_resource_score"]
    assert coherence["predicted_optimum_total_pi_pulses"] == 264
    assert pulse["predicted_optimum_total_pi_pulses"] == 84
    assert full["predicted_optimum_total_pi_pulses"] == 86
    assert not coherence["observed_optimum_inside_10pct_window"]
    assert pulse["observed_optimum_inside_10pct_window"]
    assert full["observed_optimum_inside_10pct_window"]


def test_all_requested_curve_metrics_are_reported_without_overclaiming():
    result = evaluate()
    by_id = {row["method_id"]: row for row in result["rows"]}
    coherence = by_id["coherence_only"]
    pulse = by_id["coherence_plus_pulse_fidelity"]
    full = by_id["full_resource_score"]

    assert coherence["absolute_error_vs_observed_optimum_pulses"] == 158
    assert pulse["absolute_error_vs_observed_optimum_pulses"] == 22
    assert full["absolute_error_vs_observed_optimum_pulses"] == 20
    assert coherence["normalized_rmse"] == pytest.approx(0.14213486245877863)
    assert full["normalized_rmse"] == pytest.approx(0.1551134616599272)
    assert coherence["spearman_rank_correlation"] == pytest.approx(0.024316221747202587)
    assert full["spearman_rank_correlation"] == pytest.approx(0.516719712128055)
    assert coherence["normalized_rmse"] < full["normalized_rmse"]


def test_fixed_N16_is_128_pulses_and_curve_metrics_are_not_fabricated():
    result = evaluate()
    fixed = next(row for row in result["rows"] if row["method_id"] == "fixed_xy8_N16")
    assert fixed["predicted_optimum_total_pi_pulses"] == 8 * 16
    assert fixed["absolute_error_vs_observed_optimum_pulses"] == 22
    assert fixed["normalized_rmse"] == ""
    assert fixed["spearman_rank_correlation"] == ""
    assert fixed["observed_optimum_inside_10pct_window"] == ""
    assert fixed["fixed_choice_experimental_penalty_pct"] == pytest.approx(0.5530417295123158)


def test_summary_identifies_mixed_evidence_and_excludes_reference_from_model_ranking():
    summary = evaluate()["summary"]
    assert summary["observed_optimum_total_pi_pulses"] == 106
    assert summary["observed_empirical_10pct_window_min_pulses"] == 80
    assert summary["observed_empirical_10pct_window_max_pulses"] == 212
    assert summary["best_predictive_optimum_method"] == "Full resource score"
    assert summary["lowest_normalized_rmse_method"] == "Coherence only"
    assert "tie" in summary["highest_rank_correlation_method"]
    assert "does not yet establish a large advantage" in summary["interpretation"]


def test_exported_csv_json_and_figure_share_the_same_result():
    with tempfile.TemporaryDirectory() as temporary:
        outdir = Path(temporary)
        summary = benchmark_script.run(MANIFEST_PATH, DATA_PATH, outdir)
        exported_summary = json.loads(
            (outdir / "xy8_literature_model_benchmark_summary.json").read_text(
                encoding="utf-8"
            )
        )
        with (outdir / "xy8_literature_model_benchmark.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        full = next(row for row in rows if row["method_id"] == "full_resource_score")

        assert summary == exported_summary
        assert int(full["predicted_optimum_total_pi_pulses"]) == 86
        assert int(full["absolute_error_vs_observed_optimum_pulses"]) == 20
        assert (outdir / "fig_xy8_literature_model_benchmark.png").exists()
        assert (outdir / "fig_xy8_literature_model_benchmark.pdf").exists()
        assert exported_summary["input_digests"]["measurement_csv_sha256"] == validation.file_sha256(DATA_PATH)
