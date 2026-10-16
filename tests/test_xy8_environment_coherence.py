from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
ANALYSIS_DIR = CODE_DIR / "analysis"
for path in (CODE_DIR, ANALYSIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import analyze_xy8_environment_coherence as exporter
import xy8_environment_coherence as coherence
import xy8_environment_scan_config as scan_config
import nvratemodel_api as model


@pytest.mark.parametrize("variant", coherence.SUPPORTED_VARIANTS)
def test_anchoring_exactly_preserves_nominal_t2_at_300K(variant):
    result = coherence.anchored_coherence(300.0, variant=variant)
    assert result.effective_T2_s == pytest.approx(300e-6, rel=0.0, abs=1e-15)
    assert result.effective_T2_uncertainty_1sigma_s == pytest.approx(0.0, abs=1e-18)
    assert result.residual_dephasing_rate_s_inv > 0.0
    assert result.effective_T2_s <= 2.0 * result.t1_s


def test_jarmola_rate_matches_frozen_law_and_propagates_parameter_uncertainty():
    result = coherence.jarmola_t1_rate(300.0)
    assert result.rate_s_inv == pytest.approx(186.09, abs=0.1)
    assert result.t1_s * 1e3 == pytest.approx(5.374, abs=0.01)
    assert result.uncertainty_1sigma_s_inv > 0.0
    assert result.t1_uncertainty_1sigma_s > 0.0
    assert "reported A1, A2, delta, and A3" in result.uncertainty_scope


def test_mondal_log_interpolation_propagates_both_coordinate_uncertainties():
    result = coherence.mondal_t1_rate(312.5)
    assert 120.0 < result.rate_s_inv < 148.0
    assert result.uncertainty_1sigma_s_inv > 0.0
    assert result.t1_uncertainty_1sigma_s > 0.0
    assert "coordinate-extraction" in result.uncertainty_scope


def test_source_ranges_are_enforced_and_extrapolation_is_labelled():
    with pytest.raises(ValueError, match="extrapolation is disabled"):
        coherence.mondal_t1_rate(99.0)
    allowed = coherence.mondal_t1_rate(99.0, allow_extrapolation=True)
    assert any("outside_mondal_source_range" in warning for warning in allowed.warnings)
    with pytest.raises(ValueError, match="extrapolation is disabled"):
        coherence.jarmola_t1_rate(500.0)


def test_anchored_jarmola_temperature_change_cancels_sample_constant_A1():
    laws = coherence.literature.load_temperature_laws()
    config = scan_config.load_config()
    reference_t2 = config["relaxation_policy"]["reference_T2_s"]
    values = []
    for sample in ("S2", "S8", "S3"):
        current = coherence.jarmola_t1_rate(400.0, sample=sample, laws=laws)
        reference = coherence.jarmola_t1_rate(300.0, sample=sample, laws=laws)
        values.append(1.0 / (1.0 / reference_t2 + 0.5 * (current.rate_s_inv - reference.rate_s_inv)))
    assert values == pytest.approx([values[0]] * 3, rel=1e-12)


@pytest.mark.parametrize("variant", coherence.SUPPORTED_VARIANTS)
def test_temperature_conditioned_parameters_change_only_t2_1_and_metadata(variant):
    base = {"T2_1": 300e-6, "alpha": 0.6, "beta": 0.0, "tau0": 1e-6, "p": 2.0}
    conditioned, result = coherence.temperature_conditioned_t2_params(
        base, 350.0, variant=variant
    )
    assert conditioned["T2_1"] == result.effective_T2_s
    assert conditioned["T1_limit_s"] == result.t1_s
    for key in ("alpha", "beta", "tau0", "p"):
        assert conditioned[key] == base[key]
    assert base == {"T2_1": 300e-6, "alpha": 0.6, "beta": 0.0, "tau0": 1e-6, "p": 2.0}
    assert conditioned["environment_coherence_variant"] == variant


@pytest.mark.parametrize("variant", coherence.SUPPORTED_VARIANTS)
def test_depth_scaled_coherence_obeys_t1_bound_in_helper_and_score_model(variant):
    base = {"T2_1": 300e-6, "alpha": 0.6, "beta": 0.0, "tau0": 1e-6, "p": 2.0}
    for temperature in (100.0, 300.0, 400.0):
        params, anchored = coherence.temperature_conditioned_t2_params(
            base, temperature, variant=variant
        )
        for blocks in (1, 20, 64, 128):
            helper_t2, _, _ = coherence.sequence_coherence_time_s(
                blocks, 1.25e-6, base, temperature, variant=variant
            )
            model_t2 = model.t2_xy8_empirical(blocks, 1.25e-6, **params)
            assert helper_t2 == pytest.approx(model_t2)
            assert model_t2 <= 2.0 * anchored.t1_s


def test_t1_bound_is_opt_in_and_preserves_historical_coherence_behavior():
    historical = model.t2_xy8_empirical(20, 1.25e-6, T2_1=300e-6, alpha=0.6)
    assert historical == pytest.approx(300e-6 * 20**0.6)
    assert model.t2_xy8_empirical(
        20, 1.25e-6, T2_1=300e-6, alpha=0.6, T1_limit_s=200e-6
    ) == pytest.approx(400e-6)
    with pytest.raises(ValueError, match="T1_limit_s"):
        model.t2_xy8_empirical(20, 1.25e-6, T1_limit_s=0.0)


@pytest.mark.parametrize("variant", coherence.SUPPORTED_VARIANTS)
def test_t2_physical_bound_holds_across_frozen_temperature_range(variant):
    for temperature in range(100, 401, 5):
        result = coherence.anchored_coherence(float(temperature), variant=variant)
        assert result.effective_T2_s > 0.0
        assert result.effective_T2_s <= 2.0 * result.t1_s


def test_variants_remain_separate_and_unknown_variant_is_rejected():
    experimental = coherence.anchored_coherence(400.0, variant=coherence.JARMOLA_VARIANT)
    first_principles = coherence.anchored_coherence(400.0, variant=coherence.MONDAL_VARIANT)
    assert experimental.variant != first_principles.variant
    assert experimental.effective_T2_s != pytest.approx(first_principles.effective_T2_s)
    with pytest.raises(ValueError, match="Unknown T1 variant"):
        coherence.anchored_coherence(300.0, variant="combined")


def test_stage3_export_contains_no_edt_scores_and_is_deterministic(tmp_path):
    csv_path, json_path, summary = exporter.write_outputs(tmp_path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    stored = json.loads(json_path.read_text(encoding="utf-8"))
    assert stored == summary
    assert summary["edt_score_surfaces_evaluated"] is False
    assert summary["diagnostic_row_count"] == len(rows) == 44
    assert set(summary["variants"]) == set(coherence.SUPPORTED_VARIANTS)
    assert all(values["reference_T2_1_us"] == pytest.approx(300.0) for values in summary["variants"].values())
    assert all(values["warning_row_count"] == 0 for values in summary["variants"].values())
    assert all(values["all_N128_T2_le_2T1"] for values in summary["variants"].values())
    assert "eta" not in rows[0]


def test_committed_stage3_artifacts_match_the_frozen_configuration():
    csv_path = ROOT / "data" / "processed" / exporter.CSV_NAME
    json_path = ROOT / "data" / "processed" / exporter.JSON_NAME
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    freeze = scan_config.verify_freeze_record()
    assert summary["configuration_sha256"] == freeze["configuration_sha256"]
    assert summary["diagnostic_row_count"] == len(rows) == 44
    assert summary["edt_score_surfaces_evaluated"] is False
