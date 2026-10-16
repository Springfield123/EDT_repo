from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "code" / "analysis"
PLOTTING_DIR = ROOT / "code" / "plotting"
for path in [ANALYSIS_DIR, PLOTTING_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import xy8_literature_validation as validation

SCRIPT = PLOTTING_DIR / "make_xy8_literature_validation.py"
SPEC = importlib.util.spec_from_file_location("make_xy8_literature_validation", SCRIPT)
plotting = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(plotting)

MANIFEST_PATH = (
    ROOT / "data" / "raw" / "nv" / "literature_validation" / "pham_2012_manifest.json"
)
DATA_PATH = ROOT / "data" / "raw" / "nv" / "literature_validation" / "pham_2012_fig3b.csv"


def load_inputs():
    return validation.load_manifest(MANIFEST_PATH), validation.load_measurements(DATA_PATH)


class TestXY8LiteratureValidation(unittest.TestCase):
    def test_holdout_is_frozen_and_contains_a_depth_curve(self):
        manifest, measurements = load_inputs()
        self.assertEqual(manifest["selection_status"], "frozen_before_validation_model_evaluation")
        self.assertIn("no parameter is adjusted", manifest["independence_statement"])
        self.assertEqual(len(measurements), 10)
        self.assertEqual([row["total_pi_pulses"] for row in measurements[:2]], [28, 54])
        self.assertTrue(
            all(row["source_figure"] == "Fig. 3b" for row in measurements)
        )

    def test_timing_conversion_matches_the_paper_convention(self):
        frequency = 84.5e3
        tau = validation.project_tau_s(frequency)
        self.assertAlmostEqual(tau, 1.0 / (4.0 * frequency), places=15)
        self.assertAlmostEqual(2.0 * tau, 1.0 / (2.0 * frequency), places=15)
        self.assertAlmostEqual(
            float(validation.sequence_duration_s(160, frequency)),
            160.0 / (2.0 * frequency),
            places=15,
        )

    def test_baseline_assumptions_are_imported_without_optimum_fitting(self):
        manifest, _ = load_inputs()
        nominal = json.loads(
            (ROOT / "data" / "raw" / "nv" / "xy8_scenario_nominal.json").read_text(
                encoding="utf-8"
            )
        )
        baseline = manifest["predeclared_baseline_assumptions"]
        expected_overhead = sum(nominal["overheads"].values())
        self.assertEqual(baseline["per_pulse_fidelity"], nominal["pulse"]["piPulseFid"])
        self.assertAlmostEqual(baseline["preparation_readout_overhead_s"], expected_overhead)
        self.assertIn("Neither pulse fidelity", baseline["no_fit_statement"])

    def test_holdout_result_is_partial_and_quantitatively_stable(self):
        manifest, measurements = load_inputs()
        first = validation.evaluate_holdout(manifest, measurements)
        second = validation.evaluate_holdout(manifest, measurements)
        metrics = first["metrics"]

        self.assertEqual(metrics, second["metrics"])
        self.assertEqual(metrics["assessment"], "partial_trend_validation")
        self.assertEqual(metrics["observed_central_value_optimum_total_pi_pulses"], 106)
        self.assertEqual(metrics["predicted_baseline_optimum_total_pi_pulses"], 86)
        self.assertLessEqual(metrics["predicted_baseline_10pct_window_min_pulses"], 106)
        self.assertGreaterEqual(metrics["predicted_baseline_10pct_window_max_pulses"], 106)
        self.assertTrue(metrics["checks"]["low_depth_improvement_reproduced"])
        self.assertTrue(metrics["checks"]["high_depth_decline_reproduced"])
        self.assertFalse(metrics["checks"]["rank_correlation_at_least_0_7"])
        self.assertAlmostEqual(metrics["spearman_rank_correlation"], 0.5167197121, places=8)
        self.assertAlmostEqual(metrics["normalized_rmse"], 0.1551134617, places=8)

    def test_uncertainty_and_exported_artifacts_are_consistent(self):
        manifest, measurements = load_inputs()
        result = validation.evaluate_holdout(manifest, measurements)
        metrics = result["metrics"]
        self.assertEqual(metrics["uncertainty_case_count"], 41 * 22)
        self.assertEqual(metrics["uncertainty_optimum_min_pulses"], 60)
        self.assertEqual(metrics["uncertainty_optimum_max_pulses"], 152)
        self.assertAlmostEqual(
            metrics["uncertainty_cases_containing_observed_optimum_in_10pct_window_fraction"],
            1.0,
        )

        with tempfile.TemporaryDirectory() as temporary:
            outdir = Path(temporary)
            written = plotting.run(MANIFEST_PATH, DATA_PATH, outdir)
            exported = json.loads(
                (outdir / "xy8_literature_validation_metrics.json").read_text(encoding="utf-8")
            )
            with (outdir / "xy8_literature_validation_curve.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                curve_rows = list(csv.DictReader(handle))
            best_row = min(
                curve_rows,
                key=lambda row: float(row["baseline_model_relative_to_predicted_minimum"]),
            )
            self.assertEqual(written, exported)
            self.assertEqual(int(best_row["total_pi_pulses"]), exported["predicted_baseline_optimum_total_pi_pulses"])
            self.assertTrue((outdir / "fig_xy8_literature_validation.png").exists())
            self.assertTrue((outdir / "fig_xy8_literature_validation.pdf").exists())
            self.assertEqual(
                exported["input_digests"]["measurement_csv_sha256"],
                validation.file_sha256(DATA_PATH),
            )


if __name__ == "__main__":
    unittest.main()
