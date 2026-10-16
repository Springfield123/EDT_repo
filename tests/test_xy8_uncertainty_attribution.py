from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "code" / "analysis"
PLOTTING_DIR = ROOT / "code" / "plotting"
for path in [ANALYSIS_DIR, PLOTTING_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

SCRIPT = PLOTTING_DIR / "make_xy8_uncertainty_attribution.py"
SPEC = importlib.util.spec_from_file_location("xy8_uncertainty_attribution", SCRIPT)
attribution = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(attribution)


def load_nominal():
    return json.loads(
        (ROOT / "data" / "raw" / "nv" / "xy8_scenario_nominal.json").read_text(
            encoding="utf-8"
        )
    )


class TestXY8UncertaintyAttribution(unittest.TestCase):
    def test_existing_samples_produce_all_four_decision_outcomes(self):
        samples = attribution.load_sample_rows(
            ROOT / "outputs" / "reproduced_figures" / "xy8_monte_carlo_samples.csv"
        )[:8]
        rows, surfaces, taus, Ns, nominal_meta, _ = attribution.evaluate_existing_samples(
            load_nominal(), samples, 200e3
        )

        self.assertEqual(len(rows), 8)
        self.assertEqual(surfaces.shape[0], 8)
        self.assertEqual(nominal_meta["N_best"], 20)
        self.assertTrue(np.any(np.isclose(taus, 1.25e-6, rtol=0.0, atol=1e-15)))
        self.assertIn(20, Ns)
        for row in rows:
            self.assertGreaterEqual(row["fixed_N20_penalty_pct"], -1e-10)
            self.assertGreaterEqual(row["edt_window_width_blocks"], 0)
            self.assertIn(row["edt_window_touches_boundary"], {0, 1})

    def test_rank_correlation_bootstrap_is_seeded(self):
        rows = []
        for index in range(30):
            row = {column: 1.0 + 0.01 * index for column in attribution.INPUT_COLUMNS}
            row.update(
                {
                    "N_best": index,
                    "fixed_N20_penalty_pct": 30 - index,
                    "edt_window_width_blocks": index // 2,
                    "edt_window_touches_boundary": int(index > 24),
                }
            )
            rows.append(row)

        first = attribution.rank_correlations(rows, bootstrap_samples=40, seed=123)
        second = attribution.rank_correlations(rows, bootstrap_samples=40, seed=123)
        self.assertEqual(first, second)
        pulse_depth = next(
            row
            for row in first
            if row["measurement"] == "piPulseFid" and row["outcome"] == "N_best"
        )
        self.assertAlmostEqual(pulse_depth["spearman_rho"], 1.0)

    def test_grouped_permutation_importance_finds_known_driver(self):
        rng = np.random.default_rng(7)
        rows = []
        for index in range(120):
            pulse = rng.uniform(0.0, 1.0)
            row = {
                "piPulseFid": pulse,
                "T2_scale": rng.normal(),
                "T2_alpha": rng.normal(),
                "contrast_scale": rng.normal(),
                "photon_rate_scale": rng.normal(),
                "overhead_scale": rng.normal(),
                "N_best": int(round(8 + 56 * pulse)),
            }
            rows.append(row)

        first = attribution.permutation_importance(rows, repeats=20, seed=99, neighbors=9)
        second = attribution.permutation_importance(rows, repeats=20, seed=99, neighbors=9)
        self.assertEqual(first, second)
        by_resource = {row["resource"]: row for row in first}
        self.assertGreater(
            by_resource["pulse_fidelity"]["permutation_mae_increase_blocks"],
            by_resource["coherence"]["permutation_mae_increase_blocks"],
        )

    def test_single_resource_calibration_and_oat_are_deterministic(self):
        samples = attribution.load_sample_rows(
            ROOT / "outputs" / "reproduced_figures" / "xy8_monte_carlo_samples.csv"
        )[:6]
        nominal = load_nominal()
        rows, surfaces, taus, Ns, nominal_meta, _ = attribution.evaluate_existing_samples(
            nominal, samples, 200e3
        )
        first, first_decisions = attribution.calibration_value(
            nominal,
            rows,
            surfaces,
            taus,
            Ns,
            nominal_meta,
            200e3,
            bootstrap_samples=20,
            seed=456,
        )
        second, second_decisions = attribution.calibration_value(
            nominal,
            rows,
            surfaces,
            taus,
            Ns,
            nominal_meta,
            200e3,
            bootstrap_samples=20,
            seed=456,
        )
        self.assertEqual(first, second)
        self.assertEqual(first_decisions, second_decisions)
        self.assertEqual({row["resource"] for row in first}, set(attribution.RESOURCE_GROUPS))
        self.assertTrue(all(row["fixed_control_N"] == 20 for row in first))

        oat = attribution.run_oat_sweeps(nominal, 200e3)
        contrast_depths = {row["N_best"] for row in oat if row["measurement"] == "contrast_scale"}
        photon_depths = {row["N_best"] for row in oat if row["measurement"] == "photon_rate_scale"}
        self.assertEqual(contrast_depths, {20})
        self.assertEqual(photon_depths, {20})


if __name__ == "__main__":
    unittest.main()
