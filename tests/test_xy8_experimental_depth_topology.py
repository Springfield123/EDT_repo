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
for path in (ANALYSIS_DIR, PLOTTING_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import xy8_experimental_depth_topology as topology
import xy8_literature_validation as validation


SCRIPT = PLOTTING_DIR / "make_xy8_experimental_depth_topology.py"
SPEC = importlib.util.spec_from_file_location(
    "make_xy8_experimental_depth_topology", SCRIPT
)
plotting = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(plotting)

MANIFEST_PATH = (
    ROOT / "data" / "raw" / "nv" / "literature_validation" / "pham_2012_manifest.json"
)
DATA_PATH = (
    ROOT / "data" / "raw" / "nv" / "literature_validation" / "pham_2012_fig3b.csv"
)
REQUEST_MANIFEST = (
    ROOT
    / "data"
    / "raw"
    / "nv"
    / "literature_validation"
    / "tsunaki_2025_2d_request_manifest.json"
)


def load_inputs():
    return validation.load_manifest(MANIFEST_PATH), validation.load_measurements(DATA_PATH)


class TestXY8ExperimentalDepthTopology(unittest.TestCase):
    def test_connected_runs_and_anchor_component_are_counted(self):
        pulses = np.asarray([8, 16, 24, 32, 40])
        summary = topology.summarize_masks(
            np.asarray([True, True, False, True, True]), pulses, 3
        )
        self.assertEqual(int(summary["beta_0"][0]), 2)
        self.assertEqual(int(summary["lower_boundary_pulses"][0]), 32)
        self.assertEqual(int(summary["upper_boundary_pulses"][0]), 40)
        self.assertEqual(int(summary["window_width_pulses"][0]), 8)

    def test_central_pham_curve_has_one_region_across_thresholds(self):
        manifest, measurements = load_inputs()
        result = topology.analyze_pham_topology(
            manifest,
            measurements,
            thresholds_percent=(1, 10, 100),
            checkpoints_percent=(1, 10, 100),
            trials=2_000,
            seed=42,
        )
        self.assertEqual(result["summary"]["central_measured_optimum_pulses"], 106)
        self.assertEqual(result["summary"]["frozen_model_optimum_dense_pulses"], 86)
        self.assertTrue(result["summary"]["central_beta_0_equals_1_from_1_to_100pct"])
        ten = result["threshold_rows"][1]
        self.assertEqual(ten["central_beta_0"], 1)
        self.assertEqual(ten["central_lower_boundary_pulses"], 80)
        self.assertEqual(ten["central_upper_boundary_pulses"], 212)
        self.assertEqual(ten["central_window_width_pulses"], 132)
        self.assertGreater(ten["bootstrap_probability_beta_0_greater_than_1"], 0.35)
        self.assertLess(ten["bootstrap_probability_beta_0_greater_than_1"], 0.60)

    def test_seeded_bootstrap_is_reproducible(self):
        manifest, measurements = load_inputs()
        kwargs = {
            "thresholds_percent": (5, 10),
            "checkpoints_percent": (5, 10),
            "trials": 1_500,
            "seed": 17,
        }
        first = topology.analyze_pham_topology(manifest, measurements, **kwargs)
        second = topology.analyze_pham_topology(manifest, measurements, **kwargs)
        self.assertEqual(first["threshold_rows"], second["threshold_rows"])
        self.assertEqual(first["distribution_rows"], second["distribution_rows"])
        self.assertEqual(first["summary"], second["summary"])

    def test_exports_are_consistent_and_farfurnik_remains_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            outdir = Path(temporary) / "figures"
            mirror = Path(temporary) / "processed"
            summary = plotting.run(
                MANIFEST_PATH,
                DATA_PATH,
                outdir,
                mirror_data_dir=mirror,
                trials=2_000,
                seed=91,
            )
            exported = json.loads(
                (outdir / plotting.SUMMARY_FILENAME).read_text(encoding="utf-8")
            )
            with (outdir / plotting.THRESHOLD_FILENAME).open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(summary, exported)
            self.assertEqual(len(rows), 100)
            self.assertEqual(
                exported["input_digests"]["measurement_csv_sha256"],
                validation.file_sha256(DATA_PATH),
            )
            self.assertIn("not included", exported["separation_statement"])
            self.assertTrue((outdir / f"{plotting.FIGURE_BASENAME}.png").exists())
            self.assertTrue((outdir / f"{plotting.FIGURE_BASENAME}.pdf").exists())
            for filename in (
                plotting.THRESHOLD_FILENAME,
                plotting.DISTRIBUTION_FILENAME,
                plotting.SUMMARY_FILENAME,
            ):
                self.assertEqual(
                    (outdir / filename).read_bytes(), (mirror / filename).read_bytes()
                )

    def test_tsunaki_request_is_frozen_before_data_receipt(self):
        request = json.loads(REQUEST_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(request["status"], "request_drafted_not_sent")
        self.assertEqual(request["received_files"], [])
        self.assertFalse(request["frozen_evaluation_policy"]["fit_to_observed_optimum"])
        self.assertGreaterEqual(
            request["minimum_acceptance_criteria"]["experimental_timing_sweep_count"], 2
        )
        self.assertIn(
            "pulse-error mechanism",
            request["separation_policy"]["farfurnik_2015"],
        )


if __name__ == "__main__":
    unittest.main()
