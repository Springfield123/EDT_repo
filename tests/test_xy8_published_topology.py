from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "code" / "analysis"
PLOTTING = ROOT / "code" / "plotting"
for path in (ANALYSIS, PLOTTING):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import xy8_devience_topology as devience
import xy8_loretz_spurious_topology as loretz


def load_script(name: str):
    path = PLOTTING / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


devience_plot = load_script("make_xy8_devience_topology")
loretz_plot = load_script("make_xy8_loretz_spurious_topology")
RAW = ROOT / "data" / "raw" / "nv" / "literature_validation"
DEV_MANIFEST = RAW / "devience_2015_fig2b_manifest.json"
DEV_DATA = RAW / "devience_2015_fig2b_xy8.csv"
LOR_MANIFEST = RAW / "loretz_2015_fig4a_topology_manifest.json"
LOR_DATA = RAW / "loretz_2015_fig4a_xy8_spectrum.csv"


class TestPublishedTopology(unittest.TestCase):
    def test_devience_selection_and_offsets_were_frozen(self):
        manifest = devience.load_manifest(DEV_MANIFEST)
        rows = devience.load_measurements(DEV_DATA, manifest)
        self.assertEqual(manifest["predeclared_depths"]["xy8_blocks"], [10, 20, 25, 40])
        self.assertEqual(len(rows), 95)
        self.assertFalse(manifest["normalization_policy"]["per_depth_normalization_allowed"])
        self.assertFalse(manifest["topology_policy"]["timing_shift_allowed"])
        with DEV_DATA.open("r", encoding="utf-8", newline="") as handle:
            first = next(csv.DictReader(handle))
        self.assertAlmostEqual(
            float(first["plotted_contrast"]) - float(first["display_offset"]),
            float(first["corrected_contrast"]),
            places=8,
        )

    def test_devience_two_regions_are_stable_at_ten_percent(self):
        manifest = devience.load_manifest(DEV_MANIFEST)
        rows = devience.load_measurements(DEV_DATA, manifest)
        result = devience.analyze(manifest, rows, bootstrap_trials=1_000, seed=2014313)
        ten = result["threshold_rows"][9]
        self.assertEqual(ten["central_beta_0"], 2)
        self.assertGreater(ten["bootstrap_probability_matches_central_digitization_only"], 0.90)
        strongest = result["summary"]["strongest_measured_response"]
        self.assertEqual(strongest["xy8_blocks"], 40)
        self.assertAlmostEqual(strongest["frequency_MHz"], 0.945)
        self.assertIn("does not validate", result["summary"]["guardrail"])

    def test_loretz_is_a_separate_experimental_harmonic_check(self):
        manifest, rows = loretz.load(LOR_MANIFEST, LOR_DATA)
        result = loretz.analyze(manifest, rows)
        self.assertEqual(len(rows), 135)
        first = result["summary"]["first_threshold_percent_where_identity_is_selected"]
        self.assertEqual(first["fundamental"], 1)
        self.assertEqual(first["second_harmonic"], 61)
        self.assertEqual(first["fourth_harmonic"], 61)
        self.assertIn("published simulation was excluded", result["summary"]["no_fit_statement"].lower())
        self.assertIn("not combined", result["summary"]["separation_guardrail"])

    def test_plotting_exports_match_summaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            outdir = Path(temporary)
            dev_summary = devience_plot.run(
                DEV_MANIFEST, DEV_DATA, outdir, bootstrap_trials=200, seed=7
            )
            lor_summary = loretz_plot.run(LOR_MANIFEST, LOR_DATA, outdir)
            self.assertEqual(
                dev_summary,
                json.loads((outdir / devience_plot.FILENAMES["summary"]).read_text()),
            )
            self.assertEqual(
                lor_summary,
                json.loads((outdir / loretz_plot.FILES["summary"]).read_text()),
            )
            self.assertTrue((outdir / f"{devience_plot.FILENAMES['figure']}.png").exists())
            self.assertTrue((outdir / f"{loretz_plot.FILES['figure']}.png").exists())


if __name__ == "__main__":
    unittest.main()
