#!/usr/bin/env python3
"""Generate the separate Loretz XY8 spurious-region mechanism check."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "code" / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))
import xy8_loretz_spurious_topology as topology


DEFAULT_MANIFEST = ROOT / "data" / "raw" / "nv" / "literature_validation" / "loretz_2015_fig4a_topology_manifest.json"
DEFAULT_DATA = ROOT / "data" / "raw" / "nv" / "literature_validation" / "loretz_2015_fig4a_xy8_spectrum.csv"
DEFAULT_OUTDIR = ROOT / "outputs" / "reproduced_figures"
FILES = {
    "segments": "xy8_loretz_spurious_spectrum.csv",
    "thresholds": "xy8_loretz_spurious_thresholds.csv",
    "components": "xy8_loretz_spurious_components.csv",
    "summary": "xy8_loretz_spurious_summary.json",
    "figure": "fig_xy8_loretz_spurious_topology",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    value.update(path.read_bytes())
    return value.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(result: dict, outdir: Path) -> None:
    colors = {128: "#d73027", 256: "#4575b4", 352: "#74add1", 400: "#313695", 512: "#d73027"}
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5), constrained_layout=True)
    fig.suptitle(
        "Independent XY8 data reveal spurious resonant regions",
        fontsize=14,
        fontweight="bold",
    )
    ax = axes[0]
    segment_ids = []
    for row in result["segment_rows"]:
        if row["segment_id"] not in segment_ids:
            segment_ids.append(row["segment_id"])
    for segment_id in segment_ids:
        rows = [row for row in result["segment_rows"] if row["segment_id"] == segment_id]
        pulses = rows[0]["total_pi_pulses"]
        ax.plot(
            [row["frequency_MHz"] for row in rows],
            [row["transition_probability"] for row in rows],
            color=colors[pulses],
            linewidth=1.8,
            marker="o",
            markersize=2.5,
            label=f"{pulses} pulses",
        )
        ax.hlines(
            rows[0]["baseline_probability"],
            rows[0]["frequency_MHz"],
            rows[-1]["frequency_MHz"],
            color=colors[pulses],
            linewidth=0.8,
            linestyle="--",
        )
    ax.set_xlabel("Detection frequency (MHz)")
    ax.set_ylabel("Transition probability")
    ax.set_xlim(1.5, 9.0)
    ax.set_ylim(0.02, 0.66)
    ax.grid(alpha=0.18)
    ax.legend(ncol=2, loc="upper right")
    ax.set_title("A  Vector-digitized experimental segments")

    ax = axes[1]
    rows = [row for row in result["threshold_rows"] if row["threshold_percent"] <= 80]
    ax.step(
        [row["threshold_percent"] for row in rows],
        [row["beta_0_segmented_spectrum"] for row in rows],
        where="post",
        color="#1f4e79",
        linewidth=2.3,
    )
    first = result["summary"]["first_threshold_percent_where_identity_is_selected"]
    ax.axvline(first["second_harmonic"], color="#e66101", linestyle="--", linewidth=1.4)
    ax.annotate(
        "Second and fourth harmonics enter\nat 61% allowed loss",
        xy=(first["second_harmonic"], 3),
        xytext=(45, 4.6),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
    )
    ax.set_xlabel("Allowed loss from strongest response (%)")
    ax.set_ylabel(r"Disconnected spectral regions, $\beta_0$")
    ax.set_xlim(1, 80)
    ax.set_ylim(0.7, 5.3)
    ax.set_yticks(range(1, 6))
    ax.grid(alpha=0.2)
    ax.set_title("B  Harmonic regions emerge as the threshold expands")

    fig.text(
        0.5,
        -0.02,
        "Each acquisition segment is baseline-corrected; one global normalization is used. Published simulations are excluded. No frequency shift or fitting.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    basename = FILES["figure"]
    fig.savefig(outdir / f"{basename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"{basename}.pdf", bbox_inches="tight")
    plt.close(fig)


def run(manifest_path: Path, data_path: Path, outdir: Path, mirror_data_dir: Path | None = None) -> dict:
    manifest, rows = topology.load(manifest_path, data_path)
    result = topology.analyze(manifest, rows)
    result["summary"]["source"] = manifest["source"]
    result["summary"]["input_digests"] = {
        "manifest_sha256": digest(manifest_path),
        "digitized_measurements_sha256": digest(data_path),
    }
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / FILES["segments"], result["segment_rows"])
    write_csv(outdir / FILES["thresholds"], result["threshold_rows"])
    write_csv(outdir / FILES["components"], result["component_rows"])
    (outdir / FILES["summary"]).write_text(json.dumps(result["summary"], indent=2) + "\n", encoding="utf-8")
    make_figure(result, outdir)
    if mirror_data_dir is not None:
        mirror_data_dir.mkdir(parents=True, exist_ok=True)
        for key in ("segments", "thresholds", "components", "summary"):
            shutil.copy2(outdir / FILES[key], mirror_data_dir / FILES[key])
    return result["summary"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--mirror-data-dir", type=Path)
    args = parser.parse_args()
    summary = run(args.manifest, args.data, args.outdir, args.mirror_data_dir)
    print("Loretz spurious topology:", summary["first_threshold_percent_where_identity_is_selected"])


if __name__ == "__main__":
    main()
