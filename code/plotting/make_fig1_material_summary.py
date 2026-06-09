from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "reproduced_figures"
OUTPUT_PDF = OUTPUT_DIR / "figure1_material_summary.pdf"
OUTPUT_PNG = OUTPUT_DIR / "figure1_material_summary.png"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_k3_summary():
    mod = load_module("k3_fig2_summary", ROOT / "code" / "plotting" / "make_fig2_k3c60_edt.py")
    datasets = mod.parse_agr(mod.CURVATURE_FILE)
    metadata = mod.parse_agr_metadata(mod.CURVATURE_FILE)
    analysis = mod.analyze_pair(
        datasets,
        metadata,
        mod.DATASET_MINUS,
        mod.DATASET_PLUS,
        start_idx=mod.K_START,
        end_idx=mod.K_END,
        fit_half_window=mod.FIT_HALF_WINDOW,
        threshold=mod.RESIDUAL_THRESHOLD,
    )
    boot = mod.bootstrap_widths(analysis)
    return {
        "title": r"K$_3$C$_{60}$",
        "pair_label": "208/209",
        "width": analysis.edt_width,
        "ci": np.percentile(boot, [2.5, 97.5]),
    }


def load_material_summaries():
    mod = load_module("mat_panels_summary", ROOT / "code" / "plotting" / "make_material_figure2_panels.py")
    datasets_by_path = {}
    out = []
    for config in mod.CONFIGS:
        datasets = datasets_by_path.setdefault(config.source_file, mod.parse_agr(config.source_file))
        analysis = mod.analyze_pair(config, datasets)
        boot = mod.bootstrap_widths(analysis)
        out.append(
            {
                "title": config.title,
                "pair_label": f"{config.dataset_minus + 1}/{config.dataset_plus + 1}",
                "width": analysis.edt_width,
                "ci": np.percentile(boot, [2.5, 97.5]),
            }
        )
    return out


def main() -> None:
    entries = [load_k3_summary()] + load_material_summaries()

    labels = [entry["title"] for entry in entries]
    widths = np.array([entry["width"] for entry in entries], dtype=float)
    ci = np.array([entry["ci"] for entry in entries], dtype=float)
    lower = widths - ci[:, 0]
    upper = ci[:, 1] - widths
    x = np.arange(len(labels))
    colors = ["#4daf4a", "#377eb8", "#984ea3", "#ff7f00", "#e41a1c"]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.bar(x, widths, color=colors, alpha=0.8)
    ax.errorbar(x, widths, yerr=np.vstack([lower, upper]), fmt="none", ecolor="0.15", capsize=4, lw=1.2)
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.set_ylabel(r"$\Delta k_{\rm EDT}$")
    ax.set_title("Cross-material comparison of extracted EDT windows", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    for idx, entry in enumerate(entries):
        ax.text(
            idx,
            widths[idx] + max(upper[idx], 0.015) + 0.01,
            entry["pair_label"],
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PDF}")
    print(f"Saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
