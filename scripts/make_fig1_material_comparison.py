from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "figs"
OUTPUT_PDF = OUTPUT_DIR / "figure1_material_comparison.pdf"
OUTPUT_PNG = OUTPUT_DIR / "figure1_material_comparison.png"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_k3_analysis():
    mod = load_module("k3_fig2", ROOT / "scripts" / "make_fig2_k3c60_edt.py")
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
        "k": analysis.k,
        "residual_minus": analysis.residual_minus,
        "residual_plus": analysis.residual_plus,
        "k_left": analysis.k_left,
        "k_right": analysis.k_right,
        "k0": analysis.k0,
        "width": analysis.edt_width,
        "threshold": mod.RESIDUAL_THRESHOLD,
        "ci": np.percentile(boot, [2.5, 97.5]),
    }


def load_material_analyses():
    mod = load_module("mat_panels", ROOT / "scripts" / "make_material_figure2_panels.py")
    datasets_by_path = {}
    analyses = []
    for config in mod.CONFIGS:
        datasets = datasets_by_path.setdefault(config.source_file, mod.parse_agr(config.source_file))
        analysis = mod.analyze_pair(config, datasets)
        boot = mod.bootstrap_widths(analysis)
        analyses.append(
            {
                "title": config.title,
                "pair_label": f"{config.dataset_minus + 1}/{config.dataset_plus + 1}",
                "k": analysis.k,
                "residual_minus": analysis.residual_minus,
                "residual_plus": analysis.residual_plus,
                "k_left": analysis.k_left,
                "k_right": analysis.k_right,
                "k0": analysis.k0,
                "width": analysis.edt_width,
                "threshold": config.threshold,
                "ci": np.percentile(boot, [2.5, 97.5]),
            }
        )
    return analyses


def plot_material_panel(ax, entry: dict, panel_label: str) -> None:
    ax.axvspan(entry["k_left"], entry["k_right"], color="0.88", alpha=0.8)
    ax.plot(entry["k"], entry["residual_minus"], color="#b2182b", lw=1.4)
    ax.plot(entry["k"], entry["residual_plus"], color="#2166ac", lw=1.4)
    ax.axhline(entry["threshold"], color="0.25", ls="--", lw=1.0)
    ax.axvline(entry["k0"], color="0.35", ls=":", lw=1.0)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$R_{\rm quad}(k)$")
    ymax = 1.05 * float(max(np.max(entry["residual_minus"]), np.max(entry["residual_plus"])))
    ax.set_ylim(0.0, max(0.20, ymax))
    ax.grid(alpha=0.25)
    ax.set_title(f"{panel_label} {entry['title']}", loc="left", fontweight="bold")
    ax.text(
        0.98,
        0.96,
        f"pair {entry['pair_label']}\n$k_0={entry['k0']:.3f}$\n$\\Delta k_{{\\rm EDT}}={entry['width']:.3f}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "0.8", "alpha": 0.9},
    )


def plot_summary_panel(ax, entries: list[dict], panel_label: str) -> None:
    labels = [entry["title"] for entry in entries]
    widths = np.array([entry["width"] for entry in entries], dtype=float)
    ci = np.array([entry["ci"] for entry in entries], dtype=float)
    lower = widths - ci[:, 0]
    upper = ci[:, 1] - widths
    x = np.arange(len(labels))
    colors = ["#4daf4a", "#377eb8", "#984ea3", "#ff7f00", "#e41a1c"]

    ax.bar(x, widths, color=colors, alpha=0.75)
    ax.errorbar(x, widths, yerr=np.vstack([lower, upper]), fmt="none", ecolor="0.15", capsize=3, lw=1.2)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel(r"$\Delta k_{\rm EDT}$")
    ax.set_title(f"{panel_label} Comparison of extracted EDT windows", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    for idx, entry in enumerate(entries):
        ax.text(
            idx,
            widths[idx] + max(upper[idx], 0.02) + 0.015,
            f"{entry['pair_label']}",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def main() -> None:
    k3 = load_k3_analysis()
    materials = load_material_analyses()
    ordered = [k3] + materials

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.8))
    axes = axes.ravel()
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    for ax, entry, label in zip(axes[:5], ordered, panel_labels):
        plot_material_panel(ax, entry, label)
    plot_summary_panel(axes[5], ordered, "(f)")
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PDF}")
    print(f"Saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
