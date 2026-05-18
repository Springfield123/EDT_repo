from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code" / "utilities") not in sys.path:
    sys.path.insert(0, str(ROOT / "code" / "utilities"))

from edt.curvature import compute_quadratic_residual, fit_quadratic_pair


ROOT = Path(__file__).resolve().parents[2]
FIGURE_PATH = ROOT / "outputs" / "reproduced_figures" / "S4_material_controls.pdf"
CSV_PATH = ROOT / "data" / "processed" / "material_controls_summary.csv"

NOMINAL_THRESHOLD = 0.08
THRESHOLD_SWEEP = np.array([0.04, 0.06, 0.08, 0.10, 0.12, 0.16], dtype=float)


@dataclass(frozen=True)
class MaterialConfig:
    key: str
    title: str
    role: str
    source: Path
    dataset_minus: int
    dataset_plus: int
    start_idx: int | None = None
    end_idx: int | None = None
    max_half_window: int = 24


@dataclass(frozen=True)
class MaterialAnalysis:
    config: MaterialConfig
    k: np.ndarray
    kappa_minus: np.ndarray
    kappa_plus: np.ndarray
    center_idx: int
    center_k: float
    local_fit_half_window: int
    local_residual: float
    nominal_half_window: int
    nominal_width: float
    threshold_widths: np.ndarray
    threshold_min_width: float
    threshold_max_width: float


CONFIGS = [
    MaterialConfig(
        key="k3c60",
        title="K$_3$C$_{60}$",
        role="Main positive reference",
        source=ROOT / "data" / "raw" / "k3c60" / "tmp.agr",
        dataset_minus=207,
        dataset_plus=208,
        start_idx=150,
        end_idx=281,
        max_half_window=28,
    ),
    MaterialConfig(
        key="v",
        title="V",
        role="Transition-metal comparison",
        source=ROOT / "data" / "raw" / "v" / "V-EP-F1.agr",
        dataset_minus=22,
        dataset_plus=23,
        max_half_window=20,
    ),
    MaterialConfig(
        key="na",
        title="Na",
        role="Simple-metal baseline",
        source=ROOT / "data" / "raw" / "na" / "Na-m5.agr",
        dataset_minus=10,
        dataset_plus=11,
        max_half_window=18,
    ),
    MaterialConfig(
        key="lah10",
        title="LaH$_{10}$",
        role="High-curvature hydride control",
        source=ROOT / "data" / "raw" / "lah10" / "tmp_LaH10_m16.agr",
        dataset_minus=33,
        dataset_plus=35,
        max_half_window=12,
    ),
    MaterialConfig(
        key="ybacu3o7",
        title="YBa$_2$Cu$_3$O$_7$",
        role="Correlated-oxide control",
        source=ROOT / "data" / "raw" / "ybacu3o7" / "YBa2Cu3O7_m33.agr",
        dataset_minus=145,
        dataset_plus=147,
        max_half_window=12,
    ),
]


def parse_agr(path: Path) -> list[np.ndarray]:
    datasets: list[np.ndarray] = []
    current: list[list[float]] = []
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("@") or line.startswith("#") or not line.strip():
            if current:
                datasets.append(np.array(current, dtype=float))
                current = []
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            current.append([float(parts[0]), float(parts[1])])
        except ValueError:
            continue
    if current:
        datasets.append(np.array(current, dtype=float))
    return datasets


def width_scan(
    k: np.ndarray,
    kappa_plus: np.ndarray,
    kappa_minus: np.ndarray,
    center_idx: int,
    threshold: float,
    max_half_window: int,
) -> tuple[float, int]:
    half_limit = min(max_half_window, center_idx, len(k) - center_idx - 1)
    best_width = 0.0
    best_half = 0
    for half in range(3, half_limit + 1):
        sl = slice(center_idx - half, center_idx + half + 1)
        fit = fit_quadratic_pair(k[sl], kappa_plus[sl], kappa_minus[sl])
        residual = compute_quadratic_residual(k[sl], kappa_plus[sl], kappa_minus[sl], fit)
        width = float(k[center_idx + half] - k[center_idx - half])
        if residual < threshold and width > best_width:
            best_width = width
            best_half = half
    return best_width, best_half


def analyze_material(config: MaterialConfig) -> MaterialAnalysis:
    datasets = parse_agr(config.source)
    arr_minus = datasets[config.dataset_minus]
    arr_plus = datasets[config.dataset_plus]
    data_slice = slice(config.start_idx, config.end_idx)
    k = arr_minus[data_slice, 0]
    kappa_minus = arr_minus[data_slice, 1]
    kappa_plus = arr_plus[data_slice, 1]

    idx_minus = int(np.argmin(kappa_minus))
    idx_plus = int(np.argmax(kappa_plus))
    center_idx = int(round(0.5 * (idx_minus + idx_plus)))
    center_idx = min(max(center_idx, 3), len(k) - 4)

    local_half = min(8, config.max_half_window, center_idx, len(k) - center_idx - 1)
    local_slice = slice(center_idx - local_half, center_idx + local_half + 1)
    local_fit = fit_quadratic_pair(k[local_slice], kappa_plus[local_slice], kappa_minus[local_slice])
    local_residual = compute_quadratic_residual(
        k[local_slice],
        kappa_plus[local_slice],
        kappa_minus[local_slice],
        local_fit,
    )

    threshold_widths = []
    nominal_half = 0
    for threshold in THRESHOLD_SWEEP:
        width, half = width_scan(k, kappa_plus, kappa_minus, center_idx, threshold, config.max_half_window)
        threshold_widths.append(width)
        if np.isclose(threshold, NOMINAL_THRESHOLD):
            nominal_half = half
    threshold_widths_arr = np.asarray(threshold_widths, dtype=float)

    return MaterialAnalysis(
        config=config,
        k=k,
        kappa_minus=kappa_minus,
        kappa_plus=kappa_plus,
        center_idx=center_idx,
        center_k=float(k[center_idx]),
        local_fit_half_window=local_half,
        local_residual=float(local_residual),
        nominal_half_window=nominal_half,
        nominal_width=float(threshold_widths_arr[np.where(np.isclose(THRESHOLD_SWEEP, NOMINAL_THRESHOLD))[0][0]]),
        threshold_widths=threshold_widths_arr,
        threshold_min_width=float(np.min(threshold_widths_arr)),
        threshold_max_width=float(np.max(threshold_widths_arr)),
    )


def build_summary_table(analyses: list[MaterialAnalysis]) -> pd.DataFrame:
    rows = []
    for analysis in analyses:
        cfg = analysis.config
        rows.append(
            {
                "material": cfg.key,
                "title": cfg.title,
                "role": cfg.role,
                "source_file": str(cfg.source.relative_to(ROOT)),
                "dataset_minus": cfg.dataset_minus,
                "dataset_plus": cfg.dataset_plus,
                "slice_start": cfg.start_idx,
                "slice_end": cfg.end_idx,
                "center_k": analysis.center_k,
                "nominal_threshold": NOMINAL_THRESHOLD,
                "local_residual": analysis.local_residual,
                "nominal_width": analysis.nominal_width,
                "threshold_min_width": analysis.threshold_min_width,
                "threshold_max_width": analysis.threshold_max_width,
            }
        )
    return pd.DataFrame(rows)


def plot_material_panel(ax: plt.Axes, analysis: MaterialAnalysis) -> None:
    k = analysis.k
    center_k = analysis.center_k
    ax.plot(k, analysis.kappa_minus, color="#C44E52", lw=1.3, label=r"$\kappa_-$")
    ax.plot(k, analysis.kappa_plus, color="#4C72B0", lw=1.3, label=r"$\kappa_+$")
    if analysis.nominal_half_window > 0:
        left = analysis.center_idx - analysis.nominal_half_window
        right = analysis.center_idx + analysis.nominal_half_window
        ax.axvspan(k[left], k[right], color="#55A868", alpha=0.16)
    ax.axvline(center_k, color="0.3", lw=0.9, ls="--")
    ax.set_title(analysis.config.title, fontsize=10)
    ax.text(
        0.02,
        0.96,
        (
            f"{analysis.config.role}\n"
            f"pair=({analysis.config.dataset_minus},{analysis.config.dataset_plus})\n"
            f"$R_{{\\rm local}}={analysis.local_residual:.3f}$, "
            f"$\\Delta k_{{0.08}}={analysis.nominal_width:.3f}$"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "0.85"},
    )
    ax.tick_params(labelsize=8)
    ax.set_xlabel(r"$k$", fontsize=9)
    ax.set_ylabel(r"$\kappa$", fontsize=9)


def plot_summary_panel(ax: plt.Axes, analyses: list[MaterialAnalysis]) -> None:
    labels = [analysis.config.title for analysis in analyses]
    widths = np.array([analysis.nominal_width for analysis in analyses], dtype=float)
    mins = np.array([analysis.threshold_min_width for analysis in analyses], dtype=float)
    maxs = np.array([analysis.threshold_max_width for analysis in analyses], dtype=float)
    lower = widths - mins
    upper = maxs - widths
    y = np.arange(len(analyses))
    ax.barh(y, widths, color="#8172B2", alpha=0.82)
    ax.errorbar(widths, y, xerr=np.vstack([lower, upper]), fmt="none", ecolor="0.2", capsize=3, lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(r"Candidate $\Delta k$ under common threshold scan", fontsize=9)
    ax.set_title("Threshold-scan summary", fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    for idx, analysis in enumerate(analyses):
        ax.text(
            widths[idx] + upper[idx] + 0.01 * max(maxs.max(), 1.0),
            idx,
            f"{analysis.local_residual:.3f}",
            va="center",
            fontsize=8,
            color="0.25",
        )
    ax.text(
        0.02,
        0.02,
        r"numbers at right: $R_{\rm local}$",
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="bottom",
    )


def main() -> None:
    analyses = [analyze_material(config) for config in CONFIGS]
    summary = build_summary_table(analyses)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(CSV_PATH, index=False)

    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(3, 2, figsize=(11.2, 11.8), constrained_layout=True)
    axes_flat = axes.ravel()
    for ax, analysis in zip(axes_flat[:5], analyses, strict=True):
        plot_material_panel(ax, analysis)
    plot_summary_panel(axes_flat[5], analyses)
    fig.suptitle(
        "Cross-material curvature controls using representative archived candidate branch pairs",
        fontsize=12,
        y=1.01,
    )

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", FIGURE_PATH)
    print("Saved:", CSV_PATH)


if __name__ == "__main__":
    main()
