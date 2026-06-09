from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CURVATURE_FILE = ROOT / "data" / "k3c60" / "tmp.agr"
MAIN_DIR = ROOT / "figures" / "main"
SUPP_DIR = ROOT / "figures" / "supplementary"
SUMMARY_JSON = ROOT / "data" / "k3c60" / "k3c60_edt_window_summary.json"

DATASET_MINUS = 207
DATASET_PLUS = 208
MODE_LABEL = "m120"
MODE_FREQUENCY_CM = 1090.35
K_START = 150
K_END = 281
FIT_HALF_WINDOW = 18
RESIDUAL_THRESHOLD = 0.08
BOOTSTRAP_SAMPLES = 500
RNG_SEED = 12345


@dataclass(frozen=True)
class PairAnalysis:
    dataset_minus: int
    dataset_plus: int
    band_minus_label: str
    band_plus_label: str
    k: np.ndarray
    y_minus: np.ndarray
    y_plus: np.ndarray
    idx_minus: int
    idx_plus: int
    fit_minus: dict[str, np.ndarray | float]
    fit_plus: dict[str, np.ndarray | float]
    fit_curve_minus: np.ndarray
    fit_curve_plus: np.ndarray
    residual_minus: np.ndarray
    residual_plus: np.ndarray
    left_minus: int
    right_minus: int
    left_plus: int
    right_plus: int
    left_edt: int
    right_edt: int
    left_core: int
    right_core: int
    threshold: float

    @property
    def k_left(self) -> float:
        return float(self.k[self.left_edt])

    @property
    def k_right(self) -> float:
        return float(self.k[self.right_edt - 1])

    @property
    def edt_width(self) -> float:
        return float(self.k_right - self.k_left)

    @property
    def k0(self) -> float:
        return 0.5 * (float(self.fit_minus["k_apex"]) + float(self.fit_plus["k_apex"]))

    @property
    def k_core_left(self) -> float:
        return float(self.k[self.left_core])

    @property
    def k_core_right(self) -> float:
        return float(self.k[self.right_core - 1])

    @property
    def core_width(self) -> float:
        return float(self.k_core_right - self.k_core_left)


def parse_agr(filename: Path) -> list[np.ndarray]:
    datasets: list[np.ndarray] = []
    current: list[list[float]] = []
    for line in filename.read_text().splitlines():
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


def parse_agr_metadata(filename: Path) -> dict[str, object]:
    text = filename.read_text()
    y_label_match = re.search(r'@\s+yaxis\s+label\s+"([^"]+)"', text)
    x_strings = re.findall(r'@\s+string def "([^"]+)"', text)
    dataset_labels: dict[int, str] = {}
    for idx, label in re.findall(r'@\s+s(\d+)\s+legend\s+"([^"]*)"', text):
        dataset_labels[int(idx)] = label.strip()
    return {
        "y_label_raw": y_label_match.group(1) if y_label_match else "",
        "path_labels": x_strings[-5:] if len(x_strings) >= 5 else x_strings,
        "dataset_labels": dataset_labels,
    }


def clean_grace_label(raw: str) -> str:
    cleaned = raw.replace("\\S2\\N", "^2").replace("\\s", "_").replace("\\cE\\C\\N", "A")
    cleaned = cleaned.replace("\\", "")
    return cleaned


def fit_local_parabola(k: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray | float]:
    coeffs = np.polyfit(k, y, 2)
    a, b, _ = coeffs
    k_apex = -b / (2.0 * a)
    y_apex = np.polyval(coeffs, k_apex)
    return {
        "coeffs": coeffs,
        "a": float(a),
        "k_apex": float(k_apex),
        "y_apex": float(y_apex),
    }


def extract_local_window(k: np.ndarray, y: np.ndarray, center_idx: int, half_window: int) -> tuple[np.ndarray, np.ndarray]:
    i0 = max(0, center_idx - half_window)
    i1 = min(len(k), center_idx + half_window + 1)
    return k[i0:i1], y[i0:i1]


def contiguous_true_region(mask: np.ndarray, center_idx: int) -> tuple[int, int]:
    if not mask[center_idx]:
        return center_idx, center_idx + 1
    left = center_idx
    right = center_idx
    while left - 1 >= 0 and mask[left - 1]:
        left -= 1
    while right + 1 < len(mask) and mask[right + 1]:
        right += 1
    return left, right + 1


def analyze_pair(
    datasets: list[np.ndarray],
    metadata: dict[str, object],
    dataset_minus: int,
    dataset_plus: int,
    start_idx: int = K_START,
    end_idx: int = K_END,
    fit_half_window: int = FIT_HALF_WINDOW,
    threshold: float = RESIDUAL_THRESHOLD,
) -> PairAnalysis:
    k = datasets[dataset_minus][start_idx:end_idx, 0]
    y_minus = datasets[dataset_minus][start_idx:end_idx, 1]
    y_plus = datasets[dataset_plus][start_idx:end_idx, 1]

    idx_minus = int(np.argmin(y_minus))
    idx_plus = int(np.argmax(y_plus))

    k_minus_fit, y_minus_fit = extract_local_window(k, y_minus, idx_minus, fit_half_window)
    k_plus_fit, y_plus_fit = extract_local_window(k, y_plus, idx_plus, fit_half_window)
    fit_minus = fit_local_parabola(k_minus_fit, y_minus_fit)
    fit_plus = fit_local_parabola(k_plus_fit, y_plus_fit)

    fit_curve_minus = np.polyval(fit_minus["coeffs"], k)
    fit_curve_plus = np.polyval(fit_plus["coeffs"], k)
    residual_minus = np.abs(y_minus - fit_curve_minus) / max(np.max(np.abs(y_minus)), 1e-14)
    residual_plus = np.abs(y_plus - fit_curve_plus) / max(np.max(np.abs(y_plus)), 1e-14)

    left_minus, right_minus = contiguous_true_region(residual_minus <= threshold, idx_minus)
    left_plus, right_plus = contiguous_true_region(residual_plus <= threshold, idx_plus)
    left_edt = min(left_minus, left_plus)
    right_edt = max(right_minus, right_plus)
    left_core = max(left_minus, left_plus)
    right_core = min(right_minus, right_plus)
    if right_core <= left_core:
        left_core, right_core = left_edt, right_edt

    labels = metadata["dataset_labels"]  # type: ignore[index]
    band_minus_label = labels.get(dataset_minus, f"Band {dataset_minus + 1}")  # type: ignore[union-attr]
    band_plus_label = labels.get(dataset_plus, f"Band {dataset_plus + 1}")  # type: ignore[union-attr]
    return PairAnalysis(
        dataset_minus=dataset_minus,
        dataset_plus=dataset_plus,
        band_minus_label=band_minus_label,
        band_plus_label=band_plus_label,
        k=k,
        y_minus=y_minus,
        y_plus=y_plus,
        idx_minus=idx_minus,
        idx_plus=idx_plus,
        fit_minus=fit_minus,
        fit_plus=fit_plus,
        fit_curve_minus=fit_curve_minus,
        fit_curve_plus=fit_curve_plus,
        residual_minus=residual_minus,
        residual_plus=residual_plus,
        left_minus=left_minus,
        right_minus=right_minus,
        left_plus=left_plus,
        right_plus=right_plus,
        left_edt=left_edt,
        right_edt=right_edt,
        left_core=left_core,
        right_core=right_core,
        threshold=threshold,
    )


def bootstrap_widths(
    analysis: PairAnalysis,
    fit_half_window: int = FIT_HALF_WINDOW,
    threshold: float = RESIDUAL_THRESHOLD,
    n_boot: int = BOOTSTRAP_SAMPLES,
    seed: int = RNG_SEED,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k_minus_fit, y_minus_fit = extract_local_window(analysis.k, analysis.y_minus, analysis.idx_minus, fit_half_window)
    k_plus_fit, y_plus_fit = extract_local_window(analysis.k, analysis.y_plus, analysis.idx_plus, fit_half_window)
    widths = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        minus_idx = rng.integers(0, k_minus_fit.size, size=k_minus_fit.size)
        plus_idx = rng.integers(0, k_plus_fit.size, size=k_plus_fit.size)
        fit_minus = fit_local_parabola(k_minus_fit[minus_idx], y_minus_fit[minus_idx])
        fit_plus = fit_local_parabola(k_plus_fit[plus_idx], y_plus_fit[plus_idx])
        fit_curve_minus = np.polyval(fit_minus["coeffs"], analysis.k)
        fit_curve_plus = np.polyval(fit_plus["coeffs"], analysis.k)
        residual_minus = np.abs(analysis.y_minus - fit_curve_minus) / max(np.max(np.abs(analysis.y_minus)), 1e-14)
        residual_plus = np.abs(analysis.y_plus - fit_curve_plus) / max(np.max(np.abs(analysis.y_plus)), 1e-14)
        left_minus, right_minus = contiguous_true_region(residual_minus <= threshold, analysis.idx_minus)
        left_plus, right_plus = contiguous_true_region(residual_plus <= threshold, analysis.idx_plus)
        left_edt = min(left_minus, left_plus)
        right_edt = max(right_minus, right_plus)
        widths[idx] = float(analysis.k[right_edt - 1] - analysis.k[left_edt])
    return widths


def threshold_sweep(
    datasets: list[np.ndarray],
    metadata: dict[str, object],
    thresholds: np.ndarray,
    fit_half_window: int = FIT_HALF_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    widths = []
    for threshold in thresholds:
        result = analyze_pair(
            datasets,
            metadata,
            DATASET_MINUS,
            DATASET_PLUS,
            fit_half_window=fit_half_window,
            threshold=float(threshold),
        )
        widths.append(result.edt_width)
    return thresholds, np.asarray(widths, dtype=float)


def fit_window_sweep(
    datasets: list[np.ndarray],
    metadata: dict[str, object],
    half_windows: np.ndarray,
    threshold: float = RESIDUAL_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    widths = []
    for half_window in half_windows:
        result = analyze_pair(
            datasets,
            metadata,
            DATASET_MINUS,
            DATASET_PLUS,
            fit_half_window=int(half_window),
            threshold=threshold,
        )
        widths.append(result.edt_width)
    return half_windows, np.asarray(widths, dtype=float)


def kgrid_sweep(
    datasets: list[np.ndarray],
    metadata: dict[str, object],
    stride_values: list[int],
    fit_half_window: int = FIT_HALF_WINDOW,
    threshold: float = RESIDUAL_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    widths = []
    sample_counts = []
    for stride in stride_values:
        coarse_datasets = []
        for idx in (DATASET_MINUS, DATASET_PLUS):
            arr = datasets[idx][K_START:K_END:stride]
            coarse_datasets.append(arr)
        half_window = max(4, int(np.ceil(fit_half_window / stride)))
        result = analyze_pair(
            coarse_datasets,
            {
                "dataset_labels": {
                    0: metadata["dataset_labels"][DATASET_MINUS],  # type: ignore[index]
                    1: metadata["dataset_labels"][DATASET_PLUS],  # type: ignore[index]
                }
            },
            0,
            1,
            start_idx=0,
            end_idx=coarse_datasets[0].shape[0],
            fit_half_window=half_window,
            threshold=threshold,
        )
        widths.append(result.edt_width)
        sample_counts.append(coarse_datasets[0].shape[0])
    return np.asarray(sample_counts, dtype=int), np.asarray(widths, dtype=float)


def band_pair_sweep(
    datasets: list[np.ndarray],
    metadata: dict[str, object],
    candidate_pairs: list[tuple[int, int]],
    fit_half_window: int = FIT_HALF_WINDOW,
    threshold: float = RESIDUAL_THRESHOLD,
) -> tuple[list[str], np.ndarray]:
    labels = []
    widths = []
    for dataset_minus, dataset_plus in candidate_pairs:
        result = analyze_pair(
            datasets,
            metadata,
            dataset_minus,
            dataset_plus,
            fit_half_window=fit_half_window,
            threshold=threshold,
        )
        minus_label = result.band_minus_label or f"Band {dataset_minus + 1}"
        plus_label = result.band_plus_label or f"Band {dataset_plus + 1}"
        labels.append(f"{minus_label}/{plus_label}")
        widths.append(result.edt_width)
    return labels, np.asarray(widths, dtype=float)


def save_single_panel(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_panel_b(analysis: PairAnalysis, y_label: str) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    ax.axvspan(analysis.k_left, analysis.k_right, color="0.88", alpha=0.8)
    ax.plot(analysis.k, analysis.y_minus, lw=1.5, color="#b2182b", label=analysis.band_minus_label)
    ax.plot(analysis.k, analysis.y_plus, lw=1.5, color="#2166ac", label=analysis.band_plus_label)
    ax.scatter(
        [analysis.k[analysis.idx_minus], analysis.k[analysis.idx_plus]],
        [analysis.y_minus[analysis.idx_minus], analysis.y_plus[analysis.idx_plus]],
        s=18,
        color=["#b2182b", "#2166ac"],
        zorder=3,
    )
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("(b) Curvature branches for the selected pair", loc="left", fontweight="bold")
    ax.text(
        0.98,
        0.03,
        rf"{MODE_LABEL}, $\omega={MODE_FREQUENCY_CM:.2f}\,\mathrm{{cm^{{-1}}}}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    save_single_panel(fig, MAIN_DIR / "figure2_k3c60_panel_b_curvature.pdf")


def make_panel_c(analysis: PairAnalysis, y_label: str) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    ax.axvspan(analysis.k_core_left, analysis.k_core_right, color="0.90", alpha=0.9)
    ax.plot(analysis.k, analysis.y_minus, color="#b2182b", alpha=0.35, lw=1.0)
    ax.plot(analysis.k, analysis.y_plus, color="#2166ac", alpha=0.35, lw=1.0)
    ax.plot(analysis.k, analysis.fit_curve_minus, color="#b2182b", lw=1.8, label=f"{analysis.band_minus_label} fit")
    ax.plot(analysis.k, analysis.fit_curve_plus, color="#2166ac", lw=1.8, label=f"{analysis.band_plus_label} fit")
    ax.scatter(float(analysis.fit_minus["k_apex"]), float(analysis.fit_minus["y_apex"]), s=20, color="#b2182b")
    ax.scatter(float(analysis.fit_plus["k_apex"]), float(analysis.fit_plus["y_apex"]), s=20, color="#2166ac")
    ax.axvline(analysis.k0, color="0.25", ls="--", lw=1.0)
    ax.set_xlim(max(analysis.k.min(), analysis.k_core_left - 0.06), min(analysis.k.max(), analysis.k_core_right + 0.06))
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("(c) Local quadratic fit of paired branches", loc="left", fontweight="bold")
    ax.text(0.98, 0.04, rf"$k_0={analysis.k0:.3f}$", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    save_single_panel(fig, MAIN_DIR / "figure2_k3c60_panel_c_fit.pdf")


def make_panel_d(analysis: PairAnalysis) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    ax.axvspan(analysis.k_left, analysis.k_right, color="0.88", alpha=0.8, label=r"$\Delta k_{\rm EDT}$")
    ax.axvspan(analysis.k_core_left, analysis.k_core_right, color="0.78", alpha=0.45, label=r"$\Delta k_{\rm core}$")
    ax.plot(analysis.k, analysis.residual_minus, color="#b2182b", lw=1.5, label=f"{analysis.band_minus_label} residual")
    ax.plot(analysis.k, analysis.residual_plus, color="#2166ac", lw=1.5, label=f"{analysis.band_plus_label} residual")
    ax.axhline(analysis.threshold, color="0.25", ls="--", lw=1.1, label=rf"$R_{{\rm th}}={analysis.threshold:.2f}$")
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$R_{\rm quad}(k)$")
    ax.set_ylim(0.0, max(0.20, 1.05 * float(max(analysis.residual_minus.max(), analysis.residual_plus.max()))))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("(d) EDT window from the residual threshold", loc="left", fontweight="bold")
    ax.text(
        0.98,
        0.04,
        rf"$\Delta k_{{\rm EDT}}={analysis.edt_width:.3f}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    save_single_panel(fig, MAIN_DIR / "figure2_k3c60_panel_d_window.pdf")


def make_panel_e(
    threshold_scan: tuple[np.ndarray, np.ndarray],
    bootstrap_widths_values: np.ndarray,
    nominal_width: float,
    threshold_value: float,
) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    thresholds, widths = threshold_scan
    ax.plot(thresholds, widths, color="black", lw=1.5)
    ax.axvline(threshold_value, color="0.35", ls="--", lw=1.0)
    ax.axhline(nominal_width, color="#2166ac", ls=":", lw=1.0)
    ci_low, ci_high = np.percentile(bootstrap_widths_values, [2.5, 97.5])
    ax.fill_between(thresholds, ci_low, ci_high, color="#9ecae1", alpha=0.25, linewidth=0.0)
    ax.set_xlabel(r"$R_{\rm th}$")
    ax.set_ylabel(r"$\Delta k_{\rm EDT}$")
    ax.grid(alpha=0.3)
    ax.set_title("(e) Threshold sensitivity and bootstrap width", loc="left", fontweight="bold")
    ax.text(
        0.02,
        0.96,
        rf"95\% bootstrap CI: [{ci_low:.3f}, {ci_high:.3f}]",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    inset = ax.inset_axes([0.60, 0.17, 0.34, 0.32])
    inset.hist(bootstrap_widths_values, bins=20, color="#6baed6", edgecolor="white")
    inset.axvline(ci_low, color="0.25", ls="--", lw=0.9)
    inset.axvline(ci_high, color="0.25", ls="--", lw=0.9)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("bootstrap", fontsize=7)
    save_single_panel(fig, MAIN_DIR / "figure2_k3c60_panel_e_bootstrap.pdf")


def make_supplementary_figure(
    threshold_scan: tuple[np.ndarray, np.ndarray],
    fit_scan: tuple[np.ndarray, np.ndarray],
    pair_scan: tuple[list[str], np.ndarray],
    kgrid_scan: tuple[np.ndarray, np.ndarray],
    bootstrap_widths_values: np.ndarray,
    nominal_width: float,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.8))
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.ravel()

    thresholds, widths = threshold_scan
    ax_a.plot(thresholds, widths, color="black", lw=1.5)
    ax_a.axhline(nominal_width, color="#2166ac", ls=":", lw=1.0)
    ax_a.set_title("(a) Residual-threshold sweep", loc="left", fontweight="bold")
    ax_a.set_xlabel(r"$R_{\rm th}$")
    ax_a.set_ylabel(r"$\Delta k_{\rm EDT}$")
    ax_a.grid(alpha=0.3)

    ax_b.hist(bootstrap_widths_values, bins=24, color="#6baed6", edgecolor="white")
    ci_low, ci_high = np.percentile(bootstrap_widths_values, [2.5, 97.5])
    ax_b.axvline(ci_low, color="0.25", ls="--", lw=1.0)
    ax_b.axvline(ci_high, color="0.25", ls="--", lw=1.0)
    ax_b.set_title("(b) Bootstrap confidence interval", loc="left", fontweight="bold")
    ax_b.set_xlabel(r"$\Delta k_{\rm EDT}$")
    ax_b.set_ylabel("count")
    ax_b.grid(alpha=0.2)

    fit_windows, fit_widths = fit_scan
    ax_c.plot(fit_windows, fit_widths, marker="o", ms=3.5, color="#762a83")
    ax_c.set_title("(c) Fit-window sensitivity", loc="left", fontweight="bold")
    ax_c.set_xlabel("half-window (points)")
    ax_c.set_ylabel(r"$\Delta k_{\rm EDT}$")
    ax_c.grid(alpha=0.3)

    pair_labels, pair_widths = pair_scan
    ax_d.bar(np.arange(len(pair_labels)), pair_widths, color=["#d7301f", "#fc8d59", "#1f78b4", "#33a02c"][: len(pair_labels)])
    ax_d.set_title("(d) Band-pair selection", loc="left", fontweight="bold")
    ax_d.set_ylabel(r"$\Delta k_{\rm EDT}$")
    ax_d.set_xticks(np.arange(len(pair_labels)))
    ax_d.set_xticklabels(pair_labels, rotation=20, ha="right")
    ax_d.grid(axis="y", alpha=0.3)

    sample_counts, kgrid_widths = kgrid_scan
    ax_e.plot(sample_counts, kgrid_widths, marker="s", ms=4, color="#1b9e77")
    ax_e.set_title("(e) k-grid subsampling", loc="left", fontweight="bold")
    ax_e.set_xlabel("samples in analysis window")
    ax_e.set_ylabel(r"$\Delta k_{\rm EDT}$")
    ax_e.grid(alpha=0.3)

    ax_f.remove()

    fig.tight_layout()
    save_single_panel(fig, SUPP_DIR / "S5_threshold_bootstrap.pdf")


def write_summary(
    analysis: PairAnalysis,
    threshold_scan: tuple[np.ndarray, np.ndarray],
    fit_scan: tuple[np.ndarray, np.ndarray],
    pair_scan: tuple[list[str], np.ndarray],
    kgrid_scan: tuple[np.ndarray, np.ndarray],
    bootstrap_widths_values: np.ndarray,
    metadata: dict[str, object],
) -> None:
    ci_low, ci_high = np.percentile(bootstrap_widths_values, [2.5, 97.5])
    summary = {
        "data_file": str(CURVATURE_FILE.relative_to(ROOT)),
        "mode_label": MODE_LABEL,
        "mode_frequency_cm^-1": MODE_FREQUENCY_CM,
        "analysis_window_indices": [K_START, K_END],
        "fit_half_window_points": FIT_HALF_WINDOW,
        "residual_threshold": RESIDUAL_THRESHOLD,
        "selected_pair": {
            "dataset_minus": DATASET_MINUS,
            "dataset_plus": DATASET_PLUS,
            "band_minus_label": analysis.band_minus_label,
            "band_plus_label": analysis.band_plus_label,
        },
        "nominal_result": {
            "k0": analysis.k0,
            "k_left": analysis.k_left,
            "k_right": analysis.k_right,
            "edt_width": analysis.edt_width,
            "core_width": analysis.core_width,
        },
        "bootstrap": {
            "samples": int(bootstrap_widths_values.size),
            "width_mean": float(np.mean(bootstrap_widths_values)),
            "width_std": float(np.std(bootstrap_widths_values, ddof=1)),
            "width_ci_95": [float(ci_low), float(ci_high)],
        },
        "threshold_scan": {
            "thresholds": threshold_scan[0].tolist(),
            "widths": threshold_scan[1].tolist(),
        },
        "fit_window_scan": {
            "half_windows": fit_scan[0].tolist(),
            "widths": fit_scan[1].tolist(),
        },
        "band_pair_scan": {
            "pairs": pair_scan[0],
            "widths": pair_scan[1].tolist(),
        },
        "kgrid_scan": {
            "sample_counts": kgrid_scan[0].tolist(),
            "widths": kgrid_scan[1].tolist(),
        },
        "archived_y_label": clean_grace_label(str(metadata["y_label_raw"])),
        "path_labels": metadata["path_labels"],
        "displacement_amplitude_scan_available": False,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.linewidth": 0.6,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    datasets = parse_agr(CURVATURE_FILE)
    metadata = parse_agr_metadata(CURVATURE_FILE)
    y_label = r"$\kappa_{n\alpha}(k)$ (eV/\AA$^2$)"

    nominal = analyze_pair(datasets, metadata, DATASET_MINUS, DATASET_PLUS)
    thresholds = np.linspace(0.04, 0.18, 15)
    threshold_scan = threshold_sweep(datasets, metadata, thresholds)
    fit_scan = fit_window_sweep(datasets, metadata, np.arange(10, 29, 2))
    pair_scan = band_pair_sweep(
        datasets,
        metadata,
        candidate_pairs=[(205, 206), (206, 207), (207, 208), (208, 209)],
    )
    kgrid_scan = kgrid_sweep(datasets, metadata, stride_values=[1, 2, 3, 4])
    bootstrap_values = bootstrap_widths(nominal)

    make_panel_b(nominal, y_label)
    make_panel_c(nominal, y_label)
    make_panel_d(nominal)
    make_panel_e(threshold_scan, bootstrap_values, nominal.edt_width, RESIDUAL_THRESHOLD)
    make_supplementary_figure(
        threshold_scan,
        fit_scan,
        pair_scan,
        kgrid_scan,
        bootstrap_values,
        nominal.edt_width,
    )
    write_summary(nominal, threshold_scan, fit_scan, pair_scan, kgrid_scan, bootstrap_values, metadata)

    print("Saved:", MAIN_DIR / "figure2_k3c60_panel_b_curvature.pdf")
    print("Saved:", MAIN_DIR / "figure2_k3c60_panel_c_fit.pdf")
    print("Saved:", MAIN_DIR / "figure2_k3c60_panel_d_window.pdf")
    print("Saved:", MAIN_DIR / "figure2_k3c60_panel_e_bootstrap.pdf")
    print("Saved:", SUPP_DIR / "S5_threshold_bootstrap.pdf")
    print("Saved:", SUMMARY_JSON)


if __name__ == "__main__":
    main()
