from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUPP_DIR = ROOT / "figures" / "supplementary"
SUMMARY_DIR = ROOT / "data" / "processed"

THRESHOLDS = np.linspace(0.04, 0.18, 15)
BOOTSTRAP_SAMPLES = 500
RNG_SEED = 12345
Y_LABEL = r"$\kappa_{n\alpha}(k)$ (eV/\AA$^2$)"


@dataclass(frozen=True)
class MaterialConfig:
    key: str
    figure_tag: str
    title: str
    mode_label: str
    source_file: Path
    band_figure: Path
    dataset_minus: int
    dataset_plus: int
    start_idx: int | None = None
    end_idx: int | None = None
    fit_half_window: int = 8
    threshold: float = 0.08
    manual_k_left: float | None = None
    manual_k_right: float | None = None
    manual_window_reason: str | None = None

    @property
    def output_prefix(self) -> str:
        return f"{self.figure_tag}_{self.key}"


@dataclass(frozen=True)
class PairAnalysis:
    config: MaterialConfig
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

    @property
    def auto_k_left(self) -> float:
        return float(self.k[self.left_edt])

    @property
    def auto_k_right(self) -> float:
        return float(self.k[self.right_edt - 1])

    @property
    def auto_edt_width(self) -> float:
        return float(self.auto_k_right - self.auto_k_left)

    @property
    def k_left(self) -> float:
        if self.config.manual_k_left is not None:
            return float(self.config.manual_k_left)
        return float(self.k[self.left_edt])

    @property
    def k_right(self) -> float:
        if self.config.manual_k_right is not None:
            return float(self.config.manual_k_right)
        return float(self.k[self.right_edt - 1])

    @property
    def edt_width(self) -> float:
        return float(self.k_right - self.k_left)

    @property
    def k0(self) -> float:
        if self.config.manual_k_left is not None and self.config.manual_k_right is not None:
            return 0.5 * (float(self.config.manual_k_left) + float(self.config.manual_k_right))
        return 0.5 * (float(self.fit_minus["k_apex"]) + float(self.fit_plus["k_apex"]))

    @property
    def fit_k0(self) -> float:
        return 0.5 * (float(self.fit_minus["k_apex"]) + float(self.fit_plus["k_apex"]))

    @property
    def k_core_left(self) -> float:
        if self.config.manual_k_left is not None:
            return float(self.config.manual_k_left)
        return float(self.k[self.left_core])

    @property
    def k_core_right(self) -> float:
        if self.config.manual_k_right is not None:
            return float(self.config.manual_k_right)
        return float(self.k[self.right_core - 1])

    @property
    def core_width(self) -> float:
        return float(self.k_core_right - self.k_core_left)


CONFIGS = [
    MaterialConfig(
        key="Na",
        figure_tag="S1",
        title="Na",
        mode_label="m5",
        source_file=ROOT / "data" / "na" / "Na-m5.agr",
        band_figure=SUPP_DIR / "Na_bands.pdf",
        dataset_minus=10,
        dataset_plus=11,
        fit_half_window=9,
    ),
    MaterialConfig(
        key="V",
        figure_tag="S2",
        title="V",
        mode_label="F1",
        source_file=ROOT / "data" / "v" / "V-EP-F1.agr",
        band_figure=SUPP_DIR / "V-bands.pdf",
        dataset_minus=22,
        dataset_plus=23,
        fit_half_window=8,
    ),
    MaterialConfig(
        key="YBa2Cu3O7",
        figure_tag="S4",
        title=r"YBa$_2$Cu$_3$O$_7$",
        mode_label="m33",
        source_file=ROOT / "data" / "ybacu3o7" / "YBa2Cu3O7_m33.agr",
        band_figure=SUPP_DIR / "YBa2Cu3O7_bands.pdf",
        dataset_minus=146,
        dataset_plus=147,
        fit_half_window=5,
        manual_k_left=1.062522,
        manual_k_right=1.339270,
        manual_window_reason="User-corrected Band 147/148 EDT window on the T-X AGR interval.",
    ),
    MaterialConfig(
        key="LaH10",
        figure_tag="S5",
        title=r"LaH$_{10}$",
        mode_label="m16",
        source_file=ROOT / "data" / "lah10" / "tmp_LaH10_m16.agr",
        band_figure=SUPP_DIR / "LaH10-300GPa-bands.pdf",
        dataset_minus=33,
        dataset_plus=35,
        fit_half_window=4,
    ),
]


def parse_agr(filename: Path) -> list[np.ndarray]:
    datasets: list[np.ndarray] = []
    current: list[list[float]] = []
    for line in filename.read_text(errors="ignore").splitlines():
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
    text = filename.read_text(errors="ignore")
    y_label_match = re.search(r'@\s+yaxis\s+label\s+"([^"]+)"', text)
    x_strings = re.findall(r'@\s+string def "([^"]+)"', text)
    dataset_labels: dict[int, str] = {}
    for idx, label in re.findall(r'@\s+s(\d+)\s+legend\s+"([^"]*)"', text):
        dataset_labels[int(idx)] = label.strip()
    return {
        "y_label_raw": y_label_match.group(1) if y_label_match else "",
        "path_labels": x_strings,
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


def analyze_pair(config: MaterialConfig, datasets: list[np.ndarray]) -> PairAnalysis:
    data_slice = slice(config.start_idx, config.end_idx)
    k = datasets[config.dataset_minus][data_slice, 0]
    y_minus = datasets[config.dataset_minus][data_slice, 1]
    y_plus = datasets[config.dataset_plus][data_slice, 1]

    if config.manual_k_left is not None and config.manual_k_right is not None:
        manual_mask = (k >= config.manual_k_left) & (k <= config.manual_k_right)
        manual_indices = np.flatnonzero(manual_mask)
        if manual_indices.size == 0:
            raise ValueError(f"Manual window for {config.key} contains no AGR samples.")
        idx_minus = int(manual_indices[np.argmin(y_minus[manual_mask])])
        idx_plus = int(manual_indices[np.argmax(y_plus[manual_mask])])
    else:
        idx_minus = int(np.argmin(y_minus))
        idx_plus = int(np.argmax(y_plus))

    k_minus_fit, y_minus_fit = extract_local_window(k, y_minus, idx_minus, config.fit_half_window)
    k_plus_fit, y_plus_fit = extract_local_window(k, y_plus, idx_plus, config.fit_half_window)
    fit_minus = fit_local_parabola(k_minus_fit, y_minus_fit)
    fit_plus = fit_local_parabola(k_plus_fit, y_plus_fit)

    fit_curve_minus = np.polyval(fit_minus["coeffs"], k)
    fit_curve_plus = np.polyval(fit_plus["coeffs"], k)
    residual_minus = np.abs(y_minus - fit_curve_minus) / max(np.max(np.abs(y_minus)), 1e-14)
    residual_plus = np.abs(y_plus - fit_curve_plus) / max(np.max(np.abs(y_plus)), 1e-14)

    left_minus, right_minus = contiguous_true_region(residual_minus <= config.threshold, idx_minus)
    left_plus, right_plus = contiguous_true_region(residual_plus <= config.threshold, idx_plus)
    left_edt = min(left_minus, left_plus)
    right_edt = max(right_minus, right_plus)
    left_core = max(left_minus, left_plus)
    right_core = min(right_minus, right_plus)
    if right_core <= left_core:
        left_core, right_core = left_edt, right_edt

    return PairAnalysis(
        config=config,
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
    )


def bootstrap_widths(analysis: PairAnalysis, n_boot: int = BOOTSTRAP_SAMPLES, seed: int = RNG_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k_minus_fit, y_minus_fit = extract_local_window(
        analysis.k, analysis.y_minus, analysis.idx_minus, analysis.config.fit_half_window
    )
    k_plus_fit, y_plus_fit = extract_local_window(
        analysis.k, analysis.y_plus, analysis.idx_plus, analysis.config.fit_half_window
    )
    base_fit_minus = fit_local_parabola(k_minus_fit, y_minus_fit)
    base_fit_plus = fit_local_parabola(k_plus_fit, y_plus_fit)
    resid_minus_local = y_minus_fit - np.polyval(base_fit_minus["coeffs"], k_minus_fit)
    resid_plus_local = y_plus_fit - np.polyval(base_fit_plus["coeffs"], k_plus_fit)
    widths = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        minus_idx = rng.integers(0, resid_minus_local.size, size=resid_minus_local.size)
        plus_idx = rng.integers(0, resid_plus_local.size, size=resid_plus_local.size)
        y_minus_boot = np.polyval(base_fit_minus["coeffs"], k_minus_fit) + resid_minus_local[minus_idx]
        y_plus_boot = np.polyval(base_fit_plus["coeffs"], k_plus_fit) + resid_plus_local[plus_idx]
        fit_minus = fit_local_parabola(k_minus_fit, y_minus_boot)
        fit_plus = fit_local_parabola(k_plus_fit, y_plus_boot)
        fit_curve_minus = np.polyval(fit_minus["coeffs"], analysis.k)
        fit_curve_plus = np.polyval(fit_plus["coeffs"], analysis.k)
        residual_minus = np.abs(analysis.y_minus - fit_curve_minus) / max(np.max(np.abs(analysis.y_minus)), 1e-14)
        residual_plus = np.abs(analysis.y_plus - fit_curve_plus) / max(np.max(np.abs(analysis.y_plus)), 1e-14)
        left_minus, right_minus = contiguous_true_region(residual_minus <= analysis.config.threshold, analysis.idx_minus)
        left_plus, right_plus = contiguous_true_region(residual_plus <= analysis.config.threshold, analysis.idx_plus)
        left_edt = min(left_minus, left_plus)
        right_edt = max(right_minus, right_plus)
        widths[idx] = float(analysis.k[right_edt - 1] - analysis.k[left_edt])
    return widths


def threshold_sweep(config: MaterialConfig, datasets: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    widths = []
    for threshold in THRESHOLDS:
        sweep_config = MaterialConfig(**{**config.__dict__, "threshold": float(threshold)})
        result = analyze_pair(sweep_config, datasets)
        widths.append(result.auto_edt_width)
    return THRESHOLDS, np.asarray(widths, dtype=float)


def save_single_panel(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def panel_paths(config: MaterialConfig) -> dict[str, Path]:
    prefix = config.output_prefix
    return {
        "b": SUPP_DIR / f"{prefix}_panel_b_curvature.pdf",
        "c": SUPP_DIR / f"{prefix}_panel_c_fit.pdf",
        "d": SUPP_DIR / f"{prefix}_panel_d_window.pdf",
        "e": SUPP_DIR / f"{prefix}_panel_e_bootstrap.pdf",
        "summary": SUMMARY_DIR / f"{prefix}_summary.json",
    }


def make_panel_b(config: MaterialConfig, analysis: PairAnalysis) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    ax.axvspan(analysis.k_left, analysis.k_right, color="0.88", alpha=0.8)
    ax.plot(analysis.k, analysis.y_minus, lw=1.5, color="#b2182b", label=f"Band {config.dataset_minus + 1}")
    ax.plot(analysis.k, analysis.y_plus, lw=1.5, color="#2166ac", label=f"Band {config.dataset_plus + 1}")
    ax.scatter(
        [analysis.k[analysis.idx_minus], analysis.k[analysis.idx_plus]],
        [analysis.y_minus[analysis.idx_minus], analysis.y_plus[analysis.idx_plus]],
        s=18,
        color=["#b2182b", "#2166ac"],
        zorder=3,
    )
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(Y_LABEL)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("(a) Curvature branches for the selected pair", loc="left", fontweight="bold")
    ax.text(
        0.98,
        0.03,
        f"{config.title}, mode {config.mode_label}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    save_single_panel(fig, panel_paths(config)["b"])


def make_panel_c(config: MaterialConfig, analysis: PairAnalysis) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    ax.axvspan(analysis.k_core_left, analysis.k_core_right, color="0.90", alpha=0.9)
    ax.plot(analysis.k, analysis.y_minus, color="#b2182b", alpha=0.35, lw=1.0)
    ax.plot(analysis.k, analysis.y_plus, color="#2166ac", alpha=0.35, lw=1.0)
    ax.plot(analysis.k, analysis.fit_curve_minus, color="#b2182b", lw=1.8, label=f"Band {config.dataset_minus + 1} fit")
    ax.plot(analysis.k, analysis.fit_curve_plus, color="#2166ac", lw=1.8, label=f"Band {config.dataset_plus + 1} fit")
    ax.scatter(float(analysis.fit_minus["k_apex"]), float(analysis.fit_minus["y_apex"]), s=20, color="#b2182b")
    ax.scatter(float(analysis.fit_plus["k_apex"]), float(analysis.fit_plus["y_apex"]), s=20, color="#2166ac")
    ax.axvline(analysis.k0, color="0.25", ls="--", lw=1.0)
    ax.set_xlim(max(analysis.k.min(), analysis.k_core_left - 0.06), min(analysis.k.max(), analysis.k_core_right + 0.06))
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(Y_LABEL)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("(b) Local quadratic fit of paired branches", loc="left", fontweight="bold")
    ax.text(0.98, 0.04, rf"$k_0={analysis.k0:.3f}$", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    save_single_panel(fig, panel_paths(config)["c"])


def make_panel_d(config: MaterialConfig, analysis: PairAnalysis) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    ax.axvspan(analysis.k_left, analysis.k_right, color="0.88", alpha=0.8, label=r"$\Delta k_{\rm EDT}$")
    ax.axvspan(analysis.k_core_left, analysis.k_core_right, color="0.78", alpha=0.45, label=r"$\Delta k_{\rm core}$")
    ax.plot(analysis.k, analysis.residual_minus, color="#b2182b", lw=1.5, label=f"Band {config.dataset_minus + 1} residual")
    ax.plot(analysis.k, analysis.residual_plus, color="#2166ac", lw=1.5, label=f"Band {config.dataset_plus + 1} residual")
    ax.axhline(config.threshold, color="0.25", ls="--", lw=1.1, label=rf"$R_{{\rm th}}={config.threshold:.2f}$")
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$R_{\rm quad}(k)$")
    ymax = 1.05 * float(max(analysis.residual_minus.max(), analysis.residual_plus.max()))
    ax.set_ylim(0.0, max(0.20, ymax))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("(c) EDT window from the residual threshold", loc="left", fontweight="bold")
    ax.text(
        0.98,
        0.04,
        rf"$\Delta k_{{\rm EDT}}={analysis.edt_width:.3f}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    save_single_panel(fig, panel_paths(config)["d"])


def make_panel_e(config: MaterialConfig, threshold_scan_values: tuple[np.ndarray, np.ndarray], bootstrap_values: np.ndarray, nominal_width: float) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    thresholds, widths = threshold_scan_values
    ax.plot(thresholds, widths, color="black", lw=1.5)
    ax.axvline(config.threshold, color="0.35", ls="--", lw=1.0)
    ax.axhline(nominal_width, color="#2166ac", ls=":", lw=1.0)
    ci_low, ci_high = np.percentile(bootstrap_values, [2.5, 97.5])
    ax.fill_between(thresholds, ci_low, ci_high, color="#9ecae1", alpha=0.25, linewidth=0.0)
    ax.set_xlabel(r"$R_{\rm th}$")
    ax.set_ylabel(r"$\Delta k_{\rm EDT}$")
    ax.grid(alpha=0.3)
    ax.set_title("(d) Threshold sensitivity and bootstrap width", loc="left", fontweight="bold")
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
    inset.hist(bootstrap_values, bins=20, color="#6baed6", edgecolor="white")
    inset.axvline(ci_low, color="0.25", ls="--", lw=0.9)
    inset.axvline(ci_high, color="0.25", ls="--", lw=0.9)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("bootstrap", fontsize=7)
    save_single_panel(fig, panel_paths(config)["e"])


def write_summary(config: MaterialConfig, analysis: PairAnalysis, threshold_scan_values: tuple[np.ndarray, np.ndarray], bootstrap_values: np.ndarray, metadata: dict[str, object]) -> None:
    ci_low, ci_high = np.percentile(bootstrap_values, [2.5, 97.5])
    summary = {
        "material": config.key,
        "title": config.title,
        "mode_label": config.mode_label,
        "data_file": str(config.source_file.relative_to(ROOT)),
        "band_figure": str(config.band_figure.relative_to(ROOT)),
        "dataset_minus": config.dataset_minus,
        "dataset_plus": config.dataset_plus,
        "analysis_window_indices": [config.start_idx, config.end_idx],
        "fit_half_window_points": config.fit_half_window,
        "residual_threshold": config.threshold,
        "nominal_result": {
            "k0": analysis.k0,
            "k_left": analysis.k_left,
            "k_right": analysis.k_right,
            "edt_width": analysis.edt_width,
            "core_width": analysis.core_width,
        },
        "residual_threshold_result": {
            "k0_from_quadratic_apices": analysis.fit_k0,
            "k_left": analysis.auto_k_left,
            "k_right": analysis.auto_k_right,
            "edt_width": analysis.auto_edt_width,
            "core_width": float(analysis.k[analysis.right_core - 1] - analysis.k[analysis.left_core]),
        },
        "local_quadratic_fit": {
            "minus": {
                "dataset": config.dataset_minus,
                "band": config.dataset_minus + 1,
                "fit_center_k": float(analysis.k[analysis.idx_minus]),
                "k_apex": float(analysis.fit_minus["k_apex"]),
                "y_apex": float(analysis.fit_minus["y_apex"]),
                "coeffs": [float(value) for value in analysis.fit_minus["coeffs"]],
            },
            "plus": {
                "dataset": config.dataset_plus,
                "band": config.dataset_plus + 1,
                "fit_center_k": float(analysis.k[analysis.idx_plus]),
                "k_apex": float(analysis.fit_plus["k_apex"]),
                "y_apex": float(analysis.fit_plus["y_apex"]),
                "coeffs": [float(value) for value in analysis.fit_plus["coeffs"]],
            },
            "fit_half_window_points": config.fit_half_window,
        },
        "bootstrap": {
            "samples": int(bootstrap_values.size),
            "width_mean": float(np.mean(bootstrap_values)),
            "width_std": float(np.std(bootstrap_values, ddof=1)),
            "width_ci_95": [float(ci_low), float(ci_high)],
            "width_definition": "residual-threshold EDT width from the recalculated local quadratic fit",
        },
        "threshold_scan": {
            "thresholds": threshold_scan_values[0].tolist(),
            "widths": threshold_scan_values[1].tolist(),
            "width_definition": "residual-threshold EDT width from the recalculated local quadratic fit",
        },
        "archived_y_label": clean_grace_label(str(metadata["y_label_raw"])),
        "path_labels": metadata["path_labels"],
    }
    if config.manual_k_left is not None and config.manual_k_right is not None:
        summary["manual_window_override"] = {
            "reason": config.manual_window_reason,
            "dataset_minus": config.dataset_minus,
            "dataset_plus": config.dataset_plus,
            "band_minus": config.dataset_minus + 1,
            "band_plus": config.dataset_plus + 1,
            "k_left": config.manual_k_left,
            "k_right": config.manual_k_right,
            "corrected_k0_rule": "midpoint of corrected [k_left, k_right] interval",
        }
    panel_paths(config)["summary"].write_text(json.dumps(summary, indent=2))


def render_material(config: MaterialConfig) -> None:
    datasets = parse_agr(config.source_file)
    metadata = parse_agr_metadata(config.source_file)
    analysis = analyze_pair(config, datasets)
    threshold_scan_values = threshold_sweep(config, datasets)
    bootstrap_values = bootstrap_widths(analysis)
    make_panel_b(config, analysis)
    make_panel_c(config, analysis)
    make_panel_d(config, analysis)
    make_panel_e(config, threshold_scan_values, bootstrap_values, analysis.edt_width)
    write_summary(config, analysis, threshold_scan_values, bootstrap_values, metadata)


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
    SUPP_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    for config in CONFIGS:
        render_material(config)
        paths = panel_paths(config)
        print("Saved:", paths["b"])
        print("Saved:", paths["c"])
        print("Saved:", paths["d"])
        print("Saved:", paths["e"])
        print("Saved:", paths["summary"])


if __name__ == "__main__":
    main()
