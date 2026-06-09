#!/usr/bin/env python3
"""Compare AGR EDT curvature windows with EPW/Wannier-derived quantities.

This is a scaffold for the post-EPW comparison step. It already parses Grace
AGR datasets and EDT summary JSON files, and it can compute inside/outside
ratios when EPW quantities are exported to a simple CSV.

Expected future EPW CSV columns:
    k, g2, g_tr

where k is the 1D AGR path coordinate, g2 is a mode/band-resolved EPC quantity
such as |g_mnnu(k,q)|^2, and g_tr is the quantum-metric trace or another
geometry scalar interpolated from Wannier data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AgrDataset:
    label: str
    x: list[float]
    y: list[float]


@dataclass
class EdtWindow:
    k0: float
    k_left: float
    k_right: float
    delta_k_edt: float
    mode_label: str = ""
    material: str = ""


def parse_agr(path: Path) -> dict[str, AgrDataset]:
    datasets: dict[str, AgrDataset] = {}
    current = "unknown"
    xs: list[float] = []
    ys: list[float] = []

    def flush() -> None:
        nonlocal xs, ys
        if xs and ys:
            datasets[current] = AgrDataset(current, xs, ys)
        xs = []
        ys = []

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("@target"):
            flush()
            current = line.split()[-1]
            continue
        if line.startswith("@") or line.startswith("#"):
            continue
        if line == "&":
            flush()
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
            except ValueError:
                continue
    flush()
    return datasets


def load_edt_window(summary_path: Path) -> EdtWindow:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    nominal = data.get("nominal_result", {})
    k_left = float(nominal["k_left"])
    k_right = float(nominal["k_right"])
    return EdtWindow(
        k0=float(nominal["k0"]),
        k_left=k_left,
        k_right=k_right,
        delta_k_edt=float(nominal.get("edt_width", k_right - k_left)),
        mode_label=str(data.get("mode_label", "")),
        material=str(data.get("material", "")),
    )


def load_epw_quantity_csv(path: Path) -> dict[str, list[float]]:
    columns = {"k": [], "g2": [], "g_tr": []}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(columns) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing EPW CSV columns: {sorted(missing)}")
        for row in reader:
            for key in columns:
                value = row.get(key, "")
                columns[key].append(float(value) if value else math.nan)
    return columns


def linear_interpolate(x_src: list[float], y_src: list[float], x_new: list[float]) -> list[float]:
    if not x_src:
        return [math.nan for _ in x_new]
    pairs = sorted(zip(x_src, y_src), key=lambda item: item[0])
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    out = []
    j = 0
    for x in x_new:
        if x < xs[0] or x > xs[-1]:
            out.append(math.nan)
            continue
        while j + 1 < len(xs) and xs[j + 1] < x:
            j += 1
        if j + 1 == len(xs):
            out.append(ys[-1])
            continue
        x0, x1 = xs[j], xs[j + 1]
        y0, y1 = ys[j], ys[j + 1]
        if x1 == x0:
            out.append(y0)
        else:
            t = (x - x0) / (x1 - x0)
            out.append(y0 + t * (y1 - y0))
    return out


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def inside_outside_ratio(k: list[float], quantity: list[float], window: EdtWindow) -> tuple[float, float, float]:
    inside = [q for kk, q in zip(k, quantity) if window.k_left <= kk <= window.k_right]
    outside = [q for kk, q in zip(k, quantity) if kk < window.k_left or kk > window.k_right]
    inside_mean = mean(inside)
    outside_mean = mean(outside)
    ratio = inside_mean / outside_mean if outside_mean and math.isfinite(outside_mean) else math.nan
    return inside_mean, outside_mean, ratio


def make_overlay_plot(
    agr_dataset: AgrDataset,
    epw_k: list[float],
    epw_g2_on_agr: list[float],
    window: EdtWindow,
    output: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    ax1.plot(agr_dataset.x, agr_dataset.y, color="black", lw=1.4, label="AGR curvature")
    ax1.axvspan(window.k_left, window.k_right, color="tab:orange", alpha=0.18, label="EDT window")
    ax1.axvline(window.k0, color="tab:orange", lw=1.0)
    ax1.set_xlabel("k-path coordinate")
    ax1.set_ylabel("AGR curvature")

    ax2 = ax1.twinx()
    ax2.plot(agr_dataset.x, epw_g2_on_agr, color="tab:blue", lw=1.0, label="EPW |g|^2")
    ax2.set_ylabel("EPW |g|^2")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best", frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agr", type=Path, required=True, help="AGR curvature file.")
    parser.add_argument("--summary", type=Path, required=True, help="EDT summary JSON.")
    parser.add_argument("--epw-csv", type=Path, default=None, help="Future EPW quantity CSV with k,g2,g_tr columns.")
    parser.add_argument("--dataset", default=None, help="AGR dataset label, for example G0.S10. Defaults to first dataset.")
    parser.add_argument("--plot", type=Path, default=None, help="Optional overlay plot path.")
    args = parser.parse_args()

    datasets = parse_agr(args.agr)
    if not datasets:
        raise SystemExit(f"No datasets parsed from {args.agr}")
    dataset = datasets[args.dataset] if args.dataset else next(iter(datasets.values()))
    window = load_edt_window(args.summary)

    print(f"Material: {window.material or 'unknown'}")
    print(f"Mode label: {window.mode_label or 'unknown'}")
    print(f"AGR dataset: {dataset.label}")
    print(f"k0={window.k0:.8g}, k_left={window.k_left:.8g}, k_right={window.k_right:.8g}, delta_k_EDT={window.delta_k_edt:.8g}")

    if args.epw_csv is None:
        print("EPW CSV not provided yet. Expected columns for later comparison: k,g2,g_tr")
        return 0

    epw = load_epw_quantity_csv(args.epw_csv)
    g2_on_agr = linear_interpolate(epw["k"], epw["g2"], dataset.x)
    gtr_on_agr = linear_interpolate(epw["k"], epw["g_tr"], dataset.x)
    g2_inside, g2_outside, r_epc = inside_outside_ratio(dataset.x, g2_on_agr, window)
    gtr_inside, gtr_outside, r_geom = inside_outside_ratio(dataset.x, gtr_on_agr, window)

    print(f"<|g|^2>inside={g2_inside:.8g}, outside={g2_outside:.8g}, R_EPC={r_epc:.8g}")
    print(f"<g_tr>inside={gtr_inside:.8g}, outside={gtr_outside:.8g}, R_geom={r_geom:.8g}")

    if args.plot:
        make_overlay_plot(dataset, epw["k"], g2_on_agr, window, args.plot)
        print(f"Wrote plot: {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
