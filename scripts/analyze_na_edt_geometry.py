#!/usr/bin/env python3
"""Analyze Na Wannier/EDT geometry outputs and make report plots."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path("epw_analysis") / "Na"
EPW_ROOT = Path("epw_runs") / "Na"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f(value: str | float | int | None) -> float:
    if value in (None, ""):
        return math.nan
    return float(value)


def rows_by_band(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    out: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault(int(row["band"]), []).append(row)
    return out


def plot_bands(rows: list[dict[str, str]], summary: dict[str, Any], output: Path) -> None:
    grouped = rows_by_band(rows)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for band, band_rows in sorted(grouped.items()):
        x = [f(row["k_coord"]) for row in band_rows]
        y = [f(row["energy"]) for row in band_rows]
        ax.plot(x, y, lw=1.1, label=f"Band {band}")
    window = summary["edt_window"]
    ax.axvspan(window["k_left"], window["k_right"], color="tab:orange", alpha=0.16, label="EDT window")
    ax.axvline(window["k0"], color="tab:orange", lw=1.0, alpha=0.9)
    ax.set_xlabel("AGR path coordinate")
    ax.set_ylabel("Wannier-interpolated energy (eV)")
    ax.set_title("Na Wannier Bands on AGR Path")
    ax.legend(ncols=4, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_metric(rows: list[dict[str, str]], summary: dict[str, Any], output: Path) -> None:
    grouped = rows_by_band(rows)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for band, band_rows in sorted(grouped.items()):
        x = [f(row["k_coord"]) for row in band_rows]
        y = [f(row["quantum_metric_parallel"]) for row in band_rows]
        ax.plot(x, y, lw=1.0, label=f"Band {band}")
    window = summary["edt_window"]
    ax.axvspan(window["k_left"], window["k_right"], color="tab:orange", alpha=0.18, label="EDT window")
    ax.axvline(window["k0"], color="tab:orange", lw=1.0, alpha=0.9)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("AGR path coordinate")
    ax.set_ylabel("g_parallel,n(k)")
    ax.set_title("Na Path-Projected Quantum Metric")
    ax.legend(ncols=4, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_ratios(rows: list[dict[str, str]], output: Path) -> None:
    bands = [int(row["band"]) for row in rows]
    ratios = [f(row["inside_outside_metric_ratio"]) for row in rows]
    colors = ["#2f6f8f" if ratio < 1.0 else "#b45f3c" for ratio in ratios]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.bar(bands, ratios, color=colors, width=0.72)
    ax.axhline(1.0, color="black", lw=1.0, ls="--")
    ax.set_xlabel("Wannier band")
    ax.set_ylabel("Inside / outside metric mean")
    ax.set_title("Na EDT-Window Quantum-Metric Ratios")
    ax.set_xticks(bands)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def epc_status() -> dict[str, Any]:
    patterns = ("epb", "epmat", "eph", "lambda", "linewidth", "gamma", "specfun", "selfen", "gkk")
    files = []
    for path in EPW_ROOT.rglob("*"):
        if path.is_file() and any(pattern in path.name.lower() for pattern in patterns):
            files.append({"path": str(path), "bytes": path.stat().st_size})
    return {
        "material": "Na",
        "status": "blocked_missing_epw_epc_outputs" if not files else "candidate_files_found",
        "candidate_epc_files": files,
        "outputs_dir_has_files": any((EPW_ROOT / "outputs").iterdir()) if (EPW_ROOT / "outputs").exists() else False,
        "save_dir_has_files": any((EPW_ROOT / "save").iterdir()) if (EPW_ROOT / "save").exists() else False,
        "next_action": (
            "Run or stage EPW EPC outputs such as epw.out plus epb/epmat/lambda/linewidth/specfun files, "
            "then compute EPC inside/outside ratios on the same EDT window."
        ),
    }


def write_markdown(
    path: Path,
    qc: dict[str, Any],
    summary: dict[str, Any],
    comparison_rows: list[dict[str, str]],
    epc: dict[str, Any],
) -> None:
    wout = qc["wout"]
    win = qc["win"]
    lines = [
        "# Na EDT Geometry Analysis",
        "",
        "## Wannier QC",
        f"- Required files present: {all(qc['required_files_available'].values())}",
        f"- num_wann: {win['num_wann']}",
        f"- projections: {', '.join(win['projections'])}",
        f"- disentanglement window: [{win['dis_win_min_ev']}, {win['dis_win_max_ev']}] eV",
        f"- frozen window: [{win['dis_froz_min_ev']}, {win['dis_froz_max_ev']}] eV",
        f"- final spreads Ang^2: {', '.join(f'{v:.6g}' for v in wout['final_spreads_ang2'])}",
        f"- Omega Total: {wout['omega_total_ang2']:.9g} Ang^2",
        f"- normal Wannier90 termination: {wout['normal_termination']}",
        f"- explicit convergence message: {wout['explicit_convergence_message']}",
        "",
        "Warning: Wannier90 terminated normally, but no explicit convergence message was parsed and the run reached "
        f"cycle {wout['max_cycle']}. Treat the spreads as usable QC data, not as proof of strict minimization convergence.",
        "",
        "## Geometry Summary",
        f"- EDT window: k = [{summary['edt_window']['k_left']}, {summary['edt_window']['k_right']}], k0 = {summary['edt_window']['k0']}",
        f"- inside quantum metric mean, all bands: {summary['inside_quantum_metric_parallel_mean_all_bands']:.8g}",
        f"- outside quantum metric mean, all bands: {summary['outside_quantum_metric_parallel_mean_all_bands']:.8g}",
        f"- inside/outside ratio, all bands: {summary['inside_outside_metric_ratio_all_bands']:.8g}",
        "",
        "Band-resolved ratios:",
    ]
    for row in comparison_rows:
        lines.append(
            f"- band {row['band']}: inside={f(row['inside_quantum_metric_parallel_mean']):.8g}, "
            f"outside={f(row['outside_quantum_metric_parallel_mean']):.8g}, "
            f"ratio={f(row['inside_outside_metric_ratio']):.8g}"
        )
    lines.extend(
        [
            "",
            "Control-result note: For Na, the EDT curvature window is not where the path-projected quantum metric is largest. "
            "This should be treated as a control result, not as a failure.",
            "",
            "## EPC Status",
            f"- status: {epc['status']}",
            f"- outputs/ has files: {epc['outputs_dir_has_files']}",
            f"- save/ has files: {epc['save_dir_has_files']}",
            f"- next action: {epc['next_action']}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    qc = read_json(ROOT / "wannier_qc.json")
    summary = read_json(ROOT / "edt_geometry_summary.json")
    comparison_rows = read_csv(ROOT / "edt_geometry_comparison.csv")
    metric_rows = read_csv(ROOT / "quantum_metric_on_agr_path.csv")
    band_rows = read_csv(ROOT / "wannier_bands_on_agr_path.csv")

    plot_bands(band_rows, summary, ROOT / "fig_na_wannier_bands_on_agr_path.png")
    plot_metric(metric_rows, summary, ROOT / "fig_na_quantum_metric_parallel.png")
    plot_ratios(comparison_rows, ROOT / "fig_na_metric_inside_outside_ratios.png")

    epc = epc_status()
    (ROOT / "epc_extraction_status.json").write_text(json.dumps(epc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    analysis = {
        "material": "Na",
        "qc_confirmed": {
            "required_files_present": all(qc["required_files_available"].values()),
            "num_wann": qc["win"]["num_wann"],
            "projections": qc["win"]["projections"],
            "disentanglement_window_ev": [qc["win"]["dis_win_min_ev"], qc["win"]["dis_win_max_ev"]],
            "frozen_window_ev": [qc["win"]["dis_froz_min_ev"], qc["win"]["dis_froz_max_ev"]],
            "final_spreads_ang2": qc["wout"]["final_spreads_ang2"],
            "omega_total_ang2": qc["wout"]["omega_total_ang2"],
            "normal_termination": qc["wout"]["normal_termination"],
            "convergence_warning": not qc["wout"]["explicit_convergence_message"],
        },
        "geometry_summary": {
            "inside_mean_all_bands": summary["inside_quantum_metric_parallel_mean_all_bands"],
            "outside_mean_all_bands": summary["outside_quantum_metric_parallel_mean_all_bands"],
            "inside_outside_ratio_all_bands": summary["inside_outside_metric_ratio_all_bands"],
            "band_resolved": [
                {
                    "band": int(row["band"]),
                    "inside_mean": f(row["inside_quantum_metric_parallel_mean"]),
                    "outside_mean": f(row["outside_quantum_metric_parallel_mean"]),
                    "inside_outside_ratio": f(row["inside_outside_metric_ratio"]),
                }
                for row in comparison_rows
            ],
        },
        "note": "For Na, the EDT curvature window is not where the path-projected quantum metric is largest. Treat this as a control result, not as a failure.",
        "plots": {
            "bands": str(ROOT / "fig_na_wannier_bands_on_agr_path.png"),
            "metric": str(ROOT / "fig_na_quantum_metric_parallel.png"),
            "ratios": str(ROOT / "fig_na_metric_inside_outside_ratios.png"),
        },
        "epc_extraction": epc,
    }
    (ROOT / "na_edt_geometry_analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(ROOT / "na_edt_geometry_analysis.md", qc, summary, comparison_rows, epc)
    print("Wrote Na EDT geometry analysis and plots")
    print(f"Inside mean all bands: {summary['inside_quantum_metric_parallel_mean_all_bands']:.8g}")
    print(f"Outside mean all bands: {summary['outside_quantum_metric_parallel_mean_all_bands']:.8g}")
    print(f"Inside/outside ratio all bands: {summary['inside_outside_metric_ratio_all_bands']:.8g}")
    print(f"EPC extraction status: {epc['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
