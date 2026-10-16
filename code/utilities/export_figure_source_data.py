from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIGURE_SOURCE_DIR = ROOT / "data" / "figure_source_data"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def export_fig1_summary() -> None:
    entries = []
    summaries = [
        ("K3C60", "208/209", ROOT / "data" / "processed" / "k3c60_edt_window_summary.json"),
        ("Na", "11/12", ROOT / "data" / "processed" / "S1_Na_summary.json"),
        ("V", "23/24", ROOT / "data" / "processed" / "S2_V_summary.json"),
        ("YBa2Cu3O7", "146/148", ROOT / "data" / "processed" / "S4_YBa2Cu3O7_summary.json"),
        ("LaH10", "34/36", ROOT / "data" / "processed" / "S5_LaH10_summary.json"),
    ]
    for material, pair, path in summaries:
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries.append(
            {
                "material": material,
                "pair": pair,
                "k0": payload["nominal_result"]["k0"],
                "delta_k_edt": payload["nominal_result"]["edt_width"],
                "bootstrap_ci_low": payload["bootstrap"]["width_ci_95"][0],
                "bootstrap_ci_high": payload["bootstrap"]["width_ci_95"][1],
            }
        )
    write_csv(FIGURE_SOURCE_DIR / "Fig1_source_data.csv", entries)
    write_csv(FIGURE_SOURCE_DIR / "SuppFig_S1_source_data.csv", entries)


def export_fig2_and_supp() -> None:
    fig2 = load_module("companion_fig2", ROOT / "code" / "plotting" / "make_fig2_k3c60_edt.py")
    datasets = fig2.parse_agr(fig2.CURVATURE_FILE)
    metadata = fig2.parse_agr_metadata(fig2.CURVATURE_FILE)
    analysis = fig2.analyze_pair(
        datasets,
        metadata,
        fig2.DATASET_MINUS,
        fig2.DATASET_PLUS,
        start_idx=fig2.K_START,
        end_idx=fig2.K_END,
        fit_half_window=fig2.FIT_HALF_WINDOW,
        threshold=fig2.RESIDUAL_THRESHOLD,
    )
    y_label = fig2.clean_grace_label(metadata["y_label_raw"])
    rows_b = [
        {
            "k": float(k),
            "kappa_minus": float(y_minus),
            "kappa_plus": float(y_plus),
            "window_left_k": analysis.k_left,
            "window_right_k": analysis.k_right,
            "y_label": y_label,
        }
        for k, y_minus, y_plus in zip(analysis.k, analysis.y_minus, analysis.y_plus)
    ]
    rows_c = [
        {
            "k": float(k),
            "kappa_minus": float(y_minus),
            "fit_minus": float(fit_minus),
            "kappa_plus": float(y_plus),
            "fit_plus": float(fit_plus),
            "k0": analysis.k0,
        }
        for k, y_minus, fit_minus, y_plus, fit_plus in zip(
            analysis.k,
            analysis.y_minus,
            analysis.fit_curve_minus,
            analysis.y_plus,
            analysis.fit_curve_plus,
        )
    ]
    rows_d = [
        {
            "k": float(k),
            "residual_minus": float(rm),
            "residual_plus": float(rp),
            "threshold": analysis.threshold,
        }
        for k, rm, rp in zip(analysis.k, analysis.residual_minus, analysis.residual_plus)
    ]
    threshold_values = np.linspace(0.04, 0.18, 15)
    threshold_x, threshold_y = fig2.threshold_sweep(datasets, metadata, threshold_values)
    bootstrap = fig2.bootstrap_widths(analysis)
    rows_e = [{"series": "threshold_scan", "x": float(x), "y": float(y)} for x, y in zip(threshold_x, threshold_y)]
    rows_e.extend(
        {"series": "bootstrap_width_samples", "x": idx, "y": float(y)}
        for idx, y in enumerate(bootstrap)
    )
    write_csv(FIGURE_SOURCE_DIR / "Fig2_panel_b_source_data.csv", rows_b)
    write_csv(FIGURE_SOURCE_DIR / "Fig2_panel_c_source_data.csv", rows_c)
    write_csv(FIGURE_SOURCE_DIR / "Fig2_panel_d_source_data.csv", rows_d)
    write_csv(FIGURE_SOURCE_DIR / "Fig2_panel_e_source_data.csv", rows_e)
    write_csv(FIGURE_SOURCE_DIR / "SuppFig_S2_source_data.csv", rows_b)
    write_csv(FIGURE_SOURCE_DIR / "SuppFig_S7_source_data.csv", rows_e)


def export_material_panels() -> None:
    mod = load_module("companion_material_panels", ROOT / "code" / "plotting" / "make_material_figure2_panels.py")
    datasets_by_path = {}
    file_map = {
        "Na": "SuppFig_S3_source_data.csv",
        "V": "SuppFig_S4_source_data.csv",
        "YBa2Cu3O7": "SuppFig_S5_source_data.csv",
        "LaH10": "SuppFig_S6_source_data.csv",
    }
    for config in mod.CONFIGS:
        datasets = datasets_by_path.setdefault(config.source_file, mod.parse_agr(config.source_file))
        analysis = mod.analyze_pair(config, datasets)
        rows = [
            {
                "k": float(k),
                "kappa_minus": float(y_minus),
                "fit_minus": float(fit_minus),
                "residual_minus": float(rm),
                "kappa_plus": float(y_plus),
                "fit_plus": float(fit_plus),
                "residual_plus": float(rp),
                "window_left_k": analysis.k_left,
                "window_right_k": analysis.k_right,
                "k0": analysis.k0,
            }
            for k, y_minus, fit_minus, rm, y_plus, fit_plus, rp in zip(
                analysis.k,
                analysis.y_minus,
                analysis.fit_curve_minus,
                analysis.residual_minus,
                analysis.y_plus,
                analysis.fit_curve_plus,
                analysis.residual_plus,
            )
        ]
        write_csv(FIGURE_SOURCE_DIR / file_map[config.key], rows)


def export_plaquette() -> None:
    plaquette_rows = []
    with (ROOT / "data" / "processed" / "plaquette_lambda_sweep.csv").open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        plaquette_rows.extend(reader)
    write_csv(FIGURE_SOURCE_DIR / "Fig3_source_data.csv", [dict(row) for row in plaquette_rows])
    write_csv(FIGURE_SOURCE_DIR / "Fig4_source_data.csv", [dict(row) for row in plaquette_rows])


def export_nv() -> None:
    metadata = {}
    with (ROOT / "data" / "processed" / "xy8_qre_nv_results.csv").open("r", encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle]
    start = None
    for idx, line in enumerate(lines):
        if line == "tau_s,N,eta_proxy":
            start = idx
            break
        if line and "," in line:
            key, value = line.split(",", 1)
            if key in {"scenario", "calibration_digest", "f_target_Hz", "tau_best_s", "N_best", "eta_best"}:
                metadata[key] = value.rstrip("\r")
    if start is None:
        raise RuntimeError("Could not find XY8 sweep table in xy8_qre_nv_results.csv")
    reader = csv.DictReader(lines[start:])
    nv_rows = []
    for row in reader:
        nv_rows.append({**metadata, **dict(row)})
    write_csv(FIGURE_SOURCE_DIR / "Fig5_source_data.csv", nv_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export figure-level source data.")
    parser.add_argument("--only", choices=["fig5"], default=None, help="Export only the named figure source data.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(FIGURE_SOURCE_DIR)
    if args.only == "fig5":
        export_nv()
        return
    export_fig1_summary()
    export_fig2_and_supp()
    export_material_panels()
    export_plaquette()
    export_nv()


if __name__ == "__main__":
    main()
