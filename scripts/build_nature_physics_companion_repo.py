from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from textwrap import dedent

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[1]
DEST_ROOT = SOURCE_ROOT.parent / "EDT_NaturePhysics_Data_Code"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(
        src,
        dst,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache"),
    )


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict | list) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def patch_script_text(name: str, text: str) -> str:
    if name == "make_fig2_k3c60_edt.py":
        text = text.replace(
            'ROOT = Path(__file__).resolve().parents[1]',
            'ROOT = Path(__file__).resolve().parents[2]',
        )
        text = text.replace(
            'CURVATURE_FILE = ROOT / "data" / "k3c60" / "tmp.agr"',
            'CURVATURE_FILE = ROOT / "data" / "raw" / "k3c60" / "tmp.agr"',
        )
        text = text.replace(
            'MAIN_DIR = ROOT / "figures" / "main"',
            'MAIN_DIR = ROOT / "outputs" / "reproduced_figures"',
        )
        text = text.replace(
            'SUPP_DIR = ROOT / "figures" / "supplementary"',
            'SUPP_DIR = ROOT / "outputs" / "reproduced_figures"',
        )
        text = text.replace(
            'SUMMARY_JSON = ROOT / "data" / "k3c60" / "k3c60_edt_window_summary.json"',
            'SUMMARY_JSON = ROOT / "data" / "processed" / "k3c60_edt_window_summary.json"',
        )
        return text

    if name == "make_material_figure2_panels.py":
        text = text.replace(
            'ROOT = Path(__file__).resolve().parents[1]',
            'ROOT = Path(__file__).resolve().parents[2]',
        )
        text = text.replace(
            'SUPP_DIR = ROOT / "figures" / "supplementary"',
            'SUPP_DIR = ROOT / "outputs" / "reproduced_figures"',
        )
        text = text.replace('ROOT / "data" / "na"', 'ROOT / "data" / "raw" / "na"')
        text = text.replace('ROOT / "data" / "v"', 'ROOT / "data" / "raw" / "v"')
        text = text.replace('ROOT / "data" / "ybacu3o7"', 'ROOT / "data" / "raw" / "ybacu3o7"')
        text = text.replace('ROOT / "data" / "lah10"', 'ROOT / "data" / "raw" / "lah10"')
        text = text.replace(
            'band_figure=SUPP_DIR / "Na_bands.pdf"',
            'band_figure=ROOT / "outputs" / "reproduced_figures" / "Na_bands.pdf"',
        )
        text = text.replace(
            'band_figure=SUPP_DIR / "V-bands.pdf"',
            'band_figure=ROOT / "outputs" / "reproduced_figures" / "V-bands.pdf"',
        )
        text = text.replace(
            'band_figure=SUPP_DIR / "YBa2Cu3O7_bands.pdf"',
            'band_figure=ROOT / "outputs" / "reproduced_figures" / "YBa2Cu3O7_bands.pdf"',
        )
        text = text.replace(
            'band_figure=SUPP_DIR / "LaH10-300GPa-bands.pdf"',
            'band_figure=ROOT / "outputs" / "reproduced_figures" / "LaH10-300GPa-bands.pdf"',
        )
        return text

    if name == "make_figS4_material_controls.py":
        text = text.replace(
            "from pathlib import Path\n\nimport matplotlib.pyplot as plt",
            dedent(
                """
                from pathlib import Path
                import sys

                import matplotlib.pyplot as plt
                """
            ).strip("\n"),
        )
        text = text.replace(
            "from edt.curvature import compute_quadratic_residual, fit_quadratic_pair",
            dedent(
                """
                ROOT = Path(__file__).resolve().parents[2]
                if str(ROOT / "code" / "utilities") not in sys.path:
                    sys.path.insert(0, str(ROOT / "code" / "utilities"))

                from edt.curvature import compute_quadratic_residual, fit_quadratic_pair
                """
            ).strip("\n"),
        )
        text = text.replace(
            'ROOT = Path(__file__).resolve().parents[1]',
            'ROOT = Path(__file__).resolve().parents[2]',
        )
        text = text.replace(
            'FIGURE_PATH = ROOT / "figures" / "supplementary" / "S4_material_controls.pdf"',
            'FIGURE_PATH = ROOT / "outputs" / "reproduced_figures" / "S4_material_controls.pdf"',
        )
        text = text.replace('ROOT / "data" / "k3c60"', 'ROOT / "data" / "raw" / "k3c60"')
        text = text.replace('ROOT / "data" / "v"', 'ROOT / "data" / "raw" / "v"')
        text = text.replace('ROOT / "data" / "na"', 'ROOT / "data" / "raw" / "na"')
        text = text.replace('ROOT / "data" / "lah10"', 'ROOT / "data" / "raw" / "lah10"')
        text = text.replace('ROOT / "data" / "ybacu3o7"', 'ROOT / "data" / "raw" / "ybacu3o7"')
        return text

    if name == "make_fig1_material_summary.py":
        text = text.replace(
            'ROOT = Path(__file__).resolve().parents[1]',
            'ROOT = Path(__file__).resolve().parents[2]',
        )
        text = text.replace(
            'OUTPUT_DIR = ROOT / "figs"',
            'OUTPUT_DIR = ROOT / "outputs" / "reproduced_figures"',
        )
        text = text.replace('ROOT / "scripts"', 'ROOT / "code" / "plotting"')
        return text

    if name == "fig4_plaquette_dynamics.py":
        text = text.replace(
            'ROOT = Path(__file__).resolve().parents[1]\nif str(ROOT / "src") not in sys.path:\n    sys.path.insert(0, str(ROOT / "src"))',
            'ROOT = Path(__file__).resolve().parents[2]\nif str(ROOT / "code" / "utilities") not in sys.path:\n    sys.path.insert(0, str(ROOT / "code" / "utilities"))',
        )
        text = text.replace(
            'FIGURE_PATH = ROOT / "figures" / "main" / "fig4_plaquette_dynamics.pdf"',
            'FIGURE_PATH = ROOT / "outputs" / "reproduced_figures" / "fig4_plaquette_dynamics.pdf"',
        )
        return text

    if name == "make_fig_xy8_qre.py":
        text = text.replace(
            "#!/usr/bin/env python3\nimport argparse\nimport csv\nimport json\nimport os\nfrom pathlib import Path\n\nimport matplotlib.pyplot as plt\nimport numpy as np\n\nimport nvratemodel_api as pipe",
            dedent(
                """
                #!/usr/bin/env python3
                import argparse
                import csv
                import json
                import os
                import sys
                from pathlib import Path

                ROOT = Path(__file__).resolve().parents[2]
                for extra in [ROOT / "code" / "analysis", ROOT / "code" / "utilities"]:
                    if str(extra) not in sys.path:
                        sys.path.insert(0, str(extra))

                import matplotlib.pyplot as plt
                import numpy as np

                import nvratemodel_api as pipe
                """
            ).strip("\n"),
        )
        text = text.replace('parser.add_argument("--outdir", default="figs"', 'parser.add_argument("--outdir", default="outputs/reproduced_figures"')
        return text

    if name == "nvratemodel_api.py":
        return text.replace('matplotlib.use("TkAgg")', 'matplotlib.use("Agg")')

    if name == "demo_edt_gyro_trotter_4q_fig34_clean.py":
        text = text.replace(
            'OUTDIR = Path("figs")\nOUTDIR.mkdir(exist_ok=True)',
            'ROOT = Path(__file__).resolve().parents[2]\nOUTDIR = ROOT / "outputs" / "reproduced_figures"\nOUTDIR.mkdir(parents=True, exist_ok=True)',
        )
        return text

    return text


def copy_curated_code() -> None:
    utilities_root = DEST_ROOT / "code" / "utilities"
    plotting_root = DEST_ROOT / "code" / "plotting"
    analysis_root = DEST_ROOT / "code" / "analysis"
    ensure_dir(utilities_root)
    ensure_dir(plotting_root)
    ensure_dir(analysis_root)

    copy_tree(SOURCE_ROOT / "src" / "edt", utilities_root / "edt")
    copy_tree(SOURCE_ROOT / "src" / "plaquette_model", utilities_root / "plaquette_model")

    plot_scripts = [
        "make_fig2_k3c60_edt.py",
        "make_material_figure2_panels.py",
        "make_figS4_material_controls.py",
        "make_fig1_material_summary.py",
        "fig4_plaquette_dynamics.py",
        "make_fig_xy8_qre.py",
        "demo_edt_gyro_trotter_4q_fig34_clean.py",
    ]
    for name in plot_scripts:
        src = SOURCE_ROOT / "scripts" / name
        text = patch_script_text(name, src.read_text(encoding="utf-8"))
        write_text(plotting_root / name, text)

    analysis_scripts = ["nvratemodel_api.py"]
    for name in analysis_scripts:
        src = SOURCE_ROOT / "scripts" / name
        text = patch_script_text(name, src.read_text(encoding="utf-8"))
        write_text(analysis_root / name, text)

    write_text(
        analysis_root / "__init__.py",
        "",
    )
    write_text(
        plotting_root / "__init__.py",
        "",
    )


def copy_curated_data_and_outputs() -> None:
    raw_pairs = [
        ("k3c60/tmp.agr", "data/raw/k3c60/tmp.agr"),
        ("na/Na-m5.agr", "data/raw/na/Na-m5.agr"),
        ("v/V-EP-F1.agr", "data/raw/v/V-EP-F1.agr"),
        ("ybacu3o7/YBa2Cu3O7_m33.agr", "data/raw/ybacu3o7/YBa2Cu3O7_m33.agr"),
        ("lah10/tmp_LaH10_m16.agr", "data/raw/lah10/tmp_LaH10_m16.agr"),
        ("nv/xy8_calibration_template.json", "data/raw/nv/xy8_calibration_template.json"),
        ("nv/xy8_scenario_conservative.json", "data/raw/nv/xy8_scenario_conservative.json"),
        ("nv/xy8_scenario_nominal.json", "data/raw/nv/xy8_scenario_nominal.json"),
        ("nv/xy8_scenario_optimistic.json", "data/raw/nv/xy8_scenario_optimistic.json"),
    ]
    for src_rel, dst_rel in raw_pairs:
        copy_file(SOURCE_ROOT / "data" / src_rel, DEST_ROOT / dst_rel)

    processed_pairs = [
        (SOURCE_ROOT / "data" / "k3c60" / "k3c60_edt_window_summary.json", DEST_ROOT / "data" / "processed" / "k3c60_edt_window_summary.json"),
        (SOURCE_ROOT / "data" / "processed" / "S1_Na_summary.json", DEST_ROOT / "data" / "processed" / "S1_Na_summary.json"),
        (SOURCE_ROOT / "data" / "processed" / "S2_V_summary.json", DEST_ROOT / "data" / "processed" / "S2_V_summary.json"),
        (SOURCE_ROOT / "data" / "processed" / "S4_YBa2Cu3O7_summary.json", DEST_ROOT / "data" / "processed" / "S4_YBa2Cu3O7_summary.json"),
        (SOURCE_ROOT / "data" / "processed" / "S5_LaH10_summary.json", DEST_ROOT / "data" / "processed" / "S5_LaH10_summary.json"),
        (SOURCE_ROOT / "data" / "processed" / "material_controls_summary.csv", DEST_ROOT / "data" / "processed" / "material_controls_summary.csv"),
        (SOURCE_ROOT / "data" / "processed" / "plaquette_lambda_sweep.csv", DEST_ROOT / "data" / "processed" / "plaquette_lambda_sweep.csv"),
        (SOURCE_ROOT / "figs" / "xy8_qre_nv_results.csv", DEST_ROOT / "data" / "processed" / "xy8_qre_nv_results.csv"),
        (SOURCE_ROOT / "figs" / "xy8_qre_nv_breakdown.json", DEST_ROOT / "data" / "processed" / "xy8_qre_nv_breakdown.json"),
    ]
    for src, dst in processed_pairs:
        copy_file(src, dst)

    figure_pairs = [
        (SOURCE_ROOT / "figs" / "figure1_material_summary.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "figure1_material_summary.pdf"),
        (SOURCE_ROOT / "figs" / "figure1_material_summary.png", DEST_ROOT / "outputs" / "reproduced_figures" / "figure1_material_summary.png"),
        (SOURCE_ROOT / "Latex_paper" / "Figure2.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "Figure2.pdf"),
        (SOURCE_ROOT / "figures" / "main" / "figure2_k3c60_panel_b_curvature.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "figure2_k3c60_panel_b_curvature.pdf"),
        (SOURCE_ROOT / "figures" / "main" / "figure2_k3c60_panel_c_fit.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "figure2_k3c60_panel_c_fit.pdf"),
        (SOURCE_ROOT / "figures" / "main" / "figure2_k3c60_panel_d_window.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "figure2_k3c60_panel_d_window.pdf"),
        (SOURCE_ROOT / "figures" / "main" / "figure2_k3c60_panel_e_bootstrap.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "figure2_k3c60_panel_e_bootstrap.pdf"),
        (SOURCE_ROOT / "Latex_paper" / "figure3_edt_regime_panel.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "figure3_edt_regime_panel.pdf"),
        (SOURCE_ROOT / "Latex_paper" / "demo_edt_gyro_trotter_4q_plaquette_otoc.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "demo_edt_gyro_trotter_4q_plaquette_otoc.pdf"),
        (SOURCE_ROOT / "Latex_paper" / "demo_edt_gyro_trotter_4q_circuit_example.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "demo_edt_gyro_trotter_4q_circuit_example.pdf"),
        (SOURCE_ROOT / "figures" / "main" / "fig4_plaquette_dynamics.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "fig4_plaquette_dynamics.pdf"),
        (SOURCE_ROOT / "figs" / "fig_xy8_qre_nv.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "fig_xy8_qre_nv.pdf"),
        (SOURCE_ROOT / "figs" / "fig_xy8_qre_nv.png", DEST_ROOT / "outputs" / "reproduced_figures" / "fig_xy8_qre_nv.png"),
        (SOURCE_ROOT / "figures" / "supplementary" / "S3_material_summary.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "S3_material_summary.pdf"),
        (SOURCE_ROOT / "figures" / "supplementary" / "K3C60-bands.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "K3C60-bands.pdf"),
        (SOURCE_ROOT / "figures" / "supplementary" / "K3C60-m120.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "K3C60-m120.pdf"),
        (SOURCE_ROOT / "figures" / "supplementary" / "S1_Na_panel_b_curvature.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "S1_Na_panel_b_curvature.pdf"),
        (SOURCE_ROOT / "figures" / "supplementary" / "S1_Na_panel_c_fit.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "S1_Na_panel_c_fit.pdf"),
        (SOURCE_ROOT / "figures" / "supplementary" / "S1_Na_panel_d_window.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "S1_Na_panel_d_window.pdf"),
        (SOURCE_ROOT / "figures" / "supplementary" / "S1_Na_panel_e_bootstrap.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "S1_Na_panel_e_bootstrap.pdf"),
        (SOURCE_ROOT / "figures" / "supplementary" / "S2_V_panel_b_curvature.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "S2_V_panel_b_curvature.pdf"),
        (SOURCE_ROOT / "figures" / "supplementary" / "S2_V_panel_c_fit.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "S2_V_panel_c_fit.pdf"),
        (SOURCE_ROOT / "figures" / "supplementary" / "S2_V_panel_d_window.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "S2_V_panel_d_window.pdf"),
        (SOURCE_ROOT / "figures" / "supplementary" / "S2_V_panel_e_bootstrap.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "S2_V_panel_e_bootstrap.pdf"),
        (SOURCE_ROOT / "figures" / "supplementary" / "S4_YBa2Cu3O7_panel_b_curvature.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "S4_YBa2Cu3O7_panel_b_curvature.pdf"),
        (SOURCE_ROOT / "figures" / "supplementary" / "S4_YBa2Cu3O7_panel_c_fit.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "S4_YBa2Cu3O7_panel_c_fit.pdf"),
        (SOURCE_ROOT / "figures" / "supplementary" / "S4_YBa2Cu3O7_panel_d_window.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "S4_YBa2Cu3O7_panel_d_window.pdf"),
        (SOURCE_ROOT / "figures" / "supplementary" / "S4_YBa2Cu3O7_panel_e_bootstrap.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "S4_YBa2Cu3O7_panel_e_bootstrap.pdf"),
        (SOURCE_ROOT / "figures" / "supplementary" / "S5_LaH10_panel_b_curvature.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "S5_LaH10_panel_b_curvature.pdf"),
        (SOURCE_ROOT / "figures" / "supplementary" / "S5_LaH10_panel_c_fit.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "S5_LaH10_panel_c_fit.pdf"),
        (SOURCE_ROOT / "figures" / "supplementary" / "S5_LaH10_panel_d_window.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "S5_LaH10_panel_d_window.pdf"),
        (SOURCE_ROOT / "figures" / "supplementary" / "S5_LaH10_panel_e_bootstrap.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "S5_LaH10_panel_e_bootstrap.pdf"),
        (SOURCE_ROOT / "figures" / "supplementary" / "S5_threshold_bootstrap.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "S5_threshold_bootstrap.pdf"),
        (SOURCE_ROOT / "figures" / "supplementary" / "figS7_curvature_topology.pdf", DEST_ROOT / "outputs" / "reproduced_figures" / "figS7_curvature_topology.pdf"),
    ]
    for src, dst in figure_pairs:
        copy_file(src, dst)


def make_table_s1() -> None:
    rows = []
    for name, pair_label, material_label in [
        ("k3c60_edt_window_summary.json", "208/209", r"K$_3$C$_{60}$"),
        ("S2_V_summary.json", "23/24", "V"),
        ("S1_Na_summary.json", "11/12", "Na"),
        ("S4_YBa2Cu3O7_summary.json", "146/148", r"YBa$_2$Cu$_3$O$_7$"),
        ("S5_LaH10_summary.json", "34/36", r"LaH$_{10}$"),
    ]:
        payload = json.loads((DEST_ROOT / "data" / "processed" / name).read_text(encoding="utf-8"))
        rows.append(
            {
                "material": payload.get("title", material_label),
                "pair": pair_label,
                "k0": payload["nominal_result"]["k0"],
                "delta_k_edt": payload["nominal_result"]["edt_width"],
                "bootstrap_ci_low": payload["bootstrap"]["width_ci_95"][0],
                "bootstrap_ci_high": payload["bootstrap"]["width_ci_95"][1],
            }
        )

    out_path = DEST_ROOT / "outputs" / "reproduced_tables" / "Table_S1_material_branch_pairs.csv"
    ensure_dir(out_path.parent)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_figure_source_data_exporter() -> None:
    text = dedent(
        """
        from __future__ import annotations

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


        def export_plaquette_and_nv() -> None:
            plaquette_rows = []
            with (ROOT / "data" / "processed" / "plaquette_lambda_sweep.csv").open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                plaquette_rows.extend(reader)
            write_csv(FIGURE_SOURCE_DIR / "Fig3_source_data.csv", [dict(row) for row in plaquette_rows])
            write_csv(FIGURE_SOURCE_DIR / "Fig4_source_data.csv", [dict(row) for row in plaquette_rows])

            nv_rows = []
            with (ROOT / "data" / "processed" / "xy8_qre_nv_results.csv").open("r", encoding="utf-8") as handle:
                lines = [line.rstrip("\\n") for line in handle]
            start = None
            for idx, line in enumerate(lines):
                if line == "tau_s,N,eta_proxy":
                    start = idx
                    break
            if start is None:
                raise RuntimeError("Could not find XY8 sweep table in xy8_qre_nv_results.csv")
            reader = csv.DictReader(lines[start:])
            nv_rows.extend(reader)
            write_csv(FIGURE_SOURCE_DIR / "Fig5_source_data.csv", [dict(row) for row in nv_rows])


        def main() -> None:
            ensure_dir(FIGURE_SOURCE_DIR)
            export_fig1_summary()
            export_fig2_and_supp()
            export_material_panels()
            export_plaquette_and_nv()


        if __name__ == "__main__":
            main()
        """
    ).strip() + "\n"
    write_text(DEST_ROOT / "code" / "utilities" / "export_figure_source_data.py", text)


def write_reproduction_driver() -> None:
    text = dedent(
        """
        from __future__ import annotations

        import importlib.util
        import shutil
        import subprocess
        import sys
        from pathlib import Path


        ROOT = Path(__file__).resolve().parents[1]
        PLOTTING = ROOT / "code" / "plotting"
        UTILITIES = ROOT / "code" / "utilities"


        def run(script: str, *args: str) -> None:
            cmd = [sys.executable, str(PLOTTING / script), *args]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True, cwd=ROOT)


        def has_qiskit() -> bool:
            return importlib.util.find_spec("qiskit") is not None


        def main() -> None:
            run("make_fig2_k3c60_edt.py")
            run("make_material_figure2_panels.py")
            run("make_fig1_material_summary.py")
            run("make_figS4_material_controls.py")
            run("fig4_plaquette_dynamics.py")
            run("make_fig_xy8_qre.py", "--calibration", "data/raw/nv/xy8_scenario_nominal.json", "--outdir", "outputs/reproduced_figures")
            shutil.copy2(ROOT / "outputs" / "reproduced_figures" / "xy8_qre_nv_results.csv", ROOT / "data" / "processed" / "xy8_qre_nv_results.csv")
            shutil.copy2(ROOT / "outputs" / "reproduced_figures" / "xy8_qre_nv_breakdown.json", ROOT / "data" / "processed" / "xy8_qre_nv_breakdown.json")
            subprocess.run([sys.executable, str(UTILITIES / "export_figure_source_data.py")], check=True, cwd=ROOT)
            if has_qiskit():
                run("demo_edt_gyro_trotter_4q_fig34_clean.py")
            else:
                print("Skipping demo_edt_gyro_trotter_4q_fig34_clean.py because qiskit is not installed.")


        if __name__ == "__main__":
            main()
        """
    ).strip() + "\n"
    write_text(DEST_ROOT / "code" / "run_reproduction_pipeline.py", text)


def write_environment_files() -> None:
    write_text(
        DEST_ROOT / "environment.yml",
        dedent(
            """
            name: edt-nature-physics
            channels:
              - conda-forge
            dependencies:
              - python=3.11
              - numpy=1.26.*
              - scipy=1.11.*
              - matplotlib=3.8.*
              - pandas=2.2.*
              - jupyterlab
              - nbformat
              - pytest=8.*
              - qiskit=1.*
            """
        ).strip()
        + "\n",
    )
    write_text(
        DEST_ROOT / ".gitignore",
        dedent(
            """
            __pycache__/
            *.py[cod]
            .pytest_cache/
            .ipynb_checkpoints/
            .DS_Store
            """
        ).strip()
        + "\n",
    )


def write_license_and_citation() -> None:
    write_text(
        DEST_ROOT / "LICENSE",
        dedent(
            """
            MIT License

            Copyright (c) 2026 Hang Hu

            Permission is hereby granted, free of charge, to any person obtaining a copy
            of this software and associated documentation files (the "Software"), to deal
            in the Software without restriction, including without limitation the rights
            to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
            copies of the Software, and to permit persons to whom the Software is
            furnished to do so, subject to the following conditions:

            The above copyright notice and this permission notice shall be included in all
            copies or substantial portions of the Software.

            THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
            IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
            FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
            AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
            LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
            OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
            SOFTWARE.
            """
        ).strip()
        + "\n",
    )
    write_text(
        DEST_ROOT / "CITATION.cff",
        dedent(
            """
            cff-version: 1.2.0
            message: "If you use this repository, please cite the repository release and the associated manuscript."
            title: "EDT_NaturePhysics_Data_Code"
            type: software
            authors:
              - family-names: Hu
                given-names: Hang
            abstract: "Companion data and code repository for the manuscript 'Programmable Quantum Resource Manifolds via Entanglement-Dynamic Topology'."
            version: 0.1.0
            date-released: 2026-05-18
            doi: "DOI_TO_BE_ADDED"
            repository-code: "GITHUB_URL_TO_BE_ADDED"
            preferred-citation:
              type: article
              title: "Programmable Quantum Resource Manifolds via Entanglement-Dynamic Topology"
              authors:
                - family-names: Hu
                  given-names: Hang
              doi: "MANUSCRIPT_DOI_TO_BE_ADDED"
            """
        ).strip()
        + "\n",
    )


def write_data_readme() -> None:
    write_text(
        DEST_ROOT / "data" / "README.md",
        dedent(
            """
            # Data directory guide

            This folder separates the companion repository data into four layers.

            - `raw/`: archived manuscript input data. The `*.agr` files are the raw mode-resolved second-order band-curvature exports used for the curvature-window analyses. Their paired columns are `(k, \\partial^2 E_n / \\partial Q_\\alpha^2)` in the native units encoded by the archive metadata.
            - `processed/`: analysis-ready outputs derived from the raw archives, including branch-pair summaries, bootstrap intervals, the plaquette `lambda` sweep table, and the NV XY8 sweep outputs.
            - `figure_source_data/`: figure-level source data tables prepared for journal sharing. These files provide the tabular data underlying the manuscript figures wherever numerical source data are available.

            ## Raw datasets

            - `raw/k3c60/tmp.agr`: K$_3$C$_{60}$ second-order band-curvature archive used for the main curvature-window diagnostic.
            - `raw/na/Na-m5.agr`: Na curvature archive used as the simple-metal baseline.
            - `raw/v/V-EP-F1.agr`: elemental V curvature archive.
            - `raw/ybacu3o7/YBa2Cu3O7_m33.agr`: YBa$_2$Cu$_3$O$_7$ curvature archive.
            - `raw/lah10/tmp_LaH10_m16.agr`: LaH$_{10}$ curvature archive.
            - `raw/nv/*.json`: literature-anchored NV XY8 scenario files used for the sensing case study.

            ## Processed datasets

            - `processed/k3c60_edt_window_summary.json`: extracted K$_3$C$_{60}$ window, fit, and bootstrap summary.
            - `processed/S1_Na_summary.json`, `processed/S2_V_summary.json`, `processed/S4_YBa2Cu3O7_summary.json`, `processed/S5_LaH10_summary.json`: per-material branch-pair summaries.
            - `processed/material_controls_summary.csv`: compact cross-material comparison table.
            - `processed/plaquette_lambda_sweep.csv`: plaquette dynamics sweep over `lambda`, `h`, and `J`.
            - `processed/xy8_qre_nv_results.csv`: XY8 sweep grid and benchmark points.
            - `processed/xy8_qre_nv_breakdown.json`: best-point resource breakdown for the NV case study.

            ## Figure source data files

            - `Fig1_source_data.csv` and `SuppFig_S1_source_data.csv`: bar heights and bootstrap confidence intervals for the cross-material summary.
            - `Fig2_panel_b_source_data.csv`, `Fig2_panel_c_source_data.csv`, `Fig2_panel_d_source_data.csv`, `Fig2_panel_e_source_data.csv`: K$_3$C$_{60}$ panel-level source data for the main curvature-window figure.
            - `SuppFig_S2_source_data.csv`: K$_3$C$_{60}$ supplementary curvature source table.
            - `SuppFig_S3_source_data.csv` to `SuppFig_S6_source_data.csv`: branch-resolved source data for Na, V, YBa$_2$Cu$_3$O$_7$, and LaH$_{10}$.
            - `SuppFig_S7_source_data.csv`: threshold sweep and bootstrap samples for the K$_3$C$_{60}$ robustness figure.
            - `Fig3_source_data.csv` and `Fig4_source_data.csv`: plaquette-dynamics sweep data used for the loop-coupling analysis.
            - `Fig5_source_data.csv`: NV XY8 sweep data (`tau_s`, `N`, `eta_proxy`).

            ## Column conventions

            Common columns include:

            - `k`: momentum coordinate along the archived high-symmetry path.
            - `kappa_minus`, `kappa_plus`: paired curvature branches selected for local quadratic analysis.
            - `fit_minus`, `fit_plus`: local quadratic fits to the paired branches.
            - `residual_minus`, `residual_plus`: normalized local quadratic residuals.
            - `delta_k_edt`: extracted width of the finite EDT window.
            - `bootstrap_ci_low`, `bootstrap_ci_high`: 95% bootstrap confidence interval bounds.
            - `lambda_loop`, `A_spread`, `A_corr`, `A_stability`, `S_dyn`: plaquette dynamics metrics.
            - `tau_s`, `N`, `eta_proxy`: NV XY8 sweep coordinates and sensitivity proxy.
            """
        ).strip()
        + "\n",
    )


def write_code_readme() -> None:
    write_text(
        DEST_ROOT / "code" / "README.md",
        dedent(
            """
            # Code layout

            The code in this repository is organized into lightweight analysis, plotting, and utility layers.

            ## `analysis/`

            - `nvratemodel_api.py`: helper routines for the literature-anchored NV XY8 case study, including readout handling, coherence models, pulse-fidelity compounding, and benchmark breakdowns.

            ## `plotting/`

            - `make_fig1_material_summary.py`: regenerates the cross-material EDT-window summary used in the main text and Supplementary Information.
            - `make_fig2_k3c60_edt.py`: reproduces the K$_3$C$_{60}$ curvature-window diagnostic and its panel-level outputs.
            - `make_material_figure2_panels.py`: reproduces the Na, V, YBa$_2$Cu$_3$O$_7$, and LaH$_{10}$ supplementary panel sets.
            - `make_figS4_material_controls.py`: generates the compact cross-material controls figure and summary CSV.
            - `fig4_plaquette_dynamics.py`: runs the plaquette `lambda` sweep and generates the loop-controlled dynamics figure.
            - `demo_edt_gyro_trotter_4q_fig34_clean.py`: archived 4-qubit plaquette/Trotter workflow used for the manuscript's implementation-oriented panels.
            - `make_fig_xy8_qre.py`: regenerates the NV XY8 sensing map from literature-anchored scenario files.

            ## `utilities/`

            - `edt/`: reusable curvature, Hamiltonian, and NV helper modules.
            - `plaquette_model/`: minimal exact-dynamics and Trotter tools for the plaquette model.
            - `export_figure_source_data.py`: creates the figure-level source data CSV files from the raw and processed datasets.

            ## Top-level reproduction driver

            - `run_reproduction_pipeline.py`: sequentially runs the core figure scripts and exports the source-data tables.

            ## Example usage

            ```bash
            python code/run_reproduction_pipeline.py
            python code/plotting/make_fig2_k3c60_edt.py
            python code/plotting/make_material_figure2_panels.py
            python code/plotting/make_fig_xy8_qre.py --calibration data/raw/nv/xy8_scenario_nominal.json --outdir outputs/reproduced_figures
            python code/utilities/export_figure_source_data.py
            ```
            """
        ).strip()
        + "\n",
    )


def write_docs() -> None:
    write_text(
        DEST_ROOT / "docs" / "reproduction_instructions.md",
        dedent(
            """
            # Reproduction instructions

            1. Create the conda environment:

               ```bash
               conda env create -f environment.yml
               conda activate edt-nature-physics
               ```

            2. Run the core reproduction pipeline:

               ```bash
               python code/run_reproduction_pipeline.py
               ```

            3. Inspect the regenerated outputs:

               - Figures: `outputs/reproduced_figures/`
               - Tables: `outputs/reproduced_tables/`
               - Figure-level source data: `data/figure_source_data/`

            4. If only a subset is needed, the principal scripts are:

               ```bash
               python code/plotting/make_fig2_k3c60_edt.py
               python code/plotting/make_material_figure2_panels.py
               python code/plotting/fig4_plaquette_dynamics.py
               python code/plotting/make_fig_xy8_qre.py --calibration data/raw/nv/xy8_scenario_nominal.json --outdir outputs/reproduced_figures
               python code/utilities/export_figure_source_data.py
               ```

            5. Optional repository smoke check:

               ```bash
               python tests/smoke_check.py
               ```

            Notes:

            - The curvature workflows run directly on the archived second-order band-curvature files in `data/raw/`.
            - The NV XY8 script is configured to run from the explicit scenario JSON files and therefore does not require `nvratemodel`.
            - The archived Trotter circuit script depends on `qiskit`. If `qiskit` is unavailable, the core curvature and NV analyses remain reproducible, and the reproduction driver will skip that optional step.
            """
        ).strip()
        + "\n",
    )

    write_text(
        DEST_ROOT / "docs" / "figure_data_mapping.md",
        dedent(
            """
            # Figure-to-data mapping

            ## Main text

            | Manuscript item | Source data file(s) | Code/script | Reproduced output |
            | --- | --- | --- | --- |
            | Fig. 1: Cross-material EDT-window summary | `data/figure_source_data/Fig1_source_data.csv` | `code/plotting/make_fig1_material_summary.py` | `outputs/reproduced_figures/figure1_material_summary.pdf` |
            | Fig. 2: K$_3$C$_{60}$ curvature-window analysis | `data/figure_source_data/Fig2_panel_b_source_data.csv`; `Fig2_panel_c_source_data.csv`; `Fig2_panel_d_source_data.csv`; `Fig2_panel_e_source_data.csv` | `code/plotting/make_fig2_k3c60_edt.py` | `outputs/reproduced_figures/Figure2.pdf`; panel PDFs in the same folder |
            | Fig. 3: Plaquette loop-coupling dynamics | `data/figure_source_data/Fig3_source_data.csv` | `code/plotting/fig4_plaquette_dynamics.py` | `outputs/reproduced_figures/fig4_plaquette_dynamics.pdf` |
            | Fig. 4: Trotter/implementation-oriented plaquette panels | `data/figure_source_data/Fig4_source_data.csv` (shared plaquette sweep) | `code/plotting/demo_edt_gyro_trotter_4q_fig34_clean.py` | `outputs/reproduced_figures/demo_edt_gyro_trotter_4q_plaquette_otoc.pdf`; `demo_edt_gyro_trotter_4q_circuit_example.pdf` |
            | Fig. 5: NV XY8 sensing case study | `data/figure_source_data/Fig5_source_data.csv` | `code/plotting/make_fig_xy8_qre.py` | `outputs/reproduced_figures/fig_xy8_qre_nv.pdf` |

            ## Supplementary figures

            | Supplementary item | Source data file(s) | Code/script | Reproduced output |
            | --- | --- | --- | --- |
            | Fig. S1: Cross-material summary | `data/figure_source_data/SuppFig_S1_source_data.csv` | `code/plotting/make_fig1_material_summary.py` | `outputs/reproduced_figures/S3_material_summary.pdf` |
            | Fig. S2: K$_3$C$_{60}$ curvature-resolved structure | `data/figure_source_data/SuppFig_S2_source_data.csv` | `code/plotting/make_fig2_k3c60_edt.py` | `outputs/reproduced_figures/K3C60-bands.pdf`; `K3C60-m120.pdf` |
            | Fig. S3: Na branch-pair analysis | `data/figure_source_data/SuppFig_S3_source_data.csv` | `code/plotting/make_material_figure2_panels.py` | `outputs/reproduced_figures/S1_Na_panel_b_curvature.pdf` and companion panels |
            | Fig. S4: V branch-pair analysis | `data/figure_source_data/SuppFig_S4_source_data.csv` | `code/plotting/make_material_figure2_panels.py` | `outputs/reproduced_figures/S2_V_panel_b_curvature.pdf` and companion panels |
            | Fig. S5: YBa$_2$Cu$_3$O$_7$ branch-pair analysis | `data/figure_source_data/SuppFig_S5_source_data.csv` | `code/plotting/make_material_figure2_panels.py` | `outputs/reproduced_figures/S4_YBa2Cu3O7_panel_b_curvature.pdf` and companion panels |
            | Fig. S6: LaH$_{10}$ branch-pair analysis | `data/figure_source_data/SuppFig_S6_source_data.csv` | `code/plotting/make_material_figure2_panels.py` | `outputs/reproduced_figures/S5_LaH10_panel_b_curvature.pdf` and companion panels |
            | Fig. S7: K$_3$C$_{60}$ robustness checks | `data/figure_source_data/SuppFig_S7_source_data.csv` | `code/plotting/make_fig2_k3c60_edt.py` | `outputs/reproduced_figures/S5_threshold_bootstrap.pdf` |
            | Fig. S8: curvature-topology schematic | N/A (schematic PDF asset) | archived manuscript asset | `outputs/reproduced_figures/figS7_curvature_topology.pdf` |
            | Fig. S9: Trotter circuit schematic | N/A (circuit PDF asset) | `code/plotting/demo_edt_gyro_trotter_4q_fig34_clean.py` | `outputs/reproduced_figures/demo_edt_gyro_trotter_4q_circuit_example.pdf` |
            """
        ).strip()
        + "\n",
    )

    write_text(
        DEST_ROOT / "docs" / "manuscript_availability_statements.md",
        dedent(
            """
            # Manuscript-ready availability statements

            ## Data availability

            Zenodo-ready version:

            > The data supporting the findings of this study are available within the Article, its Supplementary Information and the Source Data files. The numerical datasets used to generate the main and Supplementary figures are available in the accompanying repository at [Zenodo DOI to be added].

            GitHub-placeholder version:

            > The data supporting the findings of this study are available within the Article, its Supplementary Information and the Source Data files. The numerical datasets used to generate the main and Supplementary figures are available in the accompanying repository at [GitHub repository URL to be added].

            ## Code availability

            Zenodo-ready version:

            > The custom code used to perform the curvature-window analysis, evaluate local quadratic residuals, conduct the branch-pair analyses and generate the figures in this study is available in the accompanying repository at [Zenodo DOI to be added].

            GitHub-placeholder version:

            > The custom code used to perform the curvature-window analysis, evaluate local quadratic residuals, conduct the branch-pair analyses and generate the figures in this study is available in the accompanying repository at [GitHub repository URL to be added].
            """
        ).strip()
        + "\n",
    )


def write_notebook() -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Optional reproduction workflow\\n",
                    "\\n",
                    "This notebook is a lightweight entry point for readers who prefer a notebook-based reproduction path.\\n",
                    "The full scripted workflow is documented in `docs/reproduction_instructions.md`.\\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\\n",
                    "ROOT = Path.cwd().resolve()\\n",
                    "print(ROOT)\\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import subprocess, sys\\n",
                    "subprocess.run([sys.executable, 'code/run_reproduction_pipeline.py'], check=False)\\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    write_json(DEST_ROOT / "notebooks" / "optional_reproduction_workflow.ipynb", notebook)


def write_auxiliary_repo_files() -> None:
    write_text(
        DEST_ROOT / "scripts" / "README.md",
        dedent(
            """
            # Scripts directory

            The actively maintained reproduction scripts for this companion repository live under `code/plotting/` and `code/run_reproduction_pipeline.py`.
            This top-level folder is kept as a lightweight placeholder to match the expected archival repository layout and can be used for future shell wrappers or release helpers if needed.
            """
        ).strip()
        + "\n",
    )
    write_text(
        DEST_ROOT / "tests" / "smoke_check.py",
        dedent(
            """
            from pathlib import Path


            ROOT = Path(__file__).resolve().parents[1]


            def expected_directories_exist():
                for rel in [
                    "data/raw",
                    "data/processed",
                    "data/figure_source_data",
                    "code/analysis",
                    "code/plotting",
                    "code/utilities",
                    "outputs/reproduced_figures",
                    "outputs/reproduced_tables",
                    "docs",
                ]:
                    if not (ROOT / rel).exists():
                        raise SystemExit(f"Missing expected directory: {rel}")


            def core_source_data_files_exist():
                for rel in [
                    "data/figure_source_data/Fig1_source_data.csv",
                    "data/figure_source_data/Fig2_panel_b_source_data.csv",
                    "data/figure_source_data/Fig5_source_data.csv",
                ]:
                    if not (ROOT / rel).exists():
                        raise SystemExit(f"Missing expected file: {rel}")


            if __name__ == "__main__":
                expected_directories_exist()
                core_source_data_files_exist()
                print("Smoke checks passed.")
            """
        ).strip()
        + "\n",
    )


def write_main_readme() -> None:
    write_text(
        DEST_ROOT / "README.md",
        dedent(
            """
            # Companion data and code for *Programmable Quantum Resource Manifolds via Entanglement-Dynamic Topology*

            **Zenodo DOI:** DOI_TO_BE_ADDED  
            **Repository URL:** GITHUB_URL_TO_BE_ADDED  
            **Contact:** Hang Hu

            This repository is the publication-ready companion package for the EDT manuscript submitted in the style expected for Nature Physics / Springer Nature data and code availability. It contains the archived curvature datasets, processed analysis outputs, figure-level source data files, and the custom Python code needed to reproduce the manuscript's central curvature-window, plaquette-dynamics, and NV XY8 sensing analyses. The repository is organized to separate raw inputs, processed analysis products, source data tables, reproducible figure-generation code, and generated outputs.

            ## Repository contents

            - `data/`: raw manuscript inputs, processed analysis outputs, and figure-level source data tables.
            - `code/analysis/`: analysis helpers used by the figure-generation scripts.
            - `code/plotting/`: scripts that regenerate the manuscript figures and processed outputs.
            - `code/utilities/`: reusable curvature, plaquette, and export modules.
            - `notebooks/`: optional notebook entry point for a lightweight reproduction workflow.
            - `outputs/reproduced_figures/`: reproduced figure assets and archived manuscript figure PDFs.
            - `outputs/reproduced_tables/`: reproduced tables for manuscript-ready reporting.
            - `docs/`: reproduction notes, figure/data mapping, and draft availability statements.

            ## Software requirements

            - Python 3.11
            - NumPy 1.26
            - SciPy 1.11
            - Matplotlib 3.8
            - pandas 2.2
            - qiskit 1.x (optional for the archived Trotter circuit workflow)

            ## Installation

            ```bash
            conda env create -f environment.yml
            conda activate edt-nature-physics
            ```

            ## Minimal reproduction workflow

            1. Install the environment.
            2. Run the core analysis scripts:

               ```bash
               python code/run_reproduction_pipeline.py
               ```

            3. Regenerate figure-level source data and reproduced outputs:

               - Figures: `outputs/reproduced_figures/`
               - Tables: `outputs/reproduced_tables/`
               - Source data: `data/figure_source_data/`

            ## Figure and table mapping

            | Manuscript item | Source data file | Code/script | Reproduced output |
            | --- | --- | --- | --- |
            | Fig. 1 | `data/figure_source_data/Fig1_source_data.csv` | `code/plotting/make_fig1_material_summary.py` | `outputs/reproduced_figures/figure1_material_summary.pdf` |
            | Fig. 2 | `data/figure_source_data/Fig2_panel_*.csv` | `code/plotting/make_fig2_k3c60_edt.py` | `outputs/reproduced_figures/Figure2.pdf` and panel PDFs |
            | Fig. 3 | `data/figure_source_data/Fig3_source_data.csv` | `code/plotting/fig4_plaquette_dynamics.py` | `outputs/reproduced_figures/fig4_plaquette_dynamics.pdf` |
            | Fig. 4 | `data/figure_source_data/Fig4_source_data.csv` | `code/plotting/demo_edt_gyro_trotter_4q_fig34_clean.py` | `outputs/reproduced_figures/demo_edt_gyro_trotter_4q_plaquette_otoc.pdf` |
            | Fig. 5 | `data/figure_source_data/Fig5_source_data.csv` | `code/plotting/make_fig_xy8_qre.py` | `outputs/reproduced_figures/fig_xy8_qre_nv.pdf` |
            | Table S1 | `outputs/reproduced_tables/Table_S1_material_branch_pairs.csv` | `code/utilities/export_figure_source_data.py` plus processed summaries | `outputs/reproduced_tables/Table_S1_material_branch_pairs.csv` |

            A more detailed figure-by-figure map is provided in `docs/figure_data_mapping.md`.

            ## Suggested citation

            Hu, H. *et al.* Companion data and code for *Programmable Quantum Resource Manifolds via Entanglement-Dynamic Topology* (Version 0.1.0) [Data set and software]. Zenodo. DOI_TO_BE_ADDED.

            ## Notes for public release

            - Replace the DOI and repository URL placeholders in `README.md`, `CITATION.cff`, and `docs/manuscript_availability_statements.md` after the GitHub release is archived on Zenodo.
            - Review the manuscript title, author list, and citation metadata for exact journal-ready wording before public deposition.
            """
        ).strip()
        + "\n",
    )


def build_repo_structure() -> None:
    for rel in [
        "data/raw",
        "data/processed",
        "data/figure_source_data",
        "code/analysis",
        "code/plotting",
        "code/utilities",
        "notebooks",
        "outputs/reproduced_figures",
        "outputs/reproduced_tables",
        "docs",
        "scripts",
        "tests",
    ]:
        ensure_dir(DEST_ROOT / rel)


def remove_transient_caches() -> None:
    for cache_dir in DEST_ROOT.rglob("__pycache__"):
        shutil.rmtree(cache_dir, ignore_errors=True)


def main() -> None:
    build_repo_structure()
    copy_curated_code()
    copy_curated_data_and_outputs()
    write_figure_source_data_exporter()
    write_reproduction_driver()
    write_environment_files()
    write_license_and_citation()
    write_data_readme()
    write_code_readme()
    write_docs()
    write_notebook()
    write_auxiliary_repo_files()
    write_main_readme()
    make_table_s1()
    module = load_module("companion_export_source_data", DEST_ROOT / "code" / "utilities" / "export_figure_source_data.py")
    module.main()
    for stale in [
        DEST_ROOT / "tests" / "test_repository_smoke.py",
        DEST_ROOT / "code" / "__init__.py",
    ]:
        stale.unlink(missing_ok=True)
    remove_transient_caches()


if __name__ == "__main__":
    main()
