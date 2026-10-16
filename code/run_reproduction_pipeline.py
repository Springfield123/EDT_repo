from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLOTTING = ROOT / "code" / "plotting"
UTILITIES = ROOT / "code" / "utilities"
NV_SCENARIOS = {
    "conservative": ROOT / "data" / "raw" / "nv" / "xy8_scenario_conservative.json",
    "nominal": ROOT / "data" / "raw" / "nv" / "xy8_scenario_nominal.json",
    "optimistic": ROOT / "data" / "raw" / "nv" / "xy8_scenario_optimistic.json",
}


def run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(PLOTTING / script), *args]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def has_qiskit() -> bool:
    return importlib.util.find_spec("qiskit") is not None


def copy_nv_artifacts(output_dir: Path) -> None:
    processed = ROOT / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    for filename in [
        "xy8_qre_nv_results.csv",
        "xy8_qre_nv_breakdown.json",
        "xy8_edt_window_10pct.csv",
        "xy8_scenario_optima.csv",
        "xy8_resource_ablation.csv",
        "xy8_uncertainty_phase_diagram.csv",
        "xy8_robustness_summary.json",
        "xy8_frequency_sweep.csv",
        "xy8_monte_carlo_samples.csv",
        "xy8_monte_carlo_optimum_distribution.csv",
        "xy8_robust_control_comparison.csv",
        "xy8_uncertainty_attribution_samples.csv",
        "xy8_uncertainty_rank_correlations.csv",
        "xy8_uncertainty_permutation_importance.csv",
        "xy8_calibration_value.csv",
        "xy8_calibration_value_decisions.csv",
        "xy8_uncertainty_oat_sweeps.csv",
        "xy8_uncertainty_attribution_summary.json",
        "xy8_literature_validation_curve.csv",
        "xy8_literature_validation_uncertainty.csv",
        "xy8_literature_validation_metrics.json",
        "xy8_literature_model_benchmark.csv",
        "xy8_literature_model_benchmark_curves.csv",
        "xy8_literature_model_benchmark_summary.json",
        "xy8_specific_literature_validation_curve.csv",
        "xy8_specific_literature_validation_metrics.json",
        "xy8_pham_depth_topology_thresholds.csv",
        "xy8_pham_depth_topology_distributions.csv",
        "xy8_pham_depth_topology_summary.json",
        "xy8_devience_sparse_surface.csv",
        "xy8_devience_topology_thresholds.csv",
        "xy8_devience_topology_components.csv",
        "xy8_devience_topology_summary.json",
        "xy8_loretz_spurious_spectrum.csv",
        "xy8_loretz_spurious_thresholds.csv",
        "xy8_loretz_spurious_components.csv",
        "xy8_loretz_spurious_summary.json",
        "xy8_sequence_comparison.csv",
        "xy8_extended_analysis_summary.json",
        "xy8_pareto_frontier.csv",
        "xy8_prospective_validation.csv",
        "xy8_prospective_validation_protocol.json",
        "xy8_prospective_power_curve.csv",
        "xy8_prospective_power_summary.json",
        "xy8_control_recommendation.json",
        "xy8_environment_depth_maps.csv",
        "xy8_environment_contour_source.csv",
        "xy8_environment_figure_summary.json",
    ]:
        shutil.copy2(output_dir / filename, processed / filename)


def run_nv_pipeline() -> None:
    output_dir = ROOT / "outputs" / "reproduced_figures"
    scenario_dir = output_dir / "scenarios"
    run(
        "make_fig_xy8_qre.py",
        "--calibration",
        str(NV_SCENARIOS["nominal"]),
        "--outdir",
        str(output_dir),
    )
    for label in ["conservative", "optimistic"]:
        run(
            "make_fig_xy8_qre.py",
            "--calibration",
            str(NV_SCENARIOS[label]),
            "--outdir",
            str(scenario_dir / label),
        )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "code" / "recommend_xy8_control.py"),
            "--frequency-khz",
            "200",
            "--pulse-fidelity",
            "0.997",
            "--t2-us",
            "300",
            "--calibration",
            str(NV_SCENARIOS["nominal"]),
            "--json-out",
            str(output_dir / "xy8_control_recommendation.json"),
            "--quiet",
        ],
        check=True,
        cwd=ROOT,
    )
    run(
        "make_xy8_robustness_figures.py",
        "--calibration",
        str(NV_SCENARIOS["nominal"]),
        "--outdir",
        str(output_dir),
    )
    run(
        "make_xy8_prospective_validation.py",
        "--calibration",
        str(NV_SCENARIOS["nominal"]),
        "--outdir",
        str(output_dir),
    )
    run(
        "make_xy8_prospective_power.py",
        "--protocol",
        str(output_dir / "xy8_prospective_validation_protocol.json"),
        "--outdir",
        str(output_dir),
    )
    run(
        "make_xy8_extended_analyses.py",
        "--calibration",
        str(NV_SCENARIOS["nominal"]),
        "--outdir",
        str(output_dir),
    )
    run(
        "make_xy8_uncertainty_attribution.py",
        "--calibration",
        str(NV_SCENARIOS["nominal"]),
        "--samples-csv",
        str(output_dir / "xy8_monte_carlo_samples.csv"),
        "--outdir",
        str(output_dir),
    )
    run(
        "make_xy8_literature_validation.py",
        "--outdir",
        str(output_dir),
    )
    run(
        "make_xy8_model_benchmark.py",
        "--outdir",
        str(output_dir),
    )
    run(
        "make_xy8_specific_literature_validation.py",
        "--outdir",
        str(output_dir),
    )
    run(
        "make_xy8_experimental_depth_topology.py",
        "--outdir",
        str(output_dir),
    )
    run(
        "make_xy8_devience_topology.py",
        "--outdir",
        str(output_dir),
    )
    run(
        "make_xy8_loretz_spurious_topology.py",
        "--outdir",
        str(output_dir),
    )
    run(
        "make_xy8_environment_topology.py",
        "--processed-dir",
        str(ROOT / "data" / "processed"),
        "--outdir",
        str(output_dir),
    )
    copy_nv_artifacts(output_dir)
    subprocess.run(
        [sys.executable, str(UTILITIES / "export_figure_source_data.py"), "--only", "fig5"],
        check=True,
        cwd=ROOT,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate EDT paper artifacts.")
    parser.add_argument("--nv-only", action="store_true", help="Regenerate only the NV XY8 figures and source data.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.nv_only:
        run_nv_pipeline()
        return
    run("make_fig2_k3c60_edt.py")
    run("make_material_figure2_panels.py")
    run("make_fig1_material_summary.py")
    run("make_figS4_material_controls.py")
    run("fig4_plaquette_dynamics.py")
    run_nv_pipeline()
    subprocess.run([sys.executable, str(UTILITIES / "export_figure_source_data.py")], check=True, cwd=ROOT)
    if has_qiskit():
        run("demo_edt_gyro_trotter_4q_fig34_clean.py")
    else:
        print("Skipping demo_edt_gyro_trotter_4q_fig34_clean.py because qiskit is not installed.")


if __name__ == "__main__":
    main()
