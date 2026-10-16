# Code layout

The code in this repository is organized into lightweight analysis, plotting, and utility layers.

## `analysis/`

- `nvratemodel_api.py`: helper routines for the literature-anchored NV XY8 case study, including readout handling, coherence models, pulse-fidelity compounding, and benchmark breakdowns.
- `xy8_coherent_control.py`: two-level XY8 propagator with exact phase cycling, finite-duration driven pulses, weighted quasi-static amplitude/detuning ensembles, and a phase-aware central-difference AC sensing response.
- `xy8_spam_model.py`: independently calibrated state-preparation, analysis-pulse, and optical-readout model, plus prospective-protocol validation that prevents leakage from the frozen XY8-depth holdout.
- `xy8_spam_calibration.py`: shot-level Stage 1 estimator for bright/dark references, zero-depth quadrature visibility and offsets, propagated covariance, acquisition gates, and calibration-source digests.
- `xy8_prospective_power.py`: seeded scaled-chi power model that converts the seven-setting predictions into decision probabilities, required independent batches, and acquisition-time estimates.
- `xy8_resonant_topology.py`: validates rectangular XY8 score surfaces and counts connected components of a relative-performance sublevel set using explicit four- or eight-neighbour grid connectivity.
- `xy8_topology_persistence.py`: tracks connected-region births, continuations, and mergers across nested relative-performance thresholds and bounds each region's timing width from adjacent rejected samples.
- `nvratemodel_api.py`: sensitivity-ranking API with the historical scalar control model as its default and the sequence-specific coherent XY8 response available through `control_model="coherent_xy8"`.
- `xy8_control_recommender.py`: reusable hardware-to-control decision rule with 10% windows, Pareto alternatives, guardrails, and optional seeded robust selection.
- `xy8_literature_validation.py`: total-pulse holdout evaluator, uncertainty envelope, and trend-validation metrics for independently selected experimental data.
- `xy8_specific_literature_validation.py`: focused conventional-XY8 holdout evaluator for the scalar per-pulse survival assumption.
- `xy8_experimental_depth_topology.py`: seeded one-dimensional connected-region and threshold-persistence analysis for the frozen Pham depth curve.
- `xy8_devience_topology.py`: sparse frequency-by-depth connected-region analysis for four vector-digitized experimental XY8 sweeps from DeVience et al.
- `xy8_loretz_spurious_topology.py`: segmented one-dimensional persistence test for experimentally observed fundamental and spurious XY8 passbands.
- `xy8_environment_conditioning.py`, `xy8_environment_coherence.py`, and `xy8_environment_score.py`: frozen strain/temperature transition, coherence, and score-surface adapters.
- `xy8_environment_topology.py`: connected-region analysis, preregistered refinement selection, and medium/fine-grid confirmation for the environment-conditioned surfaces.

## `plotting/`

- `make_fig1_material_summary.py`: regenerates the cross-material EDT-window summary used in the main text and Supplementary Information.
- `make_fig2_k3c60_edt.py`: reproduces the K$_3$C$_{60}$ curvature-window diagnostic and its panel-level outputs.
- `make_material_figure2_panels.py`: reproduces the Na, V, YBa$_2$Cu$_3$O$_7$, and LaH$_{10}$ supplementary panel sets.
- `make_figS4_material_controls.py`: generates the compact cross-material controls figure and summary CSV.
- `fig4_plaquette_dynamics.py`: runs the plaquette `lambda` sweep and generates the loop-controlled dynamics figure.
- `demo_edt_gyro_trotter_4q_fig34_clean.py`: archived 4-qubit plaquette/Trotter workflow used for the manuscript's implementation-oriented panels.
- `make_fig_xy8_qre.py`: regenerates the NV XY8 sensing map from literature-anchored scenario files.
- `make_xy8_robustness_figures.py`: generates the resource ablation and pulse-fidelity/coherence phase diagram.
- `make_xy8_prospective_validation.py`: freezes a seven-setting validation protocol with predictions, uncertainty intervals, and blank experimental fields.
- `make_xy8_prospective_power.py`: generates the prospective sample-size curves and shows why the operating window is easier to establish than the exact optimum.
- `make_xy8_literature_validation.py`: compares the scalar control-depth prediction with a frozen, independently selected experimental sensitivity curve and exports uncertainty-aware validation metrics.
- `make_xy8_specific_literature_validation.py`: compares frozen scalar and sequence-specific coherent predictions with the Farfurnik et al. conventional-XY8 contrast series, without fitting, and reports when added physics fails to improve transfer.
- `make_xy8_experimental_depth_topology.py`: bootstraps the frozen Pham error bars to compare measured and no-fit predicted region count, boundaries, width, and persistence.
- `make_xy8_devience_topology.py`: visualizes the independent DeVience XY8 timing-depth surface, region separation/merger, and digitization-only stability.
- `make_xy8_loretz_spurious_topology.py`: keeps Loretz as a separate harmonic-mechanism check and excludes the paper's simulations.
- `make_xy8_model_benchmark.py`: compares coherence-only, pulse-fidelity, full-resource, fixed-control, and experimental-reference alternatives on the frozen holdout.
- `make_xy8_extended_analyses.py`: generates the resource--performance Pareto frontier, frequency sweep, seeded Monte Carlo uncertainty, robust-control comparison, and equal-total-pulse sequence capability check.
- `make_xy8_uncertainty_attribution.py`: reuses the seeded Monte Carlo cases to quantify one-at-a-time effects, rank correlations, grouped permutation importance, and the decision value of measuring each hardware resource.
- `make_xy8_topology_figure.py`: generates the presentation-ready nested EDT regions, connected-component persistence curve, and categorical depth/timing growth panel from the converged topology analysis.
- `make_xy8_environment_topology.py`: generates the Stage-6 discrete temperature/strain decision maps and paired fine-grid EDT contours from frozen environment artifacts.

## `utilities/`

- `edt/`: reusable curvature, Hamiltonian, and NV helper modules.
- `plaquette_model/`: minimal exact-dynamics and Trotter tools for the plaquette model.
- `export_figure_source_data.py`: creates the figure-level source data CSV files from the raw and processed datasets.

## Top-level reproduction driver

- `run_reproduction_pipeline.py`: sequentially runs the core figure scripts and exports the source-data tables.
- `analyze_xy8_resonant_topology.py`: characterizes the connected components of the canonical XY8 relative-performance set and exports region-level CSV and JSON records.
- `analyze_xy8_topology_refinement.py`: evaluates progressively denser timing grids around every target-frequency resonance and checks component-count, connectivity, optimum, timing-width, and boundary convergence.
- `analyze_xy8_topology_persistence.py`: follows the converged fine-grid EDT region from 1% to 100%, exporting component counts, width and depth growth, topology events, and stable component identities.
- `analyze_xy8_environment_topology.py`: analyzes all discovery surfaces, freezes the confirmation selection, and verifies the selected surfaces on medium and fine timing grids.

## Example usage

```bash
python code/run_reproduction_pipeline.py
python code/plotting/make_fig2_k3c60_edt.py
python code/plotting/make_material_figure2_panels.py
python code/plotting/make_fig_xy8_qre.py --calibration data/raw/nv/xy8_scenario_nominal.json --outdir outputs/reproduced_figures
python code/plotting/make_xy8_prospective_validation.py --calibration data/raw/nv/xy8_scenario_nominal.json --outdir outputs/reproduced_figures
python code/plotting/make_xy8_prospective_power.py --protocol outputs/reproduced_figures/xy8_prospective_validation_protocol.json --outdir outputs/reproduced_figures
python code/plotting/make_xy8_literature_validation.py --outdir outputs/reproduced_figures
python code/plotting/make_xy8_experimental_depth_topology.py --outdir outputs/reproduced_figures --mirror-data-dir data/processed
python code/plotting/make_xy8_devience_topology.py --outdir outputs/reproduced_figures --mirror-data-dir data/processed
python code/plotting/make_xy8_loretz_spurious_topology.py --outdir outputs/reproduced_figures --mirror-data-dir data/processed
python code/plotting/make_xy8_model_benchmark.py --outdir outputs/reproduced_figures
python code/plotting/make_xy8_extended_analyses.py --calibration data/raw/nv/xy8_scenario_nominal.json --outdir outputs/reproduced_figures
python code/plotting/make_xy8_uncertainty_attribution.py --calibration data/raw/nv/xy8_scenario_nominal.json --samples-csv outputs/reproduced_figures/xy8_monte_carlo_samples.csv --outdir outputs/reproduced_figures
python code/plotting/make_xy8_topology_figure.py --mirror-data-dir data/processed
python code/plotting/make_xy8_environment_topology.py --processed-dir data/processed --outdir outputs/reproduced_figures
python code/analyze_xy8_resonant_topology.py --relative-limit 1.10 --frequency-khz 200
python code/analyze_xy8_topology_refinement.py --frequency-khz 200 --relative-limit 1.10 --mirror-outdir outputs/reproduced_figures
python code/analyze_xy8_topology_persistence.py --frequency-khz 200 --mirror-outdir outputs/reproduced_figures
python code/recommend_xy8_control.py --frequency-khz 200 --pulse-fidelity 0.997 --t2-us 300
python code/calibrate_xy8_spam.py --shots /path/to/independent_spam_shots.csv
python code/utilities/export_figure_source_data.py
```
