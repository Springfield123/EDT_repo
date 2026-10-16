# Reproduction instructions

1. Create the conda environment:

   ```bash
   conda env create -f environment.yml
   conda activate edt-codebase
   ```

2. Run the core reproduction pipeline:

   ```bash
   python code/run_reproduction_pipeline.py
   ```

   To regenerate only the NV XY8 paper figure, 10% EDT window, scenario
   comparison, resource ablation, Pareto frontier, prospective validation protocol,
   two independent literature holdout comparisons,
   hardware-to-control recommendation, frequency sweep, uncertainty analyses,
   robust-control comparison, the frozen environment-conditioned EDT summary,
   processed data, and Figure 5 source data:

   ```bash
   python code/run_reproduction_pipeline.py --nv-only
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
   python code/plotting/make_xy8_robustness_figures.py --calibration data/raw/nv/xy8_scenario_nominal.json --outdir outputs/reproduced_figures
   python code/plotting/make_xy8_prospective_validation.py --calibration data/raw/nv/xy8_scenario_nominal.json --outdir outputs/reproduced_figures
   python code/plotting/make_xy8_prospective_power.py --protocol outputs/reproduced_figures/xy8_prospective_validation_protocol.json --outdir outputs/reproduced_figures
   python code/plotting/make_xy8_literature_validation.py --outdir outputs/reproduced_figures
   python code/plotting/make_xy8_model_benchmark.py --outdir outputs/reproduced_figures
   python code/plotting/make_xy8_extended_analyses.py --calibration data/raw/nv/xy8_scenario_nominal.json --outdir outputs/reproduced_figures
   python code/plotting/make_xy8_uncertainty_attribution.py --calibration data/raw/nv/xy8_scenario_nominal.json --samples-csv outputs/reproduced_figures/xy8_monte_carlo_samples.csv --outdir outputs/reproduced_figures
   python code/plotting/make_xy8_topology_figure.py --mirror-data-dir data/processed
   python code/plotting/make_xy8_environment_topology.py --processed-dir data/processed --outdir outputs/reproduced_figures
   python code/analyze_xy8_resonant_topology.py --relative-limit 1.10 --frequency-khz 200
   python code/analyze_xy8_topology_refinement.py --frequency-khz 200 --relative-limit 1.10 --mirror-outdir outputs/reproduced_figures
   python code/analyze_xy8_topology_persistence.py --frequency-khz 200 --mirror-outdir outputs/reproduced_figures
   python code/recommend_xy8_control.py --frequency-khz 200 --pulse-fidelity 0.997 --t2-us 300 --json-out outputs/reproduced_figures/xy8_control_recommendation.json
   python code/utilities/export_figure_source_data.py
   ```

5. Optional repository smoke check:

   ```bash
   python tests/smoke_check.py
   ```

Notes:

- The curvature workflows run directly on the archived second-order band-curvature files in `data/raw/`.
- The NV XY8 script is configured to run from the explicit scenario JSON files and therefore does not require `nvratemodel`.
- `code/plotting/make_fig_xy8_qre.py` is the canonical implementation. The similarly named file under `scripts/` is only a compatibility entry point.
- `xy8_prospective_validation.csv` is generated before measurement entry. Preserve its protocol digest when adding observations so predictions remain auditable.
- `xy8_prospective_power_summary.json` uses one-second independent Gaussian field-estimate batches and a sample-standard-deviation estimator. Its reported times are prospective conditional calculations, not measured acquisition times; correlated noise, drift, slope-calibration uncertainty, and model-form error can increase the requirement.
- The topology-refinement analysis uses four-neighbour connectivity on the sampled `(tau, N)` grid and checks the result against eight-neighbour connectivity. Its medium and fine resonance brackets measure numerical grid convergence; they do not replace experimental uncertainty or hardware-uncertainty tests.
- The threshold-persistence analysis reuses the converged fine grid at one-percentage-point increments from 1% to 100%. Component identities are propagated by grid-cell overlap; merger thresholds are therefore resolved only to the requested threshold step.
- The environment-conditioned Stage-6 figure consumes the frozen discovery and confirmation artifacts; it does not rescan, fit, or retune the model. Its displayed maps use the preregistered axial-strain/Jarmola operational slice, while its topology statement summarizes all 226 selected fine-grid confirmations.
- Both literature datasets, their selection rationales, parameter provenance, and digitization notes are frozen under `data/raw/nv/literature_validation`. Neither comparison tunes a parameter to the measured curve. The Pham holdout tests the complete depth-ranking trend at equal total pulse count; the Farfurnik holdout tests conventional XY8 contrast loss in the short-free-evolution limit.
- The categorical phase diagram varies only pulse fidelity and `T2_1`; scenario markers are projections because contrast, overhead, photon rate, and coherence exponents remain nominal in the map.
- The archived Trotter circuit script depends on `qiskit`. If `qiskit` is unavailable, the core curvature and NV analyses remain reproducible, and the reproduction driver will skip that optional step.
