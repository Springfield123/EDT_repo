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
