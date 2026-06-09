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
