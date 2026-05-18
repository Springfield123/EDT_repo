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
