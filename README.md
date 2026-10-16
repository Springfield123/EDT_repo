# EDT Codebase

A paper-companion research repository for **quantitative EDT diagnostics** across three reproducible workflows: curvature-window extraction, plaquette-dynamics scoring, and NV sensing optimization.

The current NV XY8 development status, scientific conclusions, verification record, release cautions, and next-step handoff are documented in [docs/edt_repo_update_and_handoff_2026-10-16.md](docs/edt_repo_update_and_handoff_2026-10-16.md).

This updated version is grounded not only in the manuscript draft but also in the uploaded supplementary information and the actual figure-generation scripts you provided. It now includes the core paper assets directly in the repository:

1. **K$_3$C$_{60}$ local-fit EDT analysis** from the uploaded `superfinger_k3c60_edt_localfit_v6.py` script, using the provided `.agr` data file.
2. **Four-qubit plaquette / Trotter EDT dynamics** from `demo_edt_gyro_trotter_4q_fig34_clean.py`.
3. **NV XY8 sensing workflow** from the canonical `code/plotting/make_fig_xy8_qre.py` together with `code/analysis/nvratemodel_api.py`.
4. **Figure 1 concept graphic** from `fig1_edt_framework_camera_ready.py` as a narrative schematic rather than an evidence-bearing diagnostic.
5. The uploaded manuscript and supplementary text under `docs/`.

## What changed relative to the starter scaffold

The earlier scaffold abstracted the paper into reusable modules. This version keeps that reusable structure **and** adds the actual paper-side scripts and input data so the repository can serve as a true companion codebase for the manuscript.

## Repository layout

```text
scripts/
  fig1_edt_framework_camera_ready.py
  superfinger_k3c60_edt_localfit_v6.py
  demo_edt_gyro_trotter_4q_fig34_clean.py
  make_fig_xy8_qre.py
  nvratemodel_api.py

data/
  k3c60/
    tmp.agr

docs/
  main_paper_draft.tex
  supplementary_information.tex
  edt_quantitative_positioning.md

src/edt/
  analysis/
  nv/
  quantum/
  plotting/
  paper/               # paths to manuscript assets

examples/
  run_curvature_demo.py
  run_plaquette_demo.py
  run_nv_demo.py
  run_uploaded_assets.py
```

## Mapping to the paper

### 1) K$_3$C$_{60}$ curvature and EDT window

The supplementary text defines the EDT window through paired quadratic fits and a residual threshold in momentum space. The uploaded K$_3$C$_{60}$ script implements exactly that workflow: it parses `.agr` datasets, rescales curvature using a zero-point displacement factor, fits local parabolas to paired branches, computes normalized residuals, and extracts both a core window and an EDT window.

Use the canonical paper workflow:

```bash
python scripts/superfinger_k3c60_edt_localfit_v6.py
```

By default this script expects the AGR file in the working directory as `tmp.agr`. If you run from the repository root, either copy/link `data/k3c60/tmp.agr` into the root or edit `AGR_FILE` in the script to point to `data/k3c60/tmp.agr`.

### 2) Four-qubit plaquette EDT dynamics

The SI section on quantum-circuit realization is implemented directly by the uploaded 4-qubit script. It builds a plaquette Hamiltonian

```math
H = \Omega Z_0 + g Z_0 X_3 + J_{01} Z_0 Z_1 + J_{12} Z_1 Z_2 + h_x X_0 + \lambda Z_0 Z_1 Z_2 Z_3,
```

computes exact and Trotterized dynamics, OTOCs, lambda sweeps, and figure panels for a quantitative EDT score based on mid-time OTOC weight and long-time correlation.

Use:

```bash
python scripts/demo_edt_gyro_trotter_4q_fig34_clean.py
```

This requires `qiskit`.

### 3) NV XY8 sensing workflow

The NV scripts implement a sensitivity proxy based on readout rate, contrast, coherence decay, pulse fidelity, filter gain, and duty-cycle overhead. The repository-local `nvratemodel_api.py` exposes the paper-facing functions used in the figure scripts.

Use:

```bash
python code/plotting/make_fig_xy8_qre.py
```

To swap in measured readout/coherence/pulse/ensemble values, pass a JSON calibration file such as
`data/nv/xy8_calibration_template.json`:

```bash
python code/plotting/make_fig_xy8_qre.py --calibration data/raw/nv/xy8_calibration_template.json
```

If you do not yet have experimental data, you can run three built-in literature-anchored scenario files:

```bash
python code/plotting/make_fig_xy8_qre.py --calibration data/raw/nv/xy8_scenario_conservative.json
python code/plotting/make_fig_xy8_qre.py --calibration data/raw/nv/xy8_scenario_nominal.json
python code/plotting/make_fig_xy8_qre.py --calibration data/raw/nv/xy8_scenario_optimistic.json
```

Regenerate the complete NV presentation evidence package with one command:

```bash
python code/run_reproduction_pipeline.py --nv-only
```

These presets are intended for modeled sensitivity comparisons and EDT operating-point optimization only. They should be described as `projected`, `modeled`, or `literature-anchored`, not experimentally validated.
For these scenario files, the readout block can use a single `effective_detected_rate_cps` parameter to avoid over-interpreting `Rph_1 * N_eff` as a physically calibrated ensemble rate.

The NV-only command also generates the 10% EDT operating window, a resource--performance Pareto frontier, a prospective seven-setting validation protocol and power analysis, a 50--500 kHz frequency sweep, seeded Monte Carlo calibration uncertainty, hardware-uncertainty attribution, a frozen temperature/strain-conditioned EDT summary, and several frozen literature checks. The environment study shows whether the connected region and recommended depth survive preregistered timing-grid refinement; its display is generated from frozen artifacts rather than a refitted scan. The power analysis distinguishes the modest sample size needed to test the operating window from the much larger requirement for resolving the 0.97% nominal difference between `N=16` and `N=20`. The attribution analysis reports which single hardware measurement most reduces the expected cost of selecting a fixed control. Pham et al. (2012) provides a one-dimensional equal-total-pulse trend and uncertainty-aware depth-topology check. Farfurnik et al. (2015) remains a three-depth pulse-error check. Four experimental XY8 timing sweeps from DeVience et al. (2015) form a sparse frequency-by-depth surface that tests whether two nuclear resonances separate and merge as the response threshold changes; it does not test the 200 kHz optimum. Loretz et al. (2015) is analyzed separately to determine whether connected-region persistence exposes experimental second- and fourth-harmonic XY8 passbands, with the paper's simulations excluded. Every selection, offset/baseline correction, normalization, and digitization rule is frozen under `data/raw/nv/literature_validation`. A predeclared request for denser experimental XY8 timing sweeps from Tsunaki et al. remains stored in `docs/tsunaki_xy8_experimental_data_request.md`.

The prospective protocol is explicitly unmeasured: its CSV contains frozen model predictions and empty measurement, uncertainty, and pass/fail columns for later experimental entry.

Convert measured target frequency, pulse fidelity, and single-block coherence into a control recommendation with:

```bash
python code/recommend_xy8_control.py --frequency-khz 200 --pulse-fidelity 0.997 --t2-us 300
```

Optional `--pulse-fidelity-sigma` and `--t2-sigma-us` inputs add seeded modal and 95th-percentile robust recommendations. The decision rule always reports when non-input resources remain nominal assumptions or when the optimum/window reaches the tested sweep boundary.

### 4) EDT concept figure

The uploaded `fig1_edt_framework_camera_ready.py` script produces a conceptual multi-panel EDT figure used to organize the paper narrative. It is illustrative and should not be cited as evidence for EDT by itself.

Use:

```bash
python scripts/fig1_edt_framework_camera_ready.py
```

## Install

Base install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional extras:

```bash
pip install -e .[quantum]
```

The NV workflow depends on `nvratemodel`, which is not bundled here.

## Notes

- The uploaded figure scripts are preserved largely as provided so the repository stays close to the manuscript workflow.
- The evidence-bearing EDT claims in this repository come from the quantitative scripts, not from the concept figure.
- A draft claim-tightening guide is provided in [docs/edt_quantitative_positioning.md](</C:/Users/agahu/Dropbox/NRC/EDT/code/docs/edt_quantitative_positioning.md>).
- The reusable package code under `src/edt/` remains useful for later cleanup, refactoring, and publication-ready modularization.
- The bundled `tmp.agr` file is the raw K$_3$C$_{60}$ curvature input currently available in this conversation.


Latex manuscript assets are now organized as:

```text
main.tex
sections/
  01_introduction.tex
  02_edt_definition.tex
  03_curvature_diagnostic.tex
  04_k3c60_analysis.tex
  05_minimal_plaquette_model.tex
  06_operator_spreading_and_robustness.tex
  07_nv_xy8_case_study.tex
  08_discussion.tex
  09_methods.tex
supplement/
  supplement.tex
  S1_curvature_derivation.tex
  ...
  S10_reproducibility.tex
figures/
  main/
  supplementary/
references.bib
```
