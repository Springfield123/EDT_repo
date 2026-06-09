# Data directory guide

This folder separates the companion repository data into four layers.

- `raw/`: archived manuscript input data. The `*.agr` files are the raw mode-resolved second-order band-curvature exports used for the curvature-window analyses. Their paired columns are `(k, \partial^2 E_n / \partial Q_\alpha^2)` in the native units encoded by the archive metadata.
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
