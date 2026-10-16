# EDT Repository Update and Handoff

**Date:** 2026-10-16  
**Repository:** `edt_repo`  
**Branch:** `main`  
**Base commit:** `8d9f3ec` (`Update K3C60 EPW windows and add QE inputs`)  
**Primary scope of this handoff:** NV-centre XY8 sensing, EDT connected-region analysis, uncertainty, validation, environmental conditioning, and NATO presentation assets

## Executive status

The NV XY8 study has progressed from a coarse optimization plot to a resource-aware, topology-based control workflow. The current codebase can:

- regenerate the canonical 200 kHz XY8 sensitivity landscape;
- resolve the nominal optimum and its 10% near-optimal operating region;
- count and track connected near-optimal regions across thresholds;
- identify the modeled resource that creates the finite optimum;
- quantify calibration uncertainty and robust fixed-control performance;
- recommend control depth from measured pulse fidelity, coherence time, and target frequency;
- freeze a seven-setting prospective validation protocol and estimate its statistical power;
- benchmark the scalar model and sequence-specific coherent model against frozen literature data;
- condition the EDT landscape on published strain and temperature inputs; and
- generate the figures, source data, short paper, NATO presentation, and speaker notes used for communication.

The central result remains a **literature-anchored model prediction**, not a calibrated device measurement. At 200 kHz, the nominal model selects

- model timing parameter `tau = 1.25 us`;
- `N = 20` XY8 blocks;
- `160` pi pulses;
- sequence duration `400 us`, before initialization and readout overhead; and
- a 10% operating window from `N = 10` through `N = 34`.

The model uses `N` for complete XY8 blocks, so `N_pi = 8N`. In this repository, `tau` is half the centre-to-centre interpulse gap; the full gap at 200 kHz is approximately `2.5 us`.

## Repository state at handoff

The working tree is intentionally **not clean**:

- 29 tracked files are modified.
- 276 paths are untracked.
- No files were staged, committed, deleted, or reorganized during preparation of this handoff.

The untracked set includes new analysis modules, tests, frozen datasets, processed results, figures, presentations, temporary build products, macOS metadata, and office lock files. Do not run `git add .` or a bulk cleanup. Build the next commit from an explicit allowlist after deciding which raw, processed, and presentation artifacts belong in the release.

## Verification completed

The focused XY8 suite was run on 2026-10-16 with:

```bash
env MPLCONFIGDIR=/private/tmp/edt_pytest_mpl \
  conda run -n base python -m pytest -q \
  tests/test_xy8_coherent_control.py \
  tests/test_xy8_coherent_eta.py \
  tests/test_xy8_environment_coherence.py \
  tests/test_xy8_environment_conditioning.py \
  tests/test_xy8_environment_figure.py \
  tests/test_xy8_environment_scan_config.py \
  tests/test_xy8_environment_score.py \
  tests/test_xy8_environment_topology.py \
  tests/test_xy8_experimental_depth_topology.py \
  tests/test_xy8_literature_validation.py \
  tests/test_xy8_model_benchmark.py \
  tests/test_xy8_nv_environment_literature.py \
  tests/test_xy8_paper_pipeline.py \
  tests/test_xy8_prospective_power.py \
  tests/test_xy8_published_topology.py \
  tests/test_xy8_resonant_topology.py \
  tests/test_xy8_spam_calibration.py \
  tests/test_xy8_spam_model.py \
  tests/test_xy8_specific_literature_validation.py \
  tests/test_xy8_uncertainty_attribution.py
```

Result:

```text
212 passed in 30.86s
```

This verifies the current working tree, not the base commit. A clean-clone reproduction has not yet been demonstrated because many required NV inputs and artifacts are still untracked.

## Scientific results currently supported

### Canonical 200 kHz recommendation

The canonical implementation is `code/plotting/make_fig_xy8_qre.py`. The similarly named script under `scripts/` is now only a compatibility wrapper.

The nominal result is:

| Quantity | Result |
|---|---:|
| Best timing parameter | `1.25 us` |
| Best depth | `N = 20` |
| Total pi pulses | `160` |
| Sequence duration | `400 us` |
| 10% depth window | `N = 10-34` |
| Boundary status | Resolved interior optimum |
| Calibration digest | `fe2b8d2c6faee1d4` |

The high-depth `N = 64` setting has a relative score of `1.7243`, or a 72.4% penalty relative to the optimum. This supports the conclusion that more control eventually becomes counterproductive under the nominal assumptions.

### Lower-pulse operating choice

`N = 16` uses 128 pi pulses, which is 20% fewer than `N = 20`. Its modeled penalty is 0.973%. This is the current lower-complexity alternative for operational planning; it is not the strict minimum of the nominal score.

### Resource mechanism

The resource ablation gives:

| Model condition | Best N | Interpretation |
|---|---:|---|
| Nominal | 20 | Finite interior optimum |
| Pulse errors removed | 128 | Optimum reaches the tested depth boundary |
| Decoherence removed | 20 | Score improves, depth remains unchanged |
| Timing overhead removed | 20 | Score improves slightly, depth remains unchanged |

Within the current scalar model, accumulated pulse infidelity is the main resource that sets the nominal upper-depth limit. This is conditional on the model and should not be presented as a universal property of XY8.

### Hardware scenarios and decision rule

The conservative, nominal, and optimistic scenarios select approximately `N = 8`, `20`, and `64`, respectively. The recommender in `code/analysis/xy8_control_recommender.py`, exposed through `code/recommend_xy8_control.py`, converts

```text
(target frequency, per-pulse fidelity, T2)
```

into resonant timing, recommended depth, total pulse count, the 10% depth window, a lower-pulse alternative, and boundary warnings.

The phase diagram varies pulse fidelity and `T2` while holding contrast, photon rate, overhead, the coherence exponent, and other calibration terms at nominal values. It is a controlled decision map rather than a complete device model.

### Monte Carlo uncertainty and robust choice

The seeded study contains 400 simulated hardware conditions.

- 359 of 400 optima, or 89.75%, fall within the nominal `N = 10-34` window.
- `N = 16` is the most frequent sample-specific optimum: 90 of 400 cases, or 22.5%.
- `N = 20` is the best fixed choice before the exact hardware state is known.
- For fixed `N = 20`, mean regret is 4.08% and 95th-percentile regret is 15.36%.
- The nominal, minimum-mean-regret, and minimum-95th-percentile-regret criteria all select `N = 20` at `tau = 1.25 us`.

The distinction matters: the most frequently selected optimum is not automatically the safest preselected setting, because frequency of selection does not measure the penalty when that setting is wrong.

### Uncertainty attribution

Pulse fidelity is the highest-value calibration measurement under the assumed independent uncertainty ranges and the scalar-fidelity model.

- Spearman correlation with selected depth: `rho = 0.967`.
- Expected reduction in the mean penalty of a fixed control: `3.88` percentage points.
- Bootstrap 95% interval for that reduction: `3.22-4.64` percentage points.
- Learning pulse fidelity changes the selected recommendation in 78.75% of the simulated cases.

Coherence calibration has a smaller decision value in this particular uncertainty model. Contrast, photon rate, and overhead scale the score but do not change the selected depth when varied as independent multiplicative factors in the current formulation. This attribution must not be generalized beyond the assumed ranges and model structure.

### Connected-region topology

The refined nominal grid contains 1,922 timing points and 32 candidate depths. The near-optimal set has one connected fundamental-resonance component from a 1% through a 100% penalty threshold.

At the 10% threshold:

- `beta_0 = 1` under both four- and eight-neighbour connectivity;
- the depth range is `N = 10-34`;
- the refined timing range is approximately `1.24575-1.25450 us`;
- the estimated timing width is `0.009 us`; and
- the component remains inside the tested timing and depth boundaries.

The higher resonance near `3.75 us` is about 134% worse than the global minimum and therefore does not enter the nominal 1-100% persistence range.

In this project, topology means connected components of a near-optimal parameter set. It does not mean a topological material or a topological quantum phase.

### Prospective validation and power

The frozen protocol contains seven control settings:

```text
N = 8, 10, 16, 20, 34, 64 at resonance
N = 20 at a model-selected off-resonant timing
```

The protocol remains unmeasured. Its CSV intentionally leaves measured value, uncertainty, repeat count, and pass/fail fields blank.

The power analysis shows that the operating-window claim is easier to test than the exact `N = 20` optimum:

- The current 3-to-5-batch allocation has about 75.5% power for placing the measured best setting inside the window and 97.0% power for timing selectivity, but only 14.1% power for selecting the exact optimum.
- Sixteen independent batches per resonant control reach about 90% power for the operating-window claim under the assumed measurement model.
- A core design using 160 independent batches for the six resonant settings reaches about 99.6% power for the window and 95.1% for the shallow/overdriven comparison.
- Resolving the modeled 0.97% ordering between `N = 16` and `N = 20` is much harder and reaches the 25,000-batch search limit in the current calculation.
- Demonstrating 10% equivalence between `N = 16` and `N = 20` requires about 1,600 batches per setting under the same assumptions.

These acquisition times are conditional calculations based on independent one-second Gaussian field-estimate batches. Drift, correlations, calibration-slope uncertainty, and model-form error are not included.

### Literature validation

The independent-data program now has four distinct roles.

1. **Pham et al. depth trend:** The no-fit scalar model reproduces resonant timing, low-depth improvement, and high-depth decline. The observed central optimum lies inside the predicted 10% window, but global rank agreement is only moderate. At the 10% threshold, the bootstrap probability of one measured connected region is 0.546, so the sparse experimental boundaries remain uncertain.
2. **Model benchmark:** The full resource model gives the closest optimum. A coherence-only model has the lowest normalized RMSE but misses the finite optimum and has little rank agreement. Pulse fidelity is necessary to recover the high-depth decline, but the current evidence does not show a large advantage of the full score over every simpler alternative.
3. **Farfurnik et al. conventional XY8 check:** The sequence-specific coherent propagator reproduces the direction of contrast loss but overpredicts retained contrast. Its normalized RMSE is 0.614, worse than the scalar baseline at 0.288. This freezes a negative result and points to independently calibrated loss, error correlations, SPAM, drift, or ensemble physics.
4. **DeVience and Loretz topology checks:** The sparse DeVience timing-by-depth surface shows separate resonance regions and threshold-dependent mergers. The Loretz spectrum shows fundamental and spurious harmonic regions entering at different thresholds. These support connected-region analysis as a spectral-structure diagnostic; they do not validate the 200 kHz optimum.

### Environment-conditioned EDT

Published spin-strain coefficients, temperature-dependent zero-field splitting, and spin-phonon relaxation data were frozen before the environment scan.

- 1,496 discovery surfaces were analyzed.
- 226 preregistered or supplementary surfaces were confirmed on medium and fine grids.
- All 226 fine-grid cases have `beta_0 = 1` under four-neighbour connectivity.
- Four- and eight-neighbour counts agree in every confirmed case.
- The fine-grid best-depth distribution is `N = 12` for 2 cases, `14` for 48, `16` for 60, and `20` for 116.
- In the displayed fixed-carrier example at 330 K and +20 microstrain, detuning moves the recommendation to `N = 12`.
- When the microwave carrier is retracked, `N = 20` remains optimal across the displayed axial-strain slice.

The supported conclusion is that the tested environment deforms and moves the operating region without splitting it. The result is conditional on the frozen parameter laws and tested range.

## Code and artifact map

### Canonical model and decision code

- `code/analysis/nvratemodel_api.py` - scalar resource model and coherent-control integration.
- `code/analysis/xy8_coherent_control.py` - deterministic two-level XY8 propagator, finite pulse duration, detuning, and quasi-static error distributions.
- `code/plotting/make_fig_xy8_qre.py` - sole canonical figure implementation.
- `scripts/make_fig_xy8_qre.py` - compatibility wrapper only.
- `code/analysis/xy8_control_recommender.py` - reusable recommendation logic.
- `code/recommend_xy8_control.py` - command-line recommendation interface.

### Topology

- `code/analysis/xy8_resonant_topology.py`
- `code/analysis/xy8_topology_persistence.py`
- `code/analyze_xy8_resonant_topology.py`
- `code/analyze_xy8_topology_refinement.py`
- `code/analyze_xy8_topology_persistence.py`
- `code/plotting/make_xy8_topology_figure.py`

### Robustness, uncertainty, and prospective testing

- `code/plotting/make_xy8_robustness_figures.py`
- `code/plotting/make_xy8_extended_analyses.py`
- `code/plotting/make_xy8_uncertainty_attribution.py`
- `code/plotting/make_xy8_prospective_validation.py`
- `code/analysis/xy8_prospective_power.py`
- `code/plotting/make_xy8_prospective_power.py`

### Literature validation

- `code/analysis/xy8_literature_validation.py`
- `code/analysis/xy8_specific_literature_validation.py`
- `code/analysis/xy8_experimental_depth_topology.py`
- `code/analysis/xy8_devience_topology.py`
- `code/analysis/xy8_loretz_spurious_topology.py`
- `data/raw/nv/literature_validation/` - frozen selections, manifests, digitized data, and extraction records.

### SPAM calibration preparation

- `code/analysis/xy8_spam_model.py`
- `code/analysis/xy8_spam_calibration.py`
- `code/calibrate_xy8_spam.py`
- `data/raw/nv/calibration/` - protocol, shot template, physical-characterization template, and calibration manifest template.

No device-specific shot file has been collected. The SPAM calibration pipeline is implemented and tested but remains unpopulated with actual measurements.

### Environment and first-principles inputs

- `code/analysis/xy8_nv_environment_literature.py`
- `code/analysis/xy8_environment_conditioning.py`
- `code/analysis/xy8_environment_coherence.py`
- `code/analysis/xy8_environment_score.py`
- `code/analysis/xy8_environment_topology.py`
- `data/raw/nv/environment_literature/` - frozen coefficients, laws, validation data, scan configuration, and digests.
- `code/plotting/make_xy8_environment_topology.py`

### Principal communication artifacts

- `Nato_paper/EDT_XY8_IST_SET_226_Short_Paper.docx` - symposium short paper source.
- `Nato_paper/EDT_XY8_IST_SET_226_Short_Paper.pdf` - current rendered paper.
- `outputs/presentations/EDT_robust_quantum_sensing_20min_NATO_template_v2.pptx` - current 17-slide NATO presentation.
- `outputs/presentations/EDT_robust_quantum_sensing_NATO_speaker_notes.docx` - complete 20-minute speaker guide.
- `outputs/reproduced_figures/` - presentation and paper figures plus machine-readable outputs.

The short paper predates some of the final environment-conditioned presentation work and should be checked for consistency before submission.

## Reproduction

### Preferred fresh environment

```bash
conda env create -f environment.yml
conda activate edt-codebase
```

The verified test run used the existing `base` conda environment with Python 3.12 and pytest 9.1.1. The declared project environment uses Python 3.11 and pytest 8 or newer.

### Rebuild the NV evidence package

```bash
python code/run_reproduction_pipeline.py --nv-only
```

Expected output locations:

- figures and mirrored tables: `outputs/reproduced_figures/`;
- processed copies: `data/processed/`; and
- Figure 5 source data: `data/figure_source_data/Fig5_source_data.csv`.

Important: the environment figure consumes frozen discovery and confirmation artifacts under `data/processed/`. A clean-clone run will not be reliable until the release decision records and tracks the necessary frozen inputs or adds the missing generation stages to the pipeline.

### Run the recommender

```bash
python code/recommend_xy8_control.py \
  --frequency-khz 200 \
  --pulse-fidelity 0.997 \
  --t2-us 300 \
  --json-out outputs/reproduced_figures/xy8_control_recommendation.json
```

### Run SPAM calibration after measurements exist

```bash
python code/calibrate_xy8_spam.py \
  --shots path/to/spam_shots.csv \
  --physical-characterization path/to/physical_characterization.json
```

Do not use the three-point literature holdout to tune these calibration values.

## Known limitations and guardrails

1. **No device calibration:** Current nominal values are literature-anchored assumptions, not measurements from a particular NV platform.
2. **Relative score:** `eta` ranks controls. It is not an absolute magnetic sensitivity in tesla per square-root hertz.
3. **Scalar default:** The main recommendation still uses an independent per-pulse survival model. The coherent propagator exists, but the frozen Farfurnik holdout showed that it does not yet improve transfer.
4. **Missing independently calibrated physics:** Short-time loss, correlated or drifting errors, ensemble inhomogeneity, and preparation/readout imperfections remain candidates for the holdout discrepancy.
5. **Sparse literature surfaces:** DeVience is digitized and sparse, with digitization uncertainty but no published experimental error bars for the selected figure.
6. **Topology is model conditional:** Connected-region stability has strong numerical checks, but experimental and full model-form uncertainty have not been propagated through every two-dimensional topology calculation.
7. **Prospective power is conditional:** The batch model excludes drift, cross-setting correlation, signal-slope uncertainty, and model-form error.
8. **Presentation scope:** The current work demonstrates the dynamic-topology component of EDT in a single-spin sensor. It does not yet demonstrate entanglement-enhanced sensing.
9. **Repository release state:** The focused tests pass in the current working tree and the expected output artifacts are present, but the full pipeline was not rerun during this handoff audit. Many required files also remain untracked, so a clean-clone pipeline acceptance test is still required.

## Recommended next actions

### Priority 0  Preserve and release the current state safely

1. Create a release allowlist for code, tests, frozen raw data, processed decision artifacts, figures, paper, and presentation deliverables.
2. Update `.gitignore` for `.DS_Store`, `.tmp/`, office lock files, caches, and genuinely disposable render intermediates.
3. Do not discard or overwrite the present working tree before the allowlist is reviewed.
4. Commit the NV work in logical groups rather than as one opaque commit.
5. Clone the resulting branch into a new directory and run the NV pipeline and all focused tests from that clean clone.

### Priority 1  Perform the prospective experiment

1. Collect bright, dark, and zero-depth shot-level SPAM measurements.
2. Record pulse fidelity, detuning distribution, finite pulse duration, coherence, contrast, photon rate, and overhead with source metadata.
3. Freeze the calibration manifest before examining the seven holdout settings.
4. Use at least 16 independent batches per resonant setting to target the operating-window claim; use the larger core design if the shallow and overdriven comparisons are central to the paper.
5. Treat `N = 16` and `N = 20` as practically close unless the experiment is powered for their sub-1% modeled difference.

### Priority 2  Improve sequence-specific transfer without fitting the holdout

1. Calibrate preparation and readout independently.
2. Measure amplitude and detuning distributions across the ensemble.
3. Determine whether errors are quasi-static, correlated, or drifting.
4. Measure the short-time coherence envelope.
5. Freeze these inputs, then rerun the Farfurnik and prospective comparisons without adjusting parameters to improve agreement.

### Priority 3  Strengthen external validation

1. Obtain a second dataset with several XY8 depths, pulse timing, and uncertainty bars.
2. Request or obtain denser timing-by-depth data from the Tsunaki authors for a genuine two-dimensional test.
3. Keep DeVience as a sparse topology transfer test and Loretz as a spurious-passband mechanism test.
4. Report negative or partial transfer results without retuning the model.

### Priority 4  Extend EDT beyond the single-spin baseline

Only after the single-spin model is independently calibrated, add multi-spin or entangled resources. The comparison should evaluate whether an entangled protocol improves the connected operating region after including state-preparation cost, shorter coherence, control error, and readout complexity. This is the step that would support the entanglement component of EDT rather than only the dynamic-topology component.

## Handoff checklist

- [x] Canonical XY8 implementation selected.
- [x] Nominal optimum and 10% window regenerated.
- [x] Scenario, frequency, Pareto, resource-ablation, and uncertainty analyses implemented.
- [x] Hardware-to-control recommender implemented.
- [x] Prospective validation and power analysis frozen.
- [x] Connected-region refinement and persistence implemented.
- [x] Literature comparisons frozen and documented.
- [x] Sequence-specific coherent model implemented and evaluated on the frozen holdout.
- [x] SPAM calibration code and templates implemented.
- [x] Temperature- and strain-conditioned topology confirmed on the fine grid.
- [x] NATO presentation and speaker notes prepared.
- [x] Focused XY8 tests pass in the current working tree.
- [ ] Curated release commit created.
- [ ] Clean-clone pipeline acceptance test completed.
- [ ] Device-specific calibration measurements collected.
- [ ] Seven-setting prospective experiment completed.
- [ ] Independent dense XY8 timing-by-depth dataset obtained.
- [ ] Entangled or multi-spin extension calibrated and tested.
