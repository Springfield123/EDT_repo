# Independently calibrated XY8 holdout action plan

## Objective and validation boundary

The objective is to determine whether independently measured hardware properties predict the depth-dependent conventional-XY8 trend without tuning the model to that trend. The three Farfurnik holdout depths, `N = 1, 9, 73` (`8, 72, 584` pi pulses), are reserved for the final comparison and must not be used to estimate, select, or reject calibration parameters.

Before acquiring calibration data, freeze the measurement list, analysis conventions, acceptance criteria, and holdout exclusions. This prospective step prevents the calibration workflow from being changed after the holdout result is seen. The frozen Stage 1 protocol is `data/raw/nv/calibration/xy8_spam_calibration_protocol.json`.

## Delivery sequence and gates

### Stage 1 - preparation and readout calibration (software implemented; measurements pending)

**Purpose:** replace ideal state preparation and optical readout with independently measured quantities before attributing high-depth disagreement to XY8 control.

Acquire interleaved bright (`m_s = 0`) and dark (`m_s = 1`) fluorescence references, together with zero-depth preparation/readout checks in the `+X`, `-X`, `+Y`, and `-Y` quadratures and a mixed-population midpoint reference. Retain individual shots and timestamps so drift and shot-to-shot variance remain observable. Randomize setting order and record microwave frequency and power, laser power, and readout duration. The paired signs are required to separate quadrature visibility from measurement offset.

The implementation in `code/analysis/xy8_spam_model.py` provides:

- a physical Bloch-vector initial state;
- calibrated preparation and analysis rotations, including angle, phase, and axis errors;
- bright/dark fluorescence references with standard errors;
- prediction of expected counts and normalized spin signal after an arbitrary two-level control unitary; and
- a protocol validator that rejects populated prospective rows or use of a holdout depth.

The shot-level estimator in `code/analysis/xy8_spam_calibration.py` reads the acquisition schema in `data/raw/nv/calibration/xy8_spam_shots_template.csv`, estimates bright/dark references and effective X/Y visibility and offsets, propagates the group-mean covariance, evaluates sampling, reference-separation, drift, midpoint, and covariance gates, and computes a combined SHA-256 digest of the protocol and raw shots. `code/calibrate_xy8_spam.py` is the command-line entry point. It writes a Stage 1 summary but cannot label absent measurements as calibrated.

Populate `data/raw/nv/calibration/xy8_spam_calibration_template.json` only from the frozen Stage 1 measurements. Preparation and analysis errors are not separately identifiable from the paired zero-depth fluorescence values, so their independently fitted physical estimates and covariance must first be entered in `data/raw/nv/calibration/xy8_spam_physical_characterization_template.json`. Compute a SHA-256 digest from all retained calibration sources, change the status to `calibrated`, and archive both the raw sources and completed calibration. Do not create placeholder fitted values.

**Gate:** bright and dark references are statistically distinguishable; their drift remains below 20% of their separation during an acquisition; the calibrated model reproduces every zero-depth check within combined 95% uncertainty; parameter covariance is finite; and no holdout observation has entered the fit. If this gate fails, improve the calibration experiment rather than advancing to the depth holdout.

The separate prospective power analysis treats one independent acquisition batch as an approximately Gaussian field estimate and estimates sensitivity from the variance across batches. It reports the probability of recovering the operating window, rejecting shallow and overdriven controls, resolving timing selectivity, establishing 10% equivalence of `N=16` and `N=20`, and selecting the exact `N=20` optimum. Its acquisition-time estimates are conditional on one-second independent batches and exclude model-form error, drift, inter-batch correlation, and signal-slope uncertainty.

### Stage 2 - ensemble amplitude and detuning calibration

Measure Rabi-frequency or pulse-area distributions and microwave detuning distributions across the sensed ensemble. Record the pulse duration, mean offset, distribution width and shape, spatial weighting, and resolved hyperfine populations. Convert detuning in hertz to angular frequency only at the model boundary and record that convention in the manifest.

**Gate:** the declared distributions reproduce independent Rabi and spectroscopy calibration curves within uncertainty, and changing quadrature or Monte Carlo resolution does not materially change the predicted XY8 response.

### Stage 3 - error timescale classification

Use repeated, timestamped Rabi/Ramsey calibrations and randomized sequence blocks to separate quasi-static inhomogeneity, pulse-to-pulse correlation, and slow drift. Estimate autocorrelation or Allan-deviation timescales rather than assigning every error to an independent distribution.

**Gate:** each error source has a declared stochastic model and timescale supported by calibration data. Competing plausible classifications should remain explicit uncertainty scenarios.

### Stage 4 - short-time coherence calibration

Measure the coherence envelope on the time range occupied by the holdout sequences, using independent Ramsey, echo, and short dynamical-decoupling scans that do not include the three validation depths. Preserve the measured envelope or a predeclared fit with confidence intervals; avoid inferring an additional loss term from the holdout decline.

**Gate:** the coherence model passes residual and repeatability checks on its own calibration data and is stable over the acquisition period relevant to validation.

### Stage 5 - freeze a calibration manifest

Create one machine-readable manifest containing all parameter values, uncertainties, units, stochastic assumptions, data-source digests, model and code versions, random seeds, and excluded validation observations. Record which values were measured directly, derived from those measurements, or adopted from literature.

**Gate:** the manifest validates against its schema, every source digest resolves to an archived file, and the full prediction can be reproduced without reading the holdout observations.

### Stage 6 - untouched three-point prediction

Load the frozen manifest and predict the normalized response at `N = 1, 9, 73` before revealing or importing the measured values. Then compare the prediction with the frozen holdout using normalized RMSE, rank correlation, pointwise uncertainty coverage, and any predeclared half-contrast metric. Report both agreement and failure without retuning.

**Gate:** the comparison script verifies that the manifest predates holdout evaluation and produces an immutable result artifact. A failed transfer test is still an informative result: it identifies missing physics for a separately calibrated follow-up rather than authorizing a holdout fit.

## Current Stage 1 status

- Prospective protocol frozen with no measurements entered.
- Holdout depths explicitly excluded from parameter estimation.
- SPAM state, pulse, and fluorescence model implemented.
- Calibration template created but intentionally unpopulated.
- Focused unit tests cover ideal and imperfect behavior, serialization, invalid inputs, and holdout-leakage rejection.
- A shot-level estimator, covariance calculation, acceptance gates, source digest, CSV schema, and command-line entry point are implemented.
- Independent experimental acquisition, parameter estimation, covariance calculation, and source-data digest remain to be completed on the calibrated NV platform.

## Decision rule

Advance only after each calibration gate passes. Do not use disagreement at the final three holdout points to revise a parameter already frozen in the manifest. Any new physical effect should be calibrated with a separate experiment and evaluated in a new, versioned validation round.
