# XY8 coherent-control model action plan

## Objective

Replace the depth-independent stochastic approximation `Fpi^(8N)` with an opt-in, sequence-specific XY8 response while keeping the existing scalar pipeline as a reproducible baseline. Each stage has an acceptance gate; no new effect changes the published recommendation until its own tests pass.

## Stage 1 - deterministic two-level XY8 propagator (implemented)

**Deliverable:** `code/analysis/xy8_coherent_control.py`.

The module defines the exact `X-Y-X-Y-Y-X-Y-X` phase cycle, composes `8N` two-level rotation matrices, accepts scalar or pulse-resolved coherent angle, phase, and axis errors, and propagates state vectors or density matrices. It contains no decoherence, ensemble averaging, finite pulse duration, or sensitivity fitting.

**Acceptance gate:** ideal XY8 preserves arbitrary states up to global phase; all propagators are unitary; density-matrix trace and purity are preserved; pulse count is exactly `8N`; malformed inputs fail clearly; the existing scalar model and outputs remain unchanged.

## Stage 2 - finite pulse duration and detuning (implemented)

Piecewise free and driven Hamiltonians now use the existing pulse-centre convention. Pulse centres remain at `(2k-1)tau`, total sequence duration remains `16*N*tau`, static detuning acts during free and driven intervals, and `t_pi < 2tau` prevents overlapping pulses. Detuning inputs use angular-frequency units (rad/s); convert a measured detuning in hertz with `2*pi*detuning_Hz`.

The finite driven Hamiltonian includes the nominal Rabi rate `pi/t_pi`, deterministic amplitude error, intended XY8 phase, phase error, independently calibrated drive-axis tilt, and static detuning. Axis tilt and detuning are separate inputs for cases where both are independently measured; do not use both to encode the same off-resonance error.

**Acceptance gate passed:** the exact `t_pi=0`, zero-detuning limit reproduces Stage 1; resonant finite pulses reproduce instantaneous XY8; a single detuned driven pulse agrees with the analytic effective-axis rotation; ideal instantaneous XY8 refocuses static detuning; finite detuned pulses produce a deterministic coherent residual; schedules have non-negative intervals, exact pulse centres, `8N` pulses, and total duration `16*N*tau`.

## Stage 3 - quasi-static amplitude and detuning distributions (implemented)

Microwave-amplitude and detuning inhomogeneity can now be supplied as explicit weighted samples, deterministic Gaussian-Hermite quadrature, or seeded Monte Carlo draws. Each sample's fractional amplitude error and angular-frequency detuning remain fixed for the complete sequence. Optional hyperfine offsets and populations form a weighted mixture with the common detuning distribution.

The propagation averages physical density matrices and provides an explicit signed-observable average for the small-signal derivative in Stage 4. It does not average absolute contrast or independently randomize errors on every pulse. Weights are validated, normalized, and made read-only; the returned result records sample states, Bloch vectors, purities, timing, and the originating ensemble.

**Acceptance gate passed:** a one-sample delta distribution exactly reproduces Stage 2; product-quadrature moments and hyperfine populations match their declarations; weights sum to one; the ensemble density matrix equals the weighted sample average; symmetric amplitude-error broadening decreases purity in the regression regime; and fixed Monte Carlo seeds reproduce identical samples.

## Stage 4 - small-signal sensing derivative (implemented)

The model now propagates an initially prepared equatorial spin state through the complete XY8 control schedule at `+deltaB` and `-deltaB`, averages each signed ensemble response, and calculates the central magnetic derivative. Dividing its magnitude by `gamma*Tseq` produces a dimensionless coherent gain that can replace the scalar filter-and-pulse factor in Stage 5. The idealized preparation is the `+X` state by default, and readout is an equatorial Pauli projection; either can be declared by the caller. Preparation and readout pulse errors are not yet added as separate driven operations.

The longitudinal AC field is integrated analytically during free intervals. During finite pulses, where the sensing and drive Hamiltonians do not commute, midpoint substeps provide a controlled numerical integration. Signal and readout phases can both be supplied explicitly. If either is omitted, the local response is optimized over the two AC quadratures and/or the equatorial readout axis; joint optimization uses the singular value of the two-quadrature response matrix.

**Acceptance gate passed:** the ideal instantaneous result reproduces the existing bounded XY8 gain at resonant and off-resonant frequencies and gives `2/pi` at the primary resonance; the central derivative converges as `deltaB` decreases; explicit and optimized phase modes agree in the ideal case; zero signal recovers Stage 2; ensemble broadening reduces the small-signal gain in the regression case; and finite-pulse integration converges with increasing substeps.

## Stage 5 - opt-in integration with `eta_ac` (implemented)

`eta_ac` and `best_point_breakdown` now accept `control_model="scalar"|"coherent_xy8"`; scalar remains the default. Scalar mode retains the historical `filter_gain * piPulseFid^(8N)` control factor. Coherent mode replaces that complete product with the Stage 4 ensemble-averaged small-signal gain. It sets the separately applied pulse factor to one and does not multiply the ideal filter gain again. The empirical bath-coherence envelope, optical contrast, detected photon rate, sequence duration, and preparation/readout overhead remain common resources.

Coherent mode requires a `coherent_control` dictionary with an explicitly declared `pi_pulse_duration_s`. The caller may provide either a validated ensemble or Gaussian-quadrature distribution parameters, plus optional signal/readout phases, rotation-axis and phase errors, finite-difference field, gyromagnetic ratio, and pulse integration resolution. It deliberately does not infer a coherent amplitude-error distribution from the scalar `piPulseFid`; that conversion requires an independently stated calibration model.

Breakdowns report the selected backend, effective control gain, ideal gain for reference, whether independent pulse and ideal-filter factors were applied, selected phase behavior, pulse duration, ensemble moments and sample count, hyperfine populations, and numerical settings.

**Acceptance gate passed:** default and explicit scalar calculations reproduce the legacy formula; ideal coherent and lossless scalar calculations agree; changing `piPulseFid` cannot change a coherent result; coherent errors worsen the score in the regression case; optical, bath, photon-rate, and overhead resources remain active; malformed configurations fail clearly; and scalar/coherent breakdowns are JSON serializable and identify all applied factors.

## Stage 6 - frozen-holdout comparison (implemented)

The Farfurnik conventional-XY8 series is evaluated with two frozen models: the original scalar pulse-survival baseline and a sequence-specific two-level propagator using the source supplement's Gaussian rotation-angle and axis-error scales. The coherent calculation averages 10,000 seeded quasi-static NV realizations and uses a predeclared 27-case parameter-sensitivity envelope. It reproduces the paper's short-free-evolution approximation, in which free-evolution operators are unity, and normalizes the signed `Sx` response to an imperfect one-pulse Hahn reference. No parameter is fitted to the three contrast measurements.

**Acceptance gate completed without an improvement claim:** both models have rank correlation `1.0`. Scalar normalized RMSE is `0.288`, pointwise coverage is `2/3`, and its half-contrast crossing is 232 pulses. Coherent normalized RMSE is `0.614`, coverage is `1/3`, and it does not cross half contrast within the tested 1,024 pulses. The coherent backend therefore fails the predeclared improvement test and exposes missing high-depth loss physics rather than validating the sequence-specific prediction.

## Stage 7 - recompute the recommendation

Run the canonical 200 kHz sweep using the coherent backend. Export the optimum, total pulses, 10% window, boundary warnings, and comparison with scalar `N=20` and the lower-pulse `N=16` alternative.

**Acceptance gate:** grid-refinement stability, exact resonance inclusion, artifact consistency, and uncertainty tests all pass. If the optimum or window changes, update the paper and prospective protocol rather than forcing the previous result.

## Calibration measurements needed

- π-pulse duration or Rabi frequency.
- Mean and distribution of microwave-amplitude error.
- Mean and distribution of detuning, including hyperfine populations.
- X/Y phase imbalance or independently measured rotation-axis tilt.
- Whether errors are quasi-static across a sequence or fluctuate pulse to pulse.
