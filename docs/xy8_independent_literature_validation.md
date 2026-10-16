# Independent literature validation

## Frozen holdout

The first holdout is Figure 3b of L. M. Pham et al., *Enhanced solid-state multi-spin metrology using dynamical decoupling*, Physical Review B 86, 045214 (2012), [doi:10.1103/PhysRevB.86.045214](https://doi.org/10.1103/PhysRevB.86.045214), [arXiv:1201.5686](https://arxiv.org/abs/1201.5686). It was selected because the figure reports sensitivity at ten pulse counts, uncertainty bars, an 84.5 kHz target, and a visible high-depth decline. The authors also report the parameters used for their analytic curve: `C=0.1`, `T2=250 us`, `p=1`, and coherence scaling `s=0.37`.

The dataset and selection rationale are frozen in:

- `data/raw/nv/literature_validation/pham_2012_fig3b.csv`
- `data/raw/nv/literature_validation/pham_2012_manifest.json`
- `data/raw/nv/literature_validation/literature_screen.csv`

Pham et al. was not a direct source for the explicit values in the project's Table 1 scenario JSON files. The existing review references may discuss this experiment, so the result is described conservatively as a **holdout trend validation**, not as proof of complete bibliographic independence.

## Comparison protocol

The experiment uses an `n`-pulse XY sequence, not complete XY8 blocks. The comparison therefore uses equal total pulse count and resonant timing. At 84.5 kHz the project's timing convention gives `tau = 1/(4f) = 2.959 us`, corresponding to a physical interpulse separation of `5.917 us`. This validates the scalar resource trade-off but not XY8-specific phase cycling.

The prediction uses the paper-reported frequency and coherence parameters. Pulse fidelity and preparation/readout overhead were not reported, so their baseline values are imported unchanged from the pre-existing nominal scenario: `Fpi=0.997` and `7.3 us`. No parameter is fitted to the measured optimum. An uncertainty calculation spans the project's predeclared fidelity range `0.995–0.999` and overhead range `0–10.5 us`. Photon rate and any depth-independent contrast scale cancel when each curve is normalized to its minimum.

## Result

The digitized experimental central values reach their minimum at 106 pulses. The model predicts 86 pulses and a 10% window of 42–158 pulses, so the measured minimum lies inside the predicted operating window without fitting. Both experiment and model improve from low depth and decline at high depth. Across all 902 predeclared missing-parameter cases, the predicted optimum spans 60–152 pulses and the observed 106-pulse minimum remains inside the corresponding 10% window.

Agreement is useful but incomplete. The Spearman rank correlation is `0.517` and the normalized RMSE is `0.155`; nine of ten experimental uncertainty bars overlap the model's 95% missing-parameter band. The scalar model understates the low-depth penalty and overstates part of the high-depth penalty. The appropriate figure title is therefore:

> Independent data identify where the model succeeds and where refinement is needed

This is a stronger conclusion than either unqualified validation or simple failure: the finite-depth decision transfers to a holdout experiment, while the remaining curve asymmetry identifies the next model improvement—finite pulse duration and coherent, axis-dependent pulse errors.

## Benchmark against simpler alternatives

The holdout was also evaluated with progressively richer alternatives. A coherence-only model predicts 264 pulses and its 10% window does not include the measured 106-pulse minimum. Adding the predeclared independent pulse fidelity moves the prediction to 84 pulses; the full resource score, including preparation and readout overhead, predicts 86 pulses. Both resource-aware windows contain the measured minimum. The full score therefore gives the closest predicted optimum, reducing the absolute error from 158 pulses for coherence alone to 20 pulses.

The comparison does not support a blanket claim that the full score is superior on every metric. Coherence alone has the lowest normalized RMSE (`0.142`) because it follows the steep low-depth improvement, but its rank correlation is only `0.024` and it misses the finite turn. The full score has a higher RMSE (`0.155`) but a substantially better rank correlation (`0.517`) and recovers the finite operating region. A fixed conventional XY8 choice of `N=16`, or 128 pulses, is also only about `0.55%` above the interpolated experimental minimum in this dataset. The defensible conclusion is therefore that pulse-error accounting is necessary to locate the finite region, while additional datasets and improved error physics are needed to demonstrate a large advantage over simpler fixed choices.

## Reproduction

```bash
python code/plotting/make_xy8_literature_validation.py
python code/plotting/make_xy8_model_benchmark.py
python code/plotting/make_xy8_specific_literature_validation.py
```

The script exports the digitized/model overlay, pointwise uncertainty band, metrics JSON, dense curve CSV, and all uncertainty cases. It is also included in `python code/run_reproduction_pipeline.py --nv-only`.

## Second holdout: a genuine XY8 depth series

The XY8-specific holdout is Figure 3b of D. Farfurnik et al., *Optimizing a dynamical decoupling protocol for solid-state electronic spin ensembles in diamond*, Physical Review B 92, 060301(R) (2015), [doi:10.1103/PhysRevB.92.060301](https://doi.org/10.1103/PhysRevB.92.060301), [arXiv:1505.00636](https://arxiv.org/abs/1505.00636). The experimental series uses conventional XY8 at 8, 72, and 584 π pulses, exactly `N=1`, `9`, and `73` complete XY8 blocks. The measured relative contrasts are approximately `1.00`, `0.55`, and `0.36`, with plotted uncertainty bars.

This is a deliberately focused rather than complete sensitivity validation. The measurements use the short-free-evolution condition `2 n tau << T2`, so bath decoherence is negligible. The observable isolates accumulated pulse errors. Because the experiment is normalized to a one-pulse Hahn echo, the scalar comparison uses `Fpi^(8N-1)`. The sequence-specific comparison instead reproduces the source paper's instantaneous-pulse model: each simulated NV receives zero-mean Gaussian angle, out-of-plane axis, and in-plane axis errors with reported standard deviations `0.06`, `0.06`, and `0.05`. Each draw remains fixed through a complete sequence, and the signed `Sx` response is averaged over 10,000 NV realizations. The random seed is fixed from the paper identifier, not selected from the measured curve.

The paper reports these error scales approximately but gives no parameter confidence intervals. Before the final evaluation, a sensitivity envelope was declared by multiplying each scale independently by `0.75`, `1.00`, or `1.25`, producing 27 cases. The first numerical draft used low-order Gaussian quadrature, but a convergence check rejected it at `N=73`; the final frozen calculation uses the same 10,000-realization ensemble size as the source paper and common seeded draws for all uncertainty cases. No physical parameter changed during that numerical correction.

The result is partial agreement, not successful transfer. Both frozen models reproduce the measured decreasing order, giving Spearman rank correlation `rho=1.0`. The scalar baseline has normalized RMSE `0.288`, overlaps two of the three measurements with its predeclared range, and first falls below half contrast at 232 pulses, inside the coarse measured 72-to-584-pulse bracket. The coherent model predicts relative contrasts `1.012`, `0.998`, and `0.872` at 8, 72, and 584 pulses. Its normalized RMSE is `0.614`, its uncertainty range overlaps only one point, and it does not fall below half contrast within 1,024 pulses.

The added coherent physics therefore does not improve this holdout. It predicts too much retained contrast at intermediate and high depth, which is consistent with the original paper's statement that its own pulse-error simulation did not reproduce the experimental loss quantitatively. The discrepancy identifies what remains unresolved: additional short-time decoherence, correlations or drift not represented by one quasi-static draw, and preparation/readout or ensemble effects. Those mechanisms require independent calibration and must not be fitted to these three points.

The defensible presentation message is:

> Independent XY8 data reproduce the expected loss direction, but reported coherent pulse errors alone do not explain the observed high-depth loss.

The frozen inputs are `farfurnik_2015_fig3b_xy8.csv` and `farfurnik_2015_xy8_manifest.json` under `data/raw/nv/literature_validation`. The generated metrics and curve are `xy8_specific_literature_validation_metrics.json` and `xy8_specific_literature_validation_curve.csv`.
