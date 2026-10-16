# Draft request for experimental XY8 timing sweeps

Status: **prepared, not sent**  
Prepared: 15 September 2026  
Suggested recipient: Dr Boris Naydenov, corresponding author  
Email: boris.naydenov@helmholtz-berlin.de  
Paper: S. Tsunaki *et al.*, “Ambiguous resonances in multipulse quantum sensing with nitrogen-vacancy centers,” *Physical Review A* **111**, 022606 (2025), DOI: 10.1103/PhysRevA.111.022606.

## Suggested subject

Request for experimental XY8 timing-sweep data for no-fit topology validation

## Ready-to-send message

Dear Dr Naydenov,

I am studying how connected near-optimal regions in an NV-center control landscape change with XY8 timing and control depth. Your 2025 paper on ambiguous resonances is particularly relevant because it examines resonance structure across XY8 order and pulse timing.

Would you be willing to share the experimental timing-sweep data underlying the paper, ideally for two or more XY8 depths? The public Figshare/GitHub release appears to contain the simulation dataset, while I am specifically seeking the experimental traces needed for an independent two-dimensional test.

For each sweep, the most useful fields would be:

- pulse timing or total free-precession interval, with its convention and units;
- XY8 block count/order and total number of π pulses;
- measured fluorescence, contrast, coherence, or other plotted observable;
- uncertainty bars, repeat-level values, or the number of repetitions;
- normalization/reference traces and any exclusion or preprocessing rules;
- π-pulse duration, Rabi frequency, detuning, bias field, and NV transition used;
- sample/ensemble identifier and the calibration values associated with that run.

Processed numerical values would already be valuable; raw repeats and analysis scripts would be especially helpful if available. I would freeze the supplied dataset and all reported calibration inputs before evaluating it. The comparison would not fit the model to the observed optimum. It would test whether the measured and predicted landscapes agree on the number, location, boundary, and persistence of connected resonant regions as the allowed performance penalty is increased.

The data would be used for a scientific symposium paper and presentation. I would cite the article and repository, clearly distinguish reported from assumed parameters, and follow any requested data-use or attribution terms. I would also be happy to share the resulting comparison before public use.

Thank you for considering the request.

Best regards,

[Name]  
[Affiliation]  
[Email]

## Minimum usable dataset

The genuine two-dimensional test requires at least two experimental XY8 timing sweeps acquired at different block counts. A stronger dataset would contain four or more depths spanning the shallow, middle, and high-depth regimes. Every trace needs a common timing convention and enough metadata to distinguish physical resonances from harmonics or spurious responses.

## Predeclared analysis

1. Preserve the received files unchanged and record SHA-256 digests.
2. Freeze inclusion rules and all paper-reported calibration values before model evaluation.
3. Normalize each measured and predicted landscape only by its own best value; do not shift timing, tune coherence, or fit pulse-error terms to improve agreement.
4. At penalty thresholds from 1% to 100%, calculate the number of connected regions, the component containing the global optimum, its timing/depth boundaries, and its persistence.
5. Propagate reported measurement uncertainty by seeded resampling and report topology probabilities rather than only the central curve.
6. Compare measured and predicted region count, resonance identity, boundaries, and mask overlap. Report partial or failed transfer without retuning.

## Scope guardrail

The frozen Pham curve remains the one-dimensional, equal-total-pulse depth test. The Farfurnik conventional-XY8 three-depth series remains a separate check of pulse-error mechanisms. Neither dataset will be pooled with the requested Tsunaki timing sweeps or represented as a substitute for a two-dimensional experimental landscape.
