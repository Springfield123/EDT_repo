# Frozen NV environment literature inputs

This directory is a provenance boundary for a later strain- and
temperature-conditioned EDT analysis. The files were selected and frozen
before calculating any environment-conditioned EDT result.

## Contents

- `udvarhelyi_2018_spin_strain_coefficients.csv`: six coefficients transcribed
  directly from Table I, including reported uncertainties and units.
- `nv_temperature_laws.json`: published analytical laws for zero-field
  splitting and longitudinal relaxation, plus a first-principles room-
  temperature slope used for an independent check.
- `mondal_2023_fig3_nv_t1_dft.csv`: twelve coordinates extracted from the
  vector source for Figure 3 because the complete numerical curve was not
  tabulated. Coordinate-extraction uncertainty is stored in every row.
- `acosta_2011_zfs_slope_validation.csv`: a validation-only experimental
  result. It is not permitted to tune the input laws.
- `nv_environment_literature_manifest.json`: DOI, figure or table location,
  source role, units, extraction method, archive digest, and freeze policy.
- `nv_environment_literature_freeze.json`: file-level and aggregate SHA-256
  digests for the immutable version-1 bundle.

Verify the frozen inputs and regenerate the separated validation report with:

```bash
python code/freeze_nv_environment_literature.py --verify
python code/validate_nv_environment_literature.py
```

The separate Stage-1 environment scan is preregistered in
`xy8_environment_scan_config.json`. Its temperature axes, independent strain
paths, fixed-versus-retracked carrier comparison, topology thresholds, and
grid-refinement rules are frozen in `xy8_environment_scan_freeze.json` before
any conditioned EDT landscape is calculated. Verify it with:

```bash
python code/freeze_xy8_environment_scan.py --verify
```

Stage 2 implements the environment-to-spin-Hamiltonian adapter in
`code/analysis/xy8_environment_conditioning.py`. It combines the frozen D(T)
law, magnetic bias, and six-coefficient strain Hamiltonian, then tracks the
selected transition by eigenvector overlap. Generate the transition-only
diagnostics with:

```bash
python code/analyze_xy8_environment_transitions.py
```

The resulting CSV and JSON deliberately contain no EDT score. They verify the
transition shifts, fixed-versus-retracked carrier detunings, source-range
warnings, and state-tracking ambiguity guard before the score model is used.

Stage 3 implements the anchored temperature-dependent coherence calculation in
`code/analysis/xy8_environment_coherence.py`. It preserves the nominal
300-microsecond T2 value at 300 K and changes only the independently sourced T1
contribution away from that reference. Generate its diagnostics with:

```bash
python code/analyze_xy8_environment_coherence.py
```

The Jarmola experimental law and Mondal first-principles curve remain separate
variants. They are never averaged or fitted to an EDT result.

Stage 4 combines the transition detuning and anchored coherence corrections
with the canonical score surface. Run:

```bash
python code/analyze_xy8_environment_scores.py
```

The command writes a compact CSV containing one summary row per environmental
surface and a compressed NPZ containing the full discovery-grid arrays. The
fixed-carrier correction uses the frozen finite-pulse response ratio, while the
ideal-retracked branch has zero environmental detuning by definition.

Stage 5 first labels connected regions on every discovery surface and writes a
confirmation selection before evaluating refined grids. The selection is
cryptographically frozen in
`data/processed/xy8_environment_confirmation_selection_freeze.json`. Run the
two steps in order:

```bash
python code/analyze_xy8_environment_topology.py
python code/analyze_xy8_environment_topology.py --confirm
```

Use `--finalize-only` to regenerate pass/fail diagnostics from existing medium
and fine artifacts without rerunning the expensive coherent propagation.

If author-supplied numerical data later replace a digitized series, retain this
directory version and create a new bundle identifier and freeze record. Do not
overwrite the existing record.
