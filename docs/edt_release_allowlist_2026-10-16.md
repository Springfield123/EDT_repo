# EDT NV Release Allowlist

This file records the paths intentionally selected for the 2026-10-16 NV-centre
XY8/EDT repository update. It exists to prevent unrelated research files,
temporary products, and superseded presentation revisions from entering the
commit accidentally.

## Included

- Canonical NV analysis, command-line, plotting, calibration, and utility code
  under `code/` whose names contain `xy8`, `nv_environment`, or
  `make_fig_xy8_qre`, together with the reproduction driver and figure-source
  exporter.
- The compatibility entry point `scripts/make_fig_xy8_qre.py`.
- Focused `tests/test_xy8_*.py` tests.
- Frozen NV configuration, calibration templates, environmental literature,
  and holdout datasets under `data/raw/nv/`.
- Machine-readable CSV/JSON results under `data/processed/`, excluding large
  scan-surface NPZ intermediates.
- Canonical and supporting NV figures and mirrored machine-readable tables
  under `outputs/reproduced_figures/`.
- Current scenario outputs under `figs_conservative/`, `figs_nominal/`, and
  `figs_optimistic/`.
- The current NATO presentation, its PDF, speaker notes, and referenced image
  assets.
- The current symposium short paper, its figures, and authoring notes.
- NV/EDT documentation, reproduction instructions, and the repository handoff.
- The updated Figure 5 source-data export.

## Deliberately excluded

- `.DS_Store`, `.tmp/`, `tmp/`, Office lock files, caches, and inspection logs.
- `data/processed/xy8_environment_discovery_surfaces.npz` and
  `data/processed/xy8_environment_confirmation_surfaces.npz`; these are large
  generated scan intermediates and need an explicit Git LFS or release-archive
  decision.
- Superseded PowerPoint revisions. The release retains only
  `EDT_robust_quantum_sensing_20min_NATO_template_v2.pptx`, its PDF, the speaker
  notes, and their assets.
- The obsolete symposium-paper PDF marked `OUTDATED` and user-supplied template,
  instruction, sample, and author-information documents.
- Unrelated QE, EPW, SHGO, Schur-CG, workstation-handoff, cover-letter, and
  source-paper files.

Anything outside the included paths remains unstaged unless separately reviewed.
