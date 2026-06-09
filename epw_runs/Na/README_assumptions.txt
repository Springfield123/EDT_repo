# EPW assumptions for Na

Source directory: C:\Users\agahu\Dropbox\NRC\EDT\qe_input\Na
Completed QE run directory: not found during setup
Prefix: pwscf
Dynamic matrix file: dmat.Na
Original SCF/relax k-grid: 10 10 10
Generated SCF k-grid: 10 10 10
Generated NSCF k-grid: 8 8 8
Generated NSCF k-point style: explicit K_POINTS crystal list with equal weights, matching the previous BN EPW workflow pattern.
EPW fine k-grid: 32 32 32
EPW fine q-grid: 1 1 1
NSCF nbnd: 48
EPW nbndsub/num_wann: 8
Disentanglement window: [-12.0, 18.0] eV
Frozen window: [-5.0, 6.0] eV
Used final relaxed geometry from pw.x output: False

Projection assumptions:
- Alkali metal: Na s+p projections on two Na atoms, giving 8 Wannier trial functions.
- proj: Na:s
- proj: Na:p
- AGR curvature file planned for comparison: data/na/Na-m5.agr
- EDT summary file planned for comparison: data/processed/S1_Na_summary.json

Validation notes:
- Gamma-only DFPT from completed workflow; suitable for the current mode-resolved EDT comparison, not full-BZ lambda.
- Stability retry: NSCF grid/nbnd reduced after dense first batch stressed pw.x.
- ph.x output not found locally; assume-completed enabled for setup only
- dynmat.x output not found locally; assume-completed enabled for setup only
- final relaxed coordinates not parsed; generated inputs use source QE geometry
- dynamic-matrix file dmat.Na not found; EPW may fail until linked manually
- no dvscf/pattern artifacts found for save/; if EPW fails after namelist parsing, run EPW pp.py or copy DFPT save files into epw_runs/<material>/save

EPW spectral-function assumptions:
- Enabled phonselfen=.true. and specfun_ph=.true. with wmin/wmax/nw = 0.00/0.25/300 and T=300 K, following the previous BN workflow.
- lpolar and shortrange are not enabled by default because the completed ph.x inputs used epsil=.false.; enable them only after a DFPT run with the required dielectric/Born-charge data.
- filqf is not set by default because the completed DFPT workflow is Gamma/q=(0,0,0); add a material-specific q-point file only after generating matching DFPT q data.

Restart notes:
- First run writes epb/epw data with epbread=.false. and epwread=.false.
- After a successful first run, later restarts can use epbread=.true. and/or epwread=.true. if appropriate.
- K3C60/KC60 is intentionally skipped until its DFPT run is complete.
