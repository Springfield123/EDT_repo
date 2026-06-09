# EPW assumptions for V

Source directory: C:\Users\agahu\Dropbox\NRC\EDT\qe_input\V
Completed QE run directory: not found during setup
Prefix: pwscf
Dynamic matrix file: dmat.V
Original SCF/relax k-grid: 8 8 8
Generated SCF k-grid: 8 8 8
Generated NSCF k-grid: 8 8 8
Generated NSCF k-point style: explicit K_POINTS crystal list with equal weights, matching the previous BN EPW workflow pattern.
EPW fine k-grid: 32 32 32
EPW fine q-grid: 1 1 1
NSCF nbnd: 60
EPW nbndsub/num_wann: 18
Disentanglement window: [-12.0, 16.0] eV
Frozen window: [-6.0, 6.0] eV
Used final relaxed geometry from pw.x output: False

Projection assumptions:
- Transition metal: V s+p+d projections on two V atoms, giving 18 Wannier trial functions.
- proj: V:s
- proj: V:p
- proj: V:d
- AGR curvature file planned for comparison: data/v/V-EP-F1.agr
- EDT summary file planned for comparison: data/processed/S2_V_summary.json

Validation notes:
- Metallic d manifold; first-batch windows intentionally include nearby s/p bands.
- V completed in the previous dense batch; these reduced settings are for reproducible retries only.
- ph.x output not found locally; assume-completed enabled for setup only
- dynmat.x output not found locally; assume-completed enabled for setup only
- final relaxed coordinates not parsed; generated inputs use source QE geometry
- dynamic-matrix file dmat.V not found; EPW may fail until linked manually
- no dvscf/pattern artifacts found for save/; if EPW fails after namelist parsing, run EPW pp.py or copy DFPT save files into epw_runs/<material>/save

EPW spectral-function assumptions:
- Enabled phonselfen=.true. and specfun_ph=.true. with wmin/wmax/nw = 0.00/0.25/300 and T=300 K, following the previous BN workflow.
- lpolar and shortrange are not enabled by default because the completed ph.x inputs used epsil=.false.; enable them only after a DFPT run with the required dielectric/Born-charge data.
- filqf is not set by default because the completed DFPT workflow is Gamma/q=(0,0,0); add a material-specific q-point file only after generating matching DFPT q data.

Restart notes:
- First run writes epb/epw data with epbread=.false. and epwread=.false.
- After a successful first run, later restarts can use epbread=.true. and/or epwread=.true. if appropriate.
- K3C60/KC60 is intentionally skipped until its DFPT run is complete.
