# EPW assumptions for YBaCuO

Source directory: C:\Users\agahu\Dropbox\NRC\EDT\qe_input\YBaCuO
Completed QE run directory: not found during setup
Prefix: pwscf
Dynamic matrix file: dmat.YBaCuO
Original SCF/relax k-grid: 8 8 4
Generated SCF k-grid: 8 8 4
Generated NSCF k-grid: 6 6 3
Generated NSCF k-point style: explicit K_POINTS crystal list with equal weights, matching the previous BN EPW workflow pattern.
EPW fine k-grid: 18 18 9
EPW fine q-grid: 1 1 1
NSCF nbnd: 110
EPW nbndsub/num_wann: 44
Disentanglement window: [-8.0, 10.0] eV
Frozen window: [-5.0, 3.0] eV
Used final relaxed geometry from pw.x output: False

Projection assumptions:
- Cuprate oxide: Cu d + O p manifold with Y d/s and Ba s, giving exactly 44 Wannier trial functions.
- proj: Cu:d
- proj: O:p
- proj: Y:d
- proj: Y:s
- proj: Ba:s
- AGR curvature file planned for comparison: data/ybacu3o7/YBa2Cu3O7_m33.agr
- EDT summary file planned for comparison: data/processed/S4_YBa2Cu3O7_summary.json

Validation notes:
- Oxide/cuprate first batch; expect projection/window tuning may be needed after inspecting Wannier spreads.
- Stability retry: reduced from 12x12x6/220 bands after YBaCuO NSCF band non-convergence; second retry uses 6x6x3/110 bands.
- ph.x output not found locally; assume-completed enabled for setup only
- dynmat.x output not found locally; assume-completed enabled for setup only
- missing pseudopotentials in local preview path C:\Users\agahu\Dropbox\NRC\EDT\qe_input\pseudo: Ba.pbe-spn-kjpaw_psl.1.0.0.UPF
- final relaxed coordinates not parsed; generated inputs use source QE geometry
- dynamic-matrix file dmat.YBaCuO not found; EPW may fail until linked manually
- no dvscf/pattern artifacts found for save/; if EPW fails after namelist parsing, run EPW pp.py or copy DFPT save files into epw_runs/<material>/save

EPW spectral-function assumptions:
- Enabled phonselfen=.true. and specfun_ph=.true. with wmin/wmax/nw = 0.00/0.25/300 and T=300 K, following the previous BN workflow.
- lpolar and shortrange are not enabled by default because the completed ph.x inputs used epsil=.false.; enable them only after a DFPT run with the required dielectric/Born-charge data.
- filqf is not set by default because the completed DFPT workflow is Gamma/q=(0,0,0); add a material-specific q-point file only after generating matching DFPT q data.

Restart notes:
- First run writes epb/epw data with epbread=.false. and epwread=.false.
- After a successful first run, later restarts can use epbread=.true. and/or epwread=.true. if appropriate.
- K3C60/KC60 is intentionally skipped until its DFPT run is complete.
