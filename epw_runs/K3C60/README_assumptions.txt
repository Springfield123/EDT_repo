# EPW assumptions for K3C60

Source directory: C:\Users\agahu\Dropbox\NRC\EDT\qe_input\KC60
Completed QE run directory: not found during setup
Prefix: pwscf
Local pseudopotential directory checked during setup: C:\Users\agahu\Dropbox\NRC\EDT\qe_input\pseudo
Runtime pseudopotential directory written to QE inputs: /home/hanghu/links/scratch/EDT/pseudo
Dynamic matrix file: dmat.AA
Original SCF/relax k-grid: 4 4 4
Generated SCF k-grid: 4 4 4
Generated NSCF k-grid: 4 4 4
Generated NSCF k-point style: explicit K_POINTS crystal list with equal weights, matching the previous BN EPW workflow pattern.
EPW fine k-grid: 8 8 8
EPW fine q-grid: 1 1 1
NSCF nbnd: 320
EPW nbndsub/num_wann: 183
Disentanglement window: [-30.0, 30.0] eV
Frozen window: [-2.0, 2.0] eV
Used final relaxed geometry from pw.x output: False

Projection assumptions:
- Fulleride first pass: C p orbitals for the C60 t1u/t1g-like manifold plus K s states. This gives 183 Wannier trial functions for 60 C atoms and 3 K atoms; add C:s only if the post-run spreads/interpolated bands show the disentanglement is too narrow.
- proj: C:p
- proj: K:s
- AGR curvature file planned for comparison: data/k3c60/tmp.agr
- EDT summary file planned for comparison: data/k3c60/k3c60_edt_window_summary.json

Validation notes:
- K3C60 DFPT is now treated as completed; stage this from the KC60 source input folder used by the QE runner.
- Stage 1 mirrors the successful Na-style EPW input: Wannierize, write EPB data, and enable phonselfen/specfun_ph in the first run.
- The K3C60 outer disentanglement window is intentionally wide because [-8, 8] eV contained only 104 states at k-point 1 for 183 target WFs.
- The AGR EDT pair is Band 208/Band 209 for mode m120 along Gamma-X-L-W-Gamma.
- ph.x output not found locally; assume-completed enabled for setup only
- dynmat.x output not found locally; assume-completed enabled for setup only
- missing pseudopotentials in local preview path C:\Users\agahu\Dropbox\NRC\EDT\qe_input\pseudo: K.pbe-spn-kjpaw_psl.1.0.0.UPF, C.pbe-n-kjpaw_psl.1.0.0.UPF
- final relaxed coordinates not parsed; generated inputs use source QE geometry
- dynamic-matrix file dmat.AA not found; EPW may fail until linked manually
- no dvscf/pattern artifacts found for save/; if EPW fails after namelist parsing, run EPW pp.py or copy DFPT save files into epw_runs/<material>/save

EPW spectral-function assumptions:
- Enabled phonselfen=.true. and specfun_ph=.true. with wmin/wmax/nw = 0.00/0.25/300 and T=300 K, following the previous BN workflow.
- lpolar and shortrange are not enabled by default because the completed ph.x inputs used epsil=.false.; enable them only after a DFPT run with the required dielectric/Born-charge data.
- filqf is not set by default because the completed DFPT workflow is Gamma/q=(0,0,0); add a material-specific q-point file only after generating matching DFPT q data.

Restart notes:
- First run writes epb/epw data with epbread=.false. and epwread=.false.
- After a successful first run, later restarts can use epbread=.true. and/or epwread=.true. if appropriate.
- Submit the second-stage EPC restart only after pwscf.wout, pwscf_hr.dat, EPB files, and epwdata.fmt are present.
