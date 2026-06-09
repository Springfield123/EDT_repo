# Narval second-stage EPW restart for Nb

Generated files:
- epw_restart_narval.in
- submit_epw_restart_narval.sh

Purpose:
- restart from completed first-stage EPW/Wannier/EPB data;
- enable epbread=.true. and epwread=.true.;
- enable phonselfen=.true., a2f=.true., and specfun_ph=.true.;
- use denser fine q-grid nqf = 4 4 4.

Prerequisites:
- The first EPW stage must have produced EPB/EPW restart payloads such as *.epb*, *.epmatwp*, or epwdata.fmt.
- DFPT save/dvscf/dyn/pattern data must be staged consistently with dvscf_dir.
- If this EPW build rejects epwread/epwwrite, set EPW_INPUT to a manually edited input with epwread/epwwrite removed while keeping epbread=.true.

Submit on Narval from this material directory:
  sbatch submit_epw_restart_narval.sh

Optional runtime overrides:
  EPW_NPOOL, EPW_NIMAGE, OMP_NUM_THREADS, MPI_LAUNCHER, EPW_INPUT

No EPC comparison should be claimed until the restart writes resolved EPC/lambda/linewidth outputs that can be parsed.
