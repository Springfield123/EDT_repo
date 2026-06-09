# Na EDT Geometry Analysis

## Wannier QC
- Required files present: True
- num_wann: 8
- projections: Na:s, Na:p
- disentanglement window: [-12.0, 18.0] eV
- frozen window: [-5.0, 6.0] eV
- final spreads Ang^2: 3.80631, 3.80631, 5.05635, 5.05635, 5.05622, 5.05634, 5.05635, 5.05623
- Omega Total: 37.9504363 Ang^2
- normal Wannier90 termination: True
- explicit convergence message: False

Warning: Wannier90 terminated normally, but no explicit convergence message was parsed and the run reached cycle 1000. Treat the spreads as usable QC data, not as proof of strict minimization convergence.

## Geometry Summary
- EDT window: k = [1.280224, 2.314964], k0 = 1.836899216185377
- inside quantum metric mean, all bands: 42.233883
- outside quantum metric mean, all bands: 116.59264
- inside/outside ratio, all bands: 0.36223455

Band-resolved ratios:
- band 1: inside=55.806172, outside=78.633318, ratio=0.70970136
- band 2: inside=54.336612, outside=175.96557, ratio=0.30879115
- band 3: inside=20.949089, outside=123.09499, ratio=0.17018636
- band 4: inside=39.127651, outside=119.90948, ratio=0.3263099
- band 5: inside=45.411433, outside=123.0455, ratio=0.36906212
- band 6: inside=45.411361, outside=89.895034, ratio=0.50515984
- band 7: inside=20.94925, outside=82.050842, ratio=0.25532035
- band 8: inside=55.879498, outside=140.1464, ratio=0.39872232

Control-result note: For Na, the EDT curvature window is not where the path-projected quantum metric is largest. This should be treated as a control result, not as a failure.

## EPC Status
- status: blocked_missing_epw_epc_outputs
- outputs/ has files: False
- save/ has files: False
- next action: Run or stage EPW EPC outputs such as epw.out plus epb/epmat/lambda/linewidth/specfun files, then compute EPC inside/outside ratios on the same EDT window.
