# EDT as a Quantitative Diagnostic

This note reframes EDT from a broad conceptual framework into a falsifiable quantitative diagnostic. Every major claim below is tied to a figure class, equation class, or reproducible script already present in the repository.

## Working Definition

In this repository, `EDT` should mean:

> A finite interval or region in parameter space that can be extracted from observables by an explicit diagnostic and that degrades outside that window.

This is narrower than a universal theory. It is an operational claim that can be tested and falsified.

## Evidence-Bearing Diagnostics

### 1) Curvature diagnostic

System: K$_3$C$_{60}$

Observable:
- paired curvature branches derived from electronic structure

Diagnostic:
- a nonzero interval `Delta k_EDT` over which both branches admit local quadratic fits below a stated residual tolerance

Code:
- [scripts/superfinger_k3c60_edt_localfit_v6.py](</C:/Users/agahu/Dropbox/NRC/EDT/code/scripts/superfinger_k3c60_edt_localfit_v6.py>)
- [src/edt/analysis/curvature.py](</C:/Users/agahu/Dropbox/NRC/EDT/code/src/edt/analysis/curvature.py>)

Allowed claim:
- "EDT is identified here as a finite momentum window extracted by paired quadratic residual thresholding."

Avoid:
- "EDT is proven to be the microscopic origin of quantum advantage."

### 2) Dynamical diagnostic

System: four-qubit plaquette model

Observables:
- mid-time OTOC
- long-time correlation / loop observable
- Trotter error under approximate implementation

Diagnostic:
- an interior maximum of the combined score used in the script,
  `S_EDT = normalized(mid-time OTOC) * normalized(C_inf)^1.2`,
  together with a finite support region in control parameter `lambda`

Code:
- [scripts/demo_edt_gyro_trotter_4q_fig34_clean.py](</C:/Users/agahu/Dropbox/NRC/EDT/code/scripts/demo_edt_gyro_trotter_4q_fig34_clean.py>)
- [src/edt/analysis/diagnostics.py](</C:/Users/agahu/Dropbox/NRC/EDT/code/src/edt/analysis/diagnostics.py>)

Allowed claim:
- "The plaquette model exhibits a finite EDT-like operating window under the chosen score and control scan."

Avoid:
- "EDT generically emerges in all loop-induced many-body dynamics."

### 3) Sensing diagnostic

System: NV-center XY8 AC magnetometry

Observables:
- resource-resolved sensitivity proxy over `tau` and `N`
- filter gain
- coherence factor
- accumulated pulse fidelity
- readout efficiency / effective detected rate

Diagnostic:
- an interior optimum of the sensing proxy that outperforms both heuristic and overdriven control points

Code:
- [scripts/make_fig_xy8_qre.py](</C:/Users/agahu/Dropbox/NRC/EDT/code/scripts/make_fig_xy8_qre.py>)
- [scripts/nvratemodel_api.py](</C:/Users/agahu/Dropbox/NRC/EDT/code/scripts/nvratemodel_api.py>)

Allowed claim:
- "For the modeled XY8 scenarios studied here, EDT corresponds to a finite control window in `(tau, N)` where sensing resources remain jointly effective."

Avoid:
- "EDT is experimentally established as a general sensing principle."

## Recommended Abstract Language

### Replacement abstract core

Quantum technologies rely on entanglement, coherence, superposition, and interference as operational resources, but in realistic systems these resources compete with decoherence, control error, and measurement overhead. In this work we use Entanglement--Dynamic Topology (EDT) in a restricted and falsifiable sense: EDT is said to be present only when a finite parameter window can be extracted from observables by an explicit diagnostic. We implement this idea in three settings. In K$_3$C$_{60}$, EDT is identified by a nonzero momentum interval `Delta k_EDT` over which paired curvature branches admit a local quadratic description below a residual threshold. In a four-qubit plaquette model, EDT is identified by an interior maximum of a combined dynamical score built from mid-time OTOC weight and long-time correlation, together with a finite control window and bounded Trotter error under approximate implementation. In NV-center AC magnetometry, EDT is identified by an interior optimum of a resource-resolved XY8 sensing proxy over pulse spacing and block count, with the best point outperforming both heuristic and overdriven controls through a measurable balance of filter gain, coherence retention, pulse fidelity, and readout efficiency. Taken together, these results do not establish a universal quantum principle; rather, they provide a reproducible set of diagnostics for detecting finite, quantitatively identifiable operating windows in three distinct model systems.

## Recommended Introduction Positioning

Use wording like:

> In this paper, EDT is not treated as a general explanatory framework by default. Instead, it is used as an operational label for finite windows extracted by explicit diagnostics from observables. Figure 1 is therefore heuristic, while the evidence-bearing claims are confined to the residual-based curvature window in Fig. 2, the score-defined dynamical window in Figs. 3--4, and the resource-resolved sensing optimum in Fig. 5.

## Claims to Remove or Weaken

Replace strong claims such as:
- "EDT establishes a new paradigm"
- "EDT is a unifying principle"
- "EDT is the microscopic origin of usable quantum resources"
- "EDT emerges generically"

with narrower claims such as:
- "EDT provides a testable diagnostic workflow"
- "EDT identifies a finite operating window under the diagnostics used here"
- "EDT is transportable across the three model systems examined here"

## Safe Conclusion Template

The present results support EDT as a reproducible diagnostic for finite operating windows in the three systems studied here. They do not, by themselves, establish universality. Any broader claim should require additional systems, explicit failure cases, and consistent threshold-based diagnostics.
