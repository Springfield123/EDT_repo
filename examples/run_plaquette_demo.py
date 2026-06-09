from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from edt.quantum.plaquette import FourQubitPlaquetteModel, PlaquetteParams


def main() -> None:
    model = FourQubitPlaquetteModel(PlaquetteParams(hx=0.7, jzz=0.5, lam=0.9))
    psi0 = model.basis_state("0000")

    times = np.linspace(0.0, 5.0, 100)
    edt_scores = []
    for t in times:
        psi_t = model.evolve_exact(psi0, float(t))
        edt_scores.append(model.edt_proxy(psi_t))

    peak_idx = 1 + int(np.argmax(edt_scores[1:]))
    print(f"Peak EDT proxy away from t=0: {edt_scores[peak_idx]:.6f}")
    print(f"Time of peak: {times[peak_idx]:.6f}")


if __name__ == "__main__":
    main()
