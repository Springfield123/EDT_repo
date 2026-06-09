from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from edt.nv.xy8 import NVParams, XY8Model


def main() -> None:
    model = XY8Model(NVParams())
    taus = np.linspace(0.5e-6, 20e-6, 120)
    n_pulses = np.arange(8, 257, 8)
    result = model.sweep(taus, n_pulses)

    print(f"Best tau (s): {result.best_tau:.6e}")
    print(f"Best N_pi: {result.best_n_pulses}")
    print(f"Best sensitivity proxy: {result.best_value:.6e}")


if __name__ == "__main__":
    main()
