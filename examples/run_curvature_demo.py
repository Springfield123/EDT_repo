from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from edt.analysis.curvature import (
    CurvatureBranch,
    estimate_k0_from_gap_minimum,
    scan_edt_window,
)


def main() -> None:
    k = np.linspace(-0.15, 0.15, 401)
    branch_a = 4.0 * (k + 0.01) ** 2 + 0.04 * np.sin(40 * k)
    branch_b = 4.6 * (k - 0.01) ** 2 + 0.04 * np.sin(40 * k + 0.3)

    a = CurvatureBranch(k=k, curvature=branch_a, label="upper")
    b = CurvatureBranch(k=k, curvature=branch_b, label="lower")

    k0 = estimate_k0_from_gap_minimum(a, b)
    result = scan_edt_window(
        a,
        b,
        k0=k0,
        half_widths=np.linspace(0.005, 0.08, 40),
        residual_threshold=0.03,
    )

    print(f"Estimated k0: {result.k0:.6f}")
    print(f"EDT window: [{result.k_min:.6f}, {result.k_max:.6f}]")
    print(f"Window width: {result.width:.6f}")


if __name__ == "__main__":
    main()
