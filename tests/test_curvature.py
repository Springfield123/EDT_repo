import numpy as np

from edt.analysis.curvature import CurvatureBranch, estimate_k0_from_gap_minimum, scan_edt_window


def test_edt_window_has_nonnegative_width():
    k = np.linspace(-0.1, 0.1, 101)
    a = CurvatureBranch(k, (k + 0.01) ** 2)
    b = CurvatureBranch(k, (k - 0.01) ** 2)
    k0 = estimate_k0_from_gap_minimum(a, b)
    result = scan_edt_window(a, b, k0=k0, half_widths=np.linspace(0.01, 0.04, 4), residual_threshold=1.0)
    assert result.width >= 0.0
