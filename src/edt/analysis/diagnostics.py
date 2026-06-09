from __future__ import annotations

import numpy as np


def normalized_peak_score(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    denom = np.max(np.abs(values))
    if denom == 0:
        return 0.0
    return float(np.max(values) / denom)


def finite_regime_score(signal: np.ndarray, grid: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if signal.shape != grid.shape:
        raise ValueError("signal and grid must have same shape")
    positive = np.clip(signal, a_min=0.0, a_max=None)
    return float(np.trapz(positive, grid))
