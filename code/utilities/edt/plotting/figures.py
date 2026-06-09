from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_curvature_branches(k: np.ndarray, branch_a: np.ndarray, branch_b: np.ndarray):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k, branch_a, label="branch a")
    ax.plot(k, branch_b, label="branch b")
    ax.set_xlabel("k")
    ax.set_ylabel("curvature")
    ax.legend()
    fig.tight_layout()
    return fig
