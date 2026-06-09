from __future__ import annotations

import numpy as np


def compute_otoc(
    H: np.ndarray,
    W: np.ndarray,
    V: np.ndarray,
    psi0: np.ndarray,
    times: np.ndarray | list[float],
) -> np.ndarray:
    """Compute the pure-state OTOC 1 - Re <W(t) V W(t) V>."""
    hamiltonian = np.asarray(H, dtype=complex)
    op_w = np.asarray(W, dtype=complex)
    op_v = np.asarray(V, dtype=complex)
    state0 = np.asarray(psi0, dtype=complex)
    time_grid = np.asarray(times, dtype=float)

    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("H must be a square matrix.")
    if op_w.shape != hamiltonian.shape or op_v.shape != hamiltonian.shape:
        raise ValueError("W and V must match the Hamiltonian dimension.")
    if state0.ndim != 1 or state0.shape[0] != hamiltonian.shape[0]:
        raise ValueError("psi0 must be a vector matching the Hamiltonian dimension.")

    evals, evecs = np.linalg.eigh(hamiltonian)
    results = []
    for time in time_grid:
        unitary = evecs @ np.diag(np.exp(-1j * evals * time)) @ evecs.conj().T
        wt = unitary.conj().T @ op_w @ unitary
        echo = np.vdot(state0, wt @ op_v @ wt @ op_v @ state0)
        results.append(1.0 - np.real(echo))
    return np.asarray(results, dtype=float)
