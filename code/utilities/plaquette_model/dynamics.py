from __future__ import annotations

import numpy as np

from edt.models.hamiltonians import Z, product_on_sites


def _coerce_states(states: np.ndarray | list[np.ndarray]) -> np.ndarray:
    arr = np.asarray(states, dtype=complex)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2:
        raise ValueError("states must be a state vector or a stack of state vectors.")
    return arr


def _infer_n_qubits(dim: int) -> int:
    n_qubits = int(np.log2(dim))
    if 2**n_qubits != dim:
        raise ValueError("State dimension must be a power of two.")
    return n_qubits


def _normalize_metric(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    vmax = np.max(values) if values.size else 0.0
    if vmax <= 0.0:
        return np.zeros_like(values)
    return values / vmax


def exact_time_evolution(H: np.ndarray, psi0: np.ndarray, times: np.ndarray | list[float]) -> np.ndarray:
    """Return the exact state trajectory under H for the supplied times."""
    hamiltonian = np.asarray(H, dtype=complex)
    state0 = np.asarray(psi0, dtype=complex)
    time_grid = np.asarray(times, dtype=float)

    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("H must be a square matrix.")
    if state0.ndim != 1 or state0.shape[0] != hamiltonian.shape[0]:
        raise ValueError("psi0 must be a vector matching the Hamiltonian dimension.")

    evals, evecs = np.linalg.eigh(hamiltonian)
    coeffs = evecs.conj().T @ state0
    trajectory = []
    for time in time_grid:
        phases = np.exp(-1j * evals * time)
        trajectory.append(evecs @ (phases * coeffs))
    return np.asarray(trajectory)


def compute_two_point_correlations(states: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """Compute nearest-neighbor ZZ correlations for each state."""
    state_stack = _coerce_states(states)
    n_qubits = _infer_n_qubits(state_stack.shape[1])
    pair_ops = [
        product_on_sites({site: Z, (site + 1) % n_qubits: Z}, n_qubits)
        for site in range(n_qubits)
    ]
    values = []
    for op in pair_ops:
        values.append(np.real(np.einsum("ti,ij,tj->t", np.conjugate(state_stack), op, state_stack)))
    return np.stack(values, axis=-1)


def compute_edt_dynamical_score(
    otoc: np.ndarray | list[float],
    correlations: np.ndarray | list[float],
    entropy: np.ndarray | list[float],
) -> np.ndarray:
    """Combine spreading, correlation retention, and moderate entanglement.

    The entropy contribution is peaked at intermediate normalized entropy so
    the score reflects a finite regime rather than maximal scrambling alone.
    """
    otoc_values = np.asarray(otoc, dtype=float)
    corr_values = np.asarray(correlations, dtype=float)
    entropy_values = np.asarray(entropy, dtype=float)

    if corr_values.ndim > 1:
        corr_metric = np.mean(np.abs(corr_values), axis=-1)
    else:
        corr_metric = np.abs(corr_values)

    otoc_metric = _normalize_metric(np.abs(otoc_values))
    corr_metric = _normalize_metric(corr_metric)
    entropy_norm = _normalize_metric(entropy_values)
    entropy_metric = 4.0 * entropy_norm * (1.0 - entropy_norm)
    return otoc_metric * corr_metric * entropy_metric
