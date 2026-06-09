from __future__ import annotations

import numpy as np


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


def compute_entanglement_entropy(
    states: np.ndarray | list[np.ndarray],
    subsystem: list[int] | tuple[int, ...],
) -> np.ndarray:
    """Compute the von Neumann entropy of a qubit subsystem for pure states."""
    state_stack = _coerce_states(states)
    subsystem_tuple = tuple(sorted(subsystem))
    n_qubits = _infer_n_qubits(state_stack.shape[1])

    if not subsystem_tuple:
        raise ValueError("subsystem must contain at least one qubit index.")
    if subsystem_tuple[-1] >= n_qubits or subsystem_tuple[0] < 0:
        raise ValueError("subsystem indices must lie within the qubit register.")

    rest = tuple(site for site in range(n_qubits) if site not in subsystem_tuple)
    order = subsystem_tuple + rest
    dim_a = 2 ** len(subsystem_tuple)
    entropies = []
    for state in state_stack:
        tensor = state.reshape((2,) * n_qubits)
        psi_matrix = np.transpose(tensor, order).reshape(dim_a, -1)
        rho_a = psi_matrix @ psi_matrix.conj().T
        rho_a = 0.5 * (rho_a + rho_a.conj().T)
        eigenvalues = np.linalg.eigvalsh(rho_a)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues)) if eigenvalues.size else 0.0
        entropies.append(float(np.real_if_close(entropy)))
    return np.asarray(entropies, dtype=float)
