from __future__ import annotations

import numpy as np

from edt.models.hamiltonians import X, Z, pauli_on_site, product_on_sites


def _nearest_neighbor_pairs(n_qubits: int) -> list[tuple[int, int]]:
    return [(site, (site + 1) % n_qubits) for site in range(n_qubits)]


def build_hamiltonian(J: float, h: float, lambda_loop: float, n_qubits: int = 4) -> np.ndarray:
    """Build the minimal plaquette Hamiltonian on a ring of qubits.

    The default four-qubit case corresponds to a square plaquette with
    nearest-neighbor ZZ couplings, a uniform transverse field, and a
    nonlocal loop term.
    """
    if n_qubits < 2:
        raise ValueError("n_qubits must be at least 2.")

    dim = 2**n_qubits
    hamiltonian = np.zeros((dim, dim), dtype=complex)

    for i, j in _nearest_neighbor_pairs(n_qubits):
        hamiltonian += J * product_on_sites({i: Z, j: Z}, n_qubits)
    for site in range(n_qubits):
        hamiltonian += h * pauli_on_site(X, site, n_qubits)
    hamiltonian += lambda_loop * product_on_sites({site: Z for site in range(n_qubits)}, n_qubits)

    return hamiltonian
