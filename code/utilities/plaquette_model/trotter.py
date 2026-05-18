from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from edt.models.hamiltonians import X, Z, pauli_on_site, product_on_sites


def trotter_evolution(J: float, h: float, lambda_loop: float, dt: float, n_steps: int) -> np.ndarray:
    """Return the first-order Trotter unitary for the 4-qubit plaquette model."""
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1.")
    if dt < 0.0:
        raise ValueError("dt must be nonnegative.")

    n_qubits = 4
    dim = 2**n_qubits
    step = np.eye(dim, dtype=complex)

    zz_terms = [
        product_on_sites({site: Z, (site + 1) % n_qubits: Z}, n_qubits)
        for site in range(n_qubits)
    ]
    x_terms = [pauli_on_site(X, site, n_qubits) for site in range(n_qubits)]
    loop_term = product_on_sites({site: Z for site in range(n_qubits)}, n_qubits)

    for term in zz_terms:
        step = expm(-1j * dt * J * term) @ step
    for term in x_terms:
        step = expm(-1j * dt * h * term) @ step
    step = expm(-1j * dt * lambda_loop * loop_term) @ step

    total = np.eye(dim, dtype=complex)
    for _ in range(n_steps):
        total = step @ total
    return total
