from __future__ import annotations

import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron_n(ops: list[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def pauli_on_site(pauli: np.ndarray, site: int, n_qubits: int) -> np.ndarray:
    ops = [I2] * n_qubits
    ops[site] = pauli
    return kron_n(ops)


def product_on_sites(pauli_map: dict[int, np.ndarray], n_qubits: int) -> np.ndarray:
    ops = [I2] * n_qubits
    for site, op in pauli_map.items():
        ops[site] = op
    return kron_n(ops)
