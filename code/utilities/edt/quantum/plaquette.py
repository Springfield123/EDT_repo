from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from edt.models.hamiltonians import X, Z, pauli_on_site, product_on_sites


@dataclass(frozen=True)
class PlaquetteParams:
    hx: float = 0.8
    hz: float = 0.0
    jzz: float = 0.5
    lam: float = 0.8
    n_qubits: int = 4


class FourQubitPlaquetteModel:
    def __init__(self, params: PlaquetteParams):
        if params.n_qubits != 4:
            raise ValueError("This minimal model is implemented for 4 qubits.")
        self.params = params
        self.dim = 2 ** params.n_qubits
        self._build_terms()

    def _build_terms(self) -> None:
        p = self.params
        n = p.n_qubits

        self.local_x_terms = [pauli_on_site(X, i, n) for i in range(n)]
        self.local_z_terms = [pauli_on_site(Z, i, n) for i in range(n)]
        self.zz_terms = [
            product_on_sites({0: Z, 1: Z}, n),
            product_on_sites({1: Z, 2: Z}, n),
            product_on_sites({2: Z, 3: Z}, n),
            product_on_sites({3: Z, 0: Z}, n),
        ]
        self.plaquette_term = product_on_sites({0: Z, 1: Z, 2: Z, 3: Z}, n)

    @property
    def hamiltonian(self) -> np.ndarray:
        p = self.params
        h = np.zeros((self.dim, self.dim), dtype=complex)
        h += p.hx * sum(self.local_x_terms)
        h += p.hz * sum(self.local_z_terms)
        h += p.jzz * sum(self.zz_terms)
        h += p.lam * self.plaquette_term
        return h

    def basis_state(self, bitstring: str) -> np.ndarray:
        if len(bitstring) != self.params.n_qubits:
            raise ValueError("Bitstring length must match number of qubits.")
        index = int(bitstring, 2)
        state = np.zeros(self.dim, dtype=complex)
        state[index] = 1.0
        return state

    def evolve_exact(self, state: np.ndarray, time: float) -> np.ndarray:
        u = expm(-1j * time * self.hamiltonian)
        return u @ state

    def evolve_trotter(self, state: np.ndarray, time: float, steps: int) -> np.ndarray:
        if steps < 1:
            raise ValueError("steps must be >= 1")
        dt = time / steps
        u = np.eye(self.dim, dtype=complex)
        p = self.params
        for term in self.local_x_terms:
            u = expm(-1j * dt * p.hx * term) @ u
        for term in self.local_z_terms:
            if p.hz != 0.0:
                u = expm(-1j * dt * p.hz * term) @ u
        for term in self.zz_terms:
            u = expm(-1j * dt * p.jzz * term) @ u
        u = expm(-1j * dt * p.lam * self.plaquette_term) @ u

        out = state.copy()
        for _ in range(steps):
            out = u @ out
        return out

    def expectation(self, state: np.ndarray, op: np.ndarray) -> float:
        return float(np.real(np.vdot(state, op @ state)))

    def plaquette_expectation(self, state: np.ndarray) -> float:
        return self.expectation(state, self.plaquette_term)

    def total_zz_correlation(self, state: np.ndarray) -> float:
        vals = [self.expectation(state, term) for term in self.zz_terms]
        return float(np.mean(vals))

    def edt_proxy(self, state: np.ndarray) -> float:
        return float(self.plaquette_expectation(state) * self.total_zz_correlation(state))
