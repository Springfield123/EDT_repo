import numpy as np
from scipy.linalg import expm

from edt.models.hamiltonians import X, Z, pauli_on_site
from plaquette_model import (
    build_hamiltonian,
    compute_edt_dynamical_score,
    compute_entanglement_entropy,
    compute_otoc,
    compute_two_point_correlations,
    exact_time_evolution,
    trotter_evolution,
)


def test_build_hamiltonian_is_hermitian():
    hamiltonian = build_hamiltonian(J=0.5, h=0.2, lambda_loop=0.1)
    assert hamiltonian.shape == (16, 16)
    assert np.allclose(hamiltonian, hamiltonian.conj().T)


def test_exact_time_evolution_preserves_state_norm():
    hamiltonian = build_hamiltonian(J=0.4, h=0.3, lambda_loop=0.2)
    psi0 = np.zeros(16, dtype=complex)
    psi0[0] = 1.0
    states = exact_time_evolution(hamiltonian, psi0, np.linspace(0.0, 1.0, 5))
    norms = np.sum(np.abs(states) ** 2, axis=1)
    assert np.allclose(norms, 1.0)
    assert np.allclose(states[0], psi0)


def test_compute_otoc_is_zero_for_commuting_operators_when_h_is_zero():
    hamiltonian = build_hamiltonian(J=0.5, h=0.0, lambda_loop=0.2)
    psi0 = np.zeros(16, dtype=complex)
    psi0[0] = 1.0
    w = pauli_on_site(Z, 0, 4)
    v = pauli_on_site(X, 1, 4)
    otoc = compute_otoc(hamiltonian, w, v, psi0, [0.0, 0.2, 0.5])
    assert np.allclose(otoc, 0.0)


def test_correlations_and_entropy_for_product_state():
    psi0 = np.zeros(16, dtype=complex)
    psi0[0] = 1.0
    correlations = compute_two_point_correlations(np.array([psi0]))
    entropy = compute_entanglement_entropy(np.array([psi0]), subsystem=[0, 1])
    assert np.allclose(correlations, 1.0)
    assert np.allclose(entropy, 0.0)


def test_edt_score_prefers_intermediate_entropy():
    otoc = np.array([0.2, 0.8, 1.0])
    correlations = np.array([0.8, 0.9, 0.9])
    entropy = np.array([0.1, 0.5, 1.0])
    score = compute_edt_dynamical_score(otoc, correlations, entropy)
    assert score[1] > score[0]
    assert score[1] > score[2]


def test_trotter_evolution_matches_exact_short_time_unitary():
    j_val = 0.3
    h_val = 0.25
    loop_val = 0.15
    dt = 0.01
    n_steps = 5

    hamiltonian = build_hamiltonian(J=j_val, h=h_val, lambda_loop=loop_val)
    u_exact = expm(-1j * hamiltonian * dt * n_steps)
    u_trotter = trotter_evolution(j_val, h_val, loop_val, dt, n_steps)
    assert np.allclose(u_trotter.conj().T @ u_trotter, np.eye(16), atol=1e-10)
    assert np.linalg.norm(u_exact - u_trotter) < 5e-2
