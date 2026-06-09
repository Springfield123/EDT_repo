#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
demo_edt_gyro_trotter_4q_fig34_clean.py

Clean Figure-3 / Figure-4 pipeline for a 4-qubit EDT gyroscope toy model
with a true plaquette loop term:
    W_p = Z0 Z1 Z2 Z3

Hamiltonian:
    H = Omega Z0 + g Z0 X3 + J01 Z0 Z1 + J12 Z1 Z2 + hx X0 + lam Z0 Z1 Z2 Z3

Figures generated:
    figs/demo_edt_gyro_trotter_4q_plaquette_otoc.pdf
    figs/demo_edt_gyro_trotter_4q_plaquette_otoc.png
    figs/demo_edt_gyro_trotter_4q_circuit_example.pdf
    figs/demo_edt_gyro_trotter_4q_circuit_example.png
    figs/figure3_edt_regime_panel.pdf
    figs/figure3_edt_regime_panel.png
    figs/figure4_edt_scaling_complexity_panel.pdf
    figs/figure4_edt_scaling_complexity_panel.png
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator


# ============================================================
# Output / style
# ============================================================
ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "outputs" / "reproduced_figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavuserif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "lines.linewidth": 1.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Parameters
# ============================================================
@dataclass
class Plaquette4QParams:
    t_min: float = 0.0
    t_max: float = 10.0
    n_t: int = 41

    # Hamiltonian parameters
    Omega: float = 0.80
    g: float = 0.65
    J01: float = 0.32
    J12: float = 0.24
    hx: float = 0.22
    lam: float = 0.38

    # Initial-state preparation
    theta0: float = np.pi / 3.4
    theta1: float = np.pi / 5.0

    # Trotterization
    trotter_steps_per_unit_time: int = 5

    # Representative circuit draw time
    t_circuit_example: float = 3.0

    # OTOC settings: mixed-sector probe aligned with g Z0 X3
    otoc_w_qubit: int = 0
    otoc_v_qubit: int = 3
    otoc_w_axis: str = "Z"
    otoc_v_axis: str = "X"

    # Lambda sweep
    lam_min: float = 0.0
    lam_max: float = 1.0
    n_lam: int = 31


# ============================================================
# Helpers
# ============================================================
def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12, 1.03, label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def smooth_1d(y: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return y.copy()
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(ypad, kernel, mode="valid")


def time_window_average(times: np.ndarray, y: np.ndarray, t1: float, t2: float) -> float:
    mask = (times >= t1) & (times <= t2)
    if np.count_nonzero(mask) < 2:
        return np.nan
    yy = y[mask]
    tt = times[mask]
    return float(np.trapz(yy, tt) / (tt[-1] - tt[0] + 1e-12))


def first_threshold_crossing(times: np.ndarray, y: np.ndarray, threshold: float) -> float:
    idx = np.where(y >= threshold)[0]
    if len(idx) == 0:
        return np.nan
    return float(times[idx[0]])


def normalize_metric(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    m = np.nanmax(y)
    if not np.isfinite(m) or m <= 0:
        return np.zeros_like(y)
    return y / m


def extract_window(metric: np.ndarray, xvals: np.ndarray, frac: float = 0.94):
    valid = np.isfinite(metric)
    if not np.any(valid):
        return np.nan, np.nan, np.zeros_like(metric, dtype=bool)

    mmax = np.nanmax(metric)
    mask = valid & (metric >= frac * mmax)
    if not np.any(mask):
        return np.nan, np.nan, mask

    xx = xvals[mask]
    return float(xx[0]), float(xx[-1]), mask


def pauli_from_axis(axis: str) -> np.ndarray:
    axis = axis.upper()
    if axis == "X":
        return X
    if axis == "Y":
        return Y
    if axis == "Z":
        return Z
    raise ValueError(f"Unknown Pauli axis: {axis}")


# ============================================================
# Pauli matrices / operator builders
# ============================================================
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron_n(ops: list[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def dagger(mat: np.ndarray) -> np.ndarray:
    return np.conjugate(mat.T)


def op_on_qubit_exact(op: np.ndarray, q: int, n: int = 4) -> np.ndarray:
    ops = [I2] * n
    ops[q] = op
    return kron_n(ops)


def op_product_exact(op_map: dict[int, np.ndarray], n: int = 4) -> np.ndarray:
    ops = [I2] * n
    for q, op in op_map.items():
        ops[q] = op
    return kron_n(ops)


def op_on_qubit_qiskit(op: np.ndarray, q: int, n: int = 4) -> np.ndarray:
    ops = [I2] * n
    ops[n - 1 - q] = op
    return kron_n(ops)


def op_product_qiskit(op_map: dict[int, np.ndarray], n: int = 4) -> np.ndarray:
    ops = [I2] * n
    for q, op in op_map.items():
        ops[n - 1 - q] = op
    return kron_n(ops)


# ============================================================
# Exact model
# ============================================================
def build_initial_state_vector_exact(p: Plaquette4QParams) -> np.ndarray:
    ket0 = np.array([1.0, 0.0], dtype=complex)
    ket1 = np.array([0.0, 1.0], dtype=complex)
    ket_plus = (ket0 + ket1) / np.sqrt(2.0)

    psi_q0 = np.cos(p.theta0 / 2.0) * ket0 + np.sin(p.theta0 / 2.0) * ket1
    psi_q1 = np.cos(p.theta1 / 2.0) * ket0 + np.sin(p.theta1 / 2.0) * ket1
    psi_q2 = ket_plus
    psi_q3 = ket_plus

    psi0 = kron_n([psi_q0, psi_q1, psi_q2, psi_q3])
    return psi0 / np.linalg.norm(psi0)


def build_exact_hamiltonian(p: Plaquette4QParams) -> np.ndarray:
    Z0 = op_on_qubit_exact(Z, 0)
    X0 = op_on_qubit_exact(X, 0)
    Z0X3 = op_product_exact({0: Z, 3: X})
    Z0Z1 = op_product_exact({0: Z, 1: Z})
    Z1Z2 = op_product_exact({1: Z, 2: Z})
    Wp = op_product_exact({0: Z, 1: Z, 2: Z, 3: Z})

    return (
        p.Omega * Z0
        + p.g * Z0X3
        + p.J01 * Z0Z1
        + p.J12 * Z1Z2
        + p.hx * X0
        + p.lam * Wp
    )


def exact_evolution(H: np.ndarray, psi0: np.ndarray, times: np.ndarray) -> np.ndarray:
    evals, evecs = np.linalg.eigh(H)
    coeffs = dagger(evecs) @ psi0
    states = []
    for t in times:
        phase = np.exp(-1j * evals * t)
        psi_t = evecs @ (phase * coeffs)
        states.append(psi_t)
    return np.array(states)


def expectation_trajectory(states: np.ndarray, O: np.ndarray) -> np.ndarray:
    vals = np.einsum("ti,ij,tj->t", np.conjugate(states), O, states)
    return np.real_if_close(vals)


def compute_cinf_from_states(states: np.ndarray, O: np.ndarray, frac: float = 0.25) -> float:
    vals = expectation_trajectory(states, O)
    n = len(vals)
    m = max(2, int(frac * n))
    return float(np.mean(np.abs(vals[-m:])))


# ============================================================
# Qiskit circuit blocks
# ============================================================
def prepare_initial_state_circuit(qc: QuantumCircuit, p: Plaquette4QParams) -> None:
    qc.ry(p.theta0, 0)
    qc.ry(p.theta1, 1)
    qc.h(2)
    qc.h(3)


def append_exp_Z(qc: QuantumCircuit, qubit: int, angle: float) -> None:
    qc.rz(2.0 * angle, qubit)


def append_exp_X(qc: QuantumCircuit, qubit: int, angle: float) -> None:
    qc.h(qubit)
    qc.rz(2.0 * angle, qubit)
    qc.h(qubit)


def append_exp_ZZ(qc: QuantumCircuit, q0: int, q1: int, angle: float) -> None:
    qc.cx(q0, q1)
    qc.rz(2.0 * angle, q1)
    qc.cx(q0, q1)


def append_exp_ZX(qc: QuantumCircuit, qz: int, qx: int, angle: float) -> None:
    qc.h(qx)
    qc.cx(qz, qx)
    qc.rz(2.0 * angle, qx)
    qc.cx(qz, qx)
    qc.h(qx)


def append_exp_ZZZZ(qc: QuantumCircuit, q0: int, q1: int, q2: int, q3: int, angle: float) -> None:
    qc.cx(q0, q3)
    qc.cx(q1, q3)
    qc.cx(q2, q3)
    qc.rz(2.0 * angle, q3)
    qc.cx(q2, q3)
    qc.cx(q1, q3)
    qc.cx(q0, q3)


def build_trotter_evolution_circuit(total_time: float, p: Plaquette4QParams) -> QuantumCircuit:
    qc = QuantumCircuit(4)
    if total_time <= 0.0:
        return qc

    n_steps = max(1, int(np.ceil(p.trotter_steps_per_unit_time * total_time)))
    dt = total_time / n_steps

    for _ in range(n_steps):
        append_exp_Z(qc, 0, p.Omega * dt)
        append_exp_ZX(qc, 0, 3, p.g * dt)
        append_exp_ZZ(qc, 0, 1, p.J01 * dt)
        append_exp_ZZ(qc, 1, 2, p.J12 * dt)
        append_exp_X(qc, 0, p.hx * dt)
        append_exp_ZZZZ(qc, 0, 1, 2, 3, p.lam * dt)

    return qc


def build_trotter_circuit(total_time: float, p: Plaquette4QParams) -> QuantumCircuit:
    qc = QuantumCircuit(4)
    prepare_initial_state_circuit(qc, p)
    qc.compose(build_trotter_evolution_circuit(total_time, p), inplace=True)
    return qc


def qiskit_statevector_from_circuit(qc: QuantumCircuit) -> np.ndarray:
    sv = Statevector.from_instruction(qc)
    return np.asarray(sv.data, dtype=complex)


def trotter_evolution_states(times: np.ndarray, p: Plaquette4QParams) -> np.ndarray:
    states = []
    for t in times:
        qc = build_trotter_circuit(float(t), p)
        states.append(qiskit_statevector_from_circuit(qc))
    return np.array(states)


# ============================================================
# OTOC
# ============================================================
def heisenberg_operator_exact(H: np.ndarray, O: np.ndarray, t: float) -> np.ndarray:
    evals, evecs = np.linalg.eigh(H)
    U = evecs @ np.diag(np.exp(-1j * evals * t)) @ dagger(evecs)
    return dagger(U) @ O @ U


def heisenberg_operator_from_circuit(qc: QuantumCircuit, O_qiskit: np.ndarray) -> np.ndarray:
    U = Operator(qc).data
    return np.conjugate(U.T) @ O_qiskit @ U


def pure_state_expectation(psi: np.ndarray, O: np.ndarray) -> complex:
    return np.vdot(psi, O @ psi)


def compute_otoc_exact(
    H: np.ndarray,
    psi0: np.ndarray,
    W: np.ndarray,
    V: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    vals = []
    for t in times:
        Wt = heisenberg_operator_exact(H, W, t)
        F = pure_state_expectation(psi0, Wt @ V @ Wt @ V)
        vals.append(1.0 - np.real(F))
    return np.array(vals)


def compute_otoc_trotter(
    psi0_qiskit: np.ndarray,
    W_qiskit: np.ndarray,
    V_qiskit: np.ndarray,
    times: np.ndarray,
    p: Plaquette4QParams,
) -> np.ndarray:
    vals = []
    for t in times:
        qc_evol = build_trotter_evolution_circuit(float(t), p)
        Wt = heisenberg_operator_from_circuit(qc_evol, W_qiskit)
        F = pure_state_expectation(psi0_qiskit, Wt @ V_qiskit @ Wt @ V_qiskit)
        vals.append(1.0 - np.real(F))
    return np.array(vals)


# ============================================================
# Core analyses
# ============================================================
def run_demo(p: Plaquette4QParams):
    times = np.linspace(p.t_min, p.t_max, p.n_t)

    psi0_exact = build_initial_state_vector_exact(p)
    H_exact = build_exact_hamiltonian(p)
    exact_states = exact_evolution(H_exact, psi0_exact, times)

    trotter_states = trotter_evolution_states(times, p)

    Z0_exact = op_on_qubit_exact(Z, 0)
    Z0Z1_exact = op_product_exact({0: Z, 1: Z})
    Wp_exact = op_product_exact({0: Z, 1: Z, 2: Z, 3: Z})

    Z0_qiskit = op_on_qubit_qiskit(Z, 0)
    Z0Z1_qiskit = op_product_qiskit({0: Z, 1: Z})
    Wp_qiskit = op_product_qiskit({0: Z, 1: Z, 2: Z, 3: Z})

    sig_exact = expectation_trajectory(exact_states, Z0_exact)
    corr_exact = expectation_trajectory(exact_states, Z0Z1_exact)
    loop_exact = expectation_trajectory(exact_states, Wp_exact)

    sig_trot = expectation_trajectory(trotter_states, Z0_qiskit)
    corr_trot = expectation_trajectory(trotter_states, Z0Z1_qiskit)
    loop_trot = expectation_trajectory(trotter_states, Wp_qiskit)

    sig_err = np.abs(sig_exact - sig_trot)
    corr_err = np.abs(corr_exact - corr_trot)
    loop_err = np.abs(loop_exact - loop_trot)

    W_exact = op_on_qubit_exact(pauli_from_axis(p.otoc_w_axis), p.otoc_w_qubit)
    V_exact = op_on_qubit_exact(pauli_from_axis(p.otoc_v_axis), p.otoc_v_qubit)
    W_qiskit = op_on_qubit_qiskit(pauli_from_axis(p.otoc_w_axis), p.otoc_w_qubit)
    V_qiskit = op_on_qubit_qiskit(pauli_from_axis(p.otoc_v_axis), p.otoc_v_qubit)

    qc0 = QuantumCircuit(4)
    prepare_initial_state_circuit(qc0, p)
    psi0_qiskit = qiskit_statevector_from_circuit(qc0)

    otoc_exact = compute_otoc_exact(H_exact, psi0_exact, W_exact, V_exact, times)
    otoc_trot = compute_otoc_trotter(psi0_qiskit, W_qiskit, V_qiskit, times, p)
    otoc_err = np.abs(otoc_exact - otoc_trot)

    return {
        "times": times,
        "sig_exact": sig_exact,
        "sig_trot": sig_trot,
        "corr_exact": corr_exact,
        "corr_trot": corr_trot,
        "loop_exact": loop_exact,
        "loop_trot": loop_trot,
        "sig_err": sig_err,
        "corr_err": corr_err,
        "loop_err": loop_err,
        "otoc_exact": otoc_exact,
        "otoc_trot": otoc_trot,
        "otoc_err": otoc_err,
    }


def run_lambda_sweep(p: Plaquette4QParams):
    times = np.linspace(p.t_min, p.t_max, p.n_t)
    lam_values = np.linspace(p.lam_min, p.lam_max, p.n_lam)

    otoc_map = []
    peak_otoc = []
    mid_otoc = []
    t_peak_otoc = []
    tstar_otoc = []
    cinf_vals = []

    t1_mid = 0.20 * p.t_max
    t2_mid = 0.45 * p.t_max
    otoc_threshold = 0.05

    for lam in lam_values:
        pp = replace(p, lam=float(lam))
        psi0_exact = build_initial_state_vector_exact(pp)
        H_exact = build_exact_hamiltonian(pp)
        states_exact = exact_evolution(H_exact, psi0_exact, times)

        W_exact = op_on_qubit_exact(pauli_from_axis(pp.otoc_w_axis), pp.otoc_w_qubit)
        V_exact = op_on_qubit_exact(pauli_from_axis(pp.otoc_v_axis), pp.otoc_v_qubit)

        otoc = compute_otoc_exact(H_exact, psi0_exact, W_exact, V_exact, times)
        otoc_s = smooth_1d(otoc, window=5)

        otoc_map.append(otoc)
        peak_otoc.append(float(np.max(otoc_s)))
        mid_otoc.append(time_window_average(times, otoc_s, t1_mid, t2_mid))
        t_peak_otoc.append(float(times[np.argmax(otoc_s)]))
        tstar_otoc.append(first_threshold_crossing(times, otoc_s, otoc_threshold))

        Z0Z1 = op_product_exact({0: Z, 1: Z})
        cinf_vals.append(compute_cinf_from_states(states_exact, Z0Z1, frac=0.25))

    return (
        times,
        lam_values,
        np.array(otoc_map),
        np.array(peak_otoc),
        np.array(mid_otoc),
        np.array(t_peak_otoc),
        np.array(tstar_otoc),
        np.array(cinf_vals),
    )


def classify_phase(mid_otoc: np.ndarray, cinf_vals: np.ndarray, lam_values: np.ndarray) -> np.ndarray:
    phase = np.zeros_like(lam_values, dtype=int)
    mid_norm = normalize_metric(mid_otoc)
    cinf_norm = normalize_metric(cinf_vals)

    for i in range(len(lam_values)):
        if mid_norm[i] < 0.3:
            phase[i] = 0  # Local
        elif cinf_norm[i] < 0.4:
            phase[i] = 2  # Overdriven
        else:
            phase[i] = 1  # EDT
    return phase


def compute_edt_score(mid_otoc: np.ndarray, cinf_vals: np.ndarray) -> np.ndarray:
    return normalize_metric(mid_otoc) * normalize_metric(cinf_vals) ** 1.2


def run_hx_scaling_scan(p: Plaquette4QParams, hx_values: np.ndarray):
    lam_star = []
    score_star = []

    for hx in hx_values:
        pp = replace(p, hx=float(hx))
        (
            _times,
            lam_values,
            _otoc_map,
            _peak_otoc,
            mid_otoc,
            _t_peak_otoc,
            _tstar_otoc,
            cinf_vals,
        ) = run_lambda_sweep(pp)
        score = compute_edt_score(mid_otoc, cinf_vals)
        idx = int(np.argmax(score))
        lam_star.append(float(lam_values[idx]))
        score_star.append(float(score[idx]))

    return np.asarray(hx_values, dtype=float), np.asarray(lam_star), np.asarray(score_star)


def run_circuit_cost_scan(p: Plaquette4QParams, times: np.ndarray):
    depths = []
    sizes = []
    for t in times:
        qc = build_trotter_circuit(float(t), p)
        depths.append(float(qc.depth()))
        sizes.append(float(qc.size()))
    return np.asarray(times), np.asarray(depths), np.asarray(sizes)


def run_trotter_error_scan(p: Plaquette4QParams):
    out = run_demo(p)
    obs_err = np.maximum.reduce([
        out["sig_err"],
        out["corr_err"],
        out["loop_err"],
        out["otoc_err"],
    ])
    return out["times"], obs_err, out["sig_err"], out["corr_err"], out["loop_err"], out["otoc_err"]


def run_trotter_budget_scan(p: Plaquette4QParams, budgets: np.ndarray):
    lam_star = []
    score_star = []
    for budget in budgets:
        pp = replace(p, trotter_steps_per_unit_time=int(budget))
        (
            _times,
            lam_values,
            _otoc_map,
            _peak_otoc,
            mid_otoc,
            _t_peak_otoc,
            _tstar_otoc,
            cinf_vals,
        ) = run_lambda_sweep(pp)
        score = compute_edt_score(mid_otoc, cinf_vals)
        idx = int(np.argmax(score))
        lam_star.append(float(lam_values[idx]))
        score_star.append(float(score[idx]))
    return np.asarray(budgets, dtype=float), np.asarray(lam_star), np.asarray(score_star)


# ============================================================
# Figure builders
# ============================================================
def make_main_figure(p: Plaquette4QParams):
    out = run_demo(p)
    times = out["times"]

    fig = plt.figure(figsize=(7.2, 11.2))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.0, 1.0, 0.95, 0.9], hspace=0.58, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    add_panel_label(ax1, "(a)")
    ax1.plot(times, out["sig_exact"], label="exact")
    ax1.plot(times, out["sig_trot"], ls="--", label="Trotter circuit")
    ax1.set_xlabel("time")
    ax1.set_ylabel(r"$\langle Z_0(t)\rangle$")
    ax1.set_title("4-qubit plaquette gyroscope signal")
    ax1.legend(frameon=False, loc="upper right")
    clean_axes(ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    add_panel_label(ax2, "(b)")
    ax2.plot(times, out["corr_exact"], label="exact")
    ax2.plot(times, out["corr_trot"], ls="--", label="Trotter circuit")
    ax2.set_xlabel("time")
    ax2.set_ylabel(r"$\langle Z_0 Z_1(t)\rangle$")
    ax2.set_title("Two-body correlator")
    ax2.legend(frameon=False, loc="upper right")
    clean_axes(ax2)

    ax3 = fig.add_subplot(gs[1, 0])
    add_panel_label(ax3, "(c)")
    ax3.plot(times, out["loop_exact"], label="exact")
    ax3.plot(times, out["loop_trot"], ls="--", label="Trotter circuit")
    ax3.set_xlabel("time")
    ax3.set_ylabel(r"$\langle W_p(t)\rangle$")
    ax3.set_title(r"Plaquette observable, $W_p = Z_0 Z_1 Z_2 Z_3$")
    ax3.legend(frameon=False, loc="upper right")
    clean_axes(ax3)

    ax4 = fig.add_subplot(gs[1, 1])
    add_panel_label(ax4, "(d)")
    ax4.plot(times, out["otoc_exact"], label="exact")
    ax4.plot(times, out["otoc_trot"], ls="--", label="Trotter circuit")
    ax4.set_xlabel("time")
    ax4.set_ylabel(r"$C_{\mathrm{OTOC}}(t)$")
    ax4.set_title(r"OTOC with $W=Z_0,\ V=X_3$")
    ax4.legend(frameon=False, loc="upper right")
    clean_axes(ax4)

    ax5 = fig.add_subplot(gs[2, 0])
    add_panel_label(ax5, "(e)")
    ax5.plot(times, out["sig_err"], label=r"$|\Delta \langle Z_0\rangle|$")
    ax5.plot(times, out["corr_err"], ls="--", label=r"$|\Delta \langle Z_0 Z_1\rangle|$")
    ax5.set_xlabel("time")
    ax5.set_ylabel("absolute error")
    ax5.set_title("Trotter error: signal and correlator")
    ax5.legend(frameon=False, loc="upper left")
    clean_axes(ax5)

    ax6 = fig.add_subplot(gs[2, 1])
    add_panel_label(ax6, "(f)")
    ax6.plot(times, out["loop_err"], label=r"$|\Delta \langle W_p\rangle|$")
    ax6.plot(times, out["otoc_err"], ls="--", label=r"$|\Delta C_{\mathrm{OTOC}}|$")
    ax6.set_xlabel("time")
    ax6.set_ylabel("absolute error")
    ax6.set_title("Trotter error: loop and OTOC")
    ax6.legend(frameon=False, loc="upper right")
    clean_axes(ax6)

    ax7 = fig.add_subplot(gs[3, :])
    add_panel_label(ax7, "(g)")
    ax7.plot(times, out["loop_exact"], label=r"$\langle W_p(t)\rangle$")
    ax7.plot(times, out["otoc_exact"], ls="--", label=r"$C_{\mathrm{OTOC}}(t)$")
    ax7.set_xlabel("time")
    ax7.set_ylabel("exact metric value")
    ax7.set_title("Loop observable and OTOC as complementary EDT diagnostics")
    ax7.legend(frameon=False, loc="upper right", ncol=2)
    clean_axes(ax7)

    pdf_path = OUTDIR / "demo_edt_gyro_trotter_4q_plaquette_otoc.pdf"
    png_path = OUTDIR / "demo_edt_gyro_trotter_4q_plaquette_otoc.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    return out


def save_circuit_example(p: Plaquette4QParams) -> None:
    qc = build_trotter_circuit(p.t_circuit_example, p)
    pdf2 = OUTDIR / "demo_edt_gyro_trotter_4q_circuit_example.pdf"
    png2 = OUTDIR / "demo_edt_gyro_trotter_4q_circuit_example.png"
    qc.draw(output="mpl", filename=str(pdf2), fold=25, idle_wires=False)
    qc.draw(output="mpl", filename=str(png2), fold=25, idle_wires=False)

    print(f"Saved: {pdf2}")
    print(f"Saved: {png2}")
    print("\nRepresentative circuit stats")
    print("----------------------------")
    print(f"time  = {p.t_circuit_example}")
    print(f"depth = {qc.depth()}")
    print(f"size  = {qc.size()}")


def make_figure3(p: Plaquette4QParams):
    (
        times,
        lam_values,
        otoc_map,
        peak_otoc,
        mid_otoc,
        t_peak_otoc,
        tstar_otoc,
        cinf_vals,
    ) = run_lambda_sweep(p)

    lam_lo_otoc, lam_hi_otoc, otoc_mask = extract_window(mid_otoc, lam_values, frac=0.94)
    lam_lo_cinf, lam_hi_cinf, cinf_mask = extract_window(cinf_vals, lam_values, frac=0.94)
    edt_mask = otoc_mask & cinf_mask

    if np.any(edt_mask):
        edt_lams = lam_values[edt_mask]
        lam_lo_edt = float(edt_lams[0])
        lam_hi_edt = float(edt_lams[-1])
    else:
        lam_lo_edt, lam_hi_edt = np.nan, np.nan

    score_edt = compute_edt_score(mid_otoc, cinf_vals)
    phase = classify_phase(mid_otoc, cinf_vals, lam_values)

    hx_values = np.linspace(max(0.08, 0.5 * p.hx), 1.6 * p.hx, 9)
    hx_values, lam_star, _score_star = run_hx_scaling_scan(p, hx_values)

    fig = plt.figure(figsize=(7.25, 9.35))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.15, 1.0, 0.92, 0.92], width_ratios=[1.0, 1.0], hspace=0.62, wspace=0.42)

    axa = fig.add_subplot(gs[0, :])
    add_panel_label(axa, "(a)")
    im = axa.imshow(otoc_map, aspect="auto", origin="lower", extent=[times[0], times[-1], lam_values[0], lam_values[-1]])
    axa.set_xlabel("time")
    axa.set_ylabel(r"loop strength $\lambda$")
#    axa.set_title(r"OTOC map $C_{\mathrm{OTOC}}(t,\lambda)$", pad=6)
    cbar = plt.colorbar(im, ax=axa, pad=0.012, fraction=0.03)
    cbar.set_label(r"$C_{\mathrm{OTOC}}(t)$")
    clean_axes(axa)

    axb = fig.add_subplot(gs[1, :])
    add_panel_label(axb, "(b)")
    axb.plot(lam_values, peak_otoc, label="peak OTOC")
    axb.plot(lam_values, mid_otoc, ls="--", label="mid-time OTOC")
    axb.plot(lam_values, cinf_vals, ls="-.", label=r"$C_\infty$")
    if np.isfinite(lam_lo_otoc) and np.isfinite(lam_hi_otoc):
        axb.axvspan(lam_lo_otoc, lam_hi_otoc, alpha=0.08, color="C0")
    if np.isfinite(lam_lo_cinf) and np.isfinite(lam_hi_cinf):
        axb.axvspan(lam_lo_cinf, lam_hi_cinf, alpha=0.08, color="C2")
    if np.isfinite(lam_lo_edt) and np.isfinite(lam_hi_edt):
        axb.axvspan(lam_lo_edt, lam_hi_edt, alpha=0.25, color="gray")
    axb.axvline(p.lam, ls=":", linewidth=1.0)
    axb.set_xlabel(r"loop strength $\lambda$")
    axb.set_ylabel("amplitude metric")
#    axb.set_title(r"Alignment of OTOC summaries with $C_\infty$", pad=6)
    axb.legend(frameon=False, ncol=3, loc="upper left")
    clean_axes(axb)

    axc = fig.add_subplot(gs[2, 0])
    add_panel_label(axc, "(c)")
    axc.plot(lam_values, t_peak_otoc, label=r"$t_{\mathrm{peak}}$")
    axc.plot(lam_values, tstar_otoc, ls="--", label=r"$t^\star$")
    if np.isfinite(lam_lo_edt) and np.isfinite(lam_hi_edt):
        axc.axvspan(lam_lo_edt, lam_hi_edt, alpha=0.18)
    axc.axvline(p.lam, ls=":", linewidth=1.0)
    axc.set_xlabel(r"$\lambda$")
    axc.set_ylabel("time")
#    axc.set_title("OTOC time-structure metrics", pad=6)
    axc.legend(frameon=False, loc="upper left")
    clean_axes(axc)

    axd = fig.add_subplot(gs[2, 1])
    add_panel_label(axd, "(d)")
    axd.plot(lam_values, score_edt, label=r"$\tilde C_{\mathrm{mid}}\tilde C_\infty$")
    if np.isfinite(lam_lo_edt) and np.isfinite(lam_hi_edt):
        axd.axvspan(lam_lo_edt, lam_hi_edt, alpha=0.18, label="EDT overlap")
    lam_star_main = lam_values[np.argmax(score_edt)]
    axd.axvline(lam_star_main, ls="--", linewidth=1.2)
    axd.axvline(p.lam, ls=":", linewidth=1.0)
    axd.set_xlabel(r"$\lambda$")
    axd.set_ylabel("combined score")
#    axd.set_title("Combined EDT phase score", pad=6)
    axd.legend(frameon=False, loc="upper left")
    clean_axes(axd)

    axe = fig.add_subplot(gs[3, 0])
    add_panel_label(axe, "(e)")
    phase_img = phase.reshape(1, -1)
    axe.imshow(phase_img, aspect="auto", cmap="viridis", extent=[lam_values[0], lam_values[-1], 0, 1])
    axe.set_yticks([])
    axe.set_xlabel(r"$\lambda$")
#    axe.set_title("EDT phase classification")
    axe.text(0.05, 0.8, "Local", transform=axe.transAxes, color="white")
    axe.text(0.37, 0.8, "EDT", transform=axe.transAxes, color="white")
    axe.text(0.70, 0.8, "Overdriven", transform=axe.transAxes)
    clean_axes(axe)

    axf = fig.add_subplot(gs[3, 1])
    add_panel_label(axf, "(f)")
    axf.plot(hx_values, lam_star, marker="o", linewidth=2, label=r"$\lambda^\star(h_x)$")
    axf.axhline(np.mean(lam_star), ls="--", alpha=0.4)
    axf.axhline(p.lam, ls=":", linewidth=1.0)
    axf.set_xlabel(r"$h_x$")
    axf.set_ylabel(r"$\lambda^\star$")
#    axf.set_title("Scaling of the optimal EDT regime", pad=6)
    axf.legend(frameon=False, loc="best")
    clean_axes(axf)

#    fig.text(
#        0.02, 0.012,
#        "Figure 3 | Finite EDT regime from loop-controlled dynamics. "
#        "The intermediate EDT regime is identified by the coexistence of structured operator spreading "
#        "and persistent long-time correlation.",
#        ha="left", va="bottom", fontsize=7
#    )

    pdf_path = OUTDIR / "figure3_edt_regime_panel.pdf"
    png_path = OUTDIR / "figure3_edt_regime_panel.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

def run_hx_scaling_scan_fig4(p: Plaquette4QParams, hx_values: np.ndarray):
    lam_star = []
    score_star = []

    for hx in hx_values:
        pp = replace(p, hx=float(hx))
        (
            _times,
            lam_values,
            _otoc_map,
            _peak_otoc,
            mid_otoc,
            _t_peak_otoc,
            _tstar_otoc,
            cinf_vals,
        ) = run_lambda_sweep(pp)

        score = normalize_metric(mid_otoc) * normalize_metric(cinf_vals) ** 1.2
        i_star = int(np.argmax(score))

        lam_star.append(float(lam_values[i_star]))
        score_star.append(float(score[i_star]))

    return np.array(lam_star), np.array(score_star)

def make_figure4(p: Plaquette4QParams):
    hx_values = np.linspace(max(0.08, 0.5 * p.hx), 1.6 * p.hx, 9)
    hx_values, lam_star_hx, score_star_hx = run_hx_scaling_scan(p, hx_values)

    cost_times = np.linspace(max(0.2, p.t_min), p.t_max, 16)
    cost_times, depths, sizes = run_circuit_cost_scan(p, cost_times)

    err_times, obs_err, sig_err, corr_err, loop_err, otoc_err = run_trotter_error_scan(p)

    budgets = np.arange(2, 9)
    budgets, lam_star_budget, score_star_budget = run_trotter_budget_scan(p, budgets)

    fig = plt.figure(figsize=(7.35, 7.9))
    gs = fig.add_gridspec(3, 2, hspace=0.62, wspace=0.42)
#
    lam_star, score_star = run_hx_scaling_scan_fig4(p, hx_values)

    hx_min = hx_values[np.argmin(score_star)]
    score_min = score_star[np.argmin(score_star)]

    axa = fig.add_subplot(gs[0, 0])
    add_panel_label(axa, "(a)")
    axa.plot(hx_values, lam_star_hx, marker="o", label=r"$\lambda^\star$")
    axa.axhline(np.mean(lam_star_hx), ls="--", alpha=0.4)
    axa.axvline(0.28, ls="--", alpha=0.4)
    axa.set_xlabel(r"$h_x$")
    axa.set_ylabel(r"$\lambda^\star$")
    axa.set_title("Optimal loop strength versus control parameter")
    axa.legend(frameon=False, loc="best")
    clean_axes(axa)

    axb = fig.add_subplot(gs[0, 1])
    add_panel_label(axb, "(b)")
    #hx_min = hx_values[np.argmin(S_star)]
    axb.scatter(hx_min, score_min, s=60, zorder=3)
    axb.plot(hx_values, score_star_hx, marker="o", label=r"$S_{\mathrm{EDT}}^\star$")
    axb.set_xlabel(r"$h_x$")
    axb.set_ylabel("peak EDT score")
    axb.set_title("Strength of the EDT regime")
    axb.legend(frameon=False, loc="best")
    clean_axes(axb)

    axc = fig.add_subplot(gs[1, 0])
    add_panel_label(axc, "(c)")
    axc.plot(cost_times, depths, label="depth")
    axc.plot(cost_times, sizes, ls="--", label="size")
    axc.set_xlabel("time")
    axc.set_ylabel("circuit cost")
    axc.set_title("Trotter circuit cost versus evolution time")
    axc.text(0.05, 0.9, "∝ t", transform=axc.transAxes)
    axc.legend(frameon=False, loc="upper left")
    clean_axes(axc)

    axd = fig.add_subplot(gs[1, 1])
    add_panel_label(axd, "(d)")
    #axd.plot(err_times, obs_err, label="max observable error")
    #axd.plot(err_times, loop_err, ls="--", label=r"$W_p$ error")
    #axd.plot(err_times, otoc_err, ls=":", label="OTOC error")
    axd.plot(err_times, obs_err, label="observables")
    axd.plot(err_times, otoc_err, ls=":", label="OTOC")
    axd.set_xlabel("time")
    axd.set_ylabel("absolute error")
    axd.set_title("Trotter error versus evolution time")
    axd.legend(frameon=False, loc="upper left")
    clean_axes(axd)

    axe = fig.add_subplot(gs[2, 0])
    add_panel_label(axe, "(e)")
    axe.plot(budgets, lam_star_budget, marker="o", label=r"$\lambda^\star$")
    axe.set_xlabel("Trotter steps per unit time")
    axe.set_ylabel(r"$\lambda^\star$")
    axe.set_ylim(0.75, 0.78)
    axe.set_title("EDT optimum versus Trotter budget")
    axe.legend(frameon=False, loc="best")
    clean_axes(axe)

    axf = fig.add_subplot(gs[2, 1])
    add_panel_label(axf, "(f)")
    axf.plot(budgets, score_star_budget, marker="o", label=r"$S_{\mathrm{EDT}}^\star$")
    axf.set_xlabel("Trotter steps per unit time")
    axf.set_ylabel("peak EDT score")
    axf.set_ylim(0.67, 0.70)
    axf.set_title("EDT robustness under approximate compilation")
    axf.legend(frameon=False, loc="best")
    clean_axes(axf)

    fig.text(
        0.02, 0.012,
        "Figure 4 | Scaling, complexity, and hardware relevance of the EDT regime. "
        "The EDT operating window remains identifiable under parameter variation and approximate circuit implementation.",
        ha="left", va="bottom", fontsize=7
    )

    pdf_path = OUTDIR / "figure4_edt_scaling_complexity_panel.pdf"
    png_path = OUTDIR / "figure4_edt_scaling_complexity_panel.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


# ============================================================
# Main
# ============================================================
def main():
    p = Plaquette4QParams()
    out = make_main_figure(p)
    save_circuit_example(p)
    make_figure3(p)
    make_figure4(p)

    print("\nSanity checks")
    print("-------------")
    print(f"<Z0>(t=0) exact   = {out['sig_exact'][0]: .6f}")
    print(f"<Z0>(t=0) trotter = {out['sig_trot'][0]: .6f}")
    print(f"<Z0Z1>(t=0) exact   = {out['corr_exact'][0]: .6f}")
    print(f"<Z0Z1>(t=0) trotter = {out['corr_trot'][0]: .6f}")
    print(f"<Wp>(t=0) exact   = {out['loop_exact'][0]: .6f}")
    print(f"<Wp>(t=0) trotter = {out['loop_trot'][0]: .6f}")
    print(f"OTOC(t=0) exact   = {out['otoc_exact'][0]: .6f}")
    print(f"OTOC(t=0) trotter = {out['otoc_trot'][0]: .6f}")


if __name__ == "__main__":
    main()
