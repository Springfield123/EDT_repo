from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code" / "utilities") not in sys.path:
    sys.path.insert(0, str(ROOT / "code" / "utilities"))

from edt.models.hamiltonians import X, Z, pauli_on_site
from plaquette_model import (
    build_hamiltonian,
    compute_entanglement_entropy,
    compute_otoc,
    compute_two_point_correlations,
    exact_time_evolution,
)


FIGURE_PATH = ROOT / "outputs" / "reproduced_figures" / "fig4_plaquette_dynamics.pdf"
CSV_PATH = ROOT / "data" / "processed" / "plaquette_lambda_sweep.csv"

N_QUBITS = 4
LAMBDA_VALUES = np.linspace(0.0, 1.40, 29)
H_VALUES = np.array([0.25, 0.40, 0.55, 0.70])
J_VALUES = np.array([0.40, 0.60, 0.80])
J_NOMINAL = 0.40
H_NOMINAL = 0.40
TIMES = np.linspace(0.0, 10.0, 201)
MID_MASK = (TIMES >= 2.0) & (TIMES <= 5.5)
LATE_MASK = TIMES >= 7.0
ENTROPY_SUBSYSTEM = [0]
MAX_ENTROPY = float(len(ENTROPY_SUBSYSTEM))


@dataclass(frozen=True)
class Probe:
    label: str
    psi0: np.ndarray
    W: np.ndarray
    V: np.ndarray


def basis_state(bitstring: str) -> np.ndarray:
    state = np.zeros(2 ** len(bitstring), dtype=complex)
    state[int(bitstring, 2)] = 1.0
    return state


def plus_state(n_qubits: int) -> np.ndarray:
    state = np.ones(2**n_qubits, dtype=complex) / np.sqrt(2**n_qubits)
    return state


def build_probes() -> list[Probe]:
    return [
        Probe(
            label="psi0000_Z0_X2",
            psi0=basis_state("0000"),
            W=pauli_on_site(Z, 0, N_QUBITS),
            V=pauli_on_site(X, 2, N_QUBITS),
        ),
        Probe(
            label="psi0101_Z1_X3",
            psi0=basis_state("0101"),
            W=pauli_on_site(Z, 1, N_QUBITS),
            V=pauli_on_site(X, 3, N_QUBITS),
        ),
        Probe(
            label="psiplus_Z0_X1",
            psi0=plus_state(N_QUBITS),
            W=pauli_on_site(Z, 0, N_QUBITS),
            V=pauli_on_site(X, 1, N_QUBITS),
        ),
    ]


def normalized(values: np.ndarray) -> np.ndarray:
    vmax = np.max(values) if values.size else 0.0
    if vmax <= 0.0:
        return np.zeros_like(values)
    return values / vmax


def stability_from_entropy(entropy_mean: np.ndarray) -> np.ndarray:
    entropy_norm = np.clip(entropy_mean / MAX_ENTROPY, 0.0, 1.0)
    return 4.0 * entropy_norm * (1.0 - entropy_norm)


def evaluate_point(hamiltonian: np.ndarray, probe: Probe) -> dict[str, object]:
    states = exact_time_evolution(hamiltonian, probe.psi0, TIMES)
    otoc = compute_otoc(hamiltonian, probe.W, probe.V, probe.psi0, TIMES)
    correlations = compute_two_point_correlations(states)
    entropy = compute_entanglement_entropy(states, subsystem=ENTROPY_SUBSYSTEM)

    return {
        "probe_label": probe.label,
        "otoc_trace": otoc,
        "correlation_trace": np.mean(np.abs(correlations), axis=1),
        "entropy_trace": entropy,
        "spread_raw": float(np.mean(otoc[MID_MASK])),
        "corr_raw": float(np.mean(np.abs(correlations[LATE_MASK]))),
        "entropy_raw": float(np.mean(entropy[LATE_MASK])),
    }


def run_sweep() -> tuple[pd.DataFrame, dict[tuple[float, float, float], list[dict[str, object]]]]:
    probes = build_probes()
    rows: list[dict[str, float | int | str | bool]] = []
    traces: dict[tuple[float, float, float], list[dict[str, object]]] = {}

    for j_val in J_VALUES:
        for h_val in H_VALUES:
            curve_entries: list[dict[str, object]] = []
            for lambda_loop in LAMBDA_VALUES:
                hamiltonian = build_hamiltonian(J=j_val, h=h_val, lambda_loop=lambda_loop, n_qubits=N_QUBITS)
                probe_results = [evaluate_point(hamiltonian, probe) for probe in probes]

                spread_raw = np.mean([result["spread_raw"] for result in probe_results], dtype=float)
                corr_raw = np.mean([result["corr_raw"] for result in probe_results], dtype=float)
                entropy_raw = np.mean([result["entropy_raw"] for result in probe_results], dtype=float)
                spread_std = np.std([result["spread_raw"] for result in probe_results], ddof=1)
                corr_std = np.std([result["corr_raw"] for result in probe_results], ddof=1)
                entropy_std = np.std([result["entropy_raw"] for result in probe_results], ddof=1)

                entry = {
                    "J": float(j_val),
                    "h": float(h_val),
                    "lambda_loop": float(lambda_loop),
                    "spread_raw": float(spread_raw),
                    "corr_raw": float(corr_raw),
                    "entropy_raw": float(entropy_raw),
                    "spread_std": float(spread_std),
                    "corr_std": float(corr_std),
                    "entropy_std": float(entropy_std),
                    "probe_count": len(probes),
                }
                curve_entries.append(entry)
                traces[(float(j_val), float(h_val), float(lambda_loop))] = probe_results

            spread_curve = np.array([entry["spread_raw"] for entry in curve_entries], dtype=float)
            corr_curve = np.array([entry["corr_raw"] for entry in curve_entries], dtype=float)
            entropy_curve = np.array([entry["entropy_raw"] for entry in curve_entries], dtype=float)
            a_spread_curve = normalized(spread_curve)
            a_corr_curve = normalized(corr_curve)
            a_stability_curve = stability_from_entropy(entropy_curve)
            s_dyn_curve = a_spread_curve * a_corr_curve * a_stability_curve
            idx_star = int(np.argmax(s_dyn_curve))
            lambda_star = float(LAMBDA_VALUES[idx_star])

            for idx, entry in enumerate(curve_entries):
                entry["A_spread"] = float(a_spread_curve[idx])
                entry["A_corr"] = float(a_corr_curve[idx])
                entry["A_stability"] = float(a_stability_curve[idx])
                entry["S_dyn"] = float(s_dyn_curve[idx])
                entry["lambda_star_curve"] = lambda_star
                entry["is_optimal_curve"] = idx == idx_star
                rows.append(entry)

    return pd.DataFrame(rows), traces


def nearest_lambda(target: float) -> float:
    return float(LAMBDA_VALUES[np.argmin(np.abs(LAMBDA_VALUES - target))])


def make_figure(df: pd.DataFrame, traces: dict[tuple[float, float, float], list[dict[str, object]]]) -> None:
    nominal = df[(df["J"] == J_NOMINAL) & (df["h"] == H_NOMINAL)].sort_values("lambda_loop").reset_index(drop=True)
    lambda_star = float(nominal.loc[nominal["S_dyn"].idxmax(), "lambda_loop"])
    lambda_small = nearest_lambda(0.15)
    lambda_large = nearest_lambda(1.25)

    fig, axes = plt.subplots(3, 2, figsize=(9.0, 9.2))
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.ravel()

    style_map = {
        lambda_small: ("small $\\lambda$", "#1f78b4"),
        lambda_star: (r"optimal $\lambda^\star$", "#33a02c"),
        lambda_large: ("large $\\lambda$", "#e31a1c"),
    }
    for lambda_loop, (label, color) in style_map.items():
        key = (float(J_NOMINAL), float(H_NOMINAL), float(lambda_loop))
        probe_results = traces[key]
        otoc_stack = np.stack([result["otoc_trace"] for result in probe_results], axis=0)
        ax_a.plot(TIMES, np.mean(otoc_stack, axis=0), color=color, lw=1.8, label=label)
        ax_a.fill_between(
            TIMES,
            np.mean(otoc_stack, axis=0) - np.std(otoc_stack, axis=0),
            np.mean(otoc_stack, axis=0) + np.std(otoc_stack, axis=0),
            color=color,
            alpha=0.15,
            linewidth=0.0,
        )
    ax_a.set_title("(a) OTOC traces across the three regimes", loc="left", fontweight="bold")
    ax_a.set_xlabel("time")
    ax_a.set_ylabel(r"$1 - \mathrm{Re}\,F(t)$")
    ax_a.legend(frameon=False, loc="best")
    ax_a.grid(alpha=0.25)

    ax_b.plot(nominal["lambda_loop"], nominal["corr_raw"], color="#6a3d9a", lw=2.0)
    ax_b.fill_between(
        nominal["lambda_loop"],
        nominal["corr_raw"] - nominal["corr_std"],
        nominal["corr_raw"] + nominal["corr_std"],
        color="#cab2d6",
        alpha=0.35,
        linewidth=0.0,
    )
    ax_b.axvline(lambda_star, color="0.3", ls="--", lw=1.0)
    ax_b.set_title("(b) Long-time correlation persistence", loc="left", fontweight="bold")
    ax_b.set_xlabel(r"$\lambda$")
    ax_b.set_ylabel(r"$A_{\rm corr}^{\rm raw}$")
    ax_b.grid(alpha=0.25)

    ax_c.plot(nominal["lambda_loop"], nominal["entropy_raw"], color="#ff7f00", lw=2.0)
    ax_c.fill_between(
        nominal["lambda_loop"],
        nominal["entropy_raw"] - nominal["entropy_std"],
        nominal["entropy_raw"] + nominal["entropy_std"],
        color="#fdbf6f",
        alpha=0.35,
        linewidth=0.0,
    )
    ax_c.axvline(lambda_star, color="0.3", ls="--", lw=1.0)
    ax_c.set_title("(c) Entanglement entropy versus loop strength", loc="left", fontweight="bold")
    ax_c.set_xlabel(r"$\lambda$")
    ax_c.set_ylabel(r"$\overline{S_A}$ (bits)")
    ax_c.grid(alpha=0.25)

    ax_d.plot(nominal["lambda_loop"], nominal["S_dyn"], color="black", lw=2.2, label=r"$S_{\rm dyn}$")
    ax_d.plot(nominal["lambda_loop"], nominal["A_spread"], color="#1f78b4", lw=1.2, alpha=0.8, label=r"$A_{\rm spread}$")
    ax_d.plot(nominal["lambda_loop"], nominal["A_corr"], color="#6a3d9a", lw=1.2, alpha=0.8, label=r"$A_{\rm corr}$")
    ax_d.plot(nominal["lambda_loop"], nominal["A_stability"], color="#ff7f00", lw=1.2, alpha=0.8, label=r"$A_{\rm stability}$")
    ax_d.axvline(lambda_star, color="0.3", ls="--", lw=1.0)
    ax_d.scatter([lambda_star], [nominal["S_dyn"].max()], color="black", s=28, zorder=3)
    ax_d.set_title("(d) Finite optimum of $S_{\\rm dyn}$", loc="left", fontweight="bold")
    ax_d.set_xlabel(r"$\lambda$")
    ax_d.set_ylabel("normalized score")
    ax_d.legend(frameon=False, loc="upper left", ncol=2)
    ax_d.grid(alpha=0.25)

    for j_val, color in zip(J_VALUES, ["#1b9e77", "#d95f02", "#7570b3"]):
        subset = (
            df[df["J"] == j_val]
            .sort_values(["h", "lambda_loop"])
            .groupby("h", sort=True)
            .first()
            .reset_index()
        )
        ax_e.plot(subset["h"], subset["lambda_star_curve"], marker="o", color=color, lw=1.8, label=rf"$J={j_val:.2f}$")
    ax_e.set_title("(e) Robustness of $\\lambda^\\star$ under $h$ and $J$ variation", loc="left", fontweight="bold")
    ax_e.set_xlabel(r"$h$")
    ax_e.set_ylabel(r"$\lambda^\star$")
    ax_e.legend(frameon=False, loc="best")
    ax_e.grid(alpha=0.25)

    ax_f.axis("off")
    ax_f.text(
        0.0,
        1.0,
        "Regime summary\n\n"
        rf"small $\lambda={lambda_small:.2f}$: weak spreading\n"
        rf"optimal $\lambda^\star={lambda_star:.2f}$: structured spreading + persistent correlation\n"
        rf"large $\lambda={lambda_large:.2f}$: overdriven dynamics with degraded persistence\n\n"
        "All curves are averaged over three probes\n"
        "(two computational-basis initial states and one |++++> state,\n"
        "with distinct Pauli W,V choices).",
        ha="left",
        va="top",
        fontsize=8.5,
    )

    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.06, top=0.96, hspace=0.52, wspace=0.34)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH)
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.linewidth": 0.6,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    df, traces = run_sweep()
    df = df.sort_values(["J", "h", "lambda_loop"]).reset_index(drop=True)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    make_figure(df, traces)

    print("Saved:", FIGURE_PATH)
    print("Saved:", CSV_PATH)


if __name__ == "__main__":
    main()
