from __future__ import annotations

import argparse

from edt.quantum.plaquette import FourQubitPlaquetteModel, PlaquetteParams


def main() -> None:
    parser = argparse.ArgumentParser(description="EDT research scaffold CLI")
    parser.add_argument("--lam", type=float, default=0.8)
    args = parser.parse_args()

    model = FourQubitPlaquetteModel(PlaquetteParams(lam=args.lam))
    print("Hamiltonian shape:", model.hamiltonian.shape)
    print("Plaquette term trace:", model.plaquette_term.trace())


if __name__ == "__main__":
    main()
