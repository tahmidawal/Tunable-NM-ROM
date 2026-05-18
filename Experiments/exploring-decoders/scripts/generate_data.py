#!/usr/bin/env python
"""Generate analytical Poisson snapshots for the exploring-decoders experiment.

Each snapshot is the analytical solution
    u(x; mu) = (A / (pi^2 * sum k_i^2)) * prod_i sin(k_i pi x_i)
evaluated on the FD grid. Off-mesh ground truth is generated at training
and eval time from the same closed form — see training.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tunable_rom_decoders.fom.poisson_analytical import generate_dataset
from tunable_rom_decoders.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_total = cfg.n_train + cfg.n_val
    print(
        f"Generating {n_total} analytical Poisson-{cfg.spatial_dim}D snapshots "
        f"at N={cfg.N}"
    )
    U, freqs = generate_dataset(
        N=cfg.N, spatial_dim=cfg.spatial_dim, n_samples=n_total, seed=cfg.seed
    )
    U_train = U[: cfg.n_train]
    U_val = U[cfg.n_train :]
    freqs_train = freqs[: cfg.n_train]
    freqs_val = freqs[cfg.n_train :]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        U_train=U_train, U_val=U_val,
        freqs_train=freqs_train, freqs_val=freqs_val,
        N=cfg.N, spatial_dim=cfg.spatial_dim,
    )
    print(f"Saved {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
