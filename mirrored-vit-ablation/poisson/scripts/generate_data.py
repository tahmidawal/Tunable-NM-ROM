#!/usr/bin/env python
"""Generate parametric Poisson FOM training data for a given resolution.

Switches between analytical (continuous) and CG (discrete) generators
based on config.data_source. Always use cg for N>=256.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tunable_rom_vit_poisson.fom.data import generate_analytical, generate_cg
from tunable_rom_vit_poisson.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_total = cfg.n_train + cfg.n_val
    print(
        f"Generating {n_total} Poisson-{cfg.spatial_dim}D snapshots at N={cfg.N}"
        f" using data_source={cfg.data_source!r}"
    )

    if cfg.data_source == "analytical":
        U, freqs = generate_analytical(cfg.N, cfg.spatial_dim, n_total, seed=cfg.seed)
    elif cfg.data_source == "cg":
        U, freqs = generate_cg(cfg.N, cfg.spatial_dim, n_total, seed=cfg.seed)
    else:
        raise ValueError(f"Unknown data_source: {cfg.data_source}")

    U_train = U[: cfg.n_train]
    U_val = U[cfg.n_train:]
    freqs_train = freqs[: cfg.n_train]
    freqs_val = freqs[cfg.n_train:]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        U_train=U_train, U_val=U_val,
        freqs_train=freqs_train, freqs_val=freqs_val,
        N=cfg.N, spatial_dim=cfg.spatial_dim,
        data_source=cfg.data_source,
    )
    print(f"Saved {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
