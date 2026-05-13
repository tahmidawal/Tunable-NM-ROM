#!/usr/bin/env python
"""Generate parametric Heat FOM training data for a given resolution.

Usage:
    python -m scripts.generate_data --config configs/heat2d_n64.yaml --out data/heat2d_n64.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np

from tunable_rom_heat.fom.heat import (
    HeatFOM,
    NUM_STEPS,
    generate_trajectory,
    sample_parameters,
)
from tunable_rom_heat.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    fom = HeatFOM(N=cfg.N, spatial_dim=cfg.spatial_dim)
    rng = np.random.default_rng(args.seed)

    def gen_set(n):
        snapshots = []
        per_traj = []
        params_list = []
        kappas = []
        for _ in range(n):
            p = sample_parameters(rng, cfg.spatial_dim)
            traj = np.asarray(generate_trajectory(fom, p, num_steps=NUM_STEPS))
            snapshots.append(traj)
            per_traj.append(traj)
            params_list.append(p)
            kappas.append(p["kappa"])
        return (
            np.concatenate(snapshots, axis=0).astype(np.float32),
            per_traj,
            params_list,
            np.asarray(kappas, dtype=np.float32),
        )

    print(f"Generating {cfg.n_train} train + {cfg.n_val} val trajectories at N={cfg.N}, d={cfg.spatial_dim}")
    U_train, train_traj, train_params, train_kappa = gen_set(cfg.n_train)
    U_val, val_traj, val_params, val_kappa = gen_set(cfg.n_val)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        U_train=U_train,
        U_val=U_val,
        train_kappa=train_kappa,
        val_kappa=val_kappa,
        N=cfg.N,
        spatial_dim=cfg.spatial_dim,
    )
    print(f"Saved {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
