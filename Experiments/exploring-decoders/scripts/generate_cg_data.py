#!/usr/bin/env python
"""Generate CG-discrete Poisson snapshots for ROM-faithful retraining + ROM eval.

Writes data/poisson2d_cg_n{N}_s{seed}.npz with train/val/test splits.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tunable_rom_decoders.fom.poisson_cg import generate_cg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--spatial-dim", type=int, default=2)
    p.add_argument("--n-train", type=int, default=700)
    p.add_argument("--n-val", type=int, default=140)
    p.add_argument("--n-test", type=int, default=160)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    total = args.n_train + args.n_val + args.n_test
    print(f"[gen_cg] N={args.N} d={args.spatial_dim} samples={total} seed={args.seed}")
    U, freqs, F = generate_cg(args.N, args.spatial_dim, total, seed=args.seed)

    U_train, U_val, U_test = (
        U[: args.n_train],
        U[args.n_train : args.n_train + args.n_val],
        U[args.n_train + args.n_val :],
    )
    fr_train, fr_val, fr_test = (
        freqs[: args.n_train],
        freqs[args.n_train : args.n_train + args.n_val],
        freqs[args.n_train + args.n_val :],
    )
    F_train, F_val, F_test = (
        F[: args.n_train],
        F[args.n_train : args.n_train + args.n_val],
        F[args.n_train + args.n_val :],
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        U_train=U_train, U_val=U_val, U_test=U_test,
        freqs_train=fr_train, freqs_val=fr_val, freqs_test=fr_test,
        F_train=F_train, F_val=F_val, F_test=F_test,
    )
    print(f"[gen_cg] Saved to {args.out}")
    print(f"[gen_cg] |U_train|={U_train.shape}  |U_val|={U_val.shape}  |U_test|={U_test.shape}")


if __name__ == "__main__":
    main()
