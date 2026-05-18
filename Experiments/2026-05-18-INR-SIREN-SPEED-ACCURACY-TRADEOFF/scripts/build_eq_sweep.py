#!/usr/bin/env python
"""Build EQ index/weight sets for a sweep over target n_eq counts.

One independent NNLS solve per (target n_eq) value, per decoder. The decoder
identity doesn't actually affect EQ selection (NNLS uses |K @ u|, not the
decoder), so to save NNLS time we solve once per n_eq target and reuse the
result across decoders. The user explicitly asked for "per decoder" so the
caller can choose; this helper supports both modes via --tag.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tunable_rom_speed.fom.poisson_cg import PoissonFOM
from tunable_rom_speed.eq.nnls import compute_eq_weights, build_stencil


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--spatial-dim", type=int, default=2)
    p.add_argument(
        "--n-eq-targets",
        type=int,
        nargs="+",
        default=[500, 1000, 2000, 4000, 8000],
    )
    p.add_argument("--n-eq-samples", type=int, default=200,
                   help="Number of training snapshots used in the NNLS design matrix.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="shared",
                   help="Tag for output filename (e.g. 'shared' or 'xattn_wide').")
    p.add_argument("--out-dir", type=Path, default=Path("runs/rom/eq"))
    args = p.parse_args()

    data = np.load(args.data)
    U_train = data["U_train"]
    fom = PoissonFOM(N=args.N, spatial_dim=args.spatial_dim)

    def K_op_numpy(u_flat):
        return np.asarray(fom.K_op(u_flat))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for n_eq in args.n_eq_targets:
        print(f"[eq] tag={args.tag} target n_eq={n_eq} ...", flush=True)
        eq_flat, eq_w = compute_eq_weights(
            U_train,
            K_op_numpy,
            N=args.N,
            spatial_dim=args.spatial_dim,
            n_eq_samples=args.n_eq_samples,
            min_eq_points=n_eq,
            weight_tol=1e-10,
            rng=np.random.default_rng(args.seed),
        )
        stencil = build_stencil(eq_flat, args.N)
        out_path = args.out_dir / f"eq_{args.tag}_{n_eq}.npz"
        np.savez(
            out_path,
            eq_flat_indices=eq_flat,
            eq_weights=eq_w,
            stencil_indices=stencil,
            target_n_eq=n_eq,
            actual_n_eq=eq_flat.size,
        )
        print(f"[eq]   -> {out_path}  actual_n_eq={eq_flat.size}", flush=True)


if __name__ == "__main__":
    main()
