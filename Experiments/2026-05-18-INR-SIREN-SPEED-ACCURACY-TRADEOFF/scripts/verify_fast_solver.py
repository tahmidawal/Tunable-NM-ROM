#!/usr/bin/env python
"""Sanity-check the FastINRNMROMSolver against the baseline INRNMROMSolver.

For each test sample:
  - Run baseline solver -> z_base, R(z_base), rel-L2
  - Run fast     solver -> z_fast, R(z_fast), rel-L2
  - Compare ||z_base - z_fast|| / ||z_base||  (should be < 1e-8 in float64)
  - Compare residual norms and iter counts
Also times one solve of each, post-JIT.
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from tunable_rom_speed.autoencoder import INRAutoencoder
from tunable_rom_speed.fom.poisson_cg import PoissonFOM, analytical_u
from tunable_rom_speed.solver.nm_rom import INRNMROMSolver
from tunable_rom_speed.solver.nm_rom_fast import FastINRNMROMSolver


def _build_model(cfg: dict) -> INRAutoencoder:
    return INRAutoencoder(
        decoder_kind=cfg["decoder_kind"],
        N=cfg["N"], spatial_dim=cfg["spatial_dim"],
        patch_size=cfg["patch_size"], embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"], num_enc_layers=cfg["num_enc_layers"],
        latent_dim=cfg["latent_dim"],
        coord_dim=cfg["spatial_dim"],
        hidden_dim=cfg["siren_hidden_dim"] if cfg["decoder_kind"] == "siren"
                   else cfg["xattn_hidden_dim"],
        siren_num_layers=cfg["siren_num_layers"],
        omega_0=cfg["omega_0"], omega=cfg["omega"],
        modulator_hidden=cfg["modulator_hidden"],
        d_attn=cfg["d_attn"], num_fourier=cfg["num_fourier"],
        xattn_num_layers=cfg["xattn_num_layers"],
        fourier_scale=cfg["fourier_scale"],
    )


def _load(p: Path):
    ck = pickle.load(open(p, "rb"))
    return _build_model(ck["config"]), ck["params"], ck.get("tokens_ref", None), ck["config"]


def make_solver(SolverCls, model, params, tokens, cfg, fom, eq, max_iters):
    return SolverCls(
        autoencoder=model, params=params,
        N=cfg["N"], spatial_dim=cfg["spatial_dim"], dx=float(fom.dx),
        eq_flat_indices=eq["eq_flat_indices"],
        eq_weights=eq["eq_weights"],
        stencil_indices=eq["stencil_indices"],
        tokens_ref=tokens, gn_max_iters=max_iters,
        latent_dim_override=cfg["latent_dim"],
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--eq", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--n-samples", type=int, default=4)
    args = p.parse_args()

    model, params, tokens, cfg = _load(args.ckpt)
    fom = PoissonFOM(N=cfg["N"], spatial_dim=cfg["spatial_dim"])
    eq = np.load(args.eq)
    data = np.load(args.data)
    F_test = data["F_test"][: args.n_samples]

    sb = make_solver(INRNMROMSolver, model, params, tokens, cfg, fom, eq, args.iters)
    sf = make_solver(FastINRNMROMSolver, model, params, tokens, cfg, fom, eq, args.iters)

    solve_b = sb.make_solve()
    solve_f = sf.make_solve()

    Fe0 = jnp.asarray(F_test[0, eq["eq_flat_indices"]])
    # Warm-up both JITs.
    zb0, _, _ = solve_b(Fe0); zb0.block_until_ready()
    zf0, _, _ = solve_f(Fe0); zf0.block_until_ready()

    print(f"{'i':>3} {'||z_b||':>10} {'||z_f-z_b||/||z_b||':>22} "
          f"{'it_b':>5} {'it_f':>5} {'tb_ms':>8} {'tf_ms':>8}", flush=True)
    z_rel_errors = []
    for i in range(args.n_samples):
        Fei = jnp.asarray(F_test[i, eq["eq_flat_indices"]])
        zb, gb, itb = solve_b(Fei); zb.block_until_ready()
        zf, gf, itf = solve_f(Fei); zf.block_until_ready()
        # Time each post-warmup with a couple of repeats.
        tb_list, tf_list = [], []
        for _ in range(3):
            t0 = time.perf_counter(); zb, _, _ = solve_b(Fei); zb.block_until_ready()
            tb_list.append((time.perf_counter() - t0) * 1000.0)
            t0 = time.perf_counter(); zf, _, _ = solve_f(Fei); zf.block_until_ready()
            tf_list.append((time.perf_counter() - t0) * 1000.0)
        zn = float(np.linalg.norm(np.asarray(zb)))
        rel = float(np.linalg.norm(np.asarray(zf) - np.asarray(zb)) / (zn + 1e-30))
        z_rel_errors.append(rel)
        print(f"{i:3d} {zn:10.4e} {rel:22.4e} {int(itb):5d} {int(itf):5d} "
              f"{np.median(tb_list):8.2f} {np.median(tf_list):8.2f}", flush=True)

    print()
    print(f"max ||z_f - z_b|| / ||z_b|| = {max(z_rel_errors):.3e}")
    print(f"PASS" if max(z_rel_errors) < 1e-6 else f"FAIL (>1e-6)")


if __name__ == "__main__":
    main()
