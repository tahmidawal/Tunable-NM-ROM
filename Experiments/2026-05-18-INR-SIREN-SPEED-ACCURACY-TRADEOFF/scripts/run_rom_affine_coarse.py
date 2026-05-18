#!/usr/bin/env python
"""Cold-start ROM solve for an AffineZDecoder checkpoint using AffineNMROMSolver.

This is the lever-D analogue of run_rom_coarse_only.py — single-stage,
single-checkpoint, cold-start. The affine-specialized solver precomputes
Phi(x_eq) once per solve, giving CP-decoder-style per-iter cost.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from tunable_rom_speed.autoencoder import INRAutoencoder
from tunable_rom_speed.fom.poisson_cg import PoissonFOM, analytical_u
from tunable_rom_speed.solver.nm_rom_affine import AffineNMROMSolver


def _build_model(cfg: dict) -> INRAutoencoder:
    return INRAutoencoder(
        decoder_kind=cfg["decoder_kind"],
        N=cfg["N"], spatial_dim=cfg["spatial_dim"],
        patch_size=cfg["patch_size"], embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"], num_enc_layers=cfg["num_enc_layers"],
        latent_dim=cfg["latent_dim"],
        coord_dim=cfg["spatial_dim"],
        hidden_dim=cfg["siren_hidden_dim"],
        siren_num_layers=cfg["siren_num_layers"],
        omega_0=cfg["omega_0"], omega=cfg["omega"],
        modulator_hidden=cfg["modulator_hidden"],
        # The affine decoder ignores xattn params but they must be present.
        d_attn=cfg.get("d_attn", 64),
        num_fourier=cfg.get("num_fourier", 16),
        xattn_num_layers=cfg.get("xattn_num_layers", 3),
        fourier_scale=cfg.get("fourier_scale", 4.0),
        affine_bias_hidden=cfg.get("affine_bias_hidden", 128),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--eq", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    ck = pickle.load(open(args.ckpt, "rb"))
    model = _build_model(ck["config"])
    params = ck["params"]
    cfg = ck["config"]
    fom = PoissonFOM(N=cfg["N"], spatial_dim=cfg["spatial_dim"])

    eq = np.load(args.eq); data = np.load(args.data)
    F_test = data["F_test"]; freqs_test = data["freqs_test"]; U_test = data["U_test"]
    print(f"[aff] ckpt={args.ckpt.name} eq n={eq['eq_flat_indices'].size} iters={args.iters}",
          flush=True)

    solver = AffineNMROMSolver(
        autoencoder=model, params=params,
        N=cfg["N"], spatial_dim=cfg["spatial_dim"], dx=float(fom.dx),
        eq_flat_indices=eq["eq_flat_indices"],
        eq_weights=eq["eq_weights"],
        stencil_indices=eq["stencil_indices"],
        tokens_ref=None, gn_max_iters=args.iters,
        latent_dim_override=cfg["latent_dim"],
    )
    solve = solver.make_solve()

    F0 = jnp.asarray(F_test[0]); x0 = analytical_u(fom, freqs_test[0].tolist())
    _ = fom.cg_solve(F0, x0=x0).block_until_ready()

    n = F_test.shape[0]; fom_t = np.empty(n)
    for i in range(n):
        Fi = jnp.asarray(F_test[i]); xi = analytical_u(fom, freqs_test[i].tolist())
        t0 = time.perf_counter()
        u = fom.cg_solve(Fi, x0=xi); u.block_until_ready()
        fom_t[i] = time.perf_counter() - t0
    fom_med = float(np.median(fom_t))

    # Warm up JIT.
    Fe0 = jnp.asarray(F_test[0, eq["eq_flat_indices"]])
    z_w, _, _ = solve(Fe0); z_w.block_until_ready()

    F_test_e = F_test[:, eq["eq_flat_indices"]]
    mask_flat = np.asarray(fom.mask).reshape(-1)
    rel_l2 = np.empty(n); tot_t = np.empty(n); iters_arr = np.empty(n, dtype=np.int32)
    for i in range(n):
        Fe = jnp.asarray(F_test_e[i])
        t0 = time.perf_counter()
        z, _, it = solve(Fe); z.block_until_ready()
        tot_t[i] = time.perf_counter() - t0
        iters_arr[i] = int(it)
        u_pred = np.asarray(solver.decode_full_grid(z)) * mask_flat
        rel_l2[i] = float(np.linalg.norm(u_pred - U_test[i]) /
                          (np.linalg.norm(U_test[i]) + 1e-12))

    out = {
        "ckpt": str(args.ckpt), "eq": str(args.eq), "iters": args.iters,
        "fom_time_median": fom_med,
        "rom_time_median": float(np.median(tot_t)),
        "speedup_median": fom_med / float(np.median(tot_t)),
        "rom_relL2_median": float(np.median(rel_l2)),
        "rom_relL2_p10": float(np.quantile(rel_l2, 0.10)),
        "rom_relL2_p90": float(np.quantile(rel_l2, 0.90)),
        "rom_relL2_max": float(rel_l2.max()),
        "frac_le_1e-2": float((rel_l2 < 1e-2).mean()),
        "frac_le_5e-3": float((rel_l2 < 5e-3).mean()),
        "iters_median": float(np.median(iters_arr)),
        "iters_max": int(iters_arr.max()),
        "iters_coarse_median": float(np.median(iters_arr)),
        "iters_fine_median": 0.0,
        "rel_l2_all": rel_l2.tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[aff] rel-L2 med {out['rom_relL2_median']:.3e} p90 {out['rom_relL2_p90']:.3e}  "
          f"speed {out['speedup_median']:.2f}x  iters med {out['iters_median']:.0f}/{out['iters_max']}",
          flush=True)


if __name__ == "__main__":
    main()
