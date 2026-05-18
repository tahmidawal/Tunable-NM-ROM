#!/usr/bin/env python
"""ROM solve with EQ-continuation cold-start:

  1) Solve cold (z=0) at n_eq=eq_coarse  (cheap).
  2) Use that z as warm-start for n_eq=eq_fine (more accurate).
  3) Report rel-L2 (vs CG FOM) and total wall time.

This costs ~2× the fine solve, so to keep the speedup story honest we
ALSO report median wall-time = (coarse_t + fine_t).

Designed for the iter2 anchor checkpoint (siren) but works for any
INRAutoencoder. Uses the existing solver with extended `gn_max_iters`.
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

from tunable_rom_speed.autoencoder import INRAutoencoder
from tunable_rom_speed.fom.poisson_cg import PoissonFOM
from tunable_rom_speed.solver.nm_rom import INRNMROMSolver


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


def _benchmark_fom(fom, F_test, freqs_test):
    from tunable_rom_speed.fom.poisson_cg import analytical_u
    n = F_test.shape[0]
    U = np.empty_like(F_test)
    times = np.empty(n, dtype=np.float64)
    F0 = jnp.asarray(F_test[0])
    x0 = analytical_u(fom, freqs_test[0].tolist())
    _ = fom.cg_solve(F0, x0=x0).block_until_ready()
    for i in range(n):
        Fi = jnp.asarray(F_test[i])
        x0 = analytical_u(fom, freqs_test[i].tolist())
        t0 = time.perf_counter()
        u = fom.cg_solve(Fi, x0=x0); u.block_until_ready()
        times[i] = time.perf_counter() - t0
        U[i] = np.asarray(u)
    return U, times


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--eq-coarse", required=True, type=Path,
                   help="EQ npz to use for the coarse cold-start.")
    p.add_argument("--eq-fine", required=True, type=Path,
                   help="EQ npz to use for the fine refinement.")
    p.add_argument("--gn-max-iters-coarse", type=int, default=30)
    p.add_argument("--gn-max-iters-fine", type=int, default=30)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    with open(args.ckpt, "rb") as f:
        ck = pickle.load(f)
    cfg = ck["config"]
    params = ck["params"]
    tokens_ref = ck.get("tokens_ref", None)

    model = _build_model(cfg)
    data = np.load(args.data)
    F_test = data["F_test"]; freqs_test = data["freqs_test"]; U_test = data["U_test"]

    fom = PoissonFOM(N=cfg["N"], spatial_dim=cfg["spatial_dim"])
    U_fom, fom_times = _benchmark_fom(fom, F_test, freqs_test)
    fom_median = float(np.median(fom_times))
    print(f"[cont] FOM CG median = {fom_median*1000:.2f} ms", flush=True)

    eq_c = np.load(args.eq_coarse); eq_f = np.load(args.eq_fine)
    print(f"[cont] coarse n_eq={eq_c['eq_flat_indices'].size}  "
          f"fine n_eq={eq_f['eq_flat_indices'].size}", flush=True)

    solver_c = INRNMROMSolver(
        autoencoder=model, params=params,
        N=cfg["N"], spatial_dim=cfg["spatial_dim"], dx=float(fom.dx),
        eq_flat_indices=eq_c["eq_flat_indices"],
        eq_weights=eq_c["eq_weights"],
        stencil_indices=eq_c["stencil_indices"],
        tokens_ref=tokens_ref,
        gn_max_iters=args.gn_max_iters_coarse,
    )
    solver_f = INRNMROMSolver(
        autoencoder=model, params=params,
        N=cfg["N"], spatial_dim=cfg["spatial_dim"], dx=float(fom.dx),
        eq_flat_indices=eq_f["eq_flat_indices"],
        eq_weights=eq_f["eq_weights"],
        stencil_indices=eq_f["stencil_indices"],
        tokens_ref=tokens_ref,
        gn_max_iters=args.gn_max_iters_fine,
    )
    solve_coarse_cold = solver_c.make_solve()
    solve_fine_warm = solver_f.make_solve_warm()

    eq_flat_c = eq_c["eq_flat_indices"]
    eq_flat_f = eq_f["eq_flat_indices"]
    F_test_c = F_test[:, eq_flat_c]
    F_test_f = F_test[:, eq_flat_f]

    # JIT warm-up.
    z_w, _, _ = solve_coarse_cold(jnp.asarray(F_test_c[0]))
    z_w.block_until_ready()
    z_w2, _, _ = solve_fine_warm(jnp.asarray(F_test_f[0]), z_w)
    z_w2.block_until_ready()

    n = F_test.shape[0]
    rel_l2 = np.empty(n)
    rom_t = np.empty(n)
    iters_coarse = np.empty(n, dtype=np.int32)
    iters_fine = np.empty(n, dtype=np.int32)
    mask_flat = np.asarray(fom.mask).reshape(-1)

    for i in range(n):
        Fc = jnp.asarray(F_test_c[i])
        Ff = jnp.asarray(F_test_f[i])
        t0 = time.perf_counter()
        z_c, _, it_c = solve_coarse_cold(Fc)
        z_c.block_until_ready()
        z_f, _, it_f = solve_fine_warm(Ff, z_c)
        z_f.block_until_ready()
        rom_t[i] = time.perf_counter() - t0
        iters_coarse[i] = int(it_c); iters_fine[i] = int(it_f)
        u_pred = np.asarray(solver_f.decode_full_grid(z_f)) * mask_flat
        u_true = U_test[i]
        rel_l2[i] = float(np.linalg.norm(u_pred - u_true) /
                          (np.linalg.norm(u_true) + 1e-12))

    out = {
        "ckpt": str(args.ckpt),
        "eq_coarse": str(args.eq_coarse),
        "eq_fine": str(args.eq_fine),
        "gn_max_iters_coarse": int(args.gn_max_iters_coarse),
        "gn_max_iters_fine": int(args.gn_max_iters_fine),
        "fom_time_median": fom_median,
        "rom_relL2_median": float(np.median(rel_l2)),
        "rom_relL2_p25": float(np.quantile(rel_l2, 0.25)),
        "rom_relL2_p75": float(np.quantile(rel_l2, 0.75)),
        "rom_relL2_p90": float(np.quantile(rel_l2, 0.9)),
        "rom_relL2_max": float(rel_l2.max()),
        "frac_le_5e-2": float((rel_l2 < 5e-2).mean()),
        "frac_le_1e-2": float((rel_l2 < 1e-2).mean()),
        "rom_time_median": float(np.median(rom_t)),
        "speedup_median": fom_median / float(np.median(rom_t)),
        "iters_coarse_median": float(np.median(iters_coarse)),
        "iters_fine_median": float(np.median(iters_fine)),
        "rel_l2_all": rel_l2.tolist(),
        "iters_coarse_all": iters_coarse.tolist(),
        "iters_fine_all": iters_fine.tolist(),
    }
    print(f"\n[cont] median rel-L2 = {out['rom_relL2_median']:.4e}  "
          f"p90 = {out['rom_relL2_p90']:.4e}  max = {out['rom_relL2_max']:.4e}  "
          f"frac<5e-2 = {out['frac_le_5e-2']:.3f}  "
          f"frac<1e-2 = {out['frac_le_1e-2']:.3f}  "
          f"spd = {out['speedup_median']:.1f}x")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[cont] saved {args.out}")


if __name__ == "__main__":
    main()
