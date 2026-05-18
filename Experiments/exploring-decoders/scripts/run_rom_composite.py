#!/usr/bin/env python
"""Composite ROM solve: use one checkpoint to find a cheap coarse z, then
warm-start the actual ROM solve with a (possibly different) checkpoint.

Three modes are supported via flags:

  --mode self        Use the same checkpoint for coarse and fine. Seeds
                     by either z=0 (cold) or by encode(u_seed_field). Set
                     --seed-mode to one of {zero, u_mean, u_zero_field}.
  --mode cross       Two checkpoints: coarse_ckpt and fine_ckpt. The
                     coarse runs cold (z=0) and decodes to a field; that
                     field is encoded with the fine_ckpt to produce z₀
                     for the fine solve.

The fine-solve sweep is done at n_eq=fine_n_eq with `gn_max_iters`.
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

from tunable_rom_decoders.autoencoder import INRAutoencoder
from tunable_rom_decoders.fom.poisson_cg import PoissonFOM, analytical_u
from tunable_rom_decoders.solver.nm_rom import INRNMROMSolver


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


def _load_pair(ckpt_path: Path):
    with open(ckpt_path, "rb") as f:
        ck = pickle.load(f)
    cfg = ck["config"]
    params = ck["params"]
    tokens_ref = ck.get("tokens_ref", None)
    return _build_model(cfg), params, tokens_ref, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coarse-ckpt", required=True, type=Path)
    p.add_argument("--fine-ckpt", required=True, type=Path)
    p.add_argument("--coarse-eq", required=True, type=Path)
    p.add_argument("--fine-eq", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--coarse-iters", type=int, default=30)
    p.add_argument("--fine-iters", type=int, default=30)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    model_c, params_c, tokens_c, cfg_c = _load_pair(args.coarse_ckpt)
    model_f, params_f, tokens_f, cfg_f = _load_pair(args.fine_ckpt)
    assert cfg_c["N"] == cfg_f["N"]
    assert cfg_c["spatial_dim"] == cfg_f["spatial_dim"]
    print(f"[comp] coarse_ckpt={args.coarse_ckpt}  fine_ckpt={args.fine_ckpt}", flush=True)
    print(f"[comp]   coarse latent={cfg_c['latent_dim']}  fine latent={cfg_f['latent_dim']}",
          flush=True)

    fom = PoissonFOM(N=cfg_c["N"], spatial_dim=cfg_c["spatial_dim"])
    data = np.load(args.data)
    F_test = data["F_test"]; freqs_test = data["freqs_test"]; U_test = data["U_test"]

    # FOM CG benchmark.
    F0 = jnp.asarray(F_test[0]); x0 = analytical_u(fom, freqs_test[0].tolist())
    _ = fom.cg_solve(F0, x0=x0).block_until_ready()
    n = F_test.shape[0]
    U_fom = np.empty_like(F_test); fom_t = np.empty(n)
    for i in range(n):
        Fi = jnp.asarray(F_test[i])
        x0 = analytical_u(fom, freqs_test[i].tolist())
        t0 = time.perf_counter()
        u = fom.cg_solve(Fi, x0=x0); u.block_until_ready()
        fom_t[i] = time.perf_counter() - t0
        U_fom[i] = np.asarray(u)
    fom_med = float(np.median(fom_t))
    print(f"[comp] FOM CG median = {fom_med*1000:.2f} ms", flush=True)

    eq_c = np.load(args.coarse_eq); eq_f = np.load(args.fine_eq)
    print(f"[comp] eq coarse n={eq_c['eq_flat_indices'].size}  "
          f"eq fine n={eq_f['eq_flat_indices'].size}", flush=True)

    solver_c = INRNMROMSolver(
        autoencoder=model_c, params=params_c,
        N=cfg_c["N"], spatial_dim=cfg_c["spatial_dim"], dx=float(fom.dx),
        eq_flat_indices=eq_c["eq_flat_indices"],
        eq_weights=eq_c["eq_weights"],
        stencil_indices=eq_c["stencil_indices"],
        tokens_ref=tokens_c,
        gn_max_iters=args.coarse_iters,
        latent_dim_override=cfg_c["latent_dim"],
    )
    solver_f = INRNMROMSolver(
        autoencoder=model_f, params=params_f,
        N=cfg_f["N"], spatial_dim=cfg_f["spatial_dim"], dx=float(fom.dx),
        eq_flat_indices=eq_f["eq_flat_indices"],
        eq_weights=eq_f["eq_weights"],
        stencil_indices=eq_f["stencil_indices"],
        tokens_ref=tokens_f,
        gn_max_iters=args.fine_iters,
        latent_dim_override=cfg_f["latent_dim"],
    )

    solve_c_cold = solver_c.make_solve()
    solve_f_warm = solver_f.make_solve_warm()

    @jax.jit
    def encode_fine(u_flat):
        return solver_f.encode_warm_start(u_flat)

    # Warm-up JIT.
    Fc0 = jnp.asarray(F_test[0, eq_c["eq_flat_indices"]])
    z_c0, _, _ = solve_c_cold(Fc0); z_c0.block_until_ready()
    u_c0 = solver_c.decode_full_grid(z_c0)
    z_f0 = encode_fine(u_c0); z_f0.block_until_ready()
    Ff0 = jnp.asarray(F_test[0, eq_f["eq_flat_indices"]])
    z_fr, _, _ = solve_f_warm(Ff0, z_f0); z_fr.block_until_ready()

    F_test_c = F_test[:, eq_c["eq_flat_indices"]]
    F_test_f = F_test[:, eq_f["eq_flat_indices"]]
    mask_flat = np.asarray(fom.mask).reshape(-1)

    rel_l2 = np.empty(n); tot_t = np.empty(n)
    iters_c_arr = np.empty(n, dtype=np.int32); iters_f_arr = np.empty(n, dtype=np.int32)
    for i in range(n):
        Fc = jnp.asarray(F_test_c[i]); Ff = jnp.asarray(F_test_f[i])
        t0 = time.perf_counter()
        # Coarse cold-start on the coarse model.
        z_c, _, it_c = solve_c_cold(Fc); z_c.block_until_ready()
        # Decode coarse z to a full-grid field.
        u_coarse = solver_c.decode_full_grid(z_c)
        # Encode that field with the fine model's encoder.
        z_seed = encode_fine(u_coarse)
        # Fine warm-start solve.
        z_f, _, it_f = solve_f_warm(Ff, z_seed); z_f.block_until_ready()
        tot_t[i] = time.perf_counter() - t0
        iters_c_arr[i] = int(it_c); iters_f_arr[i] = int(it_f)
        u_pred = np.asarray(solver_f.decode_full_grid(z_f)) * mask_flat
        rel_l2[i] = float(np.linalg.norm(u_pred - U_test[i]) /
                          (np.linalg.norm(U_test[i]) + 1e-12))

    out = {
        "coarse_ckpt": str(args.coarse_ckpt), "fine_ckpt": str(args.fine_ckpt),
        "coarse_eq": str(args.coarse_eq), "fine_eq": str(args.fine_eq),
        "coarse_iters": args.coarse_iters, "fine_iters": args.fine_iters,
        "fom_time_median": fom_med,
        "rom_relL2_median": float(np.median(rel_l2)),
        "rom_relL2_p10": float(np.quantile(rel_l2, 0.10)),
        "rom_relL2_p25": float(np.quantile(rel_l2, 0.25)),
        "rom_relL2_p75": float(np.quantile(rel_l2, 0.75)),
        "rom_relL2_p90": float(np.quantile(rel_l2, 0.9)),
        "rom_relL2_max": float(rel_l2.max()),
        "frac_le_5e-2": float((rel_l2 < 5e-2).mean()),
        "frac_le_1e-2": float((rel_l2 < 1e-2).mean()),
        "frac_le_5e-3": float((rel_l2 < 5e-3).mean()),
        "rom_time_median": float(np.median(tot_t)),
        "speedup_median": fom_med / float(np.median(tot_t)),
        "iters_coarse_median": float(np.median(iters_c_arr)),
        "iters_fine_median": float(np.median(iters_f_arr)),
        "rel_l2_all": rel_l2.tolist(),
    }
    print(f"\n[comp] median rel-L2 = {out['rom_relL2_median']:.4e}  "
          f"p10 = {out['rom_relL2_p10']:.4e}  p90 = {out['rom_relL2_p90']:.4e}  "
          f"max = {out['rom_relL2_max']:.4e}  "
          f"frac<5e-2 = {out['frac_le_5e-2']:.3f}  "
          f"frac<1e-2 = {out['frac_le_1e-2']:.3f}  "
          f"spd = {out['speedup_median']:.2f}x")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[comp] saved {args.out}")


if __name__ == "__main__":
    main()
