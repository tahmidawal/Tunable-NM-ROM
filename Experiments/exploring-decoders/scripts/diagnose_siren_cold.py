#!/usr/bin/env python
"""Diagnose why cold-start (z=0) NM-ROM fails for SIREN_WIDE on Poisson-2D.

Four diagnostics, all on the existing checkpoint
`checkpoints/poisson2d_siren_wide_cg.pkl` and a single n_eq=1000 EQ set:

  (1) Singular-value spectrum of J(z=0) = dR/dz at z=0 (R evaluated on the
      EQ stencil residual). If sigma_min/sigma_max is tiny, J is rank-
      deficient at z=0 and GN cannot make full progress.

  (2) Residual analysis: ||R(z=0)||, ||proj_{range(J)} R(z=0)||, ratio
      (the fraction of the residual that GN can chase from z=0). If the
      ratio is small, the cold-start residual lives mostly OUTSIDE the
      Jacobian's range and GN converges quickly but uselessly.

  (3) 1D loss landscape: for 3 random unit directions v in R^16, plot
      0.5*||R(t*v)||^2 over t in [-3, 3]. Looking for: flat near 0?
      curvature far from 0? presence of nearby better basins?

  (4) Try 5 z0 initializations on one fixed test parameter (the median
      |freq|-norm test sample). Report final rel-L2 vs CG FOM.

Writes a markdown block to journey-to-good-manifold.md under "Iteration 0".
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
from tunable_rom_decoders.fom.poisson_cg import (
    PoissonFOM,
    source_field,
    analytical_u,
)
from tunable_rom_decoders.solver.nm_rom import INRNMROMSolver


def _build_model(cfg: dict) -> INRAutoencoder:
    return INRAutoencoder(
        decoder_kind=cfg["decoder_kind"],
        N=cfg["N"],
        spatial_dim=cfg["spatial_dim"],
        patch_size=cfg["patch_size"],
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        num_enc_layers=cfg["num_enc_layers"],
        latent_dim=cfg["latent_dim"],
        coord_dim=cfg["spatial_dim"],
        hidden_dim=cfg["siren_hidden_dim"] if cfg["decoder_kind"] == "siren"
                   else cfg["xattn_hidden_dim"],
        siren_num_layers=cfg["siren_num_layers"],
        omega_0=cfg["omega_0"],
        omega=cfg["omega"],
        modulator_hidden=cfg["modulator_hidden"],
        d_attn=cfg["d_attn"],
        num_fourier=cfg["num_fourier"],
        xattn_num_layers=cfg["xattn_num_layers"],
        fourier_scale=cfg["fourier_scale"],
    )


def _residual_jac(solver: INRNMROMSolver, z: jnp.ndarray, F_eq: jnp.ndarray):
    R = solver.residual(z, F_eq)
    J = jax.jacfwd(lambda zz: solver.residual(zz, F_eq))(z)
    return R, J


def diag_jacobian_spectrum(solver, F_eq, latent_dim):
    z0 = jnp.zeros((latent_dim,), dtype=jnp.float64)
    R0, J0 = _residual_jac(solver, z0, F_eq)
    R0 = np.asarray(R0)
    J0 = np.asarray(J0)
    s = np.linalg.svd(J0, compute_uv=False)
    s_min, s_max = float(s.min()), float(s.max())
    cond = s_max / max(s_min, 1e-300)
    return {
        "sigma_max": s_max,
        "sigma_min": s_min,
        "cond_J0": cond,
        "sigma_all": s.tolist(),
        "norm_R0": float(np.linalg.norm(R0)),
        "norm_R_dim": int(R0.size),
    }, R0, J0


def diag_residual_alignment(R0, J0):
    Q, _ = np.linalg.qr(J0)
    Rproj = Q @ (Q.T @ R0)
    norm_R = np.linalg.norm(R0)
    norm_proj = np.linalg.norm(Rproj)
    return {
        "norm_R": float(norm_R),
        "norm_proj_R_into_range_J": float(norm_proj),
        "ratio": float(norm_proj / max(norm_R, 1e-300)),
    }


def diag_landscape(solver, F_eq, latent_dim, n_dirs=3, n_pts=51, seed=0):
    rng = np.random.default_rng(seed)
    ts = np.linspace(-3.0, 3.0, n_pts)
    losses = []
    @jax.jit
    def loss(z):
        R = solver.residual(z, F_eq)
        return 0.5 * jnp.sum(solver.w_eq * R**2)
    for d in range(n_dirs):
        v = rng.normal(size=latent_dim).astype(np.float64)
        v = v / np.linalg.norm(v)
        vj = jnp.asarray(v)
        row = []
        for t in ts:
            l = float(loss(jnp.asarray(t) * vj))
            row.append(l)
        losses.append(row)
    return {"t": ts.tolist(), "losses": losses}


def _u_to_field(solver, z):
    """Decode z to full N x N field for rel-L2 reporting."""
    return solver.decode_full_grid(z)


def diag_z0_alternatives(model, params, solver, fom, F_test, freqs_test, U_test, mask, idx,
                         eq_flat):
    F_full = F_test[idx]
    F_eq = F_full[eq_flat]
    freqs_i = freqs_test[idx].tolist()

    solve_cold = solver.make_solve()
    solve_warm = solver.make_solve_warm()

    # Encoder accessor (single forward)
    @jax.jit
    def enc(u):
        return solver.encode_warm_start(u)

    latent_dim = solver.latent_dim
    rng = np.random.default_rng(1)
    candidates = []

    # 1) zero
    candidates.append(("zero", jnp.zeros((latent_dim,), dtype=jnp.float64)))
    # 2) small random
    for sigma in [0.01, 0.1, 1.0]:
        z = rng.normal(size=latent_dim).astype(np.float64) * sigma
        candidates.append((f"randn_sigma{sigma}", jnp.asarray(z)))
    # 3) encoder(zero field)
    z_zero_field = enc(jnp.zeros((fom.N * fom.N,), dtype=jnp.float32))
    candidates.append(("enc_zero_field", z_zero_field))
    # 4) encoder(F as if it were u)
    z_enc_F = enc(jnp.asarray(F_full, dtype=jnp.float32))
    candidates.append(("enc_F_as_u", z_enc_F))
    # 5) encoder(analytical_u(mu))  — same as warm-start ROM
    u_ana = analytical_u(fom, freqs_i)
    z_warm = enc(u_ana)
    candidates.append(("enc_analytical_u", z_warm))

    F_eq_j = jnp.asarray(F_eq, dtype=jnp.float64)
    mask_flat = np.asarray(mask).reshape(-1)
    u_true = np.asarray(U_test[idx])

    results = []
    for name, z0 in candidates:
        z0 = jnp.asarray(z0, dtype=jnp.float64)
        try:
            z_f, gnorm_f, iters = solve_warm(F_eq_j, z0)
            z_f.block_until_ready()
            u_pred = np.asarray(_u_to_field(solver, z_f)) * mask_flat
            rel = float(np.linalg.norm(u_pred - u_true) /
                        (np.linalg.norm(u_true) + 1e-12))
            results.append({
                "name": name,
                "iters": int(iters),
                "gnorm": float(gnorm_f),
                "rel_l2": rel,
                "z0_norm": float(np.linalg.norm(np.asarray(z0))),
                "z_f_norm": float(np.linalg.norm(np.asarray(z_f))),
            })
        except Exception as e:
            results.append({"name": name, "error": str(e)})
    return {"test_idx": int(idx), "freqs": list(freqs_i), "results": results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path,
                   default=Path("checkpoints/poisson2d_siren_wide_cg.pkl"))
    p.add_argument("--eq", type=Path,
                   default=Path("runs/rom/eq/eq_siren_wide_1000.npz"))
    p.add_argument("--data", type=Path,
                   default=Path("data/poisson2d_cg_n128_s42.npz"))
    p.add_argument("--out", type=Path,
                   default=Path("runs/rom/diagnostic_siren_cold.json"))
    args = p.parse_args()

    print(f"[diag] loading ckpt: {args.ckpt}", flush=True)
    with open(args.ckpt, "rb") as f:
        ck = pickle.load(f)
    cfg = ck["config"]
    params = ck["params"]
    tokens_ref = ck.get("tokens_ref", None)
    print(f"[diag] cfg keys: decoder_kind={cfg['decoder_kind']} "
          f"latent={cfg['latent_dim']} N={cfg['N']}", flush=True)

    model = _build_model(cfg)
    eq = np.load(args.eq)
    eq_flat = eq["eq_flat_indices"]
    eq_w = eq["eq_weights"]
    stencil = eq["stencil_indices"]
    print(f"[diag] EQ: n_eq={eq_flat.size}", flush=True)

    fom = PoissonFOM(N=cfg["N"], spatial_dim=cfg["spatial_dim"])
    solver = INRNMROMSolver(
        autoencoder=model,
        params=params,
        N=cfg["N"], spatial_dim=cfg["spatial_dim"], dx=float(fom.dx),
        eq_flat_indices=eq_flat,
        eq_weights=eq_w,
        stencil_indices=stencil,
        tokens_ref=tokens_ref,
    )

    data = np.load(args.data)
    F_test = data["F_test"]
    freqs_test = data["freqs_test"]
    U_test = data["U_test"]

    # Pick a test sample close to median |k| norm.
    knorm = np.linalg.norm(freqs_test, axis=1)
    idx = int(np.argsort(knorm)[len(knorm) // 2])
    print(f"[diag] using test idx={idx}  freqs={freqs_test[idx].tolist()}", flush=True)

    # (1) Spectrum of J(z=0)
    F_eq_j = jnp.asarray(F_test[idx, eq_flat], dtype=jnp.float64)
    print("[diag] computing J(z=0) spectrum...", flush=True)
    t0 = time.perf_counter()
    spec, R0_np, J0_np = diag_jacobian_spectrum(solver, F_eq_j, solver.latent_dim)
    print(f"[diag]   sigma_max={spec['sigma_max']:.3e}  "
          f"sigma_min={spec['sigma_min']:.3e}  "
          f"cond={spec['cond_J0']:.3e}  "
          f"||R(0)||={spec['norm_R0']:.3e}  "
          f"({time.perf_counter()-t0:.1f}s)", flush=True)

    # (2) Residual / range(J) alignment
    print("[diag] computing residual-range alignment...", flush=True)
    align = diag_residual_alignment(R0_np, J0_np)
    print(f"[diag]   ||R||={align['norm_R']:.3e}  "
          f"||proj R||={align['norm_proj_R_into_range_J']:.3e}  "
          f"ratio={align['ratio']:.4f}", flush=True)

    # (3) 1D loss landscape along 3 random directions
    print("[diag] sampling 1D landscape along 3 directions...", flush=True)
    landscape = diag_landscape(solver, F_eq_j, solver.latent_dim)

    # (4) Alternative z0 initializations
    print("[diag] trying 5 z0 alternatives...", flush=True)
    z0_alts = diag_z0_alternatives(
        model, params, solver, fom, F_test, freqs_test, U_test,
        fom.mask, idx, eq_flat,
    )
    print("[diag] z0 alternatives:")
    for r in z0_alts["results"]:
        if "error" in r:
            print(f"  {r['name']}: ERROR {r['error']}")
        else:
            print(f"  {r['name']:24s}  rel_l2={r['rel_l2']:.4e}  "
                  f"iters={r['iters']}  ||z0||={r['z0_norm']:.3e}  "
                  f"||z_f||={r['z_f_norm']:.3e}")

    out = {
        "ckpt": str(args.ckpt),
        "eq": str(args.eq),
        "test_idx": idx,
        "spectrum": spec,
        "residual_alignment": align,
        "landscape": landscape,
        "z0_alternatives": z0_alts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n[diag] saved JSON to {args.out}", flush=True)


if __name__ == "__main__":
    main()
