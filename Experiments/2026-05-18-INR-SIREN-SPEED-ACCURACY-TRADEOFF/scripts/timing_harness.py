#!/usr/bin/env python
"""Per-phase timing harness for the SIREN NM-ROM solve.

Breaks the cold-start composite (iter-7 SOTA config) into named sub-phases
and reports median ms over the test set:

  - jit_full          : whole composite (encode + 2 solves + decode)
  - residual_only     : one residual eval on coarse (N_eq=500)
  - jacfwd_only       : one jax.jacfwd(residual) call on coarse
  - residual_fine     : one residual eval on fine (N_eq=8000)
  - jacfwd_fine       : one jax.jacfwd(residual) call on fine
  - lm_step           : H, g, linalg.solve, linesearch (latent_dim small)
  - encoder_warm      : encode_warm_start(u_full) on the fine encoder
  - decode_full_grid  : decode at all N**2 points (only for rel-L2 reporting)
  - coarse_full       : single make_solve(F) call (cold) - upper bound on coarse phase
  - fine_full         : single make_solve_warm(F, z) call - upper bound on fine

Times use perf_counter with block_until_ready to avoid the dispatch lag
masking real GPU work. Each phase is warmed up (JIT compile) before timing.

Default config matches the iter-7 baseline so the resulting numbers are
directly comparable to the 551 ms ROM_time_median.
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

# Enable float64 to match the rest of the codebase.
jax.config.update("jax_enable_x64", True)

from tunable_rom_speed.autoencoder import INRAutoencoder
from tunable_rom_speed.fom.poisson_cg import PoissonFOM, analytical_u
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


def _load_pair(ckpt_path: Path):
    with open(ckpt_path, "rb") as f:
        ck = pickle.load(f)
    return _build_model(ck["config"]), ck["params"], ck.get("tokens_ref", None), ck["config"]


def _median_ms(fn, n_repeat: int) -> float:
    """Time `fn` (which itself does block_until_ready) n_repeat times; return median in ms."""
    ts = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--coarse-ckpt", required=True, type=Path)
    p.add_argument("--fine-ckpt", required=True, type=Path)
    p.add_argument("--coarse-eq", required=True, type=Path)
    p.add_argument("--fine-eq", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--coarse-iters", type=int, default=30)
    p.add_argument("--fine-iters", type=int, default=30)
    p.add_argument("--n-samples", type=int, default=8, help="test samples to time")
    p.add_argument("--n-repeat", type=int, default=5, help="reps per sample for stability")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    model_c, params_c, tokens_c, cfg_c = _load_pair(args.coarse_ckpt)
    model_f, params_f, tokens_f, cfg_f = _load_pair(args.fine_ckpt)
    print(f"[time] coarse {args.coarse_ckpt.name}  fine {args.fine_ckpt.name}", flush=True)

    fom = PoissonFOM(N=cfg_c["N"], spatial_dim=cfg_c["spatial_dim"])
    data = np.load(args.data)
    F_test = data["F_test"][: args.n_samples]
    freqs_test = data["freqs_test"][: args.n_samples]

    eq_c = np.load(args.coarse_eq); eq_f = np.load(args.fine_eq)
    n_eq_c = int(eq_c["eq_flat_indices"].size); n_eq_f = int(eq_f["eq_flat_indices"].size)
    print(f"[time] n_eq coarse={n_eq_c}  n_eq fine={n_eq_f}", flush=True)

    solver_c = INRNMROMSolver(
        autoencoder=model_c, params=params_c,
        N=cfg_c["N"], spatial_dim=cfg_c["spatial_dim"], dx=float(fom.dx),
        eq_flat_indices=eq_c["eq_flat_indices"],
        eq_weights=eq_c["eq_weights"],
        stencil_indices=eq_c["stencil_indices"],
        tokens_ref=tokens_c, gn_max_iters=args.coarse_iters,
        latent_dim_override=cfg_c["latent_dim"],
    )
    solver_f = INRNMROMSolver(
        autoencoder=model_f, params=params_f,
        N=cfg_f["N"], spatial_dim=cfg_f["spatial_dim"], dx=float(fom.dx),
        eq_flat_indices=eq_f["eq_flat_indices"],
        eq_weights=eq_f["eq_weights"],
        stencil_indices=eq_f["stencil_indices"],
        tokens_ref=tokens_f, gn_max_iters=args.fine_iters,
        latent_dim_override=cfg_f["latent_dim"],
    )

    # JIT'd primitives. We rebuild them outside the solver so we can time
    # each individually.
    @jax.jit
    def residual_c(z, F):
        return solver_c.residual(z, F)

    @jax.jit
    def jacfwd_c(z, F):
        return jax.jacfwd(lambda zz: solver_c.residual(zz, F))(z)

    @jax.jit
    def residual_f(z, F):
        return solver_f.residual(z, F)

    @jax.jit
    def jacfwd_f(z, F):
        return jax.jacfwd(lambda zz: solver_f.residual(zz, F))(z)

    @jax.jit
    def encode_fine(u_flat):
        return solver_f.encode_warm_start(u_flat)

    @jax.jit
    def decode_grid_f(z):
        return solver_f.decode_full_grid(z)

    @jax.jit
    def lm_step(J, R, w):
        latent = J.shape[1]
        JtW = J.T * w[None, :]
        H = JtW @ J
        g = JtW @ R
        damp = jnp.maximum(1e-4 * jnp.trace(H) / latent, 1e-8)
        dz = jnp.linalg.solve(H + damp * jnp.eye(latent), -g)
        return dz

    solve_c_cold = solver_c.make_solve()
    solve_f_warm = solver_f.make_solve_warm()

    # FOM CG baseline.
    F0 = jnp.asarray(F_test[0]); x0_init = analytical_u(fom, freqs_test[0].tolist())
    _ = fom.cg_solve(F0, x0=x0_init).block_until_ready()
    fom_ts = []
    for i in range(args.n_samples):
        Fi = jnp.asarray(F_test[i])
        x0_i = analytical_u(fom, freqs_test[i].tolist())
        for _ in range(args.n_repeat):
            t0 = time.perf_counter()
            u = fom.cg_solve(Fi, x0=x0_i); u.block_until_ready()
            fom_ts.append((time.perf_counter() - t0) * 1000.0)
    fom_med = float(np.median(fom_ts))

    # JIT warm-ups: run each phase once on the first test sample.
    Fc0 = jnp.asarray(F_test[0, eq_c["eq_flat_indices"]])
    Ff0 = jnp.asarray(F_test[0, eq_f["eq_flat_indices"]])
    z0 = jnp.zeros((cfg_c["latent_dim"],), dtype=jnp.float64)
    _ = residual_c(z0, Fc0).block_until_ready()
    Jw = jacfwd_c(z0, Fc0); Jw.block_until_ready()
    _ = residual_f(z0, Ff0).block_until_ready()
    Jfw = jacfwd_f(z0, Ff0); Jfw.block_until_ready()
    w_c = jnp.asarray(eq_c["eq_weights"]); w_f = jnp.asarray(eq_f["eq_weights"])
    R_warm_c = residual_c(z0, Fc0)
    dz_warm = lm_step(Jw, R_warm_c, w_c); dz_warm.block_until_ready()
    z_warm, _, _ = solve_c_cold(Fc0); z_warm.block_until_ready()
    u_warm = decode_grid_f(z_warm); u_warm.block_until_ready()
    z_enc = encode_fine(u_warm); z_enc.block_until_ready()
    z_fine, _, _ = solve_f_warm(Ff0, z_enc); z_fine.block_until_ready()

    # Timings.
    times = {f"fom_cg_ms": fom_med}

    # Per-phase: one call each, blocked, n_repeat across samples.
    def time_phase(name, build_fn):
        per_sample = []
        for i in range(args.n_samples):
            fn = build_fn(i)
            per_sample.append(_median_ms(fn, args.n_repeat))
        times[name + "_med_ms"] = float(np.median(per_sample))
        times[name + "_min_ms"] = float(np.min(per_sample))
        times[name + "_max_ms"] = float(np.max(per_sample))
        print(f"[time] {name:24s} med={times[name+'_med_ms']:8.3f} ms  "
              f"min={times[name+'_min_ms']:8.3f}  max={times[name+'_max_ms']:8.3f}",
              flush=True)

    def make_residual_c(i):
        Fc = jnp.asarray(F_test[i, eq_c["eq_flat_indices"]])
        def fn(): residual_c(z0, Fc).block_until_ready()
        return fn

    def make_jacfwd_c(i):
        Fc = jnp.asarray(F_test[i, eq_c["eq_flat_indices"]])
        def fn(): jacfwd_c(z0, Fc).block_until_ready()
        return fn

    def make_residual_f(i):
        Ff = jnp.asarray(F_test[i, eq_f["eq_flat_indices"]])
        def fn(): residual_f(z0, Ff).block_until_ready()
        return fn

    def make_jacfwd_f(i):
        Ff = jnp.asarray(F_test[i, eq_f["eq_flat_indices"]])
        def fn(): jacfwd_f(z0, Ff).block_until_ready()
        return fn

    def make_lm(i):
        Fc = jnp.asarray(F_test[i, eq_c["eq_flat_indices"]])
        R = residual_c(z0, Fc); J = jacfwd_c(z0, Fc)
        def fn(): lm_step(J, R, w_c).block_until_ready()
        return fn

    def make_encode_fine(i):
        Fc = jnp.asarray(F_test[i, eq_c["eq_flat_indices"]])
        zc, _, _ = solve_c_cold(Fc); zc.block_until_ready()
        u_full = decode_grid_f(zc); u_full.block_until_ready()
        def fn(): encode_fine(u_full).block_until_ready()
        return fn

    def make_decode_grid_f(i):
        Fc = jnp.asarray(F_test[i, eq_c["eq_flat_indices"]])
        zc, _, _ = solve_c_cold(Fc); zc.block_until_ready()
        def fn(): decode_grid_f(zc).block_until_ready()
        return fn

    def make_coarse_full(i):
        Fc = jnp.asarray(F_test[i, eq_c["eq_flat_indices"]])
        def fn():
            z, _, _ = solve_c_cold(Fc); z.block_until_ready()
        return fn

    def make_fine_full(i):
        Fc = jnp.asarray(F_test[i, eq_c["eq_flat_indices"]])
        Ff = jnp.asarray(F_test[i, eq_f["eq_flat_indices"]])
        zc, _, _ = solve_c_cold(Fc); zc.block_until_ready()
        u_full = decode_grid_f(zc); u_full.block_until_ready()
        z_seed = encode_fine(u_full); z_seed.block_until_ready()
        def fn():
            z, _, _ = solve_f_warm(Ff, z_seed); z.block_until_ready()
        return fn

    def make_composite_full(i):
        Fc = jnp.asarray(F_test[i, eq_c["eq_flat_indices"]])
        Ff = jnp.asarray(F_test[i, eq_f["eq_flat_indices"]])
        def fn():
            zc, _, _ = solve_c_cold(Fc); zc.block_until_ready()
            u_full = decode_grid_f(zc); u_full.block_until_ready()
            z_seed = encode_fine(u_full); z_seed.block_until_ready()
            zf, _, _ = solve_f_warm(Ff, z_seed); zf.block_until_ready()
        return fn

    time_phase("residual_coarse", make_residual_c)
    time_phase("jacfwd_coarse",   make_jacfwd_c)
    time_phase("residual_fine",   make_residual_f)
    time_phase("jacfwd_fine",     make_jacfwd_f)
    time_phase("lm_step",         make_lm)
    time_phase("encoder_warm",    make_encode_fine)
    time_phase("decode_full_grid",make_decode_grid_f)
    time_phase("coarse_full",     make_coarse_full)
    time_phase("fine_full",       make_fine_full)
    time_phase("composite_full",  make_composite_full)

    times["composite_speedup_vs_fom"] = fom_med / times["composite_full_med_ms"]
    times["n_eq_coarse"] = n_eq_c
    times["n_eq_fine"] = n_eq_f
    times["coarse_max_iters"] = args.coarse_iters
    times["fine_max_iters"] = args.fine_iters
    times["latent_dim_coarse"] = cfg_c["latent_dim"]
    times["latent_dim_fine"] = cfg_f["latent_dim"]
    times["siren_hidden_dim_coarse"] = cfg_c["siren_hidden_dim"]
    times["siren_hidden_dim_fine"] = cfg_f["siren_hidden_dim"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(times, indent=2))
    print(f"\n[time] FOM_CG med = {fom_med:.2f} ms")
    print(f"[time] composite med = {times['composite_full_med_ms']:.2f} ms  "
          f"speedup = {times['composite_speedup_vs_fom']:.2f}x")
    print(f"[time] wrote {args.out}")


if __name__ == "__main__":
    main()
