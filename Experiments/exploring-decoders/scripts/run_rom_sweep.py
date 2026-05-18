#!/usr/bin/env python
"""Run the decoder-agnostic NM-ROM solver over (decoder, n_eq) grid.

For each (decoder ckpt, eq npz) pair: solve ROM on the test set, record
median rel-L2 (vs CG FOM), median GN iters, median wall-clock. Also runs
the FOM CG benchmark once per test parameter and reports median speedup.
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
from tunable_rom_decoders.autoencoder_vdf import INRAutoencoderVDF
from tunable_rom_decoders.fom.poisson_cg import PoissonFOM, source_field, analytical_u
from tunable_rom_decoders.solver.nm_rom import INRNMROMSolver


def _build_model(cfg: dict):
    """Build the model class implied by cfg['decoder_kind']."""
    kind = cfg["decoder_kind"]
    if kind == "siren_vdf":
        return INRAutoencoderVDF(
            N=cfg["N"], spatial_dim=cfg["spatial_dim"],
            patch_size=cfg["patch_size"], embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"], num_enc_layers=cfg["num_enc_layers"],
            latent_dim=cfg["latent_dim"],
            coord_dim=cfg["spatial_dim"],
            hidden_dim=cfg["siren_hidden_dim"],
            siren_num_layers=cfg["siren_num_layers"],
            omega_0=cfg["omega_0"], omega=cfg["omega"],
            modulator_hidden=cfg["modulator_hidden"],
            default_hidden_dim=cfg.get("default_hidden_dim", 128),
            default_num_layers=cfg.get("default_num_layers", 4),
            default_omega_0=cfg.get("default_omega_0", 15.0),
        )
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
        hidden_dim=cfg["siren_hidden_dim"] if cfg["decoder_kind"] == "siren" else cfg["xattn_hidden_dim"],
        siren_num_layers=cfg["siren_num_layers"],
        omega_0=cfg["omega_0"],
        omega=cfg["omega"],
        modulator_hidden=cfg["modulator_hidden"],
        d_attn=cfg["d_attn"],
        num_fourier=cfg["num_fourier"],
        xattn_num_layers=cfg["xattn_num_layers"],
        fourier_scale=cfg["fourier_scale"],
    )


def _benchmark_fom(fom: PoissonFOM, F_test: np.ndarray, freqs_test: np.ndarray):
    """Return (U_fom, fom_times) for the test set."""
    n_test = F_test.shape[0]
    U_fom = np.empty_like(F_test)
    times = np.empty(n_test, dtype=np.float64)

    # Warm up JIT.
    F0 = jnp.asarray(F_test[0])
    x0 = analytical_u(fom, freqs_test[0].tolist())
    _ = fom.cg_solve(F0, x0=x0).block_until_ready()

    for i in range(n_test):
        Fi = jnp.asarray(F_test[i])
        x0 = analytical_u(fom, freqs_test[i].tolist())
        t0 = time.perf_counter()
        u = fom.cg_solve(Fi, x0=x0)
        u.block_until_ready()
        times[i] = time.perf_counter() - t0
        U_fom[i] = np.asarray(u)
    return U_fom, times


def _run_one(decoder_label, ckpt_path: Path, eq_path: Path, data: dict, args, gn_max_iters: int = 12):
    print(f"\n[rom] === {decoder_label}   eq={eq_path.name} ===", flush=True)
    with open(ckpt_path, "rb") as f:
        ck = pickle.load(f)
    cfg = ck["config"]
    params = ck["params"]
    tokens_ref = ck.get("tokens_ref", None)

    model = _build_model(cfg)
    eq = np.load(eq_path)
    eq_flat = eq["eq_flat_indices"]
    eq_w = eq["eq_weights"]
    stencil = eq["stencil_indices"]

    fom = PoissonFOM(N=cfg["N"], spatial_dim=cfg["spatial_dim"])
    encode_method = "encode_mu" if cfg["decoder_kind"] == "siren_vdf" else "encode"
    solver = INRNMROMSolver(
        autoencoder=model,
        params=params,
        N=cfg["N"], spatial_dim=cfg["spatial_dim"], dx=float(fom.dx),
        eq_flat_indices=eq_flat,
        eq_weights=eq_w,
        stencil_indices=stencil,
        tokens_ref=tokens_ref,
        encode_method_name=encode_method,
        latent_dim_override=cfg["latent_dim"],
        gn_max_iters=gn_max_iters,
    )
    solve_fn = solver.make_solve_warm() if args.warm_start else solver.make_solve()

    F_test = data["F_test"]
    freqs_test = data["freqs_test"]
    U_test_cg = data["U_test"]
    n_test = F_test.shape[0]

    # Restrict F_eq to the EQ indices.
    F_test_eq = F_test[:, eq_flat]

    # Precompute warm-start latents from the analytical solution at each
    # parameter. Cost is one encoder forward per test parameter and is
    # NOT counted in the ROM timing (consistent with FOM's analytical
    # warm-start convention).
    if args.warm_start:
        @jax.jit
        def _encode(u):
            return solver.encode_warm_start(u)
        z_inits = np.empty((n_test, solver.latent_dim), dtype=np.float32)
        from tunable_rom_decoders.fom.poisson_cg import analytical_u as _ana
        for i in range(n_test):
            u_ana = _ana(fom, freqs_test[i].tolist())
            zi = _encode(u_ana)
            z_inits[i] = np.asarray(zi)
        # JIT warm-up.
        z0_init = jnp.asarray(z_inits[0])
        F0 = jnp.asarray(F_test_eq[0])
        z0, gn0, it0 = solve_fn(F0, z0_init)
        z0.block_until_ready()
        print(f"[rom]   warm-up (warm-start): iters={int(it0)}  gnorm={float(gn0):.3e}", flush=True)
    else:
        z_inits = None
        F0 = jnp.asarray(F_test_eq[0])
        z0, gn0, it0 = solve_fn(F0)
        z0.block_until_ready()
        print(f"[rom]   warm-up (cold-start): iters={int(it0)}  gnorm={float(gn0):.3e}", flush=True)

    rom_times = np.empty(n_test, dtype=np.float64)
    rom_iters = np.empty(n_test, dtype=np.int32)
    rel_l2 = np.empty(n_test, dtype=np.float64)

    for i in range(n_test):
        Fi = jnp.asarray(F_test_eq[i])
        t0 = time.perf_counter()
        if args.warm_start:
            z_f, gnorm_f, iters = solve_fn(Fi, jnp.asarray(z_inits[i]))
        else:
            z_f, gnorm_f, iters = solve_fn(Fi)
        z_f.block_until_ready()
        rom_times[i] = time.perf_counter() - t0
        rom_iters[i] = int(iters)
        u_pred = solver.decode_full_grid(z_f)
        u_pred = np.asarray(u_pred)
        # Match FOM boundary convention: zero on boundary.
        mask = np.asarray(fom.mask).reshape(-1)
        u_pred = u_pred * mask
        u_true = U_test_cg[i]
        rel_l2[i] = float(np.linalg.norm(u_pred - u_true) / (np.linalg.norm(u_true) + 1e-12))

    return {
        "decoder_label": decoder_label,
        "ckpt": str(ckpt_path),
        "eq_path": str(eq_path),
        "n_eq": int(eq_flat.size),
        "rom_relL2_median": float(np.median(rel_l2)),
        "rom_relL2_p90": float(np.quantile(rel_l2, 0.9)),
        "rom_iters_median": float(np.median(rom_iters)),
        "rom_time_median": float(np.median(rom_times)),
        "rom_time_mean": float(np.mean(rom_times)),
        "rom_relL2_all": rel_l2.tolist(),
        "rom_iters_all": rom_iters.tolist(),
        "rom_times_all": rom_times.tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--ckpts", required=True, nargs="+",
                   help="space-separated DECODER_LABEL=CKPT_PATH entries")
    p.add_argument("--eq-targets", type=int, nargs="+",
                   default=[500, 1000, 2000, 4000, 8000])
    p.add_argument("--eq-dir", type=Path, default=Path("runs/rom/eq"))
    p.add_argument("--eq-tag", default="shared")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--warm-start", action="store_true",
                   help="Use AE-encoded analytical solution as z_init (default cold).")
    p.add_argument("--gn-max-iters", type=int, default=12,
                   help="LM-GN max iterations (default 12).")
    args = p.parse_args()

    data = np.load(args.data)
    first_ckpt = args.ckpts[0].split("=", 1)[1]
    with open(first_ckpt, "rb") as _f:
        cfg_any = pickle.load(_f)["config"]
    fom = PoissonFOM(N=cfg_any["N"], spatial_dim=cfg_any["spatial_dim"])
    print(f"[rom] FOM CG benchmark: {data['F_test'].shape[0]} test parameters")
    U_fom, fom_times = _benchmark_fom(fom, data["F_test"], data["freqs_test"])
    fom_median = float(np.median(fom_times))
    print(f"[rom] FOM CG median wall time = {fom_median*1000:.2f} ms", flush=True)

    fom_vs_cg = np.linalg.norm(U_fom - data["U_test"], axis=1) / (
        np.linalg.norm(data["U_test"], axis=1) + 1e-12
    )
    print(f"[rom] FOM-vs-stored-CG median rel-L2 = {float(np.median(fom_vs_cg)):.3e} "
          f"(sanity; should be near 1e-6)", flush=True)

    all_results = []
    for entry in args.ckpts:
        label, ckpt = entry.split("=", 1)
        ckpt = Path(ckpt)
        for n_eq in args.eq_targets:
            eq_path = args.eq_dir / f"eq_{args.eq_tag}_{n_eq}.npz"
            if not eq_path.exists():
                print(f"[rom]   SKIP {label} n_eq={n_eq}: {eq_path} missing", flush=True)
                continue
            r = _run_one(label, ckpt, eq_path, data, args, gn_max_iters=args.gn_max_iters)
            r["fom_time_median"] = fom_median
            r["speedup_median"] = fom_median / r["rom_time_median"]
            print(
                f"[rom]   {label} n_eq={n_eq}: relL2={r['rom_relL2_median']:.3e}  "
                f"iters={r['rom_iters_median']:.1f}  "
                f"rom_t={r['rom_time_median']*1000:.2f}ms  "
                f"speedup={r['speedup_median']:.1f}x",
                flush=True,
            )
            all_results.append(r)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"fom_time_median": fom_median, "results": all_results}, indent=2,
    ))
    print(f"\n[rom] Saved to {args.out}")


if __name__ == "__main__":
    main()
