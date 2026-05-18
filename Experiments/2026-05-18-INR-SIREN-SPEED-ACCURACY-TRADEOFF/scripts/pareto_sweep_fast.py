#!/usr/bin/env python
"""Same as pareto_sweep.py but uses FastINRNMROMSolver everywhere.

Cold-start composite. One row per (coarse_neq, fine_neq, coarse_iters, fine_iters).
"""
from __future__ import annotations

import argparse
import csv
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


def make_solver(model, params, tokens, cfg, fom, eq, max_iters):
    return FastINRNMROMSolver(
        autoencoder=model, params=params,
        N=cfg["N"], spatial_dim=cfg["spatial_dim"], dx=float(fom.dx),
        eq_flat_indices=eq["eq_flat_indices"],
        eq_weights=eq["eq_weights"],
        stencil_indices=eq["stencil_indices"],
        tokens_ref=tokens, gn_max_iters=max_iters,
        latent_dim_override=cfg["latent_dim"],
    )


def eval_cell(model_c, params_c, tokens_c, cfg_c,
              model_f, params_f, tokens_f, cfg_f,
              eq_c, eq_f, F_test, U_test, mask_flat, fom,
              it_c, it_f):
    sc = make_solver(model_c, params_c, tokens_c, cfg_c, fom, eq_c, it_c)
    sf = make_solver(model_f, params_f, tokens_f, cfg_f, fom, eq_f, it_f)
    solve_c = sc.make_solve(); solve_f = sf.make_solve_warm()

    @jax.jit
    def encode_fine(u_flat):
        return sf.encode_warm_start(u_flat)

    Fc0 = jnp.asarray(F_test[0, eq_c["eq_flat_indices"]])
    Ff0 = jnp.asarray(F_test[0, eq_f["eq_flat_indices"]])
    zc0, _, _ = solve_c(Fc0); zc0.block_until_ready()
    u0 = sc.decode_full_grid(zc0); u0.block_until_ready()
    zs0 = encode_fine(u0); zs0.block_until_ready()
    zf0, _, _ = solve_f(Ff0, zs0); zf0.block_until_ready()

    F_test_c = F_test[:, eq_c["eq_flat_indices"]]
    F_test_f = F_test[:, eq_f["eq_flat_indices"]]
    n = F_test.shape[0]
    rel_l2 = np.empty(n); tot_t = np.empty(n)
    it_c_arr = np.empty(n, dtype=np.int32); it_f_arr = np.empty(n, dtype=np.int32)
    for i in range(n):
        Fc = jnp.asarray(F_test_c[i]); Ff = jnp.asarray(F_test_f[i])
        t0 = time.perf_counter()
        zc, _, itc = solve_c(Fc); zc.block_until_ready()
        u_c = sc.decode_full_grid(zc)
        z_seed = encode_fine(u_c)
        zf, _, itf = solve_f(Ff, z_seed); zf.block_until_ready()
        tot_t[i] = time.perf_counter() - t0
        it_c_arr[i] = int(itc); it_f_arr[i] = int(itf)
        u_pred = np.asarray(sf.decode_full_grid(zf)) * mask_flat
        rel_l2[i] = float(np.linalg.norm(u_pred - U_test[i]) /
                          (np.linalg.norm(U_test[i]) + 1e-12))
    return {
        "rom_relL2_median": float(np.median(rel_l2)),
        "rom_relL2_p10": float(np.quantile(rel_l2, 0.10)),
        "rom_relL2_p90": float(np.quantile(rel_l2, 0.90)),
        "rom_relL2_max": float(rel_l2.max()),
        "rom_time_median": float(np.median(tot_t)),
        "frac_le_1e-2": float((rel_l2 < 1e-2).mean()),
        "frac_le_5e-3": float((rel_l2 < 5e-3).mean()),
        "iters_coarse_median": float(np.median(it_c_arr)),
        "iters_fine_median": float(np.median(it_f_arr)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--coarse-ckpt", required=True, type=Path)
    p.add_argument("--fine-ckpt", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--eq-dir", required=True, type=Path)
    p.add_argument("--eq-extra-dir", type=Path, default=None)
    p.add_argument("--coarse-neq", nargs="+", type=int, default=[500])
    p.add_argument("--fine-neq", nargs="+", type=int, default=[500, 2000, 4000, 8000])
    p.add_argument("--coarse-iters", nargs="+", type=int, default=[20, 30])
    p.add_argument("--fine-iters", nargs="+", type=int, default=[3, 5, 10])
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--out-csv", required=True, type=Path)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def load_eq(n_eq):
        path = args.eq_dir / f"eq_siren_wide_{n_eq}.npz"
        if not path.exists() and args.eq_extra_dir is not None:
            path = args.eq_extra_dir / f"eq_siren_wide_{n_eq}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        return np.load(path)

    mc, pc, tc, cc = _load(args.coarse_ckpt)
    mf, pf, tf, cf = _load(args.fine_ckpt)
    fom = PoissonFOM(N=cc["N"], spatial_dim=cc["spatial_dim"])
    data = np.load(args.data)
    F_test = data["F_test"]; freqs_test = data["freqs_test"]; U_test = data["U_test"]
    mask_flat = np.asarray(fom.mask).reshape(-1)

    F0 = jnp.asarray(F_test[0]); x0 = analytical_u(fom, freqs_test[0].tolist())
    _ = fom.cg_solve(F0, x0=x0).block_until_ready()
    n = F_test.shape[0]; fom_t = np.empty(n)
    for i in range(n):
        Fi = jnp.asarray(F_test[i]); xi = analytical_u(fom, freqs_test[i].tolist())
        t0 = time.perf_counter()
        u = fom.cg_solve(Fi, x0=xi); u.block_until_ready()
        fom_t[i] = time.perf_counter() - t0
    fom_med = float(np.median(fom_t))
    print(f"[fast] FOM med = {fom_med*1000:.2f} ms", flush=True)

    cols = ["coarse_neq","fine_neq","coarse_iters","fine_iters",
            "rom_time_ms","fom_time_ms","speedup_vs_fom",
            "rel_l2_median","rel_l2_p90","rel_l2_max",
            "frac_le_1e-2","frac_le_5e-3","iters_c_med","iters_f_med"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for ne_c in args.coarse_neq:
            try: eq_c = load_eq(ne_c)
            except FileNotFoundError: print(f"skip coarse {ne_c}"); continue
            for ne_f in args.fine_neq:
                try: eq_f = load_eq(ne_f)
                except FileNotFoundError: print(f"skip fine {ne_f}"); continue
                for it_c in args.coarse_iters:
                    for it_f in args.fine_iters:
                        label = f"fast_c{ne_c}_f{ne_f}_ic{it_c}_if{it_f}"
                        print(f"[fast] >> {label} ...", flush=True)
                        try:
                            r = eval_cell(mc,pc,tc,cc, mf,pf,tf,cf,
                                          eq_c, eq_f, F_test, U_test,
                                          mask_flat, fom, it_c, it_f)
                        except Exception as e:
                            print(f"[fast] FAIL {label}: {e}", flush=True); continue
                        speedup = fom_med / r["rom_time_median"]
                        row = [ne_c, ne_f, it_c, it_f,
                               f"{r['rom_time_median']*1000:.2f}",
                               f"{fom_med*1000:.2f}",
                               f"{speedup:.3f}",
                               f"{r['rom_relL2_median']:.3e}",
                               f"{r['rom_relL2_p90']:.3e}",
                               f"{r['rom_relL2_max']:.3e}",
                               f"{r['frac_le_1e-2']:.3f}",
                               f"{r['frac_le_5e-3']:.3f}",
                               r["iters_coarse_median"], r["iters_fine_median"]]
                        w.writerow(row); f.flush()
                        (args.out_dir / f"{label}.json").write_text(json.dumps(
                            {**r, "speedup_vs_fom": speedup,
                             "fom_time_median": fom_med,
                             "coarse_neq": ne_c, "fine_neq": ne_f,
                             "coarse_iters": it_c, "fine_iters": it_f}, indent=2))
                        print(f"[fast] << {label}: rel-L2 {r['rom_relL2_median']:.3e}  "
                              f"speed {speedup:.2f}x", flush=True)
    print(f"[fast] wrote {args.out_csv}")


if __name__ == "__main__":
    main()
