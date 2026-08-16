"""Cell 2: COLLOCATION study for chosen objectives — full grid vs m-node subsets
(uniform random / source-biased importance sampling / NNLS-EQ on grid nodes)
and OFF-GRID random points (strong-form residual via decoder autodiff; only
possible because the decoder is meshfree).

Quadrature convention: every objective is written as an integral / grid-sum
approximated by sum_p wq_p (.)_p, so subsets and the full grid are directly
comparable.  Grid nodes: wq = n_i^2/m for uniform, 1/(m q_p) for importance
sampling with proposal q; NNLS-EQ: nonnegative weights fitted so that the M
test-mode projections of TRAINING residual snapshots (decoder at training
latents, +/- small latent perturbations) are reproduced by the subset.
Off-grid: wq = 1/m (unit area), continuous sine test modes.

Usage:
  PKL=<pkl> [NS=1] [N_TEST=16] [GN_ITERS=60] [OBJECTIVES=fd,spec_a1_M256]
  [MS=128,256,512,1024] [SCHEMES=uniform,biased,nnls,offgrid] [INITS=nearest,mean]
  [EQ_SNAPS=64] [EQ_PERTURB=3] python pro_colloc.py <out.json>
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import nnls

import pro_common as pc
from pro_common import mp

PKL = os.environ["PKL"]
OUT = sys.argv[1]
NS = int(os.environ.get("NS", "1"))
N_TEST = int(os.environ.get("N_TEST", "16"))
GN_ITERS = int(os.environ.get("GN_ITERS", "60"))
HARD_BC = int(os.environ.get("HARD_BC", "0"))
OBJECTIVES = os.environ.get("OBJECTIVES", "fd,spec_a1_M256").split(",")
MS = [int(v) for v in os.environ.get("MS", "128,256,512,1024").split(",")]
SCHEMES = os.environ.get("SCHEMES", "uniform,biased,nnls,offgrid").split(",")
INITS = os.environ.get("INITS", "nearest,mean").split(",")
EQ_SNAPS = int(os.environ.get("EQ_SNAPS", "64"))
EQ_PERTURB = int(os.environ.get("EQ_PERTURB", "3"))
EQ_MODES = int(os.environ.get("EQ_MODES", "64"))


def main():
    print(f"jax_backend={jax.default_backend()} x64={jax.config.jax_enable_x64}", flush=True)
    d, cfg, stages_all, Z_tr = pc.load_pkl(PKL)
    K = cfg["K_LAT"]; N = mp.N; N_TRAIN = mp.N_TRAIN
    stages = stages_all[:NS]
    dec = pc.make_decoder(stages, hard_bc=bool(HARD_BC))
    grid = pc.Grid(N)
    n_i2 = grid.n_i ** 2
    manifest = dict(pkl=os.path.basename(PKL), pkl_config=cfg, ns=NS, n_test=N_TEST,
                    gn_iters=GN_ITERS, hard_bc=HARD_BC, objectives=OBJECTIVES, ms=MS,
                    schemes=SCHEMES, inits=INITS, eq_snaps=EQ_SNAPS, eq_perturb=EQ_PERTURB,
                    eq_modes=EQ_MODES, backend=jax.default_backend())
    print("MANIFEST " + json.dumps(manifest), flush=True)

    U, z_true_all, coords, fom_res = mp.build_snapshots(N)
    U_va = U[N_TRAIN:]
    zt = np.asarray(z_true_all)
    nn_idx = np.argmin(((zt[N_TRAIN:, None, :] - zt[None, :N_TRAIN, :]) ** 2).sum(-1), axis=1)
    cx, cy, w, a, _ = mp.sample_params()
    U_test = np.asarray(U_va[:N_TEST]); tn = np.linalg.norm(U_test, axis=1)
    z_mean = Z_tr.mean(0)
    inits = {"mean": np.tile(z_mean, (N_TEST, 1)), "nearest": Z_tr[nn_idx][:N_TEST]}
    inits = {k: v for k, v in inits.items() if k in INITS}
    dec_full = jax.jit(lambda z: dec(z, coords))

    # oracle (data misfit, same budget)
    rJ = jax.jit(lambda z, u: (dec(z, coords) - u, jax.jacfwd(lambda zz: dec(zz, coords) - u)(z)))
    rn = jax.jit(lambda z, u: jnp.linalg.norm(dec(z, coords) - u))
    oracle = {}
    for name, Z0 in inits.items():
        rels = []
        for i in range(N_TEST):
            u = jnp.asarray(U_test[i])
            _, r, _ = pc.lm_solve(lambda zz: rJ(zz, u), lambda zz: rn(zz, u), jnp.asarray(Z0[i]), GN_ITERS)
            rels.append(r / tn[i])
        oracle[name] = np.asarray(rels)
    print("ORACLE " + "  ".join(f"{k}={v.mean():.3e}" for k, v in oracle.items()), flush=True)

    rng = np.random.default_rng(mp.SEED + 12345)
    Xi = np.asarray(grid.coords_int)                    # (n_i^2, 2)

    # ---- NNLS-EQ training snapshots: residual of the decoder at training latents (+ perturbations)
    eq_cache = {}
    def eq_weights(m):
        if m in eq_cache:
            return eq_cache[m]
        t0 = time.time()
        idx = rng.choice(N_TRAIN, size=min(EQ_SNAPS, N_TRAIN), replace=False)
        f_int = lambda i: mp.source_interior(N, cx[i], cy[i], w[i], a[i])
        res_fn = jax.jit(lambda z, f2d: (grid.op(dec(z, grid.coords_int).reshape(grid.n_i, grid.n_i)) - f2d).reshape(-1))
        snaps = []
        for i in idx:
            f2d = jnp.asarray(f_int(i))
            z = jnp.asarray(Z_tr[i])
            snaps.append(np.asarray(res_fn(z, f2d)))
            for _ in range(EQ_PERTURB):
                zp = z + 0.05 * jnp.asarray(rng.standard_normal(K))
                snaps.append(np.asarray(res_fn(zp, f2d)))
        R = np.stack(snaps)                                        # (n_snap, n_i^2)
        # test modes: EQ_MODES lowest discrete eigenmodes -> rows = mode-projections
        mask = np.asarray(grid.mode_mask(EQ_MODES)).astype(bool)
        I, Jm = np.nonzero(mask)
        S = np.asarray(grid.S)
        px, py = grid.ix_full - 1, grid.iy_full - 1
        Phi = S[px][:, I] * S[py][:, Jm]                            # (n_i^2, M)
        G = np.einsum("sp,pm->smp", R, Phi).reshape(-1, n_i2)       # (n_snap*M, n_i^2)
        b = G.sum(1)                                                 # full-grid targets
        # normalize rows
        sc = np.linalg.norm(G, axis=1) + 1e-300
        G, b = G / sc[:, None], b / sc
        # NNLS via Lawson-Hanson on a random row subset sized to give support ~m,
        # then pad/truncate to exactly m by weight magnitude (rebuttal protocol).
        rows = rng.choice(G.shape[0], size=min(G.shape[0], max(m, 8)), replace=False)
        wts, rnorm = nnls(G[rows], b[rows], maxiter=50 * n_i2)
        supp = np.nonzero(wts > 0)[0]
        if len(supp) >= m:
            keep = supp[np.argsort(-wts[supp])[:m]]
        else:
            rest = np.setdiff1d(np.arange(n_i2), supp)
            # pad with nodes of largest mean |residual| (top-ranked remaining)
            score = np.abs(R).mean(0)
            pad = rest[np.argsort(-score[rest])[:m - len(supp)]]
            keep = np.concatenate([supp, pad])
            wts = wts.copy(); wts[pad] = n_i2 / m * 0.5   # padded nodes get a nominal weight
        wq = wts[keep]
        # rescale so total weight equals the grid count (exact for constants)
        wq = wq * (n_i2 / max(wq.sum(), 1e-300))
        eq_cache[m] = (keep, wq, dict(support=int(len(supp)), rnorm=float(rnorm),
                                      secs=time.time() - t0, n_rows=int(len(rows))))
        print(f"  NNLS-EQ m={m}: support {len(supp)} rows {len(rows)} rnorm {rnorm:.2e} [{time.time()-t0:.0f}s]", flush=True)
        return eq_cache[m]

    def biased_nodes(m, i):
        """Importance sampling ∝ 0.5*uniform + 0.5*Gaussian(width 3w) around the source
        of test case i (the ROM knows f).  Weights 1/(m q_p) with q normalized on the grid."""
        g = np.exp(-((Xi[:, 0] - cx[i]) ** 2 + (Xi[:, 1] - cy[i]) ** 2) / (2 * (3 * w[i]) ** 2))
        q = 0.5 / n_i2 + 0.5 * g / g.sum()
        sel = rng.choice(n_i2, size=m, replace=False, p=q)
        return sel, 1.0 / (m * q[sel])

    report = dict(manifest=manifest, fom_max_rel_residual=fom_res,
                  oracle={k: float(v.mean()) for k, v in oracle.items()}, rows=[], complete=False)
    def save():
        json.dump(report, open(OUT, "w"), indent=1)

    for oname in OBJECTIVES:
        spec = pc.parse_objective(oname)
        for scheme in SCHEMES:
            for m in ([None] if scheme == "full" else MS):
                for iname, Z0 in inits.items():
                    per = {k: [] for k in ("err", "err_or", "obj", "acc", "rej", "att", "reason", "bnd")}
                    t0 = time.time()
                    for i in range(N_TEST):
                        gi = N_TRAIN + i
                        if scheme in ("uniform", "biased", "nnls", "full"):
                            if scheme == "uniform":
                                sel = rng.choice(n_i2, size=m, replace=False); wq = np.full(m, n_i2 / m)
                            elif scheme == "biased":
                                sel, wq = biased_nodes(m, gi)
                            elif scheme == "nnls":
                                sel, wq, _ = eq_weights(m)
                            else:
                                sel = np.arange(n_i2); wq = np.ones(n_i2)
                            ix, iy = grid.ix_full[sel], grid.iy_full[sel]
                            pts, keep = grid.stencil(ix, iy)
                            HgV, V, centre = pc.make_colloc_objective(dec, grid, spec, "grid", pts,
                                                                       jnp.asarray(wq), keep=keep)
                            f_m = jnp.asarray(pc.source_at(cx[gi], cy[gi], w[gi], a[gi],
                                                           ix * grid.dx, iy * grid.dx))
                        else:                                   # offgrid
                            P = jnp.asarray(rng.uniform(0.0, 1.0, size=(m, 2)))
                            wq = np.full(m, 1.0 / m)
                            HgV, V, centre = pc.make_colloc_objective(dec, grid, spec, "offgrid", P,
                                                                       jnp.asarray(wq))
                            f_m = jnp.asarray(pc.source_at(cx[gi], cy[gi], w[gi], a[gi],
                                                           np.asarray(P[:, 0]), np.asarray(P[:, 1])))
                        z, val, info = pc.lm_generic(lambda zz: HgV(zz, f_m), lambda zz: V(zz, f_m),
                                                     jnp.asarray(Z0[i]), GN_ITERS)
                        per["err"].append(float(np.linalg.norm(np.asarray(dec_full(z)) - U_test[i]) / tn[i]))
                        per["err_or"].append(float(oracle[iname][i]))
                        per["obj"].append(float(val))
                        per["acc"].append(info["accepted"]); per["rej"].append(info["rejected"])
                        per["att"].append(info["attempts"]); per["reason"].append(info["reason"])
                        per["bnd"].append(float(jnp.linalg.norm(dec(z, grid.bpts))))
                    e = np.asarray(per["err"])
                    row = dict(objective=oname, scheme=scheme, m=m if m else n_i2, init=iname, ns=NS,
                               budget=GN_ITERS, rom_rel_l2_mean=float(e.mean()),
                               rom_rel_l2_med=float(np.median(e)), rom_rel_l2_max=float(e.max()),
                               oracle_rel_l2_mean=float(np.mean(per["err_or"])),
                               obj_med=float(np.median(per["obj"])),
                               lm_accepted_med=float(np.median(per["acc"])),
                               lm_rejected_med=float(np.median(per["rej"])),
                               lm_attempts_med=float(np.median(per["att"])),
                               lm_reasons={r: per["reason"].count(r) for r in set(per["reason"])},
                               boundary_block_med=float(np.median(per["bnd"])),
                               eq_info=(eq_cache[m][2] if scheme == "nnls" else None),
                               per_sample_rom_rel_l2=[float(v) for v in e], secs=time.time() - t0)
                    report["rows"].append(row)
                    print(f"RESULT obj={oname:14s} {scheme:8s} m={row['m']:5d} init={iname:8s} "
                          f"ROM {row['rom_rel_l2_mean']:.3e} (med {row['rom_rel_l2_med']:.3e} max "
                          f"{row['rom_rel_l2_max']:.3e}) oracle {row['oracle_rel_l2_mean']:.3e} "
                          f"acc/rej {row['lm_accepted_med']:.0f}/{row['lm_rejected_med']:.0f} "
                          f"{row['lm_reasons']} [{row['secs']:.0f}s]", flush=True)
                    save()
    report["complete"] = True
    save()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
