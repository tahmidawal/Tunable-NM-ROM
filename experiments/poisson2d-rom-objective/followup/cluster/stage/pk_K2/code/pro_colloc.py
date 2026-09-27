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
'weak_a{alpha}_M{M}' objectives use the WEAK FORM: Phi^T A u = Lambda Phi^T u, so
only the smooth decoder output is quadratured at the m points (no stencil, no
derivatives); Lambda^{-alpha} Phi^T f is a per-query preprocessing of the input
(M numbers).  With m = full grid it coincides with spec_a{alpha}_M{M}.

Usage:
  PKL=<pkl> [NS=1] [N_TEST=16] [GN_ITERS=60] [OBJECTIVES=fd,spec_a1_M256] [BC_BETA=auto]
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

import pro_common as pc
from pro_common import mp

PKL = os.environ["PKL"]
OUT = sys.argv[1]
NS = int(os.environ.get("NS", "1"))
N_TEST = int(os.environ.get("N_TEST", "16"))
GN_ITERS = int(os.environ.get("GN_ITERS", "60"))
BC_BETA = os.environ.get("BC_BETA", "auto")   # off-grid soft-BC penalty; auto = 1/dx^2 (0 for hard-BC pkls)
OBJECTIVES = os.environ.get("OBJECTIVES", "fd,spec_a1_M256").split(",")
MS = [int(v) for v in os.environ.get("MS", "128,256,512,1024").split(",")]
SCHEMES = os.environ.get("SCHEMES", "uniform,biased,nnls,offgrid").split(",")
INITS = os.environ.get("INITS", "nearest,mean").split(",")
EQ_SNAPS = int(os.environ.get("EQ_SNAPS", "64"))
EQ_PERTURB = int(os.environ.get("EQ_PERTURB", "3"))
EQ_MODES = int(os.environ.get("EQ_MODES", "64"))
EQ_ROWS = int(os.environ.get("EQ_ROWS", "4096"))
# EQ_FIXED_SNAPS=1 draws the EQ snapshot indices, latent perturbations and row subset
# from a FIXED stream inside eq_weights, so every (M, m, pool) in an m/M ladder is fitted
# on the SAME decoder snapshots and the grid/meshfree pools are compared like-for-like.
# Default 0 reproduces the frozen round-1/round-2 runs bit-for-bit (shared rng stream,
# so each cache miss consumed fresh draws).  All follow-up cells set it to 1.
EQ_FIXED_SNAPS = int(os.environ.get("EQ_FIXED_SNAPS", "0"))


def main():
    print(f"jax_backend={jax.default_backend()} x64={jax.config.jax_enable_x64}", flush=True)
    d, cfg, stages_all, Z_tr, HARD_BC = pc.load_pkl(PKL)
    K = cfg["K_LAT"]; N = mp.N; N_TRAIN = mp.N_TRAIN
    assert 1 <= NS <= len(stages_all) and 1 <= N_TEST <= mp.N_VAL
    stages = stages_all[:NS]
    dec = pc.make_decoder(stages, hard_bc=bool(HARD_BC))
    grid = pc.Grid(N)
    n_i2 = grid.n_i ** 2
    bc_beta = 0.0 if HARD_BC else (1.0 / grid.dx ** 2 if BC_BETA == "auto" else float(BC_BETA))
    manifest = dict(pkl=os.path.basename(PKL), pkl_config=cfg, ns=NS, n_test=N_TEST,
                    gn_iters=GN_ITERS, hard_bc=HARD_BC, objectives=OBJECTIVES, ms=MS,
                    schemes=SCHEMES, inits=INITS, eq_snaps=EQ_SNAPS, eq_perturb=EQ_PERTURB,
                    eq_modes=EQ_MODES, eq_rows=EQ_ROWS, eq_fixed_snaps=EQ_FIXED_SNAPS,
                    bc_beta=bc_beta, backend=jax.default_backend())
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

    # ---- NNLS-EQ: candidate pool + snapshots + capped Lawson-Hanson
    N_CAND_OFF = int(os.environ.get("EQ_CAND_OFF", "4096"))
    cand_off = rng.uniform(0.0, 1.0, size=(N_CAND_OFF, 2))     # fixed off-grid candidate pool
    eq_cache = {}
    def eq_weights(m, weak=False, M=None, offgrid=False):
        """NNLS-EQ node set + weights.  Candidates: all interior grid nodes, or
        (offgrid=True) a fixed pool of N_CAND_OFF random points.  Snapshots: FD
        residuals of the decoder at training latents (+ perturbations) for the
        strong-form objectives; the decoder OUTPUTS for the weak form.  Rows:
        projections onto the M lowest test modes (discrete on-grid, continuum
        off-grid); targets = the exact full-grid projections."""
        M = M or EQ_MODES
        key = (m, weak, M, offgrid)
        if key in eq_cache:
            return eq_cache[key]
        t0 = time.time()
        r_eq = np.random.default_rng(mp.SEED + 20259) if EQ_FIXED_SNAPS else rng
        idx = r_eq.choice(N_TRAIN, size=min(EQ_SNAPS, N_TRAIN), replace=False)
        f_int = lambda i: mp.source_interior(N, cx[i], cy[i], w[i], a[i])
        cand = jnp.asarray(cand_off) if offgrid else grid.coords_int
        n_c = cand.shape[0]
        if weak:
            snap_fn = jax.jit(lambda z, f2d: dec(z, cand))
            full_fn = jax.jit(lambda z, f2d: dec(z, grid.coords_int))
        else:
            assert not offgrid, "strong-form NNLS only on grid nodes"
            snap_fn = jax.jit(lambda z, f2d: (grid.op(dec(z, grid.coords_int).reshape(grid.n_i, grid.n_i)) - f2d).reshape(-1))
            full_fn = snap_fn
        snaps, fulls = [], []
        for i in idx:
            f2d = jnp.asarray(f_int(i))
            z = jnp.asarray(Z_tr[i])
            for zz in [z] + [z + 0.05 * jnp.asarray(r_eq.standard_normal(K)) for _ in range(EQ_PERTURB)]:
                snaps.append(np.asarray(snap_fn(zz, f2d)))
                fulls.append(snaps[-1] if not offgrid else np.asarray(full_fn(zz, f2d)))
        R = np.stack(snaps)                                        # (n_snap, n_c)
        Rf = np.stack(fulls)                                       # (n_snap, n_i^2)
        if offgrid:
            spec_c = dict(kind="weak", alpha=0.0, M=M)
            PhiT_c, _ = pc.colloc_mode_table(grid, spec_c, "offgrid", np.asarray(cand))      # (M', n_c)
            PhiT_f, _ = pc.colloc_mode_table(grid, spec_c, "offgrid", np.asarray(grid.coords_int))
            Phi = np.asarray(PhiT_c).T                                                          # (n_c, M')
            b = (Rf @ np.asarray(PhiT_f).T * grid.dx ** 2).reshape(-1)                         # continuum targets by grid rule
        else:
            mask = np.asarray(grid.mode_mask(M)).astype(bool)
            I, Jm = np.nonzero(mask)
            S = np.asarray(grid.S)
            px, py = grid.ix_full - 1, grid.iy_full - 1
            Phi = S[px][:, I] * S[py][:, Jm]                            # (n_c, M')
            b = (Rf @ Phi).reshape(-1)                                   # exact full-grid targets
        G = np.einsum("sp,pm->smp", R, Phi).reshape(-1, n_c)          # (n_snap*M', n_c)
        sc = np.linalg.norm(G, axis=1) + 1e-300
        G, b = G / sc[:, None], b / sc
        rows = r_eq.choice(G.shape[0], size=min(G.shape[0], EQ_ROWS), replace=False)
        wts, rnorm, n_outer = pc.nnls_capped(G[rows], b[rows], max_support=m)
        supp = np.nonzero(wts > 0)[0]
        if len(supp) >= m:
            keep = supp[np.argsort(-wts[supp])[:m]]
            padded = 0
        else:
            rest = np.setdiff1d(np.arange(n_c), supp)
            score = np.abs(R).mean(0)
            pad = rest[np.argsort(-score[rest])[:m - len(supp)]]
            keep = np.concatenate([supp, pad]); padded = len(pad)
            wts = wts.copy(); wts[pad] = np.median(wts[supp]) if len(supp) else 1.0
        # refit nonnegative weights on the FINAL support (all rows), report its residual
        Gk = G[:, keep]
        wq, rnorm_final, _ = pc.nnls_capped(Gk, b, max_support=len(keep))
        if np.any(wq <= 0):        # nodes NNLS zeroed out keep a tiny nominal weight (still m nodes)
            wq = np.where(wq > 0, wq, 1e-8 * max(wq.max(), 1e-300))
        resid_rows = Gk @ wq - b
        rnorm_final = float(np.linalg.norm(resid_rows))
        rel_rows = np.abs(resid_rows) / (np.abs(b) + 1e-300)     # PER-ROW: a global norm hides a bad mode
        rnorm_full = float(np.linalg.norm(G @ np.ones(n_c) - b)) if not offgrid else float("nan")  # == 0 on-grid by construction
        eq_cache[key] = (keep, wq, dict(support=int(len(supp)), padded=int(padded), rnorm_capped=float(rnorm),
                                      rnorm_final=rnorm_final, rnorm_fullgrid=rnorm_full, b_norm=float(np.linalg.norm(b)),
                                      rel_fit=rnorm_final / float(np.linalg.norm(b)),
                                      row_rel_median=float(np.median(rel_rows)),
                                      row_rel_p95=float(np.quantile(rel_rows, 0.95)),
                                      row_rel_max=float(np.max(rel_rows)), fixed_snaps=EQ_FIXED_SNAPS,
                                      n_outer=int(n_outer), secs=time.time() - t0, n_rows=int(len(rows))))
        print(f"  NNLS-EQ m={m} M={M} {'meshfree' if offgrid else 'grid'}: support {len(supp)} (+{padded} padded) "
              f"rows {len(rows)} rnorm capped {rnorm:.2e} final(all rows) {rnorm_final:.2e} / ||b|| "
              f"{np.linalg.norm(b):.2e} (per-row median {np.median(rel_rows):.1e}, p95 "
              f"{np.quantile(rel_rows, 0.95):.1e}, max {np.max(rel_rows):.1e}) [{time.time()-t0:.0f}s]", flush=True)
        return eq_cache[key]

    def biased_nodes(m, i, r):
        """Importance sampling WITH replacement, proposal q = 0.5*uniform + 0.5*Gaussian
        (width 3w) around the source of test case i (the ROM knows f); unbiased
        weights 1/(m q_p) (with-replacement PPS)."""
        g = np.exp(-((Xi[:, 0] - cx[i]) ** 2 + (Xi[:, 1] - cy[i]) ** 2) / (2 * (3 * w[i]) ** 2))
        q = 0.5 / n_i2 + 0.5 * g / g.sum()
        sel = r.choice(n_i2, size=m, replace=True, p=q)
        return sel, 1.0 / (m * q[sel])

    SCHEME_ID = {"uniform": 1, "biased": 2, "nnls": 3, "offgrid": 4, "full": 0, "nnlsoff": 5}
    def case_rng(scheme, m, i):
        """Paired sampling: the SAME node set for a given (scheme, m, test case)
        across every objective and init."""
        return np.random.default_rng(mp.SEED + 7919 * SCHEME_ID[scheme] + 104729 * (m or 0) + i)

    report = dict(manifest=manifest, fom_max_rel_residual=fom_res,
                  oracle={k: float(v.mean()) for k, v in oracle.items()}, rows=[], complete=False)
    def save():
        json.dump(report, open(OUT, "w"), indent=1)

    for oname in OBJECTIVES:
        spec = pc.parse_objective(oname)
        for scheme in SCHEMES:
            for m in ([None] if scheme == "full" else MS):
                pts_kind = "offgrid" if scheme in ("offgrid", "nnlsoff") else "grid"
                if scheme == "nnlsoff" and spec["kind"] != "weak":
                    continue                                    # meshfree EQ needs the weak form
                HgV, V = pc.make_colloc_objective(dec, grid, spec, pts_kind, bc_beta=bc_beta)
                for iname, Z0 in inits.items():
                    per = {k: [] for k in ("err", "err_or", "obj", "acc", "rej", "att", "reason", "bnd")}
                    t0 = time.time()
                    for i in range(N_TEST):
                        gi = N_TRAIN + i
                        r_i = case_rng(scheme, m, i)
                        if pts_kind == "grid":
                            if scheme == "uniform":
                                sel = r_i.choice(n_i2, size=m, replace=False); wq = np.full(m, n_i2 / m)
                            elif scheme == "biased":
                                sel, wq = biased_nodes(m, gi, r_i)
                            elif scheme == "nnls":
                                sel, wq, _ = eq_weights(m, weak=(spec["kind"] == "weak"), M=spec.get("M"))
                            else:
                                sel = np.arange(n_i2); wq = np.ones(n_i2)
                            ix, iy = grid.ix_full[sel], grid.iy_full[sel]
                            pts, keep = grid.stencil(ix, iy)
                            centre_np = np.stack([ix * grid.dx, iy * grid.dx], 1)
                            f_m = pc.source_at(cx[gi], cy[gi], w[gi], a[gi], centre_np[:, 0], centre_np[:, 1])
                            if spec["kind"] == "weak":
                                pts = jnp.asarray(centre_np)      # weak form: centre points only
                        else:
                            if scheme == "nnlsoff":
                                sel, wq, _ = eq_weights(m, weak=True, M=spec.get("M"), offgrid=True)
                                centre_np = cand_off[sel]
                            else:
                                centre_np = r_i.uniform(0.0, 1.0, size=(m, 2))
                                wq = np.full(m, 1.0 / m)
                            pts = jnp.asarray(centre_np)
                            keep = jnp.asarray(pc.boundary_points(max(16, 4 * int(np.sqrt(m))), r_i))
                            f_m = pc.source_at(cx[gi], cy[gi], w[gi], a[gi], centre_np[:, 0], centre_np[:, 1])
                        if spec["kind"] in ("spec", "weak"):
                            PhiT, Wl = pc.colloc_mode_table(grid, spec, pts_kind, centre_np)
                        else:
                            PhiT, Wl = jnp.zeros((1, 1)), jnp.zeros((1,))
                        if spec["kind"] == "weak":
                            f_m = np.asarray(pc.weak_source_term(grid, spec, pts_kind,
                                                                 mp.source_interior(N, cx[gi], cy[gi], w[gi], a[gi])))
                        args = (pts, keep, jnp.asarray(wq), PhiT, Wl, jnp.asarray(f_m))
                        z, val, info = pc.lm_generic(lambda zz: HgV(zz, *args), lambda zz: V(zz, *args),
                                                     jnp.asarray(Z0[i]), GN_ITERS)
                        per["err"].append(float(np.linalg.norm(np.asarray(dec_full(z)) - U_test[i]) / tn[i]))
                        per["err_or"].append(float(oracle[iname][i]))
                        per["obj"].append(float(val))
                        per["acc"].append(info["accepted"]); per["rej"].append(info["rejected"])
                        per["att"].append(info["attempts"]); per["reason"].append(info["reason"])
                        per["bnd"].append(float(jnp.linalg.norm(dec(z, grid.bpts))))
                    e = np.asarray(per["err"])
                    n_modes = int(PhiT.shape[0]) if spec["kind"] in ("spec", "weak") else None
                    row = dict(objective=oname, scheme=scheme, m=m if m else n_i2, init=iname, ns=NS,
                               n_modes_retained=n_modes, bc_beta=(bc_beta if pts_kind == "offgrid" else None),
                               budget=GN_ITERS, rom_rel_l2_mean=float(e.mean()),
                               rom_rel_l2_med=float(np.median(e)), rom_rel_l2_max=float(e.max()),
                               oracle_rel_l2_mean=float(np.mean(per["err_or"])),
                               obj_med=float(np.median(per["obj"])),
                               lm_accepted_med=float(np.median(per["acc"])),
                               lm_rejected_med=float(np.median(per["rej"])),
                               lm_attempts_med=float(np.median(per["att"])),
                               lm_reasons={r: per["reason"].count(r) for r in set(per["reason"])},
                               boundary_block_med=float(np.median(per["bnd"])),
                               eq_info=(eq_cache[(m, spec["kind"] == "weak", spec.get("M") or EQ_MODES, scheme == "nnlsoff")][2]
                                        if scheme in ("nnls", "nnlsoff") else None),
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
