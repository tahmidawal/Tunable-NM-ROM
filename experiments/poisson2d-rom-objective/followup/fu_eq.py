"""Shared follow-up machinery: the NNLS-EQ quadrature fit for the weak form and
the jitted LM latent solve.  Split out of `fu_timing.py` so the timing tool and
the complexity-ladder tool (`fu_family.py`) use ONE implementation.

`eq_fit` mirrors the `nnls` / `nnlsoff` schemes of `pro_colloc.py` (decoder-OUTPUT
snapshots, targets = the exact full-grid projections at the grid it is given) with
the snapshot indices, latent perturbations and row subset drawn from a FIXED stream
(EQ_SEED), i.e. `pro_colloc.py`'s EQ_FIXED_SNAPS=1 behaviour.

`make_lm_jit` is the weak-form residual of `pro_common.make_colloc_objective` with
the acceptance / damping / stopping rules of `pro_common.lm_generic`.
"""
from __future__ import annotations

import os
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pro_common as pc  # noqa: E402
from pro_common import mp, F64  # noqa: E402

EQ_SNAPS = int(os.environ.get("EQ_SNAPS", "64"))
EQ_PERTURB = int(os.environ.get("EQ_PERTURB", "3"))
EQ_ROWS = int(os.environ.get("EQ_ROWS", "3072"))
EQ_CAND = int(os.environ.get("EQ_CAND_OFF", "4096"))
EQ_SEED = int(os.environ.get("EQ_SEED", str(mp.SEED + 20259)))


# --------------------------------------------------------------- NNLS-EQ
def eq_fit(dec, grid, Z_tr, K, M, m, pool):
    """NNLS-EQ quadrature weights for the weak form, mirroring the `nnls` /
    `nnlsoff` schemes of pro_colloc.py (decoder-OUTPUT snapshots; targets = the
    exact full-grid projections at THIS grid).  The snapshot indices, latent
    perturbations and row subset come from a FIXED stream (EQ_SEED), so every
    (M, m, pool) in a ladder is fitted on the same snapshots."""
    t0 = time.time()
    r_eq = np.random.default_rng(EQ_SEED)
    n_tr = Z_tr.shape[0]
    idx = r_eq.choice(n_tr, size=min(EQ_SNAPS, n_tr), replace=False)
    if pool == "offgrid":
        cand_np = np.random.default_rng(mp.SEED + 12345).uniform(0.0, 1.0, size=(EQ_CAND, 2))
        spec_c = dict(kind="weak", alpha=0.0, M=M)
        PhiT_c, _ = pc.colloc_mode_table(grid, spec_c, "offgrid", cand_np)
        PhiT_f, _ = pc.colloc_mode_table(grid, spec_c, "offgrid", np.asarray(grid.coords_int))
        Phi = np.asarray(PhiT_c).T                                   # (n_c, M')
        Phi_f = np.asarray(PhiT_f).T * grid.dx ** 2                  # grid rule at THIS grid
        cand = jnp.asarray(cand_np)
    else:
        cand_np = np.asarray(grid.coords_int)
        mask = np.asarray(grid.mode_mask(M)).astype(bool)
        I, Jm = np.nonzero(mask)
        S = np.asarray(grid.S)
        Phi = S[grid.ix_full - 1][:, I] * S[grid.iy_full - 1][:, Jm]  # (n_i^2, M')
        Phi_f = Phi
        cand = grid.coords_int
    snap_fn = jax.jit(lambda z: dec(z, cand))
    full_fn = jax.jit(lambda z: dec(z, grid.coords_int))
    snaps, fulls = [], []
    for i in idx:
        z = jnp.asarray(Z_tr[i])
        for zz in [z] + [z + 0.05 * jnp.asarray(r_eq.standard_normal(K)) for _ in range(EQ_PERTURB)]:
            snaps.append(np.asarray(snap_fn(zz)))
            fulls.append(snaps[-1] if pool == "grid" else np.asarray(full_fn(zz)))
    R = np.stack(snaps); Rf = np.stack(fulls)
    b = (Rf @ Phi_f).reshape(-1)
    n_c = cand_np.shape[0]
    G = np.einsum("sp,pm->smp", R, Phi).reshape(-1, n_c)
    sc = np.linalg.norm(G, axis=1) + 1e-300
    G, b = G / sc[:, None], b / sc
    rows = r_eq.choice(G.shape[0], size=min(G.shape[0], EQ_ROWS), replace=False)
    wts, rnorm, _ = pc.nnls_capped(G[rows], b[rows], max_support=m)
    supp = np.nonzero(wts > 0)[0]
    if len(supp) >= m:
        keep = supp[np.argsort(-wts[supp])[:m]]; padded = 0
    else:
        rest = np.setdiff1d(np.arange(n_c), supp)
        pad = rest[np.argsort(-np.abs(R).mean(0)[rest])[:m - len(supp)]]
        keep = np.concatenate([supp, pad]); padded = len(pad)
    Gk = G[:, keep]
    wq, _, _ = pc.nnls_capped(Gk, b, max_support=len(keep))
    wq = np.where(wq > 0, wq, 1e-8 * max(wq.max(), 1e-300))
    res = Gk @ wq - b
    rel_rows = np.abs(res) / (np.abs(b) + 1e-300)
    info = dict(M=M, m=int(len(keep)), pool=pool, grid_N=grid.N, support=int(len(supp)),
                padded=int(padded), rnorm_capped=float(rnorm), rnorm_final=float(np.linalg.norm(res)),
                b_norm=float(np.linalg.norm(b)),
                rel_fit=float(np.linalg.norm(res) / np.linalg.norm(b)),
                row_rel_median=float(np.median(rel_rows)), row_rel_p95=float(np.quantile(rel_rows, 0.95)),
                row_rel_max=float(np.max(rel_rows)), n_rows=int(len(rows)), n_cand=int(n_c),
                secs=time.time() - t0)
    print(f"  NNLS-EQ {pool} M={M} m={m} @N={grid.N}: support {len(supp)} (+{padded}) "
          f"rel fit {info['rel_fit']:.2e} (row p95 {info['row_rel_p95']:.1e}, max "
          f"{info['row_rel_max']:.1e}) [{info['secs']:.0f}s]", flush=True)
    return cand_np[keep], wq, info


# --------------------------------------------------------------- jitted LM
def make_lm_jit(dec, K, pts, wq, PhiT, Wl, budget, rel_tol=0.0):
    """Jitted LM on r(z) = Wl*(PhiT @ (wq * dec(z, pts))) - f_m: the same
    residual as pro_common.make_colloc_objective's weak core, with the same
    acceptance, damping and stopping rules as pro_common.lm_generic
    (use_rel_dec=True), plus an OPTIONAL invariant absolute stop
    ||r|| <= rel_tol*||f_m|| (rel_tol=0 disables it, matching the reference).
    Reason codes: 0 budget, 1 converged (rel-dec/step), 2 tol, 3 lambda_max,
    5 nan_at_init."""
    pts = jnp.asarray(pts); wq = jnp.asarray(wq); PhiT = jnp.asarray(PhiT); Wl = jnp.asarray(Wl)

    def r_of(z, f_m):
        return Wl * (PhiT @ (wq * dec(z, pts))) - f_m
    rJ = lambda z, f_m: (r_of(z, f_m), jax.jacfwd(r_of)(z, f_m))
    rn_fn = lambda z, f_m: jnp.linalg.norm(r_of(z, f_m))

    def lm(z0, f_m):
        tol = rel_tol * jnp.linalg.norm(f_m)
        r0, J0 = rJ(z0, f_m)
        v0 = jnp.linalg.norm(r0)
        init = (z0, J0, r0, v0, jnp.asarray(1e-6, F64), jnp.int32(0), jnp.int32(0), jnp.int32(1),
                jnp.where(jnp.isfinite(v0), jnp.int32(0), jnp.int32(5)))

        def cond(s):
            return (s[8] == 0) & (s[5] < budget)

        def body(s):
            z, J, r, val, lam, att, acc, nJ, _ = s
            H = J.T @ J; g = J.T @ r
            D = jnp.diag(jnp.diag(H)) + 1e-30 * jnp.eye(K, dtype=F64)
            dz = jnp.linalg.solve(H + lam * D, -g)
            finite = jnp.all(jnp.isfinite(dz))
            z_new = z + jnp.where(finite, dz, 0.0)
            v_new = jnp.where(finite, rn_fn(z_new, f_m), jnp.inf)
            accept = finite & jnp.isfinite(v_new) & (v_new < val)
            rel_dec = jnp.where(accept, (val - v_new) / (jnp.abs(val) + 1e-300), 1.0)
            step = jnp.linalg.norm(dz) / (1.0 + jnp.linalg.norm(z))
            r2, J2 = jax.lax.cond(accept, lambda: rJ(z_new, f_m), lambda: (r, J))
            z = jnp.where(accept, z_new, z); val = jnp.where(accept, v_new, val)
            lam = jnp.where(accept, jnp.maximum(lam / 3.0, 1e-12), jnp.minimum(lam * 10.0, 1e12))
            acc = acc + accept.astype(jnp.int32); nJ = nJ + accept.astype(jnp.int32)
            reason = jnp.where(accept & (val <= tol) & (tol > 0), jnp.int32(2),
                       jnp.where(accept & ((rel_dec < 1e-12) | (step < 1e-13)), jnp.int32(1),
                        jnp.where((~accept) & (lam >= 1e12), jnp.int32(3), jnp.int32(0))))
            return (z, J2, r2, val, lam, att + 1, acc, nJ, reason)

        z, J, r, val, lam, att, acc, nJ, reason = jax.lax.while_loop(cond, body, init)
        return z, val, nJ, acc, att, reason

    return jax.jit(lm)


