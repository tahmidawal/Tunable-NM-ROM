"""Memory-capped NNLS-EQ quadrature fits for the cost-to-tolerance surface.

The reference fits are `poisson2d-rom-objective/followup/fu_eq.eq_fit` and
`burgers2d-rom-latent-stepping/blat_common.fit_eq_weights`.  Both build the
ECSW design matrix G with ONE COLUMN PER CANDIDATE NODE.  The Poisson recipe
already caps its meshfree candidate pool at EQ_CAND_OFF=4096 at every mesh, so
it is mesh-safe as written; the Burgers recipe uses EVERY interior grid node,
which is 260 100 columns at N=512 and 64 516 at N=256 -- G would be 17 GB and
4.2 GB respectively at M=64, and 4x that at M=256.  This cell therefore caps
the candidate pool at CAND_CAP nodes (default 4096, the Poisson recipe's own
pool size) drawn once from a fixed stream, for BOTH pdes, BOTH pools and BOTH
methods, so every (k, N, method) cell is fitted from a pool of the same size
and the surface is internally like-for-like.

At N <= 64 the interior grid has 900 / 3844 nodes, i.e. FEWER than the cap, so
the pool IS the full interior and the fit is identical to the reference.  The
`eq_pool_control` arm re-fits one cell with the uncapped pool to bound what the
cap costs at a finer mesh.

Everything else is unchanged from the references:
  * snapshots are DECODER OUTPUTS (never residual snapshots),
  * the TARGETS b are the EXACT FULL-GRID projections at the mesh being fitted,
  * rows are scaled by their own norm, a fixed subsample of EQ_ROWS rows drives
    the capped Lawson-Hanson pass, and the final nonnegative weights are refit
    on the selected support against ALL rows,
  * the returned diagnostics are the same per-row relative-fit statistics.

`nnls_capped` itself is imported from `pro_common` -- it is not reimplemented.
"""
from __future__ import annotations

import os
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

CAND_CAP = int(os.environ.get("CTOL_CAND_CAP", "4096"))
EQ_SNAPS = int(os.environ.get("CTOL_EQ_SNAPS", "64"))
EQ_PERTURB = int(os.environ.get("CTOL_EQ_PERTURB", "3"))
EQ_ROWS = int(os.environ.get("CTOL_EQ_ROWS", "3072"))
EQ_SEED = int(os.environ.get("CTOL_EQ_SEED", "20259"))


# --------------------------------------------------------------------------
def _solve_nnls(G, b, m, rng, nnls_capped, label, extra, t0):
    """Shared tail of both fits: row scaling -> capped Lawson-Hanson on an
    EQ_ROWS subsample -> support padding -> nonnegative refit on ALL rows."""
    sc = np.linalg.norm(G, axis=1) + 1e-300
    G = G / sc[:, None]
    b = b / sc
    n_c = G.shape[1]
    rows = rng.choice(G.shape[0], size=min(G.shape[0], EQ_ROWS), replace=False)
    wts, rnorm, _ = nnls_capped(G[rows], b[rows], max_support=m)
    supp = np.nonzero(wts > 0)[0]
    if len(supp) >= m:
        keep = supp[np.argsort(-wts[supp])[:m]]
        padded = 0
    else:
        rest = np.setdiff1d(np.arange(n_c), supp)
        score = np.abs(G).mean(0)
        pad = rest[np.argsort(-score[rest])[:m - len(supp)]]
        keep = np.concatenate([supp, pad])
        padded = len(pad)
    Gk = G[:, keep]
    wq, _, _ = nnls_capped(Gk, b, max_support=len(keep))
    wq = np.where(wq > 0, wq, 1e-8 * max(wq.max(), 1e-300))
    res = Gk @ wq - b
    rel_rows = np.abs(res) / (np.abs(b) + 1e-300)
    info = dict(m=int(len(keep)), support=int(len(supp)), padded=int(padded),
                rnorm_capped=float(rnorm), rnorm_final=float(np.linalg.norm(res)),
                b_norm=float(np.linalg.norm(b)),
                rel_fit=float(np.linalg.norm(res) / (np.linalg.norm(b) + 1e-300)),
                row_rel_median=float(np.median(rel_rows)),
                row_rel_p95=float(np.quantile(rel_rows, 0.95)),
                row_rel_max=float(np.max(rel_rows)),
                n_rows_total=int(G.shape[0]), n_rows_fit=int(len(rows)),
                n_cand=int(n_c), cand_cap=CAND_CAP, eq_snaps=EQ_SNAPS,
                eq_perturb=EQ_PERTURB, eq_rows=EQ_ROWS, eq_seed=EQ_SEED,
                secs=time.time() - t0, **extra)
    print(f"  NNLS-EQ {label}: pool {n_c} support {len(supp)} (+{padded} pad) "
          f"rel fit {info['rel_fit']:.2e} (row p95 {info['row_rel_p95']:.1e}, "
          f"max {info['row_rel_max']:.1e}) [{info['secs']:.0f}s]", flush=True)
    return keep, wq, info


def candidate_pool(n_i2, cap=None, seed=None):
    """Indices into the interior-node list: all of them when the interior is no
    larger than the cap, otherwise a fixed random subsample."""
    cap = CAND_CAP if cap is None else cap
    if n_i2 <= cap:
        return np.arange(n_i2)
    r = np.random.default_rng(EQ_SEED if seed is None else seed)
    return np.sort(r.choice(n_i2, size=cap, replace=False))


# --------------------------------------------------------------------------
# POISSON: weak form   Phi^T A u = Lambda Phi^T u
# --------------------------------------------------------------------------
def eq_fit_poisson(u_cand, u_full, Phi_c, Phi_f, Z_snap, K, m, label, nnls_capped):
    """`fu_eq.eq_fit` with the candidate pool supplied by the caller.

    u_cand(z) -> (n_c,) decoder output at the candidate nodes,
    u_full(z) -> (n_i^2,) decoder output on the full interior grid,
    Phi_c (n_c, M'), Phi_f (n_i^2, M') already carry the mesh's quadrature rule
    for the target side (dx^2 for continuum modes, 1 for the discrete sine
    basis).  Snapshots are the decoder at Z_snap plus EQ_PERTURB latent
    perturbations, exactly as in the reference."""
    t0 = time.time()
    r_eq = np.random.default_rng(EQ_SEED)
    n_tr = Z_snap.shape[0]
    idx = r_eq.choice(n_tr, size=min(EQ_SNAPS, n_tr), replace=False)
    # The reference uses an ABSOLUTE 0.05 perturbation, which is ~5% for the
    # coordinate latents (rms ~ 1) but numerically nothing for POD coefficients
    # (rms ~ 1e2).  Scaling it by rms(Z) keeps the two arms symmetric and stays
    # within a few percent of the reference for the coordinate decoder.
    pert = 0.05 * float(np.sqrt(np.mean(np.asarray(Z_snap) ** 2)))
    snaps, fulls = [], []
    for i in idx:
        z = jnp.asarray(Z_snap[i])
        for zz in [z] + [z + pert * jnp.asarray(r_eq.standard_normal(K))
                         for _ in range(EQ_PERTURB)]:
            snaps.append(np.asarray(u_cand(zz)))
            fulls.append(np.asarray(u_full(zz)))
    R = np.stack(snaps)
    Rf = np.stack(fulls)
    b = (Rf @ Phi_f).reshape(-1)
    G = np.einsum("sp,pm->smp", R, Phi_c).reshape(-1, R.shape[1])
    del Rf
    return _solve_nnls(G, b, m, r_eq, nnls_capped, label,
                       dict(kind="weak_poisson", perturb_abs=pert,
                            z_rms=pert / 0.05), t0)


# --------------------------------------------------------------------------
# BURGERS: weak form with the FOM-exact upwind advection
# --------------------------------------------------------------------------
def eq_fit_burgers(u_full, adv_full, Phi_full, cand_pos, Z_snap, K, m, label,
                   nnls_capped):
    """`blat_common.fit_eq_weights(kind='weak', pool='grid')` with the candidate
    columns restricted to `cand_pos` (positions in the interior-node list).

    u_full(z)  -> (n_i^2,) decoder output on the interior grid,
    adv_full(u)-> (n_i^2,) the FOM's sign-upwind advection field N(u)
                  (`blat_common.upwind_adv_field`; the FOM-exact operator must
                  stay inside the weak advection term -- operating rule 7),
    Phi_full   -> (n_i^2, M) unit-2-norm interior sine modes.

    Two row blocks per snapshot (u and N(u)), targets = exact full-grid
    projections, identical to the reference."""
    t0 = time.time()
    r_eq = np.random.default_rng(EQ_SEED)
    n_s = Z_snap.shape[0]
    idx = r_eq.choice(n_s, size=min(EQ_SNAPS, n_s), replace=False)
    Gs, bs = [], []
    Phi_c = Phi_full[cand_pos]                                  # (n_c, M)
    for i in idx:
        z = jnp.asarray(Z_snap[i])
        uf = np.asarray(u_full(z))
        Nf = np.asarray(adv_full(jnp.asarray(uf)))
        for v_f in (uf, Nf):
            bs.append(Phi_full.T @ v_f)                          # (M,)
            Gs.append(Phi_c.T * v_f[cand_pos][None, :])          # (M, n_c)
    G = np.concatenate(Gs, axis=0)
    b = np.concatenate(bs)
    return _solve_nnls(G, b, m, r_eq, nnls_capped, label,
                       dict(kind="weak_burgers"), t0)
