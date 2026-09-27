"""Shared numerics for the p-linear cell: basis extension, test-count rules, oracles.

Everything here is an offline construction or an untimed diagnostic. The timed
query paths are the unchanged parent machinery: `correction_core` (the ladder),
`poisson_ablation` (the head-only and POD-LSPG arms), `core.fom_query` (DST) and
`iterative_core` (CG).

Staged flat: every module sits beside this file on the cluster.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import core as C
import sep_common as sc
import arms as A
import pbh_core as K_


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def test_count(rule, unknowns, fixed):
    """Requested sine-test count for a reduced model with `unknowns` degrees of freedom.

    `m4` is the cheap-corrections rule (M = 4 x unknowns); `m256` is the parent's fixed
    request of 256 modes (257 retained). Identical to `poisson_ladder.test_count`, where
    unknowns = K + q.
    """
    if rule == 'm4':
        return 4 * int(unknowns)
    if rule == 'm256':
        return int(fixed)
    raise ValueError(rule)


def reassemble(base, modes):
    """`core.assemble` for a different test count, reusing the built bank.

    Copied from `poisson_ladder.reassemble` so the `m4` arms are constructed exactly as
    `ccpoi01` constructed them: only the sine table, the retained mode set and the reduced
    operator depend on the test count.
    """
    t0 = time.perf_counter()
    n = base['intervals']
    lam = C.eigenvalues(n)
    I, J = np.nonzero(lam <= np.sort(lam.ravel())[modes - 1])
    maxmode = int(max(I.max(), J.max())) + 1
    p = np.arange(1, n)
    S = np.sqrt(2.0 / n) * np.sin(np.pi * np.outer(p, np.arange(1, maxmode + 1)) / n)
    Sj, Ij, Jj = jnp.asarray(S), jnp.asarray(I), jnp.asarray(J)
    W = jnp.asarray(lam[I, J] ** -1)
    bank = base['bank']
    cubes = bank.reshape((n - 1, n - 1, bank.shape[-1]))
    B = jnp.einsum('xa,xyr,yb->abr', Sj, cubes, Sj)[Ij, Jj]
    B.block_until_ready()
    info = {**base['info'], 'requested_modes': int(modes), 'retained_modes': int(len(I)),
            'operator_sha256': C.sha(np.asarray(B)),
            'reassembly_seconds': time.perf_counter() - t0}
    return {**base, 'B': B, 'S': Sj, 'I': Ij, 'J': Jj, 'W': W, 'info': info}


def extend_basis(params, codes, basis, train_draws, intervals, subset=None):
    """Nested correction directions for every q up to R.

    The retained basis (32 columns) is used VERBATIM, so every rung with q <= 32 is the
    parent's arm bit for bit. Columns 33..R extend it by the SAME rule the retained basis
    was built with -- right singular vectors of the normalised training residual at the
    stored training codes, in the bank's exact QR field metric at the training mesh --
    applied to the component of that residual orthogonal to the retained 32.

    Returns the full R x R coefficient-direction matrix and a diagnostic record.
    """
    t0 = time.perf_counter()
    R = int(np.asarray(params['h_lin']).shape[1])
    C32 = np.asarray(basis['coefficient_directions'])
    kept = int(C32.shape[1])
    assert len(train_draws) == len(codes), (len(train_draws), len(codes))
    train_draws, codes = np.asarray(train_draws), np.asarray(codes)
    if subset is not None and subset < len(train_draws):
        train_draws, codes = train_draws[:subset], codes[:subset]
    phases = {}
    t1 = time.perf_counter()
    G = K_.bank_of(params, intervals)
    Rg, rank = K_.bank_r(G)
    assert rank['rank_valid'], rank
    phases['bank_and_qr'] = time.perf_counter() - t1
    t1 = time.perf_counter()
    U = K_.fields(train_draws, intervals)
    phases['fields'] = time.perf_counter() - t1
    t1 = time.perf_counter()
    T, perp2, nu2 = K_.project_targets(G, Rg, U)
    coeff = jax.jit(jax.vmap(lambda z: sc.head(params, z)))(jnp.asarray(codes))
    Rg_np = np.asarray(Rg)
    E = (np.asarray(T) - np.asarray(coeff) @ Rg_np.T) / np.sqrt(np.asarray(nu2))[:, None]
    phases['targets'] = time.perf_counter() - t1
    t1 = time.perf_counter()
    # the retained directions in THIS run's metric (they were built in the same metric on
    # another machine; the difference is round-off and is recorded, not assumed)
    V32 = Rg_np @ C32
    gram_dev = float(np.linalg.norm(V32.T @ V32 - np.eye(kept)))
    # rebuilt top-`kept` subspace against the retained one: principal-angle cosines
    _, s_full, Vt_full = np.linalg.svd(E, full_matrices=False)
    cos = np.linalg.svd(Vt_full[:kept] @ V32, compute_uv=False)
    prefix_subspace_defect = float(np.sqrt(max(0.0, 1.0 - float(np.min(cos)) ** 2)))
    Eperp = E - (E @ V32) @ V32.T
    _, s_ext, Vt_ext = np.linalg.svd(Eperp, full_matrices=False)
    need = R - kept
    assert Vt_ext.shape[0] >= need, (Vt_ext.shape, need)
    Vext = Vt_ext[:need].T
    Cext = np.linalg.solve(Rg_np, Vext)
    Cfull = np.concatenate((C32, Cext), axis=1)
    Vfull = Rg_np @ Cfull
    orth = float(np.linalg.norm(Vfull.T @ Vfull - np.eye(R)))
    prefix_exact = bool(np.array_equal(Cfull[:, :kept], C32))
    phases['svd_and_extension'] = time.perf_counter() - t1
    energy = {}
    total = float(np.sum(s_full ** 2))
    for q in (0, 8, 16, 32, 64, 128, 256, 512):
        if q <= R:
            captured = float(np.sum(s_full[:q] ** 2)) / max(total, 1e-300)
            energy[str(q)] = captured
    info = dict(rule=('retained 32 directions verbatim; columns 33..R are right singular vectors '
                      'of the normalised training residual at the stored codes, in the exact '
                      'QR field metric at the training mesh, restricted to the complement of '
                      'the retained 32; nested in q'),
                training_intervals=int(intervals), training_sources=int(len(train_draws)),
                R=R, retained_columns=kept, retained_prefix_exact=prefix_exact,
                retained_gram_deviation_in_this_metric=gram_dev,
                retained_metric_R_max_abs_difference=float(np.max(np.abs(
                    np.asarray(basis['R']) - Rg_np))) if 'R' in basis else None,
                rebuilt_prefix_subspace_defect=prefix_subspace_defect,
                extension_orthonormality_error=orth,
                full_singular_values=s_full.tolist(),
                extension_singular_values=s_ext[:need].tolist(),
                residual_energy_captured=energy,
                bank_rank=rank, phase_seconds=phases,
                subset_note=(None if subset is None or subset >= len(codes) + 0 else
                             f'extension built from the first {subset} fit sources (smoke only)'),
                directions_sha256=sha_array(Cfull),
                seconds=time.perf_counter() - t0)
    del G, U, T
    return Cfull, info


def oracle_projected(head, Rg, Zcand, T, perp2, nu2, V, budget, starts=8, gtol=1e-6,
                     linear='gj', block=16):
    """Best-found error on the AUGMENTED manifold {G(h(z) + C_q y)}, per source.

    With V = R_G C_q orthonormal in the field metric, the optimal y is explicit and the
    remaining objective is the residual projected off span(V): a K-dimensional multistart
    LM with an R-dimensional residual, exactly `pbh_core.oracle_errors` with the projector
    folded in. At q = 0 (V empty) it is that oracle; at q = R it is the bank floor.

    DESIGN A10. `C_q` is orthonormal in the bank's QR metric at the TRAINING mesh (255),
    but `R_G` here is the QUERY mesh's factor, whose column norms scale like
    ((n-1)/254)^2 relative to the training mesh: `V^T V` measured 0.063 I at 64, 1.0079 I
    at 256 and 16.13 I at 1024 intervals. Jobs 3780692 (256) and 3783813 (1024) ran this
    function with `r - V V^T r`, which is a projection only when V^T V = I, so their
    q > 0 oracle values are wrong (inflated by (1 - rho)^2 |P r|^2; at 256 the inflation
    is <= 5e-4 relative, at 1024 the column is meaningless). The q = 0 value has no V and
    is unaffected. The fix is one line: V is orthonormalised by a thin QR before use, so
    span(V) -- the only thing the oracle depends on -- is unchanged and the projector is
    exact at every mesh. No timed arm, gate or D1-D3 quantity calls this function.
    """
    K = int(np.asarray(Zcand).shape[1])
    Vj = jnp.asarray(V)
    q = int(Vj.shape[1])
    if q:
        Vj, _ = jnp.linalg.qr(Vj, mode='reduced')
        assert float(jnp.max(jnp.abs(Vj.T @ Vj - jnp.eye(q)))) < 1e-8

    def resid(z, t, R):
        r = R @ head(z) - t
        return r - Vj @ (Vj.T @ r) if q else r
    lm = A.make_stationary_lm(resid, budget, gtol=gtol, linear=linear)

    @jax.jit
    def fit(starts_b, T_b, R):
        out = jax.vmap(lambda zs, t: jax.vmap(lambda z0: lm(z0, (t, R), 0.))(zs))(starts_b, T_b)
        best = jnp.argmin(out[1], axis=1)
        idx = jnp.arange(out[1].shape[0])
        return out[1][idx, best], out[2][idx, best], out[3][idx, best]

    H = jax.jit(jax.vmap(head))(jnp.asarray(Zcand))
    Hr = H @ jnp.asarray(Rg).T
    if q:
        Hr = Hr - (Hr @ Vj) @ Vj.T
    Hn = jnp.sum(Hr * Hr, axis=1)
    Zc = jnp.asarray(Zcand)
    Rgj = jnp.asarray(Rg)
    Tj = jnp.asarray(T)
    Tp = Tj - (Tj @ Vj) @ Vj.T if q else Tj
    total = int(T.shape[0])
    block = min(block, total)
    errs, iters, reasons = [], [], []
    for s in range(0, total, block):
        take = np.arange(s, min(s + block, total))
        keep = len(take)
        if keep < block:
            take = np.concatenate((take, np.full(block - keep, take[-1])))
        Tb = Tj[jnp.asarray(take)]
        score = Hn[None, :] - 2. * (Tp[jnp.asarray(take)] @ Hr.T)
        pick = jnp.argsort(score, axis=1)[:, :starts]
        rn, it, reason = jax.device_get(fit(Zc[pick], Tb, Rgj))
        p2 = np.asarray(perp2)[take[:keep]]
        n2 = np.asarray(nu2)[take[:keep]]
        errs.append(np.sqrt(np.asarray(rn)[:keep] ** 2 + p2) / np.sqrt(n2))
        iters.append(np.asarray(it)[:keep])
        reasons.append(np.asarray(reason)[:keep])
    return (np.concatenate(errs), np.concatenate(iters).astype(int),
            np.concatenate(reasons).astype(int))


def make_linear_query(B, n):
    """The rank-R linear reduced model solved directly: thin QR of the M x R weak
    operator offline, then one projection, one triangular solve and one decode online.
    Mathematically identical to the eliminated ladder at q = R (where the nonlinear
    iteration is inert) and to the free bank, without either's iteration."""
    Q, Rr = jnp.linalg.qr(jnp.asarray(B), mode='reduced')
    Qt = Q.T

    @jax.jit
    def kernel(source, S, I, J, W, Qt, Rr, B, bank):
        fm = (S.T @ source[1:-1, 1:-1] @ S)[I, J] * W
        y = jax.scipy.linalg.solve_triangular(Rr, Qt @ fm, lower=False)
        field = jnp.pad((bank @ y).reshape(n - 1, n - 1), 1)
        return field, y, jnp.linalg.norm(B @ y - fm), jnp.linalg.norm(fm)
    values = np.asarray(jnp.linalg.svd(Rr, compute_uv=False))
    info = dict(qr_condition_number=float(values[0] / values[-1]), M=int(B.shape[0]),
                linear_solve='thin QR offline, triangular solve online',
                qr_Q_sha256=sha_array(Qt), qr_R_sha256=sha_array(Rr))
    return kernel, Qt, Rr, info


def linear_query_once(kernel, source, ops, Qt, Rr, B, bank):
    start = time.perf_counter()
    src = jax.device_put(source)
    src.block_until_ready()
    input_end = time.perf_counter()
    out = kernel(src, ops['S'], ops['I'], ops['J'], ops['W'], Qt, Rr, B, bank)
    jax.block_until_ready(out)
    device_end = time.perf_counter()
    field, y, rn, fmn = jax.device_get(out)
    end = time.perf_counter()
    return np.asarray(field), dict(total_seconds=end - start, input_seconds=input_end - start,
                                   fused_device_seconds=device_end - input_end,
                                   output_seconds=end - device_end, residual=float(rn),
                                   iterations=0, reason=4, stationarity=0.0,
                                   relative_residual=float(rn) / max(float(fmn), 1e-300),
                                   latent=np.asarray(y).tolist())


def non_dominated(points):
    """Indices of the points not dominated on (error, cost), both to be minimised."""
    keep = []
    for i, (e, c) in enumerate(points):
        dominated = any((e2 <= e and c2 <= c) and (e2 < e or c2 < c)
                        for j, (e2, c2) in enumerate(points) if j != i)
        if not dominated:
            keep.append(i)
    return keep
