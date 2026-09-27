"""The audited nested correction directions, reproduced, plus a flattened fit.

`audited` is `ladder.residual_directions` copied verbatim except that it also returns
the best-found head codes, the head residual and the field-orthonormal directions, so
the enriched quadrature codes come free. Its arithmetic is unchanged, so its
`directions_sha256` is comparable with the retained
`f270e5bf682ad220f2164ced82dde0f3e0374e84e2fa002f3101e0728062d399`.

`flat` is the same fit with the doubly vectorised `vmap(vmap(while_loop))` replaced by
a single `vmap` over (snapshot, start) pairs. The audited ladder flagged that nesting
as the compilation-bound setup cost (1757.6 s, almost all of it one XLA slow-operation
alarm). Both are run so the saving, and whether the two agree bitwise, are measured
rather than assumed. The directions used downstream are always the audited ones.
"""
from __future__ import annotations

import time

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import arms as A


def sha_array(x):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def host(x):
    return jax.tree_util.tree_map(np.asarray, x)


def _common(params, Rb, coef, Zcand, cfg):
    rng = np.random.default_rng(cfg['residual_seed'])
    idx = np.sort(rng.choice(len(coef), min(cfg['residual_snapshots'], len(coef)), replace=False))
    target = jnp.asarray(np.asarray(coef)[idx])
    head = lambda z: A.sc.head(params, z)
    Hc = jax.jit(jax.vmap(head))(jnp.asarray(Zcand))
    Hrot = Hc @ Rb.T
    Hn = jnp.sum(Hrot * Hrot, 1)
    lm = A.make_stationary_lm(lambda z, t: Rb @ (head(z) - t), cfg['residual_budget'],
                              gtol=cfg['strict']['gtol'], linear='gj')
    return idx, target, head, Hrot, Hn, lm


def _finish(head, Rb, target, Z, its, reasons, idx, coef, cfg, seconds, path):
    rho = target - jax.jit(jax.vmap(head))(Z)
    tilde = rho @ Rb.T
    _, sv, Vt = jnp.linalg.svd(tilde, full_matrices=False)
    rank = int(min(Vt.shape[0], coef.shape[1]))
    Ct = Vt[:rank].T
    C = jnp.linalg.solve(Rb, Ct)
    total = float(jnp.sum(sv ** 2))
    fitted = np.sqrt(np.asarray(jnp.sum(tilde * tilde, 1)
                                / jnp.maximum(jnp.sum((target @ Rb.T) ** 2, 1), 1e-300)))
    info = dict(rule=('field-metric POD of the decoder-output residual eta - h_theta(z*), with z* the '
                      'best-found head code under the shared LM rule; nested in q'),
                path=path, snapshots=int(len(idx)),
                snapshot_indices_sha256=sha_array(idx),
                starts=cfg['residual_starts'], budget=cfg['residual_budget'],
                seed=cfg['residual_seed'], available_rank=rank,
                head_fit_relative_median=float(np.median(fitted)),
                head_fit_relative_worst=float(np.max(fitted)),
                head_fit_iterations_median=float(np.median(its)),
                head_fit_reason_counts={str(k): int(v) for k, v in
                                        zip(*np.unique(reasons, return_counts=True))},
                singular_values=np.asarray(sv[:min(len(sv), 600)]).tolist(),
                residual_energy_captured={},
                seconds=seconds, directions_sha256=sha_array(np.asarray(C)))
    for q in cfg['q_ladder']:
        info['residual_energy_captured'][str(q)] = float(np.sum(np.asarray(sv[:q]) ** 2)) / max(total, 1e-300)
    return C, Ct, Z, rho, info


def audited(params, Rb, coef, Zcand, K, cfg):
    """`ladder.residual_directions`, arithmetic unchanged, extra returns added."""
    t0 = time.perf_counter()
    idx, target, head, Hrot, Hn, lm = _common(params, Rb, coef, Zcand, cfg)

    @jax.jit
    def fit(t):
        score = Hn - 2 * (Hrot @ (Rb @ t))
        starts = jnp.asarray(Zcand)[jnp.argsort(score)[:cfg['residual_starts']]]
        outs = jax.vmap(lambda z0: lm(z0, (t,), 0.))(starts)
        best = jnp.argmin(outs[1])
        return outs[0][best], outs[1][best], outs[2][best], outs[3][best]

    zs, rns, its, reasons = [], [], [], []
    for s in range(0, len(target), 64):
        v = host(jax.vmap(fit)(target[s:s + 64]))
        zs.append(v[0]); rns.append(v[1]); its.append(v[2]); reasons.append(v[3])
    Z = jnp.asarray(np.concatenate(zs))
    return _finish(head, Rb, target, Z, np.concatenate(its), np.concatenate(reasons),
                   idx, coef, cfg, time.perf_counter() - t0, 'audited_double_vmap')


def flat(params, Rb, coef, Zcand, K, cfg):
    """One `vmap` over (snapshot, start) pairs instead of `vmap` over `vmap`."""
    t0 = time.perf_counter()
    idx, target, head, Hrot, Hn, lm = _common(params, Rb, coef, Zcand, cfg)
    ns = int(cfg['residual_starts'])
    score = Hn[None, :] - 2 * ((target @ Rb.T) @ Hrot.T)          # (S, C)
    pick = jnp.argsort(score, axis=1)[:, :ns]                     # (S, ns)
    z0 = jnp.asarray(Zcand)[pick].reshape(-1, int(np.asarray(Zcand).shape[1]))
    tt = jnp.repeat(target, ns, axis=0)
    run = jax.jit(jax.vmap(lambda a, b: lm(a, (b,), 0.)))
    zs, rns, its, reasons = [], [], [], []
    chunk = 64 * ns
    for s in range(0, len(z0), chunk):
        v = host(run(z0[s:s + chunk], tt[s:s + chunk]))
        zs.append(v[0]); rns.append(v[1]); its.append(v[2]); reasons.append(v[3])
    Zf = np.concatenate(zs).reshape(len(target), ns, -1)
    rn = np.concatenate(rns).reshape(len(target), ns)
    itf = np.concatenate(its).reshape(len(target), ns)
    rsf = np.concatenate(reasons).reshape(len(target), ns)
    best = np.argmin(rn, axis=1)
    rows = np.arange(len(target))
    Z = jnp.asarray(Zf[rows, best])
    return _finish(head, Rb, target, Z, itf[rows, best], rsf[rows, best],
                   idx, coef, cfg, time.perf_counter() - t0, 'flat_single_vmap')
