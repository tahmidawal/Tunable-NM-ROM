"""tensor-crossover: helpers shared by the 2D and 3D drivers (PROTOCOL.md is the specification).

- build_tensor: T_mij = sum_q P_qm (B_qi D_qj + D_qi B_qj), the precomputed form of the off-mesh quadratic term (PROTOCOL §2).
- select_rung: the reference-free suffix qualification of PROTOCOL §4.
- panel: interleaved, shuffled, burned-in timing of zero-argument callables (warm-up excluded), as the plateau / NS lanes.
- make_kernel: lax.scan of K chained Jacobian evaluations (NS lane microbenchmark).
- mem_analysis: XLA memory_analysis of a compiled executable.
"""
from __future__ import annotations

import time

import numpy as np
import jax
import jax.numpy as jnp

block = jax.block_until_ready


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def nbytes(*arrs):
    return int(sum(int(np.prod(a.shape)) * a.dtype.itemsize for a in arrs))


def tree_bytes(tree):
    return int(sum(int(np.prod(a.shape)) * a.dtype.itemsize for a in jax.tree_util.tree_leaves(tree) if hasattr(a, 'shape')))


def is_oom(ex):
    s = str(ex)
    return ('RESOURCE_EXHAUSTED' in s) or ('out of memory' in s.lower()) or ('Out of memory' in s)


def build_tensor(P, B, D, chunk_bytes=1 << 31):
    """(M, r, r) symmetric tensor of the quadrature term, built in chunks of test functions (no (m, M, r) host table)."""
    m, M = P.shape
    r = B.shape[1]
    mc = int(max(1, min(M, chunk_bytes // (8 * m * r))))
    @jax.jit
    def f(Pc, B, D):
        X = (Pc[:, :, None] * B[:, None, :]).reshape(m, -1)          # (m, mc*r): P_qa B_qi
        return (X.T @ D).reshape(Pc.shape[1], r, r)                  # sum_q P_qa B_qi D_qj
    parts = [block(f(P[:, a:a + mc], B, D)) for a in range(0, M, mc)]
    T = jnp.concatenate(parts, 0) if len(parts) > 1 else parts[0]
    del parts
    T = block(T + jnp.transpose(T, (0, 2, 1)))
    return T


def rel_field_dist(F, Fref):
    """max over output times t > 0 of ||F_t - Fref_t|| / ||Fref_t|| (F: (T, ...))."""
    F, Fref = np.asarray(F).reshape(len(F), -1), np.asarray(Fref).reshape(len(Fref), -1)
    num = np.linalg.norm(F[1:] - Fref[1:], axis=1)
    den = np.maximum(np.linalg.norm(Fref[1:], axis=1), 1e-300)
    return float(np.max(num / den))


def rel_max(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def select_rung(names, sizes, dists, tau):
    """Smallest rung k with d_k' <= tau for k and every larger rung (suffix rule). Returns (name or None, floor flag)."""
    order = np.argsort(sizes)
    names = [names[i] for i in order]
    d = [dists[i] for i in order]
    ok = [x is not None and np.isfinite(x) and x <= tau for x in d]
    sel = None
    for k in range(len(names)):
        if all(ok[k:]):
            sel = k
            break
    if sel is None:
        return None, False
    return names[sel], bool(sel == 0)


def make_burn():
    a = jnp.ones((2048, 2048))
    f = jax.jit(lambda x: x @ x * 1e-4)
    block(f(a))

    def burn(sec):
        t = time.perf_counter()
        while time.perf_counter() - t < sec:
            block(f(a))
    return burn


def panel(subjects, reps, burn, burn_each=0.1, initial_burn=30.0, seed=0, check=None, label=''):
    """subjects: {name: [callable per case]}; each callable returns device output. Warm-up (compile + first run) excluded."""
    for nm, fns in subjects.items():
        for fn in fns:
            block(fn())
    burn(initial_burn)
    rng = np.random.default_rng(seed)
    inv = []
    for rp in range(reps):
        order = [(nm, j) for nm, fns in subjects.items() for j in range(len(fns))]
        rng.shuffle(order)
        for nm, j in order:
            burn(burn_each)
            t1 = time.perf_counter()
            o = subjects[nm][j]()
            block(o)
            secs = time.perf_counter() - t1
            inv.append(dict(name=nm, case=j, rep=rp, seconds=secs, diff=(check(nm, j, o) if check else None)))
            del o
    summ = {}
    for nm in subjects:
        ts = np.array([d['seconds'] for d in inv if d['name'] == nm])
        q1, q3 = np.quantile(ts, [.25, .75])
        med = float(np.median(ts))
        blocks = [float(np.median([d['seconds'] for d in inv if d['name'] == nm and d['rep'] == rp])) for rp in range(reps)]
        summ[nm] = dict(median_ms=1e3 * med, q1_ms=1e3 * float(q1), q3_ms=1e3 * float(q3), iqr_ms=1e3 * float(q3 - q1),
                        min_ms=1e3 * float(ts.min()), max_ms=1e3 * float(ts.max()), n=int(ts.size),
                        block_drift=float(max(max(blocks) / med, med / min(blocks))))
    return inv, summ


def make_kernel(jac, K):
    """jac(c, d) -> (val (M,), J (M, r)). Returns jitted kern(d, c0) running K chained evaluations."""
    @jax.jit
    def kern(d, c0):
        def body(c, _):
            val, J = jac(c, d)
            return c + 1e-12 * (jnp.sum(J, axis=0) + jnp.sum(val)), None
        c, _ = jax.lax.scan(body, c0, None, length=K)
        return c
    return kern


def mem_analysis(jitted, *args):
    try:
        ma = jitted.lower(*args).compile().memory_analysis()
        if ma is None:
            return None
        out = {k: int(getattr(ma, k)) for k in ('argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes',
                                                 'alias_size_in_bytes', 'generated_code_size_in_bytes') if hasattr(ma, k)}
        out['device_bytes'] = out.get('argument_size_in_bytes', 0) + out.get('output_size_in_bytes', 0) + \
            out.get('temp_size_in_bytes', 0) - out.get('alias_size_in_bytes', 0)
        return out
    except Exception as ex:  # noqa: BLE001
        return dict(error=repr(ex)[:300])


def device_peak():
    try:
        s = jax.devices()[0].memory_stats() or {}
        return dict(peak_bytes_in_use=int(s.get('peak_bytes_in_use', -1)), bytes_in_use=int(s.get('bytes_in_use', -1)),
                    bytes_limit=int(s.get('bytes_limit', -1)))
    except Exception:  # noqa: BLE001
        return None
