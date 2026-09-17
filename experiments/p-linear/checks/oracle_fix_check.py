"""Sub-minute local check of the DESIGN A10 oracle fix, at 64 intervals on new_K32.

Runs `plin_core.oracle_projected` before-fix (V as the jobs used it) and after-fix (V
orthonormalised) with the retained 32 directions and with a full-rank R-column basis, and
records: the q = R value must equal the bank floor to 1e-10 relative after the fix, the
q = 32 value must be bracketed floor <= best-found(32) <= best-found(0), and the q = 0
value must be identical before and after (the fix does not touch it).

    PYTHONPATH=... jaxrun python checks/oracle_fix_check.py --out checks/2026-09-17-oracle-fix-check.json
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'p-bank-head'))
sys.path.insert(0, str(HERE.parent / 'multiresolution-poisson'))
sys.path.insert(0, str(HERE.parent / 'head-ablation'))
sys.path.insert(0, str(HERE.parent / 'mr-burgers2d'))
sys.path.insert(0, str(HERE.parent / 'separable-decoder'))
jax.config.update('jax_enable_x64', True)
import plin_core as L_  # noqa: E402
import pbh_core as K_  # noqa: E402
import sep_common as sc  # noqa: E402
import pbh_audit_np as N  # noqa: E402


def oracle_unfixed(head, Rg, Z, T, perp2, nu2, V, budget, starts, gtol, linear):
    """The function exactly as jobs 3780692 / 3783813 ran it (no orthonormalisation)."""
    import inspect
    src = inspect.getsource(L_.oracle_projected)
    src = src.replace("    if q:\n        Vj, _ = jnp.linalg.qr(Vj, mode='reduced')\n"
                      "        assert float(jnp.max(jnp.abs(Vj.T @ Vj - jnp.eye(q)))) < 1e-8\n", '')
    assert 'linalg.qr' not in src
    ns = dict(L_.__dict__)
    exec(src, ns)
    return ns['oracle_projected'](head, Rg, Z, T, perp2, nu2, V, budget, starts, gtol, linear)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--intervals', type=int, default=64)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    n = a.intervals
    assert jax.default_backend() == 'gpu'
    t0 = time.perf_counter()
    params, Z, _ = N.load(HERE / 'checkpoints/primary_K32.pkl')
    basis = np.load(HERE / 'checkpoints/primary_K32-basis.npz')
    C32 = np.asarray(basis['coefficient_directions'])
    R = C32.shape[0]
    cfg = json.loads((HERE / 'config-1024.json').read_text())
    dev = np.concatenate((N.source_params(cfg['eval_seed'], cfg['eval_count']),
                          N.source_params(cfg['fresh_seed'], cfg['fresh_count'])))
    same = [N.solve(n, p) for p in dev]
    G = K_.bank_of(params, n)
    Rg, rank = K_.bank_r(G)
    assert rank['rank_valid'], rank
    U = jnp.stack([jnp.asarray(u[1:-1, 1:-1].ravel()) for u in same])
    T, perp2, nu2 = K_.project_targets(G, Rg, U)
    floor = np.asarray(jnp.sqrt(perp2 / nu2))
    head = lambda z: sc.head(params, z)
    Rg_np = np.asarray(Rg)
    # a full-rank R-column basis whose first 32 columns are the retained ones: complete C32
    # in the TRAINING-mesh metric (255), exactly as extend_basis does, with a fixed random
    # complement -- any full-rank completion gives the floor at q = R.
    G255 = K_.bank_of(params, 255)
    R255, _ = K_.bank_r(G255)
    R255 = np.asarray(R255)
    V32 = R255 @ C32
    rng = np.random.default_rng(0)
    X = rng.standard_normal((R, R - 32))
    X = X - V32 @ (V32.T @ X)
    Q, _ = np.linalg.qr(X)
    Cfull = np.concatenate((C32, np.linalg.solve(R255, Q)), axis=1)
    Vt = R255 @ Cfull
    assert np.linalg.norm(Vt.T @ Vt - np.eye(R)) < 1e-6
    del G255
    res = dict(intervals=n, floor_worst=float(floor.max()), R=R)
    V_scale = float(np.mean(np.diag((Rg_np @ C32).T @ (Rg_np @ C32))))
    res['VtV_diag_mean_query_mesh'] = V_scale
    kw = dict(budget=cfg['recon_budget'], starts=cfg['recon_starts'], gtol=cfg['stationarity_tolerance'], linear='gj')
    for label, fn in (('unfixed', oracle_unfixed), ('fixed', L_.oracle_projected)):
        row = {}
        for q in (0, 32, R):
            V = Rg_np @ Cfull[:, :q]
            e, it, rs = fn(head, Rg, Z, T, perp2, nu2, V, **kw)
            row[str(q)] = dict(worst=float(e.max()), median=float(np.median(e)), iterations=it.tolist(), reasons=rs.tolist())
            print(label, 'q', q, 'worst', float(e.max()), 'floor', float(floor.max()), flush=True)
        res[label] = row
    f = res['fixed']
    res['checks'] = dict(
        q0_identical=bool(abs(res['unfixed']['0']['worst'] - f['0']['worst']) < 1e-14),
        fixed_qR_equals_floor_rel=float(abs(f[str(R)]['worst'] - floor.max()) / floor.max()),
        fixed_qR_equals_floor=bool(abs(f[str(R)]['worst'] - floor.max()) / floor.max() < 1e-10),
        fixed_bracket_q32=bool(floor.max() - 1e-15 <= f['32']['worst'] <= f['0']['worst'] + 1e-15),
        fixed_monotone=bool(f['0']['worst'] >= f['32']['worst'] >= f[str(R)]['worst']),
        unfixed_qR_relative_excess_over_floor=float(res['unfixed'][str(R)]['worst'] / floor.max() - 1))
    res['checks']['all_passed'] = all(v for k, v in res['checks'].items() if isinstance(v, bool))
    res['seconds'] = time.perf_counter() - t0
    Path(a.out).write_text(json.dumps(res, indent=2) + '\n')
    print(json.dumps(res['checks'], indent=2))


if __name__ == '__main__':
    main()
