"""Local parity smoke for the b-speed arms. Sub-minute per invocation by design.

Usage (each call is its own short `jaxrun` on the shared box):

    PYTHONPATH=... python smoke_speed.py --checkpoint CK --stage solve
    PYTHONPATH=... python smoke_speed.py --checkpoint CK --stage jac
    PYTHONPATH=... python smoke_speed.py --checkpoint CK --stage query --arms incumbent,L1
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e
import arms as A
import fast as F
import ladders


def host(x):
    return jax.tree_util.tree_map(np.asarray, x)


def rel(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    d = np.linalg.norm(a - b)
    return float(d / max(np.linalg.norm(b), 1e-300))


def stage_solve(rng):
    """gj_block(bs=1) must be `engines.gj_solve` bitwise; bs>1 must agree to 1e-12."""
    out = {}
    worst = {1: 0., 2: 0., 4: 0., 8: 0.}
    exact = 0
    for _ in range(256):
        B = rng.normal(size=(16, 16))
        H = B @ B.T + 16 * np.eye(16)
        lam = 10. ** rng.uniform(-12, 2)
        M = H + lam * np.diag(np.diag(H) + 1e-30)
        g = rng.normal(size=16)
        ref = np.asarray(e.gj_solve(jnp.asarray(M), jnp.asarray(g)))
        for bs in (1, 2, 4, 8):
            got = np.asarray(F.gj_block(jnp.asarray(M), jnp.asarray(g), bs))
            worst[bs] = max(worst[bs], rel(got, ref))
            if bs == 1:
                exact += int(np.array_equal(got, ref))
        true = np.linalg.solve(M, g)
        worst['numpy'] = max(worst.get('numpy', 0.), rel(ref, true))
    out['block1_bitwise_count'] = exact
    out['block1_bitwise'] = exact == 256
    out['relative'] = {str(k): v for k, v in worst.items()}
    return out


def stage_jac(params, K):
    """Does vmap(linearize-jvp) reproduce jacfwd on the real head?

    Mathematically these are the same JVPs, but each is compiled as its own program, so
    bit equality is an XLA fusion question and is MEASURED here rather than assumed. The
    gate that carries the weight is the end-to-end query parity in `stage_jac`'s sibling
    stage; this one only tells us which parity class to declare for `fuse`."""
    rng = np.random.default_rng(7)
    head = lambda z: A.sc.head(params, z)
    fun = lambda z: head(z)[:64] * 1.0

    @jax.jit
    def pair(z):
        ref = jax.jacfwd(fun)(z)
        r, jvp = jax.linearize(fun, z)
        return ref, r, jax.vmap(jvp, out_axes=1)(jnp.eye(K)), fun(z)

    eq = 0
    eqr = 0
    worst = 0.
    for _ in range(16):
        z = jnp.asarray(rng.normal(size=K) * .3)
        ref, r, got, rref = host(pair(z))
        eq += int(np.array_equal(got, ref))
        eqr += int(np.array_equal(r, rref))
        worst = max(worst, rel(got, ref))
    return dict(jacobian_bitwise_count=eq, jacobian_bitwise=eq == 16,
                primal_bitwise_count=eqr, primal_bitwise=eqr == 16, relative=worst)


def build(params, Zold, K, R, cfg):
    L, M, m = cfg['intervals'], cfg['M'], cfg['m']
    bank = A.CoordBank(params, K, R)
    head = A.neural_head(params)
    data, info = A.build_operators(bank, L, M, 'eq', Zcoef=Zold, m=m, eq_seed=cfg['eq_seed'],
                                   candidate_cap=cfg['candidate_cap'],
                                   fit_states=cfg['fit_states'], head=head)
    stride = max(1, len(Zold) // cfg['code_subsample'])
    cold, cinfo = A.build_cold(bank, head, np.asarray(Zold[::stride]), cfg['cold_axis_points'])
    trust = .01 * float(np.max(np.linalg.norm(Zold - Zold.mean(0), axis=1)))
    return data, cold, trust, info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--stage', required=True)
    p.add_argument('--arms', default='')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    t0 = time.perf_counter()
    rep = dict(stage=a.stage, backend=jax.default_backend(), config=cfg)

    if a.stage == 'solve':
        rep['result'] = stage_solve(np.random.default_rng(20260916))
        Path(a.out).write_text(json.dumps(rep, indent=2) + '\n')
        print(json.dumps(rep['result'], indent=2), flush=True)
        return

    ck = pickle.load(open(a.checkpoint, 'rb'))
    params = jax.tree_util.tree_map(jnp.asarray, host(ck['params']))
    Zold = np.asarray(ck['Z_tr'])
    K = int(Zold.shape[1])
    R = int(np.asarray(ck['params']['h_lin']).shape[1])

    if a.stage == 'jac':
        rep['result'] = stage_jac(params, K)
        Path(a.out).write_text(json.dumps(rep, indent=2) + '\n')
        print(json.dumps(rep['result'], indent=2), flush=True)
        return

    assert a.stage == 'query'
    L, dt, m = cfg['intervals'], cfg['dt'], cfg['m']
    data, cold, trust, info = build(params, Zold, K, R, cfg)
    phys = e.params_draw(cfg['case_seed'], cfg['cases'])
    inputs = [jnp.asarray(e.initial(L, ph)) for ph in phys]
    strict = cfg['strict']

    ref = A.make_query(A.neural_head(params), K, L, dt, trust, 'eq', linear='gj', **strict)
    base = [host(ref(inputs[c], float(phys[c, 4]), data, cold)) for c in range(len(phys))]
    rep['reference'] = dict(iterations=[int(sum(b[1])) for b in base],
                            ic_iterations=[int(b[5]) for b in base],
                            seconds=time.perf_counter() - t0)
    print('REFERENCE', rep['reference'], flush=True)

    names = [s for s in a.arms.split(',') if s]
    rows = []
    for name in names:
        o = ladders.ARMS[name]
        s0 = time.perf_counter()
        tab = F.build_tables(params, data, cold, o)
        q = F.make_query(params, K, L, dt, m, trust, o, **strict)
        row = dict(arm=name, opts=o, setup_seconds=time.perf_counter() - s0)
        fld, lat, its, rsn = 0., 0., True, True
        for c in range(len(phys)):
            v = host(q(inputs[c], float(phys[c, 4]), data, cold, tab))
            b = base[c]
            fld = max(fld, max(rel(v[0][t], b[0][t]) for t in range(len(b[0]))))
            lat = max(lat, rel(v[7], b[7]))
            its &= bool(np.array_equal(v[1], b[1])) and int(v[5]) == int(b[5])
            rsn &= bool(np.array_equal(v[3], b[3])) and int(v[6]) == int(b[6])
        row.update(field_relative=fld, latent_relative=lat,
                   iterations_identical=bool(its), reasons_identical=bool(rsn),
                   seconds=time.perf_counter() - s0)
        row['parity'] = bool(fld <= 1e-12 and its and rsn)
        rows.append(row)
        print('ARM', name, json.dumps({k: row[k] for k in
              ('field_relative', 'latent_relative', 'iterations_identical',
               'reasons_identical', 'parity', 'seconds')}), flush=True)
    rep['arms'] = rows
    rep['elapsed_seconds'] = time.perf_counter() - t0
    Path(a.out).write_text(json.dumps(rep, indent=2) + '\n')


if __name__ == '__main__':
    main()
