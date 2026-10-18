"""Gate G-C1 (CPU): budcore with nout = 6 reproduces p1core bitwise (every output, every per-step record) on a
manufactured problem, both forms, BE / CNR / BDF2 / CN; and with nout = 11 at horizon 0.5 its first six outputs equal the
nout = 6 horizon-0.25 run bitwise (the stepper is causal) and outputs 6..10 are finite and distinct.

    JAX_PLATFORMS=cpu /home/tahmid/Dev/.venv/bin/python test_budcore.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p1core as P1  # noqa: E402
import budcore as BC  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

r = np.random.default_rng(20261018)
M, RP, NU = 24, 6, .05
A = r.normal(size=(M, RP)) / np.sqrt(M)
LAM = np.geomspace(1., 1e3, M)
TT = .3 * r.normal(size=(M, RP, RP))
C0 = .5 * r.normal(size=RP)


def nl(c, d):
    return jnp.einsum('mij,i,j->m', d['T'], c, c)


def nlJ(c, d, Tm):
    T = d['T']
    return jnp.einsum('mij,i,j->m', T, c, c), jnp.einsum('mij,j->mi', T + T.transpose(0, 2, 1), c)


def main():
    out = {}
    ok = True
    for form in ('LSPG', 'GAL'):
        d = P1.form_data(dict(A=jnp.asarray(A), lam=jnp.asarray(LAM)), dict(T=jnp.asarray(TT)), form)
        g, t_ = (1e-7, 0.) if form == 'LSPG' else (0., 1e-13)
        for sc in ('BE', 'CNR', 'BDF2', 'CN'):
            dt = .005
            e1 = jax.jit(P1.make_evolve(nl, nlJ, form, RP, 1.0, nrec=51))
            e2 = jax.jit(BC.make_evolve(nl, nlJ, form, RP, 1.0, nrec=51, nout=6))
            e3 = jax.jit(BC.make_evolve(nl, nlJ, form, RP, 1.0, nrec=101, nout=11))
            s25 = P1.sched(sc, dt, g, t_)
            s50 = P1.sched(sc, dt, g, t_, horizon=.5)
            W1, st1 = e1(jnp.asarray(C0), NU, 1.0, d, s25)
            W2, st2 = e2(jnp.asarray(C0), NU, 1.0, d, s25)
            W3, st3 = e3(jnp.asarray(C0), NU, 1.0, d, s50)
            same = bool(np.array_equal(np.asarray(W1), np.asarray(W2)) and all(
                np.array_equal(np.asarray(st1[k]), np.asarray(st2[k]), equal_nan=True) for k in st1))
            pref = bool(np.array_equal(np.asarray(W3)[:6], np.asarray(W1)))
            W3n = np.asarray(W3)
            tail = bool(np.isfinite(W3n).all() and all(np.linalg.norm(W3n[k] - W3n[k - 1]) > 0 for k in range(6, 11)))
            out[f'{form}|{sc}'] = dict(nout6_bitwise=same, prefix_bitwise=pref, tail_finite_distinct=tail)
            ok = ok and same and pref and tail
    out['passed'] = ok
    (HERE / 'checks').mkdir(exist_ok=True)
    (HERE / 'checks' / 'test_budcore.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))
    assert ok


if __name__ == '__main__':
    main()
