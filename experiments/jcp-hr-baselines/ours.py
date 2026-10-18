"""Ours' queries for the A side of the A-B-A timing (DESIGN.md 7, 8; A3.10). Byte-identical copies of the combined
lane's code (INHERITED.json): 2D `c2common.Setting` (t2core LMM query at the mesh), 3D `c3common.Solvers` (fs3 /
adaptive queries). The exact X configurations are pinned at phase-II release; the phase-I smoke uses the combined lane's
deployed 2D R384 gauss128 LSPG-CN-R and 3D W256 gl24 FS-K1 BE to prove the adapter at real size.

Each adapter returns the six output fields on the scoring lattice (2D (6, 257, 257); 3D (6, 63^3)) and a timed callable
whose result is block_until_ready-ed by the caller.
"""
from __future__ import annotations

import time

import numpy as np
import jax
import jax.numpy as jnp

import hrcore as H


class Ours2D:
    def __init__(self, L, settings, log=print):
        import c2common as C2
        import t2core as T2
        self.C2, self.T2, self.L = C2, T2, L
        t0 = time.perf_counter()
        self.bank = C2.Bank(dict(settings=settings), L)
        self.S = {s['name']: C2.Setting(self.bank, s) for s in settings}
        self.setup_seconds = time.perf_counter() - t0
        log(f'   ours2d: bank + settings at L={L} in {self.setup_seconds:.1f}s')

    def fn(self, name, form, scheme, dt, level='prod'):
        """-> f(u0_full, nu) returning the raw query output (fields (6, L+1, L+1))."""
        S = self.S[name]
        g, t_ = self.C2.tol_of(form, level)
        sch = self.T2.sched(scheme, dt, g, t_)
        q, data, cold = S.q[form], S.data[('main', form)], S.cold
        return lambda u0, nu: q(u0, nu, data, cold, sch)

    def lattice(self, out):
        s = self.L // H.LATTICE[2]
        return np.asarray(out['fields'])[:, ::s, ::s]

    def u0(self, fam, j):
        return jnp.asarray(H.E2.initial(self.L, fam.D[j]))


class Ours3D:
    def __init__(self, bank_name, Rp, rule, a, n=65, trust_frac=0.05, log=print):
        import c3common as C3c
        self.C3c, self.n = C3c, n
        t0 = time.perf_counter()
        bank = C3c.load_bank(bank_name)
        X, w = C3c.load_rules()[rule]
        tb = C3c.Tables(n, bank, [Rp], log=lambda *a_: None)
        self.prep = C3c.prepare(tb.rule_data(Rp, X, w))
        self.solv = C3c.Solvers(n, Rp, trust_frac * bank['spread'][str(Rp)], 1e-3, a)
        self.setup_seconds = time.perf_counter() - t0
        log(f'   ours3d: {bank_name} R{Rp} {rule} tables in {self.setup_seconds:.1f}s')

    def fn(self, spec):
        """spec as c3common.Solvers.run: ('fs', scheme, dt, K) | ('ad', form, scheme, dt, level)."""
        return lambda u0, nu: self.solv.run(spec, u0, nu, self.prep)[0]

    def lattice(self, out):
        return np.asarray(out['fields']).reshape(6, -1)

    def u0(self, fam, j):
        return jnp.asarray(H.C3.initial_interior(self.n, fam.T, j))


S3_SETTINGS = {'M2_512': ('M2', 512, 3), 'W256': ('W1024', 256, 6), 'W512': ('W1024', 512, 6),   # speed3d r1.json
               'W768': ('W1024', 768, 6), 'W1024': ('W1024', 1024, 6)}
P1_RULES = {128: 'fib4181', 256: 'gauss128', 384: 'gauss128', 512: 'gauss192'}


class OursP1:
    """2D batch-1 comparators (A21.1): the jcp-p1-scaling configurations, built and queried with that lane's own code
    (p1common.Bank / Setting, p1core.make_split_query fused query; byte-identical copies, INHERITED.json).
    Keys as in its EXPORT: '<L>|R<r>[nom]|<BE|CNR>'. Query contract: full-field IC sampled at the cold quadrature
    points, reduced projection, LSPG stepping at dt0 = 0.005 ('prod'), full-field decode (V4-4)."""

    def __init__(self, L, keys, log=print):
        import p1common as P1
        self.P1, self.L = P1, L
        t0 = time.perf_counter()
        self.bank = P1.Bank(L, [0, 128, 256, 384, 512])
        self.S, self.spec = {}, {}
        for k in keys:
            Lk, name, scheme = k.split('|')
            assert int(Lk) == L, k
            nom = name.endswith('nom')
            r = int(name[1:].replace('nom', ''))
            sn = f'{name}'
            if sn not in self.S:
                self.S[sn] = P1.Setting(self.bank, sn, r, P1_RULES[r], policy='nominal' if nom else 'frozen')
            self.spec[k] = (sn, scheme)
        self.setup_seconds = time.perf_counter() - t0
        log(f'   oursP1: L={L} settings {sorted(self.S)} in {self.setup_seconds:.1f}s')

    def fn(self, key):
        sn, scheme = self.spec[key]
        st = self.S[sn]
        sch = st.sched('LSPG', scheme, 0.005, 'prod')
        q, data, cold = st.q0['LSPG']['fused'], st.data[('main', 'LSPG')], st.cold

        def f(u0, nu):
            return q(u0, nu, data, cold, sch)
        st.check('main', self.bank)
        return f

    def lattice(self, out):
        s = self.L // H.LATTICE[2]
        return np.asarray(out['fields'])[:, ::s, ::s]

    def u0(self, fam, j):
        return jnp.asarray(self.P1.Q.e.initial(self.L, fam.D[j]))
