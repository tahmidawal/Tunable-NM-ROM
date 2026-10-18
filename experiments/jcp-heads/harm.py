"""jcp-heads arms: one interface over the head ROM (hcommon.HeadSetting) and the linear-span ROM (qualify's
p1common.Setting), so the rule-ladder, accuracy and timing drivers treat both identically (DESIGN 1, 3, A1.3, A1.4).

    arm = make_arm(bank, spec)       spec: {"name", "kind": "head"|"lin", "head": <pkl> | "Rp": r, "M"?, "rule"}
    arm.run(role, scheme, level, u0, nu, record)     fused query -> dict(fields, W, w0, stats)
    arm.parts(level, record)         split query (init / evolve / decode / fused) of that level
    arm.cert(states, nu, role, sch)  per-step (gamma, ||r||) with a fresh autodiff Jacobian in the unknowns
    arm.coef(W)                      coefficient vectors c of saved unknowns (identity for the linear span; h~(z) for a head)
    arm.nl(c, role)                  tested advection N(c) with the role's rule tables (for rho)
    arm.jac(w, co, dt, nu)           history-free step Jacobian (Q3)

Levels (DESIGN 3.1, A1.3): prod (deployed: step budget 600, head initial fit 400 / 1e-6), tight (gtol 1e-7, step budget
2000, head initial fit 2000 / 1e-9), tighter (gtol 1e-8, step budget 2000, head initial fit 2000 / 1e-10). Step
tolerances come from p1common.tol_of('LSPG', level). Only LSPG is used in this lane.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

import p1common as P
from p1common import Q, T2
import hcommon as HM
import hcore as HC

HM.IC_TOL.update(tight=1e-9, tighter=1e-10)          # DESIGN A1.3 (replaces 1e-10 / 1e-11 of section 3.1)
TIGHT_BUDGET = 2000


def steps_of(dt):
    s, k = int(round(.25 / dt)), int(round(.05 / dt))
    assert abs(s * dt - .25) < 1e-12 and abs(k * dt - .05) < 1e-12, dt
    return s


def sched(scheme, dt, level):
    g, t_ = P.tol_of('LSPG', level)
    return T2.sched(scheme, dt, g, t_)


class HeadArm:
    kind = 'head'

    def __init__(self, bank, spec, head_cache, S):
        self.spec, self.name, self.bank = spec, spec['name'], bank
        nstep = max(T2.NSTEP, S + 1)
        hd = head_cache.get(spec['head'])
        if hd is None:
            hd = head_cache[spec['head']] = HM.Head(spec['head'], bank.mdl)
        self.head = hd
        self.st = HM.HeadSetting(bank, spec['name'], hd, spec['rule'], nrec=S + 1, nstep=nstep, M=spec.get('M'))
        self.K = self.dim = hd.K
        self.M, self.L = self.st.M, bank.L
        q = self.st.query('prod', True)
        self.hz = q['hz']
        self._nl = q['nl']
        self._cert = HC.make_head_certifier(self.hz, self._nl)
        self._jac = HC.make_head_jac(self.hz, self._nl)
        self._coef = jax.jit(jax.vmap(self.hz))

    def add_role(self, role, rule):
        self.st.add_role(self.bank, role, rule)

    def drop_role(self, role):
        self.st.drop_role(role)
        self.st.rules.pop(role, None)

    def data(self, role='main'):
        return self.st.data[role]

    @property
    def rules(self):
        return self.st.rules

    @property
    def cold(self):
        return self.st.cold

    def check(self, role):
        self.st.check(role, self.bank)

    def parts(self, level='prod', record=True):
        return self.st.query(level, record)

    def run(self, role, scheme, level, u0, nu, dt, record=True):
        self.check(role)
        q = self.parts(level, record)
        return q['fused'](u0, nu, self.data(role), self.cold, sched(scheme, dt, level))

    def cert(self, states, nu, role, sch):
        return self._cert(jnp.asarray(states), nu, self.data(role), sch)

    def coef(self, W):
        return np.asarray(self._coef(jnp.asarray(W)))

    def nl(self, c, role):
        return np.asarray(self._nl(jnp.asarray(c), self.data(role)))

    def jac(self, w, co, dt, nu, role='main'):
        return np.asarray(self._jac(jnp.asarray(w), jnp.asarray(co), dt, nu, self.data(role)))

    def describe(self):
        hd = self.head
        nparam = int(sum(np.size(x) for x in jax.tree_util.tree_leaves((hd.params['h'], hd.params['h_lin']))))
        return dict(kind='head', K=self.K, M=self.M, head=str(hd.path.name), head_sha256=hd.sha256, trust=hd.trust,
                    head_params=nparam, code_std=hd.code_std, code_max_dev=hd.code_max_dev, n_codes=hd.n_codes,
                    candidates=int(len(hd.Zsub)), cfg={k: hd.cfg.get(k) for k in ('k', 'jcp_heads_name', 'jcp_heads_seed0',
                                                                                     'jcp_heads_steps', 'hfit_arm')},
                    rules=dict(self.rules), m=dict(self.st.m), meta=self.st.meta, mode_hash=self.st.mode_hash)


class LinArm:
    kind = 'lin'

    def __init__(self, bank, spec, S):
        self.spec, self.name, self.bank = spec, spec['name'], bank
        nstep = max(T2.NSTEP, S + 1)
        self.st = P.Setting(bank, spec['name'], spec['Rp'], spec['rule'], 'frozen', forms=('LSPG',), nrec=0, M=spec.get('M'))
        self.dim = self.Rp = self.st.Rp
        self.M, self.L = self.st.M, bank.L
        self.nstep, self.S = nstep, S
        self._q = {}
        self._nl = Q.tested_value('point', bank.L)
        self._cert = T2.make_certifier(self._nl, self.Rp)
        self._jac = T2.make_jac(self._nl)

    def add_role(self, role, rule):
        self.st.add_role(self.bank, role, rule, forms=('LSPG',))

    def drop_role(self, role):
        self.st.drop_role(role)
        self.st.rules.pop(role, None)
        self.st.meta.pop(role, None)

    def data(self, role='main'):
        return self.st.data[(role, 'LSPG')]

    @property
    def rules(self):
        return self.st.rules

    @property
    def cold(self):
        return self.st.cold

    def check(self, role):
        self.st.check(role, self.bank)

    def parts(self, level='prod', record=True):
        key = (level, bool(record))
        if key not in self._q:
            budget = 600 if level == 'prod' else TIGHT_BUDGET
            self._q[key] = T2.make_split_query(Q, 'point', 'LSPG', self.Rp, self.L, self.st.trust, budget=budget,
                                               nrec=self.S + 1 if record else 0, nstep=self.nstep)
        return self._q[key]

    def run(self, role, scheme, level, u0, nu, dt, record=True):
        self.check(role)
        q = self.parts(level, record)
        return q['fused'](u0, nu, self.data(role), self.cold, sched(scheme, dt, level))

    def cert(self, states, nu, role, sch):
        return self._cert(jnp.asarray(states), nu, self.data(role), sch)

    def coef(self, W):
        return np.asarray(W)

    def nl(self, c, role):
        return np.asarray(self._nl(jnp.asarray(c), self.data(role)))

    def jac(self, w, co, dt, nu, role='main'):
        return np.asarray(self._jac(jnp.asarray(w), jnp.asarray(co), dt, nu, self.data(role)))

    def describe(self):
        return dict(kind='lin', Rp=self.Rp, M=self.M, trust=self.st.trust, rules=dict(self.rules), m=dict(self.st.m),
                    meta=self.st.meta, mode_hash=self.st.mode_hash)


def make_arm(bank, spec, head_cache, S=50):
    """S = number of time steps of the schedules the arm will run (all lane schedules use dt0: S = 50)."""
    if spec['kind'] == 'head':
        return HeadArm(bank, spec, head_cache, S)
    assert spec['kind'] == 'lin'
    return LinArm(bank, spec, S)


def ic_ok(v):
    """DESIGN A1.3 initial-fit veto: a head run whose initial fit exits neither stationary (4) nor by residual (1) is
    invalid. Linear runs have a closed-form initial projection (always ok)."""
    st = v['stats']
    if 'ic_reason' not in st:
        return True
    return int(np.asarray(st['ic_reason'])) in (1, 4)


def run_valid(v):
    return bool(P.run_valid(v) and ic_ok(v))
