"""jcp-speed3d shared setup: rules, per-(setting, rule) data, the solver specs and their end-to-end queries.

Spec tuples (spec_key gives the string form used everywhere):
  ('fs',  scheme, dt, K)          Step 2 fs3 (byte-identical copy), a_eff adaptive start then K full-Jacobian sweeps
  ('fsp', scheme, dt, K)          fs3p with the cubic + IMEX predictor candidates (DESIGN A1.3)
  ('qn',  scheme, dt, kref[, mut]) qn3 QN-k (kref 1, 4 or 0 = infinity); mut in {q0, noJuT} only for controls
  ('ad',  scheme, dt, level)      Step 2 verified adaptive LSPG solve (t2core), level tight | tighter
Every query is batch-first: u0 (Bt, Nin), nu (Bt,) -> dict(W (6, Bt, R'), W_all (steps+1, Bt, R') [not ad],
w0 (Bt, R'), fields (Bt, 6, 63^3) when decode=True, per-step records). Paths: 'mesh' (Step 2: least-squares projection on
the 65 mesh, input = interior mesh field) and 'mf' (DESIGN H3.2: quadrature projection on the setting's rule, input =
u0 at the rule points). Decode is always onto the 63^3 scoring lattice (at n = 65 the lattice is the interior mesh).
All large arrays are explicit jit arguments.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import c3common as C3
from c3common import C, TB, OM
import fs3
import fs3p
import qn3
import t2core as T2

HERE = Path(__file__).resolve().parent
DT0 = 0.01
RULE_FILES = [(HERE / 'vendor' / 'quad3d' / 'rules' / 'rules.npz',
               '099a79976d03e993dc3fbdcc153a57dae0a2dbc5f917d7ca3674bef28daf859b'),
              (HERE / 'rules3d' / 'rules_extra.npz', '8deacf019839b8cf42993d5d8926126541db8152b811f411eceeb69940bb38cf')]
DTS = (0.01, 0.0125, 0.05 / 3, 0.025)


def load_rules(names=None):
    rules = {}
    for p, want in RULE_FILES:
        assert C3.sha_file(p) == want, p
        z = np.load(p)
        for k in z.files:
            if k.endswith('_X'):
                nm = k[:-2]
                if names is not None and nm not in names:
                    continue
                X, w = np.asarray(z[k]), np.asarray(z[nm + '_w'])
                assert nm not in rules or (np.array_equal(rules[nm][0], X) and np.array_equal(rules[nm][1], w)), nm
                rules[nm] = (X, w)
    if names is not None:
        missing = set(names) - set(rules)
        assert not missing, missing
    return rules


def spec_key(sp):
    t = sp[0]
    if t in ('fs', 'fsp'):
        return f'{t}|{sp[1]}|{sp[2]:.8g}|K{sp[3]}'
    if t == 'qn':
        k = 'inf' if sp[3] == 0 else str(sp[3])
        return f'qn|{sp[1]}|{sp[2]:.8g}|k{k}' + (f'|{sp[4]}' if len(sp) > 4 else '')
    if t == 'ad':
        return f'ad|LSPG|{sp[1]}|{sp[2]:.8g}|{sp[3]}'
    raise KeyError(sp)


def parse_key(k):
    p = k.split('|')
    if p[0] in ('fs', 'fsp'):
        return (p[0], p[1], float(p[2]), int(p[3][1:]))
    if p[0] == 'qn':
        kr = 0 if p[3] == 'kinf' else int(p[3][1:])
        return ('qn', p[1], float(p[2]), kr) + ((p[4],) if len(p) > 4 else ())
    if p[0] == 'ad':
        return ('ad', p[2], float(p[3]), p[4])
    raise KeyError(k)


def nsteps(dt):
    s, kp = int(round(.25 / dt)), int(round(.05 / dt))
    assert abs(s * dt - .25) < 1e-12 and abs(kp * dt - .05) < 1e-12, dt
    return s, kp


# ------------------------------------------------------------------------------------------ data per setting/rule
class Setting:
    """One (bank, R') at mesh 65 with its trust, adaptive start a and lean tables; rule data cached on demand."""

    def __init__(self, cfg_s, banks, tables, rules, trust_fraction=0.05):
        self.name, self.bank, self.Rp, self.a = cfg_s['name'], cfg_s['bank'], int(cfg_s['Rp']), int(cfg_s['a'])
        bk = banks[self.bank]
        self.bk, self.tb = bk, tables[self.bank]
        self.trust = float(trust_fraction * bk['spread'][str(self.Rp)])
        self.lean = self.tb.lean(self.Rp)                              # GTb (mesh = lattice rows), L, A, lam
        self.rules = rules
        self.M = self.tb.M_of[self.Rp]
        self._d = {}
        qa = qn3.qr_data(self.lean['A'])
        self.qr = jax.tree_util.tree_map(jax.block_until_ready, qa)

    def data(self, rule):
        if rule not in self._d:
            X, w = self.rules[rule]
            d = self.tb.rule_data(self.Rp, X, w)                        # lean + B, D, P
            d = dict(d, **self.qr)
            d['hmask'] = (d['lam'] > jnp.median(d['lam'])).astype(jnp.float64)       # t2core LSPG form data
            Bw = d['B'] * jnp.asarray(w)[:, None]
            Gq = d['B'].T @ Bw
            ev = np.linalg.eigvalsh(np.asarray(Gq))
            d['Bw'], d['Lq'] = Bw, jnp.linalg.cholesky(Gq)
            self._d[rule] = (jax.tree_util.tree_map(jax.block_until_ready, d),
                             dict(cond=float(ev[-1] / ev[0]) if ev[0] > 0 else float('inf'), m=int(len(w))))
        return self._d[rule][0]

    def mf_cond(self, rule):
        self.data(rule)
        return self._d[rule][1]

    def drop(self, rule):
        self._d.pop(rule, None)


# ------------------------------------------------------------------------------------------------- queries
def project_mesh(u0, d):
    """Step 2 least-squares projection on the mesh, per member: (w0 (Bt, R'), ||s0|| (Bt,))."""
    def one(u):
        w0, s0 = TB.project(u, d)
        return w0, jnp.linalg.norm(s0)
    if u0.shape[0] == 1:                       # single query: no vmap, so Step 2's arithmetic is reproduced bitwise
        w0, s = one(u0[0])
        return w0[None], s[None]
    return jax.vmap(one)(u0)


def project_mf(u0X, d, n=65):
    """Quadrature projection on the rule: w0 = (B^T W B)^{-1} B^T W u0(X). The adaptive-start residual tolerance uses the
    mesh-equivalent energy scale (n-1)^{3/2} ||Lq^{-1} B^T W u0(X)|| (never active in practice: 1e-12)."""
    t = u0X @ d['Bw']                                                      # (Bt, R')
    s = jax.scipy.linalg.solve_triangular(d['Lq'], t.T, lower=True)
    w0 = jax.scipy.linalg.solve_triangular(d['Lq'].T, s, lower=False).T
    return w0, (n - 1) ** 1.5 * jnp.linalg.norm(s, axis=0)


def decode_lattice(W, d):
    """W (6, Bt, R') -> (Bt, 6, 63^3) on the lattice (mesh rows of n = 65)."""
    T_, Bt, R = W.shape
    F = TB.decode(jnp.transpose(W, (1, 0, 2)).reshape(Bt * T_, R), d)
    return F.reshape(Bt, T_, -1)


def make_query(sp, st, path='mesh', decode=True, n=65):
    """Batch-first end-to-end query for spec sp on Setting st. Returns a jitted f(u0, nu, data)."""
    kind, scheme, dt = sp[0], sp[1], sp[2]
    steps, keep = nsteps(dt)
    proj = project_mesh if path == 'mesh' else (lambda u, d: project_mf(u, d, n))
    if kind in ('fs', 'fsp'):
        K = sp[3]
        a_eff = fs3.a_effective(scheme, dt, DT0, st.a, steps)
        mod = fs3 if kind == 'fs' else fs3p
        kw = dict(extra=True) if kind == 'fsp' else {}
        ev = mod.make_fs_evolve(st.Rp, OM.contract_offmesh, scheme, dt, K, 1e-3, st.trust, a_eff, steps,
                                unroll=steps <= 50, **kw)

        def evolve(w0, nu, tol, d):
            tol = 1e-12 * tol                          # fs3.make_fs_query: tol_ad = 1e-12 ||s0||
            if w0.shape[0] == 1:
                o = ev(w0[0], nu[0], tol[0], d)
                return {k: v[:, None] for k, v in o.items()}
            o = jax.vmap(lambda w, v, t: ev(w, v, t, d))(w0, nu, tol)
            o = {k: jnp.moveaxis(v, 0, 1) for k, v in o.items()}           # (steps(+1), Bt, ...)
            return o
    elif kind == 'qn':
        kr = sp[3] if sp[3] != 0 else math.inf
        mut = sp[4] if len(sp) > 4 else None
        ev = qn3.make_qn_evolve(st.Rp, scheme, dt, steps, st.trust, kr, mutation=mut)

        def evolve(w0, nu, tol, d):
            return ev(w0, nu, d)
    elif kind == 'ad':
        g, t_ = C3.tol_of('LSPG', sp[3])
        ev = T2.make_evolve(C3.nl3, C3.nlJ3, 'LSPG', st.Rp, st.trust)
        sch = T2.sched(scheme, dt, g, t_)

        def evolve(w0, nu, tol, d):
            if w0.shape[0] == 1:
                W, stt = ev(w0[0], nu[0], tol[0], d, sch)
                return dict(W=W[:, None], stats=jax.tree_util.tree_map(lambda x: x[None], stt))
            W, stt = jax.vmap(lambda w, v, s: ev(w, v, s, d, sch))(w0, nu, tol)
            return dict(W=jnp.moveaxis(W, 0, 1), stats=stt)
    else:
        raise KeyError(sp)

    def query(u0, nu, d):
        w0, scale = proj(u0, d)
        o = evolve(w0, nu, scale, d)
        if kind != 'ad':
            o['W'] = o['W_all'][::keep]
        o['w0'] = w0
        if decode:
            o['fields'] = decode_lattice(o['W'], d)
        return o
    return jax.jit(query)


# ------------------------------------------------------------------------------------- validity and records
def run_valid(sp, o, b):
    """Per-run validity of member b (E-a). fs/fsp: finite, no reason-3 step, no damping saturation. qn: finite, no
    reason-3 step. ad: every step verified (nfail == 0) and finite."""
    W = np.asarray(o['W'][:, b])
    if not np.isfinite(W).all():
        return False
    if sp[0] == 'ad':
        return int(np.asarray(o['stats']['nfail'])[b]) == 0
    rs = np.asarray(o['reason'][:, b])
    if (rs == 3).any():
        return False
    if sp[0] in ('fs', 'fsp') and int(np.asarray(o['sat'][:, b]).sum()) > 0:
        return False
    return True


def step_record(sp, o, b):
    if sp[0] == 'ad':
        stt = o['stats']
        k = int(np.asarray(stt['steps'])[b])
        return dict(nfail=int(np.asarray(stt['nfail'])[b]), it_sum=int(np.asarray(stt['it_sum'])[b]), steps=k,
                    step_it=np.asarray(stt['step_it'])[b, :k].tolist(), step_gn=np.asarray(stt['step_gn'])[b, :k].tolist())
    keys = ('reason', 'it', 'rej', 'pred', 'sat', 'gn', 'rn', 'lam') if sp[0] in ('fs', 'fsp') else \
        ('reason', 'n_ref', 'n_exec', 'q', 'n_bt', 'n_clip', 'n_mu', 'refresh', 'pred', 'gam', 'rn')
    return {k: np.asarray(o[k][:, b]).tolist() for k in keys}


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def lattice_dist(Wa, Wb, G, n0):
    """DESIGN A2.3: per case max over evolved output times of ||(Wa - Wb)_k G^T|| / n0. Wa, Wb (cases, 6, R');
    G (63^3, R') or the transposed rows (R', 63^3) as stored in the lean tables (detected by shape)."""
    D = np.asarray(Wa) - np.asarray(Wb)
    Gm = np.asarray(G)
    F = D @ Gm if Gm.shape[0] == D.shape[-1] else D @ Gm.T
    d = np.linalg.norm(F, axis=-1) / np.asarray(n0)[:, None]
    return d[:, 1:].max(1)
