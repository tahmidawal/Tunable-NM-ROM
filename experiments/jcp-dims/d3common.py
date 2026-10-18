"""jcp-dims 3D extensions of speed3d's context (DESIGN 2.3): an explicit test count M per setting and the lane's rule
names. speed3d's Tables fixes M = complete_M(65, 4 R'); here every setting may carry 'M' (a prefix of the stable
eigenvalue order; complete_M(65, kappa R') for shell-completed lists, an exact count for the shell-breaking kappa = 1
control) and the operators are built once at the largest M asked. Nothing else of speed3d's arithmetic changes:
TablesM / SettingM are copies of c3common.Tables / s3common.Setting with M passed through (reproduction gate,
DESIGN 7: with M = complete_M(65, 4 R') they must reproduce speed3d's archived coefficients).

install(cfg) replaces c3common.Tables and s3common.Setting by these classes before s3ctx.Ctx is constructed, and
replaces s3common.load_rules by a loader that also knows the lane's rule names (rules3d_names.py).
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

import c3common as C3
import s3common as S
import qn3
import rules3d_names as RN
import offmesh as OM
from offmesh import TB

C = C3.C
DIMS_RULES = (__import__("pathlib").Path(__file__).resolve().parent / 'rules3d' / 'rules_dims.npz')
DIMS_RULES_SHA = '7b6ca36c1a521e3aeee77dd2002c6df9c1f56bba309a78bdc775ba860ed94fa1'   # rules3d/rules_dims.npz (lat262144)

_MMAX = {}                   # bank name -> largest M asked by the config (install)


class TablesM:
    """c3common.Tables with M per (R', M) request; A built at the largest M asked for this bank."""

    def __init__(self, n, bank, Rps, log=print):
        self.n, self.bank = n, bank
        self.order, self.lam_all = C.mode_order(n)
        self.M_of = {r: int(C.complete_M(n, 4 * r, self.order, self.lam_all)) for r in Rps}
        Mmax = max(max(self.M_of.values()), _MMAX.get(bank['name'], 0))
        self.Mmax = Mmax
        self.tb = TB.build_tables(n, bank['params'], bank['T'], sorted(set(Rps)), lambda r: Mmax, log=log, tensor=False)
        self.kx = self.tb['kxyz']

    def lean(self, Rp, M=None):
        M = self.M_of[Rp] if M is None else int(M)
        assert Rp <= M <= self.Mmax, (Rp, M, self.Mmax)
        tb = self.tb
        d = dict(GTb=TB.rows_prefix(tb['GTb'], Rp), L=tb['L'][:Rp, :Rp], A=tb['A'][:M, :Rp], lam=tb['lam'][:M])
        return jax.tree_util.tree_map(lambda x: jax.block_until_ready(jnp.asarray(x)), d)

    def rule_data(self, Rp, X, w, M=None):
        M = self.M_of[Rp] if M is None else int(M)
        om = OM.offmesh_tables(self.n, self.bank['params'], self.bank['T'][:, :Rp], X, w, self.kx[:M])
        return dict(self.lean(Rp, M), B=om['B'], D=om['D'], P=om['P'])


class SettingM(S.Setting):
    """s3common.Setting with an optional explicit 'M' in the setting config (default: the speed3d M)."""

    def __init__(self, cfg_s, banks, tables, rules, trust_fraction=0.05):
        self.name, self.bank, self.Rp, self.a = cfg_s['name'], cfg_s['bank'], int(cfg_s['Rp']), int(cfg_s['a'])
        bk = banks[self.bank]
        self.bk, self.tb = bk, tables[self.bank]
        self.trust = float(trust_fraction * bk['spread'][str(self.Rp)])
        self.M = int(cfg_s['M']) if cfg_s.get('M') else self.tb.M_of[self.Rp]
        self.lean = self.tb.lean(self.Rp, self.M)
        self.rules = rules
        self._d = {}
        qa = qn3.qr_data(self.lean['A'])
        self.qr = jax.tree_util.tree_map(jax.block_until_ready, qa)
        k = np.asarray(self.tb.kx[:self.M], np.int64)
        self.mode_hash = C3.sha(k)

    def data(self, rule):
        if rule not in self._d:
            X, w = self.rules[rule]
            d = self.tb.rule_data(self.Rp, X, w, self.M)
            d = dict(d, **self.qr)
            d['hmask'] = (d['lam'] > jnp.median(d['lam'])).astype(jnp.float64)
            cond = None
            if np.all(np.asarray(w) > 0):              # A1.4: the mesh-free Gram only for positive weights (never used
                Bw = d['B'] * jnp.asarray(w)[:, None]  # on the mesh path; signed Smolyak weights would make it indefinite)
                Gq = d['B'].T @ Bw
                ev = np.linalg.eigvalsh(np.asarray(Gq))
                d['Bw'], d['Lq'] = Bw, jnp.linalg.cholesky(Gq)
                cond = float(ev[-1] / ev[0]) if ev[0] > 0 else float('inf')
            self._d[rule] = (jax.tree_util.tree_map(jax.block_until_ready, d),
                             dict(cond=cond, m=int(len(w)), sha256=RN.sha_rule(X, w)))
        return self._d[rule][0]


_ORIG_LOAD = S.load_rules


def load_rules_dims(names=None):
    """Pinned speed3d files + rules_dims.npz (lat262144) + generated lane names."""
    names = sorted(names) if names is not None else None
    pinned = _ORIG_LOAD(None)
    if DIMS_RULES.exists():
        if DIMS_RULES_SHA is not None:
            assert C3.sha_file(DIMS_RULES) == DIMS_RULES_SHA, 'rules_dims.npz differs from its pin'
        z = np.load(DIMS_RULES)
        for k in z.files:
            if k.endswith('_X'):
                nm = k[:-2]
                assert nm not in pinned, nm
                pinned[nm] = (np.asarray(z[k]), np.asarray(z[nm + '_w']))
    if names is None:
        return pinned
    return RN.build(names, pinned)


def install(cfg):
    """Patch speed3d's context classes (call before s3ctx.Ctx(cfg, out))."""
    for s in cfg.get('settings', []):
        if s.get('M'):
            _MMAX[s['bank']] = max(_MMAX.get(s['bank'], 0), int(s['M']))
    C3.Tables = TablesM
    S.Setting = SettingM
    S.load_rules = load_rules_dims
