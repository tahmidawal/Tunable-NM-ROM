"""jcp-heads shared setup for the head arms (DESIGN 1, 3), the head analogue of qualify's `p1common.Setting`.

A HeadSetting is one (head checkpoint, K, M = 4K, frozen mode list, rule) on a p1common.Bank (the incumbent's rotated
bank on the mesh, edges including 512). Its tables are jit arguments: A = Phi^T G~ (first M frozen modes, all 512
rotated columns), lam, G (the full rotated bank, for decoding), hmask, and the point-rule blocks (Gq, Gs, Psi) of each
rule role (main, q+, q++, controls). Every role carries metadata (mesh, mode-list hash, rule sha256, head sha256,
table fingerprint) and `check` refuses a table that does not match the bank it runs on (wrong-mesh control).
The linear arms use qualify's p1common.Setting unchanged.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import p1common as P
from p1common import Q, T2
import hcore as HC

HERE = Path(__file__).resolve().parent
INCUMBENT = T2.CKPT
IC_TOL = dict(tight=1e-9, tighter=1e-9)           # DESIGN A1.3 as amended by A9 (smk1: 1e-10 hit the GJ floor)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


class Head:
    """A head checkpoint (new or incumbent): folded params on the rotated basis, codes, trust, candidate codes.
    Asserts that every bank (non-head) parameter equals the incumbent bank's, so all heads share one bank."""

    def __init__(self, path, bank_model):
        self.path = Path(path)
        self.sha256 = sha_file(self.path)
        if self.path.suffix == '.npz':          # compact archive (DESIGN A5): head weights on the incumbent bank
            z = np.load(self.path)
            assert str(z['source_ckpt_sha256']) == sha_file(INCUMBENT), \
                f'{self.path.name}: compact head was trained on another bank checkpoint'
            nl_ = int(z['n_layers'])
            ck = dict(params=dict(Q.LD_host(bank_model.params), h=[(z[f'h{i}_w'], z[f'h{i}_b']) for i in range(nl_)],
                                  h_lin=z['h_lin']),
                      cfg=dict(k=int(z['k']), jcp_heads_name=str(z['name']), jcp_heads_seed0=int(z['seed0']),
                               jcp_heads_steps=int(z['steps']), hfit_arm='mid', pkl_sha256=str(z['pkl_sha256'])))
            self.Zsub = np.asarray(z['Zsub'])
            self.trust = float(z['trust'])
            self.code_std, self.code_max_dev, self.n_codes = float(z['code_std']), float(z['code_max_dev']), int(z['n_codes'])
            self.Z = None
            self.K = int(self.Zsub.shape[1])
        else:
            ck = pickle.load(open(self.path, 'rb'))
            self.Z = np.asarray(ck['Z_tr'])
            self.K = int(self.Z.shape[1])
            self.Zsub = HC.zsub_of(self.Z)
            self.trust = HC.trust_of(self.Z)
            zn = np.linalg.norm(self.Z - self.Z.mean(0), axis=1)
            self.code_std, self.code_max_dev, self.n_codes = float(np.std(self.Z)), float(zn.max()), len(self.Z)
        self.cfg = {k: v for k, v in ck.get('cfg', {}).items() if k != 'hfit_pick'}
        params = jax.tree_util.tree_map(jnp.asarray, Q.LD_host(ck['params']))
        ref = bank_model.params
        for k_ in ref:
            if k_ in ('h', 'h_lin', 'hB'):
                continue
            la, lb = jax.tree_util.tree_leaves(ref[k_]), jax.tree_util.tree_leaves(params[k_])
            assert len(la) == len(lb) and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb)), \
                f'{self.path.name}: bank parameter {k_} differs from the incumbent bank'
        assert np.asarray(params['h_lin']).shape == (self.K, 512)
        self.params = params
        self.pf = Q.BK.fold_head(params, bank_model.Lrot, 512)
        self._cold = None

    def cold(self, bank_model):
        if self._cold is None:
            self._cold = Q.build_cold_head(bank_model, self.pf, self.Zsub)
        return self._cold


class HeadSetting:
    """One head configuration family on a bank: head, M = 4K frozen modes, main rule plus extra roles."""

    def __init__(self, bank, name, head, rule, roles=None, nrec=0, budget=600, mutation=None, nstep=T2.NSTEP,
                 ic_budget=400, ic_gtol=1e-6, M=None):
        self.name, self.head, self.K = name, head, head.K
        self.M = int(M or 4 * head.K)
        self.L = bank.L
        self.Rp = 512
        assert 512 in bank.edges
        ops = bank.ops('frozen')
        self.kx, self.ky = ops['kx'][:self.M], ops['ky'][:self.M]
        self.mode_hash = P.list_hash(self.kx, self.ky)
        self.base = dict(A=ops['Arot'][:self.M, :512], lam=ops['lam'][:self.M], G=Q.BK.prefix(bank.Grot, bank.edges, 512))
        self.cold = head.cold(bank.mdl)
        self.trust = head.trust
        self.rules = {'main': rule}
        self.rules.update(roles or {})
        self.m, self.data, self.meta, self.table_bytes = {}, {}, {}, {}
        for role, rl in self.rules.items():
            self.add_role(bank, role, rl)
        self.nstep, self.nrec, self.budget, self.mutation = nstep, nrec, budget, mutation
        self.ic = dict(prod=(ic_budget, ic_gtol))
        self._q = {}

    def query(self, level='prod', record=True):
        """Compiled split query per (level, recording). Levels (DESIGN 3.1): prod = deployed (step budget
        `budget`, initial fit budget 400 / gtol 1e-6); tight / tighter = step budget 2000 and the initial fit tightened
        from the same start (budget 2000, gtol 1e-9 for both, DESIGN A1.3 / A9). The step tolerances come from the schedule (P.tol_of)."""
        key = (level, bool(record))
        if key not in self._q:
            if level == 'prod':
                budget, (icb, icg) = self.budget, self.ic['prod']
            else:
                budget, icb, icg = 2000, 2000, IC_TOL[level]
            self._q[key] = HC.make_head_query(self.head.pf, self.K, self.L, self.trust, mutation=self.mutation,
                                              budget=budget, nrec=self.nrec if record else 0, nstep=self.nstep,
                                              ic_budget=icb, ic_gtol=icg)
        return self._q[key]

    def add_role(self, bank, role, rl):
        X, w, rsha = P.rule_points(rl)
        b0 = P.mem()['in_use']
        blk = Q.offmesh_data(bank.mdl, 512, X, w, self.L, self.kx, self.ky, 'point')
        self.m[role] = len(X)
        self.rules[role] = rl
        self.data[role] = T2.form_data(self.base, blk, 'LSPG')
        self.table_bytes[role] = P.mem()['in_use'] - b0
        self.meta[role] = dict(L=self.L, mesh_hash=bank.mesh_hash, mode_hash=self.mode_hash, rule=rl, rule_sha256=rsha,
                               m=len(X), K=self.K, M=self.M, head_sha256=self.head.sha256,
                               fingerprint=P.fingerprint(self.data[role]))
        del blk

    def drop_role(self, role):
        self.data.pop(role, None)

    def check(self, role, bank):
        mt = self.meta[role]
        if mt['L'] != bank.L or mt['mesh_hash'] != bank.mesh_hash:
            raise ValueError(f'table built for L={mt["L"]} fed to a bank on L={bank.L}')
        ops = bank.ops('frozen')
        if mt['mode_hash'] != P.list_hash(ops['kx'][:self.M], ops['ky'][:self.M]):
            raise ValueError('table mode list differs from the bank operators')
        if P.fingerprint(self.data[role]) != mt['fingerprint']:
            raise ValueError('table arrays differ from the ones built for this setting (substituted data)')
        if not np.array_equal(np.asarray(self.data[role]['lam']), np.asarray(ops['lam'][:self.M])):
            raise ValueError('table eigenvalues differ from the bank operators (wrong mesh or list)')

    def sched(self, scheme, dt, level):
        g, t_ = P.tol_of('LSPG', level)
        return T2.sched(scheme, dt, g, t_)


def run_head(bank, st, role, scheme, dt, level, u0, nu, record=True, q=None):
    """End-to-end fused head query, guarded by the table metadata check."""
    st.check(role, bank)
    assert int(round(.25 / dt)) <= st.nstep
    if record:
        assert int(round(.25 / dt)) + 1 == st.nrec, (dt, st.nrec)
    qq = q or st.query(level, record)
    return qq['fused'](u0, nu, st.data[role], st.cold, st.sched(scheme, dt, level))
