"""jcp-combined 2D shared setup (used by c2grid.py and c2bud.py): cohorts, references, frozen model, rotated bank with
nested column blocks at every R' of the config, operators at M_max (nested prefixes asserted, as C1 w2d), per-setting
base data, rule blocks, form data and queries (t2core.make_query, the C2 generic LMM query).

A setting is {name, Rp, rule} with M = 4 R'. Rules are off-mesh point rules (vendor qcore.offmesh_rule names).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import t2core as T2
import jax
import jax.numpy as jnp

Q = T2.load_qcore()
import qstudy as QS  # noqa: E402  (vendored: burn, clean, sha_file)

DT0 = .005
EXPECTED = {'test64': 'cd058fd2a297c20296b897e54cb186033a44a8db9527d98dcb2c2dad6044527c'}
REFSRC_NAME = '2026-10-01-quadrature-study/experiments/quadrature-study/runs/refdv/archive/output'


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def tol_of(form, level):
    """(gtol, tolf) per form and level (C2 t2run.tol_of; A1.10)."""
    if form == 'LSPG':
        return {'prod': (1e-3, 1e-9), 'tight': (1e-7, 0.), 'tighter': (1e-8, 0.)}[level]
    return {'prod': (0., 1e-10), 'tight': (0., 1e-13), 'tighter': (0., 1e-15)}[level]


def gpu_info():
    try:
        return subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid', '--format=csv,noheader'],
                                       text=True).strip().splitlines()
    except Exception:  # noqa: BLE001
        return []


def check_env(cfg):
    if not cfg.get('allow_cpu_smoke'):
        assert jax.default_backend() == 'gpu', jax.default_backend()
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    assert jax.config.jax_enable_x64
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)


def load_cases(cfg, rep):
    train = Q.e.params_draw(0, 128)
    allc = {k: Q.cohort(k) for k in ('dev6', 'val32', 'test64')}
    for k1 in allc:
        assert not any(np.allclose(t, s) for t in train for s in allc[k1]), f'{k1} overlaps training'
    cases = []
    for coh in cfg['cohorts']:
        assert coh in ('dev6', 'val32'), 'only development/validation cohorts are used in this lane'
        ph = allc[coh]
        h = sha(ph)
        rep['cohorts'][coh] = dict(n=len(ph), physical_sha256=h, physical=ph.tolist())
        sub = cfg.get('case_subset', {}).get(coh)
        for c in (range(len(ph)) if sub is None else sub):
            cases.append((coh, int(c), ph[c]))
    return cases


def load_refs(cfg, cases, rep):
    rdir = Path(cfg['refs'])
    man = json.loads((rdir / 'result.json').read_text())
    assert man.get('complete') and man.get('all_accepted'), 'reference job incomplete or rejected'
    want = cfg['ref_contract']
    assert int(man['config']['mesh']) == want['mesh']
    for tag, w_ in want['refs'].items():
        got = man['config']['refs'][tag]
        assert all(abs(float(got[k_]) - float(w_[k_])) <= 1e-15 * max(1., abs(float(w_[k_]))) for k_ in w_), (tag, got)
    idx = {(x['cohort'], x['case'], x['ref']): x for x in man['cases']}
    refs = {}
    for coh, c, ph in cases:
        if not cfg.get('local_smoke_waives_cohort_hash'):       # C2 A5.2
            assert man['cohort_sha256'][coh] == rep['cohorts'][coh]['physical_sha256'], coh
        for tag in ('ST', 'S'):
            ent = idx[(coh, c, tag)]
            assert ent['accepted'] and ent['mesh'] == want['mesh'] and abs(ent['dt'] - want['refs'][tag]['dt']) <= 1e-15
            r = np.load(rdir / f'ref_{tag}_{coh}_{c:03d}.npz')['f257']
            assert r.shape == (6, 257, 257) and np.isfinite(r).all() and sha(r) == ent['f257_sha256'], (coh, c, tag)
            refs[(coh, c, tag)] = r
            rep['references'][f'{tag}|{coh}|{c}'] = ent['f257_sha256']
    rep['references_job'] = dict(dir=str(rdir), job_id=man.get('job_id'), commit=man.get('commit'),
                                 manifest_sha256=QS.sha_file(rdir / 'result.json'))
    return refs


class Bank:
    """Frozen model, rotated bank on the mesh with edges at every R' of the settings, operators at M_max."""

    def __init__(self, cfg, L):
        self.L = L
        self.mdl = mdl = Q.Model(ckpt=T2.CKPT)
        assert mdl.R == 512 and mdl.K == 16
        Rps = sorted({int(s['Rp']) for s in cfg['settings']})
        self.edges = sorted(set([0] + Rps + [mdl.R]))
        t0 = time.perf_counter()
        self.Grot = Q.build_rotated_bank(mdl, L, self.edges, nrb=cfg.get('bank_blocks'))
        self.Mmax = max(4 * r for r in Rps)
        self.ops = Q.operators(self.Grot, self.edges, L, self.Mmax)
        lam = np.asarray(self.ops['lam'])
        for r in Rps:                           # nested modes: the M-prefix of the M_max table is modes_lean(L, M)
            kx, ky, lm = Q.H.modes_lean(L, 4 * r)
            assert np.array_equal(kx, self.ops['kx'][:4 * r]) and np.array_equal(ky, self.ops['ky'][:4 * r])
            assert np.array_equal(lm, lam[:4 * r])
        self.seconds = time.perf_counter() - t0
        Zsub = mdl.Zold[::max(1, len(mdl.Zold) // 8192)]
        self.Hsub = np.asarray(jax.jit(jax.vmap(lambda z: Q.sc.head(mdl.params, z)))(jnp.asarray(Zsub)))

    def base(self, Rp):
        M = 4 * Rp
        return dict(A=self.ops['Arot'][:M, :Rp], lam=self.ops['lam'][:M], G=Q.BK.prefix(self.Grot, self.edges, Rp))

    def cold_trust(self, Rp):
        trust, _ = self.mdl.trust_linear(Rp)
        cold = Q.build_cold_linear(self.mdl, Rp, self.Hsub @ self.mdl.Lrot[:Rp].T)
        return cold, trust

    def blocks(self, Rp, rule):
        M = 4 * Rp
        X, w = Q.offmesh_rule(rule)
        return Q.offmesh_data(self.mdl, Rp, X, w, self.L, self.ops['kx'][:M], self.ops['ky'][:M], 'point'), len(X)


class Setting:
    """One (R', rule) setting: base data, rule blocks (several rules allowed), form data, compiled queries."""

    def __init__(self, bank, spec, rules=None):
        self.name, self.Rp = spec['name'], int(spec['Rp'])
        self.M = 4 * self.Rp
        self.L = bank.L
        self.base = bank.base(self.Rp)
        self.cold, self.trust = bank.cold_trust(self.Rp)
        self.rules = {'main': spec['rule']}
        self.rules.update(rules or {})
        self.m, self.data = {}, {}
        for role, rule in self.rules.items():
            blk, m = bank.blocks(self.Rp, rule)
            self.m[role] = m
            for form in ('LSPG', 'GAL'):
                self.data[(role, form)] = T2.form_data(self.base, blk, form)
            del blk
        self.q = {form: T2.make_query(Q, 'point', form, self.Rp, self.L, self.trust) for form in ('LSPG', 'GAL')}
        self.vq, _ = Q.make_linear_query('point', self.Rp, self.L, DT0, self.trust, step_budget=600, gtol=1e-3)
        self.vdata = dict(self.base, **{k: v for k, v in self.data[('main', 'LSPG')].items() if k in ('Gq', 'Gs', 'Psi')})

    def call(self, role, form, scheme, dt, level, u0, nu):
        g, t_ = tol_of(form, level)
        return self.q[form](u0, nu, self.data[(role, form)], self.cold, T2.sched(scheme, dt, g, t_))

    def vendor(self, u0, nu):
        return self.vq(u0, nu, self.vdata, self.cold)


def errs(fr, refs, coh, c, n0r):
    """e_ST, e_S, e_TX (max over the five evolved times) and per-time lists, on the 257^2 shared nodes."""
    rST, rS = refs[(coh, c, 'ST')], refs[(coh, c, 'S')]
    o = {}
    for tag, R_ in (('ST', rST), ('S', rS), ('TX', (16. * rST - rS) / 15.)):
        pe = [float(np.linalg.norm(fr[j] - R_[j])) / n0r for j in range(6)]
        o[f'e_{tag}_per_time'] = pe
        o[f'e_{tag}'] = max(pe[1:])
    return o
