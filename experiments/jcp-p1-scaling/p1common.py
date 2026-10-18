"""jcp-p1-scaling 2D shared setup, adapted from jcp-combined `c2common.py` (copied as c2common_orig.py for provenance).

Changes against c2common (DESIGN 2):
  * explicit ordered mode lists: 'frozen' = modes_lean(1024, M) reused at every mesh (eigenvalues and sine tables
    recomputed at the mesh), 'nominal' = modes_lean(L, M) (the builder's per-mesh list, "deployment policy");
    operators are built once per (mesh, policy) for the 2048-mode list and a setting takes the prefix M = 4 r;
  * every setting carries metadata (L, mesh hash, mode-list hash, rule sha256) and the query wrapper refuses a table
    whose metadata differs from the bank it runs on (wrong-mesh control, DESIGN 6);
  * bank edges are given explicitly (combined's [0,128,256,384,512], or the prefix-retaining [0,128,256,384]);
  * queries are the split query of p1core (init / evolve / decode / fused), with optional internal-state recording;
  * no cohort other than dev6 / val32 can be generated (asserted).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import p1core as T2
import jax
import jax.numpy as jnp

Q = T2.load_qcore()
import qstudy as QS  # noqa: E402  (vendored: burn, clean, sha_file)

DT0 = .005
FROZEN_L = 1024
MMAX = 2048
NOUT = 6


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def tol_of(form, level):
    """(gtol, tolf) per form and level (combined c2common.tol_of, unchanged)."""
    if form == 'LSPG':
        return {'prod': (1e-3, 1e-9), 'tight': (1e-7, 0.), 'tighter': (1e-8, 0.)}[level]
    return {'prod': (0., 1e-10), 'tight': (0., 1e-13), 'tighter': (0., 1e-15)}[level]


def gpu_info():
    try:
        return subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.total', '--format=csv,noheader'],
                                       text=True).strip().splitlines()
    except Exception:  # noqa: BLE001
        return []


def check_env(cfg):
    if not cfg.get('allow_cpu_smoke'):
        assert jax.default_backend() == 'gpu', jax.default_backend()
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    assert jax.config.jax_enable_x64
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)


def mem():
    try:
        s = jax.devices()[0].memory_stats() or {}
        return dict(in_use=int(s.get('bytes_in_use', -1)), peak=int(s.get('peak_bytes_in_use', -1)),
                    limit=int(s.get('bytes_limit', -1)))
    except Exception:  # noqa: BLE001
        return dict(in_use=-1, peak=-1, limit=-1)


# ------------------------------------------------------------------ cohorts ----

def case_list(cfg):
    """[(cohort, case, physical descriptor)] for dev6 then val32 (only these two cohorts exist here)."""
    train = Q.e.params_draw(0, 128)
    out = []
    for coh in ('dev6', 'val32'):
        ph = Q.cohort(coh)
        for t in train:
            assert not any(np.allclose(t, s) for s in ph), f'{coh} overlaps training'
        for c in range(len(ph)):
            out.append((coh, c, ph[c]))
    return out


def cohort_hashes():
    return {coh: sha(Q.cohort(coh)) for coh in ('dev6', 'val32')}


def select(cases, which):
    """Named case sets: all38, sub16 (dev6 + val32[0:10]), dev6, dev2 (smoke: dev6 cases 0-1), or an explicit list."""
    idx = {('dev6', c): i for i, (coh, c, _) in enumerate(cases) if coh == 'dev6'}
    idx.update({('val32', c): i for i, (coh, c, _) in enumerate(cases) if coh == 'val32'})
    if which == 'all38':
        keys = [(coh, c) for coh, c, _ in cases]
    elif which == 'sub16':
        keys = [('dev6', c) for c in range(6)] + [('val32', c) for c in range(10)]
    elif which == 'dev6':
        keys = [('dev6', c) for c in range(6)]
    elif which == 'dev2':
        keys = [('dev6', 0), ('dev6', 1)]
    else:
        keys = [tuple(k) for k in which]
    return [idx[k] for k in keys]


# ------------------------------------------------------------------ modes ----

def lam_of(L, kx, ky):
    kx, ky = np.asarray(kx, float), np.asarray(ky, float)
    return 4 * L ** 2 * (np.sin(np.pi * kx / (2 * L)) ** 2 + np.sin(np.pi * ky / (2 * L)) ** 2)


def mode_list(policy, L, M=MMAX):
    """Ordered (kx, ky) of the M test modes under a policy at mesh L."""
    src = FROZEN_L if policy == 'frozen' else L
    assert policy in ('frozen', 'nominal')
    kx, ky, _ = Q.H.modes_lean(src, M)
    return np.asarray(kx), np.asarray(ky)


def list_hash(kx, ky):
    return sha(np.stack([np.asarray(kx, np.int64), np.asarray(ky, np.int64)]))


def symdiff(L, M):
    """Set difference of the nominal list at L against the frozen list (out, in), and whether the order is equal."""
    fx, fy = mode_list('frozen', L, M)
    nx, ny = mode_list('nominal', L, M)
    F, N = set(zip(fx.tolist(), fy.tolist())), set(zip(nx.tolist(), ny.tolist()))
    return dict(out=sorted(F - N), into=sorted(N - F), order_equal=bool(np.array_equal(fx, nx) and np.array_equal(fy, ny)))


# ------------------------------------------------------------------ bank ----

class Bank:
    """Frozen model, rotated bank on the L-mesh with explicit column edges; operators per mode policy (2048 modes)."""

    def __init__(self, L, edges, nrb=None):
        self.L = int(L)
        self.mdl = mdl = Q.Model(ckpt=T2.CKPT)
        assert mdl.R == 512 and mdl.K == 16
        self.edges = list(edges)
        assert self.edges[0] == 0 and self.edges == sorted(self.edges) and self.edges[-1] <= mdl.R
        t0 = time.perf_counter()
        self.Grot = Q.build_rotated_bank(mdl, self.L, self.edges, nrb=nrb)
        self.build_seconds = time.perf_counter() - t0
        self.mesh_hash = sha(np.asarray([self.L], np.int64))
        self._ops, self._cold = {}, {}
        Zsub = mdl.Zold[::max(1, len(mdl.Zold) // 8192)]
        self.Hsub = np.asarray(jax.jit(jax.vmap(lambda z: Q.sc.head(mdl.params, z)))(jnp.asarray(Zsub)))
        self.bytes = int(sum(int(np.prod(b.shape)) * 8 for rb in self.Grot for b in rb))

    def ops(self, policy):
        if policy not in self._ops:
            t0 = time.perf_counter()
            kx, ky = mode_list(policy, self.L)
            lam = lam_of(self.L, kx, ky)
            if policy == 'nominal':      # builder parity: modes_lean's own eigenvalues
                assert np.array_equal(lam, Q.H.modes_lean(self.L, MMAX)[2])
            sx, sy = (jnp.asarray(t_) for t_ in Q.H.sine_tables(self.L, kx, ky))
            Arot = Q.BK.project_nested(self.Grot, self.edges, self.edges[-1], sx, sy, self.L)
            self._ops[policy] = jax.block_until_ready(dict(kx=kx, ky=ky, lam=jnp.asarray(lam), Arot=Arot,
                                                           seconds=time.perf_counter() - t0))
            del sx, sy
        return self._ops[policy]

    def cold_trust(self, Rp):
        if Rp not in self._cold:
            trust, _ = self.mdl.trust_linear(Rp)
            cold = Q.build_cold_linear(self.mdl, Rp, self.Hsub @ self.mdl.Lrot[:Rp].T)
            self._cold[Rp] = (cold, trust)
        return self._cold[Rp]


def rule_points(rule):
    X, w = Q.offmesh_rule(rule)
    return X, w, sha(np.concatenate([X.ravel(), w]))


class Setting:
    """One configuration family (r, policy, main rule) on a bank, with extra rule roles (q+, q++, controls).

    data[(role, form)] are jit arguments; meta[role] is the identity the query wrapper checks."""

    def __init__(self, bank, name, Rp, rule, policy='frozen', roles=None, forms=('LSPG',), nrec=0, budget=600, mutation=None):
        self.name, self.Rp, self.policy = name, int(Rp), policy
        self.M = 4 * self.Rp
        self.L = bank.L
        assert self.Rp in bank.edges, (self.Rp, bank.edges)
        ops = bank.ops(policy)
        self.kx, self.ky = ops['kx'][:self.M], ops['ky'][:self.M]
        self.mode_hash = list_hash(self.kx, self.ky)
        self.base = dict(A=ops['Arot'][:self.M, :self.Rp], lam=ops['lam'][:self.M],
                         G=Q.BK.prefix(bank.Grot, bank.edges, self.Rp))
        self.cold, self.trust = bank.cold_trust(self.Rp)
        self.rules = {'main': rule}
        self.rules.update(roles or {})
        self.m, self.data, self.meta, self.table_bytes = {}, {}, {}, {}
        for role, rl in self.rules.items():
            self.add_role(bank, role, rl, forms)
        self.forms = forms
        self.q = {f: T2.make_split_query(Q, 'point', f, self.Rp, self.L, self.trust, nrec=nrec, budget=budget,
                                         mutation=mutation)
                  for f in forms}
        self.q0 = {f: T2.make_split_query(Q, 'point', f, self.Rp, self.L, self.trust, nrec=0, budget=budget,
                                          mutation=mutation)
                   for f in forms}
        self.nrec = nrec

    def add_role(self, bank, role, rl, forms=('LSPG',)):
        X, w, rsha = rule_points(rl)
        b0 = mem()['in_use']
        blk = Q.offmesh_data(bank.mdl, self.Rp, X, w, self.L, self.kx, self.ky, 'point')
        self.m[role] = len(X)
        self.rules[role] = rl
        for form in forms:
            self.data[(role, form)] = T2.form_data(self.base, blk, form)
        self.table_bytes[role] = mem()['in_use'] - b0
        self.meta[role] = dict(L=self.L, mesh_hash=bank.mesh_hash, mode_hash=self.mode_hash, rule=rl, rule_sha256=rsha,
                               m=len(X), Rp=self.Rp, M=self.M, policy=self.policy)
        del blk

    def drop_role(self, role):
        for k in [k for k in self.data if k[0] == role]:
            del self.data[k]

    def check(self, role, bank):
        """Wrong-mesh / wrong-list guard (DESIGN 6): the table's metadata must match the bank it runs on."""
        mt = self.meta[role]
        if mt['L'] != bank.L or mt['mesh_hash'] != bank.mesh_hash:
            raise ValueError(f'table built for L={mt["L"]} fed to a bank on L={bank.L}')
        ops = bank.ops(self.policy)
        if mt['mode_hash'] != list_hash(ops['kx'][:self.M], ops['ky'][:self.M]):
            raise ValueError('table mode list differs from the bank operators of its policy')

    def sched(self, form, scheme, dt, level):
        g, t_ = tol_of(form, level)
        return T2.sched(scheme, dt, g, t_)


def run_query(bank, st, role, form, scheme, dt, level, u0, nu, record=True):
    """End-to-end fused query, guarded by the table metadata check."""
    st.check(role, bank)
    q = (st.q if record else st.q0)[form]
    return q['fused'](u0, nu, st.data[(role, form)], st.cold, st.sched(form, scheme, dt, level))


def restrict(fields, L):
    s = L // 256
    assert s * 256 == L
    return np.asarray(fields[:, ::s, ::s])


def fdist(fa, fb, n0):
    """Scoring-lattice relative distance: max over the five evolved outputs of ||fa_k - fb_k|| / ||u0||_257."""
    return max(float(np.linalg.norm(fa[k] - fb[k])) / n0 for k in range(1, NOUT))


def run_valid(v):
    """Q1 validity of one run (DESIGN A1.4, the common-brief contract): finite outputs, no non-finite step, no
    damping-saturation exit (reason 3; a non-finite residual at step entry is also reason 3). Budget exits (reason 0)
    are not invalid by themselves; they are counted (budget_exits) and judged by the certified gamma."""
    st = v['stats']
    ex = np.asarray(st['exits'])
    W = np.asarray(v['W'])
    return bool(np.isfinite(W).all() and ex[3] == 0)


def budget_exits(v):
    return int(np.asarray(v['stats']['exits'])[0])


def gcert_ok(g_cert, g_rec):
    """G-cert (A1.3): |gamma_cert - gamma_rec| <= 1e-7 gamma_rec + 1e-12 on every step."""
    g_cert, g_rec = np.asarray(g_cert), np.asarray(g_rec)
    return bool(np.all(np.abs(g_cert - g_rec) <= 1e-7 * np.abs(g_rec) + 1e-12))


def rho(a, b):
    """A2.4: ||a - b|| / ||b||; both zero -> nan (trivial, excluded and counted); b = 0 != a -> +inf; non-finite -> inf."""
    a, b = np.asarray(a), np.asarray(b)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float('inf')
    nb = float(np.linalg.norm(b))
    if nb == 0.:
        return float('nan') if float(np.linalg.norm(a)) == 0. else float('inf')
    return float(np.linalg.norm(a - b) / nb)
