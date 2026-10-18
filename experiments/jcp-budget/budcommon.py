"""jcp-budget 2D shared setup (DESIGN 4, 5, 6, A1.3). Builds on jcp-qualify's `p1common.py` (copied unchanged: Bank, mode
lists, case lists, tolerances, validity, distances) and `budcore.py` (p1core + output count).

New here:
  * BSetting: p1common.Setting with an explicit test-space size M (default 4 r; the M-ablation uses M' = 2048) and
    lazily built rule roles that can be dropped (memory lifetime);
  * ls_init: full-grid least-squares projection of the mesh IC on the bank prefix by blocked Gram accumulation over
    the rotated bank's row blocks (A1.3; gate G-B5 orthogonality residual);
  * corner(): one B2 corner run through the split path init -> (w0 replacement) -> evolve, with the configuration tuple
    of the run (G-B6) and its validity;
  * fine-lattice helpers for B8 (decode at the 1025^2 nodes through a Bank(1024)).
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import jax
import jax.numpy as jnp

import p1common as P
from p1common import Q, DT0, sha
import budcore as BC
import qcert as QC

ING = ('Q', 'T', 'F', 'S', 'I')            # quadrature, time package, form, solver, initial condition
OTHER = {'LSPG': 'GAL', 'GAL': 'LSPG'}


def steps_keep(dt, horizon):
    s, k = int(round(horizon / dt)), int(round(.05 / dt))
    assert abs(s * dt - horizon) < 1e-12 and abs(k * dt - .05) < 1e-12, (dt, horizon)
    return s


def nout_of(horizon):
    n = int(round(horizon / .05))
    assert abs(n * .05 - horizon) < 1e-12
    return n + 1


class BSetting:
    """(bank prefix r, frozen mode list prefix M, rule roles) on one Bank; tables per (role, form)."""

    def __init__(self, bank, Rp, M=None, forms=('LSPG', 'GAL')):
        self.bank, self.Rp, self.L = bank, int(Rp), bank.L
        self.M = int(M or 4 * self.Rp)
        assert self.Rp in bank.edges and self.M <= P.MMAX
        ops = bank.ops('frozen')
        self.kx, self.ky = ops['kx'][:self.M], ops['ky'][:self.M]
        self.mode_hash = P.list_hash(self.kx, self.ky)
        self.base = dict(A=ops['Arot'][:self.M, :self.Rp], lam=ops['lam'][:self.M],
                         G=Q.BK.prefix(bank.Grot, bank.edges, self.Rp))
        self.cold, self.trust = bank.cold_trust(self.Rp)
        self.forms = forms
        self.rules, self.m, self.data, self.meta = {}, {}, {}, {}
        self._q = {}

    def add_role(self, role, rule):
        if role in self.rules and self.rules[role] == rule:
            return
        self.drop_role(role)
        X, w, rsha = P.rule_points(rule)
        blk = Q.offmesh_data(self.bank.mdl, self.Rp, X, w, self.L, self.kx, self.ky, 'point')
        for f in self.forms:
            self.data[(role, f)] = BC.form_data(self.base, blk, f)
        self.rules[role], self.m[role] = rule, len(X)
        self.meta[role] = dict(rule=rule, rule_sha256=rsha, m=len(X), Rp=self.Rp, M=self.M, L=self.L,
                               mode_hash=self.mode_hash, fingerprint=P.fingerprint(self.data[(role, self.forms[0])]))
        del blk

    def drop_role(self, role):
        for k in [k for k in self.data if k[0] == role]:
            del self.data[k]
        self.rules.pop(role, None)

    def q(self, form, nout=6, nrec=0, budget=600, nstep=BC.NSTEP):
        k = (form, nout, nrec, budget, nstep)
        if k not in self._q:
            self._q[k] = BC.make_split_query(Q, 'point', form, self.Rp, self.L, self.trust, nrec=nrec, budget=budget,
                                             nstep=nstep, nout=nout)
        return self._q[k]

    def check(self, role):
        """Wrong-table guard: the arrays must still be the ones built for this role (fingerprint)."""
        if P.fingerprint(self.data[(role, self.forms[0])]) != self.meta[role]['fingerprint']:
            raise ValueError(f'table of role {role} differs from the one built (substituted data)')


# ------------------------------------------------------------------------------------------- LS initialisation --

def row_blocks(bank):
    """Row ranges of the rotated bank's row blocks in the flattened interior (x slowest), from the block shapes."""
    rows, off = [], 0
    for rb in bank.Grot:
        n = int(rb[0].shape[0])
        rows.append((off, off + n))
        off += n
    assert off == (bank.L - 1) ** 2, (off, bank.L)
    return rows


def prefix_blocks(rb, edges, Rp):
    nb = list(edges).index(Rp)
    return jnp.concatenate(rb[:nb], axis=1) if nb > 1 else rb[0]


class LSInit:
    """A1.3: full-grid LS init on the prefix r. H = sum_b G_b^T G_b once per r; per case g = sum_b G_b^T u_b, Cholesky
    solve + one refinement step; G-B5 orthogonality residual ||G^T(Gc - u)|| / (||G||_F ||Gc - u||), blockwise."""

    def __init__(self, bank, Rp):
        self.bank, self.Rp = bank, int(Rp)
        self.rows = row_blocks(bank)
        t0 = time.perf_counter()
        H = None
        for rb in bank.Grot:
            g = prefix_blocks(rb, bank.edges, self.Rp)
            h = g.T @ g
            H = h if H is None else H + h
        self.H = jax.block_until_ready(H)
        self.GF = float(jnp.sqrt(jnp.trace(H)))                     # ||G||_F
        ev = np.linalg.eigvalsh(np.asarray(H))
        self.cond = float(ev[-1] / ev[0]) if ev[0] > 0 else float('inf')
        self.cho = jax.scipy.linalg.cho_factor(self.H, lower=True)
        self.seconds = time.perf_counter() - t0

    def _GT(self, u):
        acc = None
        for (a, b), rb in zip(self.rows, self.bank.Grot):
            g = prefix_blocks(rb, self.bank.edges, self.Rp)
            v = g.T @ u[a:b]
            acc = v if acc is None else acc + v
        return acc

    def _G(self, c):
        return jnp.concatenate([prefix_blocks(rb, self.bank.edges, self.Rp) @ c for rb in self.bank.Grot])

    def __call__(self, u0_full):
        """u0_full: (L+1, L+1) nodal IC incl. walls -> (c, record)."""
        L = self.bank.L
        u = jnp.asarray(np.asarray(u0_full)[1:-1, 1:-1].reshape(-1))
        c = jax.scipy.linalg.cho_solve(self.cho, self._GT(u))
        c = c + jax.scipy.linalg.cho_solve(self.cho, self._GT(u - self._G(c)))        # one refinement step
        res = self._G(c) - u
        orth = float(jnp.linalg.norm(self._GT(res)) / (self.GF * jnp.linalg.norm(res) + 1e-300))
        return c, dict(orth=orth, cond=self.cond, rel_residual=float(jnp.linalg.norm(res) / jnp.linalg.norm(u)), L=L)


# ------------------------------------------------------------------------------------------------- corners --

def corner_spec(member, S):
    """Configuration tuple of corner S (a frozenset of ING) for a member dict (form, scheme, factor, rule roles)."""
    form = OTHER[member['form']] if 'F' in S else member['form']
    scheme, dt = ('BDF2', DT0 / 8) if 'T' in S else (member['scheme'], DT0 * float(member['factor']))
    return dict(role='qinf' if 'Q' in S else 'main', form=form, scheme=scheme, dt=dt,
                level='tight' if 'S' in S else 'prod', init='ls' if 'I' in S else 'deployed')


def assert_tuple(spec0, spec, S):
    """G-B6: exactly the elements of S differ from the deployed tuple."""
    diff = {k for k in spec0 if spec0[k] != spec[k]}
    want = set()
    for x in S:
        want |= {'Q': {'role'}, 'F': {'form'}, 'T': {'scheme', 'dt'}, 'S': {'level'}, 'I': {'init'}}[x]
    # the time package may leave the scheme unchanged only if the deployed scheme is already BDF2 (never in FRONT-2D)
    if diff != want:
        raise AssertionError(f'corner {sorted(S)}: changed {sorted(diff)}, expected {sorted(want)}')


def run_corner(st, spec, u0, nu, w0_ls=None, nout=6, nrec=0, budget=600, w0_scale=1.0, horizon=.25):
    """init -> (w0 replacement) -> evolve. Returns dict(W, w0, stats, valid, scale, seconds)."""
    st.check(spec['role'])
    q = st.q(spec['form'], nout=nout, nrec=nrec, budget=budget,
             nstep=max(BC.NSTEP, steps_keep(spec['dt'], horizon) + 1))
    t0 = time.perf_counter()
    w0, scale = q['init'](u0, st.cold)
    if spec['init'] == 'ls':
        w0 = jnp.asarray(w0_ls)
    w0 = w0 * w0_scale
    g, tf = P.tol_of(spec['form'], spec['level'])
    sch = BC.sched(spec['scheme'], spec['dt'], g, tf, horizon=horizon)
    W, stt = q['evolve'](w0, nu, scale, st.data[(spec['role'], spec['form'])], sch)
    jax.block_until_ready(W)
    v = dict(W=W, w0=w0, stats=stt)
    return dict(W=np.asarray(W), w0=np.asarray(w0), stats=stt, valid=P.run_valid(v), scale=float(scale),
                seconds=time.perf_counter() - t0, budget_exits=P.budget_exits(v))


# ------------------------------------------------------------------------------------------ fine lattice (B8) --

FINE_L = 1024


def decode(st_G, W, L):
    """W (n, r) -> (n, L+1, L+1) with zero walls, through bank G (p1core decode)."""
    U = Q.H.bank_apply(st_G, jnp.asarray(W).T).T.reshape(len(W), L - 1, L - 1)
    return jnp.pad(U, ((0, 0), (1, 1), (1, 1)))


def energy_diss(F, nu, L=FINE_L):
    """A1.9: E = 1/2 h^2 sum u^2 over interior nodes, D = nu h^2 sum |grad_h u|^2 (central differences, zero walls).
    F: (n, L+1, L+1) incl. walls. Returns (E (n,), D (n,))."""
    h = 1.0 / L
    I = F[:, 1:-1, 1:-1]
    gx = (F[:, 2:, 1:-1] - F[:, :-2, 1:-1]) / (2 * h)
    gy = (F[:, 1:-1, 2:] - F[:, 1:-1, :-2]) / (2 * h)
    E = .5 * h * h * jnp.sum(I ** 2, axis=(1, 2))
    D = nu * h * h * jnp.sum(gx ** 2 + gy ** 2, axis=(1, 2))
    return np.asarray(E), np.asarray(D)


def balance(E, D, dt_out=.05):
    """B_k = E_k - E_{k-1} + dt/2 (D_k + D_{k-1}), k >= 1."""
    E, D = np.asarray(E), np.asarray(D)
    return E[1:] - E[:-1] + .5 * dt_out * (D[1:] + D[:-1])


def band_mask(L=FINE_L, width=1 / 32):
    x = np.arange(L + 1) / L
    d = np.minimum(x, 1 - x)
    m1 = d <= width + 1e-15
    inter = np.zeros(L + 1, bool)
    inter[1:-1] = True
    band = (m1[:, None] | m1[None, :]) & inter[:, None] & inter[None, :]
    return band, inter[:, None] & inter[None, :]


def alias_field(L=FINE_L, k=256):
    """phi = sin(k pi x) sin(k pi y) on the (L+1)^2 nodes: zero at x = i/256 (the scoring lattice) for k = 256."""
    x = np.arange(L + 1) / L
    s = np.sin(k * np.pi * x)
    return np.outer(s, s)


def file_sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()
