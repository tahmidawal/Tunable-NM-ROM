"""quadrature-study: off-mesh quadrature for the tested advection term of the frozen Burgers 2D model.

The model is the Table-1 Burgers 2D model (checkpoint dn256b, K=16, R=512, rotation_R512). Three settings:
  acc   linear rung, span of the first R'=384 rotated bank columns, M=1536 sine tests
  fast  linear rung, R'=128, M=512
  head  the k=16 head (q=0) on the full rotated bank (R'=512 folded into the head), M=64

The reduced residual of one backward-Euler step (the project's weak residual, unchanged):

    r(c) = S ( A c - p + dt ( N(c) + nu Lambda A c ) ),   S = 1 / (1 + dt nu Lambda),

with A = Phi^T G' (exact discrete projection of the bank; Phi the mesh-orthonormal sines (2/L) sin sin) and Lambda the
discrete eigenvalues. ONLY the tested advection N(c) depends on the quadrature rule (the 'hybrid' of DESIGN.md):

  dense   N = Phi^T a_h(G' c)              exact upwind stencil at every mesh node (the FOM's own advection)
  mesh    N = Pq^T a_h(G5 c)               upwind stencil at the rule's m mesh nodes, Pq = Phi[nodes] * w
                                           (the deployed 63x63 lattice `lat64`, the fitted EQ rule `q0scaled`)
  point   N = L sum_q w_q psi(x_q) u (u_x + u_y)(x_q)       off-mesh, Hari's gradient form
  flux    N = -L sum_q w_q (psi_x + psi_y)(x_q) u(x_q)^2/2   off-mesh, Hari's integrated-by-parts form

with psi = 2 sin(kx pi x) sin(ky pi y) the L2-orthonormal continuum sines of the SAME (kx, ky) as Phi, and
(x_q, w_q), sum w_q = 1 (except Smolyak, see offmesh_data), a classical rule on (0,1)^2. Because Phi^T F = sum_ij (2/L) s s F ~ L int psi F, the
off-mesh form reproduces the mesh-tested term up to the stencil's consistency error (Hari, grid.py docstring).
u, u_x, u_y at x_q come from the coordinate-network bank and its forward-mode derivatives (partial decoding).

Every large array is a jit argument. f64, highest matmul precision.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE / 'vendor', ROOT / 'experiments' / 'mr-burgers2d', ROOT / 'experiments' / 'separable-decoder'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e
import sep_common as sc
import arms as A
import hops as H
import hfast as HF
import bkfast as BK            # patches hops.bank_apply for the nested rotated bank
import hari_quadrature as HQ

CKPT = ROOT / 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
ROTATION = HERE / 'inputs/rotation_R512.npz'
EQ_RULE = HERE / 'inputs/rule_q0_m1024_qrg304_reachable.npz'
SETTINGS = dict(acc=dict(kind='lin', Rp=384, M=1536), fast=dict(kind='lin', Rp=128, M=512),
                head=dict(kind='head', Rp=512, M=64))


# ------------------------------------------------------------------ cohorts ----

def cohort(name):
    """Physical descriptors (cx, cy, w, a, nu) of the named cohort (project definitions, LAB-LOG)."""
    if name == 'dev6':
        return np.concatenate((e.params_draw(7090702, 4), e.params_draw(911702, 2)))
    if name == 'val32':
        return e.params_draw(20260927, 32)        # burgers-heldout 'sel32'
    if name == 'test64':
        return e.params_draw(20260916, 64)        # 'hold64' = burgers2d-test 'test64'
    raise ValueError(name)


# -------------------------------------------------------------------- rules ----

def _fib_index(n):
    a, b, k = 1, 1, 2
    while b < n:
        a, b, k = b, a + b, k + 1
    assert b == n, f'{n} is not a Fibonacci number'
    return k


def offmesh_rule(name):
    """(X (m, 2), w (m,)) of a named off-mesh rule; sum w = 1 except Smolyak (Hari's boundary-node removal). Names: gauss<p>, fib<n>, sobol<m>, halton<m>,
    smolyak<level> (Clenshaw-Curtis). All random rules use seed 0 (Hari's defaults)."""
    if name.startswith('gauss'):
        X, w = HQ.gauss_tensor(int(name[5:]))
    elif name.startswith('fib'):
        X, w = HQ.fibonacci_lattice(_fib_index(int(name[3:])), seed=0)
    elif name.startswith('sobol'):
        X, w = HQ.sobol(int(name[5:]), scramble=True, seed=0)
    elif name.startswith('halton'):
        X, w = HQ.halton(int(name[6:]), scramble=True, seed=0)
    elif name.startswith('smolyak'):
        X, w = HQ.smolyak(int(name[7:]), 'cc')
    else:
        raise ValueError(name)
    X, w = np.asarray(X, float), np.asarray(w, float)
    assert np.all((X > 0) & (X < 1)), name
    return X, w


def mesh_rule(name, L):
    """(ij (m, 2) 1-based interior indices, w) of a named mesh rule: lat<s> (uniform (s-1)^2 sub-lattice, equal
    weights (L/s)^2, the deployed lat64 for s=64) or q0scaled (b-eqtop's fitted NNLS rule, m=1024 at 256^2 nodes,
    weights x (L/256)^2, no refit; the head's deployed rule)."""
    if name.startswith('lat'):
        return H.lattice_rule(L, int(name[3:]))
    if name == 'q0scaled':
        z = np.load(EQ_RULE)
        ij = H.transfer_nodes(z['nodes'].astype(int), 256, L)
        w = np.asarray(z['weights'], float) * (L / 256) ** 2
        keep = w > 0
        return ij[keep], w[keep]
    raise ValueError(name)


# --------------------------------------------------------------------- bank ----

class Model:
    """The frozen model, its rotation and the rotated bank on/off the mesh."""

    def __init__(self, ckpt=CKPT, rotation=ROTATION):
        import pickle
        ck = pickle.load(open(ckpt, 'rb'))
        self.params = jax.tree_util.tree_map(jnp.asarray, LD_host(ck['params']))
        self.Zold = np.asarray(ck['Z_tr'])
        self.K = int(self.Zold.shape[1])
        self.R = int(np.asarray(ck['params']['h_lin']).shape[1])
        rz = np.load(rotation)
        self.T, self.Lrot = np.asarray(rz['T']), np.asarray(rz['L'])
        self.base = A.CoordBank(self.params, self.K, self.R)
        self._grad = jax.jit(jax.vmap(self._one_grad))

    def _one_grad(self, x):
        f = lambda xx: sc.features(self.params, xx[None])[0]
        v, gx = jax.jvp(f, (x,), (jnp.array([1., 0.]),))
        _, gy = jax.jvp(f, (x,), (jnp.array([0., 1.]),))
        return v, gx, gy

    def values_grads(self, X, Rp, chunk=4096):
        """Rotated bank and its x, y derivatives at points X: three (m, R') arrays (forward-mode AD)."""
        Tp = jnp.asarray(self.T[:, :Rp])
        outs = [self._grad(jnp.asarray(X[s:s + chunk])) for s in range(0, len(X), chunk)]
        v, gx, gy = (jnp.concatenate([o[i] for o in outs]) @ Tp for i in range(3))
        return jax.block_until_ready(v), jax.block_until_ready(gx), jax.block_until_ready(gy)

    def head_params(self, Rp):
        """Head whose output is L[:R'] h(z) (BK.fold_head): coefficients in the rotated basis."""
        return BK.fold_head(self.params, self.Lrot, Rp)

    def trust_linear(self, Rp):
        hv = jax.jit(jax.vmap(lambda z: sc.head(self.params, z)))
        Hall = np.concatenate([np.asarray(hv(jnp.asarray(self.Zold[s:s + 8192]))) for s in range(0, len(self.Zold), 8192)])
        codes = Hall @ self.Lrot[:Rp].T
        return .01 * float(np.max(np.linalg.norm(codes - codes.mean(0), axis=1))), Hall

    def trust_head(self):
        return .01 * float(np.max(np.linalg.norm(self.Zold - self.Zold.mean(0), axis=1)))


def LD_host(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def build_rotated_bank(model, L, edges, nrb=None):
    """b2test's construction: the mesh bank G T[:, :Rb] as row blocks x nested column blocks (edges)."""
    R = model.R
    Rb = edges[-1]
    n = (L - 1) ** 2
    nrb = int(nrb or np.ceil(n * R / H.MAX_GEMM_ELEMENTS))
    x = np.arange(1, L) / L
    redges = np.linspace(0, L - 1, nrb + 1).astype(int)
    Tj = jnp.asarray(model.T[:, :Rb])
    rotate = jax.jit(lambda g, t: tuple((g @ t)[:, a_:b_] for a_, b_ in zip(edges[:-1], edges[1:])))
    Grot = []
    for i0, i1 in zip(redges[:-1], redges[1:]):
        xy = np.stack(np.meshgrid(x[i0:i1], x, indexing='ij'), -1).reshape(-1, 2)
        g = jax.block_until_ready(model.base.at(xy, chunk=8192))
        Grot.append(jax.block_until_ready(rotate(g, Tj)))
        del g
    return tuple(Grot)


def operators(Grot, edges, L, M):
    kx, ky, lam = H.modes_lean(L, M)
    sx, sy = (jnp.asarray(t_) for t_ in H.sine_tables(L, kx, ky))
    Arot = BK.project_nested(Grot, edges, edges[-1], sx, sy, L)
    return jax.block_until_ready(dict(kx=kx, ky=ky, lam=jnp.asarray(lam), sx=sx, sy=sy, Arot=Arot))


def continuum_tests(X, kx, ky):
    """psi = 2 sin(kx pi x) sin(ky pi y) and psi_x + psi_y at X: two (m, M) arrays (Hari's grid.continuum_tests)."""
    x, y = X[:, 0:1], X[:, 1:2]
    a, b = np.asarray(kx, float)[None], np.asarray(ky, float)[None]
    sx, cx, sy, cy = np.sin(a * np.pi * x), np.cos(a * np.pi * x), np.sin(b * np.pi * y), np.cos(b * np.pi * y)
    return 2. * sx * sy, 2. * np.pi * (a * cx * sy + b * sx * cy)


def _tests_dev(X, kx, ky):
    """psi and psi_x + psi_y at X on the device: two (m, M) arrays (Hari's grid.continuum_tests, jnp)."""
    X = jnp.asarray(X)
    x, y = X[:, 0:1], X[:, 1:2]
    a, b = jnp.asarray(np.asarray(kx, float))[None], jnp.asarray(np.asarray(ky, float))[None]
    sx, cx, sy, cy = jnp.sin(a * jnp.pi * x), jnp.cos(a * jnp.pi * x), jnp.sin(b * jnp.pi * y), jnp.cos(b * jnp.pi * y)
    return 2. * sx * sy, 2. * jnp.pi * (a * cx * sy + b * sx * cy)


def offmesh_data(model, Rp, X, w, L, kx, ky, form, chunk=32768):
    """Cached blocks of an off-mesh rule, built on the device in point chunks (no host (m, M) tables).
    point: Gq, Gs = Gx + Gy, Psi = L w psi.  flux: Gq, Pxy = L w (psi_x + psi_y).  Smolyak rules carry their own
    (possibly negative) weights, whose sum is not 1 after Hari's boundary-node removal; they are never renormalised."""
    parts = []
    for s in range(0, len(X), chunk):
        Xs, ws = X[s:s + chunk], jnp.asarray(w[s:s + chunk])[:, None]
        Gq, Gx, Gy = model.values_grads(Xs, Rp)
        psi, pxy = _tests_dev(Xs, kx, ky)
        if form == 'point':
            parts.append((Gq, Gx + Gy, L * ws * psi))
        elif form == 'flux':
            parts.append((Gq, L * ws * pxy))
        else:
            raise ValueError(form)
        del Gx, Gy, psi, pxy
    cat = [jnp.concatenate([p_[i] for p_ in parts]) if len(parts) > 1 else parts[0][i] for i in range(len(parts[0]))]
    keys = ('Gq', 'Gs', 'Psi') if form == 'point' else ('Gq', 'Pxy')
    return jax.block_until_ready(dict(zip(keys, cat)))


def mesh_data(model, Rp, ij, w, L, kx, ky):
    g5 = model.base.stencil(ij, L)
    G5 = jnp.einsum('msr,rp->msp', g5, jnp.asarray(model.T[:, :Rp]))
    return dict(G5=jax.block_until_ready(G5), Pq=jnp.asarray(H.phi_rows(L, kx, ky, ij) * w[:, None]))


# ----------------------------------------------------------- tested advection ----

def tested_value(kind, L):
    """N(c) for a coefficient vector c (rotated basis, length R'); data carries the rule's blocks."""
    if kind == 'mesh':
        return lambda c, d: d['Pq'].T @ H.advect_points(jnp.einsum('msr,r->ms', d['G5'], c), L)
    if kind == 'point':
        return lambda c, d: d['Psi'].T @ ((d['Gq'] @ c) * (d['Gs'] @ c))
    if kind == 'flux':
        return lambda c, d: -(d['Pxy'].T @ (.5 * (d['Gq'] @ c) ** 2))
    if kind == 'dense':
        return lambda c, d: H.sep_project(e.spatial(H.bank_apply(d['G'], c), L)[0].reshape(L - 1, L - 1),
                                          d['sx'], d['sy'], L)
    raise ValueError(kind)


def tested_jac(kind, L):
    """(N(c), dN/dc @ Tm) for a tangent block Tm (R', d) -- analytic for mesh / point / flux."""
    dadv = jax.vmap(jax.grad(lambda u: H.advect_points(u, L)))
    if kind == 'mesh':
        def f(c, d, Tm):
            us = jnp.einsum('msr,r->ms', d['G5'], c)
            val = d['Pq'].T @ H.advect_points(us, L)
            G5T = d['G5'] if Tm is None else jnp.einsum('msr,rk->msk', d['G5'], Tm)
            return val, d['Pq'].T @ jnp.einsum('ms,msr->mr', dadv(us), G5T)
        return f
    if kind == 'point':
        def f(c, d, Tm):
            u, s = d['Gq'] @ c, d['Gs'] @ c
            Bq, Bs = (d['Gq'], d['Gs']) if Tm is None else (d['Gq'] @ Tm, d['Gs'] @ Tm)
            return d['Psi'].T @ (u * s), d['Psi'].T @ (s[:, None] * Bq + u[:, None] * Bs)
        return f
    if kind == 'flux':
        def f(c, d, Tm):
            u = d['Gq'] @ c
            Bq = d['Gq'] if Tm is None else d['Gq'] @ Tm
            return -(d['Pxy'].T @ (.5 * u * u)), -(d['Pxy'].T @ (u[:, None] * Bq))
        return f
    raise ValueError(kind)


# --------------------------------------------------------------- linear rung ----

def make_linear_query(kind, Rp, L, dt, trust, step_budget=600, gtol=1e-3, ridge=1e-10, solver='chol',
                      tangent_chunks=1, clip=True, lam_carry=True, predictor='quad'):
    """bkfast.make_linear_query (the Table-1 linear-rung query: fused LM, Cholesky, clipped step, damping carried,
    quadratic predictor, closed-form Gauss-point initial fit) with the tested advection replaced by `kind`.
    kind='mesh' is bkfast's res_q/evalJ_q text; kind='dense' bkfast's res_d/evalJ_d. Output tuple layout identical."""
    steps = int(round(.25 / dt))
    keep = int(round(.05 / dt))
    nl = tested_value(kind, L)

    def res(w, prev, nu, data, tab):
        ah = data['A'] @ w
        lam = data['lam']
        return (ah - prev + dt * (nl(w, data) + nu * lam * ah)) / (1 + dt * nu * lam)

    if kind == 'dense':
        def evalJ(w, args):
            prev, nu, data, tab = args
            r, lin = jax.linearize(lambda ww: res(ww, prev, nu, data, tab), w)
            basis = jnp.eye(Rp).reshape(tangent_chunks, Rp // tangent_chunks, Rp)
            J = jax.lax.map(lambda B: jax.vmap(lin)(B), basis).reshape(Rp, -1).T
            return r, J
    else:
        nlJ = tested_jac(kind, L)

        def evalJ(w, args):
            prev, nu, data, tab = args
            ah = data['A'] @ w
            lam = data['lam']
            S = 1. / (1 + dt * nu * lam)
            val, dN = nlJ(w, data, None)
            r = (ah - prev + dt * (val + nu * lam * ah)) * S
            return r, data['A'] + (dt * S)[:, None] * dN

    lm = HF.make_fused_lm(evalJ, Rp, 0, step_budget, trust, gtol, ridge, solver, clip)

    def initialize(u0, data, cold):
        xy, wq, Q, R, _, _, _ = cold
        ui = e.sample_field(u0, xy, L) * wq
        yv = Q.T @ ui
        w0 = jax.scipy.linalg.solve_triangular(R, yv, lower=False)
        scale = jnp.linalg.norm(ui) * jnp.sqrt(len(wq))
        return w0, scale

    def evolve(w0, nu, scale, data):
        tab = None

        def step(carry, k):
            wv, wprev, wprev2, lam0 = carry
            p = data['A'] @ wv
            we = wv + (wv - wprev)
            if predictor == 'quad':
                wqd = jnp.where(k >= 2, 3. * wv - 3. * wprev + wprev2, we)
                cand = jnp.stack((wv, we, wqd))
                rs = jax.vmap(lambda ww: jnp.linalg.norm(res(ww, p, nu, data, tab)))(cand)
                rs = jnp.where(jnp.isfinite(rs), rs, jnp.inf)
                wi = cand[jnp.argmin(rs)]
            else:
                r0 = jnp.linalg.norm(res(wv, p, nu, data, tab))
                re = jnp.linalg.norm(res(we, p, nu, data, tab))
                wi = jnp.where(jnp.isfinite(re) & (re < r0), we, wv)
            w2, rn, it, reason, gn, rej, lam = lm(wi, (p, nu, data, tab), 1e-9 * scale, lam0)
            return (w2, wv, wprev, lam if lam_carry else lam0), (w2, rn, it, reason, gn, rej)
        _, out = jax.lax.scan(step, (w0, w0, w0, jnp.asarray(1e-6, dtype=jnp.float64)), jnp.arange(steps))
        return out

    def decode_fields(W, data):
        U = H.bank_apply(data['G'], W.T).T.reshape(len(W), L - 1, L - 1)
        return jnp.pad(U, ((0, 0), (1, 1), (1, 1)))

    def query(u0, nu, data, cold):
        w0, scale = initialize(u0, data, cold)
        ws, rn, it, reason, gn, rej = evolve(w0, nu, scale, data)
        internal = jnp.concatenate((w0[None], ws))
        fields = decode_fields(internal[::keep], data)
        return dict(fields=fields, it=it, rn=rn, reason=reason, gn=gn, rej=rej, internal=internal)

    return jax.jit(query), jax.jit(decode_fields)


# ---------------------------------------------------------------------- head ----

def make_head_eval(kind):
    """Replacement for hfast.make_eq_eval (q = 0): the head's residual and analytic Jacobian with the tested
    advection of `kind`. kind='mesh' returns hfast's own evaluator unchanged."""
    orig = HF.make_eq_eval

    def factory(params, K, q, L, dt):
        if kind == 'mesh':
            return orig(params, K, q, L, dt)
        assert q == 0
        hz = lambda z: sc.head(params, z)
        jh = jax.jacfwd(hz)
        nl, nlJ = tested_value(kind, L), tested_jac(kind, L)

        def res(w, prev, nu, data, tab):
            h = hz(w[:K])
            ah = data['A'] @ h
            lam = data['lam']
            return (ah - prev + dt * (nl(h, data) + nu * lam * ah)) / (1 + dt * nu * lam)

        def evalJ(w, args):
            prev, nu, data, tab = args
            z = w[:K]
            h = hz(z)
            ah = data['A'] @ h
            lam = data['lam']
            S = 1. / (1 + dt * nu * lam)
            Jh = jh(z)
            val, dN = nlJ(h, data, Jh)
            r = (ah - prev + dt * (val + nu * lam * ah)) * S
            return r, data['A'] @ Jh + (dt * S)[:, None] * dN
        return res, evalJ, hz
    return factory


def make_head_query(kind, params, K, L, dt, trust, step_budget=600, gtol=1e-3, ridge=1e-10, ic_budget=400,
                    ic_gtol=1e-6, tangent_chunks=1):
    """hfast.make_query (q=0) with the t2-burgers-test head variant: LU, clip, damping carry, quadratic predictor.
    kind='dense' is hfast's own exact-advection query."""
    common = dict(ic_budget=ic_budget, step_budget=step_budget, gtol=gtol, ic_gtol=ic_gtol, ridge=ridge,
                  solver='lu', clip=True, lam_carry=True, predictor='quad', parts=True)
    C0 = jnp.zeros((params['h_lin'].shape[1], 0))
    if kind == 'dense':
        fq, parts = HF.make_query(params, C0, K, 0, L, dt, trust, 'dense', tangent_chunks=tangent_chunks, **common)
    else:
        saved = HF.make_eq_eval
        HF.make_eq_eval = make_head_eval(kind)
        try:
            fq, parts = HF.make_query(params, C0, K, 0, L, dt, trust, 'eq', **common)
        finally:
            HF.make_eq_eval = saved
    dec = parts['decode']

    def query(u0, nu, data, cold, tab):
        v = fq(u0, nu, data, cold, tab)
        return dict(fields=v[0], it=v[1], rn=v[2], reason=v[3], gn=v[8], rej=v[15], internal=v[7])
    return query, dec


def head_tables(params, K, data, cold):
    C0 = jnp.zeros((params['h_lin'].shape[1], 0))
    return HF.build_tables(params, C0, K, data, cold)


def build_cold_linear(model, Rp, codes):
    return A.build_cold(BK.RotBank(model.base, model.T, Rp), lambda w_: w_, codes, 48)[0]


def build_cold_head(model, params_fold, Zsub):
    return A.build_cold(BK.RotBank(model.base, model.T, 512), lambda z: sc.head(params_fold, z), Zsub, 48)[0]
