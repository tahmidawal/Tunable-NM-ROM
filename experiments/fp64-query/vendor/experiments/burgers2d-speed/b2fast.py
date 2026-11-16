"""burgers2d-speed: iterate-preserving engineering of the linear-rung and head queries (DESIGN.md section 3).

Nothing here changes the reduced model, the residual, the quadrature rule, the solver, the stopping rule or the
predictor; it changes only how the SAME arithmetic is emitted (reassociation class, gated by parity in every job):

  E1 `sep_eq`     The lattice rule's test matrix Pq (m, M), Pq[(a,b), k] = w (2/L) sin(pi x_a kx_k) sin(pi x_b ky_k)
                  on a tensor (s-1) x (s-1) sub-lattice with equal weights, is applied SEPARABLY through the distinct
                  wave numbers: Pq^T X = c * [SX_u^T (X SY_u)][ix, iy]. At M = 4R', m = 3969 this replaces the dense
                  (M x m)(m x R') product, the dominant term of every LM iteration, by two small contractions.
  E2 `ajac_dense` The exact (all-node) first step (`x1`) assembles its Jacobian analytically,
                      J = A + dt S Phi^T ( D . stencil(G) ),  D = d adv / d (c, xp, xm, yp, ym) pointwise,
                  instead of pushing R' forward-mode tangents through the dense residual (the parent's
                  `jax.linearize` path costs R' dense sine projections per Jacobian).
  E3 `sep_dense`  Every dense Phi^T v (exact-step residual and Jacobian) uses the distinct-wave-number
                  factorisation too, (L-1)^2 x n_distinct instead of (L-1)^2 x M.
  E4 `graphs`     (compile option, applied by the driver) XLA command buffers including WHILE and CONDITIONAL, so
                  the LM while loop and the time scan run as CUDA graphs without a host round trip per iteration.

`lm_cap` is NOT engineering: `lm_cap = 1` is the deployment knob "iteration cap 1" (one damped Gauss-Newton step per
time step at most), emitted as straight-line code with no while loop. It changes iterates whenever a step would
have taken more than one iteration, so arms that use it are separate candidates whose error is measured.

`make_fused_lm` is a verbatim copy of hires-burgers/hfast.make_fused_lm (@ b5c843ab) with the loop driver
factored out so the capped variant shares the body text; parity of the while-loop path against hfast is gated.
"""
from __future__ import annotations

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e
import hops as H

GRAPHS = {'xla_gpu_enable_command_buffer': 'FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,CUDNN,WHILE,CONDITIONAL',
          'xla_gpu_graph_min_graph_size': 1}


# ------------------------------------------------------------- separable tables ----

def distinct_tables(xpts, kx, ky, coef):
    """Tables for Pq^T X through the distinct wave numbers; xpts are the 1D node coordinates (same expression as
    hops.sine_tables / hops.phi_rows: sin(pi * x * k))."""
    ux, ix = np.unique(np.asarray(kx), return_inverse=True)
    uy, iy = np.unique(np.asarray(ky), return_inverse=True)
    x = np.asarray(xpts, float)
    return dict(SX=jnp.asarray(np.sin(np.pi * x[:, None] * ux)), SY=jnp.asarray(np.sin(np.pi * x[:, None] * uy)),
                ix=jnp.asarray(ix.astype(np.int32)), iy=jnp.asarray(iy.astype(np.int32)),
                coef=jnp.asarray(float(coef)))


def lattice_tables(L, s, kx, ky):
    """For hops.lattice_rule(L, s): nodes a (L/s), a = 1..s-1 per axis (a-major), weights (L/s)^2."""
    a = np.arange(1, s) * (L // s)
    return distinct_tables(a / L, kx, ky, (2. / L) * float((L // s) ** 2))


def dense_tables(L, kx, ky):
    return distinct_tables(np.arange(1, L) / L, kx, ky, 2. / L)


def proj_vec(v, T):
    """Pq^T v for v of length n*n (a-major) -> (M,)."""
    n = T['SX'].shape[0]
    V = v.reshape(n, n)
    Z = T['SX'].T @ (V @ T['SY'])                                   # (nux, nuy)
    return T['coef'] * Z[T['ix'], T['iy']]


def proj_cols(X, T):
    """Pq^T X for X of shape (n*n, R) or (n, n, R) (a-major) -> (M, R)."""
    n = T['SX'].shape[0]
    Xr = X.reshape(n, n, -1)
    T1 = jnp.einsum('abr,bv->avr', Xr, T['SY'])
    Z = jnp.einsum('au,avr->uvr', T['SX'], T1)
    return T['coef'] * Z[T['ix'], T['iy']]


# ------------------------------------------------------------------- the LM ----

def make_fused_lm(evalJ, K, q, budget, trust, gtol, ridge, solver='lu', clip=False, cap=None):
    """hfast.make_fused_lm verbatim; `cap=1` runs the body at most once as straight-line code (a knob)."""
    mask_z = jnp.concatenate((jnp.ones(K), jnp.zeros(q)))
    mask_y = 1. - mask_z

    def solve(Hm, g):
        if solver == 'chol':
            c = jax.scipy.linalg.cho_factor(Hm, lower=True)
            return jax.scipy.linalg.cho_solve(c, -g)
        return jnp.linalg.solve(Hm, -g)

    def ratio(g, J, rn):
        return jnp.linalg.norm(g) / (jnp.linalg.norm(J) * rn + 1e-300)

    def lm(w0, args, tol, lam0=1e-6):
        r, J = evalJ(w0, args)
        rn = jnp.linalg.norm(r)
        g = J.T @ r
        reason = jnp.where(jnp.isfinite(rn),
                           jnp.where(ratio(g, J, rn) <= gtol, 4, jnp.where(rn <= tol, 1, 0)), 3).astype(jnp.int32)

        def body(s):
            w, r, J, g, rn, lam, it, reason, rej = s
            Hm = J.T @ J
            d = jnp.diag(Hm) + 1e-30
            step = solve(Hm + jnp.diag(lam * d * mask_z + ridge * d * mask_y), g)
            if clip:
                nz = jnp.linalg.norm(step[:K])
                step = step * jnp.where(nz > trust, trust / (nz + 1e-300), 1.)
                ok = jnp.all(jnp.isfinite(step))
            else:
                ok = jnp.all(jnp.isfinite(step)) & (jnp.linalg.norm(step[:K]) <= trust)
            wn = w + jnp.where(ok, step, 0.)
            r2, J2 = evalJ(wn, args)
            rn2 = jnp.linalg.norm(r2)
            accept = ok & jnp.isfinite(rn2) & (rn2 < rn)
            w3 = jnp.where(accept, wn, w)
            r3 = jnp.where(accept, r2, r)
            J3 = jnp.where(accept, J2, J)
            rn3 = jnp.where(accept, rn2, rn)
            g3 = J3.T @ r3
            gn = ratio(g3, J3, rn3)
            tiny = ok & (jnp.linalg.norm(step) <= 1e-14 * (1 + jnp.linalg.norm(w)))
            reason = jnp.where(gn <= gtol, 4,
                      jnp.where(rn3 <= tol, 1,
                       jnp.where(tiny, 2, jnp.where((~accept) & (lam >= 1e14), 3, 0)))).astype(jnp.int32)
            lam2 = jnp.where(accept, jnp.maximum(lam / 3, 1e-12), jnp.minimum(lam * 10, 1e14))
            return (w3, r3, J3, g3, rn3, lam2, it + 1, reason, rej + (~accept).astype(jnp.int32))

        s0 = (w0, r, J, g, rn, jnp.asarray(lam0, dtype=jnp.float64), jnp.int32(0), reason, jnp.int32(0))
        if cap is None:
            s = jax.lax.while_loop(lambda s: (s[6] < budget) & (s[7] == 0), body, s0)
        else:
            s = s0
            for _ in range(int(cap)):
                go = (s[6] < budget) & (s[7] == 0)
                s1 = body(s)
                s = jax.tree_util.tree_map(lambda a_, b_: jnp.where(go, b_, a_), s, s1)
        w, r, J, g, rn, lam, it, reason, rej = s
        return w, rn, it, reason, ratio(g, J, rn), rej, lam
    return lm


# ------------------------------------------------------------ linear rung ----

def make_linear_query(Rp, L, dt, trust, exact_steps=0, step_budget=600, gtol=1e-6, ridge=1e-10, solver='chol',
                      tangent_chunks=1, clip=False, lam_carry=False, predictor='lin', sep_eq=False,
                      ajac_dense=False, sep_dense=False, lm_cap=None, compiler_options=None):
    """bkfast.make_linear_query (@ b5c843ab) with the E1-E3 emission switches and the lm_cap knob. With every switch
    off and lm_cap None this is the parent's query text (parity gated). Extra `data` keys: `sepq` (E1: lattice
    tables), `sepd` (E3: dense tables), `Gf` (E2: the bank prefix as one (L-1, L-1, R') array)."""
    steps = int(round(.25 / dt))
    keep = int(round(.05 / dt))
    dadv = jax.vmap(jax.grad(lambda u: H.advect_points(u, L)))
    assert Rp % tangent_chunks == 0

    def PqT_vec(data, v):
        return proj_vec(v, data['sepq']) if sep_eq else data['Pq'].T @ v

    def PqT_cols(data, X):
        return proj_cols(X, data['sepq']) if sep_eq else data['Pq'].T @ X

    def res_q(w, prev, nu, data, tab):
        us = jnp.einsum('msr,r->ms', data['G5'], w)
        ah = data['A'] @ w
        lam = data['lam']
        return (ah - prev + dt * (PqT_vec(data, H.advect_points(us, L)) + nu * lam * ah)) / (1 + dt * nu * lam)

    def evalJ_q(w, args):
        prev, nu, data, tab = args
        us = jnp.einsum('msr,r->ms', data['G5'], w)
        ah = data['A'] @ w
        lam = data['lam']
        S = 1. / (1 + dt * nu * lam)
        r = (ah - prev + dt * (PqT_vec(data, H.advect_points(us, L)) + nu * lam * ah)) * S
        dA = jnp.einsum('ms,msr->mr', dadv(us), data['G5'])
        return r, data['A'] + (dt * S)[:, None] * PqT_cols(data, dA)

    def dproj(data, adv):
        if sep_dense:
            return proj_vec(adv.reshape(-1), data['sepd'])
        return H.sep_project(adv.reshape(L - 1, L - 1), data['sx'], data['sy'], L)

    def res_d(w, prev, nu, data, tab):
        adv = e.spatial(H.bank_apply(data['G'], w), L)[0].reshape(L - 1, L - 1)
        ah = data['A'] @ w
        lam = data['lam']
        return (ah - prev + dt * (dproj(data, adv) + nu * lam * ah)) / (1 + dt * nu * lam)

    def evalJ_d_lin(w, args):
        prev, nu, data, tab = args
        r, lin = jax.linearize(lambda ww: res_d(ww, prev, nu, data, tab), w)
        basis = jnp.eye(Rp).reshape(tangent_chunks, Rp // tangent_chunks, Rp)
        J = jax.lax.map(lambda B: jax.vmap(lin)(B), basis).reshape(Rp, -1).T
        return r, J

    def evalJ_d_ana(w, args):
        prev, nu, data, tab = args
        Gf = data['Gf']                                              # (L-1, L-1, R')
        u = jnp.einsum('ijr,r->ij', Gf, w)
        p = jnp.pad(u, 1)
        st = jnp.stack((p[1:-1, 1:-1], p[2:, 1:-1], p[:-2, 1:-1], p[1:-1, 2:], p[1:-1, :-2]), -1)  # c xp xm yp ym
        adv = H.advect_points(st, L)
        D = dadv(st.reshape(-1, 5)).reshape(L - 1, L - 1, 5)
        Gp = jnp.pad(Gf, ((1, 1), (1, 1), (0, 0)))
        dAdv = (D[..., 0:1] * Gf + D[..., 1:2] * Gp[2:, 1:-1] + D[..., 2:3] * Gp[:-2, 1:-1]
                + D[..., 3:4] * Gp[1:-1, 2:] + D[..., 4:5] * Gp[1:-1, :-2])
        ah = data['A'] @ w
        lam = data['lam']
        S = 1. / (1 + dt * nu * lam)
        if sep_dense:
            pa, pJ = proj_vec(adv.reshape(-1), data['sepd']), proj_cols(dAdv, data['sepd'])
        else:
            pa = H.sep_project(adv, data['sx'], data['sy'], L)
            pJ = H.sep_project(jnp.moveaxis(dAdv, -1, 0), data['sx'], data['sy'], L).T
        r = (ah - prev + dt * (pa + nu * lam * ah)) * S
        return r, data['A'] + (dt * S)[:, None] * pJ

    evalJ_d = evalJ_d_ana if ajac_dense else evalJ_d_lin
    lm_q = make_fused_lm(evalJ_q, Rp, 0, step_budget, trust, gtol, ridge, solver, clip, cap=lm_cap)
    lm_d = make_fused_lm(evalJ_d, Rp, 0, step_budget, trust, gtol, ridge, solver, clip, cap=lm_cap)

    def initialize(u0, data, cold, tab):
        xy, wq, Q, R, _, _, _ = cold
        ui = e.sample_field(u0, xy, L) * wq
        yv = Q.T @ ui
        w0 = jax.scipy.linalg.solve_triangular(R, yv, lower=False)
        scale = jnp.linalg.norm(ui) * jnp.sqrt(len(wq))
        icrn = jnp.linalg.norm(R @ w0 - yv)
        z = jnp.zeros(())
        return w0, scale, jnp.int32(0), jnp.int32(1), z, icrn, jnp.linalg.norm(ui), z

    def make_step(res, lm, nu, scale, data, tab):
        def step(carry, k):
            wv, wprev, wprev2, lam0 = carry
            p = data['A'] @ wv
            we = wv + (wv - wprev)
            if predictor == 'quad':
                wq = jnp.where(k >= 2, 3. * wv - 3. * wprev + wprev2, we)
                cand = jnp.stack((wv, we, wq))
                rs = jax.vmap(lambda ww: jnp.linalg.norm(res(ww, p, nu, data, tab)))(cand)
                rs = jnp.where(jnp.isfinite(rs), rs, jnp.inf)
                wi = cand[jnp.argmin(rs)]
            else:
                r0 = jnp.linalg.norm(res(wv, p, nu, data, tab))
                re = jnp.linalg.norm(res(we, p, nu, data, tab))
                wi = jnp.where(jnp.isfinite(re) & (re < r0), we, wv)
            w2, rn, it, reason, gn, rej, lam = lm(wi, (p, nu, data, tab), 1e-9 * scale, lam0)
            return (w2, wv, wprev, lam if lam_carry else lam0), (w2, rn, it, reason, gn, rej)
        return step

    def evolve(w0, nu, scale, data, tab):
        carry = (w0, w0, w0, jnp.asarray(1e-6, dtype=jnp.float64))
        if exact_steps:
            carry, o1 = jax.lax.scan(make_step(res_d, lm_d, nu, scale, data, tab), carry, jnp.arange(exact_steps))
            _, o2 = jax.lax.scan(make_step(res_q, lm_q, nu, scale, data, tab), carry, jnp.arange(exact_steps, steps))
            return tuple(jnp.concatenate((a, b)) for a, b in zip(o1, o2))
        _, out = jax.lax.scan(make_step(res_q, lm_q, nu, scale, data, tab), carry, jnp.arange(steps))
        return out

    def decode_fields(W, data, tab):
        U = H.bank_apply(data['G'], W.T).T.reshape(len(W), L - 1, L - 1)
        return jnp.pad(U, ((0, 0), (1, 1), (1, 1)))

    def query(u0, nu, data, cold, tab):
        w0, scale, icit, icreason, icgn, icrn, uin, icgj = initialize(u0, data, cold, tab)
        ws, rn, it, reason, gn, rej = evolve(w0, nu, scale, data, tab)
        internal = jnp.concatenate((w0[None], ws))
        W = internal[::keep]
        fields = decode_fields(W, data, tab)
        return (fields, it, rn, reason, W, icit, icreason, internal, gn, icgn, icrn, uin, gn, icgj, gn, rej)

    return jax.jit(query, compiler_options=compiler_options) if compiler_options else jax.jit(query)


# ------------------------------------------------- the replaced general path ----

def make_linear_query_general(Rp, L, dt, trust, step_budget=600, gtol=1e-6, ridge=1e-10):
    """The GENERAL solver path the fast path replaced, on the linear rung (coordinator request, paper section 6.3):
    varpro.make_block_lm verbatim (forward-mode Jacobian of the residual by jax.jacfwd, LU solve of the damped
    normal equation, an over-long step REJECTED rather than clipped, the residual evaluated first and residual +
    Jacobian re-evaluated under lax.cond on acceptance, damping restarted at 1e-6 every time step) and the retained
    two-way linear predictor guard. Same weak residual, same dense Pq, same rule, trust radius, gtol, budget,
    initial fit and decoder as bkfast.make_linear_query. The per-step `blocks()` stationarity diagnostic of the old
    audited query is NOT included (it would only make this path slower)."""
    import varpro as VP
    steps = int(round(.25 / dt))
    keep = int(round(.05 / dt))

    def res(z, y, prev, nu, data):
        us = jnp.einsum('msr,r->ms', data['G5'], z)
        ah = data['A'] @ z
        lam = data['lam']
        return (ah - prev + dt * (data['Pq'].T @ H.advect_points(us, L) + nu * lam * ah)) / (1 + dt * nu * lam)

    lm = VP.make_block_lm(res, Rp, 0, step_budget, trust, gtol, ridge)
    y0 = jnp.zeros((0,))

    def query(u0, nu, data, cold, tab):
        xy, wq, Q, R, _, _, _ = cold
        ui = e.sample_field(u0, xy, L) * wq
        yv = Q.T @ ui
        w0 = jax.scipy.linalg.solve_triangular(R, yv, lower=False)
        scale = jnp.linalg.norm(ui) * jnp.sqrt(len(wq))
        icrn = jnp.linalg.norm(R @ w0 - yv)

        def step(carry, _):
            wv, wprev = carry
            p = data['A'] @ wv
            we = wv + (wv - wprev)
            r0 = jnp.linalg.norm(res(wv, y0, p, nu, data))
            re = jnp.linalg.norm(res(we, y0, p, nu, data))
            wi = jnp.where(jnp.isfinite(re) & (re < r0), we, wv)
            z2, _, rn, it, reason, gn = lm(wi, y0, (p, nu, data), 1e-9 * scale)
            return (z2, wv), (z2, rn, it, reason, gn, jnp.int32(0))

        _, (ws, rn, it, reason, gn, rej) = jax.lax.scan(step, (w0, w0), None, length=steps)
        internal = jnp.concatenate((w0[None], ws))
        W = internal[::keep]
        U = H.bank_apply(data['G'], W.T).T.reshape(len(W), L - 1, L - 1)
        fields = jnp.pad(U, ((0, 0), (1, 1), (1, 1)))
        z = jnp.zeros(())
        return (fields, it, rn, reason, W, jnp.int32(0), jnp.int32(1), internal, gn, z, icrn, jnp.linalg.norm(ui),
                gn, z, gn, rej)

    return jax.jit(query)
