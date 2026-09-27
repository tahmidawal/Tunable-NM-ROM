"""Kernel-level optimisations of the frozen 2D Burgers empirical-quadrature query.

Every path here computes the SAME weak residual, the same damped Levenberg-Marquardt
iteration, the same stopping rule and the same six dense outputs as
`arms.make_query(..., quadrature='eq', linear='gj')`. Only how the arithmetic is
emitted changes. Two parity classes are declared in `DESIGN.md`:

  bitwise        the identical floating-point expressions are evaluated
                 (`fuse`, `hoist`, `share`, `unroll`, and `block` at bs=1)
  reassociation  algebraically exact, different rounding
                 (`lean`, `block` at bs>1, `probe`, `nodot`, `leandec`)

Nothing in this module changes an iteration budget, a tolerance, an initializer or a
quadrature rule; anything that would is an algorithmic arm and does not belong here.
"""
from __future__ import annotations

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e
import arms as A
import sep_common as sc


DEFAULTS = dict(fuse=0, hoist=0, share=0, unroll=1, probe=0, lean=0, block=1,
                nodot=0, decode='vmap')


def opts(**kw):
    o = dict(DEFAULTS)
    for k, v in kw.items():
        assert k in o, k
        o[k] = v
    return o


def label(o):
    on = [k for k in ('fuse', 'hoist', 'share', 'probe', 'lean', 'nodot') if o[k]]
    if o['unroll'] != 1:
        on.append(f"unroll{o['unroll']}")
    if o['block'] != 1:
        on.append(f"block{o['block']}")
    if o['decode'] != 'vmap':
        on.append(o['decode'])
    return '+'.join(on) if on else 'incumbent'


# ------------------------------------------------------- small linear solve ---

def inv_block(P):
    """Closed-form inverse of a 1x1, 2x2 or 2^k x 2^k matrix by 2x2 block recursion.

    Used only on the pivot blocks of the damped SPD normal matrix, which the incumbent
    already eliminates without pivoting; a degenerate system produces non-finite values
    and is caught by the LM's existing `finite` guard exactly as a failed solve is."""
    n = P.shape[0]
    if n == 1:
        return 1.0 / P
    if n == 2:
        det = P[0, 0] * P[1, 1] - P[0, 1] * P[1, 0]
        return jnp.stack((jnp.stack((P[1, 1], -P[0, 1])),
                          jnp.stack((-P[1, 0], P[0, 0])))) / det
    h = n // 2
    Ai = inv_block(P[:h, :h])
    AiB = Ai @ P[:h, h:]
    CAi = P[h:, :h] @ Ai
    Si = inv_block(P[h:, h:] - P[h:, :h] @ AiB)
    return jnp.concatenate((jnp.concatenate((Ai + AiB @ Si @ CAi, -AiB @ Si), 1),
                            jnp.concatenate((-Si @ CAi, Si), 1)), 0)


def gj_block(mat_a, b, bs):
    """Block Gauss-Jordan: eliminate `bs` unknowns per sequential stage.

    bs=1 is `engines.gj_solve` bitwise (same division, same masked rank-one update);
    bs>1 divides the sequential depth of the (n, n+1) elimination by bs at the cost of
    a closed-form bs x bs pivot inverse per stage."""
    n = mat_a.shape[0]
    assert n % bs == 0, (n, bs)
    mat = jnp.concatenate((mat_a, b[:, None]), 1)
    rows = jnp.arange(n)
    for k0 in range(0, n, bs):
        blk = mat[k0:k0 + bs]
        P = blk[:, k0:k0 + bs]
        row = blk / P if bs == 1 else inv_block(P) @ blk
        mask = (rows >= k0) & (rows < k0 + bs)
        cols = jnp.where(mask[:, None], 0., mat[:, k0:k0 + bs])
        mat = mat - cols @ row
        mat = mat.at[k0:k0 + bs].set(row)
    return mat[:, n]


def make_solve(bs):
    return e.gj_solve if bs == 1 else (lambda H, g: gj_block(H, g, bs))


# ------------------------------------------------------------ head folding ---

def head_split(params):
    """h(z) = a2(z) W3 + b3 + z Wlin  ==>  f(z) = [a2(z); z],  h = Wm^T f + b3."""
    assert 'hB' not in params, 'latent Fourier features are not in the retained checkpoint'
    layers = params['h']
    W3, b3 = layers[-1]
    Wm = jnp.concatenate((W3, params['h_lin']), axis=0)

    def feat(z):
        x = z
        for w, bb in layers[:-1]:
            x = jax.nn.silu(x @ w + bb)
        return jnp.concatenate((x, z))

    return feat, Wm, b3


def feat_nodot(params):
    """The same f(z) with broadcast-reduce matvecs instead of cuBLAS calls."""
    layers = params['h']

    def feat(z):
        x = z
        for w, bb in layers[:-1]:
            x = jax.nn.silu(jnp.sum(x[:, None] * w, 0) + bb)
        return jnp.concatenate((x, z))

    return feat


def head_nodot(params):
    layers = params['h']
    hl = params['h_lin']

    def head(z):
        x = z
        for w, bb in layers[:-1]:
            x = jax.nn.silu(jnp.sum(x[:, None] * w, 0) + bb)
        w, bb = layers[-1]
        return jnp.sum(x[:, None] * w, 0) + bb + jnp.sum(z[:, None] * hl, 0)

    return head


def build_tables(params, data, cold, o):
    """Offline folded operators: pure functions of the frozen weights and the frozen
    operators, built once per mesh and charged to setup, never to a query."""
    tab = {}
    if not (o['lean'] or o['decode'].startswith('lean')):
        return tab
    _, Wm, b3 = head_split(params)
    m, _, R = data['G5'].shape
    if o['lean']:
        stack = jnp.concatenate((data['G5'].reshape(5 * m, R), data['A']), 0)
        tab['S'] = stack @ Wm.T
        tab['s0'] = stack @ b3
        tab['RW'] = cold[3] @ Wm.T
        tab['Rb'] = cold[3] @ b3
    if o['decode'].startswith('lean'):
        tab['GW'] = data['G'] @ Wm.T
        tab['Gb'] = data['G'] @ b3
    return jax.block_until_ready(tab)


# ------------------------------------------------------------- residuals -----

def make_weak(params, L, dt, o, m):
    """Returns (setup, kernel, assemble, project, rescale).

        setup(nu, data)                 -> con, the per-query constants (nu is fixed)
        kernel(z, data, tab)            -> (us, ah), the shared expensive part
        assemble(us, ah, p, con, data)  -> r, the weak residual in R^M
        project(z, data, tab)           -> A h(z) alone, the incumbent's step projection
        rescale(p, con)                 -> the scaling `assemble` expects for p

    Splitting the residual at (us, ah) is what lets one evaluation serve both the
    previous-step projection p = A h(z) and the probe residual at the same z."""
    head = head_nodot(params) if o['nodot'] else (lambda z: sc.head(params, z))
    # mv(W, x) == W^T x, written exactly as the incumbent writes it so the dot path
    # emits the same contraction order.
    mv = (lambda W, x: jnp.sum(x[:, None] * W, 0)) if o['nodot'] else (lambda W, x: W.T @ x)
    feat = (feat_nodot(params) if o['nodot'] else head_split(params)[0]) if o['lean'] else None

    def advect(us):
        c, xp, xm, yp, ym = (us[:, i] for i in range(5))
        return c * L * (jnp.where(c > 0, c - xm, xp - c) + jnp.where(c > 0, c - ym, yp - c))

    if o['lean']:
        def setup(nu, data):
            iden = 1. / (1. + dt * nu * data['lam'])
            return (data['Pq'] * (dt * iden)[None, :], iden)

        def kernel(z, data, tab):
            t = tab['S'] @ feat(z) + tab['s0']
            return t[:5 * m].reshape(m, 5), t[5 * m:]

        def assemble(us, ah, p, con, data):
            return ah - p + mv(con[0], advect(us))

        def project(z, data, tab):
            return tab['S'][5 * m:] @ feat(z) + tab['s0'][5 * m:]

        def rescale(p, con):
            return p * con[1]

    else:
        def setup(nu, data):
            if o['hoist']:
                nl = nu * data['lam']
                return (nl, 1. + dt * nl)
            return (nu, None)

        def kernel(z, data, tab):
            h = head(z)
            us = jnp.einsum('msr,r->ms', data['G5'], h)
            return us, (mv(data['A'].T, h) if o['nodot'] else data['A'] @ h)

        def assemble(us, ah, p, con, data):
            adv = advect(us)
            if o['hoist']:
                nl, den = con
                return (ah - p + dt * (mv(data['Pq'], adv) + nl * ah)) / den
            nu = con[0]
            lam = data['lam']
            return (ah - p + dt * (mv(data['Pq'], adv) + nu * lam * ah)) / (1 + dt * nu * lam)

        def project(z, data, tab):
            h = head(z)
            return mv(data['A'].T, h) if o['nodot'] else data['A'] @ h

        def rescale(p, con):
            return p

    return setup, kernel, assemble, project, rescale


def make_ic_residual(params, o):
    """R_cold h(z) - y, folded when `lean` is on."""
    if o['lean']:
        feat = feat_nodot(params) if o['nodot'] else head_split(params)[0]
        return lambda z, y, tab, cold: tab['RW'] @ feat(z) + tab['Rb'] - y
    head = head_nodot(params) if o['nodot'] else (lambda z: sc.head(params, z))
    return lambda z, y, tab, cold: cold[3] @ head(z) - y


# ------------------------------------------------------------------ the LM ---

def make_lm(fun, K, budget, o, trust=np.inf, gtol=1e-6):
    """`arms.make_stationary_lm` with the optimisations applied.

    `fuse` replaces (the primal used only for the accept test) plus (the `lax.cond`
    re-evaluation) by ONE `jax.linearize`, whose primal IS fun(zn) and whose vmapped
    JVPs ARE jacfwd's columns, and reuses J^T r for both the stationarity ratio and the
    next iteration's gradient."""
    solve = make_solve(o['block'])
    eye = jnp.eye(K)

    def evaluate(z, args):
        if o['fuse']:
            r, jvp = jax.linearize(lambda q: fun(q, *args), z)
            return r, jax.vmap(jvp, out_axes=1)(eye)
        return fun(z, *args), jax.jacfwd(fun)(z, *args)

    def gram(J):
        if o['nodot']:
            return jnp.sum(J[:, :, None] * J[:, None, :], 0)
        return J.T @ J

    def grad_vec(J, r):
        if o['nodot']:
            return jnp.sum(J * r[:, None], 0)
        return J.T @ r

    def ratio(J, r, g):
        return jnp.linalg.norm(g) / (jnp.linalg.norm(J) * jnp.linalg.norm(r) + 1e-300)

    def lm(z0, args, tol):
        r, J = evaluate(z0, args)
        rn = jnp.linalg.norm(r)
        g = grad_vec(J, r)
        reason = jnp.where(jnp.isfinite(rn),
                           jnp.where(ratio(J, r, g) <= gtol, 4,
                                     jnp.where(rn <= tol, 1, 0)), 3).astype(jnp.int32)

        def body(s):
            z, r, J, rn, g, lam, it, reason = s
            H = gram(J)
            dz = solve(H + lam * jnp.diag(jnp.diag(H) + 1e-30), -g)
            ok = jnp.all(jnp.isfinite(dz)) & (jnp.linalg.norm(dz) <= trust)
            zn = z + jnp.where(ok, dz, 0.)
            if o['fuse']:
                rc, Jc = evaluate(zn, args)
                rnc = jnp.linalg.norm(rc)
                accept = ok & jnp.isfinite(rnc) & (rnc < rn)
                r2 = jnp.where(accept, rc, r)
                J2 = jnp.where(accept, Jc, J)
                rn2 = jnp.where(accept, rnc, rn)
            else:
                rn2 = jnp.linalg.norm(fun(zn, *args))
                accept = ok & jnp.isfinite(rn2) & (rn2 < rn)

                def fresh():
                    rc, Jc = evaluate(zn, args)
                    return rc, Jc, jnp.linalg.norm(rc)

                r2, J2, rn2 = jax.lax.cond(accept, fresh, lambda: (r, J, rn))
            g2 = grad_vec(J2, r2)
            gn = ratio(J2, r2, g2)
            tiny = ok & (jnp.linalg.norm(dz) <= 1e-14 * (1 + jnp.linalg.norm(z)))
            reason = jnp.where(gn <= gtol, 4,
                               jnp.where(rn2 <= tol, 1,
                                         jnp.where(tiny, 2,
                                                   jnp.where((~accept) & (lam >= 1e14), 3, 0)))).astype(jnp.int32)
            return (jnp.where(accept, zn, z), r2, J2, rn2, g2,
                    jnp.where(accept, jnp.maximum(lam / 3, 1e-12), jnp.minimum(lam * 10, 1e14)),
                    it + 1, reason)

        z, r, J, rn, g, lam, it, reason = jax.lax.while_loop(
            lambda s: (s[6] < budget) & (s[7] == 0), body,
            (z0, r, J, rn, g, jnp.asarray(1e-6), jnp.int32(0), reason))
        return z, rn, it, reason, ratio(J, r, g)

    return lm


# --------------------------------------------------------------- the query ---

def make_parts(params, K, L, dt, m, trust, o, ic_budget=400, step_budget=180, gtol=1e-6):
    """The three phases as closures; `make_query` fuses them into one jit and the
    profile compiles them individually."""
    setup, kernel, assemble, project, rescale = make_weak(params, L, dt, o, m)
    ic_res = make_ic_residual(params, o)
    head = head_nodot(params) if o['nodot'] else (lambda z: sc.head(params, z))
    steps = int(round(.25 / dt))
    stride = int(round(.05 / dt))

    def residual(z, p, con, data, tab):
        us, ah = kernel(z, data, tab)
        return assemble(us, ah, p, con, data)

    ic = make_lm(ic_res, K, ic_budget, o, gtol=gtol)
    lm = make_lm(residual, K, step_budget, o, trust, gtol)

    def initialize(u0, data, cold, tab):
        xy, w, Q, R, Hrot, Hnorm, Zcand = cold
        ui = e.sample_field(u0, xy, L) * w
        y = Q.T @ ui
        idx = jnp.argmin(Hnorm - 2 * Hrot @ y)
        z, icrn, icit, icreason, icgn = ic(Zcand[idx], (y, tab, cold), 0.)
        return z, icrn, icit, icreason, icgn, jnp.linalg.norm(ui) * jnp.sqrt(len(w)), jnp.linalg.norm(ui)

    def evolve(z, nu, scale, data, tab):
        con = setup(nu, data)

        def step(carry, _):
            z, zprev = carry
            ze = z + (z - zprev)
            if o['probe']:
                usb, ahb = jax.vmap(lambda q: kernel(q, data, tab))(jnp.stack((z, ze)))
                p = rescale(ahb[0], con)
                both = jax.vmap(lambda u, a: jnp.linalg.norm(assemble(u, a, p, con, data)))(usb, ahb)
                r0, re = both[0], both[1]
            elif o['share']:
                usz, ahz = kernel(z, data, tab)
                p = rescale(ahz, con)
                r0 = jnp.linalg.norm(assemble(usz, ahz, p, con, data))
                re = jnp.linalg.norm(residual(ze, p, con, data, tab))
            else:
                p = rescale(project(z, data, tab), con)
                r0 = jnp.linalg.norm(residual(z, p, con, data, tab))
                re = jnp.linalg.norm(residual(ze, p, con, data, tab))
            zi = jnp.where(jnp.isfinite(re) & (re < r0), ze, z)
            z2, rn, it, reason, gn = lm(zi, (p, con, data, tab), 1e-9 * scale)
            return (z2, z), (z2, rn, it, reason, gn)

        _, out = jax.lax.scan(step, (z, z), None, length=steps, unroll=o['unroll'])
        return out

    def decode(Z, data, tab):
        mode = o['decode']
        if mode == 'vmap':
            fields = jax.vmap(lambda z: e.output_field(data['G'] @ head(z), L, L))(Z)
        elif mode == 'fused':
            U = data['G'] @ jax.vmap(head)(Z).T
            fields = jax.vmap(lambda c: e.output_field(c, L, L), in_axes=1)(U)
        elif mode.startswith('lean'):
            feat = feat_nodot(params) if o['nodot'] else head_split(params)[0]
            U = tab['GW'] @ jax.vmap(feat)(Z).T + tab['Gb'][:, None]
            fields = jax.vmap(lambda c: e.output_field(c, L, L), in_axes=1)(U)
        else:
            raise AssertionError(mode)
        return fields.astype(jnp.float32) if mode.endswith('_f32') else fields

    def query(u0, nu, data, cold, tab):
        z, icrn, icit, icreason, icgn, scale, uin = initialize(u0, data, cold, tab)
        zs, rn, it, reason, gn = evolve(z, nu, scale, data, tab)
        internal = jnp.concatenate((z[None], zs))
        Z = internal[::stride]
        return (decode(Z, data, tab), it, rn, reason, Z, icit, icreason, internal,
                gn, icgn, icrn, uin)

    return dict(query=query, initialize=initialize, evolve=evolve, decode=decode,
                residual=residual, kernel=kernel, assemble=assemble, setup=setup,
                project=project, rescale=rescale, lm=lm, head=head, steps=steps,
                stride=stride)


def make_query(params, K, L, dt, m, trust, o, **kw):
    return jax.jit(make_parts(params, K, L, dt, m, trust, o, **kw)['query'])


# ------------------------------------------------ diagnostic: fixed budget ---

def make_fixed_budget_evolve(params, K, L, dt, trust, budget):
    """Unconditional fixed-budget LM evolution: a DIAGNOSTIC used only to fit the
    per-iteration marginal cost. Capping iterations changes the answer (the 2026-09-14
    tuning study measured the cliff) and is never a production arm."""
    head = lambda z: sc.head(params, z)

    def fun(z, p, nu, data):
        return A.weak_eq(z, p, nu, data, head, L, dt)

    steps = int(round(.25 / dt))

    def lm(z0, args):
        def body(_, s):
            z, r, J, rn, lam = s
            H = J.T @ J
            g = J.T @ r
            dz = e.gj_solve(H + lam * jnp.diag(jnp.diag(H) + 1e-30), -g)
            ok = jnp.all(jnp.isfinite(dz)) & (jnp.linalg.norm(dz) <= trust)
            zn = z + jnp.where(ok, dz, 0.)
            rn2 = jnp.linalg.norm(fun(zn, *args))
            accept = ok & jnp.isfinite(rn2) & (rn2 < rn)
            r2, J2, rn2 = jax.lax.cond(
                accept, lambda: (fun(zn, *args), jax.jacfwd(fun)(zn, *args), rn2),
                lambda: (r, J, rn))
            return (jnp.where(accept, zn, z), r2, J2, rn2,
                    jnp.where(accept, jnp.maximum(lam / 3, 1e-12), jnp.minimum(lam * 10, 1e14)))

        r = fun(z0, *args)
        J = jax.jacfwd(fun)(z0, *args)
        s = jax.lax.fori_loop(0, budget, body, (z0, r, J, jnp.linalg.norm(r), jnp.asarray(1e-6)))
        return s[0]

    def evolve(z, nu, scale, data):
        def step(carry, _):
            z, zprev = carry
            p = data['A'] @ head(z)
            ze = z + (z - zprev)
            r0 = jnp.linalg.norm(fun(z, p, nu, data))
            re = jnp.linalg.norm(fun(ze, p, nu, data))
            zi = jnp.where(jnp.isfinite(re) & (re < r0), ze, z)
            return (lm(zi, (p, nu, data)), z), 0

        (zf, _), _ = jax.lax.scan(step, (z, z), None, length=steps)
        return zf

    return jax.jit(evolve)


# --------------------------------------------- diagnostic: phase splitting ---

def make_incumbent_parts(params, K, L, dt, trust, ic_budget=400, step_budget=180, gtol=1e-6):
    """`arms.make_query` split at its phase boundaries, arithmetic untouched, so the
    profile can time the initial fit, the evolution and the decode individually."""
    head = lambda z: sc.head(params, z)
    ic = A.make_stationary_lm(lambda z, y, R: R @ head(z) - y, ic_budget, gtol=gtol)
    lm = A.make_stationary_lm(
        lambda z, p, nu, data: A.weak_eq(z, p, nu, data, head, L, dt), step_budget, trust, gtol)
    steps = int(round(.25 / dt))
    stride = int(round(.05 / dt))

    def initialize(u0, data, cold):
        xy, w, Q, R, Hrot, Hnorm, Zcand = cold
        ui = e.sample_field(u0, xy, L) * w
        y = Q.T @ ui
        idx = jnp.argmin(Hnorm - 2 * Hrot @ y)
        z, icrn, icit, icreason, icgn = ic(Zcand[idx], (y, R), 0.)
        return z, jnp.linalg.norm(ui) * jnp.sqrt(len(w))

    def evolve(z, nu, scale, data):
        def step(carry, _):
            z, zprev = carry
            p = data['A'] @ head(z)
            ze = z + (z - zprev)
            r0 = jnp.linalg.norm(A.weak_eq(z, p, nu, data, head, L, dt))
            re = jnp.linalg.norm(A.weak_eq(ze, p, nu, data, head, L, dt))
            zi = jnp.where(jnp.isfinite(re) & (re < r0), ze, z)
            z2, rn, it, reason, gn = lm(zi, (p, nu, data), 1e-9 * scale)
            return (z2, z), z2

        _, zs = jax.lax.scan(step, (z, z), None, length=steps)
        return jnp.concatenate((z[None], zs))

    def decode(Z, data):
        return jax.vmap(lambda z: e.output_field(data['G'] @ head(z), L, L))(Z)

    return dict(initialize=initialize, evolve=evolve, decode=decode, stride=stride)
