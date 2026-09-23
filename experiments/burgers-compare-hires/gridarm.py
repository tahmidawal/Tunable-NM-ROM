"""The dense-residual POD-LSPG / quadratic-manifold arm of `arms.make_query`, without Phi and with the bank in
row blocks (burgers-compare-hires DESIGN.md section 5).

`arms.make_query(head, K, L, dt, trust, 'dense')` with `arms.build_operators(GridBank(B), L, M, 'dense')` is the
arm the 256^2 panel (qmn102) timed. At 1024^2-2048^2 it cannot run as written: it materialises the (n, M) test
matrix Phi (68 GB for POD-512 at 2048^2), and every product with an operand of more than 2^31 elements fails
XLA's Triton-gemm autotuning (hires-burgers hb4k02). This module is that arm with three changes, each an
identity in exact arithmetic:

  * Phi^T v by the separable sine projection of `hops.sep_project` (as the NM-ROM arms already do);
  * the bank B as a tuple of row blocks (`hops.bank_apply`), each under the gemm limit;
  * the Jacobian as jvp's over chunks of the identity (`lax.map` of `vmap`), so the tangent batch never holds
    n x dim at once. `jax.jacfwd` is `vmap(jvp)` over the whole identity; chunking changes memory, not values.

Everything else -- the weak residual, the test count, the fixed-Gauss initializer, the extrapolated first
guess, the stationary LM and its stopping rule, the output contract -- is `arms`' code, copied. The driver runs
the ORIGINAL `arms` arm beside this one at small rank on the real mesh (the `grid_arm_parity` gate).
"""
from __future__ import annotations

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e
import hops as H


class BlockGridBank:
    """Grid-resident bank B (n, D) held as row blocks; `at` samples it off-grid exactly as `arms.GridBank.at`."""

    kind = 'grid'

    def __init__(self, Bhost, L, chunk=64):
        self.L = int(L)
        n, D = Bhost.shape
        assert n == (L - 1) ** 2
        self.dim = int(D)
        nb = max(1, int(np.ceil(n * D / H.MAX_GEMM_ELEMENTS)))
        # block edges on whole grid rows, like hops.build_bank
        edges = (np.linspace(0, L - 1, nb + 1).astype(int)) * (L - 1)
        self.blocks = tuple(jax.block_until_ready(jnp.asarray(np.ascontiguousarray(Bhost[s:t])))
                            for s, t in zip(edges[:-1], edges[1:]))
        self.chunk = int(chunk)

    def at(self, xy, chunk=8192):
        xy = jnp.asarray(np.asarray(xy))
        L = self.L
        out = []
        for c0 in range(0, self.dim, self.chunk):
            col = jnp.concatenate([g[:, c0:c0 + self.chunk] for g in self.blocks], axis=0)
            pad = jnp.pad(col.T.reshape(-1, L - 1, L - 1), ((0, 0), (1, 1), (1, 1)))
            out.append(jax.vmap(lambda f: e.sample_field(f, xy, L), out_axes=1)(pad))
            del col, pad
        return jnp.concatenate(out, axis=1)

    @property
    def nbytes(self):
        return int(sum(g.nbytes for g in self.blocks))


def build_operators(gb, M):
    L = gb.L
    kx, ky, lam = H.modes_lean(L, M)
    sx, sy = H.sine_tables(L, kx, ky)
    sxd, syd = jnp.asarray(sx), jnp.asarray(sy)
    A = H.project_bank(gb.blocks, sxd, syd, L)
    data = dict(A=A, lam=jnp.asarray(lam), Gb=gb.blocks, sx=sxd, sy=syd)
    jax.block_until_ready(A)
    return data, dict(M=int(M), array_bytes=int(gb.nbytes + A.nbytes + sxd.nbytes + syd.nbytes), phi_free=True,
                      row_blocks=len(gb.blocks))


def weak_dense_pf(z, prev, nu, data, head, L, dt):
    """arms.weak_dense with Phi^T by separable projection and G @ h by row blocks."""
    h = head(z)
    adv, _ = e.spatial(H.bank_apply(data['Gb'], h), L)
    pa = H.sep_project(adv.reshape(L - 1, L - 1), data['sx'], data['sy'], L)
    ah = data['A'] @ h
    lam = data['lam']
    return (ah - prev + dt * (pa + nu * lam * ah)) / (1 + dt * nu * lam)


def chunked_jacobian(fun, chunk):
    def jac(z, *args):
        d = z.shape[0]
        c = min(int(chunk), d)
        assert d % c == 0, (d, c)
        basis = jnp.eye(d, dtype=z.dtype).reshape(d // c, c, d)
        one = lambda Ec: jax.vmap(lambda v: jax.jvp(lambda zz: fun(zz, *args), (z,), (v,))[1])(Ec)
        Jt = jax.lax.map(one, basis)                       # (d/c, c, m)
        return Jt.reshape(d, -1).T
    return jac


def make_stationary_lm(fun, budget, trust=np.inf, gtol=1e-6, linear='gj', chunk=64):
    """arms.make_stationary_lm verbatim except the Jacobian (chunked jvp instead of jacfwd)."""
    solve = e.gj_solve if linear == 'gj' else jnp.linalg.solve
    jacobian = chunked_jacobian(fun, chunk)

    def lm(z0, args, tol):
        def evaluate(z):
            r = fun(z, *args)
            J = jacobian(z, *args)
            return r, J, jnp.linalg.norm(r)

        def grad(r, J):
            return jnp.linalg.norm(J.T @ r) / (jnp.linalg.norm(J) * jnp.linalg.norm(r) + 1e-300)

        r, J, rn = evaluate(z0)
        reason = jnp.where(jnp.isfinite(rn),
                           jnp.where(grad(r, J) <= gtol, 4, jnp.where(rn <= tol, 1, 0)), 3).astype(jnp.int32)

        def body(s):
            z, r, J, rn, lam, it, reason = s
            Hm = J.T @ J
            g = J.T @ r
            dz = solve(Hm + lam * jnp.diag(jnp.diag(Hm) + 1e-30), -g)
            ok = jnp.all(jnp.isfinite(dz)) & (jnp.linalg.norm(dz) <= trust)
            zn = z + jnp.where(ok, dz, 0.)
            rn2 = jnp.linalg.norm(fun(zn, *args))
            accept = ok & jnp.isfinite(rn2) & (rn2 < rn)
            r2, J2, rn2 = jax.lax.cond(accept, lambda: evaluate(zn), lambda: (r, J, rn))
            gn = grad(r2, J2)
            tiny = ok & (jnp.linalg.norm(dz) <= 1e-14 * (1 + jnp.linalg.norm(z)))
            reason = jnp.where(gn <= gtol, 4,
                      jnp.where(rn2 <= tol, 1,
                       jnp.where(tiny, 2, jnp.where((~accept) & (lam >= 1e14), 3, 0)))).astype(jnp.int32)
            return (jnp.where(accept, zn, z), r2, J2, rn2,
                    jnp.where(accept, jnp.maximum(lam / 3, 1e-12), jnp.minimum(lam * 10, 1e14)),
                    it + 1, reason)

        z, r, J, rn, lam, it, reason = jax.lax.while_loop(
            lambda s: (s[5] < budget) & (s[6] == 0), body,
            (z0, r, J, rn, jnp.asarray(1e-6), jnp.int32(0), reason))
        return z, rn, it, reason, grad(r, J)
    return lm


def make_query(head, K, L, dt, trust, ic_budget=400, step_budget=180, gtol=1e-6, linear='gj', chunk=64):
    """arms.make_query(..., 'dense') on the Phi-free, row-blocked operators; same outputs, same order."""
    wk = weak_dense_pf
    # the state-fitting initializer is small (Gauss points x D): arms' own LM, jacfwd and all
    import arms as A
    ic = A.make_stationary_lm(lambda z, y, R: R @ head(z) - y, ic_budget, gtol=gtol, linear=linear)
    lm = make_stationary_lm(lambda z, p, nu, data: wk(z, p, nu, data, head, L, dt),
                            step_budget, trust, gtol, linear, chunk)

    def query(u0, nu, data, cold):
        xy, w, Q, R, Hrot, Hnorm, Zcand = cold
        ui = e.sample_field(u0, xy, L) * w
        y = Q.T @ ui
        idx = jnp.argmin(Hnorm - 2 * Hrot @ y)
        z, icrn, icit, icreason, icgn = ic(Zcand[idx], (y, R), 0.)
        scale = jnp.linalg.norm(ui) * jnp.sqrt(len(w))

        def step(carry, _):
            z, zprev = carry
            p = data['A'] @ head(z)
            ze = z + (z - zprev)
            r0 = jnp.linalg.norm(wk(z, p, nu, data, head, L, dt))
            re = jnp.linalg.norm(wk(ze, p, nu, data, head, L, dt))
            zi = jnp.where(jnp.isfinite(re) & (re < r0), ze, z)
            z2, rn, it, reason, gn = lm(zi, (p, nu, data), 1e-9 * scale)
            return (z2, z), (z2, rn, it, reason, gn)

        _, (zs, rn, it, reason, gn) = jax.lax.scan(step, (z, z), None, length=int(round(.25 / dt)))
        internal = jnp.concatenate((z[None], zs))
        Z = internal[::int(round(.05 / dt))]
        fields = jax.vmap(lambda z: e.output_field(H.bank_apply(data['Gb'], head(z)), L, L))(Z)
        return (fields, it, rn, reason, Z, icit, icreason, internal, gn, icgn,
                icrn, jnp.linalg.norm(ui))
    return jax.jit(query)
