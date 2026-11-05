"""Lattice-contract ROM query for the 2D driver (DESIGN A1.2).

`make_linear_query_lat` is a verbatim copy of vendored qcore.make_linear_query (quadrature-study @84d28e303) with ONE change:
`decode_fields` maps the output coefficients to the 257^2 scoring lattice (bank rows `data['Glat']` at the interior lattice
nodes x = i/256, walls padded) instead of the full mesh. The solve (initial fit on the mesh field, LM steps) is untouched.
Gate G-lat (plateau2d.py): coefficient trajectory and lattice fields equal the vendored query's on every timing case.
The body was produced by exact text substitution of that one function (verified in the commit that added this file).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from qcore import e, H, HF, tested_value, tested_jac   # noqa: F401  (names used by the copied body)


def lattice_bank(model, Rp, chunk=8192):
    """Rotated bank at the interior scoring-lattice nodes x = i/256, i = 1..255 (x slowest): (255^2, R')."""
    x = np.arange(1, 256) / 256
    xy = np.stack(np.meshgrid(x, x, indexing='ij'), -1).reshape(-1, 2)
    g = model.base.at(xy, chunk=chunk)
    return jax.block_until_ready(g @ jnp.asarray(model.T[:, :Rp]))


def make_linear_query_lat(kind, Rp, L, dt, trust, step_budget=600, gtol=1e-3, ridge=1e-10, solver='chol',
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

    def decode_fields(W, data):                                     # LANE CHANGE: lattice rows, (255, 255) interior
        U = (W @ data['Glat'].T).reshape(len(W), 255, 255)
        return jnp.pad(U, ((0, 0), (1, 1), (1, 1)))

    def query(u0, nu, data, cold):
        w0, scale = initialize(u0, data, cold)
        ws, rn, it, reason, gn, rej = evolve(w0, nu, scale, data)
        internal = jnp.concatenate((w0[None], ws))
        fields = decode_fields(internal[::keep], data)
        return dict(fields=fields, it=it, rn=rn, reason=reason, gn=gn, rej=rej, internal=internal)

    return jax.jit(query), jax.jit(decode_fields)
