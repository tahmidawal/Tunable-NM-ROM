"""Lattice-contract ROM query for the 3D driver (DESIGN A1.2).

`make_fsc_rule_lat` is a verbatim copy of vendored offmesh.make_fsc_rule (quadrature-burgers3d @aab6b6927) with ONE change:
the six output fields are decoded on the 63^3 scoring lattice x = k/64 (`data['G65']`, the rotated bank rows at the lattice
nodes, (R', 63^3)) instead of the full mesh. The projection of u0 on the mesh and the LM solve are untouched.
Gate G-lat (plateau3d.py): coefficients and lattice fields equal the vendored query's on every timing case.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from offmesh import C, TB, DT   # noqa: F401  (names used by the copied body)


def make_fsc_rule_lat(n, Rp, M, contract, dt=DT, gtol=1e-3, trust=jnp.inf, adaptive_first=3, linear_jac=True,
                  unroll=True):
    """Identical to vendor tables.make_fsc except that contract(data, w) is a parameter, and, when the rule's Jacobian
    is not linear in c (linear_jac=False, dense upwind), the predictor evaluates Ju at each candidate."""
    steps, keep = int(round(0.25 / dt)), int(round(0.05 / dt))
    assert abs(steps * dt - 0.25) < 1e-12 and abs(keep * dt - 0.05) < 1e-12

    def query(u0, nu, data, hp):
        A, lam_ = data['A'], data['lam']
        S = 1.0 / (1.0 + dt * nu * lam_)
        w0, s0 = TB.project(u0, data)
        ic = jnp.stack([0., 4., 0., jnp.linalg.norm(u0) ** 2 - jnp.linalg.norm(s0) ** 2])
        rj = TB._lm_step_parts(A, S, lam_, dt, nu)

        def sweep(w, Ju, r, J, rn, lam, p):
            Hm = J.T @ J
            g = J.T @ r
            d = jnp.diag(Hm) + 1e-30
            cf = jax.scipy.linalg.cho_factor(Hm + jnp.diag(lam * d), lower=True)
            dw = jax.scipy.linalg.cho_solve(cf, -g)
            nz = jnp.linalg.norm(dw)
            dw = dw * jnp.where(nz > trust, trust / (nz + 1e-300), 1.)
            ok = jnp.all(jnp.isfinite(dw))
            wn = w + jnp.where(ok, dw, 0.)
            Jun = contract(data, wn)
            r2, J2 = rj(Jun, wn, p)
            rn2 = jnp.linalg.norm(r2)
            acc = ok & jnp.isfinite(rn2) & (rn2 < rn)
            lam2 = jnp.where(acc, jnp.maximum(lam / 3, 1e-12), jnp.minimum(lam * 10, 1e14))
            sel = lambda a, b: jnp.where(acc, a, b)
            return sel(wn, w), sel(Jun, Ju), sel(r2, r), sel(J2, J), sel(rn2, rn), lam2, (~acc).astype(jnp.int32)

        def predict(k, wv, wp, wp2, Jv, Jp, Jp2, p):
            we = 2. * wv - wp
            wq = jnp.where(k >= 2, 3. * wv - 3. * wp + wp2, we)
            cand = jnp.stack((wv, we, wq))
            if linear_jac:
                Je = 2. * Jv - Jp
                Jq = jnp.where(k >= 2, 3. * Jv - 3. * Jp + Jp2, Je)
                Ju3 = jnp.stack((Jv, Je, Jq))
            else:
                Ju3 = jax.lax.map(lambda c: contract(data, c), cand)
            rs = jnp.linalg.norm(jax.vmap(lambda Ju, c: rj(Ju, c, p)[0])(Ju3, cand), axis=1)
            i = jnp.argmin(jnp.where(jnp.isfinite(rs), rs, jnp.inf))
            return cand[i], Ju3[i]

        def step(carry, k):
            wv, wp, wp2, Jv, Jp, Jp2, lam = carry
            p = A @ wv
            w, Ju = predict(k, wv, wp, wp2, Jv, Jp, Jp2, p)
            r, J = rj(Ju, w, p)
            w, Ju, r, J, rn, lam, rej = sweep(w, Ju, r, J, jnp.linalg.norm(r), lam, p)
            gn = jnp.linalg.norm(J.T @ r) / (jnp.linalg.norm(J) * rn + 1e-300)
            reason = jnp.where(jnp.isfinite(rn), jnp.where(gn <= gtol, 4, 0), 3).astype(jnp.int32)
            return (w, wv, wp, Ju, Jv, Jp, lam), (w, rn, jnp.int32(1), reason, gn, rej)

        lmA = C.make_fused_lm(lambda w, p_: rj(contract(data, w), w, p_), Rp, 50, trust, gtol, clip=True)

        def astep(carry, k):
            wv, wp, wp2, Jv, Jp, Jp2, lam = carry
            p = A @ wv
            wi, _ = predict(k, wv, wp, wp2, Jv, Jp, Jp2, p)
            w2, rn, it, reason, gn, rej, lam2 = lmA(wi, p, 1e-12 * jnp.linalg.norm(s0), lam)
            return (w2, wv, wp, contract(data, w2), Jv, Jp, lam2), (w2, rn, it, reason, gn, rej)

        J0 = contract(data, w0)
        carry = (w0, w0, w0, J0, J0, J0, jnp.asarray(1e-6, dtype=jnp.float64))
        carry, o1 = jax.lax.scan(astep, carry, jnp.arange(adaptive_first))
        _, o2 = jax.lax.scan(step, carry, jnp.arange(adaptive_first, steps), unroll=unroll)
        ws, rn, it, reason, gn, rej = (jnp.concatenate((x, y)) for x, y in zip(o1, o2))
        internal = jnp.concatenate((w0[None], ws))
        return internal[::keep] @ data['G65'], internal, it, reason, gn, rn, rej, ic   # LANE CHANGE: 63^3 lattice
    return jax.jit(query)
