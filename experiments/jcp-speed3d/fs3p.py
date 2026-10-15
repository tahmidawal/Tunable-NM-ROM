"""jcp-speed3d fs3p (DESIGN A1.3): copy of jcp-combined fs3.py with two extra predictor candidates (static flag
`extra`): P3 cubic extrapolation 4c_n-6c_{n-1}+4c_{n-2}-c_{n-3} and P4 the IMEX predictor (advection extrapolated
2adv_n-adv_{n-1}, least squares with A = QA RA). Their residuals use the cheap exact residual; if one of them is chosen
its Jacobian is formed (one extra contraction; such steps are identified by the predictor index 3 or 4). The carry gains c_{n-3}. With extra=False the
predictor is fs3's and gate G1 requires bitwise equality with fs3. Original docstring follows.

jcp-combined: fixed-sweep two-step LMM solver for the off-mesh Burgers 3D ROM (DESIGN 2.1, A1.1, A1.2, A2.1, A2.2).

A copy of vendor/quad3d/offmesh.make_fsc_rule (the deployed fixed-sweep backward-Euler solver: adaptive LM on the first
a steps, then K cached-predictor LM sweeps per step) in which the backward-Euler residual is replaced by the LSPG residual
of a two-step linear multistep method (scheme table of t2core.SCHEMES):

    r(c) = S * ( A (a0 c + a1 c_n + a2 c_{n-1}) + dt ( b0 (0.5 Ju(c) c + nu lam A c) + b1 F_n ) ),
    S = 1 / (a0 + dt b0 nu lam),  J = A + dt b0 S * Ju(c),  F_n = 0.5 Ju(c_n) c_n + nu lam A c_n   (Ju(c_n) carried).

Every off-mesh rule is a symmetric quadratic form, so Ju is linear in c and adv(c) = 0.5 Ju(c) c (exact).
* scheme 'BE' (no mutation) evaluates the vendor expression verbatim (TB._lm_step_parts), so with K = 1 the solver is the
  deployed one (gate G-FS1, A2.1);
* startup steps (k < n_startup of the scheme) use BE coefficients through the generic expression and are always inside the
  adaptive phase: a_eff = min(steps, max(ceil(a dt0 / dt), n_startup))  (A1.2);
* recorded per step: LM iterations (adaptive) or K (fixed), exit reason (4 stationary / 0 not stationary / 3 non-finite;
  adaptive steps: the LM reason), stationarity ratio, residual norm, rejected sweeps, predictor candidate index (0 c_n,
  1 linear, 2 quadratic), damping carried, damping saturation (lam >= 1e14).
Mutations (manufactured negative controls only, A2.2): 'stale_fn' (CN uses F(c_{n-1})), 'stale_hist' (BDF2 uses c_{n-2}),
'no_startup' (BDF2 without its BE first step), 'wrong_weight' (S built with b1), 'prev_pred' (predictor = c_n only).
All large arrays are explicit jit arguments (data); scheme, dt, K, a, steps are static.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor' / 'quad3d'))
import offmesh as OM  # noqa: E402  (vendored, unchanged)
from offmesh import C, TB  # noqa: E402
import t2core as T2  # noqa: E402

SCHEMES = T2.SCHEMES
MUTATIONS = (None, 'stale_fn', 'stale_hist', 'no_startup', 'wrong_weight', 'prev_pred')
SAT = 1e14


def a_effective(scheme, dt, dt0, a, steps, mutation=None):
    ns = 0 if mutation == 'no_startup' else SCHEMES[scheme][5]
    return int(min(steps, max(math.ceil(a * dt0 / dt - 1e-9), ns)))


def make_fs_evolve(Rp, contract, scheme, dt, K, gtol, trust, a_eff, steps, linear_jac=True, unroll=True,
                   mutation=None, lm_budget=50, make_lm=None, extra=False):
    """evolve(w0, nu, tol_ad, data) -> dict(W_all (steps+1, R'), per-step records). contract(data, w) -> Ju (M, R').
    tol_ad: residual tolerance of the adaptive LM (vendor: 1e-12 ||s0||). make_lm: LM factory (default the vendor
    C.make_fused_lm)."""
    assert mutation in MUTATIONS and scheme in SCHEMES and K >= 1 and 0 <= a_eff <= steps
    a0, a1, a2, b0, b1, ns = SCHEMES[scheme]
    if mutation == 'no_startup':
        ns = 0
    co_be = (1., -1., 0., 1., 0.)
    co_s = (a0, a1, a2, b0, b1)
    vendor_be = scheme == 'BE' and mutation is None
    need_fn = b1 != 0.
    make_lm = make_lm or C.make_fused_lm

    def evolve(w0, nu, tol_ad, data):
        A, lam_ = data['A'], data['lam']

        def parts(k, wv, wp, wp2, Jv, Jp):
            """Residual/Jacobian closure rj(Ju, c) for step k (history fixed)."""
            p = A @ wv
            if vendor_be:
                S = 1.0 / (1.0 + dt * nu * lam_)
                rv = TB._lm_step_parts(A, S, lam_, dt, nu)
                return (lambda Ju, c: rv(Ju, c, p)), p
            co = [jnp.where(k < ns, x_be, x_s) for x_be, x_s in zip(co_be, co_s)]
            c0_, c1_, c2_, d0_, d1_ = co
            pm = A @ (wp2 if mutation == 'stale_hist' else wp)
            if need_fn:
                if mutation == 'stale_fn':
                    Fn = 0.5 * Jp @ wp + nu * lam_ * (A @ wp)
                else:
                    Fn = 0.5 * Jv @ wv + nu * lam_ * p
            else:
                Fn = jnp.zeros_like(p)
            wgt = d1_ if mutation == 'wrong_weight' else d0_
            S = 1.0 / (c0_ + dt * wgt * nu * lam_)
            hist = c1_ * p + c2_ * pm + dt * d1_ * Fn

            def rj(Ju, c):
                Ac = A @ c
                r = (c0_ * Ac + hist + dt * (d0_ * (0.5 * Ju @ c + nu * lam_ * Ac))) * S
                if mutation == 'wrong_weight':
                    J = S[:, None] * (c0_ * A + dt * d0_ * (Ju + (nu * lam_)[:, None] * A))
                else:
                    J = A + (dt * d0_ * S)[:, None] * Ju
                return r, J
            return rj, p

        def sweep(rj, w, Ju, r, J, rn, lam):
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
            r2, J2 = rj(Jun, wn)
            rn2 = jnp.linalg.norm(r2)
            acc = ok & jnp.isfinite(rn2) & (rn2 < rn)
            lam2 = jnp.where(acc, jnp.maximum(lam / 3, 1e-12), jnp.minimum(lam * 10, 1e14))
            sel = lambda x, y: jnp.where(acc, x, y)
            return sel(wn, w), sel(Jun, Ju), sel(r2, r), sel(J2, J), sel(rn2, rn), lam2, (~acc).astype(jnp.int32)

        def cheap_res(k, wv, wp, Jv, c):
            co = [jnp.where(k < ns, x_be, x_s) for x_be, x_s in zip(co_be, co_s)]
            c0_, c1_, c2_, d0_, d1_ = co
            p = A @ wv
            Fn = (0.5 * Jv @ wv + nu * lam_ * p) if need_fn else jnp.zeros_like(p)
            S = 1.0 / (c0_ + dt * d0_ * nu * lam_)
            hist = c1_ * p + c2_ * (A @ wp) + dt * d1_ * Fn
            return S, hist, d0_, c0_

        def predict(rj, k, wv, wp, wp2, Jv, Jp, Jp2, wp3=None):
            if extra:
                cand_, Ju_, i_ = predict0(rj, k, wv, wp, wp2, Jv, Jp, Jp2)
                S, hist, d0_, c0_ = cheap_res(k, wv, wp, Jv, wv)
                advc = lambda c: ((data['B'] @ c) * (data['D'] @ c)) @ data['P']
                resn = lambda c: jnp.linalg.norm(S * (c0_ * (A @ c) + hist + dt * d0_ * (advc(c) + nu * lam_ * (A @ c))))
                p3 = 4. * wv - 6. * wp + 4. * wp2 - wp3
                adv_t = jnp.where(k >= 1, 2. * (0.5 * Jv @ wv) - 0.5 * Jp @ wp, 0.5 * Jv @ wv)
                y = (S * (hist + dt * d0_ * adv_t)) @ data['QA']
                p4 = -jax.scipy.linalg.solve_triangular(data['RA'], y, lower=False)
                r0 = jnp.linalg.norm(rj(Ju_, cand_)[0])
                r3 = jnp.where(k >= 3, resn(p3), jnp.inf)
                r4 = resn(p4)
                rs = jnp.stack((r0, r3, r4))
                rs = jnp.where(jnp.isfinite(rs), rs, jnp.inf)
                j = jnp.argmin(rs)
                cand = jnp.stack((cand_, p3, p4))[j]
                Ju = jax.lax.cond(j == 0, lambda: Ju_, lambda: contract(data, cand))
                idx = jnp.where(j == 0, i_, jnp.where(j == 1, 3, 4)).astype(jnp.int32)
                return cand, Ju, idx
            return predict0(rj, k, wv, wp, wp2, Jv, Jp, Jp2)

        def predict0(rj, k, wv, wp, wp2, Jv, Jp, Jp2):
            if mutation == 'prev_pred':
                return wv, Jv, jnp.int32(0)
            we = 2. * wv - wp
            wq = jnp.where(k >= 2, 3. * wv - 3. * wp + wp2, we)
            cand = jnp.stack((wv, we, wq))
            if linear_jac:
                Je = 2. * Jv - Jp
                Jq = jnp.where(k >= 2, 3. * Jv - 3. * Jp + Jp2, Je)
                Ju3 = jnp.stack((Jv, Je, Jq))
            else:
                Ju3 = jax.lax.map(lambda c: contract(data, c), cand)
            rs = jnp.linalg.norm(jax.vmap(lambda Ju, c: rj(Ju, c)[0])(Ju3, cand), axis=1)
            i = jnp.argmin(jnp.where(jnp.isfinite(rs), rs, jnp.inf))
            return cand[i], Ju3[i], i.astype(jnp.int32)

        def step(carry, k):
            wv, wp, wp2, Jv, Jp, Jp2, lam, wp3 = carry
            rj, _ = parts(k, wv, wp, wp2, Jv, Jp)
            w, Ju, pi = predict(rj, k, wv, wp, wp2, Jv, Jp, Jp2, wp3)
            r, J = rj(Ju, w)
            rn = jnp.linalg.norm(r)
            rej = jnp.int32(0)
            for _ in range(K):
                w, Ju, r, J, rn, lam, rj1 = sweep(rj, w, Ju, r, J, rn, lam)
                rej = rej + rj1
            gn = jnp.linalg.norm(J.T @ r) / (jnp.linalg.norm(J) * rn + 1e-300)
            reason = jnp.where(jnp.isfinite(rn), jnp.where(gn <= gtol, 4, 0), 3).astype(jnp.int32)
            rec = (w, rn, jnp.int32(K), reason, gn, rej, pi, lam, (lam >= SAT).astype(jnp.int32))
            return (w, wv, wp, Ju, Jv, Jp, lam, wp2), rec

        def astep(carry, k):
            wv, wp, wp2, Jv, Jp, Jp2, lam, wp3 = carry
            rj, _ = parts(k, wv, wp, wp2, Jv, Jp)
            wi, _, pi = predict0(rj, k, wv, wp, wp2, Jv, Jp, Jp2)
            lmA = make_lm(lambda w, _a: rj(contract(data, w), w), Rp, lm_budget, trust, gtol, clip=True)
            w2, rn, it, reason, gn, rej, lam2 = lmA(wi, None, tol_ad, lam)
            rec = (w2, rn, it, reason, gn, rej, pi, lam2, (lam2 >= SAT).astype(jnp.int32))
            return (w2, wv, wp, contract(data, w2), Jv, Jp, lam2, wp2), rec

        J0 = contract(data, w0)
        carry = (w0, w0, w0, J0, J0, J0, jnp.asarray(1e-6, dtype=jnp.float64), w0)
        outs = []
        if a_eff > 0:
            carry, o1 = jax.lax.scan(astep, carry, jnp.arange(a_eff))
            outs.append(o1)
        if steps > a_eff:
            carry, o2 = jax.lax.scan(step, carry, jnp.arange(a_eff, steps), unroll=unroll)
            outs.append(o2)
        cat = [jnp.concatenate([o[i] for o in outs]) for i in range(len(outs[0]))]
        ws, rn, it, reason, gn, rej, pi, lamh, sat = cat
        return dict(W_all=jnp.concatenate((w0[None], ws)), rn=rn, it=it, reason=reason, gn=gn, rej=rej, pred=pi,
                    lam=lamh, sat=sat)
    return evolve


def make_fs_query(n, Rp, contract, scheme, dt, K, gtol, trust, a, dt0=0.01, mutation=None, unroll=True, extra=False):
    """End-to-end 3D query (L2 projection of u0, fixed-sweep LMM stepping, decode of the six output fields), shaped like
    offmesh.make_fsc_rule: query(u0, nu, data) -> dict(fields, W (6, R'), w0, internal, per-step records, a_eff)."""
    steps, keep = int(round(0.25 / dt)), int(round(0.05 / dt))
    assert abs(steps * dt - 0.25) < 1e-12 and abs(keep * dt - 0.05) < 1e-12, dt
    a_eff = a_effective(scheme, dt, dt0, a, steps, mutation)
    ev = make_fs_evolve(Rp, contract, scheme, dt, K, gtol, trust, a_eff, steps, mutation=mutation, unroll=unroll, extra=extra)

    def query(u0, nu, data):
        w0, s0 = TB.project(u0, data)
        o = ev(w0, nu, 1e-12 * jnp.linalg.norm(s0), data)
        W = o['W_all'][::keep]
        return dict(o, fields=TB.decode(W, data), W=W, w0=w0)
    q = jax.jit(query)
    q.a_eff, q.steps, q.keep = a_eff, steps, keep
    return q
