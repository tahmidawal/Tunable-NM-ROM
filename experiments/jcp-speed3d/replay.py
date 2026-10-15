"""jcp-speed3d H1.1 step replay (DESIGN A1.6 as replaced by A2.2, A3).

For one failing case of FS-K2 at (scheme, dt): pick the replayed step n*, rebuild fs3's step from the saved states
(c_{n*}, c_{n*-1}, c_{n*-2}), the damping entering n* and the saved predictor index, reproduce the recorded K = 2 result
(baseline), then apply the interventions (i) K = 2..12 sweeps, (ii) trust x10 with K = 2, (iii) damping 1e-12 with K = 2,
(iv) QN-1 from the same start; label per A2.2. The step arithmetic is fs3's (parts / sweep / extrapolated-Jacobian
predictor), copied here for a single step with explicit trust and damping.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

import fs3
import qn3
from c3common import OM, TB

SCHEMES = fs3.SCHEMES
BE = (1., -1., 0., 1., 0.)


def pick_step(rec, gam_cert, valid, dist_out, a_eff, keep, bar=2.5e-4):
    """A2.2 step choice. rec: fs records (lists per step); gam_cert: certified gamma per step; dist_out: distance per output
    time (6,). Returns (n*, kind) or (None, reason)."""
    reason = np.asarray(rec['reason'])
    sat = np.asarray(rec['sat'])
    bad = np.where((reason == 3) | (sat > 0))[0]
    g = np.asarray(gam_cert, float)
    if not valid and len(bad):
        n = int(bad[0])
        kind = 'validity'
    else:
        ns = np.where(~(g <= 1e-3))[0]
        if len(ns):
            n, kind = int(ns[0]), 'nonstationary'
        else:
            ex = [k for k in range(1, 6) if dist_out[k] > bar]
            if not ex:
                return None, 'no failure found'
            k = ex[0]
            lo, hi = (k - 1) * keep, k * keep                       # steps producing states (lo, hi]
            n, kind = int(lo + np.argmax(g[lo:hi])), 'distance-only'
    if n < a_eff:
        return None, f'{kind} failure inside the adaptive start (step {n} < a_eff {a_eff}); not replayed'
    return n, kind


def make_fs_step(Rp, scheme, dt, nu, data):
    a0s, a1s, a2s, b0s, b1s, ns = SCHEMES[scheme]
    A, lam_ = data['A'], data['lam']
    contract = OM.contract_offmesh

    def parts(k, wv, wp, Jv):
        p = A @ wv
        if scheme == 'BE':
            S = 1.0 / (1.0 + dt * nu * lam_)
            rv = TB._lm_step_parts(A, S, lam_, dt, nu)
            return lambda Ju, c: rv(Ju, c, p)
        co = [jnp.where(k < ns, x, y) for x, y in zip(BE, (a0s, a1s, a2s, b0s, b1s))]
        c0_, c1_, c2_, d0_, d1_ = co
        pm = A @ wp
        Fn = 0.5 * Jv @ wv + nu * lam_ * p if b1s != 0. else jnp.zeros_like(p)
        S = 1.0 / (c0_ + dt * d0_ * nu * lam_)
        hist = c1_ * p + c2_ * pm + dt * d1_ * Fn

        def rj(Ju, c):
            Ac = A @ c
            r = (c0_ * Ac + hist + dt * (d0_ * (0.5 * Ju @ c + nu * lam_ * Ac))) * S
            return r, A + (dt * d0_ * S)[:, None] * Ju
        return rj

    def sweep(rj, w, Ju, r, J, rn, lam, trust):
        Hm = J.T @ J
        g = J.T @ r
        d = jnp.diag(Hm) + 1e-30
        cf = jax.scipy.linalg.cho_factor(Hm + jnp.diag(lam * d), lower=True)
        dw = jax.scipy.linalg.cho_solve(cf, -g)
        nz = jnp.linalg.norm(dw)
        clip = nz > trust
        dw = dw * jnp.where(clip, trust / (nz + 1e-300), 1.)
        ok = jnp.all(jnp.isfinite(dw))
        wn = w + jnp.where(ok, dw, 0.)
        Jun = contract(data, wn)
        r2, J2 = rj(Jun, wn)
        rn2 = jnp.linalg.norm(r2)
        acc = ok & jnp.isfinite(rn2) & (rn2 < rn)
        lam2 = jnp.where(acc, jnp.maximum(lam / 3, 1e-12), jnp.minimum(lam * 10, 1e14))
        sel = lambda x, y: jnp.where(acc, x, y)
        return sel(wn, w), sel(Jun, Ju), sel(r2, r), sel(J2, J), sel(rn2, rn), lam2, clip, ~acc

    def run(k, wv, wp, wp2, pidx, lam, trust, K):
        Jv, Jp, Jp2 = contract(data, wv), contract(data, wp), contract(data, wp2)
        rj = parts(k, wv, wp, Jv)
        we, Je = 2. * wv - wp, 2. * Jv - Jp
        wq = jnp.where(k >= 2, 3. * wv - 3. * wp + wp2, we)
        Jq = jnp.where(k >= 2, 3. * Jv - 3. * Jp + Jp2, Je)
        w = jnp.stack((wv, we, wq))[pidx]
        Ju = jnp.stack((Jv, Je, Jq))[pidx]
        r, J = rj(Ju, w)
        rn = jnp.linalg.norm(r)
        clips, rejs = [], []
        for _ in range(K):
            w, Ju, r, J, rn, lam, cl, rejected = sweep(rj, w, Ju, r, J, rn, lam, trust)
            clips.append(cl)
            rejs.append(rejected)
        gn = jnp.linalg.norm(J.T @ r) / (jnp.linalg.norm(J) * rn + 1e-300)
        return w, gn, jnp.stack(clips), jnp.stack(rejs)
    return run


def replay_case(st, data, scheme, dt, nu, W_all, rec, n_star, K_rec=2):
    """Returns the A2.2 record for one case (W_all (steps+1, R') of the FS-K2 run; rec its per-step records)."""
    W = jnp.asarray(W_all)
    k = n_star
    wv, wp, wp2 = W[k], W[max(k - 1, 0)], W[max(k - 2, 0)]
    lam_in = 1e-6 if k == 0 else float(rec['lam'][k - 1])
    pidx = int(rec['pred'][k])
    step = make_fs_step(st.Rp, scheme, dt, nu, data)
    jstep = {}

    def go(trust, lam, K):
        key = K
        if key not in jstep:
            jstep[key] = jax.jit(lambda k_, a, b, c, p, l, t: step(k_, a, b, c, p, l, t, K))
        w, gn, cl, rj = jstep[key](k, wv, wp, wp2, pidx, lam, trust)
        return np.asarray(w), float(gn), np.asarray(cl).tolist(), np.asarray(rj).tolist()
    w_b, g_b, cl_b, rj_b = go(st.trust, lam_in, K_rec)
    ref_w = np.asarray(W[k + 1])
    g_rec = float(rec['gn'][k])
    base_ok = bool(np.linalg.norm(w_b - ref_w) <= 1e-10 * max(np.linalg.norm(ref_w), 1e-300)
                   and abs(g_b - g_rec) <= 1e-6 * abs(g_rec) + 1e-9)
    out = dict(n_star=k, lam_in=lam_in, pred=pidx, baseline=dict(gamma=g_b, gamma_rec=g_rec, clips=cl_b, rejects=rj_b,
                                                                  reproduced=base_ok))
    if not base_ok:
        out['label'] = 'replay not reproducible'
        return out
    if g_b <= 1e-3:
        out['label'] = 'baseline already stationary'
        return out
    kmin = None
    for K in range(2, 13):
        _, g, _, _ = go(st.trust, lam_in, K)
        if g <= 1e-3:
            kmin = K
            break
    out['i_sweeps_needed'] = kmin if kmin is not None else '>12'
    _, g2, cl2, _ = go(10 * st.trust, lam_in, 2)
    _, g3, _, _ = go(st.trust, 1e-12, 2)
    out['ii_trust10'] = dict(gamma=g2, success=g2 <= 1e-3, clips=cl2)
    out['iii_damping'] = dict(gamma=g3, success=g3 <= 1e-3)
    # (iv) QN-1 from the same start (history c_{n*-1}, c_{n*-2}, c_{n*-3}; one step at index n*)
    ev = qn3.make_qn_evolve(st.Rp, scheme, dt, 1, st.trust, 1)
    wp3 = W[max(k - 3, 0)]
    o = jax.jit(lambda a, b, c, dd, e: ev(a, b, dd, hist=c, k0=k))(wv[None], jnp.asarray([nu]), (wp[None], wp2[None], wp3[None]),
                                                                 data, None)
    out['iv_qn1'] = dict(reason=int(np.asarray(o['reason'])[0, 0]), q=int(np.asarray(o['q'])[0, 0]),
                         gam_in=float(np.asarray(o['gam'])[0, 0]))
    succ = [kmin is not None and kmin <= 4, g2 <= 1e-3, g3 <= 1e-3]
    if sum(succ) >= 2:
        lab = 'mixed'
    elif succ[0]:
        lab = 'too few sweeps'
    elif succ[1]:
        lab = 'trust clip'
    elif succ[2]:
        lab = 'damping'
    else:
        lab = 'unexplained'
    out['label'] = lab
    return out
