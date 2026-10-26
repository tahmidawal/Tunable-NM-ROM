"""b3d-solver policy extension: slow_threshold freezes the refresh trigger while varying q_max.
Original source remains byte-identical in vendor/speed3d/qn3.py. All other arithmetic is unchanged.

jcp-speed3d: QN-k solver (DESIGN §3, A1.2) and the Step 2 stationarity certification (DESIGN §3, A1.4).

QN-k: two-step LMM LSPG step of the off-mesh Burgers 3D ROM, solved by iterations c <- c - H~^{-1} g(c) with the EXACT
residual and gradient (matrix-free products with B, D, P) and a frozen Gauss-Newton matrix H~ = J~^T J~ formed at a
refresh state (one dense Jacobian contraction). Batch-first: every array carries a leading batch axis Bt (Bt = 1 for a
single query); scalar `any` predicates decide whether the (expensive) contraction runs, and its result is applied only
to the members that asked for it, so batched and single runs take the same per-member branches up to rounding.

    r(c) = S*(a0 A c + hist + dt b0 (adv(c) + nu lam A c)),  hist = a1 A c_n + a2 A c_{n-1} + dt b1 F_n,
    F_n = adv(c_n) + nu lam A c_n,  S = 1/(a0 + dt b0 nu lam),
    adv(c) = P^T[(Bc)(Dc)],  g = A^T r + dt b0 Ju(c)^T (S r),  Ju(c)^T v = B^T[(Dc)(Pv)] + D^T[(Bc)(Pv)].

Mutations (negative controls only): 'q0' (no iterations; K2), 'noJuT' (g = A^T r; K3).
`certify` recomputes, for every step, the Step 2 ratio gamma = ||J^T r|| / (||J||_F ||r||) at the returned state with a
fresh contraction, using fs3's residual expression (adv = 0.5 Ju c, F_n = 0.5 Ju(c_n) c_n + nu lam A c_n).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor' / 'speed3d' / 'vendor' / 'quad3d'))
import offmesh as OM  # noqa: E402
import t2core as T2  # noqa: E402

SCHEMES = T2.SCHEMES
BE_CO = (1., -1., 0., 1., 0.)
G_IN, Q_MAX, MU0 = 5e-4, 12, 1e-10
BIG = 1 << 30
MUTATIONS = (None, 'q0', 'noJuT')
REFRESH_CODES = dict(none=0, first=1, switch=2, periodic=3, slow=4, failed=5)


# ------------------------------------------------------------------------------------------- matrix-free pieces
def bd(c, d):
    return c @ d['B'].T, c @ d['D'].T                                   # (Bt, m) each


def adv(c, d):
    Bc, Dc = bd(c, d)
    return (Bc * Dc) @ d['P']                                            # (Bt, M)


def jut(Bc, Dc, v, d):
    """Ju(c)^T v for a batch: v (Bt, M) -> (Bt, R')."""
    Pv = v @ d['P'].T
    return (Dc * Pv) @ d['B'] + (Bc * Pv) @ d['D']


def contract_batch(c, d):
    """Dense Ju(c) for every member (Bt, M, R'), one member at a time (bounded memory)."""
    return jax.lax.map(lambda w: OM.contract_offmesh(d, w), c)


def chol_solve(Lf, g):
    """Solve (L L^T) x = g per member."""
    f = lambda L, b: jax.scipy.linalg.cho_solve((L, True), b)
    return jax.vmap(f)(Lf, g)


def factor(H, mu):
    dg = jnp.diagonal(H, axis1=1, axis2=2)
    return jnp.linalg.cholesky(H + mu[:, None, None] * jax.vmap(jnp.diag)(dg))


# ------------------------------------------------------------------------------------------------- solver
def make_qn_evolve(Rp, scheme, dt, steps, trust, k_ref, g_in=G_IN, q_max=Q_MAX, mu0=MU0, mutation=None, force_pred=None, slow_threshold=None):
    """evolve(w0 (Bt, R'), nu (Bt,), data) -> dict(W_all (steps+1, Bt, R'), per-step records (steps, Bt)).
    data: A (M, R'), lam (M,), B, D (m, R'), P (m, M), QA (M, R'), RA (R', R') with A = QA RA."""
    assert scheme in SCHEMES and mutation in MUTATIONS
    a0s, a1s, a2s, b0s, b1s, ns = SCHEMES[scheme]
    kr = BIG if (k_ref is None or (isinstance(k_ref, float) and math.isinf(k_ref))) else int(k_ref)
    qmax = 0 if mutation == 'q0' else int(q_max)

    def evolve(w0, nu, data, hist=None, k0=0):
        """hist = (c_{n-1}, c_{n-2}, c_{n-3}) (each (Bt, R')) and k0 (static int) start a replay at step k0 with a given
        history (DESIGN A2.2); default: a full trajectory from k = 0 with c_{-1} = c_{-2} = c_{-3} = c_0."""
        A, lam = data['A'], data['lam']
        Bt = w0.shape[0]

        def coeffs(k):
            be = k < ns
            return [jnp.where(be, x, y) for x, y in zip(BE_CO, (a0s, a1s, a2s, b0s, b1s))]

        def stage(k, cn, cm, advn):
            a0, a1, a2, b0, b1 = coeffs(k)
            S = 1.0 / (a0 + dt * b0 * nu[:, None] * lam[None, :])                 # (Bt, M)
            Fn = advn + nu[:, None] * lam[None, :] * (cn @ A.T)
            hist = a1 * (cn @ A.T) + a2 * (cm @ A.T) + dt * b1 * Fn

            def res(c):
                Ac = c @ A.T
                return S * (a0 * Ac + hist + dt * b0 * (adv(c, data) + nu[:, None] * lam[None, :] * Ac))

            def grad(c, r):
                g = (r) @ A
                if mutation == 'noJuT':
                    return g
                Bc, Dc = bd(c, data)
                return g + dt * b0 * jut(Bc, Dc, S * r, data)

            def jac_parts(c):
                Ju = contract_batch(c, data)
                Jt = A[None] + (dt * b0 * S)[:, :, None] * Ju
                H = jnp.einsum('bmi,bmj->bij', Jt, Jt)
                return H, jnp.sqrt(jnp.sum(Jt * Jt, axis=(1, 2)))
            return dict(S=S, hist=hist, res=res, grad=grad, jac=jac_parts, b0=b0)

        def predict(k, st, cn, cm, cm2, cm3, advn, advm):
            act = jnp.array([True, k >= 1, k >= 2, k >= 3, True])
            p0 = cn
            p1 = 2. * cn - cm
            p2 = 3. * cn - 3. * cm + cm2
            p3 = 4. * cn - 6. * cm + 4. * cm2 - cm3
            adv_t = jnp.where(k >= 1, 2. * advn - advm, advn)
            v = st['S'] * (st['hist'] + dt * st['b0'] * adv_t)
            y = v @ data['QA']                                                       # (Bt, R')
            p4 = -jax.scipy.linalg.solve_triangular(data['RA'], y.T, lower=False).T
            cand = jnp.stack((p0, p1, p2, p3, p4))                                   # (5, Bt, R')
            rs = jnp.stack([jnp.linalg.norm(st['res'](cand[i]), axis=1) for i in range(5)])   # (5, Bt)
            rs = jnp.where(act[:, None] & jnp.isfinite(rs), rs, jnp.inf)
            i = jnp.argmin(rs, axis=0)                                               # ties: lowest index
            if force_pred is not None:                       # replay (A2.2 iv): the saved candidate of the replayed step
                i = jnp.full_like(i, int(force_pred))
            allbad = jnp.all(~jnp.isfinite(rs), axis=0)
            return jnp.take_along_axis(cand, i[None, :, None], 0)[0], i.astype(jnp.int32), allbad

        def step(carry, k):
            cn, cm, cm2, cm3, advn, advm, H, Lf, Jn, mu, last_ref, q_prev, fail_prev, dead = carry
            st = stage(k, cn, cm, advn)
            c, pidx, allbad = predict(k, st, cn, cm, cm2, cm3, advn, advm)
            sw = (k == ns) & (ns > 0)
            per = (k - last_ref) >= kr
            slow = q_prev > (qmax // 2 if slow_threshold is None else int(slow_threshold))
            due = ((k == k0) | sw | per | slow | fail_prev) & ~dead
            rcode = jnp.where(k == k0, 1, jnp.where(sw, 2, jnp.where(fail_prev, 5, jnp.where(slow, 4,
                              jnp.where(per, 3, 0))))).astype(jnp.int32)
            rcode = jnp.broadcast_to(rcode, (Bt,))

            def refresh(c_, mask, H_, Lf_, Jn_, mu_):
                Hn, Jnn = st['jac'](c_)
                mun = jnp.full((Bt,), mu0)
                Lfn = factor(Hn, mun)
                sel = lambda new, old: jnp.where(mask.reshape((-1,) + (1,) * (old.ndim - 1)), new, old)
                return sel(Hn, H_), sel(Lfn, Lf_), sel(Jnn, Jn_), sel(mun, mu_)
            H, Lf, Jn, mu = jax.lax.cond(jnp.any(due), lambda: refresh(c, due, H, Lf, Jn, mu), lambda: (H, Lf, Jn, mu))
            last_ref = jnp.where(due, k.astype(jnp.int32), last_ref)
            n_full = due.astype(jnp.int32)                                          # n_ref: refreshes applied
            n_exec = jnp.where(jnp.any(due), Bt, 0).astype(jnp.int32)               # contractions executed (step)
            r = st['res'](c)
            rn = jnp.linalg.norm(r, axis=1)
            g = st['grad'](c, r)
            gam = jnp.linalg.norm(g, axis=1) / (Jn * rn + 1e-300)
            lff = jnp.all(jnp.isfinite(Lf), axis=(1, 2))
            bad = dead | allbad | ~jnp.isfinite(rn) | ~jnp.isfinite(gam) | ~jnp.all(jnp.isfinite(g), axis=1) | ~lff
            done = ~bad & ((rn == 0.) | (gam <= g_in))
            z = jnp.zeros((Bt,), jnp.int32)
            emerg = jnp.zeros((Bt,), bool)
            s0 = (jnp.int32(0), c, r, rn, g, gam, done, bad, emerg, H, Lf, Jn, mu, last_ref, n_full, z, z, z, z, n_exec)

            def cond(s):
                return (s[0] < qmax) & jnp.any(~s[6] & ~s[7])

            def body(s):
                it, c, r, rn, g, gam, done, bad, emerg, H, Lf, Jn, mu, last_ref, n_full, q, nbt, nclip, nmu, nex = s
                act = ~done & ~bad
                d = -chol_solve(Lf, g)
                nz = jnp.linalg.norm(d, axis=1)
                clip = nz > trust
                d = d * jnp.where(clip, trust / (nz + 1e-300), 1.)[:, None]
                dfin = jnp.all(jnp.isfinite(d), axis=1)
                bad = bad | (act & ~dfin)
                act = act & dfin
                d = jnp.where(act[:, None], d, 0.)

                def trial(alpha):
                    c1 = c + alpha * d
                    r1 = st['res'](c1)
                    n1 = jnp.linalg.norm(r1, axis=1)
                    return c1, r1, n1, jnp.isfinite(n1) & (n1 < rn)
                c1, r1, n1, ok1 = trial(1.)
                need = act & ~ok1
                cA, rA, nA, okA = c1, r1, n1, act & ok1
                nb = jnp.zeros((Bt,), jnp.int32)
                for alpha in (.5, .25):
                    def do(alpha=alpha, need=need, cA=cA, rA=rA, nA=nA, okA=okA):
                        c2, r2, n2, ok2 = trial(alpha)
                        take = need & ok2
                        w = lambda x, y: jnp.where(take.reshape((-1,) + (1,) * (x.ndim - 1)), x, y)
                        return w(c2, cA), w(r2, rA), w(n2, nA), okA | take
                    cA, rA, nA, okA = jax.lax.cond(jnp.any(need), do, lambda: (cA, rA, nA, okA))
                    nb = nb + need.astype(jnp.int32)
                    need = act & ~okA
                acc = act & okA
                w = lambda m, x, y: jnp.where(m.reshape((-1,) + (1,) * (x.ndim - 1)), x, y)
                c, r, rn = w(acc, cA, c), w(acc, rA, r), w(acc, nA, rn)
                g = w(acc, st['grad'](c, r), g)
                bad = bad | (acc & ~jnp.all(jnp.isfinite(g), axis=1))
                rej = act & ~okA
                em_now = rej & ~emerg
                mu_up = rej & emerg
                H, Lf, Jn, mu = jax.lax.cond(jnp.any(em_now), lambda: refresh(c, em_now, H, Lf, Jn, mu),
                                             lambda: (H, Lf, Jn, mu))
                last_ref = jnp.where(em_now, k.astype(jnp.int32), last_ref)
                n_full = n_full + em_now.astype(jnp.int32)
                nex = nex + jnp.where(jnp.any(em_now), Bt, 0).astype(jnp.int32)
                emerg = emerg | em_now
                mu = jnp.where(mu_up, 10. * mu, mu)
                Lf = jax.lax.cond(jnp.any(mu_up), lambda: w(mu_up, factor(H, mu), Lf), lambda: Lf)
                gam = jnp.where(act, jnp.linalg.norm(g, axis=1) / (Jn * rn + 1e-300), gam)   # after any refresh
                bad = bad | (act & (~jnp.isfinite(gam) | ~jnp.all(jnp.isfinite(Lf), axis=(1, 2))))
                done = done | (act & ~bad & (gam <= g_in))
                return (it + 1, c, r, rn, g, gam, done, bad, emerg, H, Lf, Jn, mu, last_ref, n_full,
                        q + act.astype(jnp.int32), nbt + jnp.where(act, nb, 0), nclip + (act & clip).astype(jnp.int32),
                        nmu + mu_up.astype(jnp.int32), nex)
            (_, c, r, rn, g, gam, done, bad, emerg, H, Lf, Jn, mu, last_ref, n_full, q, nbt, nclip,
             nmu, n_exec) = jax.lax.while_loop(cond, body, s0)
            reason = jnp.where(bad, 3, jnp.where(done, 4, 0)).astype(jnp.int32)       # precedence 3 > 4 > 0
            dead = dead | bad
            c = jnp.where(dead[:, None], cn, c)                                        # frozen at the last finite state
            advc = jnp.where(dead[:, None], advn, adv(c, data))
            rec = dict(n_ref=n_full, n_exec=jnp.broadcast_to(n_exec, (Bt,)), q=q, n_bt=nbt, n_clip=nclip, n_mu=nmu, mu=mu,
                       refresh=jnp.where(due, rcode, 0), pred=pidx, gam=gam, rn=rn, reason=reason)
            carry = (c, cn, cm, cm2, advc, advn, H, Lf, Jn, mu, last_ref, q, reason != 4, dead)
            return carry, (c, rec)

        M = A.shape[0]
        adv0 = adv(w0, data)
        H0 = jnp.zeros((Bt, Rp, Rp))
        cm, cm2, cm3 = (w0, w0, w0) if hist is None else hist
        carry = (w0, cm, cm2, cm3, adv0, adv(cm, data), H0, H0, jnp.ones((Bt,)), jnp.full((Bt,), mu0),
                 jnp.full((Bt,), k0, jnp.int32), jnp.zeros((Bt,), jnp.int32), jnp.zeros((Bt,), bool), jnp.zeros((Bt,), bool))
        _, (cs, rec) = jax.lax.scan(step, carry, jnp.arange(k0, k0 + steps))
        return dict(W_all=jnp.concatenate((w0[None], cs)), **rec)
    return evolve


def qr_data(A):
    QA, RA = jnp.linalg.qr(A, mode='reduced')
    return dict(QA=QA, RA=RA)


# ------------------------------------------------------------------------------------------- certification
def make_certify(scheme, dt, steps):
    """certify(W_all (steps+1, R'), nu, data) -> gamma (steps,): the Step 2 stationarity ratio of every step's returned
    state with a fresh contraction, fs3's residual arithmetic (F_n = 0.5 Ju(c_n) c_n + nu lam A c_n; history at k = 0
    is c_{-1} = c_0, as fs3's initial carry)."""
    a0s, a1s, a2s, b0s, b1s, ns = SCHEMES[scheme]

    def certify(W, nu, data):
        A, lam = data['A'], data['lam']

        def body(carry, k):
            Jv, wv, wp = carry
            wn = W[k + 1]
            be = k < ns
            c0_, c1_, c2_, d0_, d1_ = [jnp.where(be, x, y) for x, y in zip(BE_CO, (a0s, a1s, a2s, b0s, b1s))]
            p = A @ wv
            Fn = 0.5 * Jv @ wv + nu * lam * p
            S = 1.0 / (c0_ + dt * d0_ * nu * lam)
            hist = c1_ * p + c2_ * (A @ wp) + dt * d1_ * Fn
            Ju = OM.contract_offmesh(data, wn)
            Ac = A @ wn
            r = (c0_ * Ac + hist + dt * (d0_ * (0.5 * Ju @ wn + nu * lam * Ac))) * S
            J = A + (dt * d0_ * S)[:, None] * Ju
            gn = jnp.linalg.norm(J.T @ r) / (jnp.linalg.norm(J) * jnp.linalg.norm(r) + 1e-300)
            return (Ju, wn, wv), gn
        J0 = OM.contract_offmesh(data, W[0])
        _, g = jax.lax.scan(body, (J0, W[0], W[0]), jnp.arange(steps))
        return g
    return jax.jit(certify)
