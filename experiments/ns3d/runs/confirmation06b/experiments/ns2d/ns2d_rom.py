"""ns2d_rom.py -- the weak least-squares NM-ROM for NS 2D with the precomputed degree-2
advection tensor, the correction ladder, POD-LSPG, and the FOM tolerance ladder (DESIGN.md
"Bank, head, test space, ROM").

Reduced state  omega = B c,  B in R^{n x D} a grid-resident bank (neural bank G or POD V),
c = h(w) with h the arm's coefficient map:
    neural rung q:  c = h_theta(z) + C_q y,  w = (z, y) in R^{K+q}
    POD-LSPG  k':   c = w,                    w in R^{k'}     (identity head)

Test space: the M lowest real Fourier modes phi_m (orthonormal on the grid, exact eigen-
vectors of Lap_h with Lap_h phi_m = -lam_m phi_m).  Projected implicit-midpoint residual,
scaled by the Helmholtz diagonal:

    r(c; c_n) = [ A c - A c_n - dt ( q_T(c_m) - nu lam * (A c_m) + b_f ) ] / (1 + dt nu lam / 2)
    c_m = (c + c_n)/2,   A = Phi^T B,   q_T(c)_m = sum_jk c_j c_k T[m,j,k],
    T[m,j,k] = sum_x phi_m(x) J_A(psi_j, b_k)(x),   Psi = -Lap_h^{-1} B.

Because J_A is exactly bilinear, T is EXACT (gate R-TQ) -- there is no positivity
restriction and no quadrature rule anywhere.  Q = T + T^(jk) so dq/dc = Q c.

Cold start: full-grid least squares in the whitened bank metric (thin QR of B), LM from the
best-scoring training code; for q > 0 the correction block starts at its least-squares value
given z.  Everything large is a jit ARGUMENT.
"""
from __future__ import annotations

import time

import numpy as np
import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp                                              # noqa: E402

import ns2d_fom as F                                                 # noqa: E402

F64 = jnp.float64
PI = np.pi


# ------------------------------------------------------------------ test modes --

def fourier_modes(N, M):
    """The M lowest real Fourier modes on the N-grid, orthonormal (2-norm on the grid),
    ordered by |k|^2 then by a fixed tie rule.  Returns (Phi (N*N, M), lam (M,), ids).
    Each k (with k != 0) contributes cos and sin; k = 0 is excluded (mean-zero space).
    Only one of +/-k is kept (cos/sin already span both)."""
    X, Y = F.coords(N)
    cands = []
    kmax = N // 2 - 1
    for kx in range(-kmax, kmax + 1):
        for ky in range(-kmax, kmax + 1):
            if (kx, ky) == (0, 0):
                continue
            if kx < 0 or (kx == 0 and ky < 0):
                continue                                  # keep one of +/-k
            cands.append((kx * kx + ky * ky, kx, ky))
    cands.sort()
    cols, lams, ids = [], [], []
    for k2, kx, ky in cands:
        ph = 2 * PI * (kx * X + ky * Y)
        lam = (2 - 2 * np.cos(2 * PI * kx / N) + 2 - 2 * np.cos(2 * PI * ky / N)) * N * N
        for fn, name in ((np.cos, 'c'), (np.sin, 's')):
            v = fn(ph).ravel()
            nrm = np.linalg.norm(v)
            cols.append(v / nrm)
            lams.append(lam)
            ids.append((kx, ky, name, float(nrm)))
            if len(cols) == M:
                return np.column_stack(cols), np.asarray(lams), ids
    raise ValueError('M too large for N')


# ---------------------------------------------------------------------- tensor --

def _jac_terms(P, Z, N):
    """The Arakawa Jacobian for a BATCH of column pairs, written as sums of products of
    shifted fields.  P, Z: (n, Rp), (n, Rz) grid-resident (row order i*N+j).  Returns the
    list of (coef, Pshift (n,Rp), Zshift (n,Rz)) triples such that
    J_A(psi_j, z_k)(x) = sum_t coef_t Pshift_t[x, j] Zshift_t[x, k]."""
    def sh(A, di, dj):
        G = A.reshape(N, N, -1)
        return jnp.roll(jnp.roll(G, -di, 0), -dj, 1).reshape(N * N, -1)
    c = N * N / 12.0
    px, mx, py, my = sh(P, 1, 0), sh(P, -1, 0), sh(P, 0, 1), sh(P, 0, -1)
    ppp, ppm, pmp, pmm = sh(P, 1, 1), sh(P, 1, -1), sh(P, -1, 1), sh(P, -1, -1)
    zx, zmx, zy, zmy = sh(Z, 1, 0), sh(Z, -1, 0), sh(Z, 0, 1), sh(Z, 0, -1)
    zpp, zpm, zmp, zmm = sh(Z, 1, 1), sh(Z, 1, -1), sh(Z, -1, 1), sh(Z, -1, -1)
    terms = [
        # J++ : (px - mx)(zy - zmy) - (py - my)(zx - zmx)
        (c, px, zy), (-c, px, zmy), (-c, mx, zy), (c, mx, zmy),
        (-c, py, zx), (c, py, zmx), (c, my, zx), (-c, my, zmx),
        # J+x : px(zpp - zpm) - mx(zmp - zmm) - py(zpp - zmp) + my(zpm - zmm)
        (c, px, zpp), (-c, px, zpm), (-c, mx, zmp), (c, mx, zmm),
        (-c, py, zpp), (c, py, zmp), (c, my, zpm), (-c, my, zmm),
        # Jx+ : zy(ppp - pmp) - zmy(ppm - pmm) - zx(ppp - ppm) + zmx(pmp - pmm)
        (c, ppp, zy), (-c, pmp, zy), (-c, ppm, zmy), (c, pmm, zmy),
        (-c, ppp, zx), (c, ppm, zx), (c, pmp, zmx), (-c, pmm, zmx),
    ]
    return terms


def jac_pairs_chunk(P, Z, N, rows):
    """J_A(psi_j, z_k) on the grid rows `rows` for all (j, k): (len(rows), Rp, Rz)."""
    out = None
    for coef, Ps, Zs in _jac_terms(P, Z, N):
        t = coef * Ps[rows][:, :, None] * Zs[rows][:, None, :]
        out = t if out is None else out + t
    return out


@jax.jit
def _chunk_T(Pc, JAc):
    """(chunk, M), (chunk, R, R) -> (M, R, R)."""
    return jnp.einsum('xm,xjk->mjk', Pc, JAc)


def build_T(Phi, B, N, chunk=256, reverse=False):
    """T[m,j,k] = sum_x Phi[x,m] J_A(psi_j, b_k)(x) with Psi = -Lap_h^{-1} B, accumulated
    over row chunks (reverse order if `reverse`; gate R-TB compares the two)."""
    B = jnp.asarray(B, F64)
    Phi = jnp.asarray(Phi, F64)
    n, R = B.shape
    M = Phi.shape[1]
    Psi = jax.vmap(lambda col: F.poisson(col.reshape(N, N), N).ravel(), in_axes=1, out_axes=1)(B)
    terms = _jac_terms(Psi, B, N)
    T = np.zeros((M, R, R))
    starts = list(range(0, n, chunk))
    if reverse:
        starts = starts[::-1]
    for s in starts:
        e = min(s + chunk, n)
        acc = None
        for coef, Ps, Zs in terms:
            t = coef * Ps[s:e][:, :, None] * Zs[s:e][:, None, :]
            acc = t if acc is None else acc + t
        T += np.asarray(_chunk_T(Phi[s:e], acc))
    return T


def build_T_fft(ids, B, N, block=64, reverse=False):
    """T[m,j,k] = sum_x phi_m(x) J_A(psi_j, b_k)(x), built with FFTs.

    THE TEST MODES ARE FOURIER MODES, so projecting a field on them is an FFT, not a dense
    (M, n) matmul.  For mode m = (kx, ky, 'c'|'s') with un-normalised grid norm `nrm`,

        phi_m^T v = Re F[kx, ky] / nrm      (cos),      -Im F[kx, ky] / nrm      (sin),
        F = fft2(v),

    which is exact, not an approximation.  The direct build (`build_T`) costs
    24 n M R^2 FLOPs -- 4.7e14 at n = 256^2, M = 1152, R = 512, i.e. hours on an A100.
    This costs O(24 n R^2 + n log n R^2) and is ~500x cheaper at those sizes, with the SAME
    output: `build_T` is retained as the independent reference and gate R-TFFT compares them.

    Blocked over the SECOND bank index k; for each k the whole j-axis is done at once, so the
    peak footprint is one (R, N, N) real plus one (R, N, N//2+1) complex array.
    """
    B = jnp.asarray(B, F64)
    n, R = B.shape
    M = len(ids)
    Psi = jax.vmap(lambda col: F.poisson(col.reshape(N, N), N).ravel(), in_axes=1, out_axes=1)(B)
    kxs = jnp.asarray([i[0] % N for i in ids])
    kys = jnp.asarray([i[1] % N for i in ids])
    is_cos = jnp.asarray([1.0 if i[2] == 'c' else 0.0 for i in ids])
    nrms = jnp.asarray([i[3] for i in ids], F64)

    @jax.jit
    def rows_for_k(Psi_all, b_k):
        """J_A(psi_j, b_k) for every j, projected on the M modes -> (M, R)."""
        Pg = Psi_all.T.reshape(R, N, N)                       # (R, N, N)
        Zg = b_k.reshape(1, N, N)
        r = lambda A, di, dj: jnp.roll(jnp.roll(A, -di, 1), -dj, 2)
        px, mx, py, my = r(Pg, 1, 0), r(Pg, -1, 0), r(Pg, 0, 1), r(Pg, 0, -1)
        ppp, ppm, pmp, pmm = r(Pg, 1, 1), r(Pg, 1, -1), r(Pg, -1, 1), r(Pg, -1, -1)
        zx, zmx, zy, zmy = r(Zg, 1, 0), r(Zg, -1, 0), r(Zg, 0, 1), r(Zg, 0, -1)
        zpp, zpm, zmp, zmm = r(Zg, 1, 1), r(Zg, 1, -1), r(Zg, -1, 1), r(Zg, -1, -1)
        J = ((px - mx) * (zy - zmy) - (py - my) * (zx - zmx)
             + px * (zpp - zpm) - mx * (zmp - zmm) - py * (zpp - zmp) + my * (zpm - zmm)
             + zy * (ppp - pmp) - zmy * (ppm - pmm) - zx * (ppp - ppm) + zmx * (pmp - pmm))
        J = J * (N * N / 12.0)
        Fk = jnp.fft.fft2(J)                                   # (R, N, N) complex
        vals = Fk[:, kxs, kys]                                 # (R, M)
        proj = (is_cos[None, :] * jnp.real(vals)
                - (1.0 - is_cos)[None, :] * jnp.imag(vals)) / nrms[None, :]
        return proj.T                                          # (M, R)

    T = np.zeros((M, R, R))
    ks = list(range(R))
    if reverse:
        ks = ks[::-1]
    for s0 in range(0, R, block):
        for k in ks[s0:s0 + block]:
            T[:, :, k] = np.asarray(rows_for_k(Psi, B[:, k]))
    return T


def symmetrize(T):
    return T + T.swapaxes(1, 2)


def dense_adv_projected(Phi, B, N, c):
    """Oracle: Phi^T J_A(psi(c), omega(c)) with omega = B c on the full grid."""
    w = (jnp.asarray(B) @ jnp.asarray(c)).reshape(N, N)
    psi = F.poisson(w, N)
    return jnp.asarray(Phi).T @ F.arakawa(psi, w, N).ravel()


# ------------------------------------------------------------------ residual ---

def make_weak(A, Q, lam, dt):
    """r(c, c_prev, nu) with the scaled projected implicit-midpoint form (module docstring)."""
    def weak(c, c_prev, nu):
        cm = 0.5 * (c + c_prev)
        qm = 0.5 * ((Q @ cm) @ cm)                       # = cm^T T cm
        am = A @ cm
        return (A @ c - A @ c_prev - dt * (qm - nu * lam * am)) / (1.0 + 0.5 * dt * nu * lam)
    return weak


def make_lm(fun, budget, gtol=1e-6, trust=np.inf):
    """arms.make_stationary_lm semantics with a pivoted dense solve.  reasons: 0 budget,
    1 tolerance, 2 tiny step, 3 rejected, 4 stationary."""
    def lm(z0, args, tol):
        def evaluate(z):
            r = fun(z, *args)
            J = jax.jacfwd(fun)(z, *args)
            return r, J, jnp.linalg.norm(r)

        def grad(r, J):
            return jnp.linalg.norm(J.T @ r) / (jnp.linalg.norm(J) * jnp.linalg.norm(r) + 1e-300)

        r, J, rn = evaluate(z0)
        reason = jnp.where(jnp.isfinite(rn), jnp.where(grad(r, J) <= gtol, 4,
                                                       jnp.where(rn <= tol, 1, 0)), 3).astype(jnp.int32)

        def body(s):
            z, r, J, rn, lam, it, reason = s
            H = J.T @ J
            g = J.T @ r
            dz = jnp.linalg.solve(H + lam * jnp.diag(jnp.diag(H) + 1e-30), -g)
            ok = jnp.all(jnp.isfinite(dz)) & (jnp.linalg.norm(dz) <= trust)
            zn = z + jnp.where(ok, dz, 0.)
            rn2 = jnp.linalg.norm(fun(zn, *args))
            accept = ok & jnp.isfinite(rn2) & (rn2 < rn)
            r2, J2, rn2 = jax.lax.cond(accept, lambda: evaluate(zn), lambda: (r, J, rn))
            gn = grad(r2, J2)
            tiny = ok & (jnp.linalg.norm(dz) <= 1e-14 * (1 + jnp.linalg.norm(z)))
            reason = jnp.where(gn <= gtol, 4, jnp.where(rn2 <= tol, 1, jnp.where(
                tiny, 2, jnp.where((~accept) & (lam >= 1e14), 3, 0)))).astype(jnp.int32)
            return (jnp.where(accept, zn, z), r2, J2, rn2,
                    jnp.where(accept, jnp.maximum(lam / 3, 1e-12), jnp.minimum(lam * 10, 1e14)),
                    it + 1, reason)

        z, r, J, rn, lam, it, reason = jax.lax.while_loop(
            lambda s: (s[5] < budget) & (s[6] == 0), body,
            (z0, r, J, rn, jnp.asarray(1e-6), jnp.int32(0), reason))
        return z, rn, it, reason, grad(r, J)
    return lm


# ------------------------------------------------------------------ the query ---

def make_query(head, K, q, ops, dt, nsteps, out_every, ic_budget=400, step_budget=200,
               gtol=1e-6, trust=np.inf):
    """One complete timed query: dense omega_0 (N,N) on device -> (nout+1, N, N) dense fields.

    ops: dict(B, Qb, Rb, A, Q, lam, Zcand, Hrot, Hn, Cq)  with B the bank (n, D), (Qb, Rb) its
    thin QR, A = Phi^T B, Q the symmetrised tensor, Zcand the candidate codes (C, K),
    Hrot = h(Zcand) @ Rb^T, Hn its row norms, Cq (D, q) the correction directions.
    head(w) -> c is the arm's full coefficient map on w in R^{K+q}."""
    weak_c = None

    def coef(w):
        return head(w)

    def fun_ic(w, y, Rb):
        return Rb @ coef(w) - y

    ic_lm = make_lm(fun_ic, ic_budget, gtol=gtol)

    def fun_step(w, c_prev, nu, A, Q, lam):
        return make_weak(A, Q, lam, dt)(coef(w), c_prev, nu)

    step_lm = make_lm(fun_step, step_budget, gtol=gtol, trust=trust)

    def query(w0, nu, ops):
        B, Qb, Rb, A, Q, lam = ops['B'], ops['Qb'], ops['Rb'], ops['A'], ops['Q'], ops['lam']
        Zcand, Hrot, Hn, Cq = ops['Zcand'], ops['Hrot'], ops['Hn'], ops['Cq']
        u0 = w0.reshape(-1)
        y = Qb.T @ u0                                            # whitened bank coefficients
        idx = jnp.argmin(Hn - 2 * Hrot @ y)
        z0 = Zcand[idx]
        if q > 0:
            # least-squares correction block given z0, in the whitened metric
            RC = Rb @ Cq
            y0 = jnp.linalg.lstsq(RC, y - Rb @ head(jnp.concatenate([z0, jnp.zeros(q)])))[0]
            w_init = jnp.concatenate([z0, y0])
        else:
            w_init = z0
        w, icrn, icit, icreason, icgn = ic_lm(w_init, (y, Rb), 0.)
        scale = jnp.linalg.norm(u0)

        def step(carry, _):
            w, wprev = carry
            c_prev = coef(w)
            we = w + (w - wprev)
            r0 = jnp.linalg.norm(fun_step(w, c_prev, nu, A, Q, lam))
            re = jnp.linalg.norm(fun_step(we, c_prev, nu, A, Q, lam))
            wi = jnp.where(jnp.isfinite(re) & (re < r0), we, w)
            w2, rn, it, reason, gn = step_lm(wi, (c_prev, nu, A, Q, lam), 1e-9 * scale)
            return (w2, w), (w2, rn, it, reason, gn)

        _, (ws, rn, it, reason, gn) = jax.lax.scan(step, (w, w), None, length=nsteps)
        internal = jnp.concatenate((w[None], ws))
        W = internal[::out_every]
        fields = jax.vmap(lambda w: (B @ coef(w)).reshape(w0.shape))(W)
        return fields, it, rn, reason, W, icit, icreason, gn, icgn, icrn
    return jax.jit(query)


def errors(fields, reference):
    """Burgers convention: e_t = ||f_t - r_t|| / ||r_0||; worst over evolved (t>0), over all
    times, and t=0 separately; plus current-relative."""
    f, r = np.asarray(fields), np.asarray(reference)
    n0 = np.linalg.norm(r[0])
    e = np.linalg.norm((f - r).reshape(len(r), -1), axis=1)
    cur = e / np.maximum(np.linalg.norm(r.reshape(len(r), -1), axis=1), 1e-300)
    return dict(fixed_per_time=(e / n0).tolist(), current_per_time=cur.tolist(),
                worst_evolved=float(np.max(e[1:] / n0)), worst_all=float(np.max(e / n0)),
                t0=float(e[0] / n0), worst_current=float(cur.max()))


def burn(seconds=0.4):
    a = jnp.ones((512, 512), F64) * 0.01
    k = jax.jit(lambda a: a @ a / 512 + 0.01)
    t = time.perf_counter()
    while time.perf_counter() - t < seconds:
        a = k(a)
        jax.block_until_ready(a)
