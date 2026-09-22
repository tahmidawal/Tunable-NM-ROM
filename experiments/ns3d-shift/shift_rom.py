"""Co-moving ("freezing") NS3D ROM: the translation is an online unknown.

u(x,t) = v(x - c(t), t) with v = G a in a fixed CENTERED bank turns the periodic
Navier-Stokes equation into

    dv/dt = cdot . grad v + P[N(v)] + nu Laplacian v,

because N, the Laplacian and the Leray projector all commute with translation.
The reduced weak residual against the project's fixed solenoidal Fourier tests is
therefore the ordinary one plus a single term linear in the frame increment
delta = c^{n+1} - c^n. Nothing is shifted at run time: A = Phi^T G, the advection
tensor T and the derivative projections D_d = Phi^T d_d G are all offline.

Arrays: bank G is (3*n**3, r); A is (M, r); T is (M, r, r); Dd is (3, M, r).
"""
from __future__ import annotations

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import ns3d_fom as F
import ns3d_rom as R
from ns2d_rom import make_lm


# ------------------------------------------------------------------ operators --

def spectral_derivative(columns, n, geom):
    """d/dx_d of each bank column. columns (r, 3, n, n, n) -> (3, r, 3, n, n, n)."""
    spec = F.fft(jnp.asarray(columns))
    k = geom[0]
    return jnp.stack([F.ifft(1j * k[d] * spec) for d in range(3)])


def build_operators(G, n, modes, check=True):
    """A, lam, T, Dd, S and their validation against direct full-grid evaluation."""
    G = np.asarray(G, dtype=np.float64)
    r = G.shape[1]
    if modes % 4:
        raise RuntimeError("test count must be a multiple of 4 so cos/sin pairs "
                           "at both polarizations are complete")
    if modes < r + 16:
        raise RuntimeError(f"need M >= rank+16, got M={modes} r={r}")
    orth = float(np.max(np.abs(G.T @ G - np.eye(r))))
    if orth > 1e-10:
        raise RuntimeError(f"bank is not orthonormal: {orth}")
    geom = F.geometry(n)
    Phi, lam, ids = R.test_modes(n, modes)
    A = np.asarray(Phi.T @ G)
    fields = jnp.asarray(G.T.reshape(r, 3, n, n, n))
    grads = spectral_derivative(fields, n, geom)          # (3, r, 3, n, n, n)
    flat = np.asarray(grads.reshape(3, r, -1))
    Dd = np.einsum("pm,drp->dmr", Phi, flat)              # (3, M, r)
    S = np.einsum("drp,ps->drs", flat, G)                 # (3, r, r)
    T = R.build_tensor(G, n, ids)
    report = dict(modes=int(modes), rank=int(r), bank_orthonormality=orth,
                  test_orthogonality=float(np.linalg.norm(Phi.T @ Phi - np.eye(modes))
                                           / np.sqrt(modes)),
                  A_condition=float(np.linalg.cond(A)),
                  A_smallest_singular=float(np.linalg.svd(A, compute_uv=False)[-1]))
    if check:
        rng = np.random.default_rng(7)
        c = rng.normal(size=r)
        want = np.asarray(R.dense_projection(Phi, jnp.asarray(G), jnp.asarray(c), n))
        got = np.asarray(R.contract(jnp.asarray(T), jnp.asarray(c)))
        report["tensor_relative"] = float(np.linalg.norm(got - want)
                                          / max(np.linalg.norm(want), 1e-300))
        # D_d against a direct projection of the spectral derivative of G c.
        field = (jnp.asarray(G) @ jnp.asarray(c)).reshape(1, 3, n, n, n)
        direct = np.asarray(spectral_derivative(field, n, geom).reshape(3, -1)) @ Phi
        report["derivative_relative"] = float(
            np.max(np.linalg.norm(direct - (Dd @ c), axis=1)
                   / np.maximum(np.linalg.norm(direct, axis=1), 1e-300)))
        # The diffusion shortcut: <phi_m, Laplacian G c> == -lam_m (A c)_m.
        lap = np.asarray(F.ifft(-geom[1] * F.fft(field)).reshape(-1)) @ Phi
        report["diffusion_relative"] = float(np.linalg.norm(lap + lam * (A @ c))
                                             / max(np.linalg.norm(lap), 1e-300))
        # S_d is skew on a periodic box; the phase row relies on it.
        report["S_skew"] = float(np.max(np.abs(S + np.swapaxes(S, 1, 2)))
                                 / max(np.max(np.abs(S)), 1e-300))
        # Independent sign fix: d/dc_d of Phi^T shift(Gc, c*n) must equal -D_d c.
        # This uses only the FFT shift helper, never the residual or D_d build.
        base = (jnp.asarray(G) @ jnp.asarray(c)).reshape(3, n, n, n)
        eps = 1e-6
        fd = []
        for d in range(3):
            off = np.zeros(3)
            off[d] = eps
            plus = np.asarray(shift_field(base, jnp.asarray(off * n)).reshape(-1))
            minus = np.asarray(shift_field(base, jnp.asarray(-off * n)).reshape(-1))
            fd.append((plus - minus) @ Phi / (2 * eps))
        fd = np.stack(fd)
        report["shift_sign_relative"] = float(
            np.max(np.linalg.norm(fd + (Dd @ c), axis=1)
                   / np.maximum(np.linalg.norm(fd, axis=1), 1e-300)))
        worst = max(report["tensor_relative"], report["derivative_relative"],
                    report["diffusion_relative"], report["test_orthogonality"],
                    report["S_skew"])
        if worst > 1e-9 or report["shift_sign_relative"] > 1e-5:
            raise RuntimeError(f"reduced operators failed validation: {report}")
    return dict(A=A, lam=np.asarray(lam), T=T, Dd=Dd, S=S, ids=ids), report


# ------------------------------------------------------------- the shift query --

def make_shift_run(dt, nsteps, out_every, n, r, mode="free", gauge=0.0,
                   budget=60, gtol=1e-7, retain=False):
    """Complete co-moving query. mode: 'free' (solve delta), 'zero' (delta==0),
    'oracle' (delta supplied per step). Returns fields at the output times."""
    assert nsteps % out_every == 0
    assert mode in ("free", "zero", "oracle")
    solve_delta = mode == "free"
    nvar = r + 3 if solve_delta else r

    def residual(w, prev, nu, A, T, lam, Dd, S, forced):
        a = w[:r]
        delta = w[r:] if solve_delta else forced
        mid = 0.5 * (a + prev)
        adv = jnp.einsum("mjk,j,k->m", T, mid, mid)
        frame = jnp.einsum("dmr,d,r->m", Dd, delta, mid)
        base = (A @ (a - prev) - dt * (adv - nu * lam * (A @ mid)) - frame) / (
            1.0 + 0.5 * dt * nu * lam)
        if not (solve_delta and gauge > 0.0):
            return base
        # Phase condition: the change in v carries no translation component.
        step = a - prev
        row = jnp.einsum("drs,r,s->d", S, prev, step)
        scale = jnp.linalg.norm(jnp.einsum("drs,r->ds", S, prev), axis=1) + 1e-30
        return jnp.concatenate((base, gauge * row / scale))

    lm = make_lm(residual, budget, gtol=gtol)

    @jax.jit
    def run(u0, nu, basis, A, T, lam, Dd, S, forced):
        """forced is (nsteps, 3); it is ignored unless mode == 'oracle'."""
        c0 = grid_centroid(u0)
        a0 = basis.T @ shift_field(u0, -c0 * n).ravel()
        w0 = jnp.concatenate((a0, jnp.zeros(3))) if solve_delta else a0

        def step(carry, index):
            w, c = carry
            prev = w[:r]
            w0i = w if solve_delta else prev
            args = (prev, nu, A, T, lam, Dd, S, forced[index])
            wn, rn, it, reason, gn = lm(w0i, args, 0.0)
            delta = wn[r:] if solve_delta else forced[index]
            cn = (c + delta) % 1.0
            return (wn, cn), (rn, it, reason, gn, delta)

        def block(carry, b):
            carry, info = jax.lax.scan(step, carry, b * out_every + jnp.arange(out_every))
            w, c = carry
            return carry, (shift_field((basis @ w[:r]).reshape(3, n, n, n), c * n), info, c)

        frame0 = shift_field((basis @ a0).reshape(3, n, n, n), c0 * n)
        carry, (fields, info, centers) = jax.lax.scan(
            block, (w0, c0), jnp.arange(nsteps // out_every))
        fields = jnp.concatenate((frame0[None], fields))
        centers = jnp.concatenate((c0[None], centers))
        flat = tuple(x.reshape(-1, *x.shape[2:]) for x in info)
        return fields, centers, flat
    return run


# ------------------------------------------------------- centroid tracker arm --

def shift_field(field, offset_samples):
    spec = jnp.fft.fftn(field, axes=(-3, -2, -1))
    n = field.shape[-1]
    modes = jnp.fft.fftfreq(n) * n
    for axis in range(3):
        phase = jnp.exp(-2j * jnp.pi * modes * (offset_samples[axis] / n))
        shape = [1, 1, 1]
        shape[axis] = n
        spec = spec * phase.reshape((1, *shape))
    return jnp.fft.ifftn(spec, axes=(-3, -2, -1)).real


def grid_centroid(field):
    weight = jnp.sum(field * field, axis=0)
    n = field.shape[-1]
    angle = 2 * jnp.pi * jnp.arange(n, dtype=jnp.float64) / n
    centers = []
    for axis in range(3):
        marginal = weight.sum(axis=tuple(i for i in range(3) if i != axis))
        centers.append(jnp.arctan2(jnp.sum(marginal * jnp.sin(angle)),
                                   jnp.sum(marginal * jnp.cos(angle))) / (2 * jnp.pi) % 1.0)
    return jnp.stack(centers)


def make_tracker_run(dt, nsteps, out_every, n):
    """Grok's arm: step the centered bank, then re-centre on the ROM's own
    centroid and reproject. Same startup step, grid nonlinearity."""
    assert nsteps % out_every == 0

    @jax.jit
    def run(u0, nu, basis, linear, geom):
        identity = jnp.eye(linear.shape[0])
        half = jnp.linalg.cholesky(identity - 0.5 * dt * nu * linear)
        full = jnp.linalg.cholesky(identity - dt * nu * linear)

        def solve(factor, b):
            return jax.scipy.linalg.cho_solve((factor, True), b)

        def advect(a):
            field = (basis @ a).reshape(3, n, n, n)
            return basis.T @ F.ifft(F.nonlinear(F.fft(field), geom)).ravel()

        def step(carry, _):
            a, c = carry
            current = advect(a)
            pred = solve(full, a + dt * current)
            new = solve(half, a + 0.5 * dt * (nu * (linear @ a) + current + advect(pred)))
            field = (basis @ new).reshape(3, n, n, n)
            delta = grid_centroid(field)
            a_new = basis.T @ shift_field(field, -delta * n).ravel()
            return (a_new, (c + delta) % 1.0), None

        def block(carry, _):
            carry, _ = jax.lax.scan(step, carry, None, length=out_every)
            a, c = carry
            return carry, shift_field((basis @ a).reshape(3, n, n, n), c * n)

        c0 = grid_centroid(u0)
        a0 = basis.T @ shift_field(u0, -c0 * n).ravel()
        frame0 = shift_field((basis @ a0).reshape(3, n, n, n), c0 * n)
        _, fields = jax.lax.scan(block, (a0, c0), None, length=nsteps // out_every)
        return jnp.concatenate((frame0[None], fields))
    return run


# ------------------------------------------------------------- conditioning ----

def jacobian_report(ops, n, r, dt, nu, prev, nxt, gauge):
    """Conditioning of d residual / d (a, delta) at a development state.

    The headline number is the deflated block (I - Ja Ja^+) Jdelta: shift
    information that a coefficient update cannot absorb. Nonzero delta columns
    alone prove nothing, and gauge rows can inflate the plain spectrum by weight.
    """
    out = {}
    for label, weight in (("no_gauge", 0.0), ("gauge", gauge)):

        def residual(w, prev=prev, weight=weight):
            a, delta = w[:r], w[r:]
            mid = 0.5 * (a + prev)
            adv = jnp.einsum("mjk,j,k->m", ops["T"], mid, mid)
            frame = jnp.einsum("dmr,d,r->m", ops["Dd"], delta, mid)
            base = (ops["A"] @ (a - prev) - dt * (adv - nu * ops["lam"] * (ops["A"] @ mid))
                    - frame) / (1.0 + 0.5 * dt * nu * ops["lam"])
            if weight <= 0.0:
                return base
            step = a - prev
            row = jnp.einsum("drs,r,s->d", ops["S"], prev, step)
            scale = jnp.linalg.norm(jnp.einsum("drs,r->ds", ops["S"], prev), axis=1) + 1e-30
            return jnp.concatenate((base, weight * row / scale))

        w = jnp.concatenate((jnp.asarray(nxt), jnp.zeros(3)))
        J = np.asarray(jax.jacfwd(residual)(w))
        Ja, Jd = J[:, :r], J[:, r:]
        sv = np.linalg.svd(J, compute_uv=False)
        sva = np.linalg.svd(Ja, compute_uv=False)
        deflated = Jd - Ja @ np.linalg.lstsq(Ja, Jd, rcond=None)[0]
        svd_def = np.linalg.svd(deflated, compute_uv=False)
        # Analytic delta columns, to confirm the AD Jacobian independently.
        mid = 0.5 * (np.asarray(nxt) + np.asarray(prev))
        pre = 1.0 + 0.5 * dt * nu * np.asarray(ops["lam"])
        analytic = -(np.einsum("dmr,r->md", ops["Dd"], mid)) / pre[:, None]
        gap = float(np.linalg.norm(Jd[:analytic.shape[0]] - analytic)
                    / max(np.linalg.norm(analytic), 1e-300))
        out[label] = dict(
            singular_values=[float(x) for x in sv],
            condition=float(sv[0] / max(sv[-1], 1e-300)),
            smallest_over_largest=float(sv[-1] / max(sv[0], 1e-300)),
            coefficient_block_largest=float(sva[0]),
            coefficient_block_smallest=float(sva[-1]),
            delta_column_norms=[float(x) for x in np.linalg.norm(Jd, axis=0)],
            deflated_singular_values=[float(x) for x in svd_def],
            deflated_smallest_over_Ja_largest=float(svd_def[-1] / max(sva[0], 1e-300)),
            analytic_delta_column_relative=gap,
        )
    return out


def translation_tangent_report(G, n, a):
    """How much of each translation tangent d_d(G a) lives inside span(G)."""
    G = np.asarray(G)
    geom = F.geometry(n)
    field = (jnp.asarray(G) @ jnp.asarray(a)).reshape(1, 3, n, n, n)
    tangents = np.asarray(spectral_derivative(field, n, geom).reshape(3, -1))
    inside = tangents @ G
    captured = np.linalg.norm(inside, axis=1) / np.maximum(
        np.linalg.norm(tangents, axis=1), 1e-300)
    return dict(captured_fraction=[float(x) for x in captured],
                tangent_norms=[float(x) for x in np.linalg.norm(tangents, axis=1)],
                tangent_rank_in_span=int(np.linalg.matrix_rank(inside, tol=1e-10)))
