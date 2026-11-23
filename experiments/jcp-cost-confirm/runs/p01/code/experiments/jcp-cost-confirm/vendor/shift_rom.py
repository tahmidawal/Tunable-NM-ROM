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


# ------------------------------------------------- mesh-scalable operator build --
#
# The dense test matrix Phi is (M, 3n^3): 1.8 GB at n=64 and 14.7 GB at n=128, and
# the einsums against it dominate the build. It is never needed. The tests are
# Fourier modes, so <phi_m, f> is one coefficient of fft(f), which is exactly the
# extraction ns3d_rom.build_tensor already uses; and <phi_m, d_d f> is the same
# coefficient scaled by i*2*pi*k_d before taking the real/imaginary part. So A and
# D_d come from a single FFT of the bank, with no Phi anywhere.

import ns3d_model as MD  # noqa: E402  (coords, for the dense validation subset)


def test_mode_ids(n, m):
    """The ids and eigenvalues of ns3d_rom.test_modes, without the dense fields.

    Enumeration order is identical by construction; `check_test_mode_ids` asserts it.
    """
    if m % 4:
        raise RuntimeError("test count must be a multiple of 4")
    maxk = int(np.ceil(n / 3)) - 1
    candidates = [(a * a + b * b + c * c, a, b, c)
                  for a in range(-maxk, maxk + 1) for b in range(-maxk, maxk + 1)
                  for c in range(-maxk, maxk + 1)
                  if (a > 0 or (a == 0 and b > 0) or (a == 0 and b == 0 and c > 0))]
    candidates.sort()
    ids, lams = [], []
    for square, a, b, c in candidates:
        wave = np.asarray([a, b, c], dtype=float)
        axis = np.eye(3)[np.argmin(np.abs(wave))]
        first = np.cross(wave, axis)
        first /= np.linalg.norm(first)
        second = np.cross(wave / np.sqrt(square), first)
        for pol in (first, second):
            for label in ("cos", "sin"):
                ids.append(dict(wave=[a, b, c], polarization=pol.tolist(), kind=label))
                lams.append((2 * np.pi) ** 2 * square)
                if len(ids) == m:
                    return ids, np.asarray(lams)
    raise ValueError("too many smooth vector tests for this mesh")


def check_test_mode_ids(n, m):
    """The cheap enumeration must equal ns3d_rom.test_modes exactly. Small n only."""
    _, lam, ids = R.test_modes(n, m)
    mine, mylam = test_mode_ids(n, m)
    if len(mine) != len(ids):
        raise RuntimeError("test id count differs")
    for got, want in zip(mine, ids):
        if got["wave"] != want["wave"] or got["kind"] != want["kind"] or \
                float(np.max(np.abs(np.asarray(got["polarization"])
                                    - np.asarray(want["polarization"])))) > 0:
            raise RuntimeError(f"test id mismatch: {got} vs {want}")
    return float(np.max(np.abs(mylam - lam)))


def dense_test_modes(n, ids):
    """Dense columns for a handful of ids, for validation. (3n^3, len(ids))."""
    xyz = np.asarray(MD.coords(n)).reshape(n, n, n, 3)
    columns = []
    for row in ids:
        wave = np.asarray(row["wave"], dtype=float)
        pol = np.asarray(row["polarization"])
        angle = 2 * np.pi * np.einsum("...d,d->...", xyz, wave)
        fn = np.cos if row["kind"] == "cos" else np.sin
        columns.append((pol[:, None, None, None] * fn(angle)
                        * np.sqrt(2 / n ** 3)).ravel())
    return np.stack(columns, axis=1)


def extract_tests(fields, ids_arrays, n):
    """<phi_m, f_j> and <phi_m, d_d f_j> by Fourier extraction.

    fields (J, 3, n, n, n) -> base (J, M), derivative (3, J, M).
    """
    index, pol, cosine, wave = ids_arrays
    spec = F.fft(jnp.asarray(fields))
    selected = spec[:, :, index[:, 0], index[:, 1], index[:, 2]]      # (J, 3, M)
    value = jnp.einsum("jcm,mc->jm", selected, pol)
    scale = np.sqrt(2 * n ** 3)

    def real_part(v):
        return scale * jnp.where(cosine[None, :], v.real, -v.imag)

    base = real_part(value)
    derivative = jnp.stack([real_part(1j * (2 * np.pi * wave[:, d])[None, :] * value)
                            for d in range(3)])
    return base, derivative


def ids_to_arrays(ids, n):
    index = np.asarray([row["wave"] for row in ids], dtype=int) % n
    pol = jnp.asarray(np.asarray([row["polarization"] for row in ids]))
    cosine = jnp.asarray(np.asarray([row["kind"] == "cos" for row in ids]))
    wave = jnp.asarray(np.asarray([row["wave"] for row in ids], dtype=np.float64))
    return jnp.asarray(index), pol, cosine, wave


def build_operators_fast(G, n, modes, check_modes=8, chunk=8, seed=11):
    """A, lam, T, Dd without ever forming the dense test matrix.

    Validation uses a random subset of `check_modes` densely built tests, which is
    a real check at every mesh at 1/(M/check_modes) of the memory.
    """
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
    ids, lam = test_mode_ids(n, modes)
    arrays = ids_to_arrays(ids, n)
    A = np.empty((modes, r), dtype=np.float64)
    Dd = np.empty((3, modes, r), dtype=np.float64)
    for start in range(0, r, chunk):
        block = jnp.asarray(G[:, start:start + chunk].T.reshape(-1, 3, n, n, n))
        base, derivative = extract_tests(block, arrays, n)
        A[:, start:start + chunk] = np.asarray(base).T
        Dd[:, :, start:start + chunk] = np.asarray(derivative).transpose(0, 2, 1)
    T = R.build_tensor(G, n, ids)

    report = dict(modes=int(modes), rank=int(r), bank_orthonormality=orth,
                  check_modes=int(check_modes),
                  A_condition=float(np.linalg.cond(A)),
                  A_smallest_singular=float(np.linalg.svd(A, compute_uv=False)[-1]))
    rng = np.random.default_rng(seed)
    subset = np.sort(rng.choice(modes, size=min(check_modes, modes), replace=False))
    Phi = dense_test_modes(n, [ids[i] for i in subset])
    geom = F.geometry(n)
    c = rng.normal(size=r)
    field = (jnp.asarray(G) @ jnp.asarray(c)).reshape(1, 3, n, n, n)
    report["subset_test_orthogonality"] = float(
        np.linalg.norm(Phi.T @ Phi - np.eye(len(subset))) / np.sqrt(len(subset)))
    report["A_relative"] = float(
        np.linalg.norm(Phi.T @ (G @ c) - A[subset] @ c)
        / max(np.linalg.norm(A[subset] @ c), 1e-300))
    direct = np.asarray(spectral_derivative(field, n, geom).reshape(3, -1)) @ Phi
    report["derivative_relative"] = float(
        np.max(np.linalg.norm(direct - (Dd[:, subset] @ c), axis=1)
               / np.maximum(np.linalg.norm(direct, axis=1), 1e-300)))
    lap = np.asarray(F.ifft(-geom[1] * F.fft(field)).reshape(-1)) @ Phi
    report["diffusion_relative"] = float(
        np.linalg.norm(lap + lam[subset] * (A[subset] @ c))
        / max(np.linalg.norm(lap), 1e-300))
    want = np.asarray(R.dense_projection(Phi, jnp.asarray(G), jnp.asarray(c), n))
    got = np.asarray(R.contract(jnp.asarray(T), jnp.asarray(c)))[subset]
    report["tensor_relative"] = float(np.linalg.norm(got - want)
                                      / max(np.linalg.norm(want), 1e-300))
    # Independent sign fix: d/dc_d of Phi^T shift(Gc, c*n) must equal -D_d c.
    base_field = (jnp.asarray(G) @ jnp.asarray(c)).reshape(3, n, n, n)
    eps = 1e-6
    fd = []
    for d in range(3):
        off = np.zeros(3)
        off[d] = eps
        plus = np.asarray(shift_field(base_field, jnp.asarray(off * n)).reshape(-1))
        minus = np.asarray(shift_field(base_field, jnp.asarray(-off * n)).reshape(-1))
        fd.append((plus - minus) @ Phi / (2 * eps))
    fd = np.stack(fd)
    report["shift_sign_relative"] = float(
        np.max(np.linalg.norm(fd + (Dd[:, subset] @ c), axis=1)
               / np.maximum(np.linalg.norm(fd, axis=1), 1e-300)))
    worst = max(report["A_relative"], report["derivative_relative"],
                report["diffusion_relative"], report["tensor_relative"],
                report["subset_test_orthogonality"])
    if worst > 1e-9 or report["shift_sign_relative"] > 1e-5:
        raise RuntimeError(f"reduced operators failed validation: {report}")
    return dict(A=A, lam=np.asarray(lam), T=T, Dd=Dd, ids=ids), report


# --------------------------------------------------------- the fast step solver --
#
# `ns2d_rom.make_lm` is generic: jacfwd, a data-dependent while_loop, an
# accept/reject trial that re-evaluates residual AND Jacobian on acceptance, and a
# normalized-gradient test. For this residual that is ~90 % of the query cost while
# the arithmetic (one M x r x r contraction and a 67-unknown least squares, a few
# times per step) is two orders of magnitude cheaper. The residual here is quadratic
# in `a` and bilinear in `(delta, a)`, so a fixed number of damped Gauss-Newton
# steps with an analytic Jacobian converges to the same point, in a statically
# unrolled scan that XLA can fuse.

def make_frozen_step(dt, n, r, iters, damping):
    """One reduced step: `iters` damped Gauss-Newton sweeps with an analytic Jacobian.

    Everything constant over the sweep is hoisted into `prepare`, which the caller
    runs once per query:
      * `sym` = T + T^T over the last two axes, so ONE contraction gives both the
        advection (contract once more) and its Jacobian (it is half of `sym @ mid`),
        instead of two contractions plus a separate product;
      * `jconst` = the coefficient-Jacobian terms that do not depend on the iterate;
      * the Crank-Nicolson preconditioner folded into every operator.
    The damping stays a Marquardt per-column diagonal. A scalar Levenberg term was
    tried and rejected: the three shift columns are more than ten times the norm of
    the coefficient columns, so a mean-diagonal damping over-damps the coefficient
    directions and moved the converged trajectory by 2e-5 relative, against 7e-10
    for the diagonal form.
    """
    size = r + 3

    def prepare(nu, A, T, lam, Dd):
        pre = 1.0 + 0.5 * dt * nu * lam
        Apre = A / pre[:, None]
        sym = (T + jnp.swapaxes(T, 1, 2)) / pre[:, None, None]
        lamApre = (nu * lam / pre)[:, None] * A
        Ddpre = Dd / pre[None, :, None]
        return Apre, sym, Apre + 0.5 * dt * lamApre, lamApre, Ddpre

    def residual_and_jacobian(w, prev, ops):
        Apre, sym, jconst, lamApre, Ddpre = ops
        a, delta = w[:r], w[r:]
        mid = 0.5 * (a + prev)
        symmid = jnp.einsum("msk,k->ms", sym, mid)            # d(2*advection)/d(mid)
        shift_cols = jnp.einsum("dmr,r->md", Ddpre, mid)
        res = (Apre @ (a - prev) - 0.5 * dt * (symmid @ mid) + dt * (lamApre @ mid)
               - shift_cols @ delta)
        jac_a = jconst - 0.5 * dt * symmid - 0.5 * jnp.einsum("dmr,d->mr", Ddpre, delta)
        return res, jnp.concatenate((jac_a, -shift_cols), axis=1)

    def solve(w0, prev, ops):
        def body(w, _):
            res, jac = residual_and_jacobian(w, prev, ops)
            hessian = jac.T @ jac
            factor = jnp.linalg.cholesky(
                hessian + jnp.diag(damping * jnp.diagonal(hessian) + 1e-300))
            step = jax.scipy.linalg.cho_solve((factor, True), -(jac.T @ res))
            return w + step, jnp.linalg.norm(res)
        w, norms = jax.lax.scan(body, w0, None, length=iters, unroll=iters)
        return w, norms[-1]
    return solve, prepare, residual_and_jacobian


def make_frozen_run(dt, nsteps, out_every, n, r, iters=4, damping=1e-6,
                    extrapolate=True, diagnose=True):
    """Complete co-moving query with the fixed-iteration solver. Gauge-free.

    `diagnose=False` drops the per-step records; the fields are bit-identical and
    it is the variant that should be timed, because a deployment does not compute
    them. `shift_ladder.py` asserts the two agree.
    """
    assert nsteps % out_every == 0
    solve, prepare, _ = make_frozen_step(dt, n, r, iters, damping)

    @jax.jit
    def run(u0, nu, basis, A, T, lam, Dd):
        ops = prepare(nu, A, T, lam, Dd)

        c0 = grid_centroid(u0)
        a0 = basis.T @ shift_field(u0, -c0 * n).ravel()
        w0 = jnp.concatenate((a0, jnp.zeros(3)))

        def step(carry, _):
            w, previous, c = carry
            guess = w + (w - previous) if extrapolate else w
            new, rn = solve(guess, w[:r], ops)
            centre = (c + new[r:]) % 1.0
            out = (rn, new[r:]) if diagnose else None
            return (new, w, centre), out

        def block(carry, _):
            carry, info = jax.lax.scan(step, carry, None, length=out_every)
            w, _, c = carry
            return carry, (shift_field((basis @ w[:r]).reshape(3, n, n, n), c * n), info)

        frame0 = shift_field((basis @ a0).reshape(3, n, n, n), c0 * n)
        _, (fields, info) = jax.lax.scan(block, (w0, w0, c0), None,
                                         length=nsteps // out_every)
        fields = jnp.concatenate((frame0[None], fields))
        if not diagnose:
            return fields
        return fields, tuple(x.reshape(nsteps, *x.shape[2:]) for x in info)
    return run
