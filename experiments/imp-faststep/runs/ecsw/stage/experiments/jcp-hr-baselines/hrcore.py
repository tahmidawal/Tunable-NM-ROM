"""jcp-hr-baselines core: problem data, the central-difference FOM the baselines project, and stencil algebra
(DESIGN.md sections 2, 3.1-3.3).

State: interior nodes of an L-interval grid per axis, flat, x slowest, zero Dirichlet walls (as the E0 references and
both vendor FOMs). The FOM operator is the E0 reference operator (refcore.adv_central, refcore.lap; same arithmetic):

    F_h(u) = N_h(u) - nu Lap_h u,   N_h(u)_i = u_i * sum_a (u_{i+e_a} - u_{i-e_a}) * (L/2).

Time stepping (one residual per step, t2core.SCHEMES coefficients):

    R_n(u) = a0 u + a1 u_n + a2 u_{n-1} + dt (b0 F_h(u) + b1 F_h(u_n)).

Stencil slots per node: 0 = centre, then (-x, +x, -y, +y[, -z, +z]); a wall neighbour points at the zero row n.
Everything is f64.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / 'experiments' / 'mr-burgers2d', HERE / 'vendor' / 'quad3d' / 'vendor' / 'burgers3d-span'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import engines as E2          # noqa: E402  (2D family, IC, DST)
import common as C3           # noqa: E402  (3D family, IC, DST)

# scheme: (a0, a1, a2, b0, b1, number of BE start-up steps)  -- t2core.SCHEMES
SCHEMES = dict(BE=(1., -1., 0., 1., 0., 0), CN=(1., -1., 0., .5, .5, 0), CNR=(1., -1., 0., .5, .5, 2),
               BDF2=(1.5, -2., .5, 1., 0., 1))
BE_CO = (1., -1., 0., 1., 0.)
DT0 = {2: 0.005, 3: 0.01}
DT_FACTORS = (1., 1.25, 5. / 3., 2.5)                          # speed3d's registered steps (3D)
# time-stepping options (scheme, dt): 2D = P1's LSPG-BE and LSPG-CN-R at dt0; 3D = speed3d's ladder (A1.4)
OPTIONS = {2: [('BE', 0.005), ('CNR', 0.005)],
           3: [(sc, 0.01 * f) for sc in ('BE', 'CN', 'BDF2') for f in DT_FACTORS]}
NMAX = {2: 50, 3: 25}                                          # most steps of any option (static buffers)
TRUST = {2: ('max', 0.01), 3: ('rms', 0.05)}                   # A1.15: per-dimension trust as ours'
SNAP_STEPS = (0, 1, 2, 4, 7, 10, 15, 20, 25, 30, 40, 50)      # the banks' snapshot steps at dt = 0.005
SNAP_DT = 0.005
SNAP_SCHEME = 'CNR'
TRAIN = {2: dict(seed=0, count=576), 3: dict(seed=923701, count=1536)}
HORIZON, SPACING = 0.25, 0.05


def sha(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


# ------------------------------------------------------------------------------------------- problem data --

def train_draw_2d(seed=0, count=576):
    """burgers2d_film.sample_params(seed, count)[:5] re-implemented (asserted equal in tests): rows (cx, cy, w, a, nu)."""
    rng = np.random.default_rng(seed)
    cx = rng.uniform(0.15, 0.85, count)
    cy = rng.uniform(0.15, 0.85, count)
    w = rng.uniform(0.05, 0.20, count)
    a = rng.uniform(0.5, 2.0, count)
    nu = np.exp(rng.uniform(np.log(0.01), np.log(0.1), count))
    return np.stack([cx, cy, w, a, nu], 1)


COHORTS_2D = {'dev6': [(7090702, 4), (911702, 2)], 'val32': [(20260927, 32)], 'test64': [(20260916, 64)]}
COHORTS_3D = {'dev4': (923651, 4), 'val64': (923801, 64), 'test32': (923901, 32)}
ALLOWED = {2: ('dev6', 'val32'), 3: ('val64',)}


class Family:
    """Uniform access to a set of cases: .n, .nu(j), .ic(L, j) -> flat interior (L-1)^d, .sha256."""

    def __init__(self, dim, name, count=None):
        """count: 3D only, a prefix of the table (rows are prefix-stable: b3d_common.draw_param_table)."""
        self.dim, self.name = dim, name
        if dim == 2:
            if name == 'train':
                self.D = train_draw_2d(**TRAIN[2])
            else:
                assert name in COHORTS_2D, name
                self.D = np.concatenate([E2.params_draw(s, c) for s, c in COHORTS_2D[name]])
            self.n = len(self.D)
            self.sha256 = sha(self.D)
        else:
            if name == 'train':
                s, c = TRAIN[3]['seed'], TRAIN[3]['count']
            else:
                s, c = COHORTS_3D[name]
            self.T = C3.table(s, c if count is None else min(count, c))
            self.n = int(self.T['m'])
            self.sha256 = self.T['sha256']

    def nu(self, j):
        return float(self.D[j, 4]) if self.dim == 2 else float(self.T['nu'][j])

    def ic(self, L, j):
        if self.dim == 2:
            return np.asarray(E2.initial(L, self.D[j]), float)[1:-1, 1:-1].reshape(-1)
        return np.asarray(C3.initial_interior(L + 1, self.T, j), float)

    def descriptor(self, j):
        if self.dim == 2:
            return np.asarray(self.D[j], float)
        T = self.T
        return np.concatenate([[T['B'][j]], np.ravel(T['c'][j]), np.ravel(T['w'][j]), np.ravel(T['rho'][j]),
                               [T['A'][j], T['nu'][j]]]).astype(float)


def _raw_rows(dim, name):
    """Descriptor rows without the s_star normalisation (3D: b3d_common.draw_param_table, the same draws)."""
    if dim == 2:
        D = train_draw_2d(**TRAIN[2]) if name == 'train' else np.concatenate([E2.params_draw(s, c)
                                                                              for s, c in COHORTS_2D[name]])
        return np.asarray(D, float)
    s, c = (TRAIN[3]['seed'], TRAIN[3]['count']) if name == 'train' else COHORTS_3D[name]
    T = C3.b3.draw_param_table(s, c)
    return np.concatenate([T['B'][:, None], T['c'].reshape(c, -1), T['w'], T['rho'], T['A'][:, None],
                           T['nu'][:, None]], 1).astype(float)


def assert_disjoint(dim):
    """No training row equals (to 1e-12) any row of any development, validation or test cohort."""
    A = _raw_rows(dim, 'train')
    names = list(COHORTS_2D) if dim == 2 else list(COHORTS_3D)
    for nm in names:
        Bm = _raw_rows(dim, nm)
        for j in range(len(Bm)):
            assert not np.any(np.all(np.abs(A - Bm[j][None]) <= 1e-12, axis=1)), (dim, nm, j)
    return True


# ------------------------------------------------------------------------------------------------ operators --

def shape_of(d, L):
    return (L - 1,) * d


def adv_central(g, L):
    """refcore.adv_central (same arithmetic)."""
    p = jnp.pad(g, 1)
    if g.ndim == 2:
        c = p[1:-1, 1:-1]
        s = (p[2:, 1:-1] - p[:-2, 1:-1]) + (p[1:-1, 2:] - p[1:-1, :-2])
    else:
        c = p[1:-1, 1:-1, 1:-1]
        s = ((p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) + (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1])
             + (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2]))
    return c * s * (L / 2.0)


def adv_upwind(g, L):
    """refcore.adv_upwind (same arithmetic): the banks' generator operator (generator probe, A2.3, only)."""
    p = jnp.pad(g, 1)
    if g.ndim == 2:
        c, xm, xp, ym, yp = p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]
        return c * L * (jnp.where(c > 0, c - xm, xp - c) + jnp.where(c > 0, c - ym, yp - c))
    dx = 1.0 / L
    c = p[1:-1, 1:-1, 1:-1]
    pos = c > 0
    ux = jnp.where(pos, (c - p[:-2, 1:-1, 1:-1]) / dx, (p[2:, 1:-1, 1:-1] - c) / dx)
    uy = jnp.where(pos, (c - p[1:-1, :-2, 1:-1]) / dx, (p[1:-1, 2:, 1:-1] - c) / dx)
    uz = jnp.where(pos, (c - p[1:-1, 1:-1, :-2]) / dx, (p[1:-1, 1:-1, 2:] - c) / dx)
    return c * (ux + uy + uz)


def dsum(g, L):
    """sum_a (u_{+a} - u_{-a}) * (L/2) (the central derivative sum, without the u_i factor)."""
    p = jnp.pad(g, 1)
    if g.ndim == 2:
        s = (p[2:, 1:-1] - p[:-2, 1:-1]) + (p[1:-1, 2:] - p[1:-1, :-2])
    else:
        s = ((p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) + (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1])
             + (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2]))
    return s * (L / 2.0)


def lap(g, L):
    """refcore.lap (same arithmetic)."""
    p = jnp.pad(g, 1)
    if g.ndim == 2:
        c, xm, xp, ym, yp = p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]
        return L ** 2 * (xm + xp + ym + yp - 4 * c)
    dx = 1.0 / L
    c = p[1:-1, 1:-1, 1:-1]
    lxy = (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1] + p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1] - 4.0 * c)
    lz = (p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2] - 2.0 * c)
    return (lxy + 1.0 * lz) / dx ** 2


def F_flat(u, nu, d, L):
    g = u.reshape(shape_of(d, L))
    return (adv_central(g, L) - nu * lap(g, L)).reshape(-1)


def lam_grid(d, L):
    """Eigenvalues of -Lap_h in the DST-I basis (refcore's arithmetic per dimension)."""
    if d == 2:
        k1 = jnp.arange(1, L, dtype=jnp.float64)
        l1 = 4 * L ** 2 * jnp.sin(jnp.pi * k1 / (2 * L)) ** 2
        return l1[:, None] + l1[None, :]
    l1 = jnp.asarray(C3.lam1(L + 1))
    return l1[:, None, None] + l1[None, :, None] + l1[None, None, :]


def dstn(g):
    return E2.dst2(g) if g.ndim == 2 else C3.dst3(g)


def lap_cols(Phi, d, L):
    """Lap_h applied to every column of Phi (n, r) -> (n, r)."""
    sh = shape_of(d, L)
    f = jax.vmap(lambda v: lap(v.reshape(sh), L).reshape(-1), in_axes=1, out_axes=1)
    return f(Phi)


def dsum_cols(Phi, d, L):
    sh = shape_of(d, L)
    f = jax.vmap(lambda v: dsum(v.reshape(sh), L).reshape(-1), in_axes=1, out_axes=1)
    return f(Phi)


# ------------------------------------------------------------------------------------------------ stencils --

def stencil_index(nodes, d, L):
    """(len(nodes), 2d+1) flat indices of [centre, -x, +x, -y, +y(, -z, +z)]; wall neighbours -> n (the zero row)."""
    m = L - 1
    n = m ** d
    nodes = np.asarray(nodes, np.int64)
    ijk = np.stack(np.unravel_index(nodes, (m,) * d), 1)
    cols = [nodes]
    for a in range(d):
        for s in (-1, +1):
            q = ijk.copy()
            q[:, a] += s
            ok = (q[:, a] >= 0) & (q[:, a] < m)
            q = np.clip(q, 0, m - 1)
            f = np.ravel_multi_index(tuple(q.T), (m,) * d)
            cols.append(np.where(ok, f, n))
    return np.stack(cols, 1)


def stencil_rows(Phi, idx):
    """Phi (n, r), idx (k, 2d+1) -> (k, 2d+1, r) with the zero row for walls."""
    Pp = jnp.concatenate([jnp.asarray(Phi), jnp.zeros((1, Phi.shape[1]), Phi.dtype)], 0)
    return Pp[jnp.asarray(idx)]


def stencil_F(us, nu, L, d):
    """Node-wise F_h from stencil values us (k, 2d+1): N - nu Lap, refcore's operand order where it matters."""
    c = us[:, 0]
    s = sum(us[:, 2 + 2 * a] - us[:, 1 + 2 * a] for a in range(d)) * (L / 2.0)
    nb = sum(us[:, 1 + 2 * a] + us[:, 2 + 2 * a] for a in range(d))
    lp = (nb - 2 * d * c) * L ** 2
    return c * s - nu * lp, c * s


def stencil_dF(us, nu, L, d):
    """d F_i / d (stencil values) (k, 2d+1) and d N_i / d (stencil values)."""
    c = us[:, 0]
    s = sum(us[:, 2 + 2 * a] - us[:, 1 + 2 * a] for a in range(d)) * (L / 2.0)
    k = us.shape[0]
    dN = jnp.zeros((k, 2 * d + 1), us.dtype).at[:, 0].set(s)
    for a in range(d):
        dN = dN.at[:, 1 + 2 * a].set(-c * (L / 2.0)).at[:, 2 + 2 * a].set(c * (L / 2.0))
    dL = jnp.full((k, 2 * d + 1), L ** 2, us.dtype).at[:, 0].set(-2.0 * d * L ** 2)
    return dN - nu * dL, dN


# -------------------------------------------------------------------------------------------- the FOM ------

def sched(scheme, dt, horizon=HORIZON, spacing=SPACING, save_steps=None, nmax=None):
    """Traced schedule. save_steps: step indices to keep (default: the six output times). The slot table has the fixed
    length nmax + 1 (default: this schedule's step count), so options with fewer steps reuse one compilation."""
    a0, a1, a2, b0, b1, ns = SCHEMES[scheme]
    steps, keep = int(round(horizon / dt)), int(round(spacing / dt))
    assert abs(steps * dt - horizon) < 1e-12 and abs(keep * dt - spacing) < 1e-12, (scheme, dt)
    nmax = steps if nmax is None else nmax
    assert steps <= nmax
    if save_steps is None:
        save_steps = tuple(range(0, steps + 1, keep))
    slot = np.full(nmax + 1, -1, np.int32)
    for i, k in enumerate(save_steps):
        assert 0 <= k <= steps
        slot[k] = i
    return dict(dt=jnp.float64(dt), steps=jnp.int32(steps), co=jnp.asarray([a0, a1, a2, b0, b1], jnp.float64),
                nstart=jnp.int32(ns), slot=jnp.asarray(slot), nsave=len(save_steps))


def make_fom(d, L, max_newton=20, lin_maxiter=None, adv='central'):
    """Central-difference FOM with the LMM residual above. Newton from the previous state, BiCGStab on the JVP,
    preconditioned by the exact DST inverse of (a0 + dt b0 nu Lambda); stop when ||R|| <= ntol ||u_n|| (refcore's rule).
    run(u0, nu, sch, ntol, ltol) -> dict(out (nsave, n), res (steps,), nwt (steps,))."""
    sh = shape_of(d, L)
    lam = lam_grid(d, L)
    lin_maxiter = lin_maxiter or {2: 200, 3: 2000}[d]
    be = jnp.asarray(BE_CO, jnp.float64)

    advf = adv_central if adv == 'central' else adv_upwind

    def F(u, nu):
        g = u.reshape(sh)
        return (advf(g, L) - nu * lap(g, L)).reshape(-1)

    def run(u0, nu, sch, ntol, ltol, lam, nsave):
        dt = sch['dt']
        nmax = sch['slot'].shape[0] - 1

        def body(c):
            k, un, um, out, res_, nwt = c
            co = jnp.where(k < sch['nstart'], be, sch['co'])
            a0, a1, a2, b0, b1 = co[0], co[1], co[2], co[3], co[4]
            Fn = F(un, nu)
            const = a1 * un + a2 * um + dt * b1 * Fn
            res = lambda u: a0 * u + const + dt * b0 * F(u, nu)
            den = a0 + dt * b0 * nu * lam

            def pre(v):
                return dstn(dstn(v.reshape(sh)) / den).reshape(-1)
            scale = jnp.maximum(jnp.linalg.norm(un), 1e-300)

            def nb(s):
                u, it, _ = s
                r = res(u)
                jv = lambda v: jax.jvp(res, (u,), (v,))[1]
                du, _ = jax.scipy.sparse.linalg.bicgstab(jv, -r, tol=ltol, maxiter=lin_maxiter, M=pre)
                u = u + du
                return u, it + 1, jnp.linalg.norm(res(u))
            u, it, rn = jax.lax.while_loop(lambda s: (s[2] > ntol * scale) & (s[1] < max_newton) & jnp.isfinite(s[2]),
                                           nb, (un, jnp.int32(0), jnp.linalg.norm(res(un))))
            kk = k + 1
            sl = sch['slot'][jnp.minimum(kk, nmax)]
            out = jax.lax.cond(sl >= 0, lambda o: o.at[sl].set(u), lambda o: o, out)
            res_ = res_.at[k].set(rn / scale)
            nwt = nwt.at[k].set(it)
            return kk, u, un, out, res_, nwt

        out0 = jnp.zeros((nsave,) + u0.shape, jnp.float64)
        out0 = jax.lax.cond(sch['slot'][0] >= 0, lambda o: o.at[sch['slot'][0]].set(u0), lambda o: o, out0)
        c0 = (jnp.int32(0), u0, u0, out0, jnp.full(nmax, jnp.nan), jnp.full(nmax, -1, jnp.int32))
        _, _, _, out, res_, nwt = jax.lax.while_loop(lambda c: c[0] < sch['steps'], body, c0)
        return dict(out=out, res=res_, nwt=nwt)

    jr = jax.jit(run, static_argnums=(6,))

    def call(u0, nu, sch, ntol=1e-10, ltol=1e-12):
        s = {k: v for k, v in sch.items() if k != 'nsave'}
        return jr(jnp.asarray(u0, jnp.float64), jnp.float64(nu), s, jnp.float64(ntol), jnp.float64(ltol), lam, sch['nsave'])
    return call


# ---------------------------------------------------------------------------------------------- lattices ---

LATTICE = {2: 256, 3: 64}


def lattice_nodes(d, L):
    """Flat interior indices of the scoring-lattice interior nodes on mesh L (x = k/lattice)."""
    m = LATTICE[d]
    s = L // m
    assert s * m == L, (L, m)
    ax = np.arange(1, m) * s - 1
    grids = np.meshgrid(*([ax] * d), indexing='ij')
    return np.ravel_multi_index(tuple(g.ravel() for g in grids), (L - 1,) * d)


def to_lattice(vals, d):
    """vals (..., (lattice-1)^d) at lattice interior nodes -> 2D (..., 257, 257) with zero walls; 3D unchanged."""
    m = LATTICE[d]
    if d == 3:
        return vals
    v = vals.reshape(vals.shape[:-1] + (m - 1, m - 1))
    pad = [(0, 0)] * (v.ndim - 2) + [(1, 1), (1, 1)]
    return np.pad(np.asarray(v), pad) if isinstance(v, np.ndarray) else jnp.pad(v, pad)
