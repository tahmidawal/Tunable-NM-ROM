"""Three-dimensional extension: grid, Burgers 3D, FOM, quadrature rules, evaluators.

Conventions mirror grid.py / rom.py.  Interior nodes x_i = i/N, i = 1..N-1 on each axis,
n = (N-1)^3 unknowns, arrays indexed u[i-1, j-1, k-1].  The mesh-orthonormal sine vector is
(2/N)^{3/2} sin sin sin = h^{3/2} psi_abc(x) with psi_abc = 2^{3/2} sin(a pi x) sin(b pi y)
sin(c pi z) (L2-orthonormal on the unit cube), so the mesh-tested quantity of a grid function F
is  (P F)_abc = h^{3/2} sum psi F = N^{3/2} * h^3 sum psi F ~ N^{3/2} int psi F, and an
off-mesh rule with weights summing to one reproduces it as N^{3/2} sum_q w_q psi(x_q) F(x_q).
The sine transforms and the Helmholtz solve use separable matmuls with the full (N-1)x(N-1)
sine matrix (cheaper than FFT-based DSTs on the GPU for N <= 128).
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import bicgstab
from scipy.stats import qmc
from .grid import sine_matrix
from .bank import bank_eval_chunked, bank_eval
from .pdes import gaussian_bump
from .quadrature import gauss_legendre_1d, clenshaw_curtis_1d


# ------------------------------------------------------------------- grid
def interior_coords3(N):
    s = np.arange(1, N) / N
    X, Y, Z = np.meshgrid(s, s, s, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


def mode_list3(M, N=None):
    amax = int(np.ceil((6 * M) ** (1 / 3))) + 3
    if N is not None:
        amax = min(amax, N - 1)
    r = range(1, amax + 1)
    pairs = [(a, b, c) for a in r for b in r for c in r]
    pairs.sort(key=lambda p: (p[0] ** 2 + p[1] ** 2 + p[2] ** 2, p[0], p[1]))
    return np.array(pairs[:M], dtype=np.int64)


def eigenvalues3(modes, N):
    h = 1.0 / N
    return (4.0 / h**2) * sum(np.sin(modes[:, j] * np.pi * h / 2) ** 2 for j in range(3))


def continuum_tests3(X, modes):
    """psi_abc(x_q) (m, M) and its three gradients."""
    a = modes[:, 0][None, :].astype(np.float64)
    b = modes[:, 1][None, :].astype(np.float64)
    c = modes[:, 2][None, :].astype(np.float64)
    x, y, z = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    sx, cx = np.sin(a * np.pi * x), np.cos(a * np.pi * x)
    sy, cy = np.sin(b * np.pi * y), np.cos(b * np.pi * y)
    sz, cz = np.sin(c * np.pi * z), np.cos(c * np.pi * z)
    s = 2.0 ** 1.5
    return (s * sx * sy * sz, s * np.pi * a * cx * sy * sz, s * np.pi * b * sx * cy * sz, s * np.pi * c * sx * sy * cz)


def sine_transform3(F, amax, bmax, cmax):
    """X_abc = sum_ijk s_a(i) s_b(j) s_c(k) F_ijk with mesh-orthonormal sine vectors."""
    N = F.shape[0] + 1
    Sa, Sb, Sc = sine_matrix(N, amax), sine_matrix(N, bmax), sine_matrix(N, cmax)
    T = jnp.tensordot(Sa, F, axes=([0], [0]))          # (amax, n1, n1)
    T = jnp.tensordot(T, Sb, axes=([1], [0]))          # (amax, n1, cmax?) -> (amax, n1, bmax)
    T = jnp.tensordot(T, Sc, axes=([1], [0]))          # (amax, bmax, cmax)
    return T


def mesh_test3(F, modes):
    amax, bmax, cmax = (int(modes[:, j].max()) for j in range(3))
    T = sine_transform3(F, amax, bmax, cmax)
    return T[modes[:, 0] - 1, modes[:, 1] - 1, modes[:, 2] - 1]


def dst3_full(F):
    """Full mesh-orthonormal 3D sine transform (its own inverse)."""
    n1 = F.shape[0]
    return sine_transform3(F, n1, n1, n1)


def helmholtz_solve3(r, alpha, beta=0.0):
    """Solve ((1+beta) I + alpha A) v = r on the interior grid, A the 7-point negative Laplacian."""
    N = r.shape[0] + 1
    h = 1.0 / N
    k = jnp.arange(1, N)
    lam1 = (4.0 / h**2) * jnp.sin(k * jnp.pi * h / 2) ** 2
    lam = lam1[:, None, None] + lam1[None, :, None] + lam1[None, None, :]
    return dst3_full(dst3_full(r) / (1.0 + beta + alpha * lam))


def laplacian3(u, N):
    up = jnp.pad(u, 1)
    return (up[2:, 1:-1, 1:-1] + up[:-2, 1:-1, 1:-1] + up[1:-1, 2:, 1:-1] + up[1:-1, :-2, 1:-1]
            + up[1:-1, 1:-1, 2:] + up[1:-1, 1:-1, :-2] - 6.0 * u) * N**2


# -------------------------------------------------------------- Burgers 3D
def burgers3_stencil(uc, uxm, uxp, uym, uyp, uzm, uzp, N, p):
    pos = uc > 0
    dx = jnp.where(pos, (uc - uxm) * N, (uxp - uc) * N)
    dy = jnp.where(pos, (uc - uym) * N, (uyp - uc) * N)
    dz = jnp.where(pos, (uc - uzm) * N, (uzp - uc) * N)
    return uc * (dx + dy + dz)


def stencil_apply3(u, N, fn, p):
    up = jnp.pad(u, 1)
    return fn(u, up[:-2, 1:-1, 1:-1], up[2:, 1:-1, 1:-1], up[1:-1, :-2, 1:-1], up[1:-1, 2:, 1:-1],
              up[1:-1, 1:-1, :-2], up[1:-1, 1:-1, 2:], N, p)


class Burgers3D:
    name = "burgers3d"
    steady = False
    dt = 0.005
    T = 0.25
    out_times = np.array([0.05, 0.10, 0.15, 0.20, 0.25])
    needs_grad = True
    description = "u_t + u(u_x+u_y+u_z) = nu Lap u on (0,1)^3, sign-upwind, backward Euler"

    @staticmethod
    def sample_params(rng):
        return dict(nu=float(np.exp(rng.uniform(np.log(0.01), np.log(0.1)))),
                    a=float(rng.uniform(0.5, 2.0)), c=rng.uniform(0.15, 0.85, size=3),
                    w=float(rng.uniform(0.05, 0.20)))

    @staticmethod
    def u0(X, p):
        r2 = ((X - p["c"][None, :]) ** 2).sum(1)
        return p["a"] * np.exp(-r2 / (2 * p["w"] ** 2))

    @staticmethod
    def source(X, p):
        return np.zeros(X.shape[0])

    @staticmethod
    def nu(p):
        return p["nu"]

    @staticmethod
    def react(p):
        return 0.0

    nonlinear_stencil = staticmethod(burgers3_stencil)

    @staticmethod
    def nonlinear_point(u, ux, uy, uz, p):
        return u * (ux + uy + uz)

    @staticmethod
    def nonlinear_flux(u, p):
        return (0.5 * u * u, 0.5 * u * u, 0.5 * u * u)

    def nonlinear_grid(self, u, N, p):
        return stencil_apply3(u, N, self.nonlinear_stencil, p)


BURGERS3D = Burgers3D()


# --------------------------------------------------------------------- FOM
def make_fom3(pde, N):
    """Newton with right-preconditioned BiCGStab (Python loop over jitted matvec / Helmholtz
    solve; JAX's fused bicgstab with the tensordot preconditioner crashes on ROCm) and a
    residual backtracking safeguard."""
    def residual(u, u_n, dt, nu, r, f, p):
        return u - u_n + dt * (pde.nonlinear_grid(u, N, p) - nu * laplacian3(u, N) - r * u - f)

    residual_j = jax.jit(residual)
    jvp_j = jax.jit(lambda u, v, u_n, dt, nu, r, f, p: jax.jvp(lambda uu: residual(uu, u_n, dt, nu, r, f, p), (u,), (v,))[1])
    helm_j = jax.jit(lambda v, alpha, beta: helmholtz_solve3(v, alpha, beta))
    vdot = jax.jit(lambda a, b: jnp.vdot(a, b))
    nrm = jax.jit(lambda a: jnp.linalg.norm(a))

    def bicgstab_py(Jv, Minv, b, tol, maxiter=200):
        x = jnp.zeros_like(b)
        r = b
        rhat = r
        rho = alpha = omega = 1.0
        v = pp = jnp.zeros_like(b)
        nb = float(nrm(b))
        for it in range(maxiter):
            rho_new = float(vdot(rhat, r))
            beta = (rho_new / rho) * (alpha / omega)
            rho = rho_new
            pp = r + beta * (pp - omega * v)
            y = Minv(pp)
            v = Jv(y)
            alpha = rho / float(vdot(rhat, v))
            s = r - alpha * v
            z = Minv(s)
            t_ = Jv(z)
            omega = float(vdot(t_, s)) / float(vdot(t_, t_))
            x = x + alpha * y + omega * z
            r = s - omega * t_
            if float(nrm(r)) <= tol * nb:
                break
        return x, it + 1

    def step(u_n, dt, nu, r, f, p, tol=1e-9, maxit=12, tol_lin=1e-4, u_guess=None):
        u = u_n if u_guess is None else u_guess
        scale = max(float(nrm(u_n)), 1e-12)
        nits = 0
        for it in range(maxit):
            R = residual_j(u, u_n, dt, nu, r, f, p)
            nR = float(nrm(R))
            nits += 1
            if nR <= tol * scale:
                return u, nits, nR / scale
            Jv = lambda v: jvp_j(u, v, u_n, dt, nu, r, f, p)
            Minv = lambda v: helm_j(v, dt * nu, -dt * r)
            d, _ = bicgstab_py(Jv, Minv, -R, tol_lin)
            a = 1.0
            nRt = float(nrm(residual_j(u + d, u_n, dt, nu, r, f, p)))
            while nRt > nR and a > 1e-3:
                a *= 0.5
                nRt = float(nrm(residual_j(u + a * d, u_n, dt, nu, r, f, p)))
            u = u + a * d
        return u, nits, nR / scale
    return step, residual_j


def solve_case3(pde, N, p, dt=None, tol=1e-9, verbose=False, store_every=1):
    X = interior_coords3(N)
    f = jnp.asarray(pde.source(X, p).reshape(N - 1, N - 1, N - 1))
    nu, r = pde.nu(p), pde.react(p)
    step, _ = make_fom3(pde, N)
    t0 = time.time()
    dt = pde.dt if dt is None else dt
    nsteps = int(round(pde.T / dt))
    u = jnp.asarray(pde.u0(X, p).reshape(N - 1, N - 1, N - 1))
    u_prev = None
    states, t_states, u_out, t_out, its = [np.asarray(u, dtype=np.float32)], [0.0], [], [], []
    out_steps = {int(round(t / dt)): t for t in pde.out_times}
    for n in range(1, nsteps + 1):
        guess = u if u_prev is None else 2 * u - u_prev
        u_new, nits, res = step(u, dt, nu, r, f, p, tol=tol, u_guess=guess)
        its.append(nits)
        u_prev, u = u, u_new
        if n % store_every == 0:
            states.append(np.asarray(u, dtype=np.float32)); t_states.append(n * dt)
        if n in out_steps:
            u_out.append(np.asarray(u)); t_out.append(out_steps[n])
        if verbose and n % 10 == 0:
            print(f"  step {n}/{nsteps} newton its {nits} res {res:.2e} {time.time()-t0:.0f}s", flush=True)
    return dict(u_out=np.stack(u_out), t_out=np.array(t_out), states=np.stack(states),
                t_states=np.array(t_states), newton_its=its, time=time.time() - t0, p=p, N=N)


# ----------------------------------------------------------- quadrature 3D
def gauss_tensor3(p):
    x, w = gauss_legendre_1d(p)
    X = np.array([(a, b, c) for a in x for b in x for c in x])
    W = np.array([wa * wb * wc for wa in w for wb in w for wc in w])
    return X, W


def _cc_size(i):
    return 1 if i == 1 else 2 ** (i - 1) + 1


def smolyak3(level):
    """Smolyak sparse grid with nested Clenshaw--Curtis rules on [0,1]^3, boundary points dropped.
    A(q,3) = sum_{q-2 <= |i| <= q} (-1)^{q-|i|} C(2, q-|i|) Q_i1 x Q_i2 x Q_i3, q = level + 3."""
    from math import comb
    q = level + 3
    pts = {}
    rules = {}

    def r1(i):
        if i not in rules:
            rules[i] = clenshaw_curtis_1d(_cc_size(i))
        return rules[i]
    for i1 in range(1, q + 1):
        for i2 in range(1, q + 1):
            for i3 in range(1, q + 1):
                s = i1 + i2 + i3
                if s < q - 2 or s > q:
                    continue
                coef = (-1) ** (q - s) * comb(2, q - s)
                x1, w1 = r1(i1); x2, w2 = r1(i2); x3, w3 = r1(i3)
                for a in range(len(x1)):
                    for b in range(len(x2)):
                        for c in range(len(x3)):
                            key = (round(x1[a], 12), round(x2[b], 12), round(x3[c], 12))
                            pts[key] = pts.get(key, 0.0) + coef * w1[a] * w2[b] * w3[c]
    keys = [k for k, v in pts.items() if abs(v) > 1e-15]
    X = np.array(keys); w = np.array([pts[k] for k in keys])
    keep = np.all((X > 0) & (X < 1), axis=1)
    return X[keep], w[keep]


def sobol3(m, seed=0):
    X = qmc.Sobol(d=3, scramble=True, seed=seed).random_base2(int(np.log2(m)))
    return X, np.full(len(X), 1.0 / len(X))


def mc3(m, seed=0):
    return np.random.default_rng(seed).uniform(size=(m, 3)), np.full(m, 1.0 / m)


KUO_Z = np.array([1, 182667, 469891])   # first 3 components of lattice-32001-1024-1048576.3600 (Cools, Kuo, Nuyens)


def rank1_lattice3(n, z, shift=None, tent=False, seed=0):
    i = np.arange(n)
    X = (np.outer(i, z) / n) % 1.0
    if shift is None:
        shift = np.random.default_rng(seed).uniform(size=3)
    X = (X + shift) % 1.0
    if tent:
        X = 1.0 - np.abs(2.0 * X - 1.0)
    return X, np.full(n, 1.0 / n)


def kuo_lattice3(n, tent=False, seed=0):
    return rank1_lattice3(n, KUO_Z, tent=tent, seed=seed)


def korobov_search3(n, chunk=256):
    """Korobov generating vector z = (1, a, a^2 mod n) minimising the P_2 figure of merit
    (product weights 1): P_2(z) = -1 + (1/n) sum_k prod_j [1 + 2 pi^2 B_2({k z_j / n})]."""
    k = np.arange(n)
    best, best_a = np.inf, None
    b2 = lambda x: x * x - x + 1.0 / 6.0
    f1 = 1.0 + 2 * np.pi**2 * b2(k / n)
    for a0 in range(1, n, chunk):
        A = np.arange(a0, min(n, a0 + chunk))
        z2 = A % n
        z3 = (A * A) % n
        x2 = ((k[None, :] * z2[:, None]) % n) / n
        x3 = ((k[None, :] * z3[:, None]) % n) / n
        P = (f1[None, :] * (1.0 + 2 * np.pi**2 * b2(x2)) * (1.0 + 2 * np.pi**2 * b2(x3))).mean(1) - 1.0
        j = int(np.argmin(P))
        if P[j] < best:
            best, best_a = float(P[j]), int(A[j])
    return np.array([1, best_a, (best_a * best_a) % n]), best


def p2_merit3(z, n, gamma=(1.0, 1.0, 1.0)):
    """P_2 figure of merit of the rank-1 lattice (n, z) on [0,1]^3 with product weights."""
    k = np.arange(n)
    b2 = lambda x: x * x - x + 1.0 / 6.0
    prod = np.ones(n)
    for j in range(3):
        prod = prod * (1.0 + gamma[j] * 2 * np.pi**2 * b2(((k * int(z[j])) % n) / n))
    return float(prod.mean() - 1.0)


def cbc_lattice_vector3(n, gamma=(1.0, 1.0, 1.0), chunk=256):
    """Component-by-component construction of a 3D generating vector minimising P_2
    (Sloan--Kuo--Joe); candidates are the integers coprime to n."""
    k = np.arange(n)
    b2 = lambda x: x * x - x + 1.0 / 6.0
    cand = np.array([a for a in range(1, n) if np.gcd(a, n) == 1])
    z = [1]
    prod = 1.0 + gamma[0] * 2 * np.pi**2 * b2(k / n)
    for j in range(1, 3):
        best, best_a = np.inf, None
        for a0 in range(0, len(cand), chunk):
            A = cand[a0:a0 + chunk]
            x = ((k[None, :] * A[:, None]) % n) / n
            P = (prod[None, :] * (1.0 + gamma[j] * 2 * np.pi**2 * b2(x))).mean(1) - 1.0
            i = int(np.argmin(P))
            if P[i] < best:
                best, best_a = float(P[i]), int(A[i])
        z.append(best_a)
        prod = prod * (1.0 + gamma[j] * 2 * np.pi**2 * b2(((k * best_a) % n) / n))
    return np.array(z), best


_CBC_CACHE = {}


def cbc_lattice3(n, tent=False, seed=0):
    if n not in _CBC_CACHE:
        _CBC_CACHE[n] = cbc_lattice_vector3(n)[0]
    return rank1_lattice3(n, _CBC_CACHE[n], tent=tent, seed=seed)


def korobov_lattice3(n, seed=0):
    z, _ = korobov_search3(n)
    return rank1_lattice3(n, z, seed=seed)


# --------------------------------------------------------------- mesh model
class MeshModel3:
    """Same attributes/methods as rom.MeshModel, for the unit cube."""

    def __init__(self, ckpt, pde, N, M, need_G=True):
        self.ckpt, self.pde, self.N, self.M = ckpt, pde, N, M
        self.R, self.k = ckpt["R"], ckpt["k"]
        self.modes = mode_list3(M, N)
        self.lam = eigenvalues3(self.modes, N)
        self.X = interior_coords3(N)
        t0 = time.time()
        self.G = bank_eval_chunked(ckpt["bank"], self.X, chunk=131072)     # (n, R)
        n1 = N - 1
        G4 = self.G.reshape(n1, n1, n1, self.R)
        cols = []
        mt = jax.jit(lambda F: mesh_test3(F, self.modes))
        for i in range(self.R):
            cols.append(np.asarray(mt(jnp.asarray(G4[..., i]))))
        self.B0 = np.stack(cols, 0).T                                          # (M, R)
        Gj = jnp.asarray(self.G)
        self.gram = np.asarray(Gj.T @ Gj)
        self.offline_time = time.time() - t0

    def project_field(self, u):
        return np.linalg.solve(self.gram, np.asarray(jnp.asarray(self.G).T @ jnp.asarray(np.asarray(u).ravel())))

    def tested_field(self, u):
        n1 = self.N - 1
        return np.asarray(mesh_test3(jnp.asarray(np.asarray(u).reshape(n1, n1, n1)), self.modes))

    def reconstruct(self, c):
        n1 = self.N - 1
        return np.asarray(jnp.asarray(self.G) @ jnp.asarray(c)).reshape(n1, n1, n1)


def make_dense_evaluator3(mm):
    pde, N, modes = mm.pde, mm.N, mm.modes
    G = jnp.asarray(mm.G)
    n1 = N - 1

    def nl(c, p):
        u = (G @ c).reshape(n1, n1, n1)
        return mesh_test3(pde.nonlinear_grid(u, N, p), modes)
    return dict(nl=nl, m=n1**3, kind="dense", label="dense")


def make_mesh_evaluator3(mm, idx, w, label):
    pde, N, modes, R = mm.pde, mm.N, mm.modes, mm.R
    n1 = N - 1
    i = idx // (n1 * n1) + 1
    j = (idx // n1) % n1 + 1
    k = idx % n1 + 1

    def gather(ii, jj, kk):
        inside = (ii >= 1) & (ii <= n1) & (jj >= 1) & (jj <= n1) & (kk >= 1) & (kk <= n1)
        lin = np.clip(((ii - 1) * n1 + (jj - 1)) * n1 + (kk - 1), 0, n1**3 - 1)
        return mm.G[lin] * inside[:, None]
    block = np.stack([gather(i, j, k), gather(i - 1, j, k), gather(i + 1, j, k), gather(i, j - 1, k),
                      gather(i, j + 1, k), gather(i, j, k - 1), gather(i, j, k + 1)], 1)
    a, b, c = modes[:, 0], modes[:, 1], modes[:, 2]
    psi = (2.0 / N) ** 1.5 * (np.sin(a[None] * np.pi * i[:, None] / N) * np.sin(b[None] * np.pi * j[:, None] / N)
                              * np.sin(c[None] * np.pi * k[:, None] / N))
    Psi = jnp.asarray(w[:, None] * psi)
    blk = jnp.asarray(block)

    def nl(cc, p):
        U = blk @ cc
        F = pde.nonlinear_stencil(U[:, 0], U[:, 1], U[:, 2], U[:, 3], U[:, 4], U[:, 5], U[:, 6], N, p)
        return Psi.T @ F
    return dict(nl=nl, m=len(idx), kind="mesh", label=label, idx=idx, w=w)


def mesh_lattice3(N, stride):
    idx1 = np.arange(stride, N, stride)
    I, J, K = np.meshgrid(idx1, idx1, idx1, indexing="ij")
    n1 = N - 1
    lin = ((I.ravel() - 1) * n1 + (J.ravel() - 1)) * n1 + (K.ravel() - 1)
    return lin, np.full(lin.size, float(stride**3))


def bank_eval_grad_batched(params, X, chunk=16384):
    """Bank values and the three coordinate gradients at points X, by one batched JVP per
    direction (points are independent, so a constant tangent field gives per-point derivatives)."""
    X = np.asarray(X)
    f = jax.jit(lambda Xb: bank_eval(params, Xb))
    g = jax.jit(lambda Xb, E: jax.jvp(lambda XX: bank_eval(params, XX), (Xb,), (E,))[1])
    outs = [[] for _ in range(1 + X.shape[1])]
    for i in range(0, X.shape[0], chunk):
        Xb = jnp.asarray(X[i:i + chunk])
        outs[0].append(np.asarray(f(Xb)))
        for j in range(X.shape[1]):
            E = jnp.zeros_like(Xb).at[:, j].set(1.0)
            outs[j + 1].append(np.asarray(g(Xb, E)))
    return tuple(np.concatenate(o) for o in outs)


def make_offmesh_evaluator3(mm, Xq, wq, label, form="point"):
    pde, N, modes = mm.pde, mm.N, mm.modes
    Gq, Gx, Gy, Gz = bank_eval_grad_batched(mm.ckpt["bank"], Xq)
    psi, psix, psiy, psiz = continuum_tests3(Xq, modes)
    scale = N ** 1.5 * wq[:, None]
    Gq, Gx, Gy, Gz = (jnp.asarray(a) for a in (Gq, Gx, Gy, Gz))
    Psi, Psix, Psiy, Psiz = (jnp.asarray(scale * a) for a in (psi, psix, psiy, psiz))
    if form == "flux":
        def nl(c, p):
            u = Gq @ c
            F1, F2, F3 = pde.nonlinear_flux(u, p)
            return -(Psix.T @ F1 + Psiy.T @ F2 + Psiz.T @ F3)
    else:
        def nl(c, p):
            u, ux, uy, uz = Gq @ c, Gx @ c, Gy @ c, Gz @ c
            return Psi.T @ pde.nonlinear_point(u, ux, uy, uz, p)
    return dict(nl=nl, m=len(wq), kind="offmesh", label=label, X=Xq, w=wq, form=form)


def rollout_errors3(mm, roll, case):
    N = mm.N
    sg = case["same_grid"][N]["u_out"]
    ref = case["refined"]["u_out"]
    Nr = case["refined"]["N"]
    s = Nr // N
    ref_sub = ref[:, s - 1::s, s - 1::s, s - 1::s] if s > 1 else ref
    U = np.stack([mm.reconstruct(c) for c in roll["out_coeffs"]])
    norm = np.linalg.norm(mm.pde.u0(mm.X, case["p"]))
    e_sg = np.linalg.norm((U - sg).reshape(len(U), -1), axis=1) / norm
    e_ref = np.linalg.norm((U - ref_sub).reshape(len(U), -1), axis=1) / norm
    e_fom_ref = np.linalg.norm((sg - ref_sub).reshape(len(U), -1), axis=1) / norm
    return dict(err_same_grid=e_sg, err_refined=e_ref, fom_err_refined=e_fom_ref)
