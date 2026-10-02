"""Uniform grid on (0,1)^2 with homogeneous Dirichlet data and sine weak tests.

Interior nodes x_i = i/N, i = 1..N-1 (n = (N-1)^2 unknowns). Arrays are indexed
u[i-1, j-1] with i the x index and j the y index.

Mesh-orthonormal sine vectors  psi_ab(i,j) = (2/N) sin(a pi i/N) sin(b pi j/N)
are eigenvectors of the five-point negative Laplacian A with eigenvalues
lambda_ab = (4/h^2)(sin^2(a pi h/2) + sin^2(b pi h/2)).

Continuum tests psi_ab(x,y) = 2 sin(a pi x) sin(b pi y) (L2-orthonormal on the
unit square).  The mesh-tested quantity of a grid function F is
    (P F)_ab = sum_ij psi_ab(i,j) F_ij = N * h^2 sum_ij psi_ab(x_ij) F_ij
             ~ N * int psi_ab F dx,
so an off-mesh quadrature rule with weights summing to one reproduces the
mesh-tested term as  N * sum_q w_q psi_ab(x_q) F(x_q).
"""
import numpy as np
import jax
import jax.numpy as jnp


def interior_coords(N):
    """(N-1)^2 x 2 array of interior node coordinates, row-major in (i, j)."""
    s = np.arange(1, N) / N
    X, Y = np.meshgrid(s, s, indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], axis=1)


def mode_list(M, N=None):
    """First M tensor sine modes (a, b), a,b >= 1, ordered by a^2+b^2 (tie: a)."""
    amax = int(np.ceil(np.sqrt(2 * M))) + 2
    if N is not None:
        amax = min(amax, N - 1)
    pairs = [(a, b) for a in range(1, amax + 1) for b in range(1, amax + 1)]
    pairs.sort(key=lambda p: (p[0] ** 2 + p[1] ** 2, p[0]))
    return np.array(pairs[:M], dtype=np.int64)


def eigenvalues(modes, N):
    h = 1.0 / N
    a, b = modes[:, 0], modes[:, 1]
    return (4.0 / h**2) * (np.sin(a * np.pi * h / 2) ** 2 + np.sin(b * np.pi * h / 2) ** 2)


def continuum_eigenvalues(modes):
    a, b = modes[:, 0], modes[:, 1]
    return np.pi**2 * (a**2 + b**2).astype(np.float64)


def dst1(u, axis):
    """DST-I along one axis: X_k = sum_{j=1}^{N-1} u_j sin(pi j k / N), k=1..N-1."""
    n = u.shape[axis]
    N = n + 1
    zeros_shape = list(u.shape)
    zeros_shape[axis] = 1
    z = jnp.zeros(zeros_shape, dtype=u.dtype)
    y = jnp.concatenate([z, u, z, -jnp.flip(u, axis=axis)], axis=axis)
    Y = jnp.fft.fft(y, axis=axis)
    X = -0.5 * jnp.imag(Y)
    return jax.lax.slice_in_dim(X, 1, N, axis=axis)


def dst1_2d(u):
    return dst1(dst1(u, 0), 1)


def idst1_2d(X):
    N = X.shape[0] + 1
    return (2.0 / N) ** 2 * dst1_2d(X)


def sine_matrix(N, amax):
    """(N-1) x amax matrix S[i-1, a-1] = sqrt(2/N) sin(a pi i / N): mesh-orthonormal sine vectors."""
    i = np.arange(1, N)[:, None]
    a = np.arange(1, amax + 1)[None, :]
    return jnp.asarray(np.sqrt(2.0 / N) * np.sin(a * np.pi * i / N))


def mesh_test(F, modes):
    """(P F)_ab for a grid function F ((N-1)x(N-1)) at the listed modes, by separable
    sine transforms restricted to the retained frequencies: O(n * amax)."""
    N = F.shape[0] + 1
    amax, bmax = int(modes[:, 0].max()), int(modes[:, 1].max())
    Sa, Sb = sine_matrix(N, amax), sine_matrix(N, bmax)
    X = Sa.T @ F @ Sb
    return X[modes[:, 0] - 1, modes[:, 1] - 1]


def mesh_test_batch(Fs, modes):
    """Fs: (B, N-1, N-1) -> (B, M)."""
    return jax.vmap(lambda F: mesh_test(F, modes))(Fs)


def continuum_tests(X, modes):
    """psi_ab(x_q) for points X (m x 2): returns (m, M), plus x and y derivatives."""
    a = modes[:, 0][None, :].astype(np.float64)
    b = modes[:, 1][None, :].astype(np.float64)
    x = X[:, 0:1]
    y = X[:, 1:2]
    sx, cx = np.sin(a * np.pi * x), np.cos(a * np.pi * x)
    sy, cy = np.sin(b * np.pi * y), np.cos(b * np.pi * y)
    psi = 2.0 * sx * sy
    psix = 2.0 * np.pi * a * cx * sy
    psiy = 2.0 * np.pi * b * sx * cy
    return psi, psix, psiy


def helmholtz_solve(r, alpha):
    """Solve (I + alpha A) v = r on the interior grid via DST-I."""
    N = r.shape[0] + 1
    h = 1.0 / N
    k = jnp.arange(1, N)
    lam1 = (4.0 / h**2) * jnp.sin(k * jnp.pi * h / 2) ** 2
    lam = lam1[:, None] + lam1[None, :]
    R = dst1_2d(r)
    return idst1_2d(R / (1.0 + alpha * lam))


def laplacian(u, N):
    up = jnp.pad(u, 1)
    return (up[2:, 1:-1] + up[:-2, 1:-1] + up[1:-1, 2:] + up[1:-1, :-2] - 4.0 * u) * N**2


def neg_laplacian(u, N):
    return -laplacian(u, N)
