"""Helpers shared by the bank job and the mesh jobs: spectral resampling, parameter
files, the per-mesh bank, and the derivative check."""
from __future__ import annotations

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import ns3d_fom as F
import shift_rom as SR
import coordnet as CN


def resample(field, n_to):
    """Spectral resampling of a periodic (3, n, n, n) field to (3, n_to, n_to, n_to).

    Copies the Fourier coefficients with |k_i| <= min(n, n_to)/2 - 1 (Nyquist planes are
    dropped on both sides). Returns the field and the fraction of its energy dropped."""
    field = np.asarray(field)
    n = field.shape[-1]
    if n == n_to:
        return field.copy(), 0.0
    spec = np.fft.fftn(field, axes=(1, 2, 3), norm="forward")
    m = min(n, n_to) // 2
    src = np.r_[0:m, n - m + 1:n]
    dst = np.r_[0:m, n_to - m + 1:n_to]
    kept = spec[:, src][:, :, src][:, :, :, src]
    out = np.zeros((3, n_to, n_to, n_to), dtype=complex)
    out[np.ix_(range(3), dst, dst, dst)] = kept
    total = float(np.sum(np.abs(spec) ** 2))
    lost = 1.0 - float(np.sum(np.abs(kept) ** 2)) / max(total, 1e-300)
    return np.fft.ifftn(out, axes=(1, 2, 3), norm="forward").real, max(lost, 0.0)


def save_params(path, p):
    flat = {"B": np.asarray(p["B"])}
    for i, (w, b) in enumerate(p["layers"]):
        flat[f"W{i}"] = np.asarray(w)
        flat[f"b{i}"] = np.asarray(b)
    np.savez(path, **flat)


def load_params(path):
    z = np.load(path)
    layers = []
    i = 0
    while f"W{i}" in z:
        layers.append((jnp.asarray(z[f"W{i}"]), jnp.asarray(z[f"b{i}"])))
        i += 1
    return dict(B=jnp.asarray(z["B"]), layers=layers)


def lowdin(G):
    """Closest orthonormal matrix with the same column order: G (G^T G)^{-1/2}.
    Returns the orthonormal matrix and the R x R factor S with G_orth = G S."""
    M = G.T @ G
    vals, vecs = np.linalg.eigh(M)
    S = vecs @ np.diag(vals ** -0.5) @ vecs.T
    return G @ S, S


def mesh_bank(p, T, n, chunk=32768):
    """The bank at mesh n: sample g at the n^3 nodes (scaled n^{-3/2}), project on the
    FOM's discrete space (2/3 mask + Leray), rotate by the frozen T, and remove the
    remaining non-orthonormality symmetrically (Lowdin). Every step is reported."""
    chunk = min(int(chunk), n ** 3)
    while (n ** 3) % chunk:
        chunk //= 2
    raw = np.asarray(CN.make_sampler(n, chunk=chunk)(p))
    proj, removed = CN.leray_mask(raw, n)
    Gr = proj @ np.asarray(T)
    dev = float(np.max(np.abs(Gr.T @ Gr - np.eye(Gr.shape[1]))))
    G, S = lowdin(Gr)
    change = float(np.linalg.norm(G - Gr) / np.linalg.norm(Gr))
    info = dict(n=int(n), leray_removed_worst=float(np.max(removed)),
                leray_removed_median=float(np.median(removed)),
                orthonormality_before_lowdin=dev, lowdin_relative_change=change,
                orthonormality_after=float(np.max(np.abs(G.T @ G - np.eye(G.shape[1])))))
    info["lowdin_factor"] = S
    return np.ascontiguousarray(G), info


def autodiff_gradient(p, n, chunk=4096):
    """d_e g at the grid nodes, scaled like the bank: (3_e, 3 n^3, R)."""
    pts = jnp.asarray(CN.grid_points(n))
    total = n ** 3
    chunk = min(chunk, total)

    @jax.jit
    def run(p, pts):
        vals = jax.lax.map(lambda b: CN.velocity_gradient(p, b), pts.reshape(total // chunk, chunk, 3))
        # vals (nchunks, 3e, chunk, 3, R)
        vals = jnp.moveaxis(vals, 1, 0).reshape(3, total, 3, -1)
        return jnp.transpose(vals, (0, 2, 1, 3)).reshape(3, 3 * total, -1) * n ** -1.5
    return np.asarray(run(p, pts))


def derivative_check(p, T, n):
    """Autodiff derivatives of the network vs spectral derivatives of the sampled
    columns, both rotated by T. Agreement is limited only by aliasing of the sampled
    (unprojected) columns, so it also measures how band-limited the bank is."""
    geom = F.geometry(n)
    raw = np.asarray(CN.make_sampler(n, chunk=min(32768, n ** 3))(p)) @ np.asarray(T)
    R = raw.shape[1]
    spec = np.asarray(SR.spectral_derivative(jnp.asarray(raw.T.reshape(R, 3, n, n, n)), n, geom))
    spec = spec.reshape(3, R, -1).transpose(0, 2, 1)                    # (3, dof, R)
    auto = np.einsum("edr,rs->eds", autodiff_gradient(p, n), np.asarray(T))
    rel = [float(np.linalg.norm(auto[e] - spec[e]) / np.linalg.norm(auto[e])) for e in range(3)]
    col = np.linalg.norm(auto - spec, axis=(0, 1)) / np.linalg.norm(auto, axis=(0, 1))
    return dict(n=int(n), relative_per_direction=rel, worst_column=float(col.max()))


def autodiff_tested(p, n, Phi, chunk=4096):
    """Phi^T (d_e g) from autodiff, accumulated chunk by chunk (no grid-sized gradient
    is ever stored). Phi (3 n^3, m) dense tests. Returns (3_e, m, R_raw)."""
    total = n ** 3
    chunk = min(chunk, total)
    while total % chunk:
        chunk //= 2
    pts = jnp.asarray(CN.grid_points(n)).reshape(total // chunk, chunk, 3)
    phi = jnp.asarray(np.asarray(Phi).reshape(3, total, -1)).reshape(3, total // chunk, chunk, -1)
    phi = jnp.moveaxis(phi, 1, 0)                                       # (nchunk, 3c, chunk, m)

    @jax.jit
    def run(p, pts, phi):
        def body(acc, xs):
            block, ph = xs
            g = CN.velocity_gradient(p, block)                          # (3e, chunk, 3c, R)
            return acc + jnp.einsum("epcr,cpm->emr", g, ph), None
        R = p["layers"][-1][0].shape[1] // 3
        acc0 = jnp.zeros((3, phi.shape[-1], R), dtype=phi.dtype)
        acc, _ = jax.lax.scan(body, acc0, (pts, phi))
        return acc * n ** -1.5
    return np.asarray(run(p, pts, phi))
