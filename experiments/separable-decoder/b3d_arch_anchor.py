"""Protected graph head in exact orthonormal coordinates of the learned bank.

The fixed orthonormal anchor U is outside the optimizer. The nonlinear
correction lies in its orthogonal complement, so U.T @ (q(z)-bias) == z and
J.T @ J >= I. This is a graph restriction over a particular trained-skip
anchor, not a claim of adequate tangent coverage, conditioning or dynamics.
"""
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from b3d_arch_baseline import init_mlp, mlp

NAME = 'protected_anchor'
MODE = 'codes'
CONFIG = dict(width=128, layers=2, anchor='frozen_source_skip',
              residual_projection='orthogonal', trainable_bias=True)


def init(key, shared):
    u = np.asarray(shared['anchor'], dtype=np.float64)
    if u.ndim != 2 or not (0 < u.shape[1] <= u.shape[0]):
        raise ValueError('anchor must be an R by K matrix with 0 < K <= R')
    r, k = u.shape
    center = np.asarray(shared['center'], dtype=np.float64)
    scale = np.asarray(shared['latent_scale'], dtype=np.float64)
    output_scale = np.asarray(shared['output_scale'], dtype=np.float64)
    if center.shape != (r,) or scale.shape != (k,) or output_scale.shape != ():
        raise ValueError('center, latent_scale and output_scale have incompatible shapes')
    if not all(np.all(np.isfinite(a)) for a in (u, center, scale, output_scale)):
        raise ValueError('shared arrays must be finite')
    if np.any(scale <= 0) or output_scale <= 0:
        raise ValueError('normalization scales must be positive')
    if np.linalg.norm(u.T @ u - np.eye(k), ord=2) > 1e-12:
        raise ValueError('anchor columns must be orthonormal')
    p = dict(bias=jnp.asarray(center),
             residual=init_mlp(key, [k, 128, 128, r], zero_last=True))
    frozen = dict(anchor=jnp.asarray(u), center=jnp.asarray(center),
                  latent_scale=jnp.asarray(scale), output_scale=jnp.asarray(output_scale))
    return p, frozen


def apply(p, frozen, z):
    u = frozen['anchor']
    nonlinear = frozen['output_scale'] * mlp(p['residual'], z / frozen['latent_scale'])
    perpendicular = nonlinear - (nonlinear @ u) @ u.T
    return p['bias'] + z @ u.T + perpendicular


def apply_np(p, frozen, z):
    """Independent NumPy network plus an explicitly formed dense projector."""
    z = np.asarray(z, dtype=np.float64)
    x = z / np.asarray(frozen['latent_scale'])
    for w, bias in p['residual'][:-1]:
        a = x @ np.asarray(w) + np.asarray(bias)
        x = a * (.5 + .5 * np.tanh(.5 * a))
    w, bias = p['residual'][-1]
    nonlinear = float(frozen['output_scale']) * (x @ np.asarray(w) + np.asarray(bias))
    u = np.asarray(frozen['anchor'])
    projector = np.eye(u.shape[0]) - u @ u.T
    return np.asarray(p['bias']) + z @ u.T + nonlinear @ projector


def diagnostics(p, frozen, z):
    """Finite sampled certificates, without interpreting them as accuracy gates."""
    z = jnp.asarray(z, dtype=jnp.float64)
    if z.ndim == 1:
        z = z[None, :]
    q = np.asarray(apply(p, frozen, z))
    u = np.asarray(frozen['anchor'])
    jac = np.asarray(jax.jit(jax.vmap(jax.jacfwd(apply, argnums=2),
                                     in_axes=(None, None, 0)))(p, frozen, z))
    k = u.shape[1]
    singular = np.linalg.svd(jac, compute_uv=False)
    gram = jac.swapaxes(1, 2) @ jac
    anchor_jac = np.einsum('rk,nrj->nkj', u, jac)
    return dict(
        sampled_states=len(q),
        anchor_orthogonality_error=float(np.linalg.norm(u.T @ u - np.eye(k))),
        coordinate_recovery_max_absolute=float(np.max(np.abs(
            (q - np.asarray(p['bias'])) @ u - np.asarray(z)))),
        anchor_jacobian_identity_max_absolute=float(np.max(np.abs(anchor_jac-np.eye(k)))),
        minimum_gram_eigenvalue=float(np.min(np.linalg.eigvalsh(gram))),
        minimum_jacobian_singular_value=float(np.min(singular)),
        maximum_jacobian_condition=float(np.max(singular[:, 0] / singular[:, -1])),
        guarantee_scope='Fixed-anchor graph; rank protection does not certify physical tangent coverage or dynamics')
