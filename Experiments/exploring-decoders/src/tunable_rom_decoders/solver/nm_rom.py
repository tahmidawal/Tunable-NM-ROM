"""Decoder-agnostic Levenberg-Marquardt Gauss-Newton NM-ROM solver.

Mirrors poisson/src/tunable_rom_poisson/solver/nm_rom.py but treats the
decoder as a black box exposing `decode_points(z, tokens, x_query)` on
the INRAutoencoder. The Jacobian dR/dz is obtained via jax.jacfwd of
the residual; no V_eq precompute is possible because INR decoders are
nonlinear in z.

Token handling (xattn only):
  - Tokens are frozen at solve time to a single reference tensor
    (typically mean(encoder.tokens(u)) over the training set). For SIREN
    the tokens slot is ignored.

Residual (Poisson, 2D, 5-point):
  R[i] = ((2*d) * u_center - sum_neighbours) / dx^2 - F_eq[i]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class INRNMROMSolver:
    autoencoder: object              # Flax INRAutoencoder
    params: dict                     # trained params
    N: int
    spatial_dim: int
    dx: float
    eq_flat_indices: np.ndarray      # (n_eq,)
    eq_weights: np.ndarray           # (n_eq,)
    stencil_indices: np.ndarray      # (n_eq, 2d+1)
    tokens_ref: Optional[np.ndarray] = None  # (n_tok, embed_dim) for xattn; None for siren
    gn_max_iters: int = 12
    gn_rel_tol: float = 1e-3
    lm_damping: float = 1e-4
    encode_method_name: str = "encode"   # "encode" (default) or "encode_mu" (variational)
    latent_dim_override: Optional[int] = None

    def __post_init__(self):
        d = self.spatial_dim
        self.coeff = 2.0 * d
        self.n_eq = int(self.eq_flat_indices.shape[0])
        self.stencil_w = 2 * d + 1
        self.w_eq = jnp.asarray(self.eq_weights)
        # Continuous coords for every stencil node, flattened to (n_eq*stencil_w, d).
        flat = self.stencil_indices.reshape(-1)
        if d == 2:
            ix = flat // self.N
            iy = flat % self.N
            coords = np.stack([ix, iy], axis=-1).astype(np.float32) / (self.N - 1)
        else:
            raise NotImplementedError("3D not yet supported")
        self.stencil_coords = jnp.asarray(coords)
        self.latent_dim = (
            int(self.latent_dim_override)
            if self.latent_dim_override is not None
            else self._infer_latent_dim()
        )
        self._tokens_ref_j = (
            jnp.asarray(self.tokens_ref) if self.tokens_ref is not None else None
        )

    def _infer_latent_dim(self):
        # The encoder's `head` Dense layer maps the pooled token to z, so
        # head.kernel.shape[-1] == latent_dim. For the variational AE,
        # `head_mu` plays that role.
        enc = self.params.get("encoder", {})
        if "head" in enc and "kernel" in enc["head"]:
            return int(enc["head"]["kernel"].shape[-1])
        if "head_mu" in enc and "kernel" in enc["head_mu"]:
            return int(enc["head_mu"]["kernel"].shape[-1])
        return self._guess_latent_dim()

    def _guess_latent_dim(self):
        # Fallback: inspect SIREN W_in shape or xattn modulator output.
        dec = self.params["decoder"]
        if "mod_in" in dec:
            return int(dec["mod_in"]["kernel"].shape[0])
        # cross_attn_inr: latent appears in the first MLP-hidden layer's input width
        # = 2F + d_attn + latent. We need the latent dim, which we get from B_fourier:
        # B_fourier has shape (F, coord_dim); 2F = num_fourier*2. The rest is recovered
        # by checking mlp_0 input dim minus 2F minus d_attn.
        if "B_fourier" in dec and "mlp_0" in dec:
            F = int(dec["B_fourier"].shape[0])
            d_attn = int(dec["W_v"]["kernel"].shape[-1])
            mlp_in = int(dec["mlp_0"]["kernel"].shape[0])
            return mlp_in - 2 * F - d_attn
        raise RuntimeError("Could not infer latent dim from params.")

    def u_at_stencil(self, z):
        """Decoded values at all stencil nodes, reshaped to (n_eq, stencil_w)."""
        u_flat = self.autoencoder.apply(
            {"params": self.params},
            z, self._tokens_ref_j, self.stencil_coords,
            method=self.autoencoder.decode_points,
        )
        return u_flat.reshape(self.n_eq, self.stencil_w)

    def residual(self, z, F_eq):
        u_st = self.u_at_stencil(z)
        lap_neg = (self.coeff * u_st[:, 0] - jnp.sum(u_st[:, 1:], axis=1)) / self.dx**2
        return lap_neg - F_eq

    def _solve_impl(self, F_eq, z_init=None):
        # The body operations (jacfwd, linalg.solve, etc.) promote to
        # float64 under jax_enable_x64=True, so the initial carry must
        # match for while_loop.
        z0 = jnp.zeros((self.latent_dim,), dtype=jnp.float64) if z_init is None \
            else jnp.asarray(z_init, dtype=jnp.float64)

        def body(carry):
            z, gnorm0, gnorm, itr = carry
            R = self.residual(z, F_eq)
            J = jax.jacfwd(lambda zz: self.residual(zz, F_eq))(z)
            JtW = J.T * self.w_eq[None, :]
            H = JtW @ J
            g = JtW @ R
            damp = jnp.maximum(self.lm_damping * jnp.trace(H) / self.latent_dim, 1e-8)
            dz = jnp.linalg.solve(H + damp * jnp.eye(self.latent_dim), -g)
            steps = jnp.asarray([1.0, 0.5, 0.25, 0.125])

            def try_step(a):
                Rc = self.residual(z + a * dz, F_eq)
                return 0.5 * jnp.sum(self.w_eq * Rc**2)

            losses = jax.vmap(try_step)(steps)
            best = jnp.argmin(losses)
            z_new = z + steps[best] * dz
            return (z_new, gnorm0, jnp.linalg.norm(g), itr + 1)

        def cond(carry):
            _, gnorm0, gnorm, itr = carry
            return jnp.logical_and(gnorm > self.gn_rel_tol * gnorm0, itr < self.gn_max_iters)

        R0 = self.residual(z0, F_eq)
        J0 = jax.jacfwd(lambda zz: self.residual(zz, F_eq))(z0)
        g0 = (J0.T * self.w_eq[None, :]) @ R0
        gnorm0 = jnp.maximum(jnp.linalg.norm(g0), 1e-30)
        z_f, _, gnorm_f, iters = jax.lax.while_loop(
            cond, body, (z0, gnorm0, gnorm0, 0)
        )
        return z_f, gnorm_f, iters

    def make_solve(self):
        """Return a JIT-compiled solve(F_eq) -> (z, gnorm, iters), cold-start z=0."""
        return jax.jit(self._solve_impl)

    def make_solve_warm(self):
        """Return a JIT-compiled solve(F_eq, z_init) -> (z, gnorm, iters)."""
        def _solve(F_eq, z_init):
            return self._solve_impl(F_eq, z_init=z_init)
        return jax.jit(_solve)

    def encode_warm_start(self, u_warm: jnp.ndarray) -> jnp.ndarray:
        """Use the AE encoder to map a warm-start field to a latent z.

        For the deterministic AE, encode(u) returns (z, tokens) — we keep z.
        For the variational AE, we use the named `encode_mu` method which
        returns mu directly.
        """
        if self.encode_method_name == "encode_mu":
            mu = self.autoencoder.apply(
                {"params": self.params}, u_warm,
                method=getattr(self.autoencoder, "encode_mu"),
            )
            return mu
        z, _T = self.autoencoder.apply(
            {"params": self.params}, u_warm, method=self.autoencoder.encode,
        )
        return z

    def decode_full_grid(self, z):
        """Decode at all N**d nodes for full-field rel-L2 reporting."""
        d = self.spatial_dim
        rng = jnp.linspace(0.0, 1.0, self.N)
        if d == 2:
            xx, yy = jnp.meshgrid(rng, rng, indexing="ij")
            x_query = jnp.stack([xx, yy], axis=-1).reshape(-1, 2)
        else:
            raise NotImplementedError
        u = self.autoencoder.apply(
            {"params": self.params},
            z, self._tokens_ref_j, x_query,
            method=self.autoencoder.decode_points,
        )
        return u
