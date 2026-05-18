"""LinearAffineZDecoder: like AffineZDecoder but A(z) is LINEAR in z.

   u(x; z) = Phi(x) @ W_A @ z + Phi(x) @ b_A + b(x)

Or equivalently  u(x; z) = (Phi(x) @ W_A) @ z + (Phi(x) @ b_A + b(x))
              = V(x) @ z + b_total(x)         where V(x) = Phi(x) @ W_A

The Jacobian wrt z is EXACTLY constant: dV/dz = 0, so du/dz = V(x).

This is structurally the CP decoder: V(x) is the learned reduced basis,
z is the coefficient vector. The whole NM-ROM solve reduces to a single
linear-least-squares problem -- GN converges in one iteration.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


def _siren_init(omega: float):
    def init(key, shape, dtype=jnp.float32):
        in_dim = shape[-2]
        bound = (6.0 / in_dim) ** 0.5 / omega
        return jax.random.uniform(key, shape, dtype, -bound, bound)
    return init


def _first_layer_init(in_dim: int):
    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, -1.0 / in_dim, +1.0 / in_dim)
    return init


class LinearAffineZDecoder(nn.Module):
    coord_dim: int = 2
    latent_dim: int = 16
    hidden_dim: int = 256
    num_layers: int = 4
    omega_0: float = 30.0
    omega: float = 1.0
    bias_hidden: int = 128

    def setup(self):
        # Phi trunk.
        self._phi_first_W = self.param(
            "phi_W_in", _first_layer_init(self.coord_dim),
            (self.coord_dim, self.hidden_dim),
        )
        self._phi_first_b = self.param(
            "phi_b_in", nn.initializers.zeros, (self.hidden_dim,),
        )
        self._phi_W = [
            self.param(f"phi_W_{l}", _siren_init(self.omega),
                       (self.hidden_dim, self.hidden_dim))
            for l in range(1, self.num_layers)
        ]
        self._phi_b = [
            self.param(f"phi_b_{l}", nn.initializers.zeros, (self.hidden_dim,))
            for l in range(1, self.num_layers)
        ]
        # bias SIREN.
        self._b_W_in = self.param("bias_W_in", _first_layer_init(self.coord_dim),
                                  (self.coord_dim, self.bias_hidden))
        self._b_b_in = self.param("bias_b_in", nn.initializers.zeros,
                                  (self.bias_hidden,))
        self._b_W_mid = self.param("bias_W_mid", _siren_init(self.omega),
                                   (self.bias_hidden, self.bias_hidden))
        self._b_b_mid = self.param("bias_b_mid", nn.initializers.zeros,
                                   (self.bias_hidden,))
        self._b_W_out = self.param("bias_W_out", _siren_init(self.omega),
                                   (self.bias_hidden, 1))
        self._b_b_out = self.param("bias_b_out", nn.initializers.zeros, (1,))
        # Linear coef map: A(z) = W_A @ z + b_A.
        self._W_A = self.param(
            "W_A", nn.initializers.lecun_normal(),
            (self.latent_dim, self.hidden_dim),
        )
        self._b_A = self.param(
            "b_A", nn.initializers.zeros, (self.hidden_dim,),
        )

    def phi(self, x):
        a = self.omega_0 * (x @ self._phi_first_W + self._phi_first_b)
        h = jnp.sin(a)
        for W, b in zip(self._phi_W, self._phi_b):
            h = jnp.sin(self.omega * (h @ W + b))
        return h

    def bias(self, x):
        a = self.omega_0 * (x @ self._b_W_in + self._b_b_in)
        h = jnp.sin(a)
        h = jnp.sin(self.omega * (h @ self._b_W_mid + self._b_b_mid))
        return (h @ self._b_W_out + self._b_b_out).squeeze(-1)

    def coef(self, z):
        """A(z) = W_A^T @ z + b_A — LINEAR in z."""
        return jnp.dot(z, self._W_A) + self._b_A

    def __call__(self, z, x):
        return jnp.dot(self.phi(x), self.coef(z)) + self.bias(x)
