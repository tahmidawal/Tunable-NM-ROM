"""Offline analysis environment (DESIGN A8.1): every offline scoring / selection / budget / analysis entry point runs
JAX on CPU with highest matmul precision and x64. Import this module FIRST."""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_DEFAULT_MATMUL_PRECISION'] = 'highest'
import jax  # noqa: E402
jax.config.update('jax_enable_x64', True)
assert jax.default_backend() == 'cpu', jax.default_backend()
