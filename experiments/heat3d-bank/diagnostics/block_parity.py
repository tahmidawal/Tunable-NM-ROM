"""Blocked encode/decode (core.row_blocks) vs the single-GEMM path, real R=320 bank at 32^3/64^3."""
import sys
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import core as C
model = C.load_model(dict(kind='mlp', bank='vp_R320/bank.pkl', head='vp_R320/head_K32.pkl'), Path(__file__).resolve().parents[1] / 'inputs')
for n in (32, 64):
    bank = C.bank_at(model, n); N = bank.shape[0]; u0 = C.initial_grid(n, 3, C.family('h3d', 921777, 1)[0]).reshape(-1)
    coefs = jnp.asarray(np.random.default_rng(0).normal(size=(6, bank.shape[1])))
    one, many = C.row_blocks(N, 1 << 30), C.row_blocks(N, 4096)
    p1, p2 = C.bank_project(bank, u0, one), C.bank_project(bank, u0, many)
    e1, e2 = C.bank_expand(coefs, bank, one), C.bank_expand(coefs, bank, many)
    print(n, 'blocks', len(many), 'project rel', float(jnp.max(jnp.abs(p1 - p2)) / jnp.max(jnp.abs(p1))),
          'expand rel', float(jnp.max(jnp.abs(e1 - e2)) / jnp.max(jnp.abs(e1))))
