"""Regression test for the frozen-rollout bug (fix/heat-rollout-warm-start).

The rollout used to hand the next step `s * f_norm_eq(z_new)` — the implicit
operator applied to the new state — which at convergence equals the PREVIOUS
step's target, so every step after the first started at (near-)zero residual
and the "T-step" rollout returned the 1-step solution.

Detection note: neither iteration counts nor bitwise state comparison detect
this reliably — under the bug, later steps can burn max_iters on ~1e-7
cross-program float noise and drift z by an ulp or not at all, run to run.
The robust discriminator is the MAGNITUDE of the state change: a frozen chain
moves the final state by float noise (measured ~5e-7 relative), a real chain
moves it at physical scale (measured ~0.2–0.9 relative for this setup).
Threshold 1e-3 sits four orders above the noise and two below the signal.
"""
import jax
import jax.numpy as jnp
import numpy as np

from tunable_rom_heat.fom.heat import HeatFOM, generate_trajectory, sample_parameters
from tunable_rom_heat.models import ViTCPAutoencoder
from tunable_rom_heat.eq.nnls import compute_eq_weights, build_v_eq
from tunable_rom_heat.solver.nm_rom import NMROMSolver

N = 8
D = 2
FROZEN_THRESHOLD = 1e-3


def _build_solver():
    fom = HeatFOM(N=N, spatial_dim=D)
    rng = np.random.default_rng(0)
    snaps = []
    for _ in range(4):
        p = sample_parameters(rng, D)
        snaps.append(np.asarray(generate_trajectory(fom, p, num_steps=3)))
    U_train = np.concatenate(snaps, axis=0)
    kappa = 0.3

    model = ViTCPAutoencoder(
        N=N, spatial_dim=D, patch_size=4, embed_dim=16,
        num_heads=2, num_enc_layers=2, latent_dim=8, rank=16, hidden_dim=32,
    )
    params = model.init(jax.random.PRNGKey(0), jnp.asarray(U_train[0]))["params"]
    eq_flat, eq_w = compute_eq_weights(
        model=model, params=params, snapshots=U_train,
        N=N, spatial_dim=D, n_eq_samples=2, min_eq_points=4,
    )
    v_eq_st, stencil_idx = build_v_eq(params["decoder"], eq_flat, N, D)
    solver = NMROMSolver(
        autoencoder=model, params=params, N=N, spatial_dim=D, dx=fom.dx,
        eq_flat_indices=eq_flat, eq_weights=eq_w,
        v_eq_stencil=v_eq_st, stencil_indices=stencil_idx,
        gn_max_iters=8,
    )
    return solver, jnp.asarray(U_train[0]), jnp.float32(kappa)


def _rel_change(solver, u0, kappa, T):
    u_1, _ = solver.rollout(u0, kappa, num_steps=1)
    u_T, _ = solver.rollout(u0, kappa, num_steps=T)
    u_1, u_T = np.asarray(u_1), np.asarray(u_T)
    return np.linalg.norm(u_T - u_1) / np.linalg.norm(u_1)


def test_second_step_advances_state():
    solver, u0, kappa = _build_solver()
    d = _rel_change(solver, u0, kappa, T=2)
    assert d > FROZEN_THRESHOLD, (
        f"step 2 changed the ROM state by only {d:.3e} (float noise): frozen "
        "warm-start chain — the rollout is handing A(u_new) instead of u_new "
        "to the next step (the pre-fix bug)."
    )


def test_long_rollout_advances_state():
    solver, u0, kappa = _build_solver()
    d = _rel_change(solver, u0, kappa, T=10)
    assert d > FROZEN_THRESHOLD, (
        f"10-step rollout differs from 1-step by only {d:.3e} (float noise): "
        "frozen warm-start chain (the pre-fix bug)."
    )
