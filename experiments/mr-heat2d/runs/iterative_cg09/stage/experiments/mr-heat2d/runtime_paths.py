"""Identical modular and compiled compositions of the existing heat query."""
import jax
import numpy as np

import heat_core as hc


def build_paths(cfg, dt):
    initialize, rollout, readout = hc.make_query(cfg, dt)

    @jax.jit
    def compiled(params, projection, triangular, library, codes, bank, matrix, mode_lam, u0):
        z0, init_info = initialize(params, projection, triangular, library, codes, u0)
        factor = hc.cn_factor(mode_lam, dt, cfg["diffusivity"])
        zs, step_info = rollout(params, matrix, factor, z0)
        fields = readout(params, bank, zs)
        return fields, init_info, step_info, zs

    return dict(initialize=initialize, rollout=rollout, readout=readout, compiled=compiled)


def parity(modular, compiled, field_tolerance, latent_tolerance):
    """Rows carry both starts, all steps and latents; nothing inferred from fields."""
    fm, im, sm, zm = [np.asarray(value) for value in modular]
    fc, ic, sc, zc = [np.asarray(value) for value in compiled]
    relative = lambda a, b: float(np.linalg.norm(a-b)/max(np.linalg.norm(a), 1e-300))
    counters_equal = bool(np.array_equal(im[:, :3], ic[:, :3]) and np.array_equal(sm[:, :3], sc[:, :3]))
    field_error = relative(fm, fc)
    latent_error = relative(zm, zc)
    return dict(field_relative_error=field_error, latent_relative_error=latent_error,
                exact_counter_reason_parity=counters_equal,
                initial_counter_mismatches=int(np.sum(im[:, :3] != ic[:, :3])),
                step_counter_mismatches=int(np.sum(sm[:, :3] != sc[:, :3])),
                passed=bool(field_error <= field_tolerance and latent_error <= latent_tolerance and counters_equal))
