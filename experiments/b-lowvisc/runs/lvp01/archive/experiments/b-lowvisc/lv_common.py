"""b-lowvisc: the one thing this lane changes about the Burgers family, in one place.

The incumbent Burgers cell draws its five physical descriptors with `engines.params_draw`:

    cx ~ U(.15,.85)   cy ~ U(.15,.85)   w ~ U(.05,.20)   a ~ U(.5,2.)
    nu = exp(U(log .01, log .1))

This lane keeps the identical sequential NumPy draw and changes only the viscosity bounds to
`(1e-3, 1e-2)`.  Because `Generator.uniform(lo, hi, n)` consumes the same underlying uniforms
whatever the bounds, and the two intervals are both exactly one decade wide in log space:

  * columns 0..3 (cx, cy, w, a) are **bit-identical** to the incumbent draw for the same
    (seed, count), so the initial conditions are literally the same fields; and
  * `nu_lowvisc == nu_incumbent / 10` to <= 4e-16 relative, case by case.

So the low-viscosity cohort is the incumbent cohort with the viscosity divided by ten, and
nothing else.  That is the controlled variable this lane needs.  `verify_family_relation()`
asserts both statements at run time and returns them for `result.json`.

`params_draw` here is `engines.params_draw` with the bounds lifted into arguments; called with
the defaults it is the same function.  It is NumPy only so the audit can import it without JAX.
"""
from __future__ import annotations

import hashlib

import numpy as np

INCUMBENT_NU = (0.01, 0.1)
LOWVISC_NU = (0.001, 0.01)


def params_draw(seed, count, nu_lo=INCUMBENT_NU[0], nu_hi=INCUMBENT_NU[1]):
    """`engines.params_draw` with the viscosity bounds as arguments (defaults = incumbent)."""
    r = np.random.default_rng(seed)
    return np.stack([r.uniform(.15, .85, count), r.uniform(.15, .85, count),
                     r.uniform(.05, .20, count), r.uniform(.5, 2., count),
                     np.exp(r.uniform(np.log(nu_lo), np.log(nu_hi), count))], axis=1)


def cohort(cfg, nu_lo, nu_hi):
    """The six development cases: 4 opened + 2 fresh, the incumbent's own two draws."""
    return np.concatenate((params_draw(cfg['eval_seed'], cfg['eval_cases'], nu_lo, nu_hi),
                           params_draw(cfg['eval_fresh_seed'], cfg['eval_fresh_cases'], nu_lo, nu_hi)))


def verify_family_relation(seed, count, lo=LOWVISC_NU, hi=INCUMBENT_NU):
    """The two claims the design rests on, recomputed rather than asserted in prose."""
    a = params_draw(seed, count, *hi)
    b = params_draw(seed, count, *lo)
    descriptors_identical = bool(np.array_equal(a[:, :4], b[:, :4]))
    ratio = a[:, 4] / b[:, 4]
    max_ratio_deviation = float(np.max(np.abs(ratio - 10.) / 10.))
    return dict(seed=int(seed), count=int(count),
                descriptors_bitwise_identical=descriptors_identical,
                viscosity_ratio=ratio.tolist(),
                max_relative_deviation_from_ten=max_ratio_deviation,
                passed=bool(descriptors_identical and max_ratio_deviation < 1e-13))


def grid_reynolds(physical, L):
    """Cell Reynolds number a*h/nu and the first-order-upwind numerical viscosity a*h/2.

    Reported, not gated: it is the honest measure of how well a mesh resolves a family, and
    the reason the low-viscosity cell cannot simply be run at an arbitrarily small nu.
    """
    a, nu = np.asarray(physical)[:, 3], np.asarray(physical)[:, 4]
    h = 1. / L
    return dict(intervals=int(L), cell_reynolds=(a * h / nu).tolist(),
                numerical_viscosity=(a * h / 2.).tolist(),
                numerical_over_physical=(a * h / (2. * nu)).tolist(),
                worst_cell_reynolds=float(np.max(a * h / nu)))


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()
