"""PDE definitions.

Every PDE is written as   u_t + N(u) = nu * Lap u + r u + f   on (0,1)^2, u = 0 on
the boundary (steady problems: nu * (-Lap u) + N(u) = f).  The linear part is
preassembled in the sine basis (it is exact online); N(u) is the only term that
needs a quadrature rule.  Each PDE provides

  nonlinear_grid(u, N, p)          N_h(u) on the mesh (the FOM discretisation)
  nonlinear_point(u, ux, uy, p)    pointwise continuum N(u) for off-mesh rules
  nonlinear_flux(u, p)             optional (F1, F2) with N(u) = div F, used in the
                                   integrated-by-parts form (no gradient of u needed)
  needs_grad                       whether nonlinear_point uses ux, uy
"""
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
import jax.numpy as jnp


_AC_TAU = 0.05


def gaussian_bump(X, a, c, w):
    r2 = (X[:, 0] - c[0]) ** 2 + (X[:, 1] - c[1]) ** 2
    return a * np.exp(-r2 / (2 * w**2))


def _pad(u):
    return jnp.pad(u, 1)


def stencil_apply(u, N, fn, p):
    """Evaluate a 5-point stencil function fn(uc, uxm, uxp, uym, uyp, N, p) on the grid."""
    up = _pad(u)
    return fn(u, up[:-2, 1:-1], up[2:, 1:-1], up[1:-1, :-2], up[1:-1, 2:], N, p)


def upwind_advection_stencil(uc, uxm, uxp, uym, uyp, N, p):
    dxb, dxf = (uc - uxm) * N, (uxp - uc) * N
    dyb, dyf = (uc - uym) * N, (uyp - uc) * N
    pos = uc > 0
    return uc * (jnp.where(pos, dxb, dxf) + jnp.where(pos, dyb, dyf))


def hj_stencil(uc, uxm, uxp, uym, uyp, N, p):
    """Osher--Sethian monotone Hamiltonian for H(p) = |p|^2/2."""
    dxb, dxf = (uc - uxm) * N, (uxp - uc) * N
    dyb, dyf = (uc - uym) * N, (uyp - uc) * N
    gx2 = jnp.maximum(dxb, 0.0) ** 2 + jnp.minimum(dxf, 0.0) ** 2
    gy2 = jnp.maximum(dyb, 0.0) ** 2 + jnp.minimum(dyf, 0.0) ** 2
    return 0.5 * (gx2 + gy2)


def ac_stencil(uc, uxm, uxp, uym, uyp, N, p):
    return uc**3 / p["tau"]


def bratu_stencil(uc, uxm, uxp, uym, uyp, N, p):
    return -p["lam"] * jnp.exp(uc)


@dataclass
class PDE:
    name: str
    steady: bool
    dt: float
    T: float
    out_times: np.ndarray
    sample_params: Callable      # rng -> dict of per-case parameters
    u0: Callable                 # (X, p) -> initial field values at points X
    source: Callable             # (X, p) -> f at points X (None-safe: zeros)
    nu: Callable                 # p -> diffusion coefficient
    react: Callable              # p -> linear reaction coefficient r
    nonlinear_stencil: Callable  # (uc, uxm, uxp, uym, uyp, N, p) -> pointwise on the mesh
    nonlinear_point: Callable    # (u, ux, uy, p) -> pointwise
    nonlinear_flux: Optional[Callable]
    needs_grad: bool
    description: str = ""

    def nonlinear_grid(self, u, N, p):
        return stencil_apply(u, N, self.nonlinear_stencil, p)


# ---------------------------------------------------------------- Burgers 2D
def _burgers_params(rng):
    return dict(nu=float(np.exp(rng.uniform(np.log(0.01), np.log(0.1)))),
                a=float(rng.uniform(0.5, 2.0)), c=rng.uniform(0.15, 0.85, size=2),
                w=float(rng.uniform(0.05, 0.20)))


BURGERS = PDE(
    name="burgers", steady=False, dt=0.005, T=0.25,
    out_times=np.array([0.05, 0.10, 0.15, 0.20, 0.25]),
    sample_params=_burgers_params,
    u0=lambda X, p: gaussian_bump(X, p["a"], p["c"], p["w"]),
    source=lambda X, p: np.zeros(X.shape[0]),
    nu=lambda p: p["nu"], react=lambda p: 0.0,
    nonlinear_stencil=upwind_advection_stencil,
    nonlinear_point=lambda u, ux, uy, p: u * (ux + uy),
    nonlinear_flux=lambda u, p: (0.5 * u * u, 0.5 * u * u),
    needs_grad=True,
    description="u_t + u(u_x+u_y) = nu Lap u, sign-upwind stencil, backward Euler",
)


# ------------------------------------------------------------ Allen--Cahn 2D


def _ac_params(rng, tau=_AC_TAU):
    return dict(eps=float(np.exp(rng.uniform(np.log(1e-3), np.log(1e-2)))),
                a=float(rng.uniform(0.8, 1.6)), c=rng.uniform(0.25, 0.75, size=2),
                w=float(rng.uniform(0.10, 0.25)), tau=tau)


ALLEN_CAHN = PDE(
    name="allen_cahn", steady=False, dt=0.005, T=0.25,
    out_times=np.array([0.05, 0.10, 0.15, 0.20, 0.25]),
    sample_params=_ac_params,
    u0=lambda X, p: gaussian_bump(X, p["a"], p["c"], p["w"]),
    source=lambda X, p: np.zeros(X.shape[0]),
    nu=lambda p: p["eps"], react=lambda p: 1.0 / p["tau"],
    nonlinear_stencil=ac_stencil,
    nonlinear_point=lambda u, ux, uy, p: u**3 / p["tau"],
    nonlinear_flux=None,
    needs_grad=False,
    description="u_t = eps Lap u + (u - u^3)/tau, tau=0.05; cubic reaction, sharp fronts",
)


ALLEN_CAHN_MILD = PDE(
    name="allen_cahn_mild", steady=False, dt=0.005, T=0.25,
    out_times=np.array([0.05, 0.10, 0.15, 0.20, 0.25]),
    sample_params=lambda rng: _ac_params(rng, tau=0.2),
    u0=lambda X, p: gaussian_bump(X, p["a"], p["c"], p["w"]),
    source=lambda X, p: np.zeros(X.shape[0]),
    nu=lambda p: p["eps"], react=lambda p: 1.0 / p["tau"],
    nonlinear_stencil=ac_stencil,
    nonlinear_point=lambda u, ux, uy, p: u**3 / p["tau"],
    nonlinear_flux=None,
    needs_grad=False,
    description="u_t = eps Lap u + (u - u^3)/tau, tau=0.2; milder reaction (less error amplification)",
)


# ------------------------------------------------- viscous Hamilton--Jacobi
def _hj_params(rng):
    return dict(nu=float(np.exp(rng.uniform(np.log(0.01), np.log(0.1)))),
                a=float(rng.uniform(0.5, 2.0)), c=rng.uniform(0.15, 0.85, size=2),
                w=float(rng.uniform(0.05, 0.20)))


HAMILTON_JACOBI = PDE(
    name="hj", steady=False, dt=0.005, T=0.25,
    out_times=np.array([0.05, 0.10, 0.15, 0.20, 0.25]),
    sample_params=_hj_params,
    u0=lambda X, p: gaussian_bump(X, p["a"], p["c"], p["w"]),
    source=lambda X, p: np.zeros(X.shape[0]),
    nu=lambda p: p["nu"], react=lambda p: 0.0,
    nonlinear_stencil=hj_stencil,
    nonlinear_point=lambda u, ux, uy, p: 0.5 * (ux * ux + uy * uy),
    nonlinear_flux=None,
    needs_grad=True,
    description="u_t + |grad u|^2/2 = nu Lap u; gradient nonlinearity (Osher-Sethian upwind), not a divergence",
)


# ------------------------------------------------------- Bratu (steady, elliptic)
def _bratu_params(rng):
    return dict(lam=float(rng.uniform(0.5, 3.0)), a=float(rng.uniform(20.0, 80.0)),
                c=rng.uniform(0.2, 0.8, size=2), w=float(rng.uniform(0.05, 0.15)))


BRATU = PDE(
    name="bratu", steady=True, dt=1.0, T=0.0, out_times=np.array([0.0]),
    sample_params=_bratu_params,
    u0=lambda X, p: np.zeros(X.shape[0]),
    source=lambda X, p: gaussian_bump(X, p["a"], p["c"], p["w"]),
    nu=lambda p: 1.0, react=lambda p: 0.0,
    nonlinear_stencil=bratu_stencil,
    nonlinear_point=lambda u, ux, uy, p: -p["lam"] * jnp.exp(u),
    nonlinear_flux=None,
    needs_grad=False,
    description="-Lap u = lam exp(u) + f (steady); exponential nonlinearity",
)

PDES = {p.name: p for p in [BURGERS, ALLEN_CAHN, ALLEN_CAHN_MILD, HAMILTON_JACOBI, BRATU]}
