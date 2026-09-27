"""Fresh scalar-wave reference: tensor edge stiffness, trapezoid mass, RK4.

No dependency on any previous wave implementation or artifact.
Axis names follow arrays: axis -2 is x; axis -1 is y.
"""
from dataclasses import dataclass
from functools import partial
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class Grid:
    n: int
    bx: str = "dirichlet"
    by: str = "dirichlet"
    length: float = 1.0

    def __post_init__(self):
        if self.n < 3 or self.length <= 0:
            raise ValueError("n >= 3 and positive length required")
        if any(b not in ("dirichlet", "absorbing", "neumann", "periodic") for b in (self.bx, self.by)):
            raise ValueError("unknown boundary condition")

    @property
    def h(self):
        return self.length / self.n

    def axis(self, bc):
        if bc == "periodic":
            return np.arange(self.n, dtype=np.float64) * self.h
        x = np.linspace(0, self.length, self.n + 1)
        return x[1:-1] if bc == "dirichlet" else x

    @property
    def shape(self):
        return len(self.axis(self.bx)), len(self.axis(self.by))

    def coordinates(self):
        return np.stack(np.meshgrid(self.axis(self.bx), self.axis(self.by), indexing="ij"), -1)

    def axis_weights(self, bc):
        w = np.full(len(self.axis(bc)), self.h)
        if bc in ("absorbing", "neumann"):
            w[[0, -1]] *= .5
        return w

    def mass(self):
        return self.axis_weights(self.bx)[:, None] * self.axis_weights(self.by)[None, :]

    def boundary_ratio(self, axis):
        bc = (self.bx, self.by)[axis]
        b = np.zeros(self.shape[axis])
        if bc == "absorbing":
            b[[0, -1]] = 2 / self.h
        return b[:, None] if axis == 0 else b[None, :]


def _edge_laplacian(u, h, bc, axis):
    """Return positive H^-1 S u; endpoint rows are derived from edge energy."""
    q = jnp.moveaxis(u, axis, -1)
    if bc == "periodic":
        out = (2*q - jnp.roll(q, 1, -1) - jnp.roll(q, -1, -1)) / h**2
    elif bc == "dirichlet":
        qp = jnp.pad(q, [(0, 0)] * (q.ndim-1) + [(1, 1)])
        out = (2*q - qp[..., :-2] - qp[..., 2:]) / h**2
    else:
        edges = (q[..., 1:] - q[..., :-1]) / h**2
        out = jnp.concatenate((-2*edges[..., :1], edges[..., :-1]-edges[..., 1:], 2*edges[..., -1:]), -1)
    return jnp.moveaxis(out, -1, axis)


def positive_laplacian(u, grid):
    return _edge_laplacian(u, grid.h, grid.bx, -2) + _edge_laplacian(u, grid.h, grid.by, -1)


def damping_ratio(grid, c):
    return c * (jnp.asarray(grid.boundary_ratio(0)) + jnp.asarray(grid.boundary_ratio(1)))


def acceleration(u, v, grid, c, force=None):
    a = -c*c*positive_laplacian(u, grid) - damping_ratio(grid, c)*v
    return a if force is None else a + force


def rhs(state, t, grid, c, forcing=None):
    u, v = state
    force = None if forcing is None else forcing(t, grid, c)
    return v, acceleration(u, v, grid, c, force)


def rk4_step(state, t, dt, grid, c, forcing=None):
    def add(y, a, scale):
        return tuple(p + scale*q for p, q in zip(y, a))
    a = rhs(state, t, grid, c, forcing)
    b = rhs(add(state, a, dt/2), t+dt/2, grid, c, forcing)
    d = rhs(add(state, b, dt/2), t+dt/2, grid, c, forcing)
    e = rhs(add(state, d, dt), t+dt, grid, c, forcing)
    return tuple(p + dt*(q+2*r+2*s+w)/6 for p, q, r, s, w in zip(state, a, b, d, e))


def rk4_step_balance(state, t, dt, grid, c):
    """RK4 on the augmented system (u,v,q), q'=v.T C v."""
    u, v, q = state
    def f(y):
        uu, vv, _ = y
        return vv, acceleration(uu, vv, grid, c), dissipation(vv, grid, c)
    def add(y, a, scale):
        return tuple(p+scale*r for p, r in zip(y, a))
    a = f(state)
    b = f(add(state, a, dt/2))
    d = f(add(state, b, dt/2))
    e = f(add(state, d, dt))
    return tuple(p+dt*(aa+2*bb+2*dd+ee)/6 for p, aa, bb, dd, ee in zip(state, a, b, d, e))


@partial(jax.jit, static_argnames=("grid", "steps", "stride"))
def integrate_balance(u0, v0, c, dt, *, grid, steps, stride=1):
    if steps % stride:
        raise ValueError("steps must be divisible by stride")
    def block(carry, _):
        state, t = carry
        def step(_, st):
            y, tt = st
            return rk4_step_balance(y, tt, dt, grid, c), tt+dt
        state, t = jax.lax.fori_loop(0, stride, step, (state, t))
        return (state, t), state
    zero = jnp.asarray(0.)
    _, (u, v, q) = jax.lax.scan(block, ((u0, v0, zero), zero), None, length=steps//stride)
    return jnp.concatenate((u0[None], u)), jnp.concatenate((v0[None], v)), jnp.concatenate((zero[None], q))


@partial(jax.jit, static_argnames=("grid", "steps", "stride", "forcing"))
def integrate(u0, v0, c, dt, *, grid, steps, stride=1, forcing=None):
    """Return (u,v) including t=0 at each stride; steps must be divisible by stride."""
    if steps % stride:
        raise ValueError("steps must be divisible by stride")
    def block(carry, _):
        state, t = carry
        def step(_, st):
            y, tt = st
            return rk4_step(y, tt, dt, grid, c, forcing), tt+dt
        state, t = jax.lax.fori_loop(0, stride, step, (state, t))
        return (state, t), state
    _, (u, v) = jax.lax.scan(block, ((u0, v0), jnp.asarray(0.0)), None, length=steps//stride)
    return jnp.concatenate((u0[None], u)), jnp.concatenate((v0[None], v))


def energy(u, v, grid, c):
    mass = jnp.asarray(grid.mass())
    return .5*jnp.sum(mass*(v*v+c*c*u*positive_laplacian(u, grid)), axis=(-2, -1))


def dissipation(v, grid, c):
    return jnp.sum(jnp.asarray(grid.mass())*damping_ratio(grid, c)*v*v, axis=(-2, -1))


def bump(s):
    """Compact smooth bump with exactly zero value and all derivatives at |s|=1."""
    r = jnp.maximum(1-s*s, 1e-30)
    return jnp.where(jnp.abs(s) < 1, jnp.exp(1-1/r), 0.)


def localized_initial(grid, parameters):
    """Parameters: cx, cy, sx, sy, amplitude, speed, vx_fraction, vy_fraction.

    Physical translation derivative v0=-c*(vx*u_x+vy*u_y), evaluated by AD.
    The compact support lies strictly inside both physical domains.
    """
    cx, cy, sx, sy, amp, c, vx, vy = parameters[:8]
    def field(xy):
        value = amp*bump((xy[0]-cx)/sx)*bump((xy[1]-cy)/sy)
        if len(parameters) == 10:
            sigx, sigy = parameters[8:]
            value = value*jnp.exp(-.5*(((xy[0]-cx)/sigx)**2+((xy[1]-cy)/sigy)**2))
        return value
    xy = jnp.asarray(grid.coordinates()).reshape(-1, 2)
    u = jax.vmap(field)(xy)
    grad = jax.vmap(jax.grad(field))(xy)
    v = -c*(vx*grad[:, 0]+vy*grad[:, 1])
    return u.reshape(grid.shape), v.reshape(grid.shape)


def provenance():
    import os, platform
    return {"jax_backend": jax.default_backend(), "jax_version": jax.__version__,
            "x64": bool(jax.config.jax_enable_x64), "matmul_precision": str(jax.config.jax_default_matmul_precision),
            "devices": [str(d) for d in jax.devices()], "device_kind": [d.device_kind for d in jax.devices()], "host": platform.node(),
            "source_commit": os.environ.get("COMMIT", os.environ.get("SOURCE_COMMIT")),
            "job_id": os.environ.get("SLURM_JOB_ID"), "numpy_version": np.__version__}
