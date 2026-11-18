"""Sub-minute GPU implementation fixture; no research result is derived here."""
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
import core as C
from run import action_check, save_json, clean

start = time.perf_counter()
n, width, count = 16, 5, 32
xy = C.e.coords(n)
modes, lam = C.mode_order(n, count)
G = jnp.asarray(2*np.sin(np.pi*xy[:, 0:1]*modes[None, :width, 0])*np.sin(np.pi*xy[:, 1:2]*modes[None, :width, 1]))
A = C.project(G, jnp.asarray(modes), n)
data = dict(G=G, A=A, modes=jnp.asarray(modes), lam=jnp.asarray(lam))
gram, eigenvalues, transform = C.physical_tangent(G, n)
c = np.array([.1, .02, -.01, .005, .003])
check = action_check(c, .99*c, .03, data, n, .005, transform)
assert check['passed'], check
analytic = C.value_jac(jnp.asarray(c), jnp.asarray(.99*c), .03, data, n, .005)[1]
automatic = jax.jacfwd(lambda cc: C.weak_all(cc, jnp.asarray(.99*c), .03, data, n, .005))(jnp.asarray(c))
ad_error = float(jnp.linalg.norm(analytic-automatic)/jnp.linalg.norm(automatic))
assert ad_error < 1e-12, ad_error
u = G @ jnp.asarray(c)
v = G @ jnp.asarray([.01, .02, .03, .04, .05])
original = lambda q: C.reference_residual(q, .99*u, .03, .005, n, False)
stable = lambda q: C.reference_residual(q, .99*u, .03, .005, n, True)
primal = float(jnp.linalg.norm(original(u)-stable(u))/jnp.linalg.norm(u))
jo, js = jax.jvp(original, (u,), (v,))[1], jax.jvp(stable, (u,), (v,))[1]
action = float(jnp.linalg.norm(jo-js)/jnp.linalg.norm(jo))
assert primal < 1e-12 and action < 1e-12, (primal, action)
solver = C.make_solver(n, .005, 2, 30)
coeff, stats, history = jax.block_until_ready(solver(jnp.asarray(c), .03, data, 16, 1e-8))
assert np.all(np.asarray(stats)[:, 1] == 1), np.asarray(stats)
assert np.all(np.isfinite(coeff))
serialization = dict(scalar=jnp.float64(1), nested=[jnp.array([1., jnp.nan]), dict(x=np.float64(np.inf))])
assert json.loads(json.dumps(clean(serialization), allow_nan=False)) == dict(scalar=1., nested=[[1., None], dict(x=None)])
result = dict(passed=True, backend=jax.default_backend(), x64=jax.config.jax_enable_x64,
              ad_relative_error=ad_error, parity=check, solver_stats=np.asarray(stats).tolist(),
              serialization_witness=serialization, reference_arithmetic_parity=dict(primal=primal, action=action), seconds=time.perf_counter()-start)
save_json(__import__('pathlib').Path('checks/local-fixture.json'), result)
print(json.dumps(clean(result)), flush=True)
