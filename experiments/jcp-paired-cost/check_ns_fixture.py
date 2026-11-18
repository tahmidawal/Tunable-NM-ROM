"""Sub-minute synthetic GPU fixture, not a scientific accuracy experiment."""
import json
import hashlib
import os
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import ns_query as Q
from paired_measurement import write_json


start = time.perf_counter()
assert jax.default_backend() == "gpu"

def affine(x):
    matrix = jnp.array([[1., 0.], [0., 2.], [1., 1.]])
    return matrix @ x-jnp.array([2., -1., .2]), matrix

lm = jax.jit(Q.make_lm(affine, 100, True))
solution, info = jax.block_until_ready(lm(jnp.zeros(2), (), 1e-14))
assert int(info[0]) == 1 and float(info[1]) <= 1e-14, info
assert np.linalg.norm(np.asarray(solution)-np.linalg.lstsq(
    np.array([[1., 0.], [0., 2.], [1., 1.]]), np.array([2., -1., .2]), rcond=None)[0]) < 1e-12
_, cap = jax.jit(Q.make_lm(affine, 0, True))(jnp.zeros(2), (), 1e-14)
assert int(cap[0]) == 4
tie = jnp.array([[2., 1e-8, 1e-4, 4., 3., 0.],
                 [1., 1e-12, 1e-4*(1+128*np.finfo(np.float64).eps), 4., 4., 0.]])
assert int(Q.choose_stationary_fit(tie)[0]) == 1
assert int(Q.choose_stationary_fit(tie.at[1, 2].set(1e-4+1e-9))[0]) == 0
assert int(Q.choose_stationary_fit(tie.at[0, 0].set(1))[0]) == 0

n, rank, latent = 8, 4, 2
G, _, _ = Q.SR.R.test_modes(n, rank)
operators, operator_checks = Q.SR.build_operators_fast(G, n, 24, check_modes=4)
rng = np.random.default_rng(731)
params = {"W1": jnp.asarray(rng.normal(size=(latent, 4))*.1), "b1": jnp.zeros(4),
          "W2": jnp.asarray(rng.normal(size=(4, 4))*.1), "b2": jnp.zeros(4),
          "W3": jnp.asarray(rng.normal(size=(4, rank))*.1), "b3": jnp.zeros(rank),
          "Ws": jnp.asarray(rng.normal(size=(latent, rank)))}
codes = jnp.asarray(rng.normal(size=(6, latent))*.1)
candidates = Q.H.head_apply(params, codes)
u0 = jnp.asarray(Q.F.initial(n, Q.F.parameters(202609202, 1)[0]))
arguments = (jnp.asarray(G), *(jnp.asarray(operators[key]) for key in ("A", "T", "lam", "Dd")),
             params, codes, candidates, jnp.float64(1e-7), jnp.float64(1e-7))
query_results = {}
for kind in ("head", "span"):
    call = Q.make_query(kind, n=n, rank=rank, latent=latent, dt=.002, steps=5,
                       output_every=1, budget=40, fit_starts=2, roundoff_repair=True)
    fields, stats, fit, all_fits = jax.block_until_ready(call(u0, .01, *arguments))
    again = jax.block_until_ready(call(u0, .01, *arguments))[0]
    assert fields.shape == (6, 3, n, n, n) and fields.dtype == jnp.float64
    assert np.isfinite(fields).all() and np.array_equal(fields, again)
    assert stats.shape == (5, 6)
    assert all_fits.shape == ((2, 6) if kind == "head" else (0, 6))
    query_results[kind] = {"field_shape": list(fields.shape), "repeat_identical": True,
                          "step_reasons": np.asarray(stats[:, 0]).astype(int).tolist(),
                          "fit_reason": int(fit[0]), "all_starts_shape": list(all_fits.shape)}
geometry = Q.F.geometry(n)
fom = Q.F.make_solver(.002, 5, 1)
classical = Q.make_classical(scoring_mesh=n, solver_mesh=n, horizon=.01, steps=5, scheme="cnab2")
native = np.asarray(fom(u0, .01, geometry))
wrapped = np.asarray(classical(u0, .01, geometry))
gap = float(np.linalg.norm(native-wrapped)/np.linalg.norm(native))
assert gap < 1e-13
report = {"synthetic_fixture": True, "passed": True, "backend": jax.default_backend(),
          "precision": str(jax.config.jax_default_matmul_precision), "dtype": str(fields.dtype),
          "lm": np.asarray(info).tolist(), "cap_reason": int(cap[0]),
          "stationary_tie_selector_inside_outside_already_stationary": True,
          "queries": query_results, "operator_checks": operator_checks,
          "classical_same_mesh_wrapper_parity": gap, "seconds": time.perf_counter()-start}
report["job_id"] = os.environ.get("SLURM_JOB_ID")
report["source_sha256"] = {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                            for name in ("ns_query.py", "check_ns_fixture.py")}
write_json(Path(os.environ.get("JCP_FIXTURE_OUTPUT", Path(__file__).parent / "checks/ns_fixture.json")), report)
print(json.dumps(report), flush=True)
