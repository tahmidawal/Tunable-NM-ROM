"""Bounded local smoke: the new arms run, record stopping status, and agree where settings agree.

L=64 saved operators only.  No accuracy, timing or reference claim is made here.
"""
import json
from pathlib import Path

import numpy as np
import data as d
import tuning as t

jax, e = d.gpu_modules()
import jax.numpy as jnp
import accuracy_paths as ap

L, M, K, R = 64, 64, 16, 512
checkpoint = d.ROOT / t.CHECKPOINT_RELATIVE
cfg = json.loads(t.CONFIG_PATH.read_text())
assert d.sha(checkpoint) == cfg['checkpoint_sha256']
params, Z, _ = e.sc.load_pkl(checkpoint)
dec = e.sc.SeparableDecoder(params, K, R)
G = dec.feat_at(e.coords(L), chunk=8192)
phi, lam, _ = e.modes(L, M)
archive = np.load(d.ROOT / 'consolidated/fixtures/burgers/operators.npz')
A = jnp.asarray(archive['A'])
assert float(jnp.linalg.norm(A - jnp.asarray(phi).T @ G) / jnp.linalg.norm(A)) < 1e-12
candidate_z = jnp.asarray(archive['candidate_Z'])
hrot = e.sc.head(params, candidate_z) @ jnp.asarray(archive['cold_R']).T
cold = tuple(jnp.asarray(archive[k]) for k in ['cold_xy', 'cold_w', 'cold_Q', 'cold_R']) + (hrot, jnp.sum(hrot * hrot, 1))
operators = (G, A, jnp.asarray(lam), jnp.asarray(archive['G5']), jnp.asarray(archive['Pq']),
             None, None, candidate_z, jnp.asarray(phi))
trust = .01 * float(np.max(np.linalg.norm(Z - Z.mean(0), axis=1)))
physical = e.params_draw(d.case_seed('calibration', 0), 1)[0]
u0 = jnp.asarray(e.initial(L, physical))
nu = jnp.asarray(physical[4])
weak_full = t.make_weak_full(e, e.sc)

# -- 1. full-grid weak residual against an independent NumPy assembly ------------
z = jnp.asarray(Z[0])
prev = A @ e.sc.head(params, z)
h = np.asarray(e.sc.head(params, z))
u = (np.asarray(G) @ h).reshape(L - 1, L - 1)
p = np.pad(u, 1)
c, xm, xp, ym, yp = p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]
adv = (c * L * (np.where(c > 0, c - xm, xp - c) + np.where(c > 0, c - ym, yp - c))).reshape(-1)
ah = np.asarray(A) @ h
expected = (ah - np.asarray(prev) + cfg['fixed']['dt'] * (np.asarray(phi).T @ adv + float(nu) * np.asarray(lam) * ah)) \
    / (1 + cfg['fixed']['dt'] * float(nu) * np.asarray(lam))
got = np.asarray(weak_full(z, prev, nu, operators, params, L, cfg['fixed']['dt']))
operator_parity = float(np.linalg.norm(got - expected) / max(np.linalg.norm(expected), 1e-300))

# -- 2. Jacobian consistency against central finite differences -----------------
jac = np.asarray(jax.jacfwd(lambda zz: weak_full(zz, prev, nu, operators, params, L, cfg['fixed']['dt']))(z))
step = 1e-6 * (1 + float(np.linalg.norm(z)))
fd = np.stack([(np.asarray(weak_full(z.at[i].add(step), prev, nu, operators, params, L, cfg['fixed']['dt']))
                - np.asarray(weak_full(z.at[i].add(-step), prev, nu, operators, params, L, cfg['fixed']['dt']))) / (2 * step)
               for i in range(K)], axis=1)
gradient_parity = float(np.linalg.norm(jac - fd) / max(np.linalg.norm(jac), 1e-300))

# -- 3. sampled versus full quadrature discrepancy on the same state -------------
sampled = np.asarray(e.weak(z, prev, nu, operators, params, L, cfg['fixed']['dt']))
quadrature_discrepancy = float(np.linalg.norm(sampled - got) / max(np.linalg.norm(got), 1e-300))

# -- 4. both arms run end to end and record their stopping status ---------------
ctx = dict(ap=ap, jax=jax, params=params, L=L, dt=cfg['fixed']['dt'], ic_budget=cfg['fixed']['initial_fit']['ic_budget'],
           initial_gtol=cfg['fixed']['initial_fit']['gtol'], weak_full=weak_full)
records = {}
for quadrature in ['m256', 'full']:
    for setting_name in ['native', 'cap2']:
        query = t.make_arm_query(ctx, quadrature, cfg['settings'][setting_name], trust)
        host = jax.tree_util.tree_map(np.asarray, query(u0, nu, operators, cold))
        assert np.array_equal(host[0][0], np.asarray(u0)), 'deployable output must return the supplied field'
        records[f'{quadrature}_{setting_name}'] = dict(t.stopping_record(host), fields_sha256=t.field_hash(host[0]))

# -- 5. explicit default-valued controls reproduce the implicit defaults exactly --
explicit = ap.make_rom(params, L, cfg['fixed']['dt'], trust, ic_budget=400, step_budget=180, gtol=1e-6,
                       evolution_gtol=1e-6, evolution_residual_scale=1e-9, weak_fn=None)
implicit = ap.make_rom(params, L, cfg['fixed']['dt'], trust, ic_budget=400, step_budget=180, gtol=1e-6)
a = jax.tree_util.tree_map(np.asarray, explicit(u0, nu, operators, cold))
b = jax.tree_util.tree_map(np.asarray, implicit(u0, nu, operators, cold))
identical = all(np.array_equal(x, y) for x, y in zip(a, b))

record = dict(scope='bounded local L=64 saved-operator smoke; no accuracy, timing or reference claim',
              passed=bool(operator_parity < 1e-12 and gradient_parity < 1e-6 and identical
                          and records['m256_cap2']['early_stopped']),
              full_grid_operator_numpy_parity=operator_parity, declared_operator_tolerance=1e-12,
              full_grid_jacobian_finite_difference_parity=gradient_parity, declared_gradient_tolerance=1e-6,
              sampled_versus_full_quadrature_relative_discrepancy=quadrature_discrepancy,
              explicit_default_controls_bit_identical=bool(identical), arms=records,
              backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, f64=bool(jax.config.jax_enable_x64),
              matmul_precision=str(jax.config.jax_default_matmul_precision), jax_version=jax.__version__,
              checkpoint_sha256=d.sha(checkpoint), tuning_source_sha256=d.sha(Path(t.__file__)),
              accuracy_paths_sha256=d.sha(Path(ap.__file__)), config_sha256=d.sha(t.CONFIG_PATH))
d.write_json(d.HERE / 'checks/tuning-smoke.json', record)
print(json.dumps(record, indent=1), flush=True)
assert record['passed']
