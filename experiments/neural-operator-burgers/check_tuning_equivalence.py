"""Default arguments of the retuned solver reproduce the archived stopping records.

Bounded local replay of already-opened saved calibration cases at the production
mesh with the archived empirical-quadrature rule.  Not a benchmark: no timing,
accuracy or reference claim is made here.
"""
import json
from pathlib import Path

import numpy as np
import data as d
import tuning as t

jax, e = d.gpu_modules()
import jax.numpy as jnp
import accuracy_paths as ap

CASES = ['burgers-calibration-00002', 'burgers-calibration-00003']
folder = d.HERE / 'runs/refinement02/live-diagnosis'
reference = json.loads((d.HERE / 'runs/refinement02/live-reference/index.json').read_text())
archived = json.loads((folder / 'index.json').read_text())
cfg = json.loads(t.CONFIG_PATH.read_text())
assert d.sha(folder / 'index.json') == cfg['source_diagnosis_index_sha256']
checkpoint = d.ROOT / t.CHECKPOINT_RELATIVE
assert d.sha(checkpoint) == archived['checkpoint_sha256'] == cfg['checkpoint_sha256']

L, M, K, R = archived['intervals'], archived['M'], archived['K'], archived['R']
params, Z, _ = e.sc.load_pkl(checkpoint)
dec = e.sc.SeparableDecoder(params, K, R)
G = dec.feat_at(e.coords(L), chunk=8192)
phi, lam, mode_ids = e.modes(L, M)
assert np.array_equal(mode_ids, np.asarray(archived['setup']['mode_ids']))
A = jnp.asarray(phi).T @ G
pos = np.asarray(cfg['archived_rule']['eq_indices'])
weights = np.asarray(cfg['archived_rule']['eq_weights'])
G5, Pq = t.quadrature_arrays(e, jnp, dec, phi, pos, weights, L, R)
candidate_z = jnp.asarray(Z[::max(1, len(Z) // 8192)])
operators = (G, A, jnp.asarray(lam), G5, Pq, None, None, candidate_z, jnp.asarray(phi))
cold, _ = e.build_gauss_cold(params, candidate_z)
trust = archived['setup']['trust_radius']
strict = archived['config']['strict']
query = ap.make_rom(params, L, archived['config']['dt'], trust, **strict)

rows = []
for case_id in CASES:
    row = next(r for r in archived['cases'] if r['case_id'] == case_id)
    artifact = folder / row['path']
    assert d.sha(artifact) == row['sha256']
    with np.load(artifact) as arrays:
        saved = {k: arrays[k] for k in ['input', 'online', 'online_z', 'step_iterations', 'step_reasons',
                                        'initial_iterations', 'initial_reason', 'step_gradients', 'initial_gradient']}
    nu = next(r['generation_descriptors']['nu'] for r in reference['records'] if r['case_id'] == case_id)
    host = jax.tree_util.tree_map(np.asarray, query(jnp.asarray(saved['input']), jnp.asarray(nu), operators, cold))
    rows.append(dict(case_id=case_id, archived_sha256=row['sha256'],
        step_iterations_identical=bool(np.array_equal(host[1], saved['step_iterations'])),
        step_reasons_identical=bool(np.array_equal(host[3], saved['step_reasons'])),
        initial_iterations_identical=bool(np.array_equal(host[5], saved['initial_iterations'])),
        initial_reason_identical=bool(np.array_equal(host[6], saved['initial_reason'])),
        step_gradients_bitwise_identical=bool(np.array_equal(host[8], saved['step_gradients'])),
        relative_field_difference=float(np.linalg.norm(host[0] - saved['online']) / np.linalg.norm(saved['online'])),
        relative_latent_difference=float(np.linalg.norm(host[4] - saved['online_z']) / np.linalg.norm(saved['online_z'])),
        maximum_absolute_step_gradient_difference=float(np.max(np.abs(host[8] - saved['step_gradients']))),
        **t.stopping_record(host)))

integer_exact = all(r['step_iterations_identical'] and r['step_reasons_identical']
                    and r['initial_iterations_identical'] and r['initial_reason_identical'] for r in rows)
float_close = all(r['relative_field_difference'] <= 1e-10 and r['relative_latent_difference'] <= 1e-10 for r in rows)
record = dict(scope='already-opened saved calibration cases replayed locally with the retuned default arguments',
              passed=bool(integer_exact and float_close),
              integer_stopping_records_bitwise_identical=integer_exact,
              float_agreement_within_declared_tolerance=float_close, declared_float_tolerance=1e-10,
              archived_gpu=archived['provenance']['gpu'], archived_job_id=archived['provenance']['job_id'],
              archived_jax=archived['provenance']['packages']['jax'],
              replay_gpu=jax.devices()[0].device_kind, replay_jax=jax.__version__,
              note='floating-point equality is not expected across GPU models; integer stopping records must be identical',
              cases=rows, backend=jax.default_backend(), f64=bool(jax.config.jax_enable_x64),
              matmul_precision=str(jax.config.jax_default_matmul_precision),
              checkpoint_sha256=d.sha(checkpoint), accuracy_paths_sha256=d.sha(Path(ap.__file__)),
              tuning_source_sha256=d.sha(Path(t.__file__)), config_sha256=d.sha(t.CONFIG_PATH),
              source_sha256=d.source_hashes())
d.write_json(d.HERE / 'checks/tuning-default-equivalence.json', record)
print(json.dumps(record, indent=1), flush=True)
assert record['passed']
