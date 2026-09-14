"""Replay one opened development case with the selected frozen solver.

Each invocation handles one PDE in a separate process, since the native research
modules use overlapping names. This is a parity smoke, not an accuracy benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def heat(model, raw):
    import heat_core as hc
    from run_pilot import assemble
    import cp_algebra_paths as ap
    from transfer_core import field_hash
    checkpoint = pickle.loads((ROOT/model['artifacts'][0]).read_bytes())
    params = jax.tree.map(jnp.asarray, checkpoint['params'])
    codes = jnp.asarray(checkpoint['codes'])
    cfg, settings = raw['config'], raw['settings']
    assert checkpoint['config'] == cfg
    ops, checks = assemble(params, codes, 64, cfg)
    mode_lam = hc.eigenvalues(64)[:cfg['modes_per_axis'], :cfg['modes_per_axis']].reshape(-1)
    initial = np.load(ROOT/model['replay']['initial'])['field']
    expected_input = next(r for r in raw['case_fields'] if (r['intervals'],r['case']) == (64,0))['initial']['sha256_array']
    assert field_hash(initial) == expected_input
    triangular, matrix = ops['triangular'], ops['matrix']
    result = ap.build(cfg, settings)['nmrom_cholesky'](
        params, ops['projection'], triangular, ops['library'], codes,
        ops['bank'], matrix, mode_lam, triangular.T@triangular,
        matrix.T@matrix, jnp.asarray(initial))
    fields, fits, steps, *_ = jax.device_get(result)
    assert np.all(fits[:,2] == 1) and np.all(steps[:,2] == 1)
    return {'field': fields.reshape(len(cfg['times']),63,63)}, dict(
        initial_and_steps_stationary=True,
        weak_operator_relative_error=checks['exact_weak_operator_relative_error'])


def poisson(model, raw):
    import correction_core as c
    params, codes, _ = c.sc.load_pkl(ROOT/model['artifacts'][0])
    cfg = raw['config']
    basis = np.load(ROOT/model['artifacts'][2])
    np.testing.assert_array_equal(codes, basis['training_latents'])
    ops = c.assemble(params,codes,64,cfg['requested_modes'],cfg['lm_budget'])
    engine = c.prepare_correction(ops,codes,basis['coefficient_directions'],32,cfg)
    source = c.full_source(64,raw['cohort']['parameters'][0])
    row = next(r for r in raw['rows'] if (r['intervals'],r['case'],r['method'],r['repetition']) == (64,0,'r128_q32',0))
    # Source fields were not archived. Regenerate from the recorded parameters
    # with the exact imported function; transcendental rounding can differ
    # between the cluster's x86 NumPy and this ARM machine.
    source_hashes = dict(regenerated=c.sha(source),archived=row['source_sha256'])
    field, info = c.correction_query(source,ops,engine,cfg)
    assert info['solver_valid']
    return {'field': field}, dict(source_hashes=source_hashes, **{k:info[k] for k in ('solver_valid','stationarity','reduced_stationarity','projected_jacobian_rank')})


def burgers(model, raw):
    import engines as e
    import accuracy_paths as ap
    params, _, _ = e.sc.load_pkl(ROOT/model['artifacts'][0])
    archived = {k:jnp.asarray(a) for k,a in np.load(ROOT/model['replay']['operators']).items()}
    bank = e.sc.SeparableDecoder(params,raw['K'],raw['R']).feat_at(e.coords(64),chunk=8192)
    # The strict path uses entries 0:5 and 7 only. Unused legacy initializer
    # entries are explicit None, rather than silently building a different fit.
    data = (bank, archived['A'], archived['lam'], archived['G5'], archived['Pq'],
            None, None, archived['candidate_Z'])
    hrot = e.sc.head(params,archived['candidate_Z'])@archived['cold_R'].T
    cold = tuple(archived[k] for k in ('cold_xy','cold_w','cold_Q','cold_R')) + (hrot,jnp.sum(hrot*hrot,axis=1))
    cfg = raw['config']
    setup = next(r for r in raw['mesh_setup'] if (r['intervals'],r['model']) == (64,'frozen'))
    physical = raw['physical_cases'][0]
    supplied = jnp.asarray(e.initial(64,physical))
    result = ap.make_rom(params,64,cfg['dt'],setup['trust_radius'],**cfg['strict'])(supplied,jnp.asarray(physical[4]),data,cold)
    fields, _, _, _, _, _, _, internal, gradients, initial_gradient = jax.device_get(result)
    assert np.max(gradients) <= cfg['strict']['gtol'] and initial_gradient <= cfg['strict']['gtol']
    return {'fields':fields}, dict(initial_and_steps_stationary=True,
                                  maximum_step_stationarity=float(np.max(gradients)),
                                  empirical_quadrature='Archived decoder-output rule, unchanged',
                                  internal_latent_sha256=array_sha(internal))


def wave(model, raw):
    import pilot as base
    import iterative_replay as previous
    import acceleration_replay as ar
    from fresh_models import tree_from_npz
    grid = base.Grid(64,'dirichlet','dirichlet')
    # Rebuild from the fresh bank itself, not only the saved reduced matrices.
    inputs = ROOT/'experiments/multiresolution-wave/runs/accel12/cluster/in/dirichlet'
    full = base.rebuild(inputs,grid)
    endpoint = tree_from_npz(ROOT/model['artifacts'][0])
    initializer = np.load(ROOT/model['artifacts'][1])
    transform = jnp.asarray(full['transform'])
    p = base.transform_head(endpoint['p'],transform)
    candidate = dict(full,p=p,frozen=endpoint['frozen'],
        common_inverse=jnp.linalg.pinv(transform@jnp.asarray(initializer['linear'])),
        common_center=transform@jnp.asarray(initializer['center']),
        fixed_codes=endpoint['codes'][np.linspace(0,len(endpoint['codes'])-1,6,dtype=int)])
    bank = previous.numerical_bank(candidate)
    initial = np.load(ROOT/model['replay']['initial'])
    reference = next(r for r in raw['references'] if (r['intervals'],r['case']) == (64,'opened_0'))
    assert array_sha(initial['u0']) == reference['input_sha256']['u']
    assert array_sha(initial['v0']) == reference['input_sha256']['v']
    supplied = tuple(jnp.asarray(a) for a in (initial['u0'],initial['v0'],initial['parameters'][5]))
    u,v,row,_ = ar.query('trained_nested40',raw['config']['selection_frozen']['dt'],
                       supplied,float(initial['parameters'][5]),grid,raw['config'],bank)
    assert row['completed'] and row['fit_stationary']
    return {'u':u,'v':v}, {k:row[k] for k in ('completed','fit_stationary','total_guard_fallbacks','minimum_dynamic_rank_ratio')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pde',choices=['poisson','heat','burgers','wave'])
    parser.add_argument('--output',type=Path,required=True,help='New JSON file for smoke evidence')
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
    assert jax.default_backend() == 'gpu' and jax.config.jax_enable_x64
    print('jax_backend=gpu x64=True precision=highest',flush=True)
    manifest = json.loads((ROOT/'consolidated/manifest.json').read_text())
    model = manifest['models'][args.pde]
    sys.path.insert(0,str(ROOT/model['cell']))
    raw = json.loads((ROOT/model['replay']['raw_result']).read_text())
    fields, diagnostics = globals()[args.pde](model,raw)
    saved = np.load(ROOT/model['replay']['expected'])
    errors = {}
    # This declared cross-device replay threshold is a consolidation check;
    # it does not replace any scientific accuracy or stationarity gate.
    tolerance = 1e-8
    for key,value in fields.items():
        value = np.asarray(value)
        assert value.dtype == np.float64 and np.isfinite(value).all()
        assert value.shape == saved[key].shape, (key,value.shape,saved[key].shape)
        relative = float(np.linalg.norm(value-saved[key])/np.linalg.norm(saved[key]))
        errors[key] = dict(relative_l2=relative, maximum_absolute=float(np.max(abs(value-saved[key]))),
                          replay_sha256=array_sha(value), expected_sha256=array_sha(saved[key]))
    passed = all(v['relative_l2'] <= tolerance for v in errors.values())
    evidence = dict(pde=args.pde, intervals=64, case=model['replay']['case'],passed=passed,
        scope='One opened saved case, complete frozen query, local smoke only; no timing or new accuracy claim',
        tolerance=tolerance,fields=errors,diagnostics=diagnostics,
        jax_backend=jax.default_backend(),jax=jax.__version__,x64=jax.config.jax_enable_x64,
        matmul_precision=str(jax.config.jax_default_matmul_precision),gpu=jax.devices()[0].device_kind,
        manifest_sha256=hashlib.sha256((ROOT/'consolidated/manifest.json').read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence),flush=True)
    assert passed, 'Saved-field parity failed; see evidence JSON'


if __name__ == '__main__':
    main()
