"""Fresh wave comparison with supplied/returned full fields resident on GPU.

Uses only independently verified post-reset mathematics. Each timing record is
scored from the actual timed output; duplicate repetitions retain their hashes
and share a full-field artifact only after byte-identical equality is verified.
"""
import argparse
from functools import partial
import json
from pathlib import Path
import time
import numpy as np
import pilot as base
from pilot import jax, jnp, Grid, localized_initial, parameter_rows, metrics
from fresh_models import tree_from_npz
from fresh_fom import provenance
from heads32 import adapt_model
from iterative_paths import implicit_wave, verify


@partial(jax.jit, static_argnames=('iterations',))
def initialize(bank, u, v, speed, *, iterations):
    z, w, fits = base.cold_fit(bank, u, v, iterations)
    return z, w, fits, speed**2*bank['k'], speed*bank['d']


def numerical_bank(bank):
    return {key: bank[key] for key in ('g', 'mass', 'p', 'frozen', 'k', 'd',
                                     'common_inverse', 'common_center', 'fixed_codes')}


def device_query(method, setting, inputs, speed_host, grid, cfg, bank=None):
    """Inputs are already resident and ready; no host field is used online."""
    u, v, speed = inputs
    aux = {}
    t0 = time.perf_counter()
    if method.startswith('frozen_mlp'):
        z, w, fits, k, d = initialize(bank, u, v, speed,
                                     iterations=cfg['fit_iterations'])
        jax.block_until_ready((z, w, fits, k, d))
        t1 = time.perf_counter()
        stride = int(round(cfg['observation_dt']/setting))
        steps = int(round(cfg['end_time']/setting))
        rr = base.rollout(bank['p'], bank['frozen'], z, w, k, d, setting,
                          kind='mlp', steps=steps, stride=stride)
        jax.block_until_ready(rr)
        t2 = time.perf_counter()
        up, vp, a, b = base.decode(bank['p'], bank['frozen'], rr['z'], rr['w'], bank['g'])
        up, vp = up.reshape(-1, *grid.shape), vp.reshape(-1, *grid.shape)
        jax.block_until_ready((up, vp))
        t3 = time.perf_counter()
        aux = dict(fits=fits, rollout=rr, coefficients=a, velocity_coefficients=b)
    else:
        if method.startswith('cg_'):
            stride = int(round(cfg['observation_dt']/cfg['primary_dt']))
            steps = int(round(cfg['end_time']/cfg['primary_dt']))
            t1 = time.perf_counter()
            up, vp, cg_info = implicit_wave(u, v, speed, cfg['primary_dt'], setting, grid=grid, steps=steps, stride=stride, max_iterations=cfg['cg_max_iterations'])
            aux = dict(cg_info=cg_info)
        elif method == 'dst':
            times = jnp.arange(int(round(cfg['end_time']/cfg['observation_dt']))+1,
                               dtype=jnp.float64)*cfg['observation_dt']
            stride, steps = 0, 0
            t1 = time.perf_counter()
            up, vp = base.spectral_propagate(u, v, speed, times)
        else:
            stride = int(np.ceil(cfg['observation_dt']/(setting*grid.h/speed_host)))
            steps = int(round(cfg['end_time']/cfg['observation_dt']))*stride
            dt = cfg['observation_dt']/stride
            t1 = time.perf_counter()
            up, vp = base.integrate(u, v, speed, dt, grid=grid, steps=steps, stride=stride)
        jax.block_until_ready((up, vp, aux))
        t2 = t3 = time.perf_counter()
    # Both requested fields are ready before this boundary. No field transfer
    # occurs until after t3; diagnostics also transfer outside the primary timer.
    up, vp = np.asarray(up), np.asarray(vp)
    transfer_end = time.perf_counter()
    aux = jax.tree.map(np.asarray, aux)
    record = dict(method=method, setting=setting, boundary=grid.bx, intervals=grid.n,
                  steps=steps, observation_stride=stride,
                  seconds=dict(initialization_and_parameter_projection=t1-t0,
                               evolution=t2-t1, dense_device_output=t3-t2,
                               complete_device_query=t3-t0),
                  post_timer_host_transfer_seconds=transfer_end-t3,
                  device_plus_output_transfer_seconds=transfer_end-t0,
                  output_sha256=dict(u=base.array_sha(up), v=base.array_sha(vp)),
                  output_bytes=up.nbytes+vp.nbytes,
                  completed=bool(np.all(np.isfinite(up)) and np.all(np.isfinite(vp))))
    if 'cg_info' in aux:
        info=aux['cg_info']
        record.update(cg_iterations=info[:,0].astype(int).tolist(), cg_recursive_relative=info[:,1].tolist(), cg_true_relative=info[:,2].tolist(), cg_all_converged=bool(np.all(info[:,2]<=1.01*setting)), cg_cap_exits=int(np.sum(info[:,0]>=cfg['cg_max_iterations'])))
    if 'fits' in aux:
        fits = aux['fits']; selected = int(fits['selected'])
        record.update(cold_fit={k: base.clean(val) for k, val in fits.items()
                                if k not in ('projected_u', 'projected_v')},
                      fit_stationary=bool(fits['finite'][selected] and fits['rank_ratio'][selected]>1e-8
                          and fits['gradient'][selected]<=1e-7
                          and (fits['stationarity'][selected]<=1e-6 or fits['objective'][selected]<=1e-20)),
                      minimum_dynamic_rank_ratio=float(np.min(aux['rollout']['rank_ratio'])),
                      configuration_dimension=bank['p']['linear'].shape[1])
        record['completed'] &= bool(np.all(aux['rollout']['completed']))
    return up, vp, record, aux


def load_models(inputs, grid):
    bank = base.rebuild(inputs, grid)
    with np.load(inputs/'initializer32.npz') as f:
        linear, center = f['linear'], f['center']
    head = tree_from_npz(inputs/'head32.npz')
    model32 = adapt_model(bank, head, linear, center, 'frozen_mlp32_seed691200')
    models = {'frozen_mlp16_seed691200': numerical_bank(bank),
              'frozen_mlp32_seed691200': numerical_bank(model32)}
    return bank, models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    out = args.out; out.mkdir(parents=True, exist_ok=False)
    meta = provenance(); print(json.dumps(meta), flush=True)
    assert meta['jax_backend']=='gpu' and meta['x64'] and meta['matmul_precision']=='highest'
    assert not cfg['final_test_opened']
    frozen = json.loads((base.FRESH/'FROZEN-MATH.json').read_text())
    for name, expected in frozen['sha256'].items():
        assert base.sha(base.FRESH/name)==expected
    result = dict(config=cfg, provenance=meta, frozen_mathematics=frozen,
                  input_sha256={str(p.relative_to(args.inputs)): base.sha(p)
                                for p in args.inputs.rglob('*') if p.is_file()},
                  operator_verification=verify(), meshes=[], references=[], invocations=[], accuracy_controls=[], warmups=[],
                  time_refinement=[], complete=False, final_test_opened=False)
    save = lambda: base.save_json(out/'result.json', result)
    pars = parameter_rows(cfg['validation_seed'], max(cfg['validation_indices'])+1)
    for bc in cfg['boundaries']:
        for n in cfg['meshes']:
            grid = Grid(n, bc, bc)
            bank, models = load_models(args.inputs/bc, grid)
            result['meshes'].append(dict(boundary=bc, intervals=n, audits=bank['audits'],
                assembly_seconds_including_first_compile=bank['assembly_seconds_including_first_compile'],
                storage_bytes=bank['storage_bytes']))
            np.savez_compressed(out/f'mesh_{bc}_{n}.npz', g=np.asarray(bank['g']),
                mass=np.asarray(bank['mass']), stiffness=np.asarray(bank['k']),
                damping=np.asarray(bank['d']), transform=bank['transform'])
            for ci in cfg['validation_indices']:
                par = pars[ci]
                u0, v0 = localized_initial(grid, par)
                supplied = (u0, v0, jnp.asarray(par[5]))
                jax.block_until_ready(supplied)
                fm = 'dst' if bc=='dirichlet' else 'rk4'
                fs = 0. if bc=='dirichlet' else cfg['fom_cfl']
                su, sv, _, _ = device_query(fm, 0. if bc=='dirichlet' else cfg['reference_cfl'],
                                            supplied, float(par[5]), grid, cfg)
                refrecord = dict(boundary=bc, intervals=n, case=ci, parameters=par,
                    seed=cfg['validation_seed'], input_sha256=dict(u=base.array_sha(u0), v=base.array_sha(v0)),
                    reference_kind='exact semidiscrete DST' if bc=='dirichlet' else 'same-grid RK4 with temporal self-refinement')
                if bc=='absorbing':
                    np.savez_compressed(out/f'reference_coarse_{bc}_{n}_{ci}.npz', u=su, v=sv)
                    fu, fv, _, _ = device_query('rk4', cfg['reference_refinement_cfl'], supplied,
                                                 float(par[5]), grid, cfg)
                    refinement = metrics(su, sv, fu, fv, grid, par[5], cfg)
                    maximum = max(refinement[k]['max_initial_normalized']
                                  for k in ('displacement', 'velocity', 'energy_state'))
                    refrecord.update(temporal_refinement=refinement,
                                     refinement_passed=maximum<=cfg['reference_refinement_target'])
                    # Use the finer field as the actual scoring reference.
                    su, sv = fu, fv
                else:
                    refrecord['refinement_passed'] = True
                np.savez_compressed(out/f'reference_{bc}_{n}_{ci}.npz', u=su, v=sv,
                                     u0=np.asarray(u0), v0=np.asarray(v0), parameters=par)
                result['references'].append(refrecord); save()
                methods = [(cfg['primary_method'], cfg['primary_dt'])]+[(f'cg_{tol:g}',tol) for tol in cfg['cg_tolerances']]+[(fm,fs)]
                def call(name, step):
                    return device_query(name, step, supplied, float(par[5]), grid, cfg, models.get(name))
                for name, step in methods:
                    print('warmup', bc, n, ci, name, flush=True)
                    _, _, row, _ = call(name, step)
                    result['warmups'].append(dict(case=ci, **row))
                first = {}
                for rep in range(cfg['repetitions']):
                    for order, (name, step) in enumerate(methods if rep%2==0 else list(reversed(methods))):
                        print('timed', bc, n, ci, rep, name, flush=True); base.burn()
                        u, v, row, aux = call(name, step)
                        ident = f'{bc}_{n}_{ci}_{name}'
                        row.update(case=ci, repetition=rep, order=order, invocation_id=f'{ident}_{rep}',
                                   comparison_eligible=True, measurement_role='timed_comparison',
                                   same_grid_discrepancy=metrics(u, v, su, sv, grid, par[5], cfg))
                        if rep==0:
                            arrays = dict(u=u, v=v)
                            if 'fits' in aux:
                                arrays.update(coefficients=aux['coefficients'], velocity_coefficients=aux['velocity_coefficients'])
                                arrays.update({'rollout_'+k: val for k, val in aux['rollout'].items()})
                            np.savez_compressed(out/f'{ident}.npz', **arrays)
                            first[name] = row['output_sha256']
                        else:
                            if row['output_sha256'] != first[name]:
                                raise RuntimeError('Nondeterministic repetition: preserve before extending deduplication')
                        row['field_artifact'] = f'{ident}.npz'
                        row['artifact_relation'] = 'Actual first timed output' if rep==0 else 'Byte-identical timed output verified by both complete-field hashes'
                        result['invocations'].append(row); save()
                for name, _ in methods[:1]:
                    print('accuracy_refinement', bc, n, ci, name, flush=True); base.burn()
                    u, v, row, aux = call(name, cfg['accuracy_only_dt'])
                    ident = f'{bc}_{n}_{ci}_{name}_refined'
                    row.update(case=ci, repetition=0, invocation_id=ident, comparison_eligible=False,
                               measurement_role='accuracy_refinement_only',
                               timing_note='May include first compilation; ineligible for speed claims.',
                               same_grid_discrepancy=metrics(u, v, su, sv, grid, par[5], cfg),
                               field_artifact=f'{ident}.npz')
                    np.savez_compressed(out/f'{ident}.npz', u=u, v=v, coefficients=aux['coefficients'],
                                        velocity_coefficients=aux['velocity_coefficients'])
                    with np.load(out/f'{bc}_{n}_{ci}_{name}.npz') as f:
                        refinement = metrics(f['u'], f['v'], u, v, grid, par[5], cfg)
                    # Same initial physical scales as the scoring reference, not each ROM's fitted initial state.
                    differences = {k: float(np.max(np.asarray(refinement[k]['absolute']) /
                        row['same_grid_discrepancy'][k]['initial_scale'])) for k in ('displacement', 'velocity', 'energy_state')}
                    result['time_refinement'].append(dict(boundary=bc, intervals=n, case=ci, method=name,
                        differences=refinement, maxima_on_reference_initial_scales=differences,
                        passed=row['completed'] and max(differences.values())<=cfg['rom_refinement_target']))
                    result['accuracy_controls'].append(row); save()
            del bank, models; jax.clear_caches()
    assert len(result['invocations'])==cfg['expected_timed_invocations']
    assert len(result['accuracy_controls'])==cfg['expected_accuracy_only_queries']
    result['output_sha256'] = {p.name: base.sha(p) for p in out.glob('*.npz')}
    result['complete'] = True; save(); print('fresh_wave_iterative_multires_complete', flush=True)


if __name__=='__main__':
    main()
