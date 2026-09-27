"""Paired reflective-wave geometry/timestep screen, preserving every invocation."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pilot as base
from pilot import jax, jnp
import iterative_replay as previous
import acceleration as fast
import modal_projection as modal
from fresh_fom import provenance
from fresh_models import head_geometry
from fresh_rom import weak_acceleration


def query(method, setting, supplied, c, grid, cfg, bank):
    if method.startswith('projected_') or method=='linear_bank64':
        return modal.device_query(method,bank,supplied,grid,cfg)
    if method.startswith('cgdt_'):
        dt=float(method.split('_')[1]);local_cfg=dict(cfg,primary_dt=dt)
        u,v,row,aux=previous.device_query(f'cg_{setting:g}',setting,supplied,c,grid,local_cfg,bank)
        row.update(method=method,fom_dt=dt)
        return u,v,row,aux
    if method in ('baseline', 'dst') or method.startswith('cg_'):
        name = cfg['primary_method'] if method == 'baseline' else method
        u, v, row, aux = previous.device_query(name, setting, supplied, c, grid, cfg, bank)
        row['method'] = method
        return u, v, row, aux
    t0 = time.perf_counter()
    z, w, fits, k, d = previous.initialize(bank, *supplied, iterations=cfg['fit_iterations'])
    jax.block_until_ready((z, w, fits, k, d)); t1 = time.perf_counter()
    steps, stride = round(cfg['end_time']/setting), round(cfg['observation_dt']/setting)
    rr = fast.rollout(bank['p'], bank['frozen'], z, w, k, d, setting,
                      variant=method, steps=steps, stride=stride)
    jax.block_until_ready(rr); t2 = time.perf_counter()
    up, vp, a, b = base.decode(bank['p'], bank['frozen'], rr['z'], rr['w'], bank['g'])
    up, vp = up.reshape(-1, *grid.shape), vp.reshape(-1, *grid.shape)
    jax.block_until_ready((up, vp)); t3 = time.perf_counter()
    up, vp = np.asarray(up), np.asarray(vp); transfer_end = time.perf_counter()
    aux = jax.tree.map(np.asarray, dict(fits=fits, rollout=rr, coefficients=a, velocity_coefficients=b))
    fits = aux['fits']; selected = int(fits['selected'])
    row = dict(method=method, setting=setting, boundary=grid.bx, intervals=grid.n,
        steps=steps, observation_stride=stride,
        seconds=dict(initialization_and_parameter_projection=t1-t0, evolution=t2-t1,
                     dense_device_output=t3-t2, complete_device_query=t3-t0),
        post_timer_host_transfer_seconds=transfer_end-t3,
        device_plus_output_transfer_seconds=transfer_end-t0,
        output_sha256=dict(u=base.array_sha(up), v=base.array_sha(vp)), output_bytes=up.nbytes+vp.nbytes,
        completed=bool(np.all(np.isfinite(up)) and np.all(np.isfinite(vp)) and np.all(aux['rollout']['completed'])),
        cold_fit={key: base.clean(value) for key, value in fits.items() if key not in ('projected_u', 'projected_v')},
        fit_stationary=bool(fits['finite'][selected] and fits['rank_ratio'][selected]>1e-8 and fits['gradient'][selected]<=1e-7
                           and (fits['stationarity'][selected]<=1e-6 or fits['objective'][selected]<=1e-20)),
        minimum_dynamic_rank_ratio=float(np.min(aux['rollout']['rank_ratio'])),
        rank_diagnostic_kind='Exact singular-value ratio' if method=='shared_svd_r' else 'Conservative lower bound, or exact ratio on fallback',
        total_guard_fallbacks=int(aux['rollout']['fallback_count'][-1]),
        maximum_normal_backward_error=float(aux['rollout']['normal_backward_error'][-1]), configuration_dimension=32)
    return up, vp, row, aux


def profile(bank, z, w, c, repetitions):
    p, f, k, d = bank['p'], bank['frozen'], c*c*bank['k'], c*bank['d']
    functions = dict(
        geometry_autodiff=jax.jit(lambda p, f, z, w, k, d: head_geometry(p, f, z, w, 'mlp')),
        geometry_shared=jax.jit(lambda p, f, z, w, k, d: fast.shared_geometry(p, f, z, w)),
        rhs_baseline=jax.jit(lambda p, f, z, w, k, d: weak_acceleration(p, f, z, w, k, d, 'mlp')))
    for variant in ('shared_svd_r', 'qr_guard', 'chol_guard'):
        functions['rhs_'+variant] = jax.jit(lambda p, f, z, w, k, d, variant=variant:
                                                   fast.accelerated_rhs(p, f, z, w, k, d, variant))
    _, _, jac, curve = fast.shared_geometry(p, f, z, w)
    functions2 = dict(qr=jax.jit(lambda x: jnp.linalg.qr(x, mode='reduced')),
        svd_j=jax.jit(fast.exact_rank), svd_r=jax.jit(lambda x: fast.exact_rank(jnp.linalg.qr(x, mode='reduced')[1])),
        gram_cholesky=jax.jit(lambda x: jnp.linalg.cholesky(x.T@x)))
    output = {}
    for name, fun in [*functions.items(), *functions2.items()]:
        args = (p, f, z, w, k, d) if name in functions else (jac,)
        jax.block_until_ready(fun(*args)); base.burn()
        timings = []
        for _ in range(repetitions):
            t = time.perf_counter(); jax.block_until_ready(fun(*args)); timings.append(time.perf_counter()-t)
        output[name] = dict(seconds=timings, median_seconds=float(np.median(timings)),
                            timer='Synchronized standalone component including dispatch; not additive fractions of fused RHS.')
    return output


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--inputs', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(); cfg = json.loads(args.config.read_text()); out = args.out
    out.mkdir(parents=True, exist_ok=False)
    meta = provenance(); print(json.dumps(meta), flush=True)
    assert meta['jax_backend']=='gpu' and meta['x64'] and meta['matmul_precision']=='highest'
    assert cfg['boundaries']==['dirichlet'] and not cfg['final_test_opened']
    frozen = json.loads((base.FRESH/'FROZEN-MATH.json').read_text())
    for name, expected in frozen['sha256'].items(): assert base.sha(base.FRESH/name)==expected
    data = dict(config=cfg, provenance=meta, frozen_mathematics=frozen,
        input_sha256={str(p.relative_to(args.inputs)):base.sha(p) for p in args.inputs.rglob('*') if p.is_file()},
        meshes=[], references=[], invocations=[], accuracy_controls=[], warmups=[], geometry_checks=[],
        profiles=[], parity=[], time_refinement=[], projection_kinematics=[], final_test_opened=False, complete=False)
    save = lambda: base.save_json(out/'result.json', data)
    for n in cfg['meshes']:
        grid = base.Grid(n, 'dirichlet', 'dirichlet')
        full, models = previous.load_models(args.inputs/'dirichlet', grid); bank = models[cfg['primary_method']]
        if any(a['method'].startswith('projected_') or a['method']=='linear_bank64' for a in cfg['arms']):
            bank=dict(bank,prepared_l2=modal.prepare(bank,'l2'),prepared_h1=modal.prepare(bank,'h1'))
        np.savez_compressed(out/f'mesh_dirichlet_{n}.npz', g=np.asarray(bank['g']), mass=np.asarray(bank['mass']),
            stiffness=np.asarray(bank['k']), damping=np.asarray(bank['d']), transform=full['transform'])
        data['meshes'].append(dict(intervals=n, audits=full['audits'], assembly_seconds=full['assembly_seconds_including_first_compile']))
        for cohort in cfg['cohorts']:
          pars = base.parameter_rows(cohort['seed'], max(cohort['indices'])+1)
          for ci in cohort['indices']:
            par = pars[ci]; case = f"{cohort['name']}_{ci}"
            u0, v0 = base.localized_initial(grid, par); supplied = (u0, v0, jnp.asarray(par[5])); jax.block_until_ready(supplied)
            su, sv, _, _ = query('dst', 0., supplied, float(par[5]), grid, cfg, bank)
            np.savez_compressed(out/f'reference_{n}_{case}.npz', u=su, v=sv, u0=np.asarray(u0), v0=np.asarray(v0), parameters=par)
            data['references'].append(dict(intervals=n, case=case, seed=cohort['seed'], parameters=par,
                input_sha256=dict(u=base.array_sha(u0), v=base.array_sha(v0)), reference_kind='exact semidiscrete DST'))
            initial = previous.initialize(bank, *supplied, iterations=cfg['fit_iterations'])
            data['geometry_checks'].append(dict(intervals=n, case=case, **fast.geometry_checks(bank, initial[0], initial[1])))
            if cfg.get('profile_repetitions', 0):
                data['profiles'].append(dict(intervals=n, case=case, components=profile(bank, initial[0], initial[1], supplied[2], cfg['profile_repetitions'])))
            methods = [(arm['method'], arm['dt']) for arm in cfg['arms']]
            methods += [(f'cg_{tol:g}', tol) for tol in cfg['cg_tolerances']]
            methods += [(f"cgdt_{arm['dt']:g}_tol_{arm['tolerance']:g}",arm['tolerance']) for arm in cfg.get('cg_timestep_arms',[])] + [('dst', 0.)]
            first = {}; outputs = {}
            for name, dt in methods:
                print('warmup', n, case, name, dt, flush=True)
                _, _, row, _ = query(name, dt, supplied, float(par[5]), grid, cfg, bank)
                data['warmups'].append(dict(case=case, **row))
            for rep in range(cfg['repetitions']):
                for order, (name, dt) in enumerate(methods if rep%2==0 else methods[::-1]):
                    print('timed', n, case, rep, name, dt, flush=True); base.burn()
                    u, v, row, aux = query(name, dt, supplied, float(par[5]), grid, cfg, bank)
                    ident = f'{n}_{case}_{name}_{dt:g}'
                    row.update(case=case, cohort=cohort['name'], seed=cohort['seed'], parameters=par,
                        repetition=rep, order=order, invocation_id=f'{ident}_{rep}', comparison_eligible=True,
                        same_grid_discrepancy=base.metrics(u, v, su, sv, grid, par[5], cfg), field_artifact=ident+'.npz')
                    if rep==0:
                        arrays = dict(u=u, v=v)
                        if 'fits' in aux:
                            arrays.update(coefficients=aux['coefficients'], velocity_coefficients=aux['velocity_coefficients'])
                            arrays.update({'rollout_'+key:value for key, value in aux['rollout'].items()})
                        if 'projection' in aux:
                            for key,value in aux['projection'].items():
                                if key=='fits':arrays.update({'projection_fit_'+k:v for k,v in value.items()})
                                else:arrays['projection_'+key]=value
                        np.savez_compressed(out/(ident+'.npz'), **arrays)
                        first[ident] = row['output_sha256']; outputs[(name, dt)] = ident+'.npz'
                    else:
                        assert first[ident]==row['output_sha256'], 'Nondeterministic output requires unique artifact'
                    row['artifact_relation'] = 'Actual first timed output' if rep==0 else 'Byte-identical full timed fields, both hashes checked'
                    data['invocations'].append(row); save()
            for arm in cfg['arms']:
                name=arm['method']
                if not name.startswith('projected_'):continue
                with np.load(out/outputs[(name,0.)]) as f:
                    aux={key:f['projection_'+key] for key in ('a0','b0','scale','z','velocity_coefficients')}
                print('projection_kinematics',n,case,name,flush=True)
                times=jnp.arange(round(cfg['end_time']/cfg['observation_dt'])+1)*cfg['observation_dt']
                checks,arrays=modal.kinematic_check(bank,bank['prepared_'+name.removeprefix('projected_')],aux,supplied[2],times,cfg['fit_iterations'])
                artifact=f'kinematics_{n}_{case}_{name}.npz';np.savez_compressed(out/artifact,**arrays)
                data['projection_kinematics'].append(dict(intervals=n,case=case,method=name,checks=checks,artifact=artifact,
                    passed=all(x['all_neighbor_fits_stationary'] and x['max_initial_velocity_scaled_difference']<=1e-3 for x in checks)))
                save()
            with np.load(out/outputs[('baseline', cfg['primary_dt'])]) as f:
                bu, bv = f['u'], f['v']
            for arm in cfg['arms']:
                if arm['method']=='baseline' or arm['dt']!=cfg['primary_dt']: continue
                with np.load(out/outputs[(arm['method'], arm['dt'])]) as f:
                    pm = base.metrics(f['u'], f['v'], bu, bv, grid, par[5], cfg)
                maxima = {key:float(np.max(pm[key]['absolute'])/data['invocations'][-1]['same_grid_discrepancy'][key]['initial_scale']) for key in ('displacement','velocity','energy_state')}
                data['parity'].append(dict(intervals=n, case=case, method=arm['method'], maxima=maxima,
                                          passed=max(maxima.values())<cfg['parity_target']))
            for arm in cfg['refinement_arms']:
                name, dt = arm['method'], arm['dt']; fine_dt = dt/2
                print('refinement', n, case, name, dt, flush=True); base.burn()
                u, v, row, aux = query(name, fine_dt, supplied, float(par[5]), grid, cfg, bank)
                ident = f'{n}_{case}_{name}_{dt:g}_refined'
                row.update(case=case, cohort=cohort['name'], seed=cohort['seed'], parameters=par,
                    comparison_eligible=False, invocation_id=ident, field_artifact=ident+'.npz',
                    same_grid_discrepancy=base.metrics(u,v,su,sv,grid,par[5],cfg))
                np.savez_compressed(out/(ident+'.npz'),u=u,v=v,coefficients=aux['coefficients'],velocity_coefficients=aux['velocity_coefficients'],
                    **{'rollout_'+key:value for key,value in aux['rollout'].items()})
                with np.load(out/outputs[(name,dt)]) as f: pm=base.metrics(f['u'],f['v'],u,v,grid,par[5],cfg)
                maxima = {key:float(np.max(pm[key]['absolute'])/row['same_grid_discrepancy'][key]['initial_scale']) for key in ('displacement','velocity','energy_state')}
                data['time_refinement'].append(dict(intervals=n,case=case,method=name,dt=dt,maxima=maxima,passed=max(maxima.values())<=cfg['rom_refinement_target']))
                data['accuracy_controls'].append(row); save()
            del su,sv,bu,bv,u,v,initial
        del full,models,bank; jax.clear_caches()
    data['complete']=True; data['output_sha256']={p.name:base.sha(p) for p in out.glob('*.npz')};save()
    print('wave_acceleration_complete', flush=True)


if __name__=='__main__': main()
