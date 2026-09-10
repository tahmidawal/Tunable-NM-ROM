"""Frozen-decoder validation and sealed evaluation on one GPU allocation.

All validation cases receive an accuracy/timing invocation for every setting.
Case zero supplies seven additional, explicitly separate selection-proxy timings.
Only frozen selected settings enter seven-repeat evaluation on untouched cases.
"""
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import pickle
import shutil
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import numpy as np
from common.decoders import DecoderConfig, decode_points, decode_grid
from physics import (Grid, localized_initial, parameter_rows, cn_rollout, spectral_propagate,
                     integrate, metrics, provenance, smooth_tests, damping_ratio, energy)
from weak import build_rule, numerical_rule, make_query, moments
from data import reference, save_json, clean, array_sha
from seal import verify_validation_bundle
from fields import save_fields, save_reference


@jax.jit
def _burn(a):
    return jnp.tanh((a@a)*.001)


def burn():
    a = jnp.full((1024, 1024), .01, dtype=jnp.float64)
    a = _burn(a)
    a.block_until_ready()
    start = time.perf_counter()
    while time.perf_counter()-start < .3:
        a = _burn(a)
        a.block_until_ready()


def load_models(inputs, cfg, boundary):
    training_manifest = json.loads((inputs/'training_result.json').read_text())
    if not training_manifest.get('complete') or training_manifest['boundary'] != boundary:
        raise RuntimeError('Training result is incomplete or has the wrong physical boundary')
    for filename in ('decoders.py', 'training.py', 'lm.py'):
        expected = training_manifest['source_sha256']['common/'+filename]
        actual = hashlib.sha256((Path(__file__).parents[1]/'common'/filename).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError('Shared model/solver source differs from training provenance: '+filename)
    models = {}
    for name in ('cp', 'modcp', 'film'):
        path = inputs/f'{name}.pkl'
        with path.open('rb') as f:
            checkpoint = pickle.load(f)
        extra = checkpoint['extra']
        expected = cfg['pretrain_steps']+cfg['comparison_steps']
        if not extra.get('complete') or extra.get('total_training_steps') != expected:
            raise RuntimeError(f'Partial or wrong-budget checkpoint: {path}')
        dc = DecoderConfig(**checkpoint['config'])
        if (dc.architecture != name or dc.k != cfg['latent'] or dc.rank != cfg['rank'] or
                dc.outputs != 2 or dc.intervals != cfg['train_intervals'] or
                dc.boundary != ('dirichlet' if boundary == 'dirichlet' else 'free') or
                extra.get('training_seed') != cfg['train_seed'] or extra.get('model_seed') != cfg['model_seed']):
            raise RuntimeError('Checkpoint architecture differs from frozen comparison')
        nt = extra['nt']
        initial_codes = np.asarray(checkpoint['Z'])[::nt]
        picked = np.unique(np.linspace(0, len(initial_codes)-1, cfg['initial_fit_starts']-1, dtype=int))
        starts = np.concatenate((np.zeros((1, dc.k)), initial_codes[picked]))
        models[name] = {'params': jax.tree.map(jnp.asarray, checkpoint['params']), 'codes': np.asarray(checkpoint['Z']),
                        'config': dc, 'scales': jnp.asarray(extra['fixed_component_scales']),
                        'starts': jnp.asarray(starts), 'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'training_data_sha256': extra['training_data_sha256']}
    scales = [np.asarray(m['scales']) for m in models.values()]
    if not all(np.array_equal(scales[0], s) for s in scales[1:]):
        raise RuntimeError('Architecture arms disagree on training-only component scales')
    if len({m['training_data_sha256'] for m in models.values()}) != 1:
        raise RuntimeError('Architecture arms were trained on different datasets')
    return models


def error_maxima(score):
    return {key: (float(np.max(np.asarray(score[key]))) if np.all(np.isfinite(np.asarray(score[key]))) else None)
            for key in ('displacement', 'velocity', 'energy_state')}


def configurations(cfg, boundary):
    result = []
    for name in ('cp', 'modcp', 'film'):
        for mul, cap, tol, dt in itertools.product(cfg['eq_multipliers'], cfg['gn_caps'], cfg['gn_tolerances'], cfg['time_steps']):
            result.append({'method': name, 'multiplier': mul, 'cap': cap, 'tol': tol, 'dt': dt,
                           'id': f'{name}_eq{mul}_cap{cap}_tol{tol:g}_dt{dt:g}'})
    for tol, dt in itertools.product(cfg['cg_tolerances'], cfg['time_steps']):
        result.append({'method': 'cg', 'tol': tol, 'dt': dt, 'id': f'cg_tol{tol:g}_dt{dt:g}'})
    for cfl in cfg['rk4_cfls']:
        result.append({'method': 'rk4', 'cfl': cfl, 'id': f'rk4_cfl{cfl:g}'})
    if boundary == 'dirichlet':
        result.append({'method': 'spectral', 'id': 'spectral_exact_semidiscrete'})
        for dt in cfg['time_steps']:
            result.append({'method': 'cn_direct', 'dt': dt, 'tol': 1e-12, 'id': f'cn_direct_dt{dt:g}'})
    return result


def load_reusable_rule(inputs, out, name, n, multiplier, model, cfg, boundary):
    old = inputs/'eq_reuse'
    folder = old/'quadrature'/f'{name}_{n}_{multiplier}'
    if not (folder/'rule.npz').exists():
        return None
    previous = json.loads((old/'handoff.json').read_text())
    keys = ('train_intervals', 'train_seed', 'model_seed', 'weak_modes', 'eq_fit_codes', 'eq_candidate_count')
    if (any(previous['config'][key] != cfg[key] for key in keys) or
            previous['checkpoint_sha256'][name] != model['checkpoint_sha256'] or
            previous['case_name'] != ('wave_reflective' if boundary == 'dirichlet' else 'wave_absorbing')):
        raise RuntimeError('Offline EQ reuse is incompatible with this frozen checkpoint/configuration')
    audit = json.loads((folder/'audit.json').read_text())
    audit['volume'].setdefault('nnls_method', 'direct')
    for face in audit['faces']:
        face.setdefault('nnls_method', 'direct')
    if (audit['architecture'] != name or audit['intervals'] != n or audit['weak_modes'] != cfg['weak_modes'] or
            audit['volume_target'] != cfg['weak_modes']*multiplier):
        raise RuntimeError('Offline EQ rule metadata mismatch')
    with np.load(folder/'rule.npz') as a:
        raw = {key: a[key] for key in a.files}
    target = out/'quadrature'/folder.name
    target.mkdir(parents=True, exist_ok=False)
    shutil.copy2(folder/'rule.npz', target/'rule.npz')
    audit['reused_offline_rule'] = {'source_job_id': previous['provenance']['job_id'],
                                    'source_commit': previous['provenance']['commit'],
                                    'rule_sha256': hashlib.sha256((folder/'rule.npz').read_bytes()).hexdigest(),
                                    'timing_or_selection_reused': False}
    save_json(target/'audit.json', audit)
    print('REUSE_OFFLINE_EQ_ONLY', name, n, multiplier, flush=True)
    return raw, audit


class Queries:
    def __init__(self, cfg, grid, models, rules):
        self.cfg, self.grid, self.models, self.rules = cfg, grid, models, rules
        self.functions = {}
        self.metric = jax.jit(lambda u, v, ut, vt, c: metrics(u, v, ut, vt, grid, c))
        self.times = jnp.arange(int(round(cfg['end_time']/cfg['observation_dt']))+1)*cfg['observation_dt']

    def query(self, setting, supplied):
        u0, v0, speed = supplied
        method, cfg, grid = setting['method'], self.cfg, self.grid
        aux = {}
        if method in self.models:
            model = self.models[method]
            key = (method, setting['cap'], setting['dt'])
            if key not in self.functions:
                initial, evolve, decode, residual = make_query(model['config'], grid, cfg, setting['cap'], setting['dt'])
                def fused(params, rule, u0, v0, speed, starts, scales, tolerance):
                    z0, fit = initial(params, rule, u0, v0, starts, scales)
                    z, diagnostics = evolve(params, rule, z0, speed, scales, tolerance)
                    u, v = decode(params, z)
                    return u, v, z, fit, diagnostics
                self.functions[key] = (initial, evolve, decode, residual, jax.jit(fused))
            initial, evolve, decode, _, fused = self.functions[key]
            rule = self.rules[(method, setting['multiplier'])]
            t0 = time.perf_counter()
            u, v, z, fit, diagnostics = fused(model['params'], rule, u0, v0, speed, model['starts'], model['scales'], jnp.asarray(setting['tol']))
            jax.block_until_ready((u, v, z, fit, diagnostics)); t3 = time.perf_counter()
            t1 = t2 = t0
            selected = int(fit['selected'])
            reason = np.asarray(diagnostics[1])
            chosen = int(fit['reason'][selected])
            finite_horizon = bool(np.all(np.asarray(diagnostics[4]))) and chosen not in (3, 4, 5)
            stationary = chosen == 1 and bool(np.all(reason == 1))
            aux = {'codes': z, 'fit': fit, 'step_diagnostics': diagnostics}
            stop = {'initial': chosen, 'steps': dict(Counter(int(x) for x in reason))}
        else:
            t0 = t1 = time.perf_counter()
            if method in ('cg', 'cn_direct'):
                dt = setting['dt']
                stride = int(round(cfg['observation_dt']/dt))
                steps = int(round(cfg['end_time']/dt))
                result = cn_rollout(u0, v0, speed, dt, setting['tol'], grid=grid, steps=steps,
                                    stride=stride, maxiter=cfg['cg_maxiter'], direct=method == 'cn_direct')
                u, v = result['u'], result['v']
                jax.block_until_ready(result); t2 = t3 = time.perf_counter()
                finite_horizon = bool(jnp.all(result['completed']))
                stationary = finite_horizon
                stop = {'cg_max_true_relative_residual': float(jnp.max(result['true_relative_residual'])),
                        'cg_max_iterations': int(jnp.max(result['iterations'])),
                        'cg_total_iterations': int(jnp.sum(result['iterations']))}
                aux = result
            elif method == 'spectral':
                u, v = spectral_propagate(u0, v0, speed, self.times)
                jax.block_until_ready((u, v)); t2 = t3 = time.perf_counter()
                finite_horizon = stationary = True
                stop = {'spectral_exact_semidiscrete': True}
            else:
                # Largest family speed fixes one safe time-step plan offline;
                # no input-dependent host transfer is hidden inside query timing.
                stride = int(np.ceil(cfg['observation_dt']*1.15/(setting['cfl']*grid.h)))
                dt = cfg['observation_dt']/stride
                steps = int(round(cfg['end_time']/cfg['observation_dt']))*stride
                u, v = integrate(u0, v0, speed, dt, grid=grid, steps=steps, stride=stride)
                jax.block_until_ready((u, v)); t2 = t3 = time.perf_counter()
                finite_horizon = stationary = True
                stop = {'time_step': dt, 'steps': steps}
        finite = bool(jnp.all(jnp.isfinite(u)) & jnp.all(jnp.isfinite(v)))
        row = {'method': method, 'configuration': setting['id'], 'intervals': grid.n,
               'seconds': t3-t0, 'cost_breakdown': {'initialization': t1-t0, 'evolution': t2-t1, 'reconstruction': t3-t2},
               'finite': finite, 'completed': finite and finite_horizon, 'stationary': stationary,
               'stop_reason': clean(stop)}
        if method in self.models:
            row['cost_breakdown'] = {'complete_fused_query': t3-t0}
        return u, v, row, aux

    def component_timing(self, setting, supplied):
        """Separate warmed diagnostics; never compose the primary query time."""
        model = self.models[setting['method']]
        initial, evolve, decode, _, _ = self.functions[(setting['method'], setting['cap'], setting['dt'])]
        rule = self.rules[(setting['method'], setting['multiplier'])]
        u0, v0, speed = supplied
        recorded = []
        for rep in range(2):
            if rep:
                burn()
            t0 = time.perf_counter()
            z0, fit = initial(model['params'], rule, u0, v0, model['starts'], model['scales'])
            jax.block_until_ready((z0, fit)); t1 = time.perf_counter()
            z, diagnostics = evolve(model['params'], rule, z0, speed, model['scales'], jnp.asarray(setting['tol']))
            jax.block_until_ready((z, diagnostics)); t2 = time.perf_counter()
            u, v = decode(model['params'], z)
            jax.block_until_ready((u, v)); t3 = time.perf_counter()
            if rep:
                recorded.append({'initialization': t1-t0, 'evolution': t2-t1, 'reconstruction': t3-t2})
        return {'configuration': setting['id'], 'intervals': self.grid.n,
                'diagnostic_only_staged_calls': recorded, 'excluded_from_primary_timing': True}


def select(rows, proxies, settings, grid, cfg):
    """No evaluation data enter this deterministic validation-only selection."""
    selections, chosen = [], set()
    methods = sorted({s['method'] for s in settings})
    for method in methods:
        candidates = []
        for setting in [s for s in settings if s['method'] == method]:
            data = [r for r in rows if r['configuration'] == setting['id'] and r['intervals'] == grid.n]
            proxy = [r for r in proxies if r['configuration'] == setting['id'] and r['intervals'] == grid.n]
            costs = [r['seconds'] for r in proxy]
            if len(data) != cfg['validation_count'] or len(costs) != cfg['repetitions']:
                raise RuntimeError('Validation membership/repetition count incomplete')
            complete = all(r['completed'] for r in data+proxy)
            error = max(max(r['errors'].values()) if all(x is not None for x in r['errors'].values()) else float('inf') for r in data+proxy)
            candidates.append((setting['id'], complete, error, float(np.median(costs))))
        for target in cfg['targets']:
            eligible = [r for r in candidates if r[1] and r[2] <= target]
            best = min(eligible, key=lambda r: (r[3], r[2], r[0])) if eligible else None
            if best:
                chosen.add(best[0])
            selections.append({'method': method, 'intervals': grid.n, 'target': target,
                               'configuration': None if best is None else best[0], 'validation_passed': best is not None,
                               'selection_cost_source': 'predetermined_validation_case_zero_seven_repetition_proxy'})
        # If no target passes, still evaluate the best measured-accuracy point as
        # a clearly labelled diagnostic, never silently omit a losing method.
        finite = [r for r in candidates if r[1] and np.isfinite(r[2])]
        best = min(finite, key=lambda r: (r[2], r[3], r[0])) if finite else min(candidates, key=lambda r: (r[2], r[0]))
        chosen.add(best[0])
        selections.append({'method': method, 'intervals': grid.n, 'target': None,
                           'configuration': best[0], 'validation_passed': best[1], 'diagnostic': 'best_validation_accuracy'})
    return selections, [s for s in settings if s['id'] in chosen]


def audit_full_weak(aux, model, raw_rule, setting, grid, speed, cfg):
    """Untimed full-grid weak audit on actual latent states, without rollout fixes."""
    if 'codes' not in aux:
        return {}
    z = np.asarray(aux['codes'])
    indices = np.unique(np.linspace(1, len(z)-1, min(4, len(z)-1), dtype=int))
    stride = int(round(cfg['observation_dt']/setting['dt']))
    previous = np.asarray(aux['step_diagnostics'][5])[indices*stride-1]
    phi, _, _ = smooth_tests(grid, raw_rule['test'].shape[1])
    decode = jax.jit(lambda p, zz: jax.lax.map(lambda a: decode_grid(p, a, grid.n, model['config']), zz))
    fields = np.asarray(decode(model['params'], jnp.asarray(np.concatenate((previous, z[indices])))))
    if grid.bx == 'dirichlet':
        fields = fields[:, 1:-1, 1:-1]
    fields = fields.reshape(2*len(indices), -1, 2)
    full = np.einsum('pm,spc,p->smc', phi, fields, grid.mass().ravel(), optimize=True)
    eq = np.einsum('pm,spc,p->smc', raw_rule['test'], fields[:, raw_rule['active_ids']], raw_rule['weight'], optimize=True)
    # Normalize absolute moment discrepancy by fixed training physical scales,
    # retaining near-zero absorbing late-time rows without division by cancellation.
    normalized = np.linalg.norm(full-eq, axis=1)/np.asarray(model['scales'])
    damping = np.asarray(damping_ratio(grid, 1.)).ravel()
    full_face = np.einsum('pm,spc,p->smc', phi, fields, grid.mass().ravel()*damping, optimize=True)
    if len(raw_rule['face_xy']):
        ij = np.rint(raw_rule['face_xy']*grid.n).astype(int)
        face_ids = ij[:, 0]*grid.shape[1]+ij[:, 1]
        eq_face = np.einsum('pm,spc,p->smc', raw_rule['face_test'], fields[:, face_ids], raw_rule['face_weight'], optimize=True)
    else:
        eq_face = np.zeros_like(full_face)
    def residual(mass, face):
        old, new = np.split(mass, 2)
        oldface, newface = np.split(face, 2)
        dt = setting['dt']
        ru = new[:, :, 0]-old[:, :, 0]-dt*(new[:, :, 1]+old[:, :, 1])/2
        rv = (new[:, :, 1]-old[:, :, 1]+dt*float(speed)*(newface[:, :, 1]+oldface[:, :, 1])/2 +
              dt*float(speed)**2*raw_rule['eigen']*(new[:, :, 0]+old[:, :, 0])/2)
        return np.concatenate((ru/float(model['scales'][0]), rv/float(model['scales'][1])), axis=1)
    full_r, eq_r = residual(full, full_face), residual(eq, eq_face)
    return {'observation_indices': indices.tolist(), 'full_minus_eq_mass_moments': normalized.tolist(),
            'maximum_fixed_scale_mass_moment_difference': float(np.max(normalized)),
            'full_weak_residual_norm': np.linalg.norm(full_r, axis=1).tolist(),
            'eq_weak_residual_norm': np.linalg.norm(eq_r, axis=1).tolist(),
            'full_minus_eq_weak_residual_norm': np.linalg.norm(full_r-eq_r, axis=1).tolist()}


def representation_diagnostic(queries, setting, supplied, truth, aux):
    """Untimed snapshot fits and weak tangent rank, not certified best-fit floors."""
    model = queries.models[setting['method']]
    rule = queries.rules[(setting['method'], setting['multiplier'])]
    init, evolve, decode, residual, _ = queries.functions[(setting['method'], setting['cap'], setting['dt'])]
    ut, vt = truth
    speed = supplied[2]
    grid = queries.grid
    mass = jnp.asarray(grid.mass())
    u_scale = jnp.sqrt(jnp.sum(mass*ut[0]**2))
    state_scale = jnp.sqrt(2*energy(ut[0], vt[0], grid, speed))
    indices = np.unique(np.linspace(0, len(ut)-1, min(4, len(ut)), dtype=int))
    jacobian = jax.jit(jax.jacfwd(residual))
    residual_eval = jax.jit(residual)
    rows = []
    for i in indices:
        z, fit = init(model['params'], rule, ut[i], vt[i], model['starts'], model['scales'])
        u, v = decode(model['params'], z[None])
        du, dv = u[0]-ut[i], v[0]-vt[i]
        errors = {'displacement': float(jnp.sqrt(jnp.sum(mass*du*du))/u_scale),
                  'velocity': float(jnp.sqrt(jnp.sum(mass*dv*dv))/state_scale),
                  'energy_state': float(jnp.sqrt(jnp.maximum(2*energy(du, dv, grid, speed), 0.))/state_scale)}
        om, of = moments(model['params'], z, rule, model['config'])
        args = (z, model['params'], rule, om, of, speed, model['scales'])
        J = jacobian(*args)
        singular = np.linalg.svd(np.asarray(J), compute_uv=False)
        chosen = int(fit['selected'])
        rows.append({'observation_index': int(i), 'full_grid_physical_error': clean(errors),
                     'fit_reason': int(fit['reason'][chosen]), 'fit_stationarity': float(fit['stationarity'][chosen]),
                     'weak_tangent_singular_values': singular.tolist(),
                     'weak_tangent_rank_ratio': float(singular[-1]/max(singular[0], 1e-300))})
        if i == indices[0]:
            for _ in range(2):
                jax.block_until_ready((residual_eval(*args), jacobian(*args)))
            times = []
            for _ in range(7):
                burn()
                start = time.perf_counter()
                r, J = residual_eval(*args), jacobian(*args)
                jax.block_until_ready((r, J))
                times.append(time.perf_counter()-start)
    return {'configuration': setting['id'], 'intervals': grid.n, 'case': 0,
            'interpretation': 'Diagnostic local snapshot fits provide achievable reconstruction errors, not exact manifold projection floors.',
            'snapshot_fits': rows, 'residual_plus_jacobian_seconds': times}


def run_evaluation(cfg, grid, models, rules, selected, result, out):
    """Called only after every mesh's selections have been persisted."""
    queries = Queries(cfg, grid, models, rules)
    save = lambda: save_json(out/'handoff.json', result)
    n = grid.n
    pars = parameter_rows(cfg['evaluation_seed'], cfg['evaluation_count'])
    for ci, parameter in enumerate(pars):
        ut, vt, audit = reference(grid, parameter, cfg, refine=True)
        supplied = (ut[0], vt[0], jnp.asarray(parameter[5]))
        jax.block_until_ready((supplied, ut, vt))
        reference_path, reference_hash = save_reference(out, f'reference_{n}_{ci}', (ut, vt), grid, supplied[2])
        result['references'].append({'split': 'evaluation', 'intervals': n, 'case': ci, 'parameters': parameter.tolist(),
                                     'reference_artifact': reference_path, 'reference_sha256': reference_hash, **audit})
        for setting in selected:
            for _ in range(cfg['warmups']):
                queries.query(setting, supplied)
        first = {}
        for rep in range(cfg['repetitions']):
            order = selected if rep % 2 == 0 else list(reversed(selected))
            for setting in order:
                print('EVALUATE', n, ci, rep, setting['id'], flush=True)
                burn()
                u, v, row, aux = queries.query(setting, supplied)
                score = queries.metric(u, v, ut, vt, supplied[2])
                host_u, host_v = np.asarray(u), np.asarray(v)
                sha = {'u': array_sha(host_u), 'v': array_sha(host_v)}
                row.update(split='evaluation', case=ci, rep=rep, errors=error_maxima(score), output_sha256=sha)
                if rep == 0:
                    path, kind = save_fields(out, f'evaluation_{n}_{ci}_{setting["id"]}', host_u, host_v, (ut, vt), grid, supplied[2], True, shared_reference=True)
                    first[setting['id']] = (sha, path, kind)
                elif sha != first[setting['id']][0]:
                    raise RuntimeError('Nonidentical repetitions: cannot deduplicate the actual output artifact')
                row.update(field_artifact=first[setting['id']][1], field_artifact_kind=first[setting['id']][2],
                           reference_artifact=reference_path, reference_sha256=reference_hash,
                           field_serialization='numpy_npz_stored_lossless',
                           artifact_relation='actual_first_timed_output' if rep == 0 else 'identical_full_field_hash_verified')
                result['invocations'].append(row)
            save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--boundary', choices=('dirichlet', 'absorbing'), required=True)
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--phase', choices=('validation', 'evaluation', 'development'), default='validation')
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.phase == 'development' and (not cfg.get('smoke_only') or cfg['evaluation_seed'] == 910603):
        raise RuntimeError('Development bypass cannot draw scientific evaluation cohort')
    args.out.mkdir(parents=True, exist_ok=False)
    meta = provenance()
    print(json.dumps(meta), flush=True)
    if meta['jax_backend'] != 'gpu' or not meta['x64'] or meta['matmul_precision'] != 'highest':
        raise RuntimeError('GPU/f64/highest required')
    models = load_models(args.inputs, cfg, args.boundary)
    result = {'case_name': 'wave_reflective' if args.boundary == 'dirichlet' else 'wave_absorbing',
              'config': {**cfg, 'validation_case_ids': list(range(cfg['validation_count'])),
                         'evaluation_case_ids': list(range(cfg['evaluation_count'])), 'validation_repetitions': 1},
              'provenance': {**meta, 'commit': meta['source_commit'], 'gpu': meta['device_kind'],
                             'backend': meta['jax_backend'], 'matmul_precision': meta['matmul_precision']},
              'status': 'running', 'phase': args.phase, 'invocations': [], 'selection_timings': [], 'selections': [],
              'references': [], 'eq_audits': [], 'full_weak_audits': [], 'component_timings': [], 'representation_diagnostics': [],
              'checkpoint_sha256': {name: m['checkpoint_sha256'] for name, m in models.items()},
              'evaluation_opened': False, 'artifacts': []}
    save = lambda: save_json(args.out/'handoff.json', result)
    save()
    settings = configurations(cfg, args.boundary)
    frozen_panels = {}
    if args.phase == 'evaluation':
        bundle = args.inputs/'validation_bundle'
        old, frozen_panels, proof = verify_validation_bundle(bundle, cfg, result['case_name'], result['checkpoint_sha256'])
        result['imported_validation_proof'] = proof
        result['selections'] = old['selections']
        # Old timings remain explicitly separate evidence from another job.
        # The evaluation handoff contains only its own same-allocation rows.
        shutil.copytree(bundle, args.out/'imported_validation')
        shutil.copytree(bundle/'quadrature', args.out/'quadrature')
        result['artifacts'].append('imported_validation/global_seal.json')
        save()
    for n in ([] if args.phase == 'evaluation' else cfg['meshes']):
        grid = Grid(n, args.boundary, args.boundary)
        rules, raws = {}, {}
        for name, model in models.items():
            for multiplier in cfg['eq_multipliers']:
                cached = load_reusable_rule(args.inputs, args.out, name, n, multiplier, model, cfg, args.boundary)
                raw, audit = cached if cached is not None else build_rule(
                    model['params'], model['codes'], model['config'], grid, cfg,
                    multiplier, args.out/'quadrature'/f'{name}_{n}_{multiplier}')
                raws[(name, multiplier)] = raw
                rules[(name, multiplier)] = numerical_rule(model['params'], model['config'], raw)
                result['eq_audits'].append(audit); save()
        queries = Queries(cfg, grid, models, rules)
        # Validate every setting on all sixteen cases before constructing any
        # evaluation parameter row. Case-zero proxy repetitions remain separate.
        pars = parameter_rows(cfg['validation_seed'], cfg['validation_count'])
        for ci, parameter in enumerate(pars):
            ut, vt, audit = reference(grid, parameter, cfg, refine=True)
            supplied = (ut[0], vt[0], jnp.asarray(parameter[5]))
            jax.block_until_ready((supplied, ut, vt))
            result['references'].append({'split': 'validation', 'intervals': n, 'case': ci, 'parameters': parameter.tolist(), **audit})
            order = settings if ci % 2 == 0 else list(reversed(settings))
            if ci == 0:
                for setting in order:
                    print('WARM', n, setting['id'], flush=True)
                    for _ in range(cfg['warmups']):
                        queries.query(setting, supplied)
                for rep in range(cfg['repetitions']):
                    for setting in (order if rep % 2 == 0 else list(reversed(order))):
                        burn()
                        u, v, row, _ = queries.query(setting, supplied)
                        score = queries.metric(u, v, ut, vt, supplied[2])
                        row.update(split='validation_selection_proxy', case=0, rep=rep,
                                   errors=error_maxima(score),
                                   timing_role='configuration_selection_only_not_final_speed_evidence')
                        result['selection_timings'].append(row)
                    save()
            for setting in order:
                print('VALIDATE', n, ci, setting['id'], flush=True)
                burn()
                u, v, row, aux = queries.query(setting, supplied)
                score = queries.metric(u, v, ut, vt, supplied[2])
                row.update(split='validation', case=ci, rep=0,
                           errors=error_maxima(score),
                           output_sha256={'u': array_sha(np.asarray(u)), 'v': array_sha(np.asarray(v))})
                path, kind = save_fields(args.out, f'validation_{n}_{ci}_{setting["id"]}', u, v, (ut, vt), grid, supplied[2], False)
                row.update(field_artifact=path, field_artifact_kind=kind)
                result['invocations'].append(row)
                if ci == 0 and setting['method'] in models:
                    details = audit_full_weak(aux, models[setting['method']], raws[(setting['method'], setting['multiplier'])], setting, grid, supplied[2], cfg)
                    result['full_weak_audits'].append({'intervals': n, 'configuration': setting['id'], 'case': ci, **details})
                    result['component_timings'].append(queries.component_timing(setting, supplied))
                    if (setting['multiplier'] == max(cfg['eq_multipliers']) and setting['cap'] == max(cfg['gn_caps']) and
                            setting['tol'] == min(cfg['gn_tolerances']) and setting['dt'] == min(cfg['time_steps'])):
                        result['representation_diagnostics'].append(representation_diagnostic(queries, setting, supplied, (ut, vt), aux))
            save()
        selections, selected = select([r for r in result['invocations'] if r['split'] == 'validation'],
                                      result['selection_timings'], settings, grid, cfg)
        result['selections'].extend(selections)
        freeze = {'intervals': n, 'selections': selections, 'selected_settings': selected,
                  'evaluation_opened_at_selection': False, 'training_sha256': result['checkpoint_sha256'],
                  'quadrature_sha256': {str(p.relative_to(args.out)): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in sorted((args.out/'quadrature').glob(f'*_{n}_*/rule.npz'))}}
        save_json(args.out/f'frozen_selection_{n}.json', freeze)
        save()
        frozen_panels[n] = selected
        del rules, raws, queries
        jax.clear_caches()
    if args.phase == 'validation':
        result['status'] = 'validation_frozen'
        save()
        print('WAVE_VALIDATION_COMPLETE_EVALUATION_SEALED', args.boundary, flush=True)
        return
    # The shared physical evaluation cohort remains unopened until all meshes
    # have their validation-only configuration selections fixed on disk.
    result['evaluation_opened'] = True
    save()
    for n in cfg['meshes']:
        grid = Grid(n, args.boundary, args.boundary)
        rules = {}
        for name, model in models.items():
            for multiplier in cfg['eq_multipliers']:
                with np.load(args.out/'quadrature'/f'{name}_{n}_{multiplier}'/'rule.npz') as archive:
                    raw = {key: archive[key] for key in archive.files}
                rules[(name, multiplier)] = numerical_rule(model['params'], model['config'], raw)
        run_evaluation(cfg, grid, models, rules, frozen_panels[n], result, args.out)
        del rules
        jax.clear_caches()
    result['status'] = 'complete'
    save()
    print('WAVE_COMPARISON_COMPLETE', args.boundary, flush=True)


if __name__ == '__main__':
    main()
