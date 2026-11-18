#!/usr/bin/env python
"""Execute the registered dense discrete Burgers visibility pilot on one GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import time
from pathlib import Path

os.environ.setdefault('JAX_ENABLE_X64', 'true')
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import core as C
from core import e
import sep_common as sc
import cohort_guard

HERE = Path(__file__).resolve().parent


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        return clean(np.asarray(value).tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def npz(path, **arrays):
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})


def spectrum(J, transform):
    raw = np.linalg.svd(J, compute_uv=False)
    physical = np.linalg.svd(J @ transform, compute_uv=False)
    return dict(raw=raw, physical=physical, raw_min=raw[-1], physical_min=physical[-1],
                raw_cond=raw[0]/raw[-1], physical_cond=physical[0]/physical[-1])


def action_check(coeff, previous, nu, data, n, dt, transform):
    """Independent direct sums and finite differences, recording sign crossings."""
    r, J = jax.block_until_ready(C.value_jac(jnp.asarray(coeff), jnp.asarray(previous), nu, data, n, dt))
    G = np.asarray(data['G'])
    modes, lam = np.asarray(data['modes']), np.asarray(data['lam'])
    u, old = G @ coeff, G @ previous
    adv, lap = (np.asarray(v) for v in e.spatial(jnp.asarray(u), n))
    full = u-old+dt*(adv-nu*lap)
    direct = C.explicit_project(full, modes, n)/(1+dt*nu*lam)
    norm = np.linalg.norm(np.asarray(r))
    residual_abs = np.linalg.norm(direct-np.asarray(r))
    residual_rel = residual_abs/max(norm, 1e-300)
    directions = []
    for seed in (1701, 1702, 1703):
        direction = np.random.default_rng(seed).normal(size=transform.shape[1])
        direction /= np.linalg.norm(direction)
        dc = transform @ direction
        du = G @ dc
        # The original spatial routine supplies an independent AD directional action.
        spatial_action = jax.jvp(lambda v: e.spatial(v, n), (jnp.asarray(u),), (jnp.asarray(du),))[1]
        full_action = du+dt*(np.asarray(spatial_action[0])-nu*np.asarray(spatial_action[1]))
        direct_action = C.explicit_project(full_action, modes, n)/(1+dt*nu*lam)
        analytic = np.asarray(J) @ dc
        error_abs = np.linalg.norm(direct_action-analytic)
        error_rel = error_abs/max(np.linalg.norm(analytic), 1e-300)
        fd = []
        for eps in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
            plus = C.weak_all(jnp.asarray(coeff+eps*dc), jnp.asarray(previous), nu, data, n, dt)
            minus = C.weak_all(jnp.asarray(coeff-eps*dc), jnp.asarray(previous), nu, data, n, dt)
            finite_diff = np.asarray((plus-minus)/(2*eps))
            fd.append(dict(epsilon=eps, relative_error=np.linalg.norm(finite_diff-analytic)/max(np.linalg.norm(analytic), 1e-300),
                           switched_nodes=np.sum((u+eps*du > 0) != (u-eps*du > 0))))
        directions.append(dict(seed=seed, explicit_action_relative=error_rel, explicit_action_absolute=error_abs,
                               finite_difference=fd, passed=(error_rel <= 1e-10 or error_abs <= 1e-12) and min(v['relative_error'] for v in fd) <= 1e-5))
    return dict(residual_relative=residual_rel, residual_absolute=residual_abs, directions=directions,
                passed=bool((residual_rel <= 1e-10 or residual_abs <= 1e-12) and all(v['passed'] for v in directions)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(HERE/'configs/pilot.json'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    assert jax.default_backend() == 'gpu', jax.default_backend()
    assert jax.config.jax_enable_x64
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print('jax_backend=gpu x64=True precision=highest', flush=True)
    assert cfg['cohort'] == 'dev6' and cfg['seed_first_draw'] == 7090702 and cfg['seed_first_count'] == 4
    if cfg.get('cohort_phase', 'initial-dev2') in ('remaining-dev6', 'all-dev6-width'):
        assert cfg['case_ids'] == ([2, 3, 4, 5] if cfg['cohort_phase'] == 'remaining-dev6' else list(range(6)))
        assert cfg['seed_second_draw'] == 911702 and cfg['seed_second_count'] == 2
    else:
        assert cfg['case_ids'] == [0, 1], 'Initial protocol opens exactly dev6 indices 0 and 1.'
    physical, cohort_record = cohort_guard.regenerate(cfg, e.params_draw)
    save_json(out/'cohort-drawing.json', cohort_record)
    assert cohort_record['guard_passed'], 'Canonical cohort identity failed; retained cohort-drawing.json records every descriptor.'
    n, width = cfg['mesh_intervals'], cfg['bank_width']
    if cfg.get('width_followup'):
        admission_path = HERE/'inputs/width-admission.json'
        admission = json.loads(admission_path.read_text())
        assert admission['W0_raw_audit_passed'] and admission['R128_reproduction_passed']
        assert admission['admitted_widths']
        assert width == 128 or width in admission['admitted_widths']
    dt, steps = cfg['dt'], int(round(cfg['horizon']/cfg['dt']))
    assert abs(steps*dt-cfg['horizon']) < 1e-12
    ms = cfg['solve_rows']
    max_m = max(ms)
    observer_start = int(cfg.get('observer_start', max_m))
    assert observer_start >= max_m
    modes, lam = C.mode_order(n, observer_start+cfg['observer_rows'])
    ckpt, rotation = HERE/'inputs/frozen.pkl', HERE/'inputs/rotation_R512.npz'
    checkpoint = pickle.load(open(ckpt, 'rb'))
    params = jax.tree_util.tree_map(jnp.asarray, checkpoint['params'])
    T = np.load(rotation)['T'][:, :width]
    del checkpoint
    feature = jax.jit(lambda p, x, t: sc.features(p, x) @ t)
    xy = e.coords(n)
    G = jnp.concatenate([feature(params, jnp.asarray(xy[k:k+2048]), jnp.asarray(T)) for k in range(0, len(xy), 2048)])
    G = jax.block_until_ready(G)
    A = jax.jit(lambda g, mode: C.project(g, mode, n))(G, jnp.asarray(modes))
    data = jax.block_until_ready(dict(G=G, A=A, modes=jnp.asarray(modes), lam=jnp.asarray(lam)))
    gram, eigenvalues, transform = C.physical_tangent(G, n, cfg['tangent_relative_eigenvalue_threshold'])
    ranks = {str(t): int(np.sum(eigenvalues > t*eigenvalues[-1])) for t in (1e-10, 1e-12, 1e-14)}
    assert transform.shape[1] == width, 'Frozen span lost rank; retain this failure before changing thresholds.'
    for start in range(0, width, 64):
        npz(out/f'bank-columns-{start:03d}.npz', bank=G[:, start:start+64])
    npz(out/'operators.npz', A=A, modes=modes, lam=lam, gram=gram, gram_eigenvalues=eigenvalues, transform=transform)
    scaling = np.exp(np.linspace(-1, 1, width))
    scaled_gram = scaling[:, None]*gram*scaling[None, :]
    se, sv = np.linalg.eigh(scaled_gram)
    sk = se > cfg['tangent_relative_eigenvalue_threshold']*se[-1]
    scaled_transform = sv[:, sk]/np.sqrt(se[sk])[None, :]
    environment = dict(config=cfg, checkpoint_sha256=sha(ckpt), rotation_sha256=sha(rotation),
                       source_sha256={str(p.relative_to(HERE)): sha(p) for p in [HERE/'core.py', HERE/'run.py', HERE/'vendor/engines.py', HERE/'vendor/sep_common.py']},
                       job_id=os.environ.get('SLURM_JOB_ID'), commit=os.environ.get('SOURCE_COMMIT'),
                       gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid', '--format=csv,noheader'], text=True).strip(),
                       backend=jax.default_backend(), x64=jax.config.jax_enable_x64, matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],
                       tangent_ranks=ranks, scaled_tangent_rank=int(sk.sum()), grid_nodes=(n-1)**2,
                       actual_unknowns=width, free_frame_variables=0, mass_per_node=1/n**2,
                       wall_start=time.time(), buffer_residual_rows=len(modes), no_production_timing_claim=True)
    environment['observer_start_row_zero_based'] = observer_start
    if cfg.get('width_followup'):
        environment['width_admission_sha256'] = sha(admission_path)
    save_json(out/'environment.json', environment)
    solver = C.make_solver(n, dt, steps, cfg['solver_max_iterations'])
    diagnostic = jax.jit(lambda c, old, nu, d: C.value_jac(c, old, nu, d, n, dt))
    metric_fn = jax.jit(lambda f, r: C.field_metrics(f, r, n))
    fom, _ = e.make_fom(n, dt, horizon=cfg['horizon'], output_spacing=dt, max_newton=20)
    repaired_reference = cfg.get('reference_arithmetic', 'original') == 'neighbor-difference'
    if repaired_reference:
        stable_fom = C.make_reference(n, dt, steps, True)
        original_instrumented_fom = C.make_reference(n, dt, steps, False)
    save_json(out/'cases.json', [{'cohort':'dev6', 'index':i, 'physical':physical[i]} for i in cfg['case_ids']])
    summary = dict(scope='same-grid same-step discrete target; preliminary development cases', cases={}, environmental_acceptance_pending=True)
    first = True
    for case_id in cfg['case_ids']:
        print(f'case={case_id} start', flush=True)
        case_dir = out/f'case-{case_id:02d}'
        case_dir.mkdir()
        u0 = jnp.asarray(e.initial(n, physical[case_id]))
        nu = float(physical[case_id, -1])
        references = []
        refstats = []
        for level, (ntol, ltol) in enumerate(cfg['reference_tolerances']):
            if repaired_reference:
                reference, iterations, residual, traces = jax.block_until_ready(stable_fom(u0, nu, ntol, ltol))
                old, old_it, old_res, old_trace = jax.block_until_ready(original_instrumented_fom(u0, nu, ntol, ltol))
                npz(case_dir/f'original-reference-{level}.npz', fields=old, iterations=old_it, residual=old_res, newton_history=old_trace)
                npz(case_dir/f'reference-newton-history-{level}.npz', residual_history=traces)
                original_residual = jax.jit(jax.vmap(lambda u, prev: e.residual(u, prev, nu, dt, n)))(reference[1:], reference[:-1])
                original_relative = jnp.linalg.norm(original_residual, axis=1)/jnp.linalg.norm(reference[:-1], axis=1)
                stable_residual = jax.jit(jax.vmap(lambda u, prev: C.reference_residual(u, prev, nu, dt, n, True)))(reference[1:], reference[:-1])
                difference = jnp.linalg.norm(original_residual-stable_residual, axis=1)/jnp.linalg.norm(reference[:-1], axis=1)
                old_difference = jnp.linalg.norm(reference-old, axis=1).max()/jnp.linalg.norm(reference[0])
                actions = []
                for time_index in (1, steps):
                    u, prev = reference[time_index], reference[time_index-1]
                    for seed in (1701, 1702, 1703):
                        v = jnp.asarray(np.random.default_rng(seed).normal(size=u.shape))
                        v = v/jnp.linalg.norm(v)
                        jo = jax.jvp(lambda q: e.residual(q, prev, nu, dt, n), (u,), (v,))[1]
                        js = jax.jvp(lambda q: C.reference_residual(q, prev, nu, dt, n, True), (u,), (v,))[1]
                        actions.append(dict(time_index=time_index, seed=seed, relative_difference=float(jnp.linalg.norm(js-jo)/jnp.linalg.norm(jo))))
                parity_record = dict(original_residual_max=original_relative.max(), primal_difference_max=difference.max(),
                    old_new_trajectory_difference=old_difference, jacobian_actions=actions,
                    passed=bool(original_relative.max() <= 1e-10 and difference.max() <= 1e-12 and old_difference <= 1e-8 and max(a['relative_difference'] for a in actions) <= 1e-12))
                npz(case_dir/f'reference-arithmetic-{level}.npz', original_relative_residual=original_relative, primal_difference=difference)
                save_json(case_dir/f'reference-arithmetic-{level}.json', parity_record)
            else:
                value, iterations, residual = jax.block_until_ready(fom(u0, nu, ntol, ltol))
                reference = value[:, 1:-1, 1:-1].reshape(steps+1, -1)
                parity_record = dict(passed=True, applicable=False)
            references.append(reference)
            npz(case_dir/f'reference-{level}.npz', fields=reference, iterations=iterations, residual=residual, physical=physical[case_id])
            refstats.append(dict(max_residual=np.max(residual), maximum_iterations=np.max(iterations),
                                 finite=bool(np.all(np.isfinite(reference))), tolerances=[ntol, ltol], arithmetic_parity=parity_record))
        norm0 = float(jnp.linalg.norm(references[0][0]))
        reference_delta = float(jnp.max(jnp.linalg.norm(references[0]-references[1], axis=1))/norm0)
        ref_eligible = all(v['finite'] and v['max_residual'] <= 1e-10 and v['maximum_iterations'] < 20 and v['arithmetic_parity']['passed'] for v in refstats) and reference_delta <= cfg['reference_max_disagreement']
        reference = references[1]
        Ghost = np.asarray(G)
        # Exact mass projection in this fixed linear span; same coefficients initialize all M.
        if cfg.get('initialization') == 'mass-qr':
            Qmass, Rmass = jax.block_until_ready(jnp.linalg.qr(G/n, mode='reduced'))
            rhs = Qmass.T @ (reference.T/n)
            project_coeff = jax.scipy.linalg.solve_triangular(Rmass, rhs, lower=False).T
            qr_projection = (Qmass @ rhs).T*n
            initialization_control = dict(algorithm='mass-qr',
                qr_orthogonality=float(jnp.linalg.norm(Qmass.T @ Qmass-jnp.eye(width))/jnp.sqrt(float(width))),
                field_parity=float(jnp.linalg.norm(project_coeff @ G.T-qr_projection, axis=1).max()/norm0))
            initialization_control['passed'] = initialization_control['qr_orthogonality'] <= 1e-10 and initialization_control['field_parity'] <= 1e-10
            npz(case_dir/'initialization-qr.npz', Rmass=Rmass)
        else:
            project_coeff = jnp.linalg.solve(jnp.asarray(gram), (G.T @ reference.T)/n**2).T
            initialization_control = dict(algorithm='mass-gram', passed=True)
        projected = project_coeff @ G.T
        c0 = project_coeff[0]
        projection_error = jnp.linalg.norm(projected-reference, axis=1)/norm0
        npz(case_dir/'projection.npz', coeff=project_coeff, error=projection_error)
        case_summary = dict(reference_eligible=bool(ref_eligible), reference_disagreement=reference_delta,
                            reference_controls=refstats, initialization_control=initialization_control, arms={}, fixed_state_spectra=[])
        # All test-space sensitivities at identical reference-projected states.
        for ti in (1, 10, 25, 50):
            if ti > steps:
                continue
            r, J = jax.block_until_ready(diagnostic(project_coeff[ti], project_coeff[ti-1], nu, data))
            spectra = {str(m): spectrum(np.asarray(J[:m]), transform) for m in ms}
            case_summary['fixed_state_spectra'].append(dict(time_index=ti, spectra=spectra))
        for m in ms:
            arm = dict(levels={}, reached_spectra=[], parity=[])
            outputs = {}
            for name, gtol in cfg['solver_gtols'].items():
                if first:
                    start = time.perf_counter()
                    warm = jax.block_until_ready(solver(c0, nu, data, m, gtol))
                    npz(out/'compile-witness.npz', coeff=warm[0], stats=warm[1], history=warm[2])
                    save_json(out/'compile-witness.json', dict(case_id=case_id, m=m, level=name, seconds=time.perf_counter()-start))
                    first = False
                reps = cfg['timing_repetitions'] if name == 'tight' else 1
                timings, metrics, witnesses = [], [], []
                for rep in range(reps):
                    e.burn(.6)
                    start = time.perf_counter()
                    coeff, stats, histories = jax.block_until_ready(solver(c0, nu, data, m, gtol))
                    elapsed = time.perf_counter()-start
                    field = coeff @ G.T
                    metric = jax.block_until_ready(metric_fn(field, reference))
                    delta = jax.block_until_ready(jnp.linalg.norm(field-projected, axis=1)/norm0)
                    identity = np.asarray(metric[:, 0])**2-np.asarray(projection_error)**2-np.asarray(delta)**2
                    npz(case_dir/f'm{m}-{name}-rep{rep}.npz', coeff=coeff, stats=stats, histories=histories,
                        metrics=metric, evolution_distance=delta, pythagorean_residual=identity, elapsed_seconds=elapsed)
                    timings.append(elapsed)
                    metrics.append(np.asarray(metric))
                    witnesses.append(hashlib.sha256(np.asarray(coeff).tobytes()).hexdigest())
                outputs[name] = coeff
                stats_n = np.asarray(stats)
                eligible_solver = bool(np.all(stats_n[:, 1] == 1) and np.all(stats_n[:, 2] <= gtol) and np.all(np.isfinite(coeff)))
                arm['levels'][name] = dict(timing_repetitions=timings, coefficient_hashes=witnesses,
                    max_error_per_repetition=[float(x[:, 0].max()) for x in metrics],
                    solver_eligible=eligible_solver, max_stationarity=float(stats_n[:, 2].max()),
                    total_iterations=int(stats_n[:, 0].sum()), rejected_steps=int(stats_n[:, 4].sum()),
                    termination_counts={str(k):int(np.sum(stats_n[:, 1] == k)) for k in range(1, 5)},
                    pythagorean_max_absolute=float(np.max(np.abs(identity))))
                print(f'case={case_id} M={m} level={name} error={metrics[-1][:,0].max():.6g} stationarity={stats_n[:,2].max():.3g} eligible={eligible_solver}', flush=True)
            tight, tighter = outputs['tight'], outputs['tighter']
            traj_diff = float(jnp.max(jnp.linalg.norm((tight-tighter) @ G.T, axis=1))/norm0)
            error = arm['levels']['tighter']['max_error_per_repetition'][0]
            threshold = min(1e-6, .01*error) if error >= 1e-4 else 1e-6
            arm['tight_tighter_difference'] = traj_diff
            arm['tight_tighter_threshold'] = threshold
            arm['solver_pair_eligible'] = bool(arm['levels']['tight']['solver_eligible'] and arm['levels']['tighter']['solver_eligible'] and traj_diff <= threshold)
            coeff = outputs['tighter']
            residuals = []
            for ti in range(1, steps+1):
                r = jax.block_until_ready(C.weak_all(coeff[ti], coeff[ti-1], nu, data, n, dt))
                residuals.append(np.asarray(r))
            residuals = np.asarray(residuals)
            unscaled = residuals*(1+dt*nu*lam)[None, :]
            npz(case_dir/f'm{m}-residuals.npz', scaled=residuals, unscaled=unscaled)
            arm['independent_scaled_shell_max'] = np.linalg.norm(residuals[:, observer_start:], axis=1).max()
            arm['independent_unscaled_shell_max'] = np.linalg.norm(unscaled[:, observer_start:], axis=1).max()
            for ti in (1, 10, 25, 50):
                if ti > steps:
                    continue
                _, J = jax.block_until_ready(diagnostic(coeff[ti], coeff[ti-1], nu, data))
                J = np.asarray(J)
                spectra = {str(mm): spectrum(J[:mm], transform) for mm in ms}
                scaled_s = np.linalg.svd((J[:m]*scaling[None, :]) @ scaled_transform, compute_uv=False)
                original_s = spectra[str(m)]['physical']
                invariant_error = np.linalg.norm(scaled_s-original_s)/np.linalg.norm(original_s) if len(scaled_s) == len(original_s) else None
                arm['reached_spectra'].append(dict(time_index=ti, spectra=spectra, scaled_physical_spectrum=scaled_s,
                    rescaling_invariance_relative=invariant_error, same_rank=bool(len(scaled_s)==len(original_s))))
            for ti in (1, steps):
                arm['parity'].append(dict(time_index=ti, **action_check(np.asarray(coeff[ti]), np.asarray(coeff[ti-1]), nu, data, n, dt, transform)))
            parity = all(p['passed'] for p in arm['parity'])
            invariance = all(s['same_rank'] and s['rescaling_invariance_relative'] <= 1e-8 for s in arm['reached_spectra'])
            arm['numerically_eligible'] = bool(ref_eligible and initialization_control['passed'] and arm['solver_pair_eligible'] and parity and invariance)
            case_summary['arms'][str(m)] = arm
            summary['cases'][str(case_id)] = case_summary
            save_json(out/'summary.json', summary)
        print(f'case={case_id} finished', flush=True)
    summary['environmental_acceptance_pending'] = True  # collector audits scheduler/logs/checksums separately
    summary['wall_finished'] = time.time()
    save_json(out/'summary.json', summary)
    print('completed=True', flush=True)


if __name__ == '__main__':
    main()
