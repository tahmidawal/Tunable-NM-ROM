"""Audit the bounded screen and generate notes directly from saved arrays.

Uses independent NumPy polynomial derivatives, without importing the model or JAX.
Run with the absolute project Python after pulling the immutable job outputs.
"""
import hashlib
import json
from pathlib import Path
import pickle
import subprocess

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal_tree(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            equal_tree(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) == type(b) and len(a) == len(b)
        for aa, bb in zip(a, b):
            equal_tree(aa, bb)
    else:
        np.testing.assert_array_equal(a, b)


def hessian_of_polynomial(p, frozen):
    r, k = p['linear'].shape
    h = np.zeros((r, k, k))
    scale = np.asarray(frozen['latent_scale'])
    output = float(frozen['output_scale'])
    pair = 0
    for i in range(k):
        for j in range(i, k):
            coefficient = output * np.asarray(p['quadratic'][pair]) / (scale[i] * scale[j])
            h[:, i, j] = coefficient
            h[:, j, i] = coefficient
            if i == j:
                h[:, i, j] *= 2
            pair += 1
    return h


def verify_stage(stage, payload, val):
    p, f = payload['trainable'], payload['frozen']
    z = np.asarray(stage['z'])
    h = hessian_of_polynomial(p, f)
    q = np.asarray(p['bias']) + z @ np.asarray(p['linear']).T
    q += .5 * np.einsum('ni,cij,nj->nc', z, h, z, optimize=True)
    jac = np.asarray(p['linear'])[None, :, :] + np.einsum('cij,nj->nci', h, z)
    residual = q - np.asarray(val['target'])
    full_residual2 = np.sum(residual**2, axis=1) + val['perpendicular2']
    errors = np.sqrt(full_residual2 / val['norm2'])
    grads = np.einsum('nci,nc->ni', jac, residual)
    optimality = np.linalg.norm(grads, axis=1) / (np.linalg.norm(jac, axis=(1, 2)) * np.sqrt(full_residual2))
    np.testing.assert_allclose(errors, stage['error'], rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(optimality, stage['optimality'], rtol=1e-6, atol=1e-10)
    best = np.argmin(stage['all_errors'], axis=1)
    np.testing.assert_array_equal(best, stage['best_start'])
    np.testing.assert_array_equal(np.min(stage['all_errors'], axis=1), stage['error'])
    all_invariant, all_tangent, curvature, negative = [], [], [], []
    for i, ji in enumerate(jac):
        u, s, _ = np.linalg.svd(ji, full_matrices=False)
        rank = int(np.sum(s > 1e-10 * s[0]))
        u = u[:, :rank]
        geom = stage['geometry'][i]
        assert rank == geom['rank']
        np.testing.assert_allclose(s, geom['singular_values'], rtol=1e-10, atol=1e-11)
        invariant = np.linalg.norm(u.T @ residual[i]) / np.sqrt(full_residual2[i])
        v = np.asarray(val['velocity_target'][i])
        missing = v - u @ (u.T @ v)
        tangent = np.sqrt((np.dot(missing, missing) + val['velocity_perpendicular2'][i]) / val['velocity_norm2'][i])
        assert abs(invariant - geom['invariant_stationarity']) < 1e-9
        assert abs(tangent - geom['tangent_relative']) < 1e-9
        # Hessian of half the full squared residual; the bank floor is constant.
        objective_hessian = ji.T @ ji + np.einsum('c,cij->ij', residual[i], h)
        eigen = np.linalg.eigvalsh(objective_hessian)
        cutoff = 1e-8 * max(np.max(np.abs(eigen)), 1e-300)
        curvature.append(float(eigen[0]))
        negative.append(bool(eigen[0] < -cutoff))
        all_invariant.append(float(invariant))
        all_tangent.append(float(tangent))
    nonstationary = np.flatnonzero(optimality > 1e-6)
    worst = int(np.argmax(errors))
    result = dict(budget=stage['budget'], independent_output_error_max=float(np.max(np.abs(errors-stage['error']))),
        independent_optimality_error_max=float(np.max(np.abs(optimality-stage['optimality']))),
        normalized_gradient=optimality.tolist(), invariant_stationarity=all_invariant,
        tangent_relative=all_tangent, objective_hessian_min_eigenvalues=curvature,
        objective_hessian_negative_cutoff='1e-8 times maximum absolute eigenvalue in original z coordinates',
        negative_curvature_count=int(np.sum(negative)),
        stationary_negative_curvature_count=int(np.sum(np.asarray(negative) & (optimality <= 1e-6))),
        nonstationary_indices=nonstationary.tolist(), nonstationary_snapshot_ids=np.asarray(val['sid'])[nonstationary].tolist(),
        worst_index=worst, worst_snapshot_id=int(val['sid'][worst]), worst_optimality=float(optimality[worst]),
        worst_invariant_stationarity=all_invariant[worst])
    assert len(nonstationary) == stage['nonstationary']
    return result


def main():
    here = Path(__file__).resolve().parent
    src = here.parents[1]
    repo = src.parents[1]
    record = src / 'runs/b3d_architecture/screen40'
    for manifest in ('PULL.sha256', 'RESULTS.sha256'):
        subprocess.run(['sha256sum', '-c', manifest, '--quiet'], cwd=record, check=True)
    report = json.loads((record / 'out/result.json').read_text())
    config = report['config']
    assert report['complete'] and config['backend'] == 'gpu' and config['x64']
    assert config['matmul_precision'] == 'highest' and config['test_table_opened'] is False
    assert config['optimizer_seeds'] == [200, 201] and len(report['runs']) == 2
    assert report['bank']['rank'] == 128
    assert report['bank']['truth_max_residual'] < 1e-8
    assert report['bank']['velocity_be_identity_max'] < 1e-8
    assert (record / 'EXIT_CODE').read_text().strip() == '0'
    assert (record / 'COMMIT.txt').read_text().strip() == config['commit']
    for line in (record / 'INPUTS.sha256').read_text().splitlines():
        expected, name = line.split('  ', 1)
        if name.startswith('code/'):
            content = subprocess.check_output(['git', '-C', str(repo), 'show',
                f'{config["commit"]}:experiments/separable-decoder/{name[5:]}'])
            assert hashlib.sha256(content).hexdigest() == expected
    assert digest(src/'b3d_arch_quadratic.py') == config['model_sha256']
    source = src / 'runs/b3d_repair/head33/out/refined_checkpoint.pkl'
    assert digest(source) == config['source_sha256']
    with source.open('rb') as stream:
        source_checkpoint = pickle.load(stream)
    logs = '\n'.join(p.read_text() for p in (record/'logs').glob('*'))
    assert 'jax_backend=gpu' in logs and 'DONE ' in logs
    for forbidden in ('Captured constant', 'captured constant', 'out of memory', 'No space left', 'Traceback'):
        assert forbidden not in logs, forbidden
    provenance = report['provenance']['fields']
    assert all(provenance[k]['exact'] for k in ('seed', 'm', 'B', 'c', 'w', 'rho', 'A'))
    val = {k: np.asarray(v) for k, v in report['validation_states'].items()}
    expected_sid = (np.arange(512, 576)[:, None] * 51 + np.asarray([0, 10, 25, 50])).ravel()
    np.testing.assert_array_equal(val['sid'], expected_sid)
    control = json.loads((src/'runs/b3d_architecture/mlp128/out/result.json').read_text())
    assert control['config']['source_sha256'] == config['source_sha256']
    inherited = json.loads((src/'runs/b3d_repair/repair_repro33/out/repro33.json').read_text())
    pod_floor = inherited['gates']['D4_heldout_oracle_validation']['pod_K_floor_mean']
    required_mean = min(.05, .5 * pod_floor)
    cohort = {}
    for name, values in val.items():
        other = np.asarray(control['validation_states'][name])
        if name in ('sid', 'blob_count'):
            np.testing.assert_array_equal(values, other)
        else:
            np.testing.assert_allclose(values, other, rtol=1e-12, atol=1e-12)
        cohort[name] = dict(exact=bool(np.array_equal(values, other)), max_absolute=float(np.max(np.abs(values-other))))
    audit = dict(status='verified', job=config['slurm_job'], source_commit=config['commit'],
        source_content_and_output_checksums_verified=True, cohort_parity_with_mlp128=cohort,
        backend=config['backend'], x64=config['x64'], matmul_precision=config['matmul_precision'],
        host_affinity_warning='hwloc_set_cpubind' in logs, test_data_opened=False,
        results_are_local_multistart_fits_not_global_minima=True,
        inherited_pod_K_floor_mean=pod_floor, effective_mean_ceiling=required_mean, runs=[])
    for run in report['runs']:
        cp = record / 'out' / f'quadratic_head_seed{run["seed"]}.pkl'
        assert digest(cp) == run['checkpoint_sha256']
        with cp.open('rb') as stream:
            payload = pickle.load(stream)
        equal_tree(payload['bank_params'], source_checkpoint['params'])
        assert all(np.asarray(v).dtype == np.float64 for v in payload['trainable'].values())
        assert all(np.asarray(v).dtype == np.float64 for v in payload['frozen'].values())
        assert payload['codes'].shape == (8192, 32)
        cfg = payload['cfg']
        assert (cfg['N'], cfg['k'], cfg['r'], cfg['seed'], cfg['n_train_traj'], cfg['max_snaps']) == (33, 32, 128, 0, 512, 8192)
        assert cfg['val_rows'] == [512, 575]
        training = run['training']
        assert (training['steps'], training['lr'], training['batch'], training['norm']) == (60000, 3e-4, 4096, 'global')
        assert training['initialization_code_max_difference'] < 1e-12
        assert training['initialization_field_max_difference'] < 1e-12
        assert sum(np.asarray(x).size for x in payload['trainable'].values()) == training['trainable_parameter_count'] == 71808
        final = run['fits'][-1]['summary']
        audit['runs'].append(dict(seed=run['seed'], bank_bit_identical=True,
            effective_mean_pass=bool(final['mean'] <= required_mean),
            worst_error_pass=bool(final['worst'] <= .15),
            oracle_over_pod_K=final['mean'] / pod_floor,
            checkpoint_sha256=digest(cp), stages=[verify_stage(stage, payload, val) for stage in run['fits']]))
    (here/'screen40_audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    notes = ['# Burgers 3D quadratic head screen', '',
        'Generated from the retained run JSON and independent polynomial audit. These validation results remain '
        'provisional pending coordinator result review; they do not establish rollout accuracy or wave transfer.', '',
        '| Head | Optimizer seed | Mean | Median | Worst | Above 15% | Unconverged | Tangent mean | Trainable parameters |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label in ('mlp128', 'mlp192', 'screen40'):
        data = json.loads((src/f'runs/b3d_architecture/{label}/out/result.json').read_text())
        display = {'mlp128': 'MLP width 128', 'mlp192': 'MLP width 192', 'screen40': 'Quadratic'}[label]
        for run in data['runs']:
            fit = run['fits'][-1]; sm = fit['summary']
            notes.append(f'| {display} | {run["seed"]} | {100*sm["mean"]:.4f}% | {100*sm["median"]:.4f}% | '
                f'{100*sm["worst"]:.4f}% | {fit["outliers_above_15pct"]} | {fit["nonstationary"]} | '
                f'{100*fit["tangent_summary"]["mean"]:.4f}% | {run["training"]["trainable_parameter_count"]} |')
    notes += ['', 'The quadratic head uses one undoubled product per latent-variable pair. Its affine '
        'initialization and zero quadratic weights are deterministic. Optimizer seeds change minibatches, '
        'not the model initialization or PDE cohort. The wider MLP supplies a close parameter-count control; '
        'equal updates do not imply equal compute. No cross-job timing comparison is made.', '']
    notes += [f'The effective inherited mean-error ceiling is {100*required_mean:.6f}%, '
        f'obtained from the POD comparison on the same cohort (POD mean {100*pod_floor:.6f}%). '
        'A mean below the looser standalone threshold is insufficient. Both the effective mean '
        'and worst-error checks remain required; this screen does not execute the full inherited pilot.', '']
    for run, checked in zip(report['runs'], audit['runs']):
        fit = run['fits'][-1]; ca = checked['stages'][-1]
        notes.append(f'- Quadratic seed {run["seed"]}: training mean {100*run["training"]["final"]["mean"]:.4f}%, '
            f'initial-state validation mean {100*fit["groups"]["initial"]["mean"]:.4f}%, later-state mean '
            f'{100*fit["groups"]["later"]["mean"]:.4f}%. Maximum budget change {run["budget_change_max"]:.6e}. '
            f'Maximum invariant stationarity {max(ca["invariant_stationarity"]):.6e}. '
            f'Worst-state normalized gradient {ca["worst_optimality"]:.6e}; '
            f'unconverged snapshot IDs {ca["nonstationary_snapshot_ids"]}. '
            f'Effective mean gate: {"pass" if checked["effective_mean_pass"] else "fail"}; '
            f'worst-error gate: {"pass" if checked["worst_error_pass"] else "fail"}; '
            f'oracle/POD ratio {checked["oracle_over_pod_K"]:.6f}. '
            f'Independent polynomial curvature audit found {ca["stationary_negative_curvature_count"]} '
            'selected stationary fits with negative objective curvature at the declared numerical cutoff.')
    notes += ['', f'Job {config["slurm_job"]} ran on {config["gpu"]} from source commit `{config["commit"]}`. '
        'Source content, output/log checksums, unchanged source bank, validation membership, and the common '
        'training schedule were independently verified. The audit recomputes loss, gradient, singular values, '
        'tangent error and projected stationarity using the explicit polynomial without importing the model '
        'or its differentiation implementation. Host affinity warnings are recorded separately in the audit. '
        'The bank-only mean validation error is '
        f'{100*report["bank_error"]["mean"]:.4f}%.', '',
        'Local multistart fits do not certify global reconstruction minima. Converged worst cases and '
        'unconverged fits must be distinguished before diagnosing representation. No inherited pilot, '
        'test-data evaluation, rollout or cost experiment is included in this bounded screen.', '',
        '[Raw screen](runs/b3d_architecture/screen40/out/result.json) · '
        '[Independent audit](runs/b3d_arch_quadratic/screen40_audit.json) · '
        '[Design](B3D-QUADRATIC-DESIGN.md)', '', '## Glossary', '',
        '- **Head / latent code:** coefficient-generating model / its adjustable reduced coordinates.',
        '- **Mean / median / worst:** relative field reconstruction errors over validation states.',
        '- **Above 15%:** count exceeding the unchanged worst-error threshold.',
        '- **Unconverged:** selected fits exceeding the inherited normalized-gradient threshold.',
        '- **Tangent mean:** average fraction of truth-state PDE velocity outside available decoder directions.',
        '- **Trainable parameters:** shared model weights, excluding per-snapshot optimized codes.',
        '- **Optimizer seed:** minibatch randomness with the same PDE data and deterministic quadratic initialization.',
        '- **Initial / later states:** unseen states before / after PDE time stepping.',
        '- **Budget change:** largest relative error change when doubling latent-fit attempts.',
        '- **Invariant stationarity:** reconstruction residual projected into decoder directions and normalized by full residual.',
        '- **Normalized gradient:** the inherited first-order local-fit convergence diagnostic.',
        '- **Snapshot ID:** trajectory index times the number of saved times plus its time index.',
        '- **Polynomial curvature:** second derivatives; negative objective curvature can rule out a local minimum.',
        '- **Numerical cutoff:** declared tolerance for deciding whether a computed quantity differs from zero.',
        '- **Bank:** fixed learned spatial features; its unrestricted coefficients give a reconstruction floor.',
        '- **POD / oracle:** training-derived linear projection comparator / truth-assisted local latent fit.',
        '- **Pilot / rollout:** inherited validation checks / prediction forward in time.', '']
    (src/'B3D-QUADRATIC-NOTES.md').write_text('\n'.join(notes))
    print(json.dumps(dict(status=audit['status'], job=audit['job'], runs=[dict(seed=x['seed'],
        nonstationary=x['stages'][-1]['nonstationary_snapshot_ids'],
        worst_optimality=x['stages'][-1]['worst_optimality']) for x in audit['runs']]), indent=2))


if __name__ == '__main__':
    main()
