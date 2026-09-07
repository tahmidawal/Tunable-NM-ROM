"""Audit architecture results across isolated worktrees without writing to them.

Output is campaign-audit.json in the coordinator worktree. Incomplete/invalid
attempts remain visible. Scientific acceptance requires complete verified jobs;
this script's preliminary screen is not the inherited pilot or a rollout.
"""
import hashlib
import json
from pathlib import Path
import pickle
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1]
REPO = SRC.parents[1]
WORKTREES = REPO.parent
ARMS = ('anchor', 'quadratic', 'encoder', 'mixture')


def candidates():
    found = []
    for name in ('mlp128', 'mlp192'):
        found.append(HERE / name / 'out/result.json')
    for arm in ARMS:
        root = WORKTREES / f'2026-09-06-b3d-{arm}' / 'experiments/separable-decoder/runs/b3d_architecture'
        for path in sorted(root.glob('*/out/result.json')):
            if path.parent.parent.name not in ('mlp128', 'mlp192'):
                found.append(path)
    return found


def assert_summary(values, reported):
    a = np.asarray(values)
    expected = dict(n=a.size, mean=a.mean(), median=np.median(a),
                    p95=np.quantile(a, .95), worst=a.max())
    for key, value in expected.items():
        np.testing.assert_allclose(value, reported[key], rtol=1e-13, atol=1e-14)


def audit(path, reference):
    report = json.loads(path.read_text())
    if report.get('kind') != 'b3d_architecture':
        return None
    config = report['config']
    record = path.parent.parent
    result = dict(path=str(path), model=config['model'], name=config['name'],
                  width=config.get('model_config', {}).get('width'), job=config['slurm_job'],
                  source_commit=config['commit'], status='invalid', complete=report.get('complete', False))
    try:
        for manifest in ('PULL.sha256', 'RESULTS.sha256'):
            subprocess.run(['sha256sum', '-c', manifest, '--quiet'], cwd=record, check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert (record / 'COMMIT.txt').read_text().strip() == config['commit']
        for line in (record / 'INPUTS.sha256').read_text().splitlines():
            expected, name = line.split('  ', 1)
            if name.startswith('code/'):
                content = subprocess.check_output(['git', '-C', str(REPO), 'show',
                    f'{config["commit"]}:experiments/separable-decoder/{name[5:]}'])
                assert hashlib.sha256(content).hexdigest() == expected, name
            elif name == 'in/checkpoint.pkl':
                assert expected == config['source_sha256'], 'staged checkpoint provenance'
            elif name == 'in/reference_table.npz':
                original = SRC/'runs/b3d_repair/reference/archived_parameter_manifest.npz'
                assert hashlib.sha256(original.read_bytes()).hexdigest() == expected
        logs = '\n'.join(p.read_text() for p in (record / 'logs').glob('*'))
        assert 'jax_backend=gpu' in logs and config['backend'] == 'gpu'
        assert config['x64'] and config['matmul_precision'] == 'highest'
        assert config['source_sha256'] == reference['config']['source_sha256']
        assert config['source_config'] == reference['config']['source_config']
        assert config['optimizer_seeds'] == [200, 201]
        assert config['test_table_opened'] is False
        result['host_affinity_warning'] = 'hwloc_set_cpubind' in logs
        for phrase in ('Captured constant', 'captured constant', 'out of memory', 'No space left'):
            assert phrase not in logs, phrase
        for name in ('seed', 'm', 'B', 'c', 'w', 'rho', 'A'):
            assert report['provenance']['fields'][name]['exact'], name
        provenance = report['provenance']
        assert provenance['archived_sha256'] == config['source_config']['table_sha256']
        assert provenance['derived_rtol'] == 8*np.finfo(float).eps
        for name in ('nu', 's_star'):
            difference = provenance['fields'][name]['max_relative']
            assert np.isfinite(difference) and 0 <= difference <= provenance['derived_rtol'], name
        parity = {}
        for name, array in reference['validation_states'].items():
            actual, expected = np.asarray(report['validation_states'][name]), np.asarray(array)
            if name in ('sid', 'blob_count'):
                np.testing.assert_array_equal(actual, expected)
            else:
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
            parity[name] = float(np.max(np.abs(actual-expected)))
        result['validation_parity_max_absolute'] = parity
        result['runs'] = []
        if report.get('complete'):
            assert [row['seed'] for row in report['runs']] == config['optimizer_seeds']
        for row in report.get('runs', []):
            checkpoint = path.parent / f'{config["name"]}_seed{row["seed"]}.pkl'
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == row['checkpoint_sha256']
            with checkpoint.open('rb') as stream:
                payload = pickle.load(stream)
            assert payload['kind'] == 'b3d_arch_checkpoint' and payload['model'] == config['model']
            assert payload['source']['model_sha256'] == config['model_sha256']
            assert row['training']['steps'] == 60000 and row['training']['batch'] == 4096
            assert row['training']['lr'] == 3e-4 and row['training']['seed'] == row['seed']
            if not row.get('fits'):
                result['runs'].append(dict(seed=row['seed'], status='incomplete_fit'))
                continue
            assert [f['budget'] for f in row['fits']] == [400, 800]
            fit = row['fits'][-1]
            errors, optimality = np.asarray(fit['error']), np.asarray(fit['optimality'])
            assert np.isfinite(errors).all() and np.isfinite(optimality).all()
            assert_summary(errors, fit['summary'])
            assert fit['nonstationary'] == int(np.sum(optimality > 1e-6))
            assert fit['outliers_above_15pct'] == int(np.sum(errors > .15))
            assert np.asarray(fit['all_errors']).shape == (256, 8)
            np.testing.assert_allclose(np.min(fit['all_errors'], axis=1), errors, rtol=0, atol=0)
            assert_summary([g['tangent_relative'] for g in fit['geometry']], fit['tangent_summary'])
            budget_change = np.max(np.abs(errors-np.asarray(row['fits'][0]['error'])) / row['fits'][0]['error'])
            np.testing.assert_allclose(budget_change, row['budget_change_max'], rtol=1e-13, atol=1e-15)
            seed_reference = next(x for x in reference['runs'] if x['seed'] == row['seed'])
            np.testing.assert_array_equal(row['start_snapshot_ids'], seed_reference['start_snapshot_ids'])
            result['runs'].append(dict(seed=row['seed'], status='verified',
                mean=float(errors.mean()), median=float(np.median(errors)), worst=float(errors.max()),
                outliers=fit['outliers_above_15pct'], unconverged=fit['nonstationary'],
                worst_is_stationary=bool(optimality[np.argmax(errors)] <= 1e-6),
                maximum_optimality=float(optimality.max()),
                maximum_invariant=float(max(g['invariant_stationarity'] for g in fit['geometry'])),
                minimum_rank=int(min(g['rank'] for g in fit['geometry'])),
                tangent_mean=fit['tangent_summary']['mean']))
        result['status'] = 'verified' if report.get('complete') and (record/'EXIT_CODE').read_text().strip() == '0' else 'incomplete'
        if result['status'] == 'verified':
            assert len(result['runs']) == 2 and all(x['status'] == 'verified' for x in result['runs'])
        else:
            result['failure'] = report.get('failure_message', 'job incomplete')
    except Exception as error:
        result['status'] = 'invalid'
        result['failure'] = f'{type(error).__name__}: {error}'
    return result


def main():
    reference = json.loads((HERE/'mlp128/out/result.json').read_text())
    records = [audit(p, reference) for p in candidates() if p.exists()]
    records = [r for r in records if r is not None]
    seen = {r['model'] for r in records}
    no_output = []
    for arm in ARMS:
        folder = WORKTREES / f'2026-09-06-b3d-{arm}' / 'experiments/separable-decoder/runs/b3d_architecture'
        for submission in sorted(folder.glob('*/submission.json')):
            metadata = json.loads(submission.read_text())
            if metadata.get('script') != 'b3d_arch_bench.py' or metadata.get('env', {}).get('MODEL') != f'b3d_arch_{arm}':
                continue
            if not (submission.parent/'out/result.json').exists():
                no_output.append(dict(model=f'b3d_arch_{arm}', label=metadata['label'],
                    path=str(submission), remote=metadata['remote'],
                    status='no_numerical_output', note='Pending/canceled/preflight attempt; inspect preserved submission and accounting records.'))
    result = dict(records=records, attempts_without_output=no_output,
                  missing_models=[f'b3d_arch_{a}' for a in ARMS if f'b3d_arch_{a}' not in seen])
    output = HERE/'review/campaign-audit.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    for row in records:
        print(row['name'], row['width'], row['job'], row['status'], row.get('failure', ''))
    print('Missing models:', ', '.join(result['missing_models']) or 'none')


if __name__ == '__main__':
    main()
