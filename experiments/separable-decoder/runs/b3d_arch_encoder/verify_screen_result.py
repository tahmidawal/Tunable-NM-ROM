"""Artifact and independent NumPy checks for this encoder screen only.

This is the implementation worker's check; coordinator independent result
review remains separate. No PDE data or GPU numerical experiment runs here.
"""
import hashlib
import json
from pathlib import Path
import pickle
import subprocess

import numpy as np

SRC = Path(__file__).resolve().parents[2]
REPO = SRC.parents[1]
RECORD = SRC/'runs/b3d_architecture/screen40'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_tree(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            verify_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for aa, bb in zip(a, b):
            verify_tree(aa, bb)
    else:
        np.testing.assert_array_equal(a, b)


def dense(layers, x):
    for w, bias in layers[:-1]:
        x = x @ w + bias
        x = x * (.5+.5*np.tanh(.5*x))
    w, bias = layers[-1]
    return x @ w + bias


def decode(p, frozen, z):
    d = p['decoder']
    return d['bias'] + z@d['linear'].T + frozen['output_scale']*dense(
        d['residual'], z/frozen['latent_scale'])


def encode(p, frozen, a):
    c = a-frozen['center']
    return c@frozen['anchor'] + frozen['latent_scale']*dense(
        p['encoder_residual'], c/frozen['output_scale'])


def jacobian(p, frozen, z):
    d = p['decoder']
    x = z/frozen['latent_scale']
    derivative = np.diag(1/frozen['latent_scale'])
    for w, bias in d['residual'][:-1]:
        x = x@w+bias
        sigmoid = .5+.5*np.tanh(.5*x)
        derivative = (sigmoid+x*sigmoid*(1-sigmoid))[:, None] * (w.T@derivative)
        x = x*sigmoid
    w, _ = d['residual'][-1]
    return d['linear'] + frozen['output_scale']*(w.T@derivative)


def main():
    for name in ('PULL.sha256', 'RESULTS.sha256'):
        subprocess.run(['sha256sum', '-c', name, '--quiet'], cwd=RECORD, check=True)
    report = json.loads((RECORD/'out/result.json').read_text())
    config = report['config']
    submission = json.loads((RECORD/'submission.json').read_text())
    assert report['complete'] and report['kind'] == 'b3d_architecture'
    assert config['model'] == 'b3d_arch_encoder'
    assert config['optimizer_seeds'] == [200, 201]
    assert config['backend'] == 'gpu' and config['x64']
    assert config['matmul_precision'] == 'highest' and not config['test_table_opened']
    assert (RECORD/'EXIT_CODE').read_text().strip() == '0'
    assert (RECORD/'COMMIT.txt').read_text().strip() == config['commit'] == submission['source_commit']
    assert (RECORD/'JID').read_text().strip() == config['slurm_job']
    for line in (RECORD/'INPUTS.sha256').read_text().splitlines():
        expected, name = line.split('  ', 1)
        if name.startswith('code/'):
            source = subprocess.check_output(['git', '-C', str(REPO), 'show',
                f'{config["commit"]}:experiments/separable-decoder/{name[5:]}'])
            assert hashlib.sha256(source).hexdigest() == expected
            if name == 'code/b3d_arch_encoder.py':
                assert expected == config['model_sha256']
    source_path = Path(submission['checkpoint'])
    assert sha(source_path) == config['source_sha256'] == submission['checkpoint_sha256']
    with source_path.open('rb') as stream:
        source_payload = pickle.load(stream)
    source_bank = source_payload['params']
    assert (source_payload['cfg']['N'], source_payload['cfg']['k'],
            source_payload['cfg']['r'], source_payload['cfg']['seed']) == (33, 32, 128, 0)
    logs = '\n'.join(p.read_text() for p in (RECORD/'logs').glob('*'))
    assert 'jax_backend=gpu' in logs
    for forbidden in ('captured constant', 'out of memory', 'no space left', 'traceback', 'oom-kill'):
        assert forbidden not in logs.lower(), forbidden
    fields = report['provenance']['fields']
    assert all(fields[k]['exact'] for k in ('seed', 'm', 'B', 'c', 'w', 'rho', 'A'))
    assert all(fields[k]['max_relative'] <= report['provenance']['derived_rtol'] for k in ('nu', 's_star'))
    val = {k:np.asarray(v) for k,v in report['validation_states'].items()}
    np.testing.assert_array_equal(val['sid'],
        (np.arange(512, 576)[:, None]*51+np.asarray([0, 10, 25, 50])).ravel())
    control = json.loads((SRC/'runs/b3d_architecture/mlp128/out/result.json').read_text())
    with (SRC/'runs/b3d_architecture/mlp128/out/mlp_control_seed200.pkl').open('rb') as stream:
        control_checkpoint = pickle.load(stream)
    comparison = {}
    for key, a in val.items():
        b = np.asarray(control['validation_states'][key])
        if key in ('sid', 'blob_count'):
            np.testing.assert_array_equal(a, b)
        else:
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
        comparison[key] = dict(exact=bool(np.array_equal(a, b)),
                               maximum_absolute=float(np.max(np.abs(a-b))))
    checks = []
    for run in report['runs']:
        path = RECORD/'out'/f'shared_encoder_seed{run["seed"]}.pkl'
        assert sha(path) == run['checkpoint_sha256']
        with path.open('rb') as stream:
            payload = pickle.load(stream)
        verify_tree(payload['bank_params'], source_bank)
        assert payload['source'] == config
        p, frozen = payload['trainable'], payload['frozen']
        np.testing.assert_array_equal(payload['r'], control_checkpoint['r'])
        for key, value in frozen.items():
            np.testing.assert_allclose(value, control_checkpoint['frozen'][key],
                                       rtol=1e-12, atol=1e-12)
        assert payload['codes'].shape == (8192, 32)
        assert run['training']['optimized_code_count'] == 0
        assert (run['training']['steps'], run['training']['lr'], run['training']['batch']) == (60000, 3e-4, 4096)
        assert run['training']['norm'] == 'global' and run['training']['mode'] == 'encoder'
        assert run['training']['initialization_code_max_difference'] < 1e-12
        assert run['training']['initialization_field_max_difference'] < 1e-12
        np.testing.assert_array_equal(run['start_training_indices'],
            np.random.default_rng(5).choice(8192, 7, replace=False))
        train_sids = np.sort(np.random.default_rng(0).choice(512*51, 8192, replace=False))
        np.testing.assert_array_equal(run['start_snapshot_ids'], train_sids[run['start_training_indices']])
        direct_q = decode(p, frozen, encode(p, frozen, val['target']))
        direct_error = np.sqrt((np.sum((direct_q-val['target'])**2, axis=1)+
                                val['perpendicular2'])/val['norm2'])
        np.testing.assert_allclose(direct_error, run['direct_encoder_error'], rtol=1e-10, atol=1e-12)
        seed_check = dict(seed=run['seed'], direct_encoder_max_difference=float(
            np.max(np.abs(direct_error-run['direct_encoder_error']))), stages=[])
        for stage in run['fits']:
            z = np.asarray(stage['z'])
            residual = decode(p, frozen, z)-val['target']
            full2 = np.sum(residual**2, axis=1)+val['perpendicular2']
            error = np.sqrt(full2/val['norm2'])
            np.testing.assert_allclose(error, stage['error'], rtol=1e-10, atol=1e-12)
            opt, tangent = [], []
            for i, zz in enumerate(z):
                jac = jacobian(p, frozen, zz)
                opt.append(np.linalg.norm(jac.T@residual[i]) /
                           (np.linalg.norm(jac)*np.sqrt(full2[i])+1e-300))
                velocity = val['velocity_target'][i]
                dz = np.linalg.lstsq(jac, velocity, rcond=config['rank_rtol'])[0]
                tangent.append(np.sqrt((np.sum((velocity-jac@dz)**2)+
                    val['velocity_perpendicular2'][i])/val['velocity_norm2'][i]))
            np.testing.assert_allclose(opt, stage['optimality'], rtol=1e-5, atol=1e-10)
            expected = [g['tangent_relative'] for g in stage['geometry']]
            np.testing.assert_allclose(tangent, expected, rtol=1e-10, atol=1e-12)
            seed_check['stages'].append(dict(budget=stage['budget'],
                error_max_difference=float(np.max(np.abs(error-stage['error']))),
                optimality_max_difference=float(np.max(np.abs(np.asarray(opt)-stage['optimality']))),
                tangent_max_difference=float(np.max(np.abs(np.asarray(tangent)-expected)))))
        checks.append(seed_check)
    output = dict(status='worker_verified_pending_coordinator_independent_review',
                  job=config['slurm_job'], source_commit=config['commit'],
                  checksums_verified=True, source_content_matches_git=True,
                  checkpoint_bank_unchanged=True, provenance=report['provenance'],
                  validation_matches_control=comparison, independent_numpy_checks=checks,
                  host_affinity_warning='hwloc_set_cpubind' in logs)
    (RECORD/'verification.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({'status':output['status'], 'checks':checks}, indent=1))


if __name__ == '__main__':
    main()
