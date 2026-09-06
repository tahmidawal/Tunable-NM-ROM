"""Verify the two initial controls and save cross-job input parity evidence."""
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import numpy as np

here = Path(__file__).resolve().parent
repo = here.parents[3]
labels = ['mlp128', 'mlp192']
reports, checkpoints = [], []
audit = dict(status='verified', jobs={}, cross_job={})
for label in labels:
    record = here / label
    subprocess.run(['sha256sum', '-c', 'PULL.sha256', '--quiet'], cwd=record, check=True)
    subprocess.run(['sha256sum', '-c', 'RESULTS.sha256', '--quiet'], cwd=record, check=True)
    report = json.loads((record / 'out/result.json').read_text())
    config = report['config']
    assert report['complete'] and config['backend'] == 'gpu'
    assert config['x64'] and config['matmul_precision'] == 'highest'
    assert config['optimizer_seeds'] == [200, 201] and len(report['runs']) == 2
    assert (record / 'EXIT_CODE').read_text().strip() == '0'
    assert (record / 'COMMIT.txt').read_text().strip() == config['commit']
    for line in (record / 'INPUTS.sha256').read_text().splitlines():
        expected, name = line.split('  ', 1)
        if name.startswith('code/'):
            source = subprocess.check_output(['git', '-C', str(repo), 'show',
                f'{config["commit"]}:experiments/separable-decoder/{name[5:]}'])
            assert hashlib.sha256(source).hexdigest() == expected
    logs = '\n'.join(p.read_text() for p in (record / 'logs').glob('*'))
    assert 'jax_backend=gpu' in logs
    for forbidden in ['Captured constant', 'captured constant', 'out of memory', 'No space left', 'Traceback']:
        assert forbidden not in logs, forbidden
    raw = report['provenance']['fields']
    assert all(raw[k]['exact'] for k in ['seed', 'm', 'B', 'c', 'w', 'rho', 'A'])
    for run in report['runs']:
        path = record / 'out' / f'mlp_control_seed{run["seed"]}.pkl'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == run['checkpoint_sha256']
    with (record / 'out/mlp_control_seed200.pkl').open('rb') as stream:
        checkpoint = pickle.load(stream)
    reports.append(report); checkpoints.append(checkpoint)
    audit['jobs'][label] = dict(job=config['slurm_job'], source_commit=config['commit'],
        source_content_verified=True, output_and_log_checksums_verified=True,
        backend='gpu', x64=True, matmul_precision='highest',
        raw_parameter_draws_exact=True, parameter_provenance=report['provenance'],
        host_affinity_warning='hwloc_set_cpubind' in logs)

assert reports[0]['config']['source_sha256'] == reports[1]['config']['source_sha256']
np.testing.assert_array_equal(checkpoints[0]['r'], checkpoints[1]['r'])
for key in checkpoints[0]['frozen']:
    a, b = [np.asarray(p['frozen'][key]) for p in checkpoints]
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    audit['cross_job']['shared_' + key] = dict(exact=bool(np.array_equal(a,b)), max_absolute=float(np.max(np.abs(a-b))))
for key in reports[0]['validation_states']:
    a, b = [np.asarray(p['validation_states'][key]) for p in reports]
    if key in ['sid', 'blob_count']:
        np.testing.assert_array_equal(a, b)
    else:
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    audit['cross_job']['validation_' + key] = dict(exact=bool(np.array_equal(a,b)), max_absolute=float(np.max(np.abs(a-b))))
audit['cross_job']['note'] = ('Independent GPU regeneration differs at floating-point rounding level; '
    'same raw random draws, membership, source bank, QR matrix and anchor. Shared hashes are not byte-identical.')
(here / 'review/control-provenance.json').write_text(json.dumps(audit, indent=2) + '\n')
print('Control provenance, checksums and cross-job parity verified.')
