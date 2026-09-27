"""Bounded local collection monitor for the owned Burgers FNO job.

No GPU work, training or model selection occurs here. It waits for the exact
owned Slurm job, verifies the transferred archive and staged source hashes,
independently recomputes every validation error with NumPy using the Burgers
lane's fixed-initial metric, re-derives the timing medians from the retained
repetition arrays, preserves the archive as ordered Git-tracked parts, and only
then removes the exact remote job directory. It never touches the shared
`pilot-data01` cache.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shlex
import subprocess
import tarfile
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
NAMESPACE = '/cluster/tufts/paralab/tawal01/no_audit_20260914'
NAME = 'fno_burgers01'
JOB_NAME = 'ctol_noa_fno_b01'
REMOTE = NAMESPACE + '/' + NAME
# Retained: every `best.pt`, every saved validation prediction, all loss curves,
# the timing repetition arrays and the model-facing validation cases.
# Excluded: the final-epoch optimiser state (training is deliberately not
# resumable and `best.pt` carries the selected weights, normalisation and its
# own hash), package caches, and the training cases and solver sidecars, which
# are already archived and hash-linked by the Burgers lane.
EXCLUDES = ['--exclude=*/last.pt', '--exclude=*.solver.npz',
            '--exclude=fno_burgers01/cache', '--exclude=fno_burgers01/tmp',
            '--exclude=fno_burgers01/data/train']


def ssh(command):
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                                    'tufts-login', command], text=True, timeout=1800)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def statistics(values):
    x = np.asarray(values, dtype=np.float64)
    q1, q3 = np.quantile(x, [.25, .75])
    return dict(mean=float(x.mean()), median=float(np.median(x)), maximum=float(x.max()),
                p95=float(np.quantile(x, .95)), minimum=float(x.min()),
                upper_tukey_outliers=int((x > q3 + 1.5 * (q3 - q1)).sum()),
                above_threshold_counts={str(t): int((x > t).sum()) for t in (.01, .02, .05)})


def fixed_initial_errors(prediction, target):
    """The Burgers lane metric: interior l2 discrepancy over the interior l2
    norm of the reference initial field, per requested output time."""
    denominator = np.linalg.norm(target[0, 0, 1:-1, 1:-1])
    assert np.isfinite(denominator) and denominator > 0
    return np.linalg.norm((prediction - target)[:, 0, 1:-1, 1:-1].reshape(len(target), -1), axis=1) / denominator


def audit_model(folder, rows, data_root):
    summary = json.loads((folder / 'result.json').read_text())
    per_case, per_case_times, declared = [], [], []
    for index, row in enumerate(rows):
        with np.load(data_root / row['path']) as case:
            target, supplied, times = case['target'].copy(), case['input'].copy(), case['times'].copy()
        assert sha(data_root / row['path']) == row['sha256']
        with np.load(folder / (row['case_id'] + '.prediction.npz')) as saved:
            prediction = saved['prediction'].copy()
        assert prediction.shape == target.shape and prediction.dtype == np.float64
        assert np.isfinite(prediction).all()
        for edge in (prediction[..., 0, :], prediction[..., -1, :], prediction[..., :, 0], prediction[..., :, -1]):
            assert not np.any(edge)
        assert np.array_equal(prediction[0], supplied), 'Supplied initial state must be returned exactly'
        assert times.size == 6 and times[0] == 0
        errors = fixed_initial_errors(prediction, target)
        assert errors[0] == 0
        stated = np.asarray(summary['validation']['errors'][index], dtype=np.float64)
        assert np.allclose(errors, stated, rtol=1e-11, atol=1e-13), (row['case_id'], errors, stated)
        per_case.append(float(errors.max()))
        per_case_times.append(errors.tolist())
        declared.append(stated.tolist())
    assert sha(folder / 'best.pt') == summary['best_checkpoint_sha256']
    return dict(complete=True, best_epoch=summary['best_epoch'], epochs_completed=summary['epochs_completed'],
                stopped_by_wall_budget=summary['stopped_by_wall_budget'],
                stopped_by_signal=summary['stopped_by_signal'],
                training_seconds=summary['training_seconds'], config=json.loads((folder / 'provenance.json').read_text())['config'],
                real_parameter_count=summary['real_parameter_count'],
                parameter_tensor_elements=summary['parameter_tensor_elements'],
                best_checkpoint_sha256=summary['best_checkpoint_sha256'],
                fixed_initial=statistics(per_case), case_maximum_errors=per_case,
                per_time_errors=per_case_times, declared_errors=declared,
                validation_case_ids=[r['case_id'] for r in rows])


def audit_timing(root):
    folder = root / 'out/timing'
    if not (folder / 'timing.json').exists():
        return dict(present=False, note='Timing block did not complete inside the allocation')
    stated = json.loads((folder / 'timing.json').read_text())
    arrays = np.load(folder / 'timing.npz')
    checked = {}
    for name, model in stated['models'].items():
        device, host = arrays[f'device_{name}'], arrays[f'host_{name}']
        assert device.shape == host.shape and device.shape[0] == len(stated['cases'])
        assert np.isfinite(device).all() and np.isfinite(host).all() and (device > 0).all()
        recomputed = float(np.median(device) * 1e3)
        assert abs(recomputed - model['device_query_pooled']['median_ms']) <= 1e-9 * max(1., recomputed)
        recomputed_host = float(np.median(host) * 1e3)
        assert abs(recomputed_host - model['host_transfer_pooled']['median_ms']) <= 1e-9 * max(1., recomputed_host)
        checked[name] = dict(repetitions=list(device.shape),
                             device_pooled_median_ms=recomputed,
                             device_median_of_case_medians_ms=float(np.median(np.median(device, axis=1)) * 1e3),
                             host_pooled_median_ms=recomputed_host,
                             device_plus_host_pooled_median_ms=float(np.median(device + host) * 1e3))
    return dict(present=True, gpu=stated['environment']['gpu'], models=checked,
                repetitions_per_case=stated['repetitions_per_case'], burn_in_per_block=stated['burn_in_per_block'],
                limitation='Same-job, same-GPU FNO timings only; never divide these by another job\'s timings.')


def audit(root):
    rows = json.loads((root / 'data/validation/index.json').read_text())['records']
    data_root = root / 'data/validation'
    models = {}
    for folder in sorted((root / 'out').glob('fno-*')):
        if (folder / 'result.json').exists():
            models[folder.name] = audit_model(folder, rows, data_root)
        else:
            models[folder.name] = dict(complete=False)
    selection = root / 'out/capacity-selection.json'
    return dict(numpy_fields_passed=bool(models) and all(m.get('complete') for m in models.values()),
                models=models, validation_cases=len(rows),
                capacity_selection=json.loads(selection.read_text()) if selection.exists() else None,
                worker=json.loads((root / 'out/worker.json').read_text()),
                timing=audit_timing(root),
                metric='maximum over the six requested output times of the interior l2 discrepancy '
                       'divided by the interior l2 norm of the supplied initial field',
                limitation='Single-seed, bounded-epoch development screen on one mesh and one Gaussian '
                           'continuum family; no ROM/FOM comparison and no cross-job timing ratio.')


def main(job_id):
    assert job_id.isdecimal()
    folder = HERE / 'runs' / NAME
    folder.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + 10 * 3600
    accounting = ''
    while time.monotonic() < deadline:
        try:
            queued = ssh(f'squeue -u tawal01 -h -o "%i %j %T" | awk \'$1 == "{job_id}"\'').strip()
            if queued:
                assert queued.split()[0] == job_id and queued.split()[1] == JOB_NAME
                print(queued, flush=True)
                time.sleep(60)
                continue
            accounting = ssh(f'sacct -X -j {job_id} -n -P -o JobID,JobName%40,State,Elapsed,NodeList')
            assert job_id in accounting and JOB_NAME in accounting
            (folder / 'accounting.txt').write_text(accounting)
            break
        except subprocess.SubprocessError as error:
            print(str(error), flush=True)
            time.sleep(60)
    else:
        raise RuntimeError('Monitor deadline reached; remote directory preserved')
    ssh(f'tar -C {shlex.quote(NAMESPACE)} {" ".join(EXCLUDES)} -czf {shlex.quote(REMOTE + ".tar.gz")} {NAME}')
    expected = ssh(f'sha256sum {shlex.quote(REMOTE + ".tar.gz")}').split()[0]
    archive = folder / (NAME + '.tar.gz')
    subprocess.run(['scp', '-q', 'tufts-login:' + REMOTE + '.tar.gz', str(archive)], check=True)
    assert sha(archive) == expected
    (folder / 'archive.sha256').write_text(f'{expected}  {archive.name}\n')
    with tarfile.open(archive, 'r:gz') as tar:
        tar.extractall(folder, filter='data')
    root = folder / NAME
    for line in (root / 'MANIFEST.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        assert sha(root / name) == digest, name
    result = audit(root)
    result.update(job_id=job_id, archive_sha256=expected, accounting=accounting,
                  archive_exclusions=EXCLUDES, archive_bytes=archive.stat().st_size)
    (folder / 'field-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    chunks = folder / 'archive-parts'
    chunks.mkdir()
    manifest = dict(archive_sha256=expected, exclusions=EXCLUDES, parts=[])
    with archive.open('rb') as stream:
        index = 0
        while data := stream.read(64 * 1024 * 1024):
            path = chunks / f'part-{index:04d}'
            path.write_bytes(data)
            manifest['parts'].append(dict(path=path.name, sha256=sha(path), bytes=len(data)))
            index += 1
    (chunks / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    archive.unlink()
    worktree = HERE.parents[1]
    keep = [folder / 'field-audit.json', folder / 'accounting.txt', folder / 'archive.sha256', chunks]
    subprocess.run(['git', '-C', str(worktree), 'add', '-f', *[str(p) for p in keep]], check=True)
    subprocess.run(['git', '-C', str(worktree), 'commit', '--only',
                    '-m', f'Archive and audit Burgers FNO job {job_id}', *[str(p) for p in keep]], check=True)
    ssh(f'rm -rf -- {shlex.quote(REMOTE)} && rm -- {shlex.quote(REMOTE + ".tar.gz")}')
    print(json.dumps({k: v for k, v in result.items() if k != 'models'}), flush=True)
    print('COLLECTION FINISHED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', required=True)
    arguments = parser.parse_args()
    try:
        main(arguments.job_id)
    except BaseException:
        traceback.print_exc()
        raise
