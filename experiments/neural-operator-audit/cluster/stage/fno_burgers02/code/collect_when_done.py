"""Bounded local collection monitor for one owned Slurm job.

No GPU work or model selection occurs here. Collection and field auditing can
continue while the main conversation is available to the user.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import shlex
import shutil
import subprocess
import tarfile
import time
import traceback
import numpy as np

HERE = Path(__file__).resolve().parent
NAMESPACE = '/cluster/tufts/paralab/tawal01/no_audit_20260914'
NAME = 'fno_poisson01'
REMOTE = NAMESPACE + '/' + NAME
LAB = Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/LAB-LOG.md')


def ssh(command):
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                                    'tufts-login', command], text=True, timeout=600)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(root):
    rows = json.loads((root/'data/validation/index.json').read_text())['records']
    result = {}
    for size in ('small', 'medium', 'large'):
        folder = root/'out'/('fno-'+size)
        if not (folder/'result.json').exists():
            result[size] = dict(complete=False)
            continue
        summary = json.loads((folder/'result.json').read_text())
        discrete, physical = [], []
        for i, row in enumerate(rows):
            with np.load(root/'data/validation'/row['path']) as data:
                target = data['target'].copy()
            with np.load(folder/(row['case_id']+'.prediction.npz')) as data:
                prediction = data['prediction'].copy()
            assert prediction.shape == target.shape and prediction.dtype == np.float64
            assert np.isfinite(prediction).all()
            for edge in (prediction[...,0,:], prediction[...,-1,:], prediction[...,:,0], prediction[...,:,-1]):
                assert not np.any(edge)
            e = np.linalg.norm(prediction-target)/np.linalg.norm(target)
            assert np.isclose(e, summary['validation']['errors'][i][0], rtol=1e-11, atol=1e-13)
            discrete.append(float(e))
            ref = row['reference']
            path = root/'data/validation'/ref['path']
            assert sha(path) == ref['sha256']
            with np.load(path) as data:
                target = data['target'].copy()
            physical.append(float(np.linalg.norm(prediction-target)/np.linalg.norm(target)))
        assert sha(folder/'best.pt') == summary['best_checkpoint_sha256']
        def statistics(values):
            x = np.asarray(values)
            q1,q3 = np.quantile(x, [.25,.75])
            return dict(mean=float(x.mean()), median=float(np.median(x)), maximum=float(x.max()),
                        p95=float(np.quantile(x,.95)), upper_tukey_outliers=int((x>q3+1.5*(q3-q1)).sum()),
                        above_threshold_counts={str(t):int((x>t).sum()) for t in (.01,.02,.05)})
        result[size] = dict(complete=True, best_epoch=summary['best_epoch'],
                           discrete=statistics(discrete), physical_candidate=statistics(physical),
                           discrete_errors=discrete, physical_candidate_errors=physical)
    return dict(numpy_fields_passed=any(r['complete'] for r in result.values()),
        all_models_complete=all(r['complete'] for r in result.values()), models=result,
        limitation='Single-seed development models and empirical numerical references; no paired runtime or FOM/ROM speed comparison.')


def main(job_id):
    assert job_id.isdecimal()
    folder = HERE/'runs'/NAME
    folder.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + 8*3600
    while time.monotonic() < deadline:
        try:
            queued = ssh(f'squeue -u tawal01 -j {job_id} -h -o "%i %j %T"').strip()
            if queued:
                assert queued.split()[0] == job_id and queued.split()[1] == 'ctol_noa_fno_p01'
                print(queued, flush=True)
                time.sleep(30)
                continue
            accounting = ssh(f'sacct -X -j {job_id} -n -P -o JobID,JobName%40,State,Elapsed,NodeList')
            assert job_id in accounting and 'ctol_noa_fno_p01' in accounting
            (folder/'accounting.txt').write_text(accounting)
            break
        except subprocess.SubprocessError as e:
            print(str(e), flush=True)
            time.sleep(30)
    else:
        raise RuntimeError('Monitor deadline reached; remote directory preserved')
    # Build an archive only after Slurm releases the exact owned job.
    ssh(f'tar -C {shlex.quote(NAMESPACE)} -czf {shlex.quote(REMOTE+".tar.gz")} {NAME}')
    expected = ssh(f'sha256sum {shlex.quote(REMOTE+".tar.gz")}').split()[0]
    archive = folder/(NAME+'.tar.gz')
    subprocess.run(['scp', '-q', 'tufts-login:'+REMOTE+'.tar.gz', str(archive)], check=True)
    assert sha(archive) == expected
    (folder/'archive.sha256').write_text(f'{expected}  {archive.name}\n')
    with tarfile.open(archive, 'r:gz') as tar:
        tar.extractall(folder, filter='data')
    root = folder/NAME
    for line in (root/'MANIFEST.sha256').read_text().splitlines():
        digest, name = line.split('  ',1)
        assert sha(root/name) == digest
    result = audit(root)
    result.update(job_id=job_id, archive_sha256=expected, accounting=accounting)
    (folder/'field-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    # Retain bounded Git blobs and an exact reconstruction manifest.
    chunks = folder/'archive-parts'
    chunks.mkdir()
    manifest = dict(archive_sha256=expected, parts=[])
    with archive.open('rb') as stream:
        i=0
        while data := stream.read(64*1024*1024):
            path = chunks/f'part-{i:04d}'
            path.write_bytes(data)
            manifest['parts'].append(dict(path=path.name, sha256=sha(path), bytes=len(data)))
            i+=1
    (chunks/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    archive.unlink()
    wt = HERE.parents[1]
    keep = [folder/'field-audit.json', folder/'accounting.txt', folder/'archive.sha256', chunks]
    subprocess.run(['git','-C',str(wt),'add','-f',*[str(p) for p in keep]], check=True)
    subprocess.run(['git','-C',str(wt),'commit','--only','-m',f'Archive and audit FNO capacity job {job_id}',
                    *[str(p) for p in keep]], check=True)
    ssh(f'rm -rf -- {shlex.quote(REMOTE)} && rm -- {shlex.quote(REMOTE+".tar.gz")}')
    completed = time.strftime('%Y-%m-%d', time.gmtime())
    text = (f'\n\n## {completed}\n### FNO capacity worker {job_id} — collected automatically\n\n'
            f'The owned Poisson FNO worker ended and was checksummed, independently field-audited '
            f'and archived at `{folder}`. Source hashes and the checkpoints, saved predictions '
            f'and declared discrete errors of completed models were verified; missing models '
            f'are listed explicitly. Physical-candidate error arrays and '
            f'mean/median/tail/outlier statistics are generated in `field-audit.json`. '
            f'Archive SHA256 `{expected}`; bounded archive parts and audit records are Git-tracked. '
            f'The exact completed remote job directory and transfer archive were removed. '
            f'Accounting: `{accounting.strip()}`. This is single-seed development evidence; '
            f'no paired FNO/ROM/FOM timing claim or final-cohort access occurred. '
            f'The broader campaign and user merge decision remain open.\n')
    with LAB.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(text)
        f.flush()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    try:
        main(args.job_id)
    except BaseException:
        traceback.print_exc()
        raise
