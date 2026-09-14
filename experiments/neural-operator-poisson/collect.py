"""Checksum-collect a completed exact job directory, audit, then clean it."""
import argparse
import json
from pathlib import Path
import subprocess
from datetime import datetime,timezone
from dataset import HERE,sha_file
from audit import audit
from audit_stationarity import inspect
from audit_matched import run as audit_matched


def collect(label, retain_remote=False, resume=False):
    record=HERE/'runs'/label
    info=json.loads((record/'submission.json').read_text()); job=info['job_id'];remote=info['remote']
    assert remote==f'/cluster/tufts/paralab/tawal01/no_poisson_20260914/{label}' and job.isdigit()
    queue=subprocess.check_output(['ssh','tufts-login','squeue -h -u tawal01 -o %i'],text=True)
    if job in queue.split(): raise RuntimeError('Job still queued/running')
    subprocess.run(['ssh','tufts-login',f'cd {remote} && test -f EXIT_CODE && sha256sum -c MANIFEST.sha256 --quiet && sha256sum -c RESULTS.sha256 --quiet && find . -type f ! -name ARCHIVE.sha256 -print0 | sort -z | xargs -0 sha256sum > ARCHIVE.sha256'],check=True)
    archive=record/'archive'
    if resume:
        assert (archive/'ARCHIVE.sha256').exists()
    else:
        archive.mkdir(exist_ok=False)
        subprocess.run(['scp','-r',f'tufts-login:{remote}/.',str(archive)],check=True)
    remote_manifest=subprocess.check_output(['ssh','tufts-login',f'sha256sum {remote}/ARCHIVE.sha256'],text=True).split()[0]
    assert remote_manifest==sha_file(archive/'ARCHIVE.sha256')
    subprocess.run(['sha256sum','-c','ARCHIVE.sha256','--quiet'],cwd=archive,check=True)
    exit_code=int((archive/'EXIT_CODE').read_text())
    logs='\n'.join(p.read_text() for p in (archive/'logs').glob('*'))
    assert 'jax_backend=gpu' in logs
    forbidden=['cuInit','CUDA_ERROR','captured constant','large constant','out of memory','No space left','Traceback']
    warnings=[x for x in forbidden if x.lower() in logs.lower()]
    checks=dict(job_id=job,exit_code=exit_code,checksums_verified=True,log_warnings=warnings,
                archive_manifest_sha256=sha_file(archive/'ARCHIVE.sha256'))
    if exit_code==0 and not warnings:
        if (archive/'out/r128').exists():
            summary=audit_matched(archive)
            (record/'matched-summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
            checks['matched_data_and_exposure_audit_pass']=True
        else:
            summary=audit(archive/'out')
            (record/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
            independent=inspect(archive)
            (record/'stationarity-audit.json').write_text(json.dumps(independent,indent=2,allow_nan=False)+'\n')
        checks['numpy_audit_pass']=True
        checks['independent_stationarity_replay_pass']=True
    else:
        checks['numpy_audit_pass']=False
    # Restore evidence is present and checksum verified, including failed jobs.
    if not retain_remote:
        subprocess.run(['ssh','tufts-login',f'test -f {remote}/EXIT_CODE && rm -rf -- {remote} && test ! -e {remote}'],check=True)
    checks.update(remote_exact_directory_removed=not retain_remote,collected_at=datetime.now(timezone.utc).isoformat())
    (record/'collection.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--retain-remote',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args();collect(a.label,a.retain_remote,a.resume)
