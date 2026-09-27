"""Stage pinned source and explicitly hashed immutable assets for a follow-up.

Training data are never staged. This does not mutate an existing attempt.
"""
import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path
from stage import ROOT, EXP, REMOTE, command, ssh


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    cfg=json.loads((EXP/a.config).read_text());attempt=cfg['attempt']
    assert attempt.isalnum() and attempt.startswith('b3d')
    before=ssh(['squeue','-u','tawal01','-h','-o','%i|%j|%T']);print('QUEUE BEFORE\n'+before,flush=True)
    assert 'ctol_paper_b3d' not in before
    assert len(before.splitlines())<4,'Four allocated/queued jobs already present'
    source=command(['git','-C',str(ROOT),'rev-parse','HEAD'])
    assert not command(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=no','--','experiments/paper-b3d'])
    run=EXP/'runs'/attempt;run.mkdir(parents=True,exist_ok=False);stage=run/'stage';stage.mkdir()
    manifest=[]
    for entry in command(['git','-C',str(ROOT),'ls-files','experiments/paper-b3d']).splitlines():
        relative=Path(entry).relative_to('experiments/paper-b3d')
        if relative.parts[0] in ['runs','checks','final-assets']:continue
        if relative.name=='.gitignore':continue
        data=(ROOT/entry).read_bytes();assert data==subprocess.check_output(['git','-C',str(ROOT),'show',source+':'+entry])
        target=stage/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        manifest.append(hashlib.sha256(data).hexdigest()+'  '+str(relative))
    for asset in cfg.get('assets',[]):
        origin=ROOT/asset['source'];relative=Path(asset['destination']);assert not relative.is_absolute() and '..' not in relative.parts
        data=origin.read_bytes();assert hashlib.sha256(data).hexdigest()==asset['sha256']
        assert data==subprocess.check_output(['git','-C',str(ROOT),'show',source+':'+asset['source']]),'Asset must be retained in the pinned Git commit'
        target=stage/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        manifest.append(asset['sha256']+'  '+str(relative))
    (stage/'SOURCE.sha256').write_text('\n'.join(manifest)+'\n')
    remote=REMOTE+'/'+attempt;print('DISK\n'+ssh(['df','-h',REMOTE]),flush=True);ssh(['mkdir',remote])
    subprocess.run(['scp','-qr',str(stage),'tufts-login:'+remote+'/code'],check=True)
    workflow=cfg['workflow'];assert workflow in ['conditioned_operator.py','final_campaign.py']
    script=f'''#!/bin/bash
#SBATCH --job-name=ctol_paper_{attempt}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time={cfg['wall_time']}
#SBATCH --output={remote}/slurm.%j.out
#SBATCH --error={remote}/slurm.%j.err
set -uo pipefail
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export SOURCE_COMMIT={source}
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
cd {remote}/code
sha256sum -c SOURCE.sha256 || exit 43
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}',flush=True); sys.exit(0 if b=='gpu' else 42)" || exit $?
mkdir -p ../out ../training
finalize() {{
    exit_code=$?
    printf '%s\\n' "$exit_code" > ../EXIT_CODE
    date -u +%FT%TZ > ../FINISHED_UTC
    (cd .. && find code out training -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256)
}}
trap finalize EXIT
date -u +%FT%TZ > ../STARTED_UTC
$PY -u {workflow} --config {shlex.quote(a.config)} --out ../out > ../workflow.log 2>&1
exit $?
'''
    (run/'run.sbatch').write_text(script);subprocess.run(['scp','-q',str(run/'run.sbatch'),'tufts-login:'+remote+'/run.sbatch'],check=True)
    job=ssh(['sbatch','--parsable',remote+'/run.sbatch']).split(';')[0]
    after=ssh(['squeue','-u','tawal01','-h','-o','%i|%j|%T'])
    record=dict(job_id=job,source_commit=source,attempt=attempt,remote=remote,config=cfg,
        config_sha256=hashlib.sha256((EXP/a.config).read_bytes()).hexdigest(),queue_before=before,queue_after=after)
    (run/'submission.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
