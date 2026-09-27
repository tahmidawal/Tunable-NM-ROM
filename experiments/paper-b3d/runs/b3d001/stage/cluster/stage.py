"""Stage a pinned unique Burgers3D attempt directly into paralab and submit."""
import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'experiments/paper-b3d'
REMOTE='/cluster/tufts/paralab/tawal01/paper_b3d_20260920'


def command(argv,**kwargs):
    return subprocess.check_output(argv,text=True,**kwargs).strip()


def ssh(args):
    return command(['ssh','tufts-login',shlex.join(args)])


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='config-pilot.json');args=p.parse_args()
    cfg=json.loads((EXP/args.config).read_text());attempt=cfg['attempt']
    assert attempt.isalnum() and attempt.startswith('b3d')
    before=ssh(['squeue','-u','tawal01','-h','-o','%i|%j|%T'])
    print('QUEUE BEFORE\n'+before,flush=True)
    assert 'ctol_paper_b3d' not in before,'This lane already has a queued/running job'
    commit=command(['git','-C',str(ROOT),'rev-parse','HEAD'])
    assert not command(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=no','--','experiments/paper-b3d'])
    run=EXP/'runs'/attempt
    run.mkdir(parents=True,exist_ok=False)
    stage=run/'stage';stage.mkdir()
    files=command(['git','-C',str(ROOT),'ls-files','experiments/paper-b3d']).splitlines()
    manifest=[]
    for entry in files:
        relative=Path(entry).relative_to('experiments/paper-b3d')
        if relative.name=='.gitignore':continue
        source=ROOT/entry;target=stage/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        data=source.read_bytes()
        recorded=subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+entry])
        assert data==recorded,(entry,'working file differs from pinned source')
        manifest.append(hashlib.sha256(data).hexdigest()+'  '+str(relative))
    (stage/'SOURCE.sha256').write_text('\n'.join(manifest)+'\n')
    remote=REMOTE+'/'+attempt
    ssh(['mkdir','-p',REMOTE])
    ssh(['mkdir',remote])
    subprocess.run(['scp','-qr',str(stage),'tufts-login:'+remote+'/code'],check=True)
    script=f'''#!/bin/bash
#SBATCH --job-name=ctol_paper_{attempt}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output={remote}/slurm.%j.out
#SBATCH --error={remote}/slurm.%j.err
set -uo pipefail
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export SOURCE_COMMIT={commit}
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
cd {remote}/code
sha256sum -c SOURCE.sha256 || exit 43
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}',flush=True); sys.exit(0 if b=='gpu' else 42)" || exit $?
mkdir -p ../out
date -u +%FT%TZ > ../STARTED_UTC
$PY -u run.py --config {shlex.quote(args.config)} --out ../out > ../driver.log 2>&1
code=$?
if [ "$code" -eq 0 ]; then
    $PY -u audit.py ../out > ../audit.log 2>&1
    code=$?
fi
printf '%s\\n' "$code" > ../EXIT_CODE
date -u +%FT%TZ > ../FINISHED_UTC
cd ..
find code out -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
exit "$code"
'''
    (run/'run.sbatch').write_text(script)
    subprocess.run(['scp','-q',str(run/'run.sbatch'),'tufts-login:'+remote+'/run.sbatch'],check=True)
    job=ssh(['sbatch','--parsable',remote+'/run.sbatch']).split(';')[0]
    after=ssh(['squeue','-u','tawal01','-h','-o','%i|%j|%T'])
    result=dict(job_id=job,source_commit=commit,attempt=attempt,remote=remote,config=cfg,
                config_sha256=hashlib.sha256((EXP/args.config).read_bytes()).hexdigest(),queue_before=before,queue_after=after)
    (run/'submission.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
