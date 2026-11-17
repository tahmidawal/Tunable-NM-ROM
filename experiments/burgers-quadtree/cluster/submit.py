from pathlib import Path
import subprocess as s,sys,json,os,hashlib
H=Path(__file__).resolve().parents[1];root=H.parents[1]
job,driver,minutes=sys.argv[1:];minutes=int(minutes)
assert job.isalnum() and driver in ('reference','baseline','representation','reference_dense','representation_repair','dynamics','spectral_cost','autoencoder','output_control') and 0<minutes<=240
ns='/cluster/tufts/paralab/tawal01/burgers-quadtree-20261117';remote=ns+'/'+job
ledger=H/'cluster/ledger.json';old=json.loads(ledger.read_text()) if ledger.exists() else []
assert sum(x.get('elapsed_seconds',0)/60 if x.get('state') in ('COMPLETED','FAILED','TIMEOUT','CANCELLED') else x['reserved_minutes'] for x in old)+minutes<=480,'initial batch reserved hours exhausted; reconcile actuals before expansion'
def ssh(c):return s.check_output(['ssh','-o','BatchMode=yes','tufts-login',c],text=True)
q=ssh('squeue -u tawal01 -h -o "%i %j %T %Z"');print(q,flush=True)
assert len([x for x in q.splitlines() if 'bqt_' in x])<2
assert not any(x['name']==job for x in old)
commit=s.check_output(['git','rev-parse','HEAD'],cwd=H,text=True).strip()
ssh(f'test ! -e {remote} && mkdir -p {remote}/logs')
archive=s.check_output(['git','archive',commit,'experiments/burgers-quadtree'],cwd=root)
s.run(['ssh','tufts-login',f'tar -x -C {remote}'],input=archive,check=True)
dependency=''
if driver=='dynamics':
 dependency='#SBATCH --dependency=afterok:'+str(next(x['id'] for x in old if x['name']=='repair01'))
script=f'''#!/bin/bash
#SBATCH --job-name=bqt_{job}
{dependency}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time={minutes}
#SBATCH --output={remote}/logs/%j.out
#SBATCH --error={remote}/logs/%j.err
set -euo pipefail
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export SOURCE_COMMIT={commit}
export XDG_CACHE_HOME={remote}/cache TMPDIR={remote}/tmp
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
cd {remote}
sha256sum -c MANIFEST.sha256 --quiet
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd experiments/burgers-quadtree
$PY {driver}.py --out {remote}/output
cd {remote}
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
'''
(H/'cluster'/f'{job}.sbatch').write_text(script)
s.run(['ssh','tufts-login',f'cat > {remote}/run.sbatch'],input=script.encode(),check=True)
# Compare each staged byte to the committed source, never ancestor git metadata.
files=s.check_output(['git','ls-tree','-r','--name-only',commit,'experiments/burgers-quadtree'],cwd=root,text=True).splitlines()
man=''.join(hashlib.sha256(s.check_output(['git','show',f'{commit}:{p}'],cwd=root)).hexdigest()+'  '+p+'\n' for p in files)
s.run(['ssh','tufts-login',f'cat > {remote}/MANIFEST.sha256'],input=man.encode(),check=True)
ssh(f'cd {remote} && sha256sum -c MANIFEST.sha256 --quiet')
q=ssh('squeue -u tawal01 -h -o "%i %j %T %Z"');assert f'bqt_{job}' not in q
out=ssh(f'cd {remote} && sbatch --parsable run.sbatch').strip();assert out.isdigit(),out
old.append(dict(name=job,id=int(out),reserved_minutes=minutes,commit=commit,remote=remote,driver=driver));ledger.write_text(json.dumps(old,indent=2)+'\n')
print(out);print(ssh('squeue -u tawal01 -o "%.18i %.20j %.8T %.60Z"'))
