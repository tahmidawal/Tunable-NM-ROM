"""Owned submission/accounting only. No account-wide cancellation or deletion."""
from pathlib import Path
import argparse, datetime, hashlib, json, shlex, subprocess as sp, tarfile, io

H=Path(__file__).resolve().parents[1];ROOT=H.parents[1]
def ssh(cmd):return sp.check_output(['ssh','-o','BatchMode=yes','tufts-login',cmd],text=True)
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def reconcile(rows):
    if rows:
        ids=','.join(str(x['id']) for x in rows if x.get('id'))
        text=ssh(f'sacct -j {ids} -X --format=JobID,State,ElapsedRaw,AllocCPUS,AllocTRES -P')
        for line in text.splitlines()[1:]:
            z=line.split('|')
            for r in rows:
                if z[0]==str(r.get('id')):r.update(state=z[1],elapsed_seconds=int(z[2]),allocated_cpus=int(z[3]),allocated_tres=z[4])
        dump(H/'cluster/ledger.json',rows)
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['submit','reconcile','status'])
    ap.add_argument('--name');ap.add_argument('--driver');ap.add_argument('--minutes',type=int)
    ap.add_argument('--args',default='');ap.add_argument('--gpu',default='a100',choices=['a100','h100','h200','l40s'])
    ap.add_argument('--after',type=int)
    a=ap.parse_args();owner=json.loads((H/'OWNER.json').read_text());ns=owner['namespace']
    rows=reconcile(json.loads((H/'cluster/ledger.json').read_text()))
    q=ssh('squeue -u tawal01 -h -o "%i %j %T %Z"')
    print(q,flush=True)
    if a.action!='submit':
        print(json.dumps(rows,indent=2));print('allocated_gpu_hours',sum(x.get('elapsed_seconds',0) for x in rows)/3600);return
    assert a.name and a.name.replace('-','').isalnum() and a.driver and (H/a.driver).is_file()
    assert 0<a.minutes<=240 and not any(x['name']==a.name for x in rows)
    assert sum('bte_' in line and ns in line for line in q.splitlines())<2
    terminal={'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}
    charged=sum(x.get('elapsed_seconds',0) if (x.get('state') or 'PENDING').split()[0] in terminal else x['reserved_minutes']*60 for x in rows)
    cap=json.loads((H/'cluster/BATCHES.json').read_text())['cumulative_ceiling_gpu_seconds']
    assert charged+a.minutes*60<=cap,(charged,a.minutes*60,cap)
    dirty=sp.check_output(['git','-c','gc.auto=0','status','--porcelain','--',str(H)],cwd=ROOT,text=True)
    # Reconciliation intentionally updates the live ledger before this check.
    source_dirty=[line for line in dirty.splitlines() if not line.endswith('/cluster/ledger.json')]
    assert not source_dirty,'Commit and mirror source before submission: '+str(source_dirty)
    commit=sp.check_output(['git','-c','gc.auto=0','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    remote=ns+'/'+a.name
    ssh(f'test ! -e {shlex.quote(remote)} && mkdir -p {shlex.quote(remote)}/logs')
    archive=sp.check_output(['git','-c','gc.auto=0','archive',commit,'experiments/burgers-tree-embeddings'],cwd=ROOT)
    sp.run(['ssh','tufts-login',f'tar -x -C {shlex.quote(remote)}'],input=archive,check=True)
    manifest=[]
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for f in tar:
            if f.isfile():manifest.append(hashlib.sha256(tar.extractfile(f).read()).hexdigest()+'  '+f.name)
    sp.run(['ssh','tufts-login',f'cat > {remote}/SOURCE.sha256'],input=('\n'.join(manifest)+'\n').encode(),check=True)
    cmd=' '.join(shlex.quote(x) for x in [a.driver,'--out',remote+'/output']+shlex.split(a.args))
    constraint='#SBATCH --constraint=a100-80G\n' if a.gpu=='a100' else ''
    if a.after:assert any(r.get('id')==a.after for r in rows),'Dependency must be an owned job'
    dependency=f'#SBATCH --dependency=afterok:{a.after}\n' if a.after else ''
    script=f'''#!/bin/bash
#SBATCH --job-name=bte_{a.name}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:{a.gpu}:1
{dependency}{constraint}#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time={a.minutes}
#SBATCH --chdir={remote}
#SBATCH --output={remote}/logs/%j.out
#SBATCH --error={remote}/logs/%j.err
set -euo pipefail
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export SOURCE_COMMIT={commit} STUDY_NAMESPACE={ns}
export XDG_CACHE_HOME={remote}/cache TMPDIR={remote}/tmp
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
cd {remote}
sha256sum -c SOURCE.sha256 --quiet
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd experiments/burgers-tree-embeddings
$PY {cmd}
cd {remote}
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
'''
    scriptpath=H/'cluster'/f'{a.name}.sbatch';scriptpath.write_text(script)
    sp.run(['ssh','tufts-login',f'cat > {remote}/run.sbatch'],input=script.encode(),check=True)
    ssh(f'cd {remote} && sha256sum -c SOURCE.sha256 --quiet')
    q=ssh('squeue -u tawal01 -h -o "%i %j %T %Z"');assert f'bte_{a.name}' not in q
    job=ssh(f'cd {remote} && sbatch --parsable run.sbatch').strip();assert job.isdigit(),job
    rows.append(dict(name=a.name,id=int(job),reserved_minutes=a.minutes,commit=commit,remote=remote,
                     driver=a.driver,args=a.args,gpu=a.gpu,submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    dump(H/'cluster/ledger.json',rows)
    q=ssh('squeue -u tawal01 -h -o "%i %j %T %Z"');print(job,flush=True);print(q,flush=True)
    assert sum(f'bte_{a.name} ' in line for line in q.splitlines())<=1
if __name__=='__main__':main()
