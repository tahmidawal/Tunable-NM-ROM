"""Stage only committed Poisson3D source into one isolated, hash-pinned attempt."""
import argparse
import hashlib
import json
import subprocess
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
LANE='experiments/paper-p3d'
NAMESPACE='/cluster/tufts/paralab/tawal01/paper_p3d_20260920'
FILES=['DESIGN.md','config.json','IMPORTS.json','common.py','train.py','shared_rom.py','poisson.py','pod_transfer.py','run.py','audit.py','audit_pretraining.py','state_audit.py','trunk_diagnostic.py','freeze.py','cluster/stage.py','cluster/collect.py']
FILES+=['operators/'+name for name in ['models3d.py','training.py','pretrained_deeponet.py','__init__.py','README.md','IMPORTS.json','LOCAL_EXTENSIONS.json','poisson_adapter.py']]
FILES+=['operators/'+name for name in ['extra_models3d.py','EXTRA_IMPORTS.json','upstream/Physics_Attention.py','upstream/LICENSE','upstream/prior_families.py','upstream/PROVENANCE.json']]


def main():
    p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--reuse-attempt')
    p.add_argument('--reuse-operator',action='append',default=[],metavar='NAME=ATTEMPT')
    a=p.parse_args()
    assert a.attempt.isalnum(),a.attempt
    out=ROOT/LANE/'runs'/a.attempt;out.mkdir(parents=True,exist_ok=False)
    remote=f'{NAMESPACE}/{a.attempt}'
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    proof=[]
    cfg=json.loads((ROOT/LANE/'config.json').read_text())
    overrides=dict(value.split('=',1) for value in a.reuse_operator)
    assert len(overrides)==len(a.reuse_operator)
    assert all(name in [entry['name'] for entry in cfg['operators'] if entry.get('reuse')] and attempt.isalnum()
               for name,attempt in overrides.items())
    assert not overrides or a.reuse_attempt
    names=FILES+(['final-freeze.json'] if cfg.get('evaluation_cohort')=='final' else [])
    for name in names:
        path=f'{LANE}/{name}'
        blob=subprocess.check_output(['git','-C',str(ROOT),'show',f'{commit}:{path}'])
        assert blob==(ROOT/path).read_bytes(),f'uncommitted source: {path}'
        dest=out/'code'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(blob)
        proof.append(dict(path=path,staged=str(dest.relative_to(out)),sha256=hashlib.sha256(blob).hexdigest()))
    (out/'COMMIT.txt').write_text(commit+'\n')
    if a.reuse_attempt:
        assert a.reuse_attempt.isalnum()
        previous=ROOT/LANE/'runs'/a.reuse_attempt
        collected=json.loads((previous/'COLLECTED.json').read_text())
        assert collected['checksums_verified']
        assert json.loads((previous/'audit-local.json').read_text())['passed']
        source=previous/'archive/out';record=json.loads((source/'result.json').read_text())
        cfg=json.loads((out/'code/config.json').read_text())
        assert cfg['reuse_checkpoint_directory']=='checkpoints'
        names=['result.json','cohorts.json','bank.pkl']+[f'head_K{k}.pkl' for k in cfg['latent_dimensions']]
        names += [p.name for p in source.glob('eq_N*_K*.npz')]
        for name in names:
            dest=out/'checkpoints'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source/name,dest)
            proof.append(dict(path=f'reused:{a.reuse_attempt}/out/{name}',staged=str(dest.relative_to(out)),
                              sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),source_commit=record['source_commit']))
        origins={}
        def origin_for(name,checkpoint):
            wanted=hashlib.sha256(checkpoint.read_bytes()).hexdigest();matches=[]
            for prior_file in (ROOT/LANE/'runs').glob('*/archive/out/result.json'):
                prior=json.loads(prior_file.read_text());candidate=prior_file.parent/name
                attempt_root=prior_file.parents[2]
                if not (attempt_root/'COLLECTED.json').exists() or not (attempt_root/'audit-local.json').exists():continue
                if not json.loads((attempt_root/'COLLECTED.json').read_text())['checksums_verified']:continue
                if not json.loads((attempt_root/'audit-local.json').read_text())['passed']:continue
                if not prior.get('complete') or prior.get('final_cohort_opened') or not candidate.exists():continue
                if name.startswith('operators/'):
                    op=name.split('/')[1]
                    entries=[entry for entry in prior['config'].get('operators',[]) if entry['name']==op]
                    freshly_trained=bool(entries) and not entries[0].get('reuse',False)
                else:freshly_trained=not prior['config'].get('reuse_checkpoint_directory')
                if freshly_trained and hashlib.sha256(candidate.read_bytes()).hexdigest()==wanted:
                    matches.append(dict(source_commit=prior['source_commit'],job_id=prior['job_id'],
                        result_sha256=hashlib.sha256(prior_file.read_bytes()).hexdigest(),checkpoint_sha256=wanted,
                        attempt=prior_file.parents[2].name))
            assert len(matches)<=1,('ambiguous original training source',name,matches)
            return matches[0] if matches else dict(source_commit=None,job_id=None,checkpoint_sha256=wanted,
                limitation='Original training job not identified; immediate reused source remains hash-pinned.')
        for name in ['bank.pkl']+[f'head_K{k}.pkl' for k in cfg['latent_dimensions']]:
            origins[name]=origin_for(name,out/'checkpoints'/name)
        for entry in cfg['operators']:
            if not entry.get('reuse'):continue
            attempt=overrides.get(entry['name'],a.reuse_attempt)
            previous=ROOT/LANE/'runs'/attempt
            assert json.loads((previous/'COLLECTED.json').read_text())['checksums_verified']
            assert json.loads((previous/'audit-local.json').read_text())['passed']
            src=previous/'archive/out';prior=json.loads((src/'result.json').read_text())
            assert prior['complete'] and not prior['final_cohort_opened']
            name=f"operators/{entry['name']}/best.pkl"
            for source_name,dest_name in [(name,name),('result.json',f"operators/{entry['name']}/SOURCE_RESULT.json")]:
                dest=out/'checkpoints'/dest_name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src/source_name,dest)
                proof.append(dict(path=f'reused:{attempt}/out/{source_name}',staged=str(dest.relative_to(out)),
                    sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),source_commit=prior['source_commit']))
            origins[name]=origin_for(name,src/name)
        origin_path=out/'checkpoints'/'ORIGINAL_TRAINING_SOURCES.json'
        origin_path.write_text(json.dumps(origins,indent=2)+'\n')
        proof.append(dict(path='generated:original-training-checkpoint-content-matches',staged=str(origin_path.relative_to(out)),
            sha256=hashlib.sha256(origin_path.read_bytes()).hexdigest()))
    (out/'PROVENANCE.json').write_text(json.dumps(dict(source_commit=commit,files=proof,remote=remote),indent=2)+'\n')
    script='''#!/bin/bash
#SBATCH --job-name=ctol_p3d_920___ATTEMPT__
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=__REMOTE__/job.out
#SBATCH --error=__REMOTE__/job.err
set -euo pipefail
TASK_ROOT=__REMOTE__
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export TMPDIR="$TASK_ROOT/tmp" XDG_CACHE_HOME="$TASK_ROOT/cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/out"
cd "$TASK_ROOT"
sha256sum -c SOURCE.sha256
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "source_commit=$SOURCE_COMMIT job_id=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
df -h /cluster/tufts/paralab
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
# Training data is generated by run.py from the committed parameter seeds.
set +e
"$PY" code/run.py --config code/config.json --out out
RUN_STATUS=$?
set -e
find out -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo "run_exit=$RUN_STATUS"
exit "$RUN_STATUS"
'''.replace('__REMOTE__',remote).replace('__ATTEMPT__',a.attempt)
    (out/'run.sbatch').write_text(script)
    source=[]
    for f in sorted(out.rglob('*')):
        if f.is_file():source.append(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(out)}')
    (out/'SOURCE.sha256').write_text('\n'.join(source)+'\n')
    print(json.dumps(dict(local=str(out),remote=remote,commit=commit,files=proof),indent=2))


if __name__=='__main__':main()
