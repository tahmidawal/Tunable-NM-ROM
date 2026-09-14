"""Stage committed coverage comparison and explicit inherited-artifact lineage."""
import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2];exp=root/'experiments/mr-burgers2d';attempt='accuracy09';parent=exp/'runs/accuracy08';archive=parent/'archive'
assert json.loads((parent/'COLLECTION-CHECK.json').read_text())['archive_verified']
dst=exp/'cluster/stage'/attempt;dst.mkdir(parents=True,exist_ok=False)
for name in ['code','in','out','logs','in/reuse']:(dst/name).mkdir(parents=True,exist_ok=True)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
files={f'experiments/mr-burgers2d/{n}':f'code/{n}' for n in ['engines.py','iterative_paths.py','accuracy_paths.py','accuracy_coverage_paths.py','ACCURACY-COVERAGE-DESIGN.md']}
files.update({'experiments/mr-burgers2d/accuracy_coverage.py':'code/accuracy.py','experiments/mr-burgers2d/config-accuracy-coverage.json':'code/config-accuracy.json',
 'experiments/separable-decoder/sep_common.py':'code/sep_common.py','experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl':'in/checkpoint.pkl'})
prov=[]
for source,target in files.items():
 data=subprocess.check_output(['git','show',f'{commit}:{source}'],cwd=root);assert (root/source).read_bytes()==data
 (dst/target).write_bytes(data);prov.append(dict(source=source,staged=target,sha256=hashlib.sha256(data).hexdigest(),commit=commit))
old=json.loads((archive/'out/result.json').read_text());meta=json.loads((parent/'ARCHIVE.json').read_text());audit=json.loads((parent/'AUDIT.json').read_text())
assert audit['passed'] and audit['result_sha256']==hashlib.sha256((archive/'out/result.json').read_bytes()).hexdigest()
assert old['complete'] and meta['job_completed_successfully']
inherit=[]
for source in ['out/trained_checkpoint.pkl','out/training_targets.npz','out/training.json','out/result.json']+[f'out/{r["artifact"]}' for r in old['reference']]:
 destination='parent-result.json' if source=='out/result.json' else Path(source).name
 inherit.append(dict(source=source,destination=destination,sha256=hashlib.sha256((archive/source).read_bytes()).hexdigest()))
lineage=dict(parent_job_id=old['job_id'],parent_source_commit=old['commit'],parent_gpu=old['gpu'],parent_remote=meta['remote'],parent_archive_sha256=meta['joined_sha256'],
 parent_owner_audit_sha256=hashlib.sha256((parent/'AUDIT.json').read_bytes()).hexdigest(),parent_result_sha256=audit['result_sha256'],parent_terminal_state=meta['job_terminal_state'],parent_reference_seed=old['config']['seed'],parent_reference_fresh_seed=old['config']['fresh_seed'],files=inherit)
(dst/'in/reuse/INHERITED.json').write_text(json.dumps(lineage,indent=2)+'\n');(dst/'INHERITED.sha256').write_text('\n'.join(f'{r["sha256"]}  in/reuse/{r["destination"]}' for r in inherit)+'\n')
(dst/'PROVENANCE.json').write_text(json.dumps(prov,indent=2)+'\n');(dst/'COMMIT.txt').write_text(commit+'\n')
remote=f'/cluster/tufts/paralab/tawal01/mr_burgers2d_20260907/{attempt}'
script='''#!/bin/bash
#SBATCH --job-name=ctol_mr_burgers_accuracy09
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=REMOTE/logs/%j.out
#SBATCH --error=REMOTE/logs/%j.err
set -euo pipefail
TASK_ROOT=REMOTE
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
sha256sum -c INHERITED.sha256 --quiet
export COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) commit=$COMMIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
"$PY" code/accuracy.py --config code/config-accuracy.json --checkpoint in/checkpoint.pkl --reuse-dir in/reuse --out out
printf 'ALL-DONE\\n'
'''.replace('REMOTE',remote)
(dst/'run.sbatch').write_text(script)
(dst/'MANIFEST.sha256').write_text('\n'.join(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(dst)}' for f in sorted(dst.rglob('*')) if f.is_file())+'\n')
print(dst)
