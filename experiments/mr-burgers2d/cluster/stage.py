"""Create a new immutable attempt from committed files, with content provenance."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[3]
attempt=sys.argv[1]
assert attempt.isalnum(), 'attempt must be alphanumeric'
dst=root/'experiments/mr-burgers2d/cluster/stage'/attempt
assert not dst.exists(), 'each attempt directory is new'
for name in ['code','in','out','logs']:(dst/name).mkdir(parents=True,exist_ok=True)
commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
files={
'experiments/mr-burgers2d/engines.py':'code/engines.py',
'experiments/mr-burgers2d/pilot.py':'code/pilot.py',
'experiments/mr-burgers2d/cold_fit.py':'code/cold_fit.py',
'experiments/mr-burgers2d/rollout.py':'code/rollout.py',
'experiments/mr-burgers2d/tests/check_kernels.py':'code/check_kernels.py',
'experiments/separable-decoder/sep_common.py':'code/sep_common.py',
'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl':'in/checkpoint.pkl',
}
provenance=[]
for source,target in files.items():
    saved=subprocess.check_output(['git','-C',str(root),'show',f'{commit}:{source}'])
    data=(root/source).read_bytes();assert saved==data, f'uncommitted source: {source}'
    (dst/target).write_bytes(data)
    provenance.append(dict(source=source,staged=target,sha256=hashlib.sha256(data).hexdigest(),commit=commit))
(dst/'COMMIT.txt').write_text(commit+'\n');(dst/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
remote=f'/cluster/tufts/paralab/tawal01/mr_burgers2d_20260907/{attempt}'
script='''#!/bin/bash
#SBATCH --job-name=ctol_mr_burgers_ATTEMPT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=REMOTE/logs/%j.out
#SBATCH --error=REMOTE/logs/%j.err
set -euo pipefail
TASK_ROOT=REMOTE
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
export TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) commit=$COMMIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
"$PY" code/check_kernels.py
"$PY" code/pilot.py --checkpoint in/checkpoint.pkl --out out --cases 4 --reps 3 PILOT_ARGS
cd "$TASK_ROOT"
find out logs -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
printf 'ALL-DONE\\n'
'''.replace('ATTEMPT',attempt).replace('REMOTE',remote).replace('PILOT_ARGS', '--meshes 256,512,1024 --reference-mesh 4096 --reference-dt .0003125 --order-audit --ic-starts 1,4' if attempt=='pilot02' else '')
if attempt.startswith('rollout') or attempt.startswith('steps'):
    script=script.replace('code/pilot.py','code/rollout.py')
if attempt.startswith('steps'):
    script=script.replace('--cases 4 --reps 3 ', '--cases 4 --reps 3 --study timestep --meshes 512,1024 --observation-intervals 256 ')
if attempt.startswith('cold'):
    script=script.replace('#SBATCH --time=02:00:00','#SBATCH --time=00:30:00')
    script=script.replace('"$PY" code/pilot.py --checkpoint in/checkpoint.pkl --out out --cases 4 --reps 3 ', '"$PY" code/cold_fit.py --checkpoint in/checkpoint.pkl --out out --cases 4 ')
(dst/'run.sbatch').write_text(script)
manifest=[]
for file in sorted(dst.rglob('*')):
    if file.is_file():manifest.append(f'{hashlib.sha256(file.read_bytes()).hexdigest()}  {file.relative_to(dst)}')
(dst/'MANIFEST.sha256').write_text('\n'.join(manifest)+'\n')
print(dst)
