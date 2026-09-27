"""Stage committed source directly into the authorized Burgers namespace."""
import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2];attempt='accuracy07'
dst=root/'experiments/mr-burgers2d/cluster/stage'/attempt;dst.mkdir(parents=True,exist_ok=False)
for name in ['code','in','out','logs']:(dst/name).mkdir()
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
files={f'experiments/mr-burgers2d/{n}':f'code/{n}' for n in ['engines.py','iterative_paths.py','accuracy_paths.py','accuracy.py','config-accuracy.json','ACCURACY-DESIGN.md']}
files['experiments/separable-decoder/sep_common.py']='code/sep_common.py'
files['experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl']='in/checkpoint.pkl'
prov=[]
for source,target in files.items():
    data=subprocess.check_output(['git','show',f'{commit}:{source}'],cwd=root);assert (root/source).read_bytes()==data
    (dst/target).write_bytes(data);prov.append(dict(source=source,staged=target,sha256=hashlib.sha256(data).hexdigest(),commit=commit))
(dst/'PROVENANCE.json').write_text(json.dumps(prov,indent=2)+'\n');(dst/'COMMIT.txt').write_text(commit+'\n')
remote=f'/cluster/tufts/paralab/tawal01/mr_burgers2d_20260907/{attempt}'
script='''#!/bin/bash
#SBATCH --job-name=ctol_mr_burgers_accuracy07
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
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
export COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) commit=$COMMIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
"$PY" code/accuracy.py --config code/config-accuracy.json --checkpoint in/checkpoint.pkl --out out
printf 'ALL-DONE\\n'
'''.replace('REMOTE',remote)
(dst/'run.sbatch').write_text(script)
(dst/'MANIFEST.sha256').write_text('\n'.join(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(dst)}' for f in sorted(dst.rglob('*')) if f.is_file())+'\n')
print(dst)
