"""Stage exactly committed sources; no data/checkpoints copied from local disk."""
import hashlib,json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[3]
attempt=sys.argv[1];phase=sys.argv[2]
resume=Path(sys.argv[3]).resolve() if len(sys.argv)>3 else None
assert attempt.isalnum() and phase in ('smoke','train','validate','evaluate','all')
dst=root/'experiments/modcp-eq/cluster/stage'/attempt
assert not dst.exists()
for name in ('code','out','logs'):(dst/name).mkdir(parents=True,exist_ok=True)
commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
files=subprocess.check_output(['git','-C',str(root),'ls-files','experiments/modcp-eq'],text=True).splitlines()
provenance=[]
for source in files:
    if '/runs/' in source or '/stage/' in source or not source.endswith(('.py','.md','.sh')):continue
    data=subprocess.check_output(['git','-C',str(root),'show',f'{commit}:{source}'])
    assert data==(root/source).read_bytes(),f'uncommitted source {source}'
    target=dst/'code'/Path(source).relative_to('experiments/modcp-eq');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    provenance.append(dict(source=source,staged=str(target.relative_to(dst)),sha256=hashlib.sha256(data).hexdigest(),commit=commit))
(dst/'COMMIT.txt').write_text(commit+'\n');(dst/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
if resume is not None:
    assert resume.is_dir()
    # Only learned/fitted artifacts and frozen selection records move phases.
    # No local truth/data array is staged: every phase regenerates seeded fields.
    selected=[]
    for directory in ('checkpoints','rules'):
        if (resume/directory).exists():selected+=list((resume/directory).rglob('*'))
    for name in ('config.json','selections.json','selection_freeze.json'):
        if (resume/name).exists():selected.append(resume/name)
    copied=[]
    for source in selected:
        if not source.is_file():continue
        rel=source.relative_to(resume);target=dst/'out'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        data=source.read_bytes();target.write_bytes(data)
        copied.append(dict(source=str(source),staged=str(target.relative_to(dst)),sha256=hashlib.sha256(data).hexdigest()))
    (dst/'RESUME_ARTIFACTS.json').write_text(json.dumps(copied,indent=2)+'\n')
remote=f'/cluster/tufts/paralab/tawal01/modcp_burgers2d_20260910/{attempt}'
script='''#!/bin/bash
#SBATCH --job-name=ctol_modcp_b2d_ATTEMPT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=TIME
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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
"$PY" code/tests/check_burgers.py
"$PY" code/burgers/campaign.py --out out OPTIONS
find out -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
printf 'ALL-DONE\\n'
'''.replace('ATTEMPT',attempt).replace('REMOTE',remote).replace('TIME','01:00:00' if phase=='smoke' else '24:00:00').replace('OPTIONS','--smoke' if phase=='smoke' else '--phase '+phase)
(dst/'run.sbatch').write_text(script)
manifest=[]
for file in sorted(dst.rglob('*')):
    if file.is_file():manifest.append(f'{hashlib.sha256(file.read_bytes()).hexdigest()}  {file.relative_to(dst)}')
(dst/'MANIFEST.sha256').write_text('\n'.join(manifest)+'\n');print(dst)
