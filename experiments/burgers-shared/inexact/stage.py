"""Stage original retained inputs plus inexact coarse screen code, with content manifests."""
from pathlib import Path
import json,hashlib,subprocess,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
original=ROOT.parent/'2026-10-28-burgers-shared-step/experiments/burgers-shared-step/artifacts/screen01'
dest=ROOT/'experiments/burgers-shared/runs/inexact01'
assert not dest.exists();dest.mkdir(parents=True)
proof=[]
for line in (original/'MANIFEST.sha256').read_text().splitlines():
    digest,name=line.split('  ',1)
    if not name.startswith(('model/','experiments/')):continue
    source=original/name
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    proof.append(dict(path=name,sha256=digest,source='retained step job6619857 input'))
for name in ['screen.py']:
    source=ROOT/'experiments/burgers-shared/inexact'/name
    target=dest/'experiments/burgers-shared/inexact'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
for line in (original/'OUTPUTS.sha256').read_text().splitlines():
    digest,name=line.split('  ',1)
    if name.endswith('result.json') or any(name.endswith(f'{arm}_case{c}.npz') for arm in ['direct','affine'] for c in range(6)):
        assert hashlib.sha256((original/name).read_bytes()).hexdigest()==digest
(dest/'expected').mkdir()
shutil.copyfile(original/'output/dev6/result.json',dest/'expected/result.json')
for arm in ['direct','affine']:
 for c in range(6):
    with np.load(original/f'output/dev6/{arm}_case{c}.npz') as z:
        np.savez_compressed(dest/f'expected/{arm}_case{c}.npz',states=z['states'],step_it=z['step_it'])
(dest/'SOURCE-PROVENANCE.json').write_text(json.dumps(proof,indent=2)+'\n')
commit=subprocess.check_output(['git','-c','gc.auto=0','rev-parse','HEAD'],text=True).strip()
(dest/'COMMIT.txt').write_text(commit+'\n')
remote='/cluster/tufts/paralab/tawal01/bshared-inexact-20261029/inexact01'
(dest/'run.sbatch').write_text(f'''#!/bin/bash
#SBATCH --job-name=bshared_inexact01
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:15:00
#SBATCH --output={remote}/logs/%j.out
#SBATCH --error={remote}/logs/%j.err
set -euo pipefail
TASK_ROOT={remote}
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp" PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}',flush=True); sys.exit(0 if b=='gpu' else 42)"
trap 'rc=$?; cd "$TASK_ROOT"; find output -type f -print0 | sort -z | xargs -0 -r sha256sum > OUTPUTS.sha256; exit "$rc"' EXIT
timeout --signal=TERM --kill-after=10s 870s "$PY" experiments/burgers-shared/inexact/screen.py
echo ALL-DONE
''')
(dest/'logs').mkdir()
paths=sorted(p for p in dest.rglob('*') if p.is_file())
(dest/'MANIFEST.sha256').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(dest)}\n' for p in paths))
print(dest)
