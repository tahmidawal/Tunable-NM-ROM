"""Stage exactly committed files. Never submits; reviewer inspects the frozen job first."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]
LANE='experiments/b3d-solver'
NS='/cluster/tufts/paralab/tawal01/b3d-solver-20261026'

def main():
    p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--config',required=True)
    p.add_argument('--minutes',type=int,default=55);a=p.parse_args()
    assert re.fullmatch('[a-z0-9-]+',a.attempt)
    assert 1<=a.minutes<=60
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',commit,LANE],cwd=ROOT,text=True).splitlines()
    names=[n for n in names if '/runs/' not in n and '/audits/' not in n and not n.endswith('.pyc')]
    assert LANE+'/'+a.config in names
    stage=ROOT/LANE/'runs'/a.attempt;stage.mkdir(parents=True,exist_ok=False)
    source=[]
    for n in names:
        content=subprocess.check_output(['git','show',commit+':'+n],cwd=ROOT)
        assert (ROOT/n).read_bytes()==content,n
        dst=stage/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(content)
        source.append(dict(path=n,sha256=hashlib.sha256(content).hexdigest(),bytes=len(content)))
    (stage/'COMMIT.txt').write_text(commit+'\n')
    (stage/'SOURCE.json').write_text(json.dumps(source,indent=1)+'\n')
    remote=NS+'/'+a.attempt
    script=f'''#!/bin/bash
#SBATCH --job-name=b3ds_{a.attempt}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=00:{a.minutes:02d}:00
#SBATCH --output={remote}/logs/%j.out
#SBATCH --error={remote}/logs/%j.err
set -euo pipefail
TASK_ROOT={remote}
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
export JCPREF_STORE=/cluster/tufts/paralab/tawal01/jcp-ref/store
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd {LANE}
"$PY" -u run.py --config {a.config} --out "$TASK_ROOT/output"
cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''
    (stage/'run.sbatch').write_text(script)
    proof=[]
    for f in sorted(stage.rglob('*')):
        if f.is_file():proof.append(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(stage)}')
    (stage/'MANIFEST.sha256').write_text('\n'.join(proof)+'\n')
    (stage/'logs').mkdir()
    print(json.dumps(dict(stage=str(stage),remote=remote,commit=commit,
                         config=a.config,config_sha256=hashlib.sha256((ROOT/LANE/a.config).read_bytes()).hexdigest(),
                         manifest_sha256=hashlib.sha256((stage/'MANIFEST.sha256').read_bytes()).hexdigest(),minutes=a.minutes)))

if __name__=='__main__':main()
