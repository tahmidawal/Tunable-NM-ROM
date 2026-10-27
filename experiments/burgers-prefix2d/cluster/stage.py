"""Export only pinned committed sources/inputs into one unique allocated job directory."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
LANE='experiments/burgers-prefix2d'
NS='/cluster/tufts/paralab/tawal01/bprefix2d-20261027'


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('attempt'); ap.add_argument('--minutes',type=int,default=20)
    a=ap.parse_args(); assert a.attempt.isalnum() and 1<=a.minutes<=60
    ledger=json.loads((ROOT/LANE/'GPU-LEDGER.json').read_text())
    used=sum(x.get('charged_seconds',x.get('reserved_seconds',0)) for x in ledger['entries'])
    assert used+a.minutes*60<=ledger['cap_seconds'],(used,a.minutes,ledger)
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    files=[f'{LANE}/{name}' for name in ('pilot.py','prefixcore.py','test_prefix.py','DESIGN.md')]
    files += ['experiments/jcp-qualify/p1core.py','experiments/jcp-qualify/p1common.py',
              'experiments/jcp-qualify/vendor/quad2d/qcore.py','experiments/jcp-qualify/vendor/quad2d/qstudy.py',
              'experiments/mr-burgers2d/engines.py','experiments/mr-burgers2d/iterative_paths.py',
              'experiments/separable-decoder/sep_common.py',
              'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl',
              'experiments/jcp-qualify/vendor/quad2d/inputs/rotation_R512.npz']
    files += [f'experiments/jcp-qualify/vendor/quad2d/vendor/{name}.py' for name in ('arms','hops','hfast','bkfast','hari_quadrature')]
    dest=ROOT/LANE/'runs'/a.attempt; dest.mkdir(parents=True,exist_ok=False)
    remote=f'{NS}/{a.attempt}'; proof=[]
    for name in files:
        blob=subprocess.check_output(['git','-C',str(ROOT),'show',f'{commit}:{name}'])
        assert blob==(ROOT/name).read_bytes(),f'dirty source {name}'
        path=dest/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(blob)
        proof.append({'path':name,'sha256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob),'commit':commit})
    (dest/'PROVENANCE.json').write_text(json.dumps(proof,indent=2)+'\n')
    (dest/'COMMIT.txt').write_text(commit+'\n')
    script=f'''#!/bin/bash
#SBATCH --job-name=bprefix2d_{a.attempt}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:{a.minutes:02d}:00
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
df -h /cluster/tufts/paralab
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}',flush=True); sys.exit(0 if b=='gpu' else 42)"
trap 'rc=$?; cd "$TASK_ROOT"; find output -type f -print0 | sort -z | xargs -0 -r sha256sum > OUTPUTS.sha256; exit "$rc"' EXIT
"$PY" {LANE}/pilot.py --out "$TASK_ROOT/output/dev6"
echo ALL-DONE
'''
    (dest/'run.sbatch').write_text(script)
    (dest/'logs').mkdir()
    (dest/'RESERVATION.json').write_text(json.dumps({'attempt':a.attempt,'reserved_seconds':a.minutes*60,'commit':commit,'remote':remote},indent=2)+'\n')
    paths=sorted(p for p in dest.rglob('*') if p.is_file())
    (dest/'MANIFEST.sha256').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(dest)}\n' for p in paths))
    ledger['entries'].append({'attempt':a.attempt,'reserved_seconds':a.minutes*60,'status':'staged_not_submitted','commit':commit,'remote':remote})
    (ROOT/LANE/'GPU-LEDGER.json').write_text(json.dumps(ledger,indent=2)+'\n')
    print(dest); print(remote)


if __name__=='__main__':main()
