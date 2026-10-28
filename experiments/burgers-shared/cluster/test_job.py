"""Copy a reviewed lane package and add its bounded allocation script; never submit."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lane', choices=('step', 'replay'), required=True)
    ap.add_argument('--package', required=True)
    ap.add_argument('--attempt', default='screen01')
    args = ap.parse_args()
    assert args.attempt.isalnum()
    source = Path(args.package).resolve()
    for line in (source/'MANIFEST.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = (source/name).resolve()
        assert path.is_relative_to(source)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    lane = f'burgers-shared-{args.lane}'
    dest = ROOT/'experiments/burgers-shared/runs'/f'{args.lane}-{args.attempt}'
    assert not dest.exists()
    shutil.copytree(source, dest)
    (dest/'LANE-MANIFEST.sha256').write_bytes((dest/'MANIFEST.sha256').read_bytes())
    remote = f'/cluster/tufts/paralab/tawal01/bshared-{args.lane}-20261028/{args.attempt}'
    script = f'''#!/bin/bash
#SBATCH --job-name=bshared_{args.lane}_{args.attempt}
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
df -h /cluster/tufts/paralab
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}',flush=True); sys.exit(0 if b=='gpu' else 42)"
trap 'rc=$?; cd "$TASK_ROOT"; find output -type f -print0 | sort -z | xargs -0 -r sha256sum > OUTPUTS.sha256; exit "$rc"' EXIT
timeout --signal=TERM --kill-after=15s 840s "$PY" experiments/{lane}/pilot.py --model-dir model --out output/dev6
echo ALL-DONE
'''
    (dest/'run.sbatch').write_text(script)
    (dest/'logs').mkdir()
    (dest/'RESERVATION.json').write_text(json.dumps(dict(
        lane=args.lane, attempt=args.attempt, remote=remote, reserved_seconds=900,
        source_commit=(dest/'COMMIT.txt').read_text().strip(),
        model_manifest_sha256=hashlib.sha256((dest/'model/MODEL.json').read_bytes()).hexdigest(),
        timing_scope='each lane compares its methods only within this allocation'), indent=2)+'\n')
    paths = sorted(p for p in dest.rglob('*') if p.is_file() and p.name != 'MANIFEST.sha256')
    (dest/'MANIFEST.sha256').write_text(''.join(
        f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(dest)}\n' for p in paths))
    print(dest)
    print(remote)


if __name__ == '__main__':
    main()
