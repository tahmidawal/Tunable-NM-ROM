"""Stage one lshape attempt flat into an isolated paralab-bound directory.

    python experiments/lshape/cluster/stage.py train lsh01
    python experiments/lshape/cluster/stage.py solve lsh02 --intervals 64 128 --extra <staged models...>

Each attempt gets its OWN submit directory and its OWN remote directory: one job per
directory, never two. Every tracked file is checked byte-for-byte against the committed blob
at HEAD before it is copied, so what runs on the cluster is exactly what is in Git; run
products staged with --extra (trained checkpoints, models.json) are hashed into
PROVENANCE.json as untracked.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/lshape_20260917'

COMMON = ['experiments/separable-decoder/sep_common.py',
          'experiments/mr-burgers2d/engines.py',
          'experiments/head-ablation/arms.py',
          'experiments/lshape/lsh_core.py',
          'experiments/lshape/cluster/stage.py']

TRAIN = COMMON + ['experiments/lshape/lsh_fit.py', 'experiments/lshape/lsh_train.py',
                  'experiments/lshape/config-train.json']

SOLVE = COMMON + ['experiments/lshape/lsh_solve.py', 'experiments/lshape/config-solve.json']

BODY = {
    'train': '"$PY" lsh_train.py --config config-train.json --out ../output',
    'solve': '"$PY" lsh_solve.py --config config-solve.json --models models.json --out ../output INTERVALS',
}

SCRIPT = '''#!/bin/bash
#SBATCH --job-name=lsh_ATTEMPT
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:GPU:1
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=HOURS:00:00
#SBATCH --output=REMOTE/logs/%j.out
#SBATCH --error=REMOTE/logs/%j.err
set -euo pipefail
TASK_ROOT=REMOTE
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
export TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd "$TASK_ROOT/code"
BODY
cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['train', 'solve'])
    p.add_argument('attempt')
    p.add_argument('--hours', type=int, default=10)
    p.add_argument('--gpu', default='a100')
    p.add_argument('--intervals', nargs='*', default=[])
    p.add_argument('--extra', nargs='*', default=[])
    a = p.parse_args()
    assert a.attempt.isalnum(), a.attempt
    out = ROOT / 'experiments/lshape/runs' / a.attempt
    out.mkdir(parents=True, exist_ok=False)
    (out / 'code').mkdir()
    (out / 'logs').mkdir()
    remote = f'{NAMESPACE}/{a.attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    proof = []
    for name in {'train': TRAIN, 'solve': SOLVE}[a.mode]:
        content = (ROOT / name).read_bytes()
        assert content == subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{name}']), \
            f'uncommitted: {name}'
        dest = out / 'code' / Path(name).name
        assert not dest.exists(), f'flat name collision: {name}'
        dest.write_bytes(content)
        proof.append(dict(source=name, flat=Path(name).name, bytes=len(content),
                          sha256=hashlib.sha256(content).hexdigest(), commit=commit, tracked=True))
    for name in a.extra:
        src = Path(name)
        content = src.read_bytes()
        dest = out / 'code' / src.name
        assert not dest.exists(), f'flat name collision: {name}'
        dest.write_bytes(content)
        proof.append(dict(source=str(src), flat=src.name, bytes=len(content),
                          sha256=hashlib.sha256(content).hexdigest(), commit=commit, tracked=False))
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=2) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    body = BODY[a.mode].replace('INTERVALS', ('--intervals ' + ' '.join(a.intervals)) if a.intervals else '')
    (out / 'run.sbatch').write_text(
        SCRIPT.replace('ATTEMPT', a.attempt).replace('REMOTE', remote).replace('GPU', a.gpu)
        .replace('HOURS', f'{a.hours:02d}').replace('BODY', body))
    manifest = [f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}'
                for p in sorted(out.rglob('*')) if p.is_file()]
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote)


if __name__ == '__main__':
    main()
