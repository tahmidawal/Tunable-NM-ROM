"""Stage committed source bytes into an isolated paralab-bound training attempt."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/b_head_train_20260916'
FILES = [
    'experiments/b-head-train/common.py',
    'experiments/b-head-train/train.py',
    'experiments/b-head-train/config-train.json',
    'experiments/b-head-train/cluster/stage_train.py',
    'experiments/head-ablation/arms.py',
    'experiments/mr-burgers2d/engines.py',
    'experiments/separable-decoder/sep_common.py',
    'experiments/separable-decoder/sep_hfit.py',
    'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl',
]
CHECKPOINT = 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
SCRIPT = '''#!/bin/bash
#SBATCH --job-name=ctol_bht_ATTEMPT
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
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
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
export PYTHONPATH="$TASK_ROOT/experiments/mr-burgers2d:$TASK_ROOT/experiments/head-ablation:$TASK_ROOT/experiments/separable-decoder:$TASK_ROOT/experiments/b-head-train"
"$PY" experiments/b-head-train/train.py \\
  --config experiments/b-head-train/config-train.json \\
  --checkpoint CKPT \\
  --out output
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''


def stage(attempt, files, script, extra=()):
    assert attempt.isalnum(), attempt
    out = ROOT / 'experiments/b-head-train/runs' / attempt
    out.mkdir(parents=True, exist_ok=False)
    remote = f'{NAMESPACE}/{attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    proof = []
    for name in files:
        content = (ROOT / name).read_bytes()
        assert content == subprocess.check_output(
            ['git', '-C', str(ROOT), 'show', f'{commit}:{name}']), f'uncommitted: {name}'
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        proof.append(dict(source=name, bytes=len(content),
                          sha256=hashlib.sha256(content).hexdigest(), commit=commit))
    for src, rel in extra:
        content = Path(src).read_bytes()
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        proof.append(dict(source=str(src), bytes=len(content),
                          sha256=hashlib.sha256(content).hexdigest(), commit='produced-artifact'))
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=2) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    (out / 'logs').mkdir()
    (out / 'run.sbatch').write_text(
        script.replace('ATTEMPT', attempt).replace('REMOTE', remote).replace('CKPT', CHECKPOINT))
    manifest = [f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}'
                for p in sorted(out.rglob('*')) if p.is_file()]
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote)
    return out, remote


if __name__ == '__main__':
    stage(sys.argv[1], FILES, SCRIPT)
