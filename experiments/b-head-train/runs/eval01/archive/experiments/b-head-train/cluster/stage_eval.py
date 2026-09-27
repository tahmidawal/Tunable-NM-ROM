"""Stage the evaluation attempt: committed source plus the frozen checkpoints a
collected training attempt produced. The incumbent is carried through as arm (a),
so it is evaluated in the same job, on the same GPU, in the same randomised
order as every trained checkpoint."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_train import ROOT, NAMESPACE, stage  # noqa: E402

FILES = [
    'experiments/b-head-train/evaluate.py',
    'experiments/b-head-train/config-eval.json',
    'experiments/b-head-train/cluster/stage_eval.py',
    'experiments/b-head-train/cluster/stage_train.py',
    'experiments/head-ablation/arms.py',
    'experiments/head-ablation/artifacts/abl01/result.json',
    'experiments/mr-burgers2d/engines.py',
    'experiments/mr-burgers2d/iterative_paths.py',
    'experiments/separable-decoder/sep_common.py',
    'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl',
]
INCUMBENT = 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
SCRIPT = '''#!/bin/bash
#SBATCH --job-name=ctol_bhe_ATTEMPT
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=08:00:00
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
"$PY" experiments/b-head-train/evaluate.py \\
  --config experiments/b-head-train/config-eval.json \\
  --checkpoints experiments/b-head-train/checkpoints \\
  --incumbent CKPT \\
  --out output
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''


def main():
    attempt, source = sys.argv[1], Path(sys.argv[2])
    ckpts = sorted(source.glob('ckpt_*.pkl'))
    assert ckpts, f'no ckpt_*.pkl under {source}'
    extra = [(p, f'experiments/b-head-train/checkpoints/{p.name}') for p in ckpts]
    extra.append((source / 'result.json', 'experiments/b-head-train/checkpoints/TRAIN.json'))
    out, remote = stage(attempt, FILES, SCRIPT.replace('CKPT', INCUMBENT), extra)
    print(f'{len(ckpts)} checkpoints staged')


if __name__ == '__main__':
    main()
