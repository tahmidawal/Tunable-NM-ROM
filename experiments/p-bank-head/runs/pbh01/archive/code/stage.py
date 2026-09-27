"""Stage one p-bank-head attempt flat into an isolated paralab-bound directory.

The Poisson cell expects every module beside it (`core.py` looks for a sibling
`sep_common.py`), so the staged tree is flat, exactly as the head-ablation
Poisson attempt stages. Every staged byte is checked against the Git object at
HEAD, so a staged tree can never contain an uncommitted edit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/p_bank_head_20260916'
CELL = 'experiments/multiresolution-poisson'
RUN = f'{CELL}/runs/correction_accuracy10'

COMMON = [f'{CELL}/core.py', f'{CELL}/speed_core.py', f'{CELL}/kernel_solver.py',
          f'{CELL}/pilot.py', f'{CELL}/correction_core.py',
          'experiments/separable-decoder/sep_common.py',
          'experiments/cost-to-tolerance/ctol_tol.py',
          'experiments/wave2d-rom-latent-stepping/deps/multistage-precision/ms_parametric.py',
          'experiments/mr-burgers2d/engines.py',
          'experiments/head-ablation/arms.py',
          'experiments/head-ablation/poisson_ablation.py',
          'experiments/p-bank-head/pbh_core.py',
          'experiments/p-bank-head/cluster/stage.py']

TRAIN = COMMON + ['experiments/p-bank-head/pbh_fit.py',
                  'experiments/p-bank-head/pbh_train.py',
                  'experiments/p-bank-head/config-train.json',
                  f'{RUN}/checkpoints/r128_joint.pkl', f'{RUN}/basis.npz']

SOLVE = COMMON + ['experiments/p-bank-head/pbh_solve.py',
                  'experiments/p-bank-head/config-solve.json',
                  'experiments/p-bank-head/pabl01-reference.json',
                  f'{RUN}/checkpoints/r128_joint.pkl', f'{RUN}/basis.npz']

BODY = {
    'train': '''"$PY" pbh_train.py \\
  --config config-train.json \\
  --incumbent r128_joint.pkl --incumbent-basis basis.npz \\
  --out ../output''',
    'solve': '''"$PY" pbh_solve.py \\
  --config config-solve.json --models models.json \\
  --reference pabl01-reference.json \\
  --out ../output''',
}

SCRIPT = '''#!/bin/bash
#SBATCH --job-name=pbh_ATTEMPT
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
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
    p.add_argument('--hours', type=int, default=6)
    p.add_argument('--extra', nargs='*', default=[],
                   help='additional files staged verbatim (untracked run products '
                        'such as the trained checkpoints and the generated models.json)')
    a = p.parse_args()
    assert a.attempt.isalnum(), a.attempt
    out = ROOT / 'experiments/p-bank-head/runs' / a.attempt
    out.mkdir(parents=True, exist_ok=False)
    (out / 'code').mkdir()
    (out / 'logs').mkdir()
    remote = f'{NAMESPACE}/{a.attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                                     text=True).strip()
    proof = []
    for name in (TRAIN if a.mode == 'train' else SOLVE):
        content = (ROOT / name).read_bytes()
        assert content == subprocess.check_output(
            ['git', '-C', str(ROOT), 'show', f'{commit}:{name}']), f'uncommitted: {name}'
        dest = out / 'code' / Path(name).name
        assert not dest.exists(), f'flat name collision: {name}'
        dest.write_bytes(content)
        proof.append(dict(source=name, flat=Path(name).name, bytes=len(content),
                          sha256=hashlib.sha256(content).hexdigest(), commit=commit,
                          tracked=True))
    for name in a.extra:
        src = Path(name)
        content = src.read_bytes()
        dest = out / 'code' / src.name
        assert not dest.exists(), f'flat name collision: {name}'
        dest.write_bytes(content)
        proof.append(dict(source=str(src), flat=src.name, bytes=len(content),
                          sha256=hashlib.sha256(content).hexdigest(), commit=commit,
                          tracked=False))
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=2) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    (out / 'run.sbatch').write_text(
        SCRIPT.replace('ATTEMPT', a.attempt).replace('REMOTE', remote)
        .replace('HOURS', f'{a.hours:02d}').replace('BODY', BODY[a.mode]))
    manifest = [f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}'
                for p in sorted(out.rglob('*')) if p.is_file()]
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote)


if __name__ == '__main__':
    main()
