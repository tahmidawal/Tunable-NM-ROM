"""Stage one p-linear attempt flat into an isolated paralab-bound directory.

    python cluster/stage.py <mode> <attempt> [--hours H]

Modes: `solve` (the ladder job; config-<intervals>.json chosen by --config) and `head`
(the head-capacity job). Each attempt gets its OWN submit directory and its OWN remote
directory: one job per directory, never two. Every staged byte is checked against the Git
object at HEAD before it is copied, so a staged tree can never contain an uncommitted
edit. The Poisson cell imports its modules as siblings, so the tree is flat.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/p_linear_20260917'
CELL = 'experiments/multiresolution-poisson'
RUN = f'{CELL}/runs/correction_accuracy10'
LANE = 'experiments/p-linear'

COMMON = [f'{CELL}/core.py', f'{CELL}/speed_core.py', f'{CELL}/kernel_solver.py',
          f'{CELL}/pilot.py', f'{CELL}/correction_core.py', f'{CELL}/iterative_core.py',
          'experiments/separable-decoder/sep_common.py',
          'experiments/cost-to-tolerance/ctol_tol.py',
          'experiments/wave2d-rom-latent-stepping/deps/multistage-precision/ms_parametric.py',
          'experiments/mr-burgers2d/engines.py',
          'experiments/head-ablation/arms.py',
          'experiments/head-ablation/poisson_ablation.py',
          'experiments/p-bank-head/pbh_core.py',
          'experiments/p-bank-head/pbh_fit.py',
          f'{LANE}/plin_core.py', f'{LANE}/directions.py', f'{LANE}/cluster/stage.py',
          f'{LANE}/checkpoints/primary_K32.pkl', f'{LANE}/checkpoints/primary_K32-basis.npz',
          f'{RUN}/checkpoints/r128_joint.pkl', f'{RUN}/basis.npz']

SOLVE = COMMON + [f'{LANE}/plin_solve.py',
                  f'{LANE}/references/pbh02-reference.json',
                  f'{LANE}/references/ccpoi01-reference.json']

HEAD = COMMON + [f'{LANE}/plin_head.py',
                 f'{LANE}/checkpoints/bankarm_head.pkl',
                 f'{LANE}/references/pbh02-reference.json']

BODY = {
    'solve': '''"$PY" plin_solve.py --config __CONFIG__ --out ../output''',
    'head': '''"$PY" plin_head.py --config __CONFIG__ --out ../output''',
}

SCRIPT = '''#!/bin/bash
#SBATCH --job-name=plin___ATTEMPT__
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:__GPU__:1
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=__MEM__
#SBATCH --time=__HOURS__:00:00
#SBATCH --output=__REMOTE__/logs/%j.out
#SBATCH --error=__REMOTE__/logs/%j.err
set -euo pipefail
TASK_ROOT=__REMOTE__
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
__BODY__
cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['solve', 'head'])
    p.add_argument('attempt')
    p.add_argument('--config', required=True, help='config file name inside experiments/p-linear')
    p.add_argument('--hours', type=int, default=8)
    p.add_argument('--gpu', default='a100', choices=['a100', 'h100', 'h200', 'l40s'])
    p.add_argument('--mem', default='180G', help='180G on A100/H100; 240G for H200-class runs')
    a = p.parse_args()
    assert a.attempt.isalnum(), a.attempt
    out = ROOT / LANE / 'runs' / a.attempt
    out.mkdir(parents=True, exist_ok=False)
    (out / 'code').mkdir()
    (out / 'logs').mkdir()
    remote = f'{NAMESPACE}/{a.attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    files = {'solve': SOLVE, 'head': HEAD}[a.mode] + [f'{LANE}/{a.config}']
    proof = []
    for name in files:
        content = (ROOT / name).read_bytes()
        assert content == subprocess.check_output(
            ['git', '-C', str(ROOT), 'show', f'{commit}:{name}']), f'uncommitted: {name}'
        dest = out / 'code' / Path(name).name
        assert not dest.exists(), f'flat name collision: {name}'
        dest.write_bytes(content)
        proof.append(dict(source=name, flat=Path(name).name, bytes=len(content),
                          sha256=hashlib.sha256(content).hexdigest(), commit=commit, tracked=True))
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=2) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    script = SCRIPT
    for token, value in (('__ATTEMPT__', a.attempt), ('__REMOTE__', remote), ('__GPU__', a.gpu),
                         ('__HOURS__', f'{a.hours:02d}'), ('__MEM__', a.mem),
                         ('__BODY__', BODY[a.mode])):
        script = script.replace(token, value)
    script = script.replace('__CONFIG__', a.config)
    assert '__' not in script.replace('__pycache__', ''), script
    (out / 'run.sbatch').write_text(script)
    manifest = [f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}'
                for p in sorted(out.rglob('*')) if p.is_file()]
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote)


if __name__ == '__main__':
    main()
