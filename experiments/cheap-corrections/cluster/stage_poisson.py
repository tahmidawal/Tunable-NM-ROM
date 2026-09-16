"""Stage the Poisson correction ladder flat into an isolated paralab-bound attempt.

The Poisson cell expects every module beside it (core.py imports a sibling
sep_common.py), so the staged tree is flat, exactly as the head-ablation Poisson
attempt did.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/cheap_corr_20260915'
CELL = 'experiments/multiresolution-poisson'
RUN = f'{CELL}/runs/correction_accuracy10'
FLAT = [f'{CELL}/core.py', f'{CELL}/speed_core.py', f'{CELL}/kernel_solver.py', f'{CELL}/pilot.py',
        f'{CELL}/correction_core.py',
        'experiments/separable-decoder/sep_common.py',
        'experiments/cost-to-tolerance/ctol_tol.py',
        'experiments/wave2d-rom-latent-stepping/deps/multistage-precision/ms_parametric.py',
        'experiments/mr-burgers2d/engines.py',
        'experiments/head-ablation/arms.py',
        'experiments/cheap-corrections/directions.py',
        'experiments/cheap-corrections/poisson_ladder.py',
        'experiments/cheap-corrections/config-poisson.json',
        'experiments/cheap-corrections/cluster/stage_poisson.py',
        f'{RUN}/checkpoints/r128_joint.pkl',
        f'{RUN}/basis.npz']
HOURS = '02:30:00'


def main():
    attempt = sys.argv[1]
    assert attempt.isalnum(), attempt
    out = ROOT / 'experiments/cheap-corrections/runs' / attempt
    out.mkdir(parents=True, exist_ok=False)
    (out / 'code').mkdir()
    remote = f'{NAMESPACE}/{attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    proof = []
    for name in FLAT:
        content = (ROOT / name).read_bytes()
        assert content == subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{name}']), name
        dest = out / 'code' / Path(name).name
        assert not dest.exists(), f'flat name collision: {name}'
        dest.write_bytes(content)
        proof.append(dict(source=name, flat=Path(name).name, bytes=len(content),
                          sha256=hashlib.sha256(content).hexdigest(), commit=commit))
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=2) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    (out / 'logs').mkdir()
    script = '''#!/bin/bash
#SBATCH --job-name=ctol_cc_ATTEMPT
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=HOURS
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
"$PY" poisson_ladder.py \
  --config config-poisson.json \
  --checkpoint r128_joint.pkl --basis basis.npz \
  --out ../output
cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''.replace('ATTEMPT', attempt).replace('REMOTE', remote).replace('HOURS', HOURS)
    (out / 'run.sbatch').write_text(script)
    manifest = [f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}'
                for p in sorted(out.rglob('*')) if p.is_file()]
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote)


if __name__ == '__main__':
    main()
