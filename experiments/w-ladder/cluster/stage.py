"""Stage one w-ladder attempt into an isolated paralab-bound directory.

    python experiments/w-ladder/cluster/stage.py <attempt> <config-file-name>

Each attempt gets its OWN submit directory and its OWN remote directory: one job per
directory, never two. Every staged file is checked byte-for-byte against the committed blob
at HEAD before it is copied, so what runs on the cluster is exactly what is in Git.
Pattern copied from experiments/q-ridge/cluster/stage.py (2026-09-16).
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/w_ladder_20260917'
INPUTS = 'experiments/multiresolution-wave/runs/accel12/cluster/in/dirichlet'
FILES = [
    'experiments/w-ladder/ladder.py',
    'experiments/w-ladder/retained-gates.json',
    'experiments/w-ladder/cluster/stage.py',
    'experiments/multiresolution-wave/pilot.py',
    'experiments/multiresolution-wave/iterative_replay.py',
    'experiments/multiresolution-wave/iterative_paths.py',
    'experiments/multiresolution-wave/acceleration.py',
    'experiments/multiresolution-wave/acceleration_replay.py',
    'experiments/multiresolution-wave/modal_projection.py',
    'experiments/multiresolution-wave/dynamics.py',
    'experiments/multiresolution-wave/heads32.py',
    'experiments/multiresolution-wave/nested_head.py',
    'experiments/fresh-wave-head/fresh_fom.py',
    'experiments/fresh-wave-head/fresh_models.py',
    'experiments/fresh-wave-head/fresh_rom.py',
    'experiments/fresh-wave-head/fresh_learning.py',
    'experiments/fresh-wave-head/fresh_evaluate.py',
    'experiments/fresh-wave-head/FROZEN-MATH.json',
    'experiments/multiresolution-wave/runs/accel12/cluster/in/ORIGIN.json',
] + [f'{INPUTS}/{name}' for name in (
    'bank_parameters.npz', 'campaign-config.json', 'coordinates.npz', 'data_manifest.json', 'head.npz',
    'head32.npz', 'initializer32.npz', 'initializer_trained_nested40.npz', 'trained_nested40.npz')]
EXCLUDE = 'pax007'


def main():
    attempt, config = sys.argv[1], sys.argv[2]
    gpu = sys.argv[3] if len(sys.argv) > 3 else 'a100'
    assert attempt.isalnum(), attempt
    assert gpu in ('a100', 'h100', 'h200', 'l40s'), gpu
    cfg = json.loads((ROOT / 'experiments/w-ladder' / config).read_text())
    files = FILES + [f'experiments/w-ladder/{config}']
    out = ROOT / 'experiments/w-ladder/runs' / attempt
    out.mkdir(parents=True, exist_ok=False)
    remote = f'{NAMESPACE}/{attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    proof = []
    for name in files:
        content = (ROOT / name).read_bytes()
        assert content == subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{name}']), name
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        proof.append(dict(source=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), commit=commit))
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=2) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    (out / 'logs').mkdir()
    script = '''#!/bin/bash
#SBATCH --job-name=wl___ATTEMPT__
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:__GPU__:1
#SBATCH --exclude=__EXCLUDE__
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=__HOURS__
#SBATCH --output=__REMOTE__/logs/%j.out
#SBATCH --error=__REMOTE__/logs/%j.err
set -euo pipefail
TASK_ROOT=__REMOTE__
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
"$PY" experiments/w-ladder/ladder.py \\
  --config experiments/w-ladder/__CONFIG__ \\
  --inputs __INPUTS__ \\
  --gates experiments/w-ladder/retained-gates.json \\
  --out output
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''
    for token, value in (('__ATTEMPT__', attempt), ('__REMOTE__', remote), ('__CONFIG__', config), ('__INPUTS__', INPUTS),
                         ('__HOURS__', cfg['walltime']), ('__EXCLUDE__', EXCLUDE), ('__GPU__', gpu)):
        script = script.replace(token, value)
    assert '__' not in script.replace('__pycache__', ''), script
    (out / 'run.sbatch').write_text(script)
    manifest = [f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}' for p in sorted(out.rglob('*')) if p.is_file()]
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote)


if __name__ == '__main__':
    main()
