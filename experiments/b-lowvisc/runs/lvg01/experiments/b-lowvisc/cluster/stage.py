"""Stage one b-lowvisc attempt into an isolated paralab-bound directory.

    python experiments/b-lowvisc/cluster/stage.py lvg01 [a100]

Mechanics copied from `b-seeds/cluster/stage.py` (itself q-ridge's): every staged file is
checked byte-for-byte against the committed blob at HEAD before it is copied, so what runs on
the cluster is exactly what is in Git; `COMMIT.txt`, `PROVENANCE.json` and `MANIFEST.sha256`
are written beside it; the sbatch activates the paralab venv, exports `JAX_ENABLE_X64=true`
and `JAX_DEFAULT_MATMUL_PRECISION=highest`, runs the GPU preflight (`jax_backend=gpu` or exit
42), writes `OUTPUTS.sha256` and ends with `ALL-DONE`.  One job per attempt directory, ever.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/b_lowvisc_20260917'
EXCLUDE = 'pax007'

GATE_FILES = [
    'experiments/b-lowvisc/gate.py',
    'experiments/b-lowvisc/lv_common.py',
    'experiments/b-lowvisc/audit_gate.py',
    'experiments/b-lowvisc/cluster/stage.py',
    'experiments/b-lowvisc/config-gate.json',
    'experiments/mr-burgers2d/engines.py',
    'experiments/mr-burgers2d/iterative_paths.py',
    'experiments/head-ablation/ladder.py',
    'experiments/head-ablation/ablation.py',
    'experiments/head-ablation/arms.py',
    'experiments/separable-decoder/sep_common.py',
]

PREAMBLE = '''#!/bin/bash
#SBATCH --job-name=lv___ATTEMPT__
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
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT attempt=__ATTEMPT__ started=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
LANE_PATH="$TASK_ROOT/experiments/mr-burgers2d:$TASK_ROOT/experiments/head-ablation:$TASK_ROOT/experiments/b-lowvisc"
'''

GATE_BODY = '''
# ---------------------------------------------------------------- the gate job ----
# Both viscosity families, one allocation, one GPU, no trained model (DESIGN.md section 5).
echo "GATE $(date -Is)"
PYTHONPATH="$LANE_PATH" "$PY" experiments/b-lowvisc/gate.py \\
  --config experiments/b-lowvisc/config-gate.json --out output/gate

echo "STAGES DONE $(date -Is)"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''

BODIES = {'lvg': (GATE_FILES, GATE_BODY, '6:00:00')}


def main():
    attempt = sys.argv[1]
    gpu = sys.argv[2] if len(sys.argv) > 2 else 'a100'
    assert attempt.isalnum(), attempt
    assert gpu in ('a100', 'h100', 'h200', 'l40s'), gpu
    kind = attempt[:3]
    assert kind in BODIES, f'unknown attempt kind {kind!r}; known: {sorted(BODIES)}'
    files, body, hours = BODIES[kind]

    out = Path(os.environ.get('STAGE_ROOT', ROOT / 'experiments/b-lowvisc/runs')) / attempt
    out.mkdir(parents=True, exist_ok=False)
    remote = f'{NAMESPACE}/{attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True)
    assert not dirty.strip(), f'working tree not clean:\n{dirty}'
    proof = []
    for name in files:
        content = (ROOT / name).read_bytes()
        assert content == subprocess.check_output(
            ['git', '-C', str(ROOT), 'show', f'{commit}:{name}']), f'uncommitted: {name}'
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        proof.append(dict(source=name, staged_as=name, bytes=len(content),
                          sha256=hashlib.sha256(content).hexdigest(), commit=commit))
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=2) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    (out / 'logs').mkdir()
    script = PREAMBLE + body
    for token, value in (('__ATTEMPT__', attempt), ('__REMOTE__', remote), ('__GPU__', gpu),
                         ('__HOURS__', hours), ('__EXCLUDE__', EXCLUDE)):
        script = script.replace(token, value)
    assert '__' not in script.replace('__pycache__', ''), script
    (out / 'run.sbatch').write_text(script)
    manifest = [f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}'
                for p in sorted(out.rglob('*')) if p.is_file()]
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote)
    print(f'rsync -a {out}/ tufts-login:{remote}/')
    print(f"ssh tufts-login 'cd {remote} && sha256sum -c MANIFEST.sha256 --quiet && sbatch run.sbatch'")


if __name__ == '__main__':
    main()
