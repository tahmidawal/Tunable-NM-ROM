"""Stage one b-qxm attempt into an isolated paralab-bound directory.

    python cluster/stage.py <attempt> <config-file-name> [<gpu>] [<constraint>] [<hours>]

Each attempt gets its OWN submit directory and its OWN remote directory: one job per
directory, never two. Every staged file is checked byte-for-byte against the committed
blob at HEAD before it is copied, so what runs on the cluster is exactly what is in Git.
Mechanics copied from `experiments/q-ridge/cluster/stage.py` (job qrg304); the one
addition is `XLA_PYTHON_CLIENT_MEM_FRACTION=0.55` (default 0.75). Compiled executables
(CUBINs) are loaded OUTSIDE XLA's preallocated pool, which is where qtd01 died
(`Failed to load in-memory CUBIN: CUDA_ERROR_OUT_OF_MEMORY`); a smaller pool leaves more
room for them. The arrays and peak workspace this lane needs (<= ~10 GB, measured by
`probe_memory.py`) fit a 22 GB pool on a 40 GB card. No arithmetic changes.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/b_qxm_20260917'
FILES = [
    'experiments/b-qxm/q_xm.py',
    'experiments/b-qxm/cluster/stage.py',
    'experiments/q-ridge/ridge.py',
    'experiments/b-ladder-top/topfix.py',
    'experiments/cheap-corrections/varpro.py',
    'experiments/cheap-corrections/directions.py',
    'experiments/head-ablation/arms.py',
    'experiments/head-ablation/ladder.py',
    'experiments/mr-burgers2d/engines.py',
    'experiments/mr-burgers2d/iterative_paths.py',
    'experiments/mr-burgers2d/accuracy_paths.py',
    'experiments/separable-decoder/sep_common.py',
    'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl',
]
CHECKPOINT = 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
HOURS = '08:00:00'
EXCLUDE = 'pax007'


def main():
    attempt, config = sys.argv[1], sys.argv[2]
    assert attempt.isalnum(), attempt
    gpu = sys.argv[3] if len(sys.argv) > 3 else 'a100'
    assert gpu in ('a100', 'h100', 'h200', 'l40s'), gpu
    constraint = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != '-' else None
    hours = sys.argv[5] if len(sys.argv) > 5 else HOURS
    files = FILES + [f'experiments/b-qxm/{config}']
    out = ROOT / 'experiments/b-qxm/runs' / attempt
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
        proof.append(dict(source=name, bytes=len(content),
                          sha256=hashlib.sha256(content).hexdigest(), commit=commit))
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=2) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    (out / 'logs').mkdir()
    script = '''#!/bin/bash
#SBATCH --job-name=bqx___ATTEMPT__
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:__GPU__:1
#SBATCH --exclude=__EXCLUDE__
__CONSTRAINT__
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=__HOURS__
#SBATCH --output=__REMOTE__/logs/%j.out
#SBATCH --error=__REMOTE__/logs/%j.err
set -euo pipefail
TASK_ROOT=__REMOTE__
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
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
export PYTHONPATH="$TASK_ROOT/experiments/mr-burgers2d:$TASK_ROOT/experiments/head-ablation:$TASK_ROOT/experiments/cheap-corrections:$TASK_ROOT/experiments/b-ladder-top:$TASK_ROOT/experiments/q-ridge:$TASK_ROOT/experiments/b-qxm"
"$PY" experiments/b-qxm/q_xm.py \\
  --config experiments/b-qxm/__CONFIG__ \\
  --checkpoint __CKPT__ \\
  --out output
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''
    for token, value in (('__ATTEMPT__', attempt), ('__REMOTE__', remote),
                         ('__CKPT__', CHECKPOINT), ('__CONFIG__', config),
                         ('__HOURS__', hours), ('__EXCLUDE__', EXCLUDE), ('__GPU__', gpu),
                         ('__CONSTRAINT__', f'#SBATCH --constraint={constraint}' if constraint else '')):
        script = script.replace(token, value)
    assert '__' not in script.replace('__pycache__', ''), script
    (out / 'run.sbatch').write_text('\n'.join(l for l in script.split('\n') if l.strip() != '') if False else script.replace('\n\n', '\n'))
    manifest = [f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}'
                for p in sorted(out.rglob('*')) if p.is_file()]
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote)


if __name__ == '__main__':
    main()
