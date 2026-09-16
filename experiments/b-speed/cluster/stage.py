"""Stage one b-speed attempt into its own isolated paralab-bound job directory.

One attempt directory per job; never two submissions from one directory. Every staged
file is byte-compared against the current commit before it is written.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/b_speed_20260916'
FILES = [
    'experiments/b-speed/speed.py',
    'experiments/b-speed/fast.py',
    'experiments/b-speed/ladders.py',
    'experiments/b-speed/cluster/stage.py',
    'experiments/head-ablation/arms.py',
    'experiments/mr-burgers2d/engines.py',
    'experiments/mr-burgers2d/iterative_paths.py',
    'experiments/separable-decoder/sep_common.py',
    'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl',
]
REFERENCE = [f'experiments/b-speed/reference/L256_a_neural_eq_case{c}_rep0.npz'
             for c in range(6)]
CHECKPOINT = 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'

SCRIPT = '''#!/bin/bash
#SBATCH --job-name=bspeed_@ATTEMPT@
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=@MEM@G
#SBATCH --time=@HOURS@
#SBATCH --output=@REMOTE@/logs/%j.out
#SBATCH --error=@REMOTE@/logs/%j.err
set -euo pipefail
TASK_ROOT=@REMOTE@
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
export PYTHONPATH="$TASK_ROOT/experiments/mr-burgers2d:$TASK_ROOT/experiments/head-ablation:$TASK_ROOT/experiments/separable-decoder:$TASK_ROOT/experiments/b-speed"
"$PY" experiments/b-speed/speed.py \\
  --config experiments/b-speed/@CONFIG@ \\
  --checkpoint @CKPT@ \\
  @REFARG@ \\
  --out output
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('--config', required=True)
    p.add_argument('--hours', default='05:30:00')
    p.add_argument('--mem', default='200')
    p.add_argument('--reference', action='store_true')
    a = p.parse_args()
    assert a.attempt.isalnum(), a.attempt
    out = ROOT / 'experiments/b-speed/runs' / a.attempt
    out.mkdir(parents=True, exist_ok=False)
    remote = f'{NAMESPACE}/{a.attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                                     text=True).strip()
    files = list(FILES) + [f'experiments/b-speed/{a.config}']
    if a.reference:
        files += REFERENCE
    proof = []
    for name in files:
        content = (ROOT / name).read_bytes()
        assert content == subprocess.check_output(
            ['git', '-C', str(ROOT), 'show', f'{commit}:{name}']), name
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        proof.append(dict(source=name, bytes=len(content),
                          sha256=hashlib.sha256(content).hexdigest(), commit=commit))
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=2) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    (out / 'logs').mkdir()
    refarg = '--reference experiments/b-speed/reference' if a.reference else ''
    script = (SCRIPT.replace('@ATTEMPT@', a.attempt).replace('@REMOTE@', remote)
              .replace('@CKPT@', CHECKPOINT).replace('@HOURS@', a.hours)
              .replace('@MEM@', a.mem).replace('@CONFIG@', a.config)
              .replace('@REFARG@', refarg))
    assert '@' not in script, script
    (out / 'run.sbatch').write_text(script)
    manifest = [f'{hashlib.sha256(q.read_bytes()).hexdigest()}  {q.relative_to(out)}'
                for q in sorted(out.rglob('*')) if q.is_file()]
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote)


if __name__ == '__main__':
    main()
