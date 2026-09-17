"""Stage one b-eqtop attempt into an isolated paralab-bound directory.

    python cluster/stage.py <attempt> <config-file-name> [--gpu a100|h100|h200|l40s]

One job per directory, never two. Every staged file is checked byte-for-byte against the
committed blob at HEAD before it is copied, so what runs on the cluster is exactly what is
in Git. The sbatch activates the paralab venv, exports x64 + highest precision, runs the
GPU preflight (exit 42 on CPU), writes OUTPUTS.sha256 and ends with ALL-DONE.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/b_eqtop_20260917'
FILES = [
    'experiments/b-eqtop/q_eqtop.py',
    'experiments/b-eqtop/eqtop.py',
    'experiments/b-eqtop/fitworker.py',
    'experiments/b-eqtop/cluster/stage.py',
    'experiments/b-eqtop/rules/qrg304/MANIFEST.json',
    'experiments/q-ridge/eqcert.py',
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
HOURS = '20:00:00'
EXCLUDE = 'pax007'
GPU = {'a100': '#SBATCH --gres=gpu:a100:1\n#SBATCH --constraint=a100-80G',
       'h100': '#SBATCH --gres=gpu:h100:1',
       'h200': '#SBATCH --gres=gpu:h200:1',
       'l40s': '#SBATCH --gres=gpu:l40s:1'}


def main():
    attempt, config = sys.argv[1], sys.argv[2]
    gpu = sys.argv[sys.argv.index('--gpu') + 1] if '--gpu' in sys.argv else 'a100'
    assert attempt.isalnum(), attempt
    files = FILES + [f'experiments/b-eqtop/{config}']
    files += [f'experiments/b-eqtop/rules/qrg304/{p.name}'
              for p in sorted((ROOT / 'experiments/b-eqtop/rules/qrg304').glob('*.npz'))]
    out = ROOT / 'experiments/b-eqtop/runs' / attempt
    out.mkdir(parents=True, exist_ok=False)
    remote = f'{NAMESPACE}/{attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
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
    script = '''#!/bin/bash
#SBATCH --job-name=bet___ATTEMPT__
#SBATCH --partition=gpu
#SBATCH --qos=normal
__GPU__
#SBATCH --exclude=__EXCLUDE__
#SBATCH --cpus-per-task=16
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
echo "host=$(hostname) source_commit=$SOURCE_COMMIT cpus=$SLURM_CPUS_PER_TASK"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
export PYTHONPATH="$TASK_ROOT/experiments/mr-burgers2d:$TASK_ROOT/experiments/head-ablation:$TASK_ROOT/experiments/cheap-corrections:$TASK_ROOT/experiments/b-ladder-top:$TASK_ROOT/experiments/q-ridge:$TASK_ROOT/experiments/b-eqtop"
"$PY" experiments/b-eqtop/q_eqtop.py \
  --config experiments/b-eqtop/__CONFIG__ \
  --checkpoint __CKPT__ \
  --rules experiments/b-eqtop/rules/qrg304 \
  --tmp "$TMPDIR/fits" \
  --out output
rm -rf "$TMPDIR/fits"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''
    for token, value in (('__ATTEMPT__', attempt), ('__REMOTE__', remote),
                         ('__CKPT__', CHECKPOINT), ('__CONFIG__', config),
                         ('__HOURS__', HOURS), ('__EXCLUDE__', EXCLUDE), ('__GPU__', GPU[gpu])):
        script = script.replace(token, value)
    assert '__' not in script.replace('__pycache__', ''), script
    (out / 'run.sbatch').write_text(script)
    manifest = [f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}'
                for p in sorted(out.rglob('*')) if p.is_file()]
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote)


if __name__ == '__main__':
    main()
