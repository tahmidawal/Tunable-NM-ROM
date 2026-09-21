"""Stage one job of the nmrom-baselines lane: a committed, hashed code bundle + run.sbatch.

  python cluster/stage.py <attempt> --gpu a100 --hours 6 --mem 120G -- <script.py> <args...>

One attempt = one directory under the lane namespace (mkdir exist_ok=False locally; the remote
copy is made by the caller with rsync and then `sbatch` from tufts-login). Refuses uncommitted
lane files so the recorded commit is the code that ran.
"""
from pathlib import Path
import argparse, hashlib, shutil, subprocess, sys

HERE = Path(__file__).resolve().parent
LANE = HERE.parent
WT = LANE.parents[1]
NAMESPACE = '/cluster/tufts/paralab/tawal01/nmrombase_20260920'
FILES = ['experiments/nmrom-baselines/kimae.py', 'experiments/nmrom-baselines/lspg.py',
         'experiments/nmrom-baselines/gate_kim2d.py', 'experiments/nmrom-baselines/family.py',
         'experiments/nmrom-baselines/vendor/arms.py', 'experiments/nmrom-baselines/vendor/ladder.py',
         'experiments/nmrom-baselines/vendor/varpro.py', 'experiments/nmrom-baselines/vendor/topfix.py',
         'experiments/nmrom-baselines/vendor/directions_qtd02.npz',
         'experiments/mr-burgers2d/engines.py', 'experiments/mr-burgers2d/iterative_paths.py',
         'experiments/separable-decoder/sep_common.py',
         'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl']

FILES += sorted(str(p.relative_to(WT)) for p in list((LANE / 'configs').glob('*.json')) + list((LANE / 'runs').glob('gate*/output/summary.json')))

SCRIPT = r'''#!/bin/bash
#SBATCH --job-name=nmb_@NAME@
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:@GRES@:1
@CONSTRAINT@
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=@MEM@
#SBATCH --time=@HOURS@:00:00
#SBATCH --output=@ROOT@/logs/%j.out
#SBATCH --error=@ROOT@/logs/%j.err
set -euo pipefail
TASK_ROOT=@ROOT@
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
export PYTHONPATH="$TASK_ROOT/experiments/mr-burgers2d:$TASK_ROOT/experiments/separable-decoder:$TASK_ROOT/experiments/nmrom-baselines/vendor:$TASK_ROOT/experiments/nmrom-baselines"
"$PY" experiments/nmrom-baselines/@CMD@
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('attempt')
    ap.add_argument('--gpu', default='a100', choices=['a100', 'a100-80G', 'h100', 'h200', 'l40s'])
    ap.add_argument('--hours', type=int, default=6)
    ap.add_argument('--mem', default='120G')
    ap.add_argument('cmd', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = ' '.join(c for c in a.cmd if c != '--')
    assert cmd
    dirty = subprocess.check_output(['git', '-C', str(WT), 'status', '--porcelain', '--'] + [f for f in FILES if (WT / f).exists()], text=True)
    assert not dirty.strip(), 'commit lane files first:\n' + dirty
    commit = subprocess.check_output(['git', '-C', str(WT), 'rev-parse', 'HEAD'], text=True).strip()
    out = HERE / 'stage' / a.attempt
    out.mkdir(parents=True, exist_ok=False)
    for f in FILES:
        if not (WT / f).exists():
            continue
        (out / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(WT / f, out / f)
        blob = subprocess.check_output(['git', '-C', str(WT), 'show', f'{commit}:{f}'])
        assert blob == (out / f).read_bytes(), f
    (out / 'logs').mkdir()
    (out / 'COMMIT.txt').write_text(commit + '\n')
    root = f'{NAMESPACE}/{a.attempt}'
    gres = 'a100' if a.gpu.startswith('a100') else a.gpu
    script = (SCRIPT.replace('@NAME@', a.attempt).replace('@GRES@', gres).replace('@MEM@', a.mem)
              .replace('@CONSTRAINT@', '#SBATCH --constraint=a100-80G' if a.gpu == 'a100-80G' else '')
              .replace('@HOURS@', f'{a.hours:02d}').replace('@ROOT@', root).replace('@CMD@', cmd))
    assert '@' not in script.replace('tawal01@', '')
    (out / 'run.sbatch').write_text(script)
    files = sorted(p for p in out.rglob('*') if p.is_file() and p.name != 'MANIFEST.sha256' and 'logs' not in p.parts)
    (out / 'MANIFEST.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(out)}\n' for p in files))
    print(out)
    print(root)


if __name__ == '__main__':
    main()
