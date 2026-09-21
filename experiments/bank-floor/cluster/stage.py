"""Stage committed source bytes into an isolated paralab-bound attempt directory.

usage: stage.py ATTEMPT KIND [--gpu a100] [--time HH:MM:SS] [--mem 120G]
KIND: rep-burgers | rep-poisson | solve-burgers | solve-poisson
Every staged byte is checked against the Git object at HEAD.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/bankfloor_20260920'
X = 'experiments'
BURGERS_CKPT = f'{X}/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
POISSON_CKPT = f'{X}/bank-floor/incumbents/poisson_primary_K32.pkl'
COMMON = [f'{X}/bank-floor/bf_core.py', f'{X}/bank-floor/bf_rep.py', f'{X}/bank-floor/cluster/stage.py',
          f'{X}/separable-decoder/sep_common.py', f'{X}/mr-burgers2d/engines.py',
          f'{X}/head-ablation/arms.py']
KINDS = {
    'rep-burgers': dict(
        files=COMMON + [f'{X}/b-head-train/common.py', f'{X}/separable-decoder/sep_hfit.py',
                        f'{X}/bank-floor/config-rep-burgers.json', BURGERS_CKPT],
        cmd=f'"$PY" {X}/bank-floor/bf_rep.py --config {X}/bank-floor/config-rep-burgers.json '
            f'--incumbent {BURGERS_CKPT} --out output'),
    'rep-poisson': dict(
        files=COMMON + [f'{X}/bank-floor/pbh_core.py', f'{X}/multiresolution-poisson/core.py',
                        f'{X}/cost-to-tolerance/ctol_tol.py',
                        f'{X}/wave2d-rom-latent-stepping/deps/multistage-precision/ms_parametric.py',
                        f'{X}/bank-floor/config-rep-poisson.json', POISSON_CKPT],
        cmd=f'"$PY" {X}/bank-floor/bf_rep.py --config {X}/bank-floor/config-rep-poisson.json '
            f'--incumbent {POISSON_CKPT} --out output'),
}
PYPATH = ':'.join(f'$TASK_ROOT/{X}/{d}' for d in (
    'bank-floor', 'mr-burgers2d', 'head-ablation', 'separable-decoder', 'b-head-train',
    'multiresolution-poisson', 'cost-to-tolerance',
    'wave2d-rom-latent-stepping/deps/multistage-precision'))
SCRIPT = '''#!/bin/bash
#SBATCH --job-name=bf_ATTEMPT
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:GPU:1
#SBATCH --exclude=pax007
CONSTRAINT#SBATCH --cpus-per-task=8
#SBATCH --mem=MEM
#SBATCH --time=TIME
#SBATCH --output=REMOTE/logs/%j.out
#SBATCH --error=REMOTE/logs/%j.err
set -euo pipefail
TASK_ROOT=REMOTE
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
export TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
export PYTHONPATH="PYPATH"
CMD
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('attempt')
    ap.add_argument('kind')
    ap.add_argument('--gpu', default='a100')
    ap.add_argument('--time', default='10:00:00')
    ap.add_argument('--mem', default='120G')
    ap.add_argument('--constraint', default='')
    ap.add_argument('--extra', nargs='*', default=[], help='extra committed files to stage')
    ap.add_argument('--cmd', default=None, help='override command')
    a = ap.parse_args()
    assert a.attempt.isalnum()
    kind = KINDS[a.kind] if a.kind in KINDS else dict(files=COMMON, cmd=a.cmd)
    out = ROOT / X / 'bank-floor/runs' / a.attempt
    out.mkdir(parents=True, exist_ok=False)
    remote = f'{NAMESPACE}/{a.attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    proof = []
    for name in dict.fromkeys(kind['files'] + a.extra):
        content = (ROOT / name).read_bytes()
        assert content == subprocess.check_output(
            ['git', '-C', str(ROOT), 'show', f'{commit}:{name}']), f'uncommitted: {name}'
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        proof.append(dict(source=name, bytes=len(content),
                          sha256=hashlib.sha256(content).hexdigest(), commit=commit))
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=2) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    (out / 'logs').mkdir()
    (out / 'run.sbatch').write_text(
        SCRIPT.replace('CMD', a.cmd or kind['cmd']).replace('PYPATH', PYPATH)
        .replace('CONSTRAINT', f'#SBATCH --constraint={a.constraint}\n' if a.constraint else '')
        .replace('ATTEMPT', a.attempt).replace('REMOTE', remote).replace('GPU', a.gpu)
        .replace('MEM', a.mem).replace('TIME', a.time))
    manifest = [f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}'
                for p in sorted(out.rglob('*')) if p.is_file()]
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote)


if __name__ == '__main__':
    main()
