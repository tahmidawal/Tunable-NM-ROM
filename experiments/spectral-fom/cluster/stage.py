"""Stage one spectral-fom attempt flat into its own submit directory (one job per directory).

    python cluster/stage.py <attempt> --runs "sp2d_solve.py:config-p2d-256.json" "sp2d_solve.py:config-p2d-1024.json" \
        --set p2d [--gpu a100 | --constraint "a100-80G|h100-80G|h200-141G"] [--mem 240G] [--hours 6]

Each `--runs` entry is  driver:config  and runs sequentially in the SAME allocation, into ../output<i>.
Every staged byte is checked against a Git object: this lane's files against HEAD, and files of other
lanes against the owning lane's pinned commit (FILESETS below), so a staged tree never contains an
uncommitted edit and the ROM code is exactly the owning lane's committed code.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/specfom_20260923'
LANE = 'experiments/spectral-fom'
MINE = [f'{LANE}/spec_core.py', f'{LANE}/spec_timing.py', f'{LANE}/sp_audit_np.py', f'{LANE}/cluster/stage.py']

MRP = 'experiments/multiresolution-poisson'
PBK = 'experiments/poisson-bank-knob'
PBK_COMMIT = '06546331'      # exp/2026-09-23-poisson-bank-knob, lane complete
P2D = [(f, PBK_COMMIT) for f in (
    f'{MRP}/core.py', f'{MRP}/speed_core.py', f'{MRP}/kernel_solver.py', f'{MRP}/correction_core.py',
    f'{MRP}/iterative_core.py', 'experiments/separable-decoder/sep_common.py',
    'experiments/cost-to-tolerance/ctol_tol.py',
    'experiments/wave2d-rom-latent-stepping/deps/multistage-precision/ms_parametric.py',
    'experiments/mr-burgers2d/engines.py', 'experiments/head-ablation/arms.py',
    'experiments/p-bank-head/pbh_core.py', 'experiments/p-linear/plin_core.py',
    'experiments/p-linear/checkpoints/primary_K32.pkl', 'experiments/p-linear/checkpoints/primary_K32-basis.npz',
    'experiments/hires-poisson/hp_core.py', f'{PBK}/pbk_core.py', f'{PBK}/runs/prep.npz')]
FILESETS = {'p2d': P2D + [(f'{LANE}/sp2d_solve.py', 'HEAD')]}

SCRIPT = '''#!/bin/bash
#SBATCH --job-name=specfom___ATTEMPT__
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:__GPU__1
__CONSTRAINT__
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
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=__MEMFRAC__
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
export TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd "$TASK_ROOT/code"
__RUNS__
cd "$TASK_ROOT"
rm -rf cache tmp
find output* -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''


def git_bytes(commit, name):
    return subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{name}'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--set', required=True, choices=sorted(FILESETS))
    p.add_argument('--extra', nargs='*', default=[], help='extra (this lane, HEAD) files')
    p.add_argument('--memfrac', default='0.90')
    p.add_argument('--hours', type=int, default=6)
    p.add_argument('--gpu', default='a100', choices=['a100', 'h100', 'h200', 'l40s'])
    p.add_argument('--mem', default='240G')
    p.add_argument('--constraint', default=None)
    p.add_argument('--local', action='store_true', help='stage for a local smoke only (runs-local/)')
    a = p.parse_args()
    assert a.attempt.isalnum(), a.attempt
    out = ROOT / LANE / ('runs-local' if a.local else 'runs') / a.attempt
    out.mkdir(parents=True, exist_ok=False)
    (out / 'code').mkdir()
    (out / 'logs').mkdir()
    remote = f'{NAMESPACE}/{a.attempt}'
    head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    files = [(f, 'HEAD') for f in MINE] + FILESETS[a.set] + [(f'{LANE}/{r.split(":")[1]}', 'HEAD') for r in a.runs] \
        + [(f'{LANE}/{e}', 'HEAD') for e in a.extra]
    files = list(dict.fromkeys(files))
    proof = []
    for name, commit in files:
        c = head if commit == 'HEAD' else subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', commit], text=True).strip()
        content = git_bytes(c, name)
        if commit == 'HEAD':
            assert content == (ROOT / name).read_bytes(), f'uncommitted: {name}'
        dest = out / 'code' / Path(name).name
        assert not dest.exists(), f'flat name collision: {name}'
        dest.write_bytes(content)
        proof.append(dict(source=name, flat=Path(name).name, bytes=len(content),
                          sha256=hashlib.sha256(content).hexdigest(), commit=c))
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=2) + '\n')
    (out / 'COMMIT.txt').write_text(head + '\n')
    runs = []
    for i, r in enumerate(a.runs):
        drv, cfg = r.split(':')
        o = f'../output{i}'
        runs.append(f'"$PY" {drv} --config {cfg} --out {o}\n"$PY" sp_audit_np.py {o} --delete-fields || echo "AUDIT-FAILED {o}"')
    script = SCRIPT.replace('__RUNS__', '\n'.join(runs))
    for token, value in (('__ATTEMPT__', a.attempt), ('__REMOTE__', remote),
                         ('__GPU__', '' if a.constraint else a.gpu + ':'),
                         ('__CONSTRAINT__', f'#SBATCH --constraint="{a.constraint}"' if a.constraint else ''),
                         ('__HOURS__', f'{a.hours:02d}'), ('__MEM__', a.mem), ('__MEMFRAC__', a.memfrac)):
        script = script.replace(token, value)
    (out / 'run.sbatch').write_text(script)
    manifest = [f'{hashlib.sha256(q.read_bytes()).hexdigest()}  {q.relative_to(out)}'
                for q in sorted(out.rglob('*')) if q.is_file()]
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote)


if __name__ == '__main__':
    main()
