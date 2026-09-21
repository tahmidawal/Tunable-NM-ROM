"""Stage one hires-poisson attempt flat into its own submit directory (one job per directory).

    python cluster/stage.py <attempt> --config config-2048.json [--gpu h200] [--mem 240G]

Every staged byte is checked against the Git object at HEAD, so a staged tree can never
contain an uncommitted edit. Adapted from experiments/p-linear/cluster/stage.py.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/hires_p_20260920'
CELL = 'experiments/multiresolution-poisson'
LANE = 'experiments/hires-poisson'

COMMON = [f'{CELL}/core.py', f'{CELL}/speed_core.py', f'{CELL}/kernel_solver.py',
          f'{CELL}/correction_core.py', f'{CELL}/iterative_core.py',
          'experiments/separable-decoder/sep_common.py',
          'experiments/cost-to-tolerance/ctol_tol.py',
          'experiments/wave2d-rom-latent-stepping/deps/multistage-precision/ms_parametric.py',
          'experiments/mr-burgers2d/engines.py',
          'experiments/head-ablation/arms.py',
          'experiments/p-bank-head/pbh_core.py',
          'experiments/p-linear/plin_core.py',
          'experiments/p-linear/checkpoints/primary_K32.pkl',
          'experiments/p-linear/checkpoints/primary_K32-basis.npz',
          f'{LANE}/hp_core.py', f'{LANE}/hp_audit_np.py', f'{LANE}/cluster/stage.py']

P3D = 'experiments/paper-p3d'
SET3D = [f'{P3D}/common.py', f'{P3D}/poisson.py', f'{P3D}/shared_rom.py', f'{P3D}/iterative_cg.py',
         f'{P3D}/runs/final08/checkpoints/bank.pkl', f'{P3D}/runs/final08/checkpoints/head_K16.pkl',
         f'{LANE}/cluster/stage.py']

LSH = f'{LANE}/lshape'
SETL = ['experiments/separable-decoder/sep_common.py', 'experiments/mr-burgers2d/engines.py',
        'experiments/head-ablation/arms.py', 'experiments/cost-to-tolerance/ctol_tol.py',
        f'{LSH}/lsh_core.py', f'{LSH}/models.json', f'{LSH}/IMPORTS.json',
        f'{LSH}/head_sdf_R512_K16.pkl', f'{LSH}/head_sdf_R512_K16-basis.npz',
        f'{LSH}/head_sdf_R512_K32.pkl', f'{LSH}/head_sdf_R512_K32-basis.npz',
        f'{LANE}/cluster/stage.py']

SCRIPT = '''#!/bin/bash
#SBATCH --job-name=hp___ATTEMPT__
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
"$PY" __DRIVER__ --config __CONFIG__ --out ../output
"$PY" __AUDIT__ ../output --subsample __SUB__ --delete-fields
__SECOND__
cd "$TASK_ROOT"
rm -rf cache tmp
find output $(test -d output2 && echo output2) -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('--config', required=True)
    p.add_argument('--driver', default='hp_solve.py')
    p.add_argument('--audit', default='hp_audit_np.py')
    p.add_argument('--extra', nargs='*', default=[], help='extra repo-relative files to stage')
    p.add_argument('--set', default='2d', choices=['2d', '3d', 'lshape'], help='which parent file set to stage')
    p.add_argument('--subsample', type=int, default=256)
    p.add_argument('--config2', default=None, help='optional second driver run in the same job (-> ../output2)')
    p.add_argument('--memfrac', default='0.75', help='XLA client memory fraction (0.95 for the 4096^2 bank)')
    p.add_argument('--hours', type=int, default=6)
    p.add_argument('--gpu', default='h200', choices=['a100', 'h100', 'h200', 'l40s'])
    p.add_argument('--mem', default='240G')
    a = p.parse_args()
    assert a.attempt.isalnum(), a.attempt
    out = ROOT / LANE / 'runs' / a.attempt
    out.mkdir(parents=True, exist_ok=False)
    (out / 'code').mkdir()
    (out / 'logs').mkdir()
    remote = f'{NAMESPACE}/{a.attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    files = list(dict.fromkeys({'2d': COMMON, '3d': SET3D, 'lshape': SETL}[a.set] + [f'{LANE}/{a.driver}', f'{LANE}/{a.audit}',
                                         f'{LANE}/{a.config}'] + ([f'{LANE}/{a.config2}'] if a.config2 else []) + a.extra))
    proof = []
    for name in files:
        content = (ROOT / name).read_bytes()
        assert content == subprocess.check_output(
            ['git', '-C', str(ROOT), 'show', f'{commit}:{name}']), f'uncommitted: {name}'
        dest = out / 'code' / Path(name).name
        assert not dest.exists(), f'flat name collision: {name}'
        dest.write_bytes(content)
        proof.append(dict(source=name, flat=Path(name).name, bytes=len(content),
                          sha256=hashlib.sha256(content).hexdigest(), commit=commit))
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=2) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    script = SCRIPT
    second = ('"$PY" __DRIVER__ --config %s --out ../output2\n"$PY" __AUDIT__ ../output2 --subsample __SUB__ --delete-fields'
              % a.config2) if a.config2 else ''
    script = script.replace('__SECOND__', second)
    for token, value in (('__ATTEMPT__', a.attempt), ('__REMOTE__', remote), ('__GPU__', a.gpu),
                         ('__HOURS__', f'{a.hours:02d}'), ('__MEM__', a.mem),
                         ('__DRIVER__', a.driver), ('__AUDIT__', a.audit),
                         ('__CONFIG__', a.config), ('__SUB__', str(a.subsample)), ('__MEMFRAC__', a.memfrac)):
        script = script.replace(token, value)
    (out / 'run.sbatch').write_text(script)
    manifest = [f'{hashlib.sha256(q.read_bytes()).hexdigest()}  {q.relative_to(out)}'
                for q in sorted(out.rglob('*')) if q.is_file()]
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote)


if __name__ == '__main__':
    main()
