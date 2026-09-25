"""poisson-lshape3d-test: stage one TEST attempt (adapted from poisson-bank-knob-3d/cluster/stage.py).

Stage one attempt flat into its own submit directory (one job per directory).

    python cluster/stage.py <attempt> --set lshape|cube --config config-lshape-2048.json [--gpu h200] [--hours 4]

Every staged byte is checked against the Git object at HEAD, so a staged tree can never contain an
uncommitted edit. Adapted from experiments/poisson-bank-knob/cluster/stage.py.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/pl3test_20260925'
LANE = 'experiments/poisson-lshape3d-test'
PARENT = 'experiments/poisson-bank-knob-3d'
LSH = 'experiments/hires-poisson/lshape'
P3D = 'experiments/paper-p3d'
SETS = dict(
    lshape=['experiments/separable-decoder/sep_common.py', 'experiments/mr-burgers2d/engines.py',
            'experiments/head-ablation/arms.py', f'{LSH}/lsh_core.py', f'{LSH}/models.json',
            f'{LSH}/head_sdf_R512_K16.pkl', f'{LSH}/head_sdf_R512_K16-basis.npz',
            f'{PARENT}/runs/prep_lshape.npz', f'{PARENT}/pbk3_core.py', f'{LANE}/plt_lshape.py',
            f'{PARENT}/pbk3_audit_np.py', f'{LANE}/cluster/stage.py', f'{LANE}/frozen-test-lshape.json'],
    cube=[f'{P3D}/common.py', f'{P3D}/poisson.py', f'{P3D}/shared_rom.py', f'{P3D}/iterative_cg.py',
          f'{P3D}/runs/final08/checkpoints/bank.pkl', f'{P3D}/runs/final08/checkpoints/head_K16.pkl',
          f'{P3D}/runs/final08/checkpoints/cohorts.json',
          f'{PARENT}/runs/prep_cube.npz', f'{PARENT}/pbk3_core.py', f'{LANE}/plt_cube.py',
          f'{PARENT}/pbk3_audit_np.py', f'{LANE}/cluster/stage.py', f'{LANE}/frozen-test-cube.json'])
DRIVER = dict(lshape='plt_lshape.py', cube='plt_cube.py')

SCRIPT = '''#!/bin/bash
#SBATCH --job-name=pl3t___ATTEMPT__
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:__GPU__:1
__CONSTRAINT__#SBATCH --exclude=pax007
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
# ---- field storage (poisson-lshape3d-test): saved fields go to node-local /tmp when it has room (tmpfs on the
# H200 nodes, charged to this job's memory, hence the larger --mem), else to the share if it keeps >= 100 GB free,
# else abort (exit 43) before any work. The audit deletes them; the EXIT trap removes the local copy in any case.
NEED_GB=__NEEDGB__
mkdir -p "$TASK_ROOT/output"
LOCAL_FIELDS=/tmp/pl3t___ATTEMPT___${SLURM_JOB_ID}
AVAIL_LOCAL=$(df --output=avail -BG /tmp | tail -1 | tr -dc 0-9)
AVAIL_SHARE=$(df --output=avail -BG /cluster/tufts/paralab | tail -1 | tr -dc 0-9)
echo "fields_need_gb=$NEED_GB avail_local_gb=$AVAIL_LOCAL avail_share_gb=$AVAIL_SHARE"
if [ "$AVAIL_LOCAL" -ge $((NEED_GB * 11 / 10 + 10)) ]; then
  mkdir -p "$LOCAL_FIELDS"; trap 'rm -rf "$LOCAL_FIELDS"' EXIT
  ln -s "$LOCAL_FIELDS" "$TASK_ROOT/output/fields"; echo "fields_store=local $LOCAL_FIELDS"
elif [ "$AVAIL_SHARE" -ge $((NEED_GB + 100)) ]; then
  mkdir -p "$TASK_ROOT/output/fields"; echo "fields_store=share"
else
  echo "fields_store=NONE (need $NEED_GB GB)"; exit 43
fi
cd "$TASK_ROOT/code"
"$PY" __DRIVER__ --config __CONFIG__ --out ../output__SMOKE__
"$PY" pbk3_audit_np.py ../output --delete-fields
cd "$TASK_ROOT"
if [ -L output/fields ]; then rm output/fields; rm -rf "$LOCAL_FIELDS"; else rmdir output/fields 2>/dev/null || true; fi
rm -rf cache tmp
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('--set', required=True, choices=sorted(SETS))
    p.add_argument('--config', required=True)
    p.add_argument('--extra', nargs='*', default=[], help='extra lane-relative files to stage')
    p.add_argument('--memfrac', default='0.90')
    p.add_argument('--hours', type=int, default=4)
    p.add_argument('--gpu', default='h200', choices=['a100', 'h100', 'h200'])
    p.add_argument('--mem', default='240G')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--constraint', default='')
    p.add_argument('--need-gb', type=int, required=True, help='estimated saved-field volume (GB)')
    a = p.parse_args()
    assert a.attempt.isalnum(), a.attempt
    out = ROOT / LANE / 'runs' / a.attempt
    out.mkdir(parents=True, exist_ok=False)
    (out / 'code').mkdir()
    (out / 'logs').mkdir()
    remote = f'{NAMESPACE}/{a.attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    files = list(dict.fromkeys(SETS[a.set] + [f'{LANE}/{a.config}'] + [f'{LANE}/{x}' for x in a.extra]))
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
    for token, value in (('__ATTEMPT__', a.attempt), ('__REMOTE__', remote), ('__GPU__', a.gpu),
                         ('__HOURS__', f'{a.hours:02d}'), ('__MEM__', a.mem), ('__DRIVER__', DRIVER[a.set]),
                         ('__CONFIG__', Path(a.config).name), ('__MEMFRAC__', a.memfrac),
                         ('__SMOKE__', ' --smoke' if a.smoke else ''),
                         ('__CONSTRAINT__', f'#SBATCH --constraint={a.constraint}\n' if a.constraint else ''),
                         ('__NEEDGB__', str(a.need_gb))):
        script = script.replace(token, value)
    (out / 'run.sbatch').write_text(script)
    manifest = [f'{hashlib.sha256(q.read_bytes()).hexdigest()}  {q.relative_to(out)}'
                for q in sorted(out.rglob('*')) if q.is_file()]
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote)


if __name__ == '__main__':
    main()
