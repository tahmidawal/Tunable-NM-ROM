"""imp-fronts ROM sbatch templates (DESIGN B4, B5, B8 / B9 as amended by Amendments 3-5). Prints (or writes with --out) a
complete sbatch script; never submits. The staged tree under the job directory must mirror the repository layout
(experiments/imp-fronts/rom/... plus experiments/mr-burgers2d/{engines,iterative_paths}.py and
experiments/separable-decoder/sep_common.py, which rom/env.sh puts on PYTHONPATH) and carry MANIFEST.sha256 + COMMIT.txt.

Kinds
  sel  <label>            SEL-<label>: r2acc on configs/e2s<label>.json (R384 group only, mkconfigs_sel.py --r384;
                          select2d_r384.py runs locally on the pulled output).
                          Wall = ceiling 0.3 GPU-h / 1 GPU (Amendment 9); r2acc has no self-timer, so it runs under
                          `timeout` at 90 % of the wall (exhaustion: incomplete result -> select2d refuses -> no
                          certified rule -> m* route FAIL, ROM routes run from the uncertified start, Amendment 4).
  pol  <bank> <config>    POL-<bank>: fpolicy on configs/<config>.json, wall = ceiling 0.4 GPU-h; fpolicy's self-timer
                          stops new work at 90 % of the wall counted from job start (--t0-epoch); a killed job is
                          finalised locally by rom/fcollect.py.
  smb-rom                 the ROM part of SM-b as a shell FRAGMENT for the SM-b job (which owns the preamble, the wall
                          0.45 h and the smoke-bank training): runtime registration of the smoke bank, fpolicy round 1
                          on FC1 dev j = 0, 1 with K1 (+ C-B1, C-B6) and with the smoke bank.
  smb-rom-standalone      the same fragment wrapped in a full 0.45 h script (K1 part only works without the smoke bank).

    python render.py sel a [--out runs/SEL-a/run.sbatch]
    python render.py pol K1 polK1
    python render.py smb-rom
"""
import argparse
import math
import sys
from pathlib import Path

NS = '/cluster/tufts/paralab/tawal01/imp-fronts'
VENV = '/cluster/tufts/paralab/tawal01/ae-research/venv'
GRES = {'a100': 'a100', 'h100': 'h100', 'h200': 'h200', 'l40s': 'l40s'}
CEIL = {'sel': .3, 'pol': .4, 'smb': .45}


def hms(hours):
    s = int(math.floor(hours * 3600 + 1e-6))
    return f'{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}'


def preamble(job, hours, gpu='a100', mem='120G', cpus=8):
    remote = f'{NS}/{job}'
    return f'''#!/bin/bash
#SBATCH --job-name=imf_{job}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:{GRES[gpu]}:1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={hms(hours)}
#SBATCH --output={remote}/logs/%j.out
#SBATCH --error={remote}/logs/%j.err
# imp-fronts {job}: ceiling {hours} GPU-h on 1 GPU = Slurm wall limit (DESIGN Amendment 4 B8). gpu partition only.
set -euo pipefail
T0=$(date +%s)
export TASK_ROOT={remote}
WALL_S={int(round(hours * 3600))}
STOP_S=$(( WALL_S * 9 / 10 ))                    # self-timer: no new work after 90 % of the wall
PY={VENV}/bin/python
source {VENV}/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OPENBLAS_NUM_THREADS={cpus} OMP_NUM_THREADS={cpus}
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output" "$TASK_ROOT/logs"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) job=$SLURM_JOB_ID source_commit=$SOURCE_COMMIT precision=$JAX_DEFAULT_MATMUL_PRECISION"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}', flush=True); sys.exit(0 if b == 'gpu' else 42)"
source experiments/imp-fronts/rom/env.sh
cd experiments/imp-fronts/rom
left() {{ echo $(( STOP_S - ($(date +%s) - T0) )); }}
'''


def epilogue():
    return '''cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''


def sel(label):
    assert label.isalnum() and label != '1', 'K1 selection is taken from jcp-robust (not rerun)'
    job = f'SEL-{label}'
    body = f'''L=$(left); [ "$L" -gt 60 ] || {{ echo "no time left"; exit 1; }}
echo "STEP r2acc configs/e2s{label}.json timeout=${{L}}s $(date -u +%FT%TZ)"
set +e
timeout --signal=TERM "$L" "$PY" r2acc.py --config configs/e2s{label}.json --out "$TASK_ROOT/output/r2acc_e2s{label}"
RC=$?
set -e
[ "$RC" = 124 ] && echo "SELF-TIMER: r2acc stopped at 90 % of the wall; result.json incomplete -> no certified rule (m* FAIL)"
[ "$RC" = 0 ] || [ "$RC" = 124 ] || {{ echo "r2acc failed rc=$RC"; exit "$RC"; }}
'''
    return preamble(job, CEIL['sel']) + body + epilogue()


def pol(bank, config):
    assert bank.isalnum() and config.isalnum()
    job = f'POL-{bank}'
    body = f'''echo "STEP fpolicy configs/{config}.json $(date -u +%FT%TZ)"
"$PY" fpolicy.py --config configs/{config}.json --out "$TASK_ROOT/output/fpolicy_{config}" --t0-epoch "$T0"
'''
    return preamble(job, CEIL['pol']) + body + epilogue()


def smb_fragment(with_smoke_bank=True):
    sb = '''# (1) smoke bank trained earlier in this job -> runtime-registered bank dir (BANK.json, runtime_hashed)
"$PY" register_arm.py --runtime "${{SMOKE_BANK_SRC:-$TASK_ROOT/output/train2d_s1}}" "$TASK_ROOT/output/smokebank" smkb
''' if with_smoke_bank else ''
    sb2 = '''# (3) policy round 1 on FC1 dev j = 0, 1 with the smoke bank
"$PY" fpolicy.py --config configs/smkpolSB.json --out "$TASK_ROOT/output/fpolicy_smkpolSB" --t0-epoch "$T0" \\
  || echo "smoke-bank policy round failed (recorded; K1 part stands)"
''' if with_smoke_bank else ''
    return f'''# ---- SM-b ROM part (imp-fronts cluster/rom/render.py smb-rom); needs T0, PY, TASK_ROOT from the SM-b preamble and
# ---- the cwd experiments/imp-fronts/rom with rom/env.sh sourced.
{sb}# (2) policy round 1 on FC1 dev j = 0, 1 with K1, forced uncertified start fib4181; C-B1 hash-mismatch rejection;
#     C-B6 perturbation fixture on FC1|0 step 10 (Amendment 3)
"$PY" fpolicy.py --config configs/smkpolK1.json --out "$TASK_ROOT/output/fpolicy_smkpolK1" --t0-epoch "$T0"
"$PY" - <<'PYEOF'
import json, os
r = json.load(open(os.environ['TASK_ROOT'] + '/output/fpolicy_smkpolK1/result.json'))
c = r['controls']
print('C-B1 fired', c['cb1']['fired'], '| C-B6', c.get('CB6', {{}}).get('verdict'), '| backend', r['backend'])
PYEOF
{sb2}'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=('sel', 'pol', 'smb-rom', 'smb-rom-standalone'))
    ap.add_argument('args', nargs='*')
    ap.add_argument('--out')
    a = ap.parse_args()
    if a.kind == 'sel':
        txt = sel(*a.args)
    elif a.kind == 'pol':
        txt = pol(*a.args)
    elif a.kind == 'smb-rom':
        txt = smb_fragment(True)
    else:
        txt = preamble('SM-b-rom', CEIL['smb']) + smb_fragment(False) + epilogue()
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt)
        print(a.out)
    else:
        sys.stdout.write(txt)


if __name__ == '__main__':
    main()
