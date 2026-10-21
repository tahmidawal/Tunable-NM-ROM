"""imp-fronts ROM sbatch templates (DESIGN B4, B5, B8 / B9 as amended by Amendments 3-12). Prints (or writes with --out) a
complete sbatch script; never submits. The staged tree under the job directory must mirror the repository layout
(experiments/imp-fronts/rom/..., experiments/imp-fronts/phaseb/frefs + cluster/refs/configs for SM-r, plus
experiments/mr-burgers2d/{engines,iterative_paths}.py and experiments/separable-decoder/sep_common.py, which rom/env.sh
puts on PYTHONPATH) and carry MANIFEST.sha256 + COMMIT.txt. Walls = ceilings (rom/schedule.py) / GPUs.

Kinds
  sel <label>          SEL-<label> (A100, as the e2s1 history): r2acc on configs/e2s<label>.json (R384 group only,
                       mkconfigs_sel.py --r384); wall 0.4 h; r2acc has no self-timer, so it runs under `timeout` at 90 %
                       of the wall. Locally: select2d_r384.py (persists an explicit incomplete outcome if needed).
  pol <bank> <config>  POL-<bank> (H200, the GPU type of the SM-r dry run that projects it): fpolicy, wall 0.5 h,
                       self-timer at 90 % from job start (--t0-epoch); a killed job is finalised by rom/fcollect.py.
  smr                  SM-r (H200, 0.4 h; Amendment 11): (1) K1 policy round 1 on FC1 dev j = 0, 1 with C-B1 and C-B6;
                       (2) the 16-case reference-free K1 policy dry run; (3) the two dev references FC1|0, FC2|0 with the
                       remaining wall (stopped at 90 %); (4) SM-r timing.json (reference seconds) and the in-job
                       evidence preview. Locally after the pull: rom/evidence.py build runs/SM-r/output, then
                       resources.py --rom .../fpolicy_smkpolK1/result.json --dry .../fpolicy_smkdryK1/result.json.

    python render.py sel b [--out runs/SEL-b/run.sbatch]
    python render.py pol K1 pol1
    python render.py smr
"""
import argparse
import math
import sys
from pathlib import Path

NS = '/cluster/tufts/paralab/tawal01/imp-fronts'
VENV = '/cluster/tufts/paralab/tawal01/ae-research/venv'
GRES = {'a100': 'a100', 'h100': 'h100', 'h200': 'h200', 'l40s': 'l40s'}
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'rom'))
import schedule as SCH  # noqa: E402
CEIL = {'sel': SCH.CEIL['select_b'], 'pol': SCH.CEIL['policy_K1'], 'smr': SCH.SMOKE['SM-r']}


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
    return preamble(job, CEIL['pol'], gpu='h200') + body + epilogue()


def smr():
    body = r"""# (1) K1 policy round 1 on FC1 dev j = 0, 1 + C-B1 + C-B6 (stops new work at 15 % of the wall)
"$PY" fpolicy.py --config configs/smkpolK1.json --out "$TASK_ROOT/output/fpolicy_smkpolK1" --t0-epoch "$T0" \
  || echo "STEP smkpolK1 failed rc=$? (evidence will fail: Phase B blocked)"
# (2) reference-free K1 policy dry run on the 16 fresh dev/val cases (stops new work at 60 % of the wall)
"$PY" fpolicy.py --config configs/smkdryK1.json --out "$TASK_ROOT/output/fpolicy_smkdryK1" --t0-epoch "$T0" \
  || echo "STEP smkdryK1 failed rc=$? (evidence will fail: policy uncertified)"
# (3) the two dev references with the remaining wall (90 % cutoff)
cd "$TASK_ROOT/experiments/imp-fronts/phaseb/frefs"
for cfg in d_FC1_0.json d_FC2_0.json; do
  L=$(left)
  if [ "$L" -le 60 ]; then echo "DEADLINE: skipping $cfg"; continue; fi
  stem=$(basename "$cfg" .json)
  echo "STEP refjob $cfg left=${L}s $(date -u +%FT%TZ)"
  timeout --signal=TERM --kill-after=30 "$L" "$PY" refjob.py --config "../../cluster/refs/configs/$cfg" \
    --out "$TASK_ROOT/output/refs_$stem" || echo "STEP $cfg rc=$? (124 = deadline)"
done
cd "$TASK_ROOT/experiments/imp-fronts/rom"
# (4) SM-r timing.json (per-case reference seconds) and an evidence preview (the binding record is rebuilt locally)
"$PY" - "$TASK_ROOT/output" <<'PYEOF'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
refs = []
for p in sorted(out.glob('refs_d_*/**/c*.json')):
    d = json.loads(p.read_text())
    if 'seconds' in d:
        refs.append(dict(case=f"{d.get('cohort')}|{d.get('case')}", seconds=float(d['seconds']), refined=bool(d.get('refined')),
                         gpu=d.get('job', {}).get('gpu'), source=str(p.relative_to(out))))
(out / 'timing.json').write_text(json.dumps(dict(schema='imp-fronts-timing/2', job='SM-r', references=refs), indent=1))
print('SM-r reference timings', refs)
PYEOF
"$PY" evidence.py build "$TASK_ROOT/output" --out "$TASK_ROOT/output/SMOKE-EVIDENCE.preview.json" || true
"""
    return preamble('SM-r', CEIL['smr'], gpu='h200', mem='160G') + body + epilogue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=('sel', 'pol', 'smr'))
    ap.add_argument('args', nargs='*')
    ap.add_argument('--out')
    a = ap.parse_args()
    txt = {'sel': sel, 'pol': pol, 'smr': smr}[a.kind](*a.args)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt)
        print(a.out)
    else:
        sys.stdout.write(txt)


if __name__ == '__main__':
    main()
