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
    python render.py sme              SM-e (E10, 0.3 h H200)
    python render.py pole10s          POL-e10s (E10, 0.8 h H200)
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
CEIL = {'sel': SCH.CEIL['select_b'], 'pol': SCH.CEIL['policy_K1'], 'smr': SCH.SMOKE['SM-r'],
        'pole10s': SCH.E10_CEIL['policy_e10s'], 'sme': SCH.E10_SMOKE['SM-e']}


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
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp" PYTHONDONTWRITEBYTECODE=1
export CUDA_CACHE_PATH="$TASK_ROOT/cache/cuda" JAX_COMPILATION_CACHE_DIR="$TASK_ROOT/cache/jax"   # every write in the job dir
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$JAX_COMPILATION_CACHE_DIR" "$TASK_ROOT/output" "$TASK_ROOT/logs"
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
timeout --signal=TERM "$L" "$PY" archrun.py r2acc.py --config configs/e2s{label}.json --out "$TASK_ROOT/output/r2acc_e2s{label}"
RC=$?
set -e
[ "$RC" = 124 ] && echo "SELF-TIMER: r2acc stopped at 90 % of the wall; result.json incomplete -> no certified rule (m* FAIL)"
[ "$RC" = 0 ] || [ "$RC" = 124 ] || {{ echo "r2acc failed rc=$RC"; exit "$RC"; }}
'''
    return preamble(job, CEIL['sel']) + body + epilogue()


def pol(bank, config, ceil_key='pol'):
    assert bank.isalnum() and config.isalnum()
    job = f'POL-{bank}'
    body = f'''export IMPF_TESTONCE_DIR={NS}/TESTONCE        # shared, atomic test-once claims (rom/testonce.py)
mkdir -p "$IMPF_TESTONCE_DIR"
echo "STEP fpolicy configs/{config}.json $(date -u +%FT%TZ)"
# outer timeout at 95 % of the wall (fpolicy's own self-timer stops new solves at 90 %); a kill -> fcollect.py locally
timeout --signal=TERM --kill-after=30 $(( WALL_S * 19 / 20 - ($(date +%s) - T0) )) \
  "$PY" fpolicy.py --config configs/{config}.json --out "$TASK_ROOT/output/fpolicy_{config}" --t0-epoch "$T0"
'''
    return preamble(job, CEIL[ceil_key], gpu='h200') + body + epilogue()


def pole10s():
    """POL-e10s (E10 bucket, 0.8 h H200): fpolicy on configs/pole10s.json (E10-s, re-used test cases, test-once claim
    keyed separately and marked re-use)."""
    return pol('e10s', 'pole10s', 'pole10s')


def sme():
    """SM-e (E10 bucket, 0.3 h H200): the E10-s policy on FC1 dev j = 0 (must-fail control) and j = 1; ALL-DONE only if the
    step succeeded and evidence.check_e10_smoke passes (control fired, all four factors, consistency, replay)."""
    body = r'''FAIL=0
L=$(left)
echo "STEP fpolicy smke10s timeout=${L}s $(date -u +%FT%TZ)"
set +e
timeout --signal=TERM --kill-after=30 "$L" "$PY" fpolicy.py --config configs/smke10s.json --out "$TASK_ROOT/output/fpolicy_smke10s" --t0-epoch "$T0"
rc=$?
set -e
[ "$rc" = 0 ] || { echo "STEP smke10s FAILED rc=$rc"; FAIL=1; }
set +e
"$PY" - "$TASK_ROOT/output/fpolicy_smke10s/result.json" <<'PYEOF'
import json, sys
import evidence as EV
c = EV.check_e10_smoke(json.load(open(sys.argv[1])))
print('SM-e checks', c)
sys.exit(0 if all(c.values()) else 4)
PYEOF
rc=$?
set -e
[ "$rc" = 0 ] || { echo "SM-e CHECKS FAILED rc=$rc (E10-s blocked)"; FAIL=1; }
cd "$TASK_ROOT"; find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
if [ "$FAIL" = 0 ]; then echo ALL-DONE; else echo "SM-e FAILED"; exit 1; fi
'''
    return preamble('SM-e', CEIL['sme'], gpu='h200', mem='160G') + body


def smr():
    """SM-r: every step must succeed for ALL-DONE (code audit SM-r-2): the two policy steps run under an outer timeout and a
    failure exits non-zero; the references may stop at the deadline (status INCOMPLETE, recorded in timing.json, which
    makes resources.py fall back to the flagged historical bound), any other reference failure exits non-zero; the
    evidence must build and pass; OUTPUTS.sha256 is written in every case so the outputs can be collected."""
    body = r"""FAIL=0
finish() {                                  # always hash the outputs; ALL-DONE only if nothing failed
  cd "$TASK_ROOT"; find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
  if [ "$FAIL" = 0 ]; then echo ALL-DONE; else echo "SM-r FAILED ($FAIL)"; exit 1; fi
}
step_policy() {                             # $1 = config stem; outer timeout = what is left of the 90 % cutoff
  local L; L=$(left)
  if [ "$L" -le 30 ]; then echo "DEADLINE before $1"; FAIL="$FAIL $1:deadline"; return; fi
  echo "STEP fpolicy $1 timeout=${L}s $(date -u +%FT%TZ)"
  set +e
  timeout --signal=TERM --kill-after=30 "$L" "$PY" fpolicy.py --config "configs/$1.json" --out "$TASK_ROOT/output/fpolicy_$1" --t0-epoch "$T0"
  local rc=$?
  set -e
  [ "$rc" = 0 ] || { echo "STEP $1 FAILED rc=$rc"; FAIL="$FAIL $1:rc$rc"; }
}
# (1) K1 policy round 1 on FC1 dev j = 0, 1 + C-B1 + C-B6 (fpolicy stops new solves at 15 % of the wall)
step_policy smkpolK1
# (2) reference-free K1 policy dry run on the 16 fresh dev/val cases (fpolicy stops new solves at 60 % of the wall)
step_policy smkdryK1
# (3) the two dev references with the remaining wall (90 % cutoff); a deadline stop is INCOMPLETE, not a failure
REFS=complete
cd "$TASK_ROOT/experiments/imp-fronts/phaseb/frefs"
for cfg in d_FC1_0.json d_FC2_0.json; do
  L=$(left)
  if [ "$L" -le 60 ]; then echo "DEADLINE: skipping $cfg"; REFS=INCOMPLETE; continue; fi
  stem=$(basename "$cfg" .json)
  echo "STEP refjob $cfg left=${L}s $(date -u +%FT%TZ)"
  set +e
  timeout --signal=TERM --kill-after=30 "$L" "$PY" refjob.py --config "../../cluster/refs/configs/$cfg" --out "$TASK_ROOT/output/refs_$stem"
  rc=$?
  set -e
  if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then echo "STEP $cfg stopped at the deadline (INCOMPLETE)"; REFS=INCOMPLETE
  elif [ "$rc" != 0 ]; then echo "STEP $cfg FAILED rc=$rc"; FAIL="$FAIL $stem:rc$rc"; REFS=FAILED; fi
done
cd "$TASK_ROOT/experiments/imp-fronts/rom"
# (4) SM-r timing.json (per-case reference seconds + status) at output/timing.json: resources.py reads it from the pulled
#     output directory (--smr <dir>, default runs/rom/SM-r/output); then the evidence, which must pass
"$PY" - "$TASK_ROOT/output" "$REFS" <<'PYEOF'
import json, pathlib, sys
out, status = pathlib.Path(sys.argv[1]), sys.argv[2]
refs = []
for p in sorted(out.glob('refs_d_*/**/c*.json')):
    d = json.loads(p.read_text())
    if 'seconds' in d:
        refs.append(dict(case=f"{d.get('cohort')}|{d.get('case')}", seconds=float(d['seconds']), refined=bool(d.get('refined')),
                         gpu=d.get('job', {}).get('gpu'), source=str(p.relative_to(out))))
if status == 'complete' and len(refs) < 2:
    status = 'INCOMPLETE'
(out / 'timing.json').write_text(json.dumps(dict(schema='imp-fronts-timing/2', job='SM-r', references_status=status,
                                                 references=refs), indent=1))
print('SM-r reference timings', status, refs)
PYEOF
set +e
"$PY" evidence.py build "$TASK_ROOT/output" --out "$TASK_ROOT/output/SMOKE-EVIDENCE.json"
rc=$?
set -e
[ "$rc" = 0 ] || { echo "EVIDENCE FAILED rc=$rc (Phase B blocked)"; FAIL="$FAIL evidence:rc$rc"; }
finish
"""
    return preamble('SM-r', CEIL['smr'], gpu='h200', mem='160G') + body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=('sel', 'pol', 'smr', 'sme', 'pole10s'))
    ap.add_argument('args', nargs='*')
    ap.add_argument('--out')
    a = ap.parse_args()
    txt = {'sel': sel, 'pol': pol, 'smr': smr, 'sme': sme, 'pole10s': pole10s}[a.kind](*a.args)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt)
        print(a.out)
    else:
        sys.stdout.write(txt)


if __name__ == '__main__':
    main()
