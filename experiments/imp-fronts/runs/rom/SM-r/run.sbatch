#!/bin/bash
#SBATCH --job-name=imf_SM-r
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:24:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/imp-fronts/SM-r/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/imp-fronts/SM-r/logs/%j.err
# imp-fronts SM-r: ceiling 0.4 GPU-h on 1 GPU = Slurm wall limit (DESIGN Amendment 4 B8). gpu partition only.
set -euo pipefail
T0=$(date +%s)
export TASK_ROOT=/cluster/tufts/paralab/tawal01/imp-fronts/SM-r
WALL_S=1440
STOP_S=$(( WALL_S * 9 / 10 ))                    # self-timer: no new work after 90 % of the wall
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp" PYTHONDONTWRITEBYTECODE=1
export CUDA_CACHE_PATH="$TASK_ROOT/cache/cuda" JAX_COMPILATION_CACHE_DIR="$TASK_ROOT/cache/jax"   # every write in the job dir
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$JAX_COMPILATION_CACHE_DIR" "$TASK_ROOT/output" "$TASK_ROOT/logs"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) job=$SLURM_JOB_ID source_commit=$SOURCE_COMMIT precision=$JAX_DEFAULT_MATMUL_PRECISION"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b == 'gpu' else 42)"
source experiments/imp-fronts/rom/env.sh
cd experiments/imp-fronts/rom
left() { echo $(( STOP_S - ($(date +%s) - T0) )); }
FAIL=0
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
