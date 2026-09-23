#!/bin/bash
# Stage one isolated job directory and submit it (heat-bank-knob; adapted from heat3d-bank cluster/submit.sh @5f1b048d).
#   cluster/submit.sh <job> <gpu: a100|h100|h200|l40s> <HH:MM:SS> <mem> <panel_config.json>
# Must be run from a CLEAN committed tree (the commit is the recorded source). The NumPy audit runs in-job, after the
# panel, on the saved fields (so they can be deleted remotely after the pull).
set -euo pipefail
JOB=$1; GPU=$2; WALL=$3; MEM=$4; CFG=$5
[[ "$JOB" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{1,40}$ ]] || { echo "bad job name"; exit 1; }
LANE=$(cd "$(dirname "$0")/.." && pwd); NS=/cluster/tufts/paralab/tawal01/hbank_20260923; REMOTE=$NS/$JOB
RUN=$LANE/runs/$JOB; [ -e "$RUN" ] && { echo "run dir exists: $RUN"; exit 1; }
git -C "$LANE" diff --quiet HEAD -- . || { echo "uncommitted lane changes"; exit 1; }
[ -z "$(git -C "$LANE" status --porcelain -- . | grep -v '^?? runs/' || true)" ] || { echo "untracked/uncommitted lane files"; exit 1; }
mkdir -p "$RUN/stage/code/configs"
cp "$LANE"/{core.py,hbk_core.py,hbk_run.py,hbk_audit_np.py,prep_2d.npz,prep_3d.npz} "$RUN/stage/code/"
cp "$LANE/configs/$CFG" "$RUN/stage/code/configs/"
cp -r "$LANE/inputs" "$RUN/stage/code/inputs"
git -C "$LANE" rev-parse HEAD > "$RUN/stage/COMMIT.txt"
cat > "$RUN/stage/run.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=hbank_$JOB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:$GPU:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=$MEM
#SBATCH --time=$WALL
#SBATCH --output=$REMOTE/job.out
#SBATCH --error=$REMOTE/job.err
set -euo pipefail
ROOT=$REMOTE
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 TMPDIR="\$ROOT/tmp" XDG_CACHE_HOME="\$ROOT/cache"
mkdir -p "\$TMPDIR" "\$XDG_CACHE_HOME" "\$ROOT/out"
cd "\$ROOT"; sha256sum -c SOURCE.sha256 > /dev/null
export SOURCE_COMMIT=\$(cat COMMIT.txt); echo "source_commit=\$SOURCE_COMMIT job_id=\$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
df -h /cluster/tufts/paralab | tail -1
"\$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd code
STATUS=0
set +e
"\$PY" hbk_run.py --config configs/$CFG --out "\$ROOT/out/panel" || STATUS=1
"\$PY" hbk_audit_np.py "\$ROOT/out/panel" > "\$ROOT/out/audit_np.json" 2> "\$ROOT/out/audit_np.err" || STATUS=2
set -e
cd "\$ROOT"
echo "run_exit=\$STATUS" > out/EXIT_STATUS.txt; cp job.out out/job.out.copy 2>/dev/null || true
find out -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo "run_exit=\$STATUS"; exit \$STATUS
EOF
(cd "$RUN/stage" && find code COMMIT.txt run.sbatch -type f -print0 | sort -z | xargs -0 sha256sum > SOURCE.sha256)
ssh tufts-login "squeue -u \$USER -o '%i %j %T %M %R'" | tee "$RUN/QUEUE-BEFORE.log"
if grep -q " hbank_$JOB " "$RUN/QUEUE-BEFORE.log"; then echo "duplicate job name queued"; exit 1; fi
[ "$(grep -c ' hbank_' "$RUN/QUEUE-BEFORE.log" || true)" -ge 2 ] && { echo "lane already has 2 jobs queued/running"; exit 1; }
ssh tufts-login "mkdir -p '$NS' && mkdir '$REMOTE'" || { echo "remote dir exists or mkdir failed (atomic)"; exit 1; }
rsync -a "$RUN/stage/" "tufts-login:$REMOTE/"
ssh tufts-login "cd $REMOTE && sha256sum -c SOURCE.sha256 > /dev/null && sbatch run.sbatch" | tee "$RUN/SUBMIT.log"
sleep 3; ssh tufts-login "squeue -u \$USER -o '%i %j %T %M %R'" | tee "$RUN/QUEUE-AFTER.log"
[ "$(grep -c " hbank_$JOB " "$RUN/QUEUE-AFTER.log")" -le 1 ] || { echo "DUPLICATE SUBMISSION"; exit 1; }
