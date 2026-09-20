#!/bin/bash
# Stage one isolated job directory and submit it. Usage:
#   cluster/submit.sh <job> <config.json> <gpu: a100|h100|h200|l40s> <HH:MM:SS> <mem> [train]
# Must be run from a CLEAN committed tree (the commit is the recorded source).
set -euo pipefail
JOB=$1; CONFIG=$2; GPU=$3; WALL=$4; MEM=$5; TRAIN=${6:-}
LANE=$(cd "$(dirname "$0")/.." && pwd); NS=/cluster/tufts/paralab/tawal01/hires_h_20260920; REMOTE=$NS/$JOB
RUN=$LANE/runs/$JOB; [ -e "$RUN" ] && { echo "run dir exists: $RUN"; exit 1; }
git -C "$LANE" diff --quiet HEAD -- . || { echo "uncommitted lane changes"; exit 1; }
mkdir -p "$RUN/stage/code/hires-heat/configs" "$RUN/stage/code/separable-decoder"
cp "$LANE"/{core.py,run.py,train.py} "$RUN/stage/code/hires-heat/"
cp "$LANE/configs/$CONFIG" "$RUN/stage/code/hires-heat/configs/"
cp -r "$LANE/inputs" "$RUN/stage/code/hires-heat/inputs"; rm -rf "$RUN/stage/code/hires-heat/inputs"/smoke*
cp "$LANE/../separable-decoder/sep_common.py" "$RUN/stage/code/separable-decoder/"
git -C "$LANE" rev-parse HEAD > "$RUN/stage/COMMIT.txt"
TRAINCMD=""; [ -n "$TRAIN" ] && TRAINCMD="\"\$PY\" train.py --config configs/$CONFIG --out inputs/$TRAIN"
cat > "$RUN/stage/run.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=hires_h_$JOB
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
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 TMPDIR="\$ROOT/tmp" XDG_CACHE_HOME="\$ROOT/cache"
mkdir -p "\$TMPDIR" "\$XDG_CACHE_HOME"
cd "\$ROOT"; sha256sum -c SOURCE.sha256 > /dev/null
export SOURCE_COMMIT=\$(cat COMMIT.txt); echo "source_commit=\$SOURCE_COMMIT job_id=\$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
df -h /cluster/tufts/paralab | tail -1
"\$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd code/hires-heat
set +e
$TRAINCMD
"\$PY" run.py --config configs/$CONFIG --out "\$ROOT/out"; STATUS=\$?
set -e
cd "\$ROOT"; [ -n "$TRAIN" ] && cp -r code/hires-heat/inputs/$TRAIN out/trained_$TRAIN
find out -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo "run_exit=\$STATUS"; exit \$STATUS
EOF
(cd "$RUN/stage" && find code COMMIT.txt run.sbatch -type f -print0 | sort -z | xargs -0 sha256sum > SOURCE.sha256)
ssh tufts-login "squeue -u \$USER -o '%i %j %T %M %R'" | tee "$RUN/QUEUE-BEFORE.log"
if grep -q "hires_h_$JOB\$\| hires_h_$JOB " "$RUN/QUEUE-BEFORE.log"; then echo "duplicate job name queued"; exit 1; fi
[ "$(grep -c RUNNING "$RUN/QUEUE-BEFORE.log" || true)" -ge 6 ] && { echo "account already has 6 running"; exit 1; }
ssh tufts-login "test ! -e $REMOTE && mkdir -p $NS"
rsync -a "$RUN/stage/" "tufts-login:$REMOTE/"
ssh tufts-login "cd $REMOTE && sha256sum -c SOURCE.sha256 > /dev/null && sbatch run.sbatch" | tee "$RUN/SUBMIT.log"
sleep 3; ssh tufts-login "squeue -u \$USER -o '%i %j %T %M %R'" | tee "$RUN/QUEUE-AFTER.log"
[ "$(grep -c "hires_h_$JOB" "$RUN/QUEUE-AFTER.log")" -le 1 ] || { echo "DUPLICATE SUBMISSION"; exit 1; }
