#!/bin/bash
# Stage ONE isolated job directory and submit it (pattern: hires-heat cluster/submit.sh).
#   cluster/submit.sh <job> <gpu: a100|h100|h200> <HH:MM:SS> <mem> '<command run from code/heat-compare-hires>'
# Clean committed tree only (the commit is the recorded source). Never reuses a remote dir.
# Optional env CONSTRAINT=<slurm feature> (e.g. a100-80G). Enforces: gpu partition; at most 2 running/pending hcmp_* jobs of this lane; squeue before and after.
set -euo pipefail
JOB=$1; GPU=$2; WALL=$3; MEM=$4; CMD=$5
LANE=$(cd "$(dirname "$0")/.." && pwd); NS=/cluster/tufts/paralab/tawal01/hcmp_20260923; REMOTE=$NS/$JOB
RUN=$LANE/runs/$JOB; [ -e "$RUN" ] && { echo "run dir exists: $RUN"; exit 1; }
git -C "$LANE" diff --quiet HEAD -- . ../hires-heat || { echo "uncommitted lane changes"; exit 1; }
[ -z "$(git -C "$LANE" status --porcelain -- . ../hires-heat | grep -v '^??' || true)" ] || { echo "uncommitted"; exit 1; }
mkdir -p "$RUN/stage/code/heat-compare-hires" "$RUN/stage/code/hires-heat"
(cd "$LANE" && git ls-files -z . | grep -zv '^runs/' | xargs -0 -I{} cp --parents {} "$RUN/stage/code/heat-compare-hires/")
HH=$LANE/../hires-heat
cp "$HH"/{core.py,run.py,audit.py} "$RUN/stage/code/hires-heat/"
mkdir -p "$RUN/stage/code/hires-heat/inputs/wide2d"
cp "$HH"/inputs/wide2d/{bank.pkl,head_K8.pkl,training.json,SHA256SUMS} "$RUN/stage/code/hires-heat/inputs/wide2d/"
(cd "$RUN/stage/code/hires-heat" && sha256sum -c --ignore-missing inputs/wide2d/SHA256SUMS)
git -C "$LANE" rev-parse HEAD > "$RUN/stage/COMMIT.txt"
cat > "$RUN/stage/run.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=hcmp_$JOB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:$GPU:1
${CONSTRAINT:+#SBATCH --constraint=$CONSTRAINT}
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
cd code/heat-compare-hires
set +e
PY=\$PY OUT=\$ROOT/out bash -c '$CMD'; STATUS=\$?
set -e
cd "\$ROOT"; find out -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo "run_exit=\$STATUS"; exit \$STATUS
EOF
(cd "$RUN/stage" && find code COMMIT.txt run.sbatch -type f -print0 | sort -z | xargs -0 sha256sum > SOURCE.sha256)
ssh tufts-login "squeue -u \$USER -o '%i %j %T %M %R'" | tee "$RUN/QUEUE-BEFORE.log"
if grep -q " hcmp_$JOB " "$RUN/QUEUE-BEFORE.log"; then echo "duplicate job name queued"; exit 1; fi
[ "$(grep -c ' hcmp_' "$RUN/QUEUE-BEFORE.log" || true)" -ge 2 ] && { echo "lane already has 2 jobs queued/running"; exit 1; }
ssh tufts-login "test ! -e $REMOTE && mkdir -p $REMOTE"
rsync -a "$RUN/stage/" "tufts-login:$REMOTE/"
ssh tufts-login "cd $REMOTE && sha256sum -c SOURCE.sha256 > /dev/null && sbatch run.sbatch" | tee "$RUN/SUBMIT.log"
sleep 3; ssh tufts-login "squeue -u \$USER -o '%i %j %T %M %R'" | tee "$RUN/QUEUE-AFTER.log"
[ "$(grep -c " hcmp_$JOB " "$RUN/QUEUE-AFTER.log")" -le 1 ] || { echo "DUPLICATE SUBMISSION"; exit 1; }
rm -rf "$RUN/stage/code"   # the staged copy is reproducible from COMMIT.txt; keep SOURCE.sha256 + run.sbatch
