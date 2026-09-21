#!/bin/bash
# Stage one isolated job directory and submit it (heat3d-bank; adapted from hires-heat cluster/submit.sh @4fb12a6d). Usage:
#   cluster/submit.sh <job> <gpu: a100|h100|h200|l40s> <HH:MM:SS> <mem> <train_config.json|-> <train_name|-> <panel_config.json> [panel_config.json ...]
# If a training config is given, train_vp.py writes inputs/<train_name> first (training + validation draws only).
# Each panel writes out/<panel stem>/. Must be run from a CLEAN committed tree (the commit is the recorded source).
set -euo pipefail
JOB=$1; GPU=$2; WALL=$3; MEM=$4; TCFG=$5; TNAME=$6; shift 6; PANELS=("$@")
LANE=$(cd "$(dirname "$0")/.." && pwd); NS=/cluster/tufts/paralab/tawal01/h3dbank_20260921; REMOTE=$NS/$JOB
RUN=$LANE/runs/$JOB; [ -e "$RUN" ] && { echo "run dir exists: $RUN"; exit 1; }
git -C "$LANE" diff --quiet HEAD -- . || { echo "uncommitted lane changes"; exit 1; }
[ -z "$(git -C "$LANE" status --porcelain -- .)" ] || { echo "untracked/uncommitted lane files"; exit 1; }
mkdir -p "$RUN/stage/code/heat3d-bank/configs"
cp "$LANE"/{core.py,run.py,train.py,train_vp.py} "$RUN/stage/code/heat3d-bank/"
for c in "${PANELS[@]}"; do cp "$LANE/configs/$c" "$RUN/stage/code/heat3d-bank/configs/"; done
[ "$TCFG" != "-" ] && cp "$LANE/configs/$TCFG" "$RUN/stage/code/heat3d-bank/configs/"
cp -r "$LANE/inputs" "$RUN/stage/code/heat3d-bank/inputs"
git -C "$LANE" rev-parse HEAD > "$RUN/stage/COMMIT.txt"
TRAINCMD="true"; [ "$TCFG" != "-" ] && TRAINCMD="\"\$PY\" train_vp.py --config configs/$TCFG --out inputs/$TNAME"
PANELCMD=""
for c in "${PANELS[@]}"; do s=${c%.json}; PANELCMD+="\"\$PY\" run.py --config configs/$c --out \"\$ROOT/out/$s\" || STATUS=1"$'\n'; done
cat > "$RUN/stage/run.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=h3db_$JOB
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
cd code/heat3d-bank
STATUS=0
set +e
$TRAINCMD || STATUS=1
if [ \$STATUS -eq 0 ]; then
$PANELCMD
fi
set -e
cd "\$ROOT"; [ "$TNAME" != "-" ] && [ -d code/heat3d-bank/inputs/$TNAME ] && cp -r code/heat3d-bank/inputs/$TNAME out/trained_$TNAME
find out -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo "run_exit=\$STATUS"; exit \$STATUS
EOF
(cd "$RUN/stage" && find code COMMIT.txt run.sbatch -type f -print0 | sort -z | xargs -0 sha256sum > SOURCE.sha256)
ssh tufts-login "squeue -u \$USER -o '%i %j %T %M %R'" | tee "$RUN/QUEUE-BEFORE.log"
if grep -q " h3db_$JOB " "$RUN/QUEUE-BEFORE.log"; then echo "duplicate job name queued"; exit 1; fi
[ "$(grep -c RUNNING "$RUN/QUEUE-BEFORE.log" || true)" -ge 6 ] && { echo "account already has 6 running"; exit 1; }
ssh tufts-login "test ! -e $REMOTE && mkdir -p $NS"
rsync -a "$RUN/stage/" "tufts-login:$REMOTE/"
ssh tufts-login "cd $REMOTE && sha256sum -c SOURCE.sha256 > /dev/null && sbatch run.sbatch" | tee "$RUN/SUBMIT.log"
sleep 3; ssh tufts-login "squeue -u \$USER -o '%i %j %T %M %R'" | tee "$RUN/QUEUE-AFTER.log"
[ "$(grep -c " h3db_$JOB " "$RUN/QUEUE-AFTER.log")" -le 1 ] || { echo "DUPLICATE SUBMISSION"; exit 1; }
