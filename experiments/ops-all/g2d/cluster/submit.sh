#!/bin/bash
# g2d: stage ONE job directory and submit it (one directory per job, never reused).
#   cluster/submit.sh <job> <gpu: a100|a100-80G|h100|h200> <HH:MM:SS> <mem> <tree: burgers|heat> <body.sh>
# The body runs from $ROOT/src with $PY, $ROOT, $OUT set and PYTHONPATH covering the tree's experiment dirs.
# Enforces: gpu partition only; group cap (g2d_* running+pending < 3, else refuses); no reuse of a remote dir;
# squeue before AND after; GPU preflight (exit 42); venv; x64 + highest precision; all output under the namespace.
set -euo pipefail
rssh() { local i rc; for i in 1 2 3 4 5 6; do ssh -o ConnectTimeout=20 "$@"; rc=$?; [ $rc -ne 255 ] && return $rc; sleep 10; done; return 255; }
JOB=$1; GPU=$2; WALL=$3; MEM=$4; TREE=$5; BODY=$6
[[ "$JOB" =~ ^[a-zA-Z0-9]+$ ]]
G2D=$(cd "$(dirname "$0")/.." && pwd)
NS=/cluster/tufts/paralab/tawal01/opsall_20260924/g2d; REMOTE=$NS/$JOB
RUN=$G2D/runs/$JOB; [ -e "$RUN" ] && { echo "run dir exists: $RUN"; exit 1; }
case $GPU in a100) GRES=gpu:a100:1; CON=;; a100-80G) GRES=gpu:a100:1; CON=a100-80G;; h100) GRES=gpu:h100:1; CON=;;
  h200) GRES=gpu:h200:1; CON=;; *) echo bad gpu; exit 1;; esac
mkdir -p "$RUN/stage/src"
cp -a "$G2D/src/$TREE/." "$RUN/stage/src/"
cp -a "$G2D/scripts" "$RUN/stage/scripts"
cp -a "$G2D/configs" "$RUN/stage/configs"
mkdir -p "$RUN/stage/speed2048" && cp -a "$G2D/speed2048/"*.py "$G2D/speed2048/"*.json "$RUN/stage/speed2048/"
mkdir -p "$RUN/stage/fomdt" && cp -a "$G2D/fomdt/scripts" "$RUN/stage/fomdt/"
mkdir -p "$RUN/stage/scaling256" && cp -a "$G2D/scaling256/scripts" "$G2D/scaling256/configs" "$RUN/stage/scaling256/"
cp "$BODY" "$RUN/stage/body.sh"
[ -d "$G2D/ckpt/$JOB" ] && cp -al "$G2D/ckpt/$JOB" "$RUN/stage/ckpt"
if [ "$TREE" = burgers ]; then
  PP='experiments/mr-burgers2d experiments/separable-decoder experiments/head-ablation experiments/cheap-corrections experiments/b-ladder-top experiments/b-panel/speed experiments/hires-burgers experiments/burgers-repanel experiments/burgers-bank-knob experiments/burgers2d-speed experiments/burgers-compare-hires experiments/burgers-compare-hires/ops'
else
  PP='experiments/hires-heat experiments/heat-compare-hires experiments/heat-compare-hires/ops experiments/heat-bank-knob'
fi
PYPATH=$(for d in $PP; do printf '%s:' "$REMOTE/src/$d"; done)$REMOTE/scripts
cat > "$RUN/stage/run.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=g2d_$JOB
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=$GRES
${CON:+#SBATCH --constraint=$CON}
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=$MEM
#SBATCH --time=$WALL
#SBATCH --output=$REMOTE/job.out
#SBATCH --error=$REMOTE/job.err
set -euo pipefail
ROOT=$REMOTE
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=\${XLA_FRAC:-0.90}
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 TMPDIR="\$ROOT/tmp" XDG_CACHE_HOME="\$ROOT/cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "\$TMPDIR" "\$XDG_CACHE_HOME" "\$ROOT/out"
cd "\$ROOT"; sha256sum -c MANIFEST.sha256 --quiet
echo "host=\$(hostname) job_id=\$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"\$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
"\$PY" -c "import torch,sys; ok=torch.cuda.is_available(); print('torch_cuda', ok, torch.cuda.get_device_name() if ok else None, flush=True); sys.exit(0 if ok else 42)"
export PYTHONPATH="$PYPATH"
export OUT=\$ROOT/out
cd "\$ROOT/src"
set +e
PY=\$PY OUT=\$OUT ROOT=\$ROOT bash "\$ROOT/body.sh"; STATUS=\$?
set -e
cd "\$ROOT"; find out -type f \( -name '*.json' -o -name '*.pt' -o -name '*.log' \) -print0 | sort -z | xargs -0 -r sha256sum > OUTPUTS.sha256
echo "run_exit=\$STATUS"; exit \$STATUS
EOF
(cd "$RUN/stage" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256)
Q=$(rssh tufts-login "squeue -u \$USER -h -o '%i %j %T %M %R'")
echo "$Q" > "$RUN/QUEUE-BEFORE.log"; echo "--- squeue before"; echo "$Q" | grep g2d_ || true
if echo "$Q" | awk '{print $2}' | grep -qx "g2d_$JOB"; then echo "g2d_$JOB already queued: refusing"; exit 3; fi
N=$(echo "$Q" | awk '$2 ~ /^g2d_/ && ($3=="RUNNING" || $3=="PENDING")' | grep -c . || true)
[ "$N" -lt 10 ] || { echo "group cap: $N g2d jobs running/pending"; rm -rf "$RUN"; exit 4; }
rssh tufts-login "test ! -e $REMOTE && mkdir -p $REMOTE"
for i in 1 2 3 4 5; do rsync -a "$RUN/stage/" "tufts-login:$REMOTE/" && break; sleep 10; done
: rsync -a "$RUN/stage/" "tufts-login:$REMOTE/"
ssh -o ConnectTimeout=30 tufts-login "cd $REMOTE && sha256sum -c MANIFEST.sha256 --quiet && sbatch run.sbatch" | tee "$RUN/SUBMIT.log"
sleep 3
Q=$(rssh tufts-login "squeue -u \$USER -h -o '%i %j %T %M %R'"); echo "$Q" > "$RUN/QUEUE-AFTER.log"
echo "--- squeue after"; echo "$Q" | grep g2d_ || true
[ "$(echo "$Q" | awk '{print $2}' | grep -cx "g2d_$JOB")" -le 1 ] || { echo "DUPLICATE SUBMISSION"; exit 1; }
rm -rf "$RUN/stage/src" "$RUN/stage/ckpt"
