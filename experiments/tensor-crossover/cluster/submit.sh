#!/bin/bash
# tensor-crossover: stage ONE job from the COMMITTED lane tree and submit it (PROTOCOL §5, §7).
#   bash cluster/submit.sh <job> <dim:2|3> <config> <mem> <minutes>
# A100 80GB only (one GPU type for the whole figure); gpu partition; venv; f64; highest; jax_backend=gpu preflight (exit 42);
# refuses if tcx_<job> is queued, the remote dir exists, the tree is dirty, or 3 lane jobs are already queued/running.
set -euo pipefail
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=gc.auto GIT_CONFIG_VALUE_0=0
JOB="$1"; DIM="$2"; CFG="$3"; MEM="$4"; MIN="$5"
[[ "$JOB" =~ ^[a-zA-Z0-9]+$ ]]
LANE_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$LANE_DIR"
NS=/cluster/tufts/paralab/tawal01/tcross-20261115
R="$NS/$JOB"
COMMIT=$(git rev-parse HEAD)
git diff --quiet HEAD -- . ':!cluster/ledger.jsonl' || { echo "uncommitted changes in the lane: refusing"; exit 4; }
git cat-file -e "HEAD:experiments/tensor-crossover/$CFG"
Q=$(ssh -o BatchMode=yes tufts-login 'squeue -u $USER -h -o "%i %j %T"')
echo "--- squeue before"; echo "$Q"
if echo "$Q" | awk '{print $2}' | grep -qx "tcx_$JOB"; then echo "tcx_$JOB already queued: refusing"; exit 3; fi
N_MINE=$(echo "$Q" | awk '{print $2}' | grep -c '^tcx_' || true)
[ "$N_MINE" -lt 3 ] || { echo "already $N_MINE lane jobs queued/running (cap 3): refusing"; exit 5; }
ssh -o BatchMode=yes tufts-login "test ! -e $R && mkdir -p $R/logs"
git -C "$(git rev-parse --show-toplevel)" archive --format=tar HEAD experiments/tensor-crossover | ssh -o BatchMode=yes tufts-login "tar -x -C $R"
TMPS=/tmp/claude-1002/tcx_run_$JOB.sbatch
cat > $TMPS <<EOS
#!/bin/bash
#SBATCH --job-name=tcx_$JOB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=$MEM
#SBATCH --time=$MIN
#SBATCH --output=$R/logs/%j.out
#SBATCH --error=$R/logs/%j.err
set -euo pipefail
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$R/cache" TMPDIR="$R/tmp"
mkdir -p "\$TMPDIR" "\$XDG_CACHE_HOME"
cd "$R"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=\$(cat COMMIT.txt) SLURM_JOB_ID
echo "host=\$(hostname) source_commit=\$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
"\$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd "$R/experiments/tensor-crossover"
"\$PY" tc${DIM}d.py --config $CFG --out "$R/output"
cd "$R"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
EOS
scp -q $TMPS "tufts-login:$R/run.sbatch"; rm -f $TMPS
ssh -o BatchMode=yes tufts-login "cd $R && echo $COMMIT > COMMIT.txt && find experiments -type f -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256"
REMOTESUM=$(ssh -o BatchMode=yes tufts-login "cd $R && find experiments -type f | LC_ALL=C sort | xargs cat | sha256sum | cut -d' ' -f1")
LOCALSUM=$(cd "$(git rev-parse --show-toplevel)" && git ls-tree -r --name-only HEAD experiments/tensor-crossover | LC_ALL=C sort | while read -r f; do git show "HEAD:$f"; done | sha256sum | cut -d' ' -f1)
[ "$REMOTESUM" = "$LOCALSUM" ] || { echo "staged tree differs from HEAD: refusing"; exit 6; }
OUT=$(ssh -o BatchMode=yes tufts-login "cd $R && sbatch run.sbatch")
echo "$OUT"
JID=$(echo "$OUT" | awk '{print $NF}')
echo "{\"job\": \"$JOB\", \"jobid\": \"$JID\", \"dim\": $DIM, \"config\": \"$CFG\", \"minutes\": $MIN, \"commit\": \"$COMMIT\", \"submitted\": \"$(date -Is)\"}" >> cluster/ledger.jsonl
echo "--- squeue after"; ssh -o BatchMode=yes tufts-login 'squeue -u $USER -o "%.10i %.24j %.8T %.10M %R"'
