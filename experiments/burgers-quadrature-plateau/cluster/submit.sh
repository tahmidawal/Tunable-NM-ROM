#!/bin/bash
# burgers-quadrature-plateau: stage ONE job from the COMMITTED lane tree and submit it (DESIGN §10, A1.8, A2.5).
#   bash cluster/submit.sh <job> <dim:2|3> <config> <gpu:a100-80G|h100|h200> <mem> <minutes>
# - refuses if bqp_<job> is queued or the remote job dir exists (one job per directory; duplicate-submit race);
# - streams `git archive HEAD experiments/burgers-quadrature-plateau` straight into the paralab job dir (no login /tmp);
# - writes run.sbatch + MANIFEST.sha256 (checked on the node) + COMMIT.txt; gpu partition only; venv; f64; highest;
#   jax_backend=gpu preflight (exit 42); E0 store read-only via JCPREF_STORE;
# - budget guard: python cluster/ledger.py check <minutes> must pass (GPU-h used + outstanding limits + new <= 4.0);
# - squeue before and after; records the job in cluster/ledger.json.
set -euo pipefail
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=gc.auto GIT_CONFIG_VALUE_0=0
JOB="$1"; DIM="$2"; CFG="$3"; GPU="$4"; MEM="$5"; MIN="$6"
[[ "$JOB" =~ ^[a-zA-Z0-9]+$ ]]
LANE_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$LANE_DIR"
NS=/cluster/tufts/paralab/tawal01/bqplateau-20261105
R="$NS/$JOB"
COMMIT=$(git rev-parse HEAD)
git diff --quiet HEAD -- . ':!cluster/ledger.json' || { echo "uncommitted changes in the lane: refusing"; exit 4; }
git cat-file -e "HEAD:experiments/burgers-quadrature-plateau/$CFG"
/home/tahmid/Dev/.venv/bin/python cluster/ledger.py check "$MIN"
case "$GPU" in
  a100-80G) GRES="gpu:a100:1"; CONS="#SBATCH --constraint=a100-80G";;
  h100) GRES="gpu:h100:1"; CONS="";;
  h200) GRES="gpu:h200:1"; CONS="";;
  *) echo "bad gpu"; exit 2;;
esac
DRIVER=plateau${DIM}d.py
Q=$(ssh tufts-login 'squeue -u $USER -h -o "%i %j %T"')
echo "--- squeue before"; echo "$Q"
if echo "$Q" | awk '{print $2}' | grep -qx "bqp_$JOB"; then echo "bqp_$JOB already queued: refusing"; exit 3; fi
N_MINE=$(echo "$Q" | awk '{print $2}' | grep -c '^bqp_' || true)
[ "$N_MINE" -lt 2 ] || { echo "already $N_MINE lane jobs queued/running (cap 2): refusing"; exit 5; }
ssh tufts-login "df -h /cluster/tufts/paralab | tail -1; test ! -e $R && mkdir -p $R/logs"
git -C "$(git rev-parse --show-toplevel)" archive --format=tar HEAD experiments/burgers-quadrature-plateau | ssh tufts-login "tar -x -C $R"
cat > /tmp/claude-1002/bqp_run_$JOB.sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=bqp_$JOB
#SBATCH --partition=gpu
#SBATCH --gres=$GRES
$CONS
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
export JCPREF_STORE=/cluster/tufts/paralab/tawal01/jcp-ref/store
mkdir -p "\$TMPDIR" "\$XDG_CACHE_HOME"
cd "$R"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=\$(cat COMMIT.txt) SLURM_JOB_ID
echo "host=\$(hostname) source_commit=\$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"\$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd "$R/experiments/burgers-quadrature-plateau"
"\$PY" $DRIVER --config $CFG --out "$R/output"
cd "$R"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
EOF
scp -q /tmp/claude-1002/bqp_run_$JOB.sbatch "tufts-login:$R/run.sbatch"
rm -f /tmp/claude-1002/bqp_run_$JOB.sbatch
ssh tufts-login "cd $R && echo $COMMIT > COMMIT.txt && find experiments -type f -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256 && sha256sum -c MANIFEST.sha256 --quiet"
# verify the staged tree equals the commit (content hash against git, not the working tree)
REMOTESUM=$(ssh tufts-login "cd $R && find experiments -type f | LC_ALL=C sort | xargs cat | sha256sum | cut -d' ' -f1")
LOCALSUM2=$(cd "$(git rev-parse --show-toplevel)" && git ls-tree -r --name-only HEAD experiments/burgers-quadrature-plateau | LC_ALL=C sort | while read -r f; do git show "HEAD:$f"; done | sha256sum | cut -d' ' -f1)
[ "$REMOTESUM" = "$LOCALSUM2" ] || { echo "staged tree differs from HEAD ($REMOTESUM vs $LOCALSUM2): refusing"; exit 6; }
OUT=$(ssh tufts-login "cd $R && sbatch run.sbatch")
echo "$OUT"
JID=$(echo "$OUT" | awk '{print $NF}')
/home/tahmid/Dev/.venv/bin/python cluster/ledger.py add "$JOB" "$JID" "$MIN" "$GPU" "$CFG" "$COMMIT"
echo "--- squeue after"; ssh tufts-login 'squeue -u $USER -o "%.10i %.24j %.8T %.10M %R"'
