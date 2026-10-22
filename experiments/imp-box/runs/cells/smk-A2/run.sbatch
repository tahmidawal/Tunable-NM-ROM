#!/bin/bash
#SBATCH --job-name=ibox-smk-A2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/imp-box/smk-A2/slurm-%j.out
set -euo pipefail
exec 9>/cluster/tufts/paralab/tawal01/imp-box/smk-A2/.exec.lock
flock -n 9 || { echo "another execution holds /cluster/tufts/paralab/tawal01/imp-box/smk-A2"; exit 3; }
cd /cluster/tufts/paralab/tawal01/imp-box/smk-A2/code
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LANE_COMMIT=d149f94e0
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "CUDA_VISIBLE_DEVICES unset: cannot identify the allocated GPU"; exit 43; }
GPUSEL=${CUDA_VISIBLE_DEVICES%%,*}
nvidia-smi -i "$GPUSEL" --query-gpu=uuid,name --format=csv,noheader | tee /cluster/tufts/paralab/tawal01/imp-box/smk-A2/allocated_gpu.txt
grep -q H200 /cluster/tufts/paralab/tawal01/imp-box/smk-A2/allocated_gpu.txt || { echo "allocated GPU is not an H200: refusing (imp-box timing contract)"; exit 43; }
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b == 'gpu' else 42)" || exit 42
df -h /cluster/tufts/paralab | tail -1
unset IGEO_NCAP IMPBOX_TINY IMPBOX_BANK_STEPS
# external memory sampling (survives SIGKILL / OOM of the Python process; job audit 1)
# columns: epoch, allocated-GPU uuid, its memory.used MiB, job cgroup memory bytes (sampled usage, not exact peaks)
CGP=$(awk -F: '$2 ~ /memory/ || $1 == "0" {print $3; exit}' /proc/self/cgroup)
CGF=""; for f in /sys/fs/cgroup$CGP/memory.current /sys/fs/cgroup/memory$CGP/memory.usage_in_bytes; do [ -r "$f" ] && { CGF=$f; break; }; done
SID=$(ps -o sid= $$ | tr -d ' ')
echo "cuda_visible=$CUDA_VISIBLE_DEVICES gpu=$GPUSEL cgroup_file=${CGF:-none} session=$SID (samples every 30 s; not exact peaks)" > /cluster/tufts/paralab/tawal01/imp-box/smk-A2/mem_samples.log
# host column: job cgroup memory bytes when readable, else the summed RSS (KiB) of this job's process session
( while true; do H=$( [ -n "$CGF" ] && echo "cg:$(cat $CGF)" || echo "rss_kib:$(ps -eo rss=,sid= | awk -v s=$SID '$2==s{t+=$1} END{print t}')" ); echo "$(date +%s) $(nvidia-smi -i "$GPUSEL" --query-gpu=uuid,memory.used --format=csv,noheader,nounits | tr -d ' ') $H"; sleep 30; done ) >> /cluster/tufts/paralab/tawal01/imp-box/smk-A2/mem_samples.log 2>&1 &
MEMPID=$!
FAILED=0
set +e +o pipefail   # (no pipefail: display pipelines like grep | tail must not count as failures); the body counts failures itself (run() increments FAILED); any other failing command too:
set -E           # errtrace: the ERR trap is inherited by functions and subshells
rm -f /cluster/tufts/paralab/tawal01/imp-box/smk-A2/.failures
trap 'echo "$BASH_COMMAND" >> /cluster/tufts/paralab/tawal01/imp-box/smk-A2/.failures' ERR   # inherited (set -E) by subshells and command substitutions
OUT=$(dirname $(pwd))/out; mkdir -p $OUT
run() { local name=$1; shift; echo "=== $name: $*"; local t0=$(date +%s); "$@" > $OUT/$name.log 2>&1; rc=$?; [ $rc -ne 0 ] && FAILED=$((FAILED+1)); echo "=== $name rc=$rc $(( $(date +%s)-t0 ))s"; tail -5 $OUT/$name.log; }
for st in ref data pod bank ladder time controls; do
  run $st $PY cell.py --W W40 --lev A2 --out $OUT/cell --stages $st 
  [ $FAILED -ne 0 ] && break
done

# bodies may only call run() (make_body.sh / bodies/*.sh); a failure anywhere in a run() step is also caught from its
# recorded exit code, so a failure inside a subshell or command substitution cannot be lost
NFAIL=$(grep -c '^=== .* rc=[1-9]' "/cluster/tufts/paralab/tawal01/imp-box/smk-A2/slurm-${SLURM_JOB_ID}.out" || true)
NERR=$(cat /cluster/tufts/paralab/tawal01/imp-box/smk-A2/.failures 2>/dev/null | wc -l)
kill $MEMPID 2>/dev/null || true
if [ "$FAILED" -ne 0 ] || [ "$NFAIL" -ne 0 ] || [ "$NERR" -ne 0 ]; then echo "JOB FAILED ($FAILED counted, $NFAIL recorded)"; exit 1; fi
echo "JOB DONE"
