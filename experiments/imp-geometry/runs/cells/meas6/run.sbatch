#!/bin/bash
#SBATCH --job-name=igeo-meas6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:09:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/imp-geometry/meas6/slurm-%j.out
set -euo pipefail
exec 9>/cluster/tufts/paralab/tawal01/imp-geometry/meas6/.exec.lock
flock -n 9 || { echo "another execution holds /cluster/tufts/paralab/tawal01/imp-geometry/meas6"; exit 3; }
cd /cluster/tufts/paralab/tawal01/imp-geometry/meas6/code
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LANE_COMMIT=f77177954
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b == 'gpu' else 42)" || exit 42
df -h /cluster/tufts/paralab | tail -1
FAILED=0
set +e +o pipefail   # (no pipefail: display pipelines like grep | tail must not count as failures); the body counts failures itself (run() increments FAILED); any other failing command too:
set -E           # errtrace: the ERR trap is inherited by functions and subshells
rm -f /cluster/tufts/paralab/tawal01/imp-geometry/meas6/.failures
trap 'echo "$BASH_COMMAND" >> /cluster/tufts/paralab/tawal01/imp-geometry/meas6/.failures' ERR   # inherited (set -E) by subshells and command substitutions
OUT=$(dirname $(pwd))/out; mkdir -p $OUT
run() { local name=$1; shift; echo "=== $name: $*"; local t0=$(date +%s); timeout 220 "$@" > $OUT/$name.log 2>&1; rc=$?; [ $rc -ne 0 ] && FAILED=$((FAILED+1)); echo "=== $name rc=$rc $(( $(date +%s)-t0 ))s"; grep -E 'bank sub' $OUT/$name.log | tail -2; tail -2 $OUT/$name.log; }
run probe_bankstep_H_S2 $PY probe.py --out $OUT/probe --dom H --lev S2 --sections bankstep
run probe_bankstep_L_S1 $PY probe.py --out $OUT/probe --dom L --lev S1 --sections bankstep

# bodies may only call run() (make_body.sh / bodies/*.sh); a failure anywhere in a run() step is also caught from its
# recorded exit code, so a failure inside a subshell or command substitution cannot be lost
NFAIL=$(grep -c '^=== .* rc=[1-9]' "/cluster/tufts/paralab/tawal01/imp-geometry/meas6/slurm-${SLURM_JOB_ID}.out" || true)
NERR=$(cat /cluster/tufts/paralab/tawal01/imp-geometry/meas6/.failures 2>/dev/null | wc -l)
if [ "$FAILED" -ne 0 ] || [ "$NFAIL" -ne 0 ] || [ "$NERR" -ne 0 ]; then echo "JOB FAILED ($FAILED counted, $NFAIL recorded)"; exit 1; fi
echo "JOB DONE"
