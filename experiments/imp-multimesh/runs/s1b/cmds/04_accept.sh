set -uo pipefail
env JAX_PLATFORMS=cpu $PY mmaccept.py $OUT $TASK_ROOT/logs/$SLURM_JOB_ID.out
