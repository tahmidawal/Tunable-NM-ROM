#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
# Lane commit a6c0c8d09f4a197d5f6fe853df0dd6e88e96f245.  Runs the staged trees one after another; a failing tree does not stop the next.
cd /cluster/tufts/paralab/tawal01/sparse-solvers-20261113/jobs/ours-p2d-small-h200
hostname; date; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
printf 'cpu_model='; lscpu | sed -n 's/^Model name:[[:space:]]*//p'
/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b == 'gpu' else 42)" || exit 42
for label in p2t256 p2t1024 p2t2048; do
  echo "=== $label start $(date)"
  t0=$(date +%s)
  if bash $label/run_here.sh > $label/logs/run.out 2> $label/logs/run.err; then st=ok; else st="failed rc=$?"; fi
  echo "=== $label $st seconds=$(( $(date +%s) - t0 ))"
  tail -2 $label/logs/run.out
done
date
echo "task_completed"
