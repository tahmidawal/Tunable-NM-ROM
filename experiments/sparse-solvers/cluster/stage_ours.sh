#!/usr/bin/env bash
# Task B (amendment 3): re-run the paper's own staged run trees, unchanged, in this lane's namespace on a chosen GPU type.
# Each entry is label=<local staged tree>=<original cluster root written in its run.sbatch>.  The tree is copied as is
# (archive/, logs/, output/ of the original run excluded), its own MANIFEST.sha256 is verified on the cluster, and the
# original run.sbatch is executed with bash after replacing only the original root path by the new one (run_here.sh;
# #SBATCH lines are then comments).  One Slurm job runs the entries one after another on one GPU.
# Usage: cluster/stage_ours.sh <attempt> <HH:MM:SS> <gpu feature> <mem> <cpus> label=tree=origroot [...]
set -euo pipefail
cd "$(dirname "$0")/.."
ATTEMPT=$1; WALL=$2; FEAT=$3; MEM=$4; CPUS=$5; shift 5
NS=/cluster/tufts/paralab/tawal01/sparse-solvers-20261113
DIR=$NS/jobs/ours-$ATTEMPT
[ -z "$(git status --porcelain -- cluster)" ] || { echo "uncommitted changes in cluster/; commit first"; exit 2; }
COMMIT=$(git rev-parse HEAD)
mine=$(ssh tufts-login "squeue -u \$USER -h -p gpu -o '%j'" | grep -c '^ss_' || true)
[ "$mine" -lt 3 ] || { echo "already $mine GPU jobs of this lane queued or running"; exit 6; }
ssh tufts-login "test ! -e $DIR && mkdir -p $DIR/logs" || { echo "attempt directory exists: $DIR"; exit 4; }
LABELS=()
for ent in "$@"; do
  IFS='=' read -r label tree orig <<<"$ent"
  [ -f "$tree/run.sbatch" ] && [ -f "$tree/MANIFEST.sha256" ] || { echo "no run.sbatch/MANIFEST in $tree"; exit 7; }
  rsync -a --exclude=/archive --exclude=/logs --exclude=/output --exclude=/out --exclude=/cache --exclude=/tmp \
    "$tree/" "tufts-login:$DIR/$label/"
  ssh tufts-login "cd $DIR/$label && mkdir -p logs && sha256sum -c MANIFEST.sha256 --quiet && grep -q '$orig' run.sbatch \
    && sed 's#$orig#$DIR/$label#g' run.sbatch > run_here.sh && (diff run.sbatch run_here.sh > run_here.diff || true) \
    && echo 'staged $label: manifest ok,' \$(grep -c . MANIFEST.sha256) files, \$(grep -c '^[<>]' run_here.diff) changed lines"
  LABELS+=("$label")
done
cat > /tmp/ss_chain_$$.sh <<EOS
#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
# Lane commit $COMMIT.  Runs the staged trees one after another; a failing tree does not stop the next.
cd $DIR
hostname; date; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
printf 'cpu_model='; lscpu | sed -n 's/^Model name:[[:space:]]*//p'
/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b == 'gpu' else 42)" || exit 42
for label in ${LABELS[*]}; do
  echo "=== \$label start \$(date)"
  t0=\$(date +%s)
  if bash \$label/run_here.sh > \$label/logs/run.out 2> \$label/logs/run.err; then st=ok; else st="failed rc=\$?"; fi
  echo "=== \$label \$st seconds=\$(( \$(date +%s) - t0 ))"
  tail -2 \$label/logs/run.out
done
date
echo "task_completed"
EOS
scp -q /tmp/ss_chain_$$.sh "tufts-login:$DIR/chain.sbatch"; rm -f /tmp/ss_chain_$$.sh
JOB=$(ssh tufts-login "cd $DIR && sbatch --parsable --job-name=ss_ours_$ATTEMPT --time=$WALL --constraint=$FEAT chain.sbatch")
n=$(ssh tufts-login "squeue -u \$USER -h -o '%i %Z'" | awk -v d="$DIR" '$2==d' | wc -l)
[ "$n" -eq 1 ] || { echo "expected exactly one job in $DIR, found $n (job $JOB)"; exit 5; }
echo "submitted job=$JOB dir=$DIR commit=$COMMIT"
mkdir -p ledger
printf '{"attempt":"ours-%s","job":"%s","dir":"%s","commit":"%s","wall":"%s","gpu_feature":"%s","tasks":"%s","submitted_utc":"%s"}\n' \
  "$ATTEMPT" "$JOB" "$DIR" "$COMMIT" "$WALL" "$FEAT" "${LABELS[*]}" "$(date -u +%FT%TZ)" >> ledger/JOBS.jsonl
