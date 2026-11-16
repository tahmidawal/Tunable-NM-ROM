#!/usr/bin/env bash
# Pull one finished mesh: bash pull.sh t2d256   (small files only; fields stay on the cluster)
set -euo pipefail
cd "$(dirname "$0")"
task=$1
[[ $task =~ ^t[23]d[0-9]+$ ]]
job=/cluster/tufts/paralab/tawal01/sealed-timing-20261115/$task
ssh tufts-login "set -e; cd '$job'; test -f output/measurement.json; grep -q 'SEALED-TIMING COMPLETE' *.out; grep -q 'jax_backend=gpu' *.out; \
  sha256sum -c OUTPUTS.sha256 --quiet; \
  for j in \$(ls *.out | sed 's/.out//'); do sacct -X -j \$j -n -P -o jobid,state,exitcode,elapsed,nodeList; done > accounting.txt"
ssh tufts-login "cd '$job' && python3 - output > output/result-timing.json" < extract.py
mkdir -p results/raw/$task
rsync -aq "tufts-login:$job/output/"{measurement.json,result-timing.json,config-provenance.json,derived-config.json,gpu-start.txt,gpu-trace.csv,node-occupancy-start.txt,node-occupancy-end.txt} results/raw/$task/
rsync -aq "tufts-login:$job/"{'*.out','*.err',accounting.txt,OUTPUTS.sha256,run.sbatch} results/raw/$task/
du -sh results/raw/$task
