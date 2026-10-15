#!/bin/bash
# jcp-speed3d cluster smoke (DESIGN A1.9, A1.10, A1.12): every mandatory gate test and every s3run mode on cases 0, 5, 7,
# each step's exit status recorded; smoke_accept.py writes output/ACCEPTANCE.json (pass / fail per gate). The script
# itself always finishes so that the evidence is collected; acceptance is read from ACCEPTANCE.json, never from completion.
set -uo pipefail
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
O="$TASK_ROOT/output"
mkdir -p "$O"
echo "job=${SLURM_JOB_ID:-none} commit=${SOURCE_COMMIT:-none}" > "$O/STEPS.txt"
step() { local name=$1; shift; local t0=$(date +%s); echo "=== $name"; "$@" > "$O/$name.log" 2>&1; local rc=$?; echo "$name rc=$rc seconds=$(( $(date +%s) - t0 ))" | tee -a "$O/STEPS.txt"; }
step test_s3_M2 "$PY" test_s3.py --setting M2_512 --rule gl32 --cases 0 5 7 --out "$O/test_s3-M2_512.json"
step test_s3_W1024 "$PY" test_s3.py --setting W1024 --rule lat65536 --cases 0 5 7 --out "$O/test_s3-W1024.json"
step test_fom "$PY" test_fom.py --L 64 --skip-e0 --out "$O/test_fom.json"
for c in smkgrid smkfom smkprof smkmesh; do step "$c" "$PY" s3run.py --config "configs/$c.json" --out "$O/$c"; done
step accept "$PY" smoke_accept.py --out "$O"
cat "$O/STEPS.txt"
exit 0
