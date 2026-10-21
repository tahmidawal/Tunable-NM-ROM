#!/usr/bin/env bash
# Campaign-wide local GB10 slot guard: at most 3 concurrent jaxrun jobs across ALL lanes.
# Usage: bash jaxslot.sh <max_wait_seconds> jaxrun /home/tahmid/Dev/.venv/bin/python script.py ...
# Takes the first free slot lock of /tmp/burgers-jaxslot-{1,2,3}.lock (held for the job's lifetime),
# waiting up to <max_wait_seconds> (polls every 30 s). Exits 75 if no slot frees in time.
set -u
wait=$1; shift
deadline=$(( $(date +%s) + wait ))
while :; do
  for i in 1 2 3; do
    exec {fd}>/tmp/burgers-jaxslot-$i.lock
    if flock -n "$fd"; then
      echo "jaxslot: holding slot $i (pid $$)" >&2
      "$@"; rc=$?
      exit $rc
    fi
    exec {fd}>&-
  done
  [ "$(date +%s)" -ge "$deadline" ] && { echo "jaxslot: no free local slot within ${wait}s" >&2; exit 75; }
  sleep 30
done
