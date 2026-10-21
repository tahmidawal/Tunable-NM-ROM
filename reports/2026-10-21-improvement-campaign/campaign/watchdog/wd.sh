#!/usr/bin/env bash
# Read-only Codex watchdog loop. Usage: wd.sh <index> <offset_seconds> <interval_seconds> <deadline_hours>
# Exits (to notify the coordinator) as soon as a report ends with OVERALL: ACTION-NEEDED, after two
# consecutive failed codex calls, or at its overall deadline.
set -u
i=$1; offset=$2; interval=$3; hours=$4
ROOT=/home/tahmid/Dev/Tunable-NM-ROM-Claude
D=$ROOT/reports/2026-10-21-improvement-campaign/campaign/watchdog
deadline=$(( $(date +%s) + hours*3600 ))
fails=0
sleep "$offset"
while [ "$(date +%s)" -lt "$deadline" ]; do
  ts=$(date -u +%Y%m%dT%H%M%SZ)
  out=$D/wd$i-$ts.md
  timeout 900 codex exec -C "$ROOT" -o "$out" "$(cat $D/prompt.txt)" < /dev/null > "$D/.wd$i-last.log" 2>&1
  if [ ! -s "$out" ]; then
    fails=$((fails+1)); echo "$ts wd$i codex call failed ($fails)" >> $D/wd-events.log
    if [ $fails -ge 2 ]; then echo "WATCHDOG wd$i EXIT: two consecutive codex failures"; exit 2; fi
  else
    fails=0
    last=$(grep -E '^OVERALL:' "$out" | tail -1)
    echo "$ts wd$i $last" >> $D/wd-events.log
    if echo "$last" | grep -q 'ACTION-NEEDED'; then echo "WATCHDOG wd$i EXIT: ACTION-NEEDED in $out"; exit 1; fi
  fi
  sleep "$interval"
done
echo "WATCHDOG wd$i EXIT: deadline reached"; exit 0
