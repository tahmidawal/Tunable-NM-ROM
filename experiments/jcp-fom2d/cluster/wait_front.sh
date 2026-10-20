#!/bin/bash
# Wait for jcp-qualify's committed FRONT-2D.json (DESIGN §7/A1.7): poll every 10 min (<= 20 min rule), heartbeat each poll,
# overall deadline given as an absolute UTC epoch. Exits 0 when the file is committed on exp/2026-10-18-jcp-qualify
# (with its .sha256), 3 at the deadline.
LANE=$(cd "$(dirname "$0")/.." && pwd)
DL=$1
Q=/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-qualify
while true; do
  C=$(git -C "$Q" log -1 --format=%H -- experiments/jcp-qualify/runs/FRONT-2D.json 2>/dev/null)
  S=$(git -C "$Q" log -1 --format=%H -- experiments/jcp-qualify/runs/FRONT-2D.sha256 2>/dev/null)
  if [ -n "$C" ] && [ -n "$S" ]; then echo "FRONT committed $C $(date -u +%FT%TZ)"; exit 0; fi
  [ "$(date +%s)" -lt "$DL" ] || { echo "deadline $(date -u +%FT%TZ)"; exit 3; }
  (cd "$LANE" && /home/tahmid/Dev/.venv/bin/python hb.py "waiting for FRONT-2D.json (poll every 10 min; deadline $(date -u -d @$DL +%FT%TZ))" local_pid=$$ expected=$(date -u -d @$DL +%FT%TZ) notes="C1 complete and accepted pending re-audit; C2 smoke accepted; qualify step: $(/home/tahmid/Dev/.venv/bin/python -c "import json;print(json.load(open('$Q/experiments/jcp-qualify/HEARTBEAT.json'))['step'][:120])" 2>/dev/null)")
  sleep 600
done
