#!/usr/bin/env bash
# External watchdog on the Burgers campaign coordinator session. Read-only. Exits (notifying the parent) on a problem.
R=/home/tahmid/Dev/Tunable-NM-ROM-Claude
OUT=$R/reports/2026-10-21-improvement-campaign/external-watch; mkdir -p $OUT
END=$(( $(date +%s) + ${HOURS:-48}*3600 )); i=0; fails=0
paths() { ls -d $R/reports/2026-10-21-improvement-campaign $R/LAB-LOG.md $R/worktrees/2026-10-2[1-9]-*/experiments $R/worktrees/2026-1[01]-*-imp-*/experiments 2>/dev/null; }
while [ $(date +%s) -lt $END ]; do
  sleep 1200; i=$((i+1)); now=$(date +%s); ts=$(date -u +%Y%m%dT%H%MZ)
  # 1. activity
  newest=$(find $(paths) -type f -not -path "*/external-watch/*" -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
  age=$(( now - ${newest%.*} ))
  if [ "$age" -gt 5400 ]; then echo "QUIET: no campaign file changed for $((age/60)) min ($ts)"; exit 3; fi
  # 2. heartbeats
  for hb in $(ls $R/worktrees/2026-10-2[1-9]-*/experiments/*/HEARTBEAT.json $R/worktrees/2026-1[01]-*-imp-*/experiments/*/HEARTBEAT.json 2>/dev/null | sort -u); do
    st=$(grep -o '"step"[^,}]*' "$hb" | head -1)
    case "${st,,}" in *done*|*hand*back*|*handed*|*complete*) continue;; esac
    hage=$(( now - $(stat -c %Y "$hb") ))
    if [ "$hage" -gt 21600 ]; then echo "STALE HEARTBEAT: $hb is $((hage/60)) min old, $st ($ts)"; exit 3; fi
  done
  # 3. hourly Codex review
  if [ $((i % 3)) -eq 0 ]; then
    f=$OUT/watch-$ts.md
    timeout 900 codex exec -C $R -o $f "You are a READ-ONLY watchdog: do not modify, create or delete any file, do not submit, cancel or kill anything, do not write to LAB-LOG.md. Check the health of the JCP improvement campaign run by a coordinator session (prompt: reports/2026-10-21-improvement-campaign/AGENT-PROMPT.md). Inspect: reports/2026-10-21-improvement-campaign/ (lane registry, watchdog reports, INTERVENTIONS.md); every lane worktree under worktrees/ dated 2026-10-21 or later (names contain -imp-) (HEARTBEAT.json age and content, last commit time, audits/ progress, runs/JOB-LEDGER.json); running processes (ps: codex exec, jaxrun, sleep/until waiter loops) and whether each awaited process or output file is alive; cluster state via timeout 60 ssh -o ConnectTimeout=20 tufts-login 'squeue -u tawal01; sacct -u tawal01 -S now-1days --format=JobID,JobName%25,State,End -X' (flag jobs COMPLETED more than 3 hours ago whose outputs were not collected into the lane, and jobs stuck PENDING or past their time); df -h /home; whether the coordinator's internal watchdogs are still producing reports. For each lane give OK / STALLED / HUNG / DEAD / AT-RISK with evidence and the exact recommended action. Normal workflow is NOT a problem: Codex audits returning WRONG/NOT READY that the lane is actively fixing, slightly stale heartbeat bookkeeping, or audit blockers awaiting fixes are expected and must be reported as OK (mention them). Say ACTION-NEEDED only for: a lane STALLED/HUNG/DEAD (no commit, file change or live process for over 90 minutes), cluster jobs completed over 3 hours ago and not collected, jobs stuck or failed without the lane reacting, /home free space under 50 GB, two lanes sharing a worktree or cluster directory, or the coordinator's own watchdogs no longer reporting. End with exactly one line: OVERALL: OK or OVERALL: ACTION-NEEDED." < /dev/null > $f.log 2>&1
    if [ ! -s "$f" ]; then fails=$((fails+1)); [ $fails -ge 2 ] && { echo "WATCHDOG CODEX FAILED twice ($ts)"; exit 4; }; continue; fi
    fails=0
    if grep -q "OVERALL: ACTION-NEEDED" "$f"; then echo "ACTION-NEEDED: see $f ($ts)"; exit 3; fi
  fi
done
echo "watch deadline reached $(date)"
