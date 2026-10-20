#!/bin/bash
# usage: run_codex.sh N  -> runs round N in the foreground with timeout, waits by PID
cd /home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/final
timeout 3600 codex exec -C /home/tahmid/Dev/Tunable-NM-ROM-Claude -o audits/codex-final-$1.md "$(cat audits/prompt-final-$1.txt)" < /dev/null > audits/codex-final-$1.log 2>&1 &
echo $!
