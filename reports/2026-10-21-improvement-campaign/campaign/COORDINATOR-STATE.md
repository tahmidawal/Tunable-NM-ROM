# Coordinator state (for a fresh coordinator session to resume)

Launched 2026-10-21 (wave 1). Agents are background subagents of the coordinator session; their IDs only
work inside that session. If the session is lost, relaunch a lane agent with its brief and tell it to read
its HEARTBEAT.json and last commits first.

| lane | agent id (session-local) | phase |
|---|---|---|
| imp-faststep (E5) | a9d5d6f61bcd3d884 | full lane |
| imp-hrcost (E6) | a47e9790790fe2e93 | full lane (waits on E5 HANDOFF-E6.md) |
| imp-multimesh (E7) | a228f44b8b2245217 | full lane |
| imp-varcoef (E3) | adb67ad37375bd98f | FOM-only first |
| imp-fronts (E8/E9/E10) | af015423829cd4b32 | Phase A (CPU); Phase B on release |
| imp-geometry (E2) | a4c7eaaf4c7697b94 | design mode; production on release |

Watchdogs: detached `WD_CONTINUE=1 setsid nohup bash campaign/watchdog/wd.sh <i> <offset> 1500 96` (check `ps -eo pid,args | grep "[w]d.sh"`); alarms in `watchdog/ACTION-NEEDED.log`.
| imp-box (E1) | ad17a93504184c2d7 | launched 2026-10-22 05:40 UTC; checkpoint after wide-family cells |

Status 2026-10-22 05:40 UTC: done = imp-faststep, imp-hrcost, imp-multimesh, imp-varcoef; imp-fronts Phase B running; imp-geometry parked on job 6175736 (watched via watchdog/parked-jobs.txt by coord-monitor.sh).
- Results page: https://claude.ai/artifact/NB5epQrD3gwVBSAoLzbwts (source reports/2026-10-21-improvement-campaign/page/index.html); update after each wave.
| final compilation | acd05d841b5f3c98c | launched 2026-10-23 01:05 UTC; builds final/make_final.py with E1 pending |
