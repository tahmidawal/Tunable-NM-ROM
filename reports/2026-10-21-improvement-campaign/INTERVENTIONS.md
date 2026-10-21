# Interventions log — improvement campaign (2026-10-21)

Every coordinator intervention (resume, kill, re-run) on a watchdog notification or otherwise.

## 2026-10-21 18:15 UTC — wd2 ACTION-NEEDED (wd2-20261021T181211Z.md)
- imp-fronts: two waiters (PIDs 3777072, 3781812) used `until ! pgrep -f "<pattern>"` where the pattern matched only the waiter itself (the real process, test_train_arms.py --dry --keep --json, alive, does not contain the literal pattern). Verified never-terminating; killed both; messaged the lane to wait on the captured PID instead (COMMON-BRIEF §8).
- Machine: PID 1580784, a 22 h self-matching `pgrep` waiter from another (side) session's save sweep (scratchpad 22f52704…). Verified it could never terminate; killed.
- imp-multimesh: job 6158333 FAILED after 3 min; heartbeat fresh, lane handling it — no action.
- 18:28 UTC wd2 ACTION-NEEDED again for imp-multimesh smoke job 6158659 FAILED with a fresh heartbeat (lane debugging, not a stall). Tightened watchdog prompt: failed jobs with heartbeat < 45 min are OK. No lane action.
- ~20:10 UTC: watchdog loops moved off the 2-h tool limit: `wd.sh` now (with WD_CONTINUE=1) appends ACTION-NEEDED / codex-failure lines to `campaign/watchdog/ACTION-NEEDED.log` and keeps running; both run detached (`setsid nohup`, 96 h deadline, 25-min interval, staggered ~12 min). The coordinator tails that log with a Monitor (re-armed every 30 min) that also alarms if a watchdog process dies.
- 22:15 UTC: all six lane agents stopped on the account session usage limit (~19:30 UTC); watchdogs reported ACTION-NEEDED (stale heartbeats) every round since 19:38 — expected. Cluster queue empty. Resumed all six via SendMessage after the limit reset. The tail -F Monitor delivered no events (unexplained); replaced it with a polling monitor.
## 2026-10-21 ~23:40 UTC — imp-fronts Phase A hand-back; Phase B released
- Phase A done (0 GPU-h): A1 smearing INCONCLUSIVE (control failed to separate), A2 wall incompatibility RUN (arm c eligible), A3 truncation screen passed (GPU timing deferred), A4 hybrid floors STOP.
- **Coordinator decision:** release Phase B as S3 = coverage arm (b) vs control K1 (10.05 GPU-h ceilings) within the 15 GPU-h cap; cap NOT raised. Arms (a) fidelity and (c) boundary-compatible deferred (S1 needs 15.75). If S3 leaves room, (c) has priority (needs its architecture amendment + audit first).
- **Deviation:** E8 retrains released before the other wave-1 lanes hand back — they are independent of wave-1 results and fully capped.
- **GitHub credentials expired on the box** (`gh auth status`: failed login; `could not read Username`). All lane mirrors stop pushing; work is committed locally only. Push notification sent to the user. Lanes keep committing; coordinator re-syncs every lane when credentials return. imp-fronts last mirror 12e983cba, local 40e650676.
