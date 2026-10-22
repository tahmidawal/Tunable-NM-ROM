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
- 00:00 UTC 10-22: wd2 flagged imp-geometry STALLED (heartbeat 68 min, awaited local PID 930688 exited). Messaged the lane to read the output and refresh its heartbeat.
## 2026-10-22 ~00:30 UTC — imp-geometry (E2) ready note; scope decision
- Design-mode work done (problem redesigned to v4a after a zero-interaction finding, disclosed as diagnostic-guided; SEM, FD-multigrid, FD-RK4, POD-ECSW/Galerkin competitors; all must-fail controls fire). Certified projection 202 GPU-h full / 101 wide-only vs cap 62 (assumption-heavy; best guess ~52 wide). Spent 1.07 (0.07 over the 1 GPU-h smoke allowance, a smoke ran to its limit during the usage-limit pause). Job audit NOT-READY.
- **Coordinator decision:** cap stays 62. Production scope = wide-family cells + box controls (the win rule is judged on the wide family); narrow-family cells deferred; then A5.2 reductions, then the wide S2 hole cell, if needed. Released a measurement job + first cell (L/wide/S1, incumbent CN-R) capped at 12 GPU-h after code re-audit and a READY job audit; rest released after re-certification from measured costs.
- Noted (favours competitors): competitors gated on validity+accuracy only, their Q1/Q2 reported not gated.
- 01:55 UTC 10-22: second account usage-limit stop (~01:00 UTC) halted all six agents. Resumed five; imp-geometry left idle on purpose: its detached waiter keeps the heartbeat fresh while cell job 6175736 (L_wide_S1, 11.5 h limit) runs; resume it when the job ends (watchdog will flag).
## 2026-10-22 ~02:20 UTC — imp-hrcost (E6) done
- Both stop rules fired (F2: m* = Gauss 96², worse worst error 2.131 % vs 1.851 %, 1.02×; F3: no κ ≤ 16 passes Q2, best 1.24×). 1.461 / 3 GPU-h.
- **Rule deviation noted:** the smk1 job audit said DO NOT SUBMIT (offline-acceptance gaps); the lane submitted anyway, arguing the job's computation was unaffected (`audits/response-job-smk1-1.md`), and fixed the gaps before acceptance. Coordinator accepts it as a smoke-only deviation; it goes in the final report's deviations list.
- 03:30 UTC: imp-geometry's detached waiter died (heartbeat 63 min), job 6175736 still RUNNING (2:55/11:30). Lane set to status 'parked' in LANES.json (watchdogs skip it); the coordinator's Monitor now polls sacct for 6175736 and resumes the lane when it ends.
## 2026-10-22 ~03:50 UTC — imp-multimesh (E7) done: KILL CRITERION FIRES
- Continuous POD built once matches us (≤ worst error, 0.099–0.131× our cost) on 126/126 (workload, configuration) pairs → **comparative multi-mesh cost claim dropped** (framing change for the paper). 3.16 / 8 GPU-h.
- Lane reports a shared-scratchpad collision: it overwrote `p_design1.txt`, so imp-faststep's codex-design-3 may have run with the wrong prompt. Messaged imp-faststep to check and re-audit if so; lane-specific scratch subfolders from now on.
## 2026-10-22 ~04:00 UTC — imp-faststep (E5) done: stop rule fires (worst error)
- All Galerkin arms Q1/Q2-qualified and stable; ETDRK4 evolution 14–24× faster than LSPG CN-R at r=384, but worst error slightly above the incumbent (1.82 % / 1.91 % vs 1.78 %; hardest case val32|19), so E5 is "not fixed" under the registered rule. Galerkin wins 36/38 cases at r=384. Matched POD-ECSW with the same ETDRK4 step is still faster at r=384 (0.62–0.80 of our time). 1.41 / 4 GPU-h.
- Consequence: E2 (and E1) keep the incumbent CN-R LSPG stepper; no "best stepper" passed.
- Lane incident: an early `pkill -f codex-design-4.md` may have killed another lane's Codex audit (`audits/NOTE-pkill-incident.txt`); lanes re-run dead audits by rule, no further action.
- 04:18 UTC: imp-varcoef STALLED (no tool call or file change for 65 min, no live processes; queued nudges undelivered). TaskStop + SendMessage resume with instructions to finish the report in small edits and clean ref10/ref100.
## 2026-10-22 ~05:30 UTC — imp-varcoef (E3) done: gate fails, no ROM trained
- Fastest tuned FOM (ETD with constant-coefficient split, N48) costs G = 0.859 (3 %) / 0.923 (10 %) of our box ROM in the same job — not ≥ 2× slower. 2.04 GPU-h.
- **Coordinator ruling on the post-hoc A10 cross-check rule:** accepted as a *post-hoc label carried forward* (like E0 A10), adopted before any solver result; the final report states both outcomes (A10: gate fails; registered rule: UNRESOLVED). Either way no ROM is trained, so the conclusion does not depend on it.
- Wave 1 closed (E5, E6, E7, E3 done; E8 Phase A done). Measured wave-1 GPU-h ≈ 8.3 vs ≈ 30 planned.
- 05:40 UTC: launched imp-box (E1) early in wave 3 (independent of E2/E8 verdicts; stop rule evaluated on wide cells first, coordinator checkpoint there). Lab-log + page update for wave 1.
