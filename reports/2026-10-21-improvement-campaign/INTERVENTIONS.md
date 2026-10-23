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
## 2026-10-22 ~07:40 UTC — imp-fronts Phase B done: arm (b) STOPS; E10 activated
- Arm (b) coverage retrain: FC1 median 5.81 % vs K1 8.12 % (1.40×, bar 2×) and 0/8 FC1 cases numerically qualified → stop rule (b) fires. In-range guard passed. E9 descriptive only (s > 112.77 flags 79 % of 24 test cases; below the 30/20 minimum). 6.327 / 15 GPU-h.
- **Coordinator decision:** release arm (c) boundary-compatible bank within E8's remaining 8.67 GPU-h (amendment + audit first); release E10 (step refinement, state-triggered quadrature escalation; single-change vs K1) with its own 10 GPU-h cap. The prompt makes E10 conditional on E8 falling short, which it did.
- 12:01 UTC: imp-geometry job 6175736 (L_wide_S1) ended TIMEOUT at 11.5 h. Lane set active and resumed to collect partial outputs and re-certify.
## 2026-10-22 — imp-fronts done: Problem 3 not fixed
- Arm (c) boundary-compatible bank STOPS (rule gauss128 = 16 384 nodes vs bar ≤ 12 800; tail 1.48× vs 3×; guard passes). E10-s (step to 1/8) STOPS (FC1 0/8 qualified, median 8.29 % vs bar 4.06 %). E10-q STOPS without a job (its trigger never fires: max ρ 0.039 over 17 600 states → quadrature is not binding). E8 10.545 / 15, E10 0.199 / 10 GPU-h.
- **Deviation:** SM-c was submitted although the lane's own `resources.admit` refused it (budget arithmetic still held); the results audit marks it WRONG as a procedure breach; disclosed. E10 reused Phase B test cases (not test-once for E10).
- **Coordinator decision:** no further steep-front design is released in this campaign (the lane's suggested next step — an embedded adaptive integrator or a shift/local-basis bank — is not a single change and is recorded as an open item for the paper / a later campaign).
## 2026-10-22 ~14:00 UTC — imp-geometry: first cell TIMEOUT diagnosed; step 2 authorized
- L_wide_S1 overran in the ladder stage (10.5 h, unfinished; SEM 4.7 h). Fixes A9 (registered reductions, tensor-form bank ladder, resumable ladder). Spent 12.69 / 62. Re-certified: full wide panel 97.6, minus S2 hole 72.5, one wide cell per geometry 24.9 GPU-h (with 1.5 headroom).
- **Coordinator decision:** authorize step 2 = resume L_wide_S1 + H_wide_S1 (≤ 30 GPU-h this step), cap unchanged. Win/withdraw rule unchanged. Pre-registered conditional (amendment before S1 selection data): if S ≥ 2 vs the best competitor at some target on ≥ 1 geometry at S1, raise the cap from the E4 reserve to fund L_wide_S2 + box controls; if S < 2 everywhere at S1, stop and report "not supported at S1; S2 untested".
- 14:40 UTC: imp-geometry agent ended its turn with step-2 jobs 6192623/6192666 running (its waiter hit the 2 h tool limit). Lane parked again; coord-monitor watches both jobs.
## 2026-10-22 ~16:40 UTC — imp-box (E1) checkpoint: stop rules not fired; continue
- W40_A1: 21/27 matched comparisons S ≥ 0.5 (max 1.13; POD arms and bank-ETDRK4 0.7–1.1; bank LSPG 0.04–0.31) → SPEED-STOP cannot fire; BANK-STOP not fired. Provisional (val selection, results audit pending).
- A11: a dense-matrix FOM (same discretisation) is faster than the FFT FOM at batch 1 on H200 and halves the ROM advantage (smoke S 1.6–2.3 → 0.7–1.1) — the fair competitor. A10: H200/JAX 0.10.2 gives non-finite FFT at N = 12 (correct on A100); excluded.
- Coordinator: continue wide cells then narrow cells within 87 GPU-h; smk2-A4 cancelled at 14 h if uncertified. 9.09 GPU-h spent.
- 17:15 UTC: watchdog flagged its own 22 h wd.sh loops as stale waiters; prompt now exempts wd.sh and coord-monitor.sh.
- 18:29 UTC: imp-geometry L_wide_S1_r 6192623 COMPLETED; lane resumed to collect + test-once.
- imp-geometry: L_wide_S1 scored incl. test-once (6202632): bank S = 0.015–0.21 vs best competitor (POD-Galerkin/POD+ECSW) on val, 0.013–0.09 test; S < 2 at every target on L. 25.16/62 GPU-h. Parked again on 6192666.
## 2026-10-22 ~19:45 UTC — GitHub credentials restored (user); all mirrors re-synced
- Prompted by the monitoring session (verified `gh auth status` myself). main pushed (aaf4b6e8b). Every lane's sync_github.sh re-run; `git ls-remote` confirms:
  imp-faststep ed52f5477 (a3e417b1e) · imp-hrcost c891ee4cc (cf351b62a) · imp-multimesh 4ade95d16 (dd4cd7fc4) · imp-varcoef 6a98c018a (e8d39fa1a) · imp-fronts a2f64ebda (506dbd3a5) · imp-geometry 80067e490 (25f55bfc0) · imp-box c77739769.
- 2026-10-23T00:01 UTC: imp-geometry H_wide_S1 6192666 COMPLETED; lane resumed for collect, test-once, results audit, S1 decision.
## 2026-10-23 — imp-geometry (E2) done: geometry claim not supported; E4 not run
- A10 condition → stop: bank S vs best tested competitor (POD-Galerkin / POD+ECSW) 0.015–0.22 (val) and 0.013–0.12 (test) on both L and H at S1; vs SEM 0.09–0.23; vs FD+multigrid > 1 only at τ = 10 % (1.36 / 1.77). H WITHDRAWN (measured loss); L not decidable (4/32 references fail the order gate). 30.68 / 62 GPU-h.
- **E4 (3D cube with holes) not run** (conditional on an E2 win). Its 80 GPU-h reserve is released unspent.
- Disclosed in the lane: geometry variant chosen after inspecting smoke ladder rows (A3.1 retraction of the "no results existed" claim); static condensation / multi-domain Chebyshev not built (S is "against the strongest tested competitor").
