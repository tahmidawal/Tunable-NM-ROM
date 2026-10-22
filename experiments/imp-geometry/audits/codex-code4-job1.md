Read-only audit of source HEAD `9a94d3de7`; no writes, jobs, or scratch arrays. Tiny NumPy/AST and shell checks only. Concurrent smoke-artifact changes appeared during review; those were not mine. Paths below are relative to `experiments/imp-geometry/`.

**Part 1 — round-3 fixes**

1. **CORRECT — reduced-step carry/trace:** ten entries agree at `rom.py:344,374,379`.
2. **CORRECT — previously repaired solver validity:** Q1/Q2 and confirmation retain their gates (`finalize.py:101,149–151`; `run_cell.py:778,816`).
3. **CORRECT — reused test verdicts/finalist binding:** target-specific verdicts and matching finalist keys remain enforced (`run_cell.py:769–785,814–826`; `decide.py:61–69`).
4. **CORRECT — timing integrity:** every validation finalist receives T-dbl; test has a before/after sentinel and invalidates S on drift. Complete matrices and every-call T-id remain intact (`run_cell.py:582–589,794–842`; `finalize.py:195–207,226–247`).
5. **CORRECT — freeze evidence/code binding:** dedicated freeze hashes final/diag/inter after generation; test compares executing Python/vendor contents (`run_cell.py:713–753`). `code_commit` remains metadata, but content hashes provide the substantive code binding.
6. **NEEDS-FIX — diagnostics/box eligibility:** feature-window and SEM IC diagnostics now exist (`run_cell.py:505–509,692–708`). Box validation eligibility is checked, but **box test-reference eligibility is still bypassed**: attribution ignores `confirmation='unconfirmable'` and reads raw test S; box losses never receive the test-reference check (`decide.py:52–59,106–119`).
7. **NEEDS-FIX — cost certificate:** four ECSW steppers, seven bank rules and conditional bank training are counted, but coverage and defensible bounds remain incomplete (`resources.py:49–55,74–100`; details below).
8. **NEEDS-FIX — submission/failure handling:** one commit now supplies both provenance and archive (`cluster/submit.sh:14–20`). The ERR trap lacks inheritance into functions/subshells (`cluster/job.sbatch.tmpl:22–23`): a failing command followed by successful output inside a function still escapes. Tiny shell reproduction returned `FAILED=0`. Explicit solver exit-code checks in the supplied bodies do work.

**Round-3’s three new bugs**

9. **NEEDS-FIX — Q1 comparison solves:** reduced-arm branch now gates comparison validity; FD/SEM computes it but omits it from `passed` (`finalize.py:108–119,125–130`). In-memory reproduction: comparison statistics report 100 failures, `comparison_solves_valid=False`, yet `passed=True`.
10. **CORRECT — fine-rule diagnostic crash:** `space_at` now uses shared `rule_of` decoding (`run_cell.py:224–226,429–430`).
11. **CORRECT — freeze-before-final ordering:** `final.json` precedes the separate freeze stage (`run_cell.py:636,713–726`; `cluster/bodies/release1_L_wide_S1.sh:3`).

**Additional new verdict-changing code finding**

- **Pooled validity is implemented per case.** `ladder.py:63–66` rejects any LSPG case exceeding 1%, whereas DESIGN’s invalid-step allowance is pooled (`DESIGN.md:218`). For equal 1,000-step cases, counts `[11,0]` satisfy pooled 0.55% but are rejected. This can remove eligible bank candidates and change the outcome.

**Part 2 — jobs and certificate**

- **CORRECT infrastructure:** GPU partition/preflight, absolute venv, highest precision, Python f64, paralab paths, exclusive directory reservation/execution lock, locked account/lane caps, and before/after queue checks are present (`cluster/job.sbatch.tmpl:3–20`; `cluster/submit.sh:11–30`; `run_cell.py:28`; `probe.py:14`).
- **Committed staging is partial:** Python tree comes from pinned `FULL`, but template/body come from live local files, with arbitrary external bodies accepted (`cluster/submit.sh:19–25`). Archive or hash the actual executed submission inputs. Current tracked lane changes also make the dirty-tree guard refuse submission.
- **Ordering is correct; cross-job resumability is missing:** release runs ref→data→bank→ladder→inter→time→diag→freeze. Markers resume within an existing output directory (`run_cell.py:871–883`), but submit refuses existing job directories and every body chooses a fresh `out/cell`. No restoration step brings archived bank/POD/ECSW/reference/freeze artifacts into a new test job (`cluster/submit.sh:18`; `cluster/bodies/cell.sh.in:1–4`). A committed freeze JSON alone cannot satisfy `stage_test`.
- **Bank measurement is not isolated steady-state timing:** three calls create fresh jitted training functions, schedules, initialization and reconstruction; subtraction does not cancel compilation, especially when the all-full call omits the sub-step compile (`probe.py:160–169`; `bank.py:72–143`). No repetition arrays or controlled burn-in establish a reliable replacement bound.
- **ECSW measurement is censored/unrepresentative:** random snapshots/states, only `(r,Mt)=(384,512)`, and a 150-second fit limit versus production’s 3,600 seconds (`probe.py:175–183`; `pod.py:106`; `run_cell.py:310–315`). Timeout/unfinished supports cannot be treated as completed-fit cost.
- **Recertification is not wired:** `resources.py:20–24,52,62` still consumes smoke4/5 and hard-coded estimates, not smoke6 or the full-cell stage timings.
- **Coverage is not a conservative bound:** SEM assumes 30 feasible configurations although L screens 90; bank rules share one measured cost despite different quadrature sizes; conditional box accounting adds training without its added ladder/qualification work (`resources.py:50,53,74–88`; `run_cell.py:261–282,325–331`). Diagnostics are not explicitly budgeted.
- **Certificate disclosure is inconsistent:** JSON says provisional, correctly, but its unmeasured descriptions still say bank 2.5 h, timing 0.5×, S1 0.6× and SEM 15 s while calculations use 3 h, 1×, 1× and 30 s (`resources.py:50–62,98–100`). Calling these established “conservative bounds” is unjustified.
- **Wall time:** meas6 requests 20 minutes and permits two 600-second subprocesses, exceeding A5’s ≤0.15 GPU-h allowance. First-cell projection is 12.604 h before 1.5× headroom, **18.907 h with it**, versus a 16-hour request (`cluster/RELEASE-PLAN.md:9–10`; `resources.py:12`; `runs/cost/projection.json:31–35`).
- **Memory:** 120/160G host requests exceed the listed individual arrays, but those arrays are not simultaneous peak bounds. `gpu:a100` does not enforce the certificate’s 80GB variant; `nvidia-smi` only reports memory (`cluster/job.sbatch.tmpl:4,18`; `resources.py:94–97`). Memory readiness remains unproven.
- **Question coverage:** first-cell outputs address L/wide/S1 validation only; meas6 addresses cost only. Geometry claims require the remaining cells, frozen test confirmation and eligible box attribution. Those are not yet an executable end-to-end release path.
- **“Not releasable on the current certificate” is CORRECT for the remaining panel:** 100.970 projected + 1.074 spent = **102.044 GPU-h > 62** (`runs/cost/projection.json:93–98`). A6 explicitly permits the initial measurement/full-cell exception; it does not authorize the remaining campaign or repair the operational blockers above.

CODE: NEEDS-FIX (FD/SEM Q1 comparison gate; box test eligibility; pooled validity; inherited failure handling)
JOBS: NOT-READY (measurement validity; complete recertification; frozen-artifact restoration; committed execution inputs; wall-time/measurement allowance; memory qualification)