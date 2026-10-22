Read-only audit; no files changed, jobs submitted, or scratch arrays created. Paths below are relative to `experiments/imp-geometry/`.

**Round-2 blocking fixes**

1. **CORRECT — Reduced-step carry/trace.** Ten entries now agree throughout (`rom.py:344,374,379`). Tiny CPU/f64 queries completed for all four steppers with finite outputs.

2. **CORRECT — Previously identified solver-validity omissions.** Q1 gates LSPG unconverged counts (`finalize.py:101,116–120`); Q2 gates both rollouts (`finalize.py:149–151,174`); confirmation calls `stats_valid` (`run_cell.py:726,763`). ETDRK4 checks intermediate-stage finiteness (`rom.py:253`).

3. **CORRECT — Test reuse and finalist binding.** Reused outputs get fresh target/statistic verdicts (`run_cell.py:717–721`); test pairs use matching, valid finalists and qualified banks (`run_cell.py:761–773`); confirmation reads that summary’s keys (`decide.py:61–69`).

4. **NEEDS-FIX — Timing integrity.** Complete-row max–min and every-call T-id are repaired (`finalize.py:195–207,224–245`; incomplete-matrix fixture returns no S). However, test timing still has **no sentinel** (`run_cell.py:736–785`), and validation T-dbl checks only the first finalist per family (`run_cell.py:565–567`).

5. **NEEDS-FIX — Immutable freeze.** Staged-copy location, descriptors, and listed hashes are checked (`run_cell.py:691–701`), but recorded `code_commit` remains unchecked. Decision inputs `final.json` and `inter.json` are hashed only if already present; `diag.json` is omitted (`run_cell.py:609–615`). This does not bind all decision evidence or executing code.

6. **NEEDS-FIX — Status, controls, diagnostics, attribution.** Unqualified status, diagnostic rejection, and interaction convergence gating are repaired (`run_cell.py:590–591,637–644`; `decide.py:29–35`). SEM IC-interpolation and matched feature-window diagnostics remain absent (`run_cell.py:568–580,647–678`). Box attribution reads raw summaries without checking box reference/diagnostic eligibility (`decide.py:107–115`).

7. **NEEDS-FIX — Cost certificate.** POD/ECSW probes now use full-grid IC projection (`probe.py:90,178`). Coverage estimates remain stale, including ECSW and the expanded bank rules; “total_registered” still excludes conditional box banks (`resources.py:50–55,77–85`). A6 permits the initial measurement cell, not the remaining campaign.

8. **NEEDS-FIX — Submission/failure handling.** Cap queries now fail closed (`cluster/submit.sh:16`). However, `COMMIT` and archived `FULL` still come from separate HEAD reads (`:14,18`), and the batch disables errexit around the entire body, allowing unwrapped failures to finish successfully (`cluster/job.sbatch.tmpl:22–25`).

**Round-2 listed new bugs**

- **CORRECT — CN/LSPG crash repaired:** `rom.py:344,374,379`; all four tiny CPU queries completed.
- **CORRECT — Wrong reused test verdict repaired:** `run_cell.py:717–721`.
- **CORRECT — Incomplete timing matrices repaired:** `finalize.py:224–245`; missing fast-competitor fixture no longer yields inflated S.
- **CORRECT — Mixed-finalist confirmation repaired:** `run_cell.py:761–773`; `decide.py:61–69`.

**Only new verdict-changing bugs**

- **Q1 accepts failed tight/tighter comparison solves.** Their statistics are discarded (`finalize.py:104–110,124–126`). An in-memory fixture with **100/100 unconverged comparison steps** and identical outputs still returns `passed=True`. Validate every comparison replay before using its distance.

- **A7 fine-rule finalists crash required diagnostics.** `space_at` passes `"f8"`/`"f12"` directly as quadrature degree instead of decoding the doubled subdivision (`run_cell.py:423–424`). `stage_diag` calls this without exception handling (`:672`); CPU probes reproduce `TypeError: deg must be an integer`. Share the rule decoding used at `:228–230`.

- **Freeze generation precedes final-result generation.** `final.json` is conditionally hashed at `run_cell.py:609–610`, then written at `:621`. A fresh run leaves it unprotected; a rerun with an existing final hashes the previous contents and then invalidates its own freeze.

The supplied smoke directory contains stage timing records, but currently lacks `final.json`, `frozen.json`, and `diag.json`; it cannot substantiate the repaired selection/freeze/diagnostic path.

**OVERALL: NEEDS-FIX**

Blocking fixes only: validate Q1 comparison solves; decode fine rules in diagnostics; complete timing controls; freeze all decision artifacts after generation and verify code provenance; finish required diagnostics and box eligibility gates; recertify campaign coverage before further releases; pin one submission commit and fail closed throughout batch execution.