**NEEDS-FIX.** Read-only audit completed: no files changed, jobs submitted, or scratch arrays created. Tiny CPU/f64 checks confirmed the solver crash, Q1 acceptance bug, and max–min failures below.

Paths are relative to `experiments/imp-geometry/`. Verdicts assess the ten required fixes against Amendment A6.

1. **CORRECT — Search coverage and screening.** SEM coverage follows A6; ECSW includes all four steppers; bank searches every rule (`run_cell.py:262–278,302–321`). Selection considers eligible rows using warm single-query times and screens implementation variants (`ladder.py:103–109`; `run_cell.py:439–460`).

2. **WRONG — Solver validity throughout.** Ladder and validation selection now use solver statistics (`ladder.py:52–66`; `run_cell.py:477–482`), and ETDRK4 conditioning is enforced (`rom.py:191–193`). However, Q1 ignores LSPG unconverged counts when deciding passage (`finalize.py:99,113–116`), and confirmation never calls `stats_valid` (`run_cell.py:692–702`). CPU fixture: **100/100 unconverged steps passes Q1**, while `stats_valid` rejects it.

3. **NEEDS-FIX — Q2 and untimed certificates.** Tested mass/stiffness blocks and IC differences are checked (`finalize.py:158–167`); ETDRK4 independent calculations are guarded by `cert` (`rom.py:203–209,253–258`). However, Q2 discards all rollout solver statistics (`finalize.py:143–145,160`), allowing failed successor solves to qualify. ETDRK4 checks finiteness only after reconstructing the updated state, not explicitly every stage (`rom.py:259–262`).

4. **WRONG — Decisions and confirmation.** Reference/zero-field gates and geometry/attribution aggregation exist (`decide.py:24–59,69–116`). But confirmation checks the **validation-winning keys**, while using a potentially different test max–min winner (`decide.py:52–59`). Test aggregation does not gate pairs on test matching, solver validity, or bank qualification (`run_cell.py:719–726`). Failed qualification can also become “censored” instead of invalid/no evidence (`run_cell.py:575–584`).

5. **WRONG — Immutable test freeze.** Artefact hashes are checked, but `--frozen` can point to the output’s own `frozen.json`; commitment is never verified (`run_cell.py:666–672`). Recorded `code_commit` and `descriptors_sha` are never checked. `ref.json` and dev references, which affect confirmation, are absent from the frozen artefact list (`run_cell.py:588–592,674–682`).

6. **WRONG — Timing integrity and reuse.** Actual timed outputs are compared, and T-dbl is serial (`finalize.py:173–209`; `run_cell.py:537–540`). But sentinel failure and T-dbl failure never invalidate summaries (`run_cell.py:543–559,596–599`). Test timing checks only T-id: no compilation counter, sentinel, or T-dbl (`run_cell.py:703–726`). `paired` retains only the final repetition’s first A and B outputs, discarding the second A and earlier outputs (`finalize.py:183–188`).

7. **CORRECT — RK4, ECSW sampling, POD IC.** The squared cut fraction is removed (`run_cell.py:242`; `fd.py:250–259`); ECSW samples the remaining nodes without truncation (`pod.py:120–125`); production POD uses the full data grid (`run_cell.py:215–220,414–417`).

8. **NEEDS-FIX — Controls and diagnostics.** Revised MG/quadrature thresholds, step-identity comparison, and FD-field perturbation are implemented (`controls.py:69–76,170–184,218–233`; `run_cell.py:122–135`). But the new trace implementation crashes all CN/LSPG queries—see below. Diagnostic failures remain report-only (`run_cell.py:637–654`); SEM IC-interpolation and matched feature-window diagnostics remain absent. Interaction attribution still discards solver status (`run_cell.py:613–620`).

9. **NEEDS-FIX — Probes and cost certificate.** Function-order and data-grid bugs are repaired (`probe.py:74–81,187–188`). However, POD/ECSW probes still use 128-grid IC projection (`probe.py:90,178`) while production uses the full grid. `resources.py:49–55,85` still estimates old coverage and excludes conditional banks from “total_registered.” A6 permits the initial measurement cell, but this does not certify the remaining campaign.

10. **NEEDS-FIX — Submission/provenance/failure handling.** Locked reservation/submission, both caps, archived code, and execution locking are present (`cluster/submit.sh:15–28`; `cluster/job.sbatch.tmpl:10–11`). Remote cap pipelines lack fail-closed handling for `squeue` errors (`submit.sh:16–17,28`). The batch template lacks `errexit` and relies entirely on body commands updating `FAILED`; setup or unwrapped body failures can still end with “JOB DONE” (`job.sbatch.tmpl:9,12–23`). Archives use mutable `HEAD` after separately recording the commit (`submit.sh:14,18–19`).

**New verdict-changing bugs**

- **All three CN/LSPG steppers crash:** `rom.py:343` unpacks nine carry entries; `:378` initializes ten and `:372–373` returns ten. Tiny CPU queries reproduce `ValueError: too many values to unpack (expected 9)` for LSPG, Galerkin CN, and linearised CN. Ladder catches this and records invalid configurations (`ladder.py:93–98`), potentially disguising a programming failure as numerical failure.
- **Test reuse copies the wrong verdict:** `run_cell.py:688–690` reuses a complete result across targets/statistics without recomputing `sv` or `three_way`. A configuration first evaluated for a median target retains that verdict for later worst-error or different-target entries.
- **Incomplete timing matrices inflate S or crash:** invalid pairs are removed before max–min (`run_cell.py:543–545`). `finalize.py:218–226` then minimizes over the remaining competitors. CPU fixture: losing the fast competitor’s pair selects **S=4**; when a whole family is missing for the selected bank, `per_family` raises `ValueError: min() iterable argument is empty`.
- **Confirmation mixes accuracy and timing evidence from different finalists:** the test summary can obtain its S from an unqualified/nonmatching bank while `decide.py` checks another bank’s qualification (`run_cell.py:722–726`; `decide.py:52–59`).

**OVERALL: NEEDS-FIX**

1. Repair the reduced-step carry/trace contract and exercise all four steppers.
2. Enforce solver validity in confirmation, Q1, and Q2 comparison replays.
3. Recompute target/statistic verdicts when reusing test outputs; bind confirmation to the finalists supplying its S.
4. Require complete valid timing comparisons; gate sentinel, T-dbl, compilation, and timed-output failures on both cohorts.
5. Verify committed freeze provenance, running code/descriptors, and every decision-affecting artefact.
6. Correct side-status classification and gate required controls/diagnostics and attribution on valid solves.
7. Align cost probes with production and recertify remaining coverage under A6.
8. Make queue checks and batch failures fail closed; archive the exact recorded commit.