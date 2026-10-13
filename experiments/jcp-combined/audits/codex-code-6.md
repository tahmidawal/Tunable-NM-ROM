**NOT READY FOR BUDGET SMOKES.**

Reviewed read-only at `bdec8563a`, against DESIGN A1–A8. No files modified, project modules imported, or GPU jobs run. Validation used source inspection and isolated AST-extracted probes. Paths below are relative to `experiments/jcp-combined/`.

**Disposition of the previous 16 items**

| Item | Status | Evidence |
|---|---|---|
| 1 | **RESOLVED** | `cluster/stage.py:23,147` stages and invokes `c2bud.py`. |
| 2 | **RESOLVED** | `freeze.py:64` preserves the exact 2D factor; `make_budget_configs.py:26` retains it. |
| 3 | **RESOLVED** | `c3run.py:210–211` saves trajectory-specific case IDs; `score.py:270–274` and `budget3d.py:81–82` use them. |
| 4 | **RESOLVED** | `c2bud.py:144–187` checks finite distances and verified rows, tries the successor, and restores Gauss320 if both checks fail. |
| 5 | **PARTIALLY — MAJOR** | Main error components now have validity guards (`budget2d.py:174–177`, `budget3d.py:161–167`); 2D B3 requires complete coverage (`budget2d.py:254–257`). But 3D B2 checks only **control** validity, not baseline validity (`c3run.py:445–448`). The 2D orthogonal-pair diagnostic also remains numerical when `star` is unverified (`budget2d.py:182–183`). |
| 6 | **PARTIALLY — MAJOR** | Missing finer counterparts/cases now fail, failed convergence withholds attribution, and the smoke baseline is fixed-sweep BE (`budget3d.py:113–124,228–229`; `configs/smk3bud.json:23–29`). However, convergence inventory is still derived from existing archive keys (`budget3d.py:110`): deleting both sides of a required comparison silently removes the check; an empty inventory passes. |
| 7 | **PARTIALLY — MAJOR** | Grid reproduction now requires equality with a comparison count (`budget2d.py:278–282`, `budget3d.py:245–249`), but remains optional and its expected population comes from observed output. Production staging still checks frozen binding only **if present** (`cluster/stage.py:68`). `make_budget_configs.py:24–45` implements neither conditional fallback job. |
| 8 | **RESOLVED** | 3D now includes the orthogonal pair, triangle check, closure, context counts, resolution flags and analytic fixtures (`budget3d.py:168–171,203–240`). 2D reproduction moved outside the arm loop (`budget2d.py:93–111`). Separate defects below remain. |
| 9 | **PARTIALLY — MAJOR** | Failed verification/non-finite replay now suppresses solver distance (`analyze3d.py:113–118`). A **finite replay exceeding `0.1 × bar`** still leaves the comparator diagnostic available: the threshold affects only `eligible_H4` (`:122–123`), contrary to A3.4. |
| 10 | **RESOLVED** | `analyze3d.py:161–169` safely handles empty witness lists and includes witnesses in budget arms. |
| 11 | **RESOLVED** | Authenticated scores are consumed at `analyze3d.py:97`; H6 remains explicitly descriptive at `:183–185`. |
| 12 | **RESOLVED** | Missing ladder evidence prevents selection (`select3d.py:97–105`); exact timing slots and duplicate counts are checked (`:125–134`). |
| 13 | **PARTIALLY — MAJOR** | 2D now checks configured ROM/FOM inventories and FOM archive coverage (`score.py:165–196`). 3D still compares only archive versus rows (`:305–314`); deleting a row and its trajectory together escapes detection. Non-finite handling also crashes on serialized values—N7 below. |
| 14 | **RESOLVED** | `score.py:123–125` hashes actual 2D descriptors. `score.py:249–250` checks regenerated 3D cohort identity against the registry pins (`refs.json:50–51`). |
| 15 | **PARTIALLY — MAJOR** | 2D endpoint propagation is implemented (`analyze2d.py:113–118,169–176`), but H2b still uses the old scalar median threshold (`:302–303`). In 3D, `--frozen` is merely required as a nonempty argument (`analyze3d.py:64`): its file is never read, authenticated or applied; selection is recomputed at `:156–158`. |
| 16 | **RESOLVED** | `analyze2d.py:53–56` retains deterministic collapse of duplicate Pareto points. |

**Disposition of N1–N6**

| Item | Status | Evidence |
|---|---|---|
| N1 | **RESOLVED** | `budget2d.py:206–210` uses `dep_`/`ide_`; the arm dictionary survives through `:229,246`. |
| N2 | **RESOLVED** | `analyze3d.py:161` uses `(witnesses or [None])[0]`. |
| N3 | **RESOLVED** | `analyze3d.py:166–168` recovers exact timesteps from configured factors; `freeze.py:89` preserves the specs. |
| N4 | **RESOLVED** | `budget2d.py:93–111` reproduces each setting once. Extracted smoke inventory gives **108 rows and 1,944 comparisons per setting**, matching the gate. |
| N5 | **RESOLVED** | The mismatch is stronger and uses one designated fixture rather than an all-case minimum (`budget2d.py:180,227,270`; `budget3d.py:167,222,238`). Its measured defect must still pass. |
| N6 | **RESOLVED** | `offline.py:3–8` forces CPU, highest precision and x64, and asserts the backend. All six offline entry points import it before JAX-backed dependencies. |

**New findings**

**N7 — BLOCKER: non-finite job rows crash both budget analyzers instead of being accounted for.**

Both drivers clean non-finite scores into JSON `null` before saving (`c2bud.py:94`, `c3run.py:151`; corresponding vendor `clean` functions).

The readers do not handle the resulting `None`:

- [budget2d.py:110](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/budget2d.py:110) evaluates `abs(mine[j] - y)`.
- [budget3d.py:105](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/budget3d.py:105) evaluates `np.isfinite(y)`, including in its intended invalid-run exception.
- `score.py:158–160,284–286` has the same problem.

**Reproduced:** subtraction with `None` and `np.isfinite(None)` both raise `TypeError`. This is conditional on a non-finite rollout, not a claim that either smoke necessarily diverges. Such outcomes must remain analyzable under A8.5, especially for deliberately inadequate controls.

**Fix:** handle serialized nulls before arithmetic; record invalid/unavailable rows separately from finite reproduction certification.

**N8 — MAJOR: B2 measures the wrong quadrature comparison; 3D can additionally pass with an invalid baseline.**

[c2bud.py:198–201](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/c2bud.py:198) compares the control rule with the **deployed** rule. [c3run.py:437–448](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/c3run.py:437) does the same.

The registered B2 substitutes the control rule as \(r\) in \(\delta_{\rm quad}\), requiring comparison with \(r_\infty\).

**Counterexample:** normalized scalar fields `converged=0`, `deployed=0.0002`, `control=0.0003` give:

- Registered distance: `0.0003` — exceeds `0.00025`.
- Implemented distance: `0.0001` — does not.

Moreover, the 3D discrimination predicate checks `ctl[…]['elig']['all_valid']` but never the baseline’s validity. A finite baseline with damping saturation can therefore produce `discriminating=true`.

**Fix:** compare the control against the matching converged-rule trajectory, join cases explicitly, and require both trajectories valid for every control case.

**N9 — MAJOR: the supplied 3D smoke falsely counts case 2 as quadrature-limited.**

The selected arm runs on cases `0,1,2`, while its ideal, solver, form and time diagnostics run only on `0,1` (`c3run.py:398–414`).

[budget3d.py:196–201](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/budget3d.py:196) nevertheless includes all selected-arm cases in `largest`, drops unavailable components, and chooses among whatever remains.

For case 2, only `delta_quad` remains in that comparison. **An extracted-code probe returns `{'delta_quad': 1}` even for a quadrature distance of `1e-8`.** No competing component was measured.

The production configuration repeats this for the extra 32 finalist-only cases, potentially manufacturing 32 quadrature-limited classifications.

**Fix:** restrict budget classifications to the configured budget population, and report incomplete comparisons as unavailable. Keep all-case finalist qualification separate.

**N10 — MAJOR: 3D never evaluates B3 or withholds time attribution when it fails.**

[budget3d.py:224–232](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/budget3d.py:224) handles B2 and convergence withholding only. There is no B3 computation, despite calculating the baseline’s `delta_time` at `:150`.

Consequently, a non-discriminating time control does not withhold time attribution as A8.5 requires.

**Fix:** evaluate baseline `delta_time > 1e-3` on the complete, valid budget population and propagate non-discrimination into `withheld`.

**N11 — MAJOR: `score.py` cannot score a 2D budget output.**

[c2bud.py:224](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/c2bud.py:224) writes `WB_<setting>.npz` with individual trajectory keys. [score.py:134–135](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/score.py:134) unconditionally opens `W_<setting>.npz` and expects the grid archive layout.

Thus scoring `smk2bud` fails with `FileNotFoundError`; merely adapting the filename would still encounter incompatible archive and grid/FOM inventory assumptions (`score.py:166–183`). The budget analyzer’s reproduction loop does not supply the scorer’s negative controls.

**Fix:** support the budget archive and configured budget inventory explicitly in `score2d`, including partial check/control populations.

The previous unconditional analyzer crashes are fixed, but invalid-output handling and the demonstrated smoke classification/control errors still prevent acceptance.

**NOT READY FOR BUDGET SMOKES.**