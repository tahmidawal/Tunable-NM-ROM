<!-- codex -o output file was not written; final answer block copied verbatim from the codex stdout log -->
tokens used
34,610
Amendment 5 fixes B8 and B9 and closes B0’s false-PASS loophole, but B0’s early-STOP exception needs clarification.

| Item | Verdict | Finding / fix |
|---|---|---|
| A1 fixtures | CORRECT | Prerequisites and failure handling remain intact. |
| A2 curvature | CORRECT | Raw-derivative comparison remains dimensionally consistent. |
| A2 decision | CORRECT | Availability checks remain separate from descriptive curvature. |
| B0 | NEEDS-RESTATEMENT | Complete evidence now gates PASS and route (a); define when incomplete evidence proves a guard violation. |
| B2 Q1 fixture | CORRECT | Actual nonlinear checks and certification blocking remain intact. |
| B5 | CORRECT | Transitions, qualification and uncertified starts remain consistent with the revised Q1 scope. |
| B8 | CORRECT | Admission reserves hard ceilings, including outstanding jobs and retries; projections no longer enforce the cap. |
| B9 | CORRECT | Production-only tight comparisons, promotion checks and terminal-only tighter anchors reconcile the solve inventory. |

**B0 — remaining ambiguity ([lines 686–690](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-fronts/experiments/imp-fronts/DESIGN.md:686)).** “Observed violation on the available cases” must not mean comparing two incomplete maxima. Available arm/control maxima of 0.03/0.01 appear to violate the guard, but a missing control floor of 0.10 can reverse that conclusion.

**Fix:** permit early STOP only when the observed arm maximum exceeds twice the **complete eligible-population K1 maximum** (or a certified upper bound). Otherwise retain UNAVAILABLE. Keep the registered population minimum explicit.

No other regression found in the reviewed design. Read-only: no files changed, scratch created, jobs submitted, or lab-log writes.
