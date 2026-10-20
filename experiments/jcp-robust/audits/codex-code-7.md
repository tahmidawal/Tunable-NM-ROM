**Overall verdict: NEEDS-FIX.** Acceptance code should not yet be used to accept smoke outputs or release production.

Audited HEAD `1162f6c31`. No files changed, scratch created, smoke outputs read, or SSH/cluster commands run. Small CPU probes used disabled bytecode writing. Paths below are relative to `experiments/jcp-robust/`.

1. **Round 6 #1 — RESOLVED — CORRECT — MINOR.**  
   `smoke_accept.py:53–57` now checks reservation attempt, commit, GPU type/count, groups and rounded time limit against the registered specification and collected commit. The GPU-hour limit remains checked at `:51–52`. These match the fields written by `cluster/stage.py:184–185`.

   **Fix:** None for this finding.

2. **Round 6 #3 — NOT RESOLVED completely — NEEDS-FIX — BLOCKING.**  
   Exact-shape guards now prevent broadcasting in `e2analyze3d.py:91–97` and `select3d.py:64–71`. However, malformed evidence returns a correctly sized list of infinities rather than an integrity failure.

   In selection, agreement checking explicitly skips non-finite recomputed distances (`select3d.py:110–115`). `draw_qualified()` consequently returns a definite `False`, and selection continues to later candidates (`:133–142`). Missing/malformed evidence can therefore change the selected rule instead of stopping selection as unavailable.

   The H-acc-r path checks agreement only for production–tight distances (`e2analyze3d.py:138–141`). A malformed tighter or successor array can leave overall integrity true while changing qualification.

   **CPU counterexample:** malformed tighter coefficients produced eight infinities, **zero integrity mismatches**, and `(False, False, 'ok')`.

   **Fix:** Propagate shape failure explicitly. Mark integrity false and qualification unavailable; do not represent malformed evidence as an ordinary failed numerical threshold.

3. **Round 6 #6 — RESOLVED for missing pilot evidence — CORRECT — MINOR.**  
   `smoke_accept.py:247–269` now verifies the fetched pilot-bank hash, installs its decoder, requires all four coefficient arrays with exact mappings/shapes, recomputes distances, and checks certificate-row coverage and step counts. The evidence check is mandatory at `:270–271`; qualification uses recomputed quantities at `:274–280`.

   Legitimate, fully evidenced `own=False` remains acceptable, as A6.3 requires.

   **Fix:** None for the original omission. Finding 5 below concerns a new defect in this implementation.

4. **Round 6 #10 — RESOLVED — CORRECT — MINOR.**  
   Both target certificates now require target-run validity on the certification subset:
   
   - `e1analyze.py:107–110`
   - `e1analyze3d.py:96–101`

   The 3D expression also checks validity-list length. Production cases outside the certification subset remain separately represented.

   **Fix:** None for this finding.

5. **NEW — Pilot distance reduction can hide non-finite coefficients — WRONG — BLOCKING.**  
   The new reduction uses Python `max()` without validating finiteness (`smoke_accept.py:264`). A NaN after an earlier finite value can disappear from the maximum. `rec_ok` checks certificate coverage, not coefficient finiteness (`:266–271`).

   **CPU counterexample:** correctly shaped arrays containing NaN at the second compared state produced **eight zero distances**. With otherwise passing flags/certificates, the actual predicate returned `own=True`. The evidence gate would also pass.

   The other 3D analysis paths already use `_safe_max`; this new path reintroduces the earlier non-finite-reduction defect.

   **Fix:** Validate finiteness of required coefficient states and decoded distances. Non-finite evidence contradicting valid-run flags must fail integrity. Preserve legitimate finite, numerically unqualified pilot outcomes.

6. **NEW — E1 smoke can accept malformed auxiliary arrays — NEEDS-FIX — BLOCKING.**  
   The new E1 shape guard likewise returns infinities (`e1analyze3d.py:68–74`), while its distance-agreement loop ignores non-finite recomputed values (`:128–133`). Qualification becomes `False`, but integrity can remain true (`:237`).

   `smoke_accept.py:123–125` requires only that own-family labels are **defined**. Thus malformed tight/tighter evidence can satisfy that check as `False`, despite failing the required evidence-integrity gate. A singleton time axis is particularly relevant: matrix decoding can succeed before the analysis shape guard rejects it.

   **Fix:** Treat shape failures as integrity failures before qualification. Keep “valid evidence demonstrates unqualified” distinct from “evidence is malformed.”

7. **NEW — E2 2D aggregation does not bind all jobs to the requested bank — NEEDS-FIX — BLOCKING.**  
   `e2analyze.py:28–46` verifies each job against its **own** declared bank. But `main()` combines selection and verdict jobs without requiring matching checkpoint/rotation identities or matching the requested bank (`:180–215`). The extra 1024² records are merged while retaining the original bank metadata (`:200–203`).

   Consequently, a correctly hashed verdict job from another seed can supply one mesh or factor panel in a nominally single-bank analysis. H-acc-N compares those records without a bank check (`:50–71`). Cross-bank verdict aggregation trusts the resulting integrity flag (`e2verdicts.py:99–102`). The timing-bank check at `e2analyze.py:98–100` does not close this accuracy-job hole.

   **Fix:** Bind selection and every contributing accuracy/timing job to one expected checkpoint-plus-rotation identity and seed. Reject mismatches before merging records.

The smoke assessment below is based on code inspection and synthetic probes, **not actual smoke acceptance**:

| Smoke | Can complete, passing evidence reach ACCEPT? | Reject every missing/failed required control or gate? |
|---|---|---|
| **smk1b** | Yes; no positive-path blocker identified. | **No:** malformed auxiliary evidence can evade integrity rejection; finding 6. |
| **smk2** | Yes; selection and H-acc-N plumbing remain connected. | No additional smoke-specific bypass demonstrated. Production bank binding remains blocking; finding 7. |
| **smk3** | Yes, conditional on the fetched bank, archive and accounting availability. | **No:** non-finite pilot evidence can pass; finding 5. |
| **smk4** | Yes. | No additional bypass identified in its explicit E5 controls/gates. |
| **smk5** | Yes. | No additional bypass identified in its explicit H200 E5 controls/gates. |

E5 single-job acceptance still explicitly requires job status, populations, doubled-work control, estimator fixtures, reproduction and P-cross gates (`e5analyze.py:722–749`). G-E5.2 remains correctly descriptive. The pure E5 drift-control and inventory tests passed. Filesystem-writing tests and smoke wrappers were not executed under the read-only restriction.