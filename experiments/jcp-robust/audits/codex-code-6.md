**Overall verdict: NEEDS-FIX. Acceptance code must not yet be used to accept smoke outputs or release production.**

Audited HEAD `448d7aa3e`. No files changed, scratch created, smoke results read, jobs submitted, or SSH/cluster commands run. Small CPU probes ran with bytecode writing disabled. References are relative to `experiments/jcp-robust/`.

1. **Round 5 #3 — NOT RESOLVED — NEEDS-FIX — BLOCKING.**  
   The registered job specification and regenerated driver inventories are now compared; shell arguments, environment and projections are checked (`smoke_accept.py:43–66`). This closes the empty-inventory counterexample.

   Reservation validation still checks only `time_limit_hours`. It ignores reservation `attempt`, `commit`, `gpu`, `gpus`, `time_limit` and `groups`, although staging writes these fields (`cluster/stage.py:184–185`).

   **In-memory counterexample:** a correct smk4 specification/inventory combined with a reservation naming an unrelated attempt, wrong commit/GPU, 99 GPUs and empty groups returned `ok=True`, provided `time_limit_hours=.5`.

   **Fix:** Compare the reservation’s identity and resource fields against the committed specification and collected commit. Keep the GPU-hour calculation.

2. **Round 5 #4 — RESOLVED — CORRECT — MINOR.**  
   The 2D path now restricts repetitions to 0–2 and cases to dev6, and vetoes subjects containing duplicate or unregistered keys (`e2analyze.py:108–128`). The 3D path applies duplicate, population and identity vetoes over **all** invocations before filtering (`e2analyze3d.py:179–192`).

   **Fix:** None for the cited defects.

3. **Round 5 #5 — NOT RESOLVED completely — NEEDS-FIX — BLOCKING.**  
   The substantive repairs are present: tighter cases must be 0–7, tight/tighter distances are recomputed, and stationarity comes from certificate rows with checked case coverage and step counts (`e2analyze3d.py:117–137`).

   However, coefficient **shapes** remain unchecked. `dist3()` subtracts arrays directly (`:91–93`); case mappings alone do not prevent broadcasting. For example, tighter coefficients shaped `(8,6,1)` broadcast against `(8,6,256)`. A small zero-array probe returned eight zero distances, rather than rejecting malformed evidence. The analogous subtraction in `select3d.py:64–67,98–104` has the same weakness.

   **Fix:** Require exact production, successor, tight and tighter coefficient shapes—case count × six states × registered width—before subtraction. Malformed evidence must make qualification unavailable and integrity fail.

4. **Round 5 #7 — RESOLVED for the cited E5 defect — CORRECT — MINOR.**  
   Single-job acceptance requires successful job status, populations, controls, reproduction and P-cross gates (`e5analyze.py:722–749`). The equivalence coverage hole referenced by this finding is repaired; see item 7 below.

   **Fix:** None for that defect. The shared resource-contract blocker in item 1 remains.

5. **Round 5 #10 — RESOLVED — CORRECT — MINOR.**  
   smk5 explicitly invokes H200 single-job E5 acceptance (`smoke_accept.py:261–268`). G-E5.2 is now included as descriptive evidence and excluded from `ALL` (`e5analyze.py:751–767`).

   **Fix:** None.

6. **Round 5 #11 — NOT RESOLVED completely — NEEDS-FIX — BLOCKING.**  
   smk3 now requires the registered JSON case identities and tighter-validity populations for both members of each rung pair (`smoke_accept.py:219–232`). The original one-tighter-case counterexample is closed.

   But pilot qualification still consumes serialized distances and aggregate stationarity (`:235–240`). It never checks the pilot’s saved coefficient arrays, their case mappings, or certificate-row coverage. The later NPZ access checks only the incumbent replay (`:243–257`).

   **In-memory predicate check:** complete-looking JSON lists with zero distances and stationarity returned `own=True` without any saved coefficient arrays or certificate rows. Removing or mis-mapping pilot arrays would not affect this acceptance path.

   **Fix:** Validate the pilot arrays and mappings, recompute its distances, and validate certificate coverage before evaluating the predicate. Preserve acceptance of a legitimate, fully evidenced `own=False` result.

7. **Round 5 #13 — RESOLVED — CORRECT — MINOR.**  
   E5 equivalence now requires exactly one passing record per registered scored subject, including the normalization subject (`e5analyze.py:135–141`; `e5common.py:217–224,235–246`). Missing, duplicate, anonymous and failed records block the job (`e5analyze.py:154–155`). Corresponding regression tests were added (`e5_tests.py:430–448`).

   **Fix:** None for the missing-equivalence defect.

8. **Round 5 #14 — RESOLVED for the cited empty-validity defect — CORRECT — MINOR.**  
   3D selection requires both target/check validity lists to match the case population and requires at least eight cases before certification (`select3d.py:76–79`). Empty lists no longer certify a target.

   **Fix:** None for that defect. The coefficient-shape issue is covered in item 3.

9. **Round 5 #15 — RESOLVED — CORRECT — MINOR.**  
   smk1b requires exactly the BE/CN-R arms (`smoke_accept.py:86`), explicitly checks the 3D GPU backend and completion marker, and requires the registered 3D configuration inventory (`:106–107`).

   **Fix:** None for the cited omissions.

10. **NEW — E1 target certification ignores failed target validity — WRONG — BLOCKING.**  
    In 2D, `tgt_cert` checks check-run validity, agreement and distance, but omits target-run validity (`e1analyze.py:107–108`). The 3D `tc_ok` has the same omission (`e1analyze3d.py:92–96`).

    smk1b acceptance requires those certificate booleans (`smoke_accept.py:90,116`), then requires only defined **own-family** labels (`:91–93,117–119`). It does not separately require valid target runs.

    **In-memory counterexamples using the actual certificate expressions:** all target-validity flags `False`, with valid checks and zero check distances, yielded `tgt_cert=True` and `tc_ok=True`.

    This violates A3.3’s requirement that both runs be valid on the certification subset.

    **Fix:** Include target validity on the registered certification subset in each certificate. For these smokes, that subset covers every case. Keep failures outside a production certification subset separate, as A4.2 requires.

For the requested smoke assessment, the positive-path column is based on code inspection, not acceptance of actual outputs:

| Smoke | Can complete passing evidence reach ACCEPT? | Reject every missing/failed required control or gate? |
|---|---|---|
| **smk1b** | Yes; no demonstrated positive-path blocker. | **No:** failed target validity can escape certification; reservation identity is unchecked. |
| **smk2** | Yes; the A6.3 selection-to-H-acc-N path remains connected. | **No:** reservation identity is unchecked. |
| **smk3** | Yes, conditional on archive availability and accounting lookup. | **No:** pilot raw-array/certificate evidence can be absent; reservation identity is unchecked. |
| **smk4** | Yes; explicit E5 controls and gates are connected. | **No overall:** shared reservation defect. The cited E5 equivalence hole is closed. |
| **smk5** | Yes; H200 E5 acceptance is explicit. | **No overall:** shared reservation defect. The cited E5 equivalence hole is closed. |

Pure E5 drift-control, empty-reduction and inventory tests passed. The filesystem-writing E5 suite and smoke wrappers were not run under this read-only audit.