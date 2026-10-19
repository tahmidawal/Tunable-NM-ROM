**Overall verdict: NEEDS-FIX. Acceptance code must not yet be used to accept smoke outputs or release production.**

Audited HEAD `76b6bbac3`, including the three response commits and A6/A7. No files changed, scratch created, smoke outputs read, jobs submitted, or SSH/cluster commands run. CPU probes used in-memory fixtures with bytecode writing disabled.

References below are relative to `experiments/jcp-robust/`. Resolved findings are marked MINOR to indicate that they leave no blocker for the cited defect.

1. **Round-4 #2 — RESOLVED for the enumerated defects — CORRECT — MINOR.**  
   Independent 2D smoke populations now exist (`e1analyze.py:36,76–82`). E1 3D checks target-array lengths, registered populations and successor/tight/tighter mappings (`e1analyze3d.py:85–101,118–120`). Selection checks certificate case identities and expected lengths (`select3d.py:88–93`).  
   **Fix:** None for those specific defects. A separate selection target-validity hole remains in finding 14.

2. **Round-4 #4 — RESOLVED — CORRECT — MINOR.**  
   Smoke acceptance passes `--output-root` to the selection subprocess (`smoke_accept.py:143`); selection installs it before resolving the bank (`select2d.py:25–29`). Runtime checkpoint and rotation hashes are checked by `e2bank.py:24–33`.  
   **Fix:** None.

3. **Round-4 #5 — NOT RESOLVED — NEEDS-FIX — BLOCKING.**  
   A6.1 legitimately replaces smoke-derived certificates for bootstrap smokes. Staging now records driver execution keys and shell arguments/projections (`cluster/stage.py:119–128`), and collection retains these artifacts (`cluster/collect.py:63–65`).

   However, acceptance validates only inventory **names**, not their contents (`smoke_accept.py:43–47`). Empty `{}` entries pass; execution keys, counts, shell arguments and projections are never compared. It also derives the required step list from the collected specification without binding it to the registered smoke specification.

   **In-memory counterexample:** empty payloads for every listed inventory entry returned `ok=True`.

   **Fix:** Validate the registered job specification, regenerate and compare driver inventories, compare shell arguments/environment/projections, and bind reservation identity and GPU-hours to that specification.

4. **Round-4 #6 — NOT RESOLVED — NEEDS-FIX — BLOCKING.**  
   Missing equivalence, failed fused identities, rotation mismatch and duplicate counting are improved (`e2analyze.py:98–122`; `e2analyze3d.py:165–172`).

   Remaining holes:
   - Neither path restricts repetitions to registered repetition IDs.
   - The 3D path silently skips duplicates instead of rejecting contradictory evidence; a later duplicate containing an identity failure is skipped before its flags are examined (`e2analyze3d.py:167–171`).

   **In-memory counterexample:** otherwise valid 2D timing rows using only `rep=99` and `rep=100` produced front state `member`.

   **Fix:** Validate registered subject/case/repetition keys before filtering. Reject duplicates and apply identity vetoes over all records before constructing costs.

5. **Round-4 #7 — NOT RESOLVED — WRONG — BLOCKING.**  
   Production/tight/gl80 mappings and serialized-distance lengths are checked (`e2analyze3d.py:110–123`). But H-acc-r still trusts serialized tighter-case membership, tighter validity and tight/tighter distances (`:53–61`). It neither requires cases 0–7 nor recomputes their tighter distances. Stationarity also comes from serialized aggregate counts.

   **In-memory counterexample:** a 64-case record with `tighter_cases=[]`, `adt_tighter_valid=[]` and `adt_tight_tighter=[]` qualified all **64 cases**.

   **Fix:** Require the registered tighter population, validate coefficient shapes/mappings, recompute tight/tighter distances, and validate stationarity certificate coverage before evaluating H-acc-r.

6. **Round-4 #8 — RESOLVED under A6.3’s revised smoke scope — CORRECT — MINOR.**  
   Selection’s actual rule now controls downstream H-acc-N row construction, with explicit unmatched/no-pick handling (`smoke_accept.py:145–162`). The pilot-bank 3D predicate is invoked (`:201–210`). A6.3 explicitly postpones the complete generated-job chain until incumbent production.  
   **Fix:** None for the original disconnected-selection defect. The pilot predicate’s evidence population remains defective under finding 11.

7. **Round-4 #9 — NOT RESOLVED overall — NEEDS-FIX — BLOCKING.**  
   The repaired single-job E5 path explicitly checks status, controls, populations and gates, and exits nonzero on failure (`e5analyze.py:716–746,761–769`). Nevertheless, its status can still pass with missing subject-specific equivalence evidence; see finding 13.  
   **Fix:** Retain the explicit checks and close the remaining evidence-coverage hole.

8. **Round-4 #12 — RESOLVED for the enumerated defects — CORRECT — MINOR.**  
   E5 populations now come from independently registered contracts (`e5common.py:217–231`; `e5analyze.py:114–129`). G-E5.1 checks reference hashes against the supplied P1 archive listing, rejects duplicate case identities, requires registered case coverage, and checks exact coefficient shapes before subtraction (`e5analyze.py:475–544`). Smoke callers use the canonical P1 archive path (`smoke_accept.py:233–235,245–247`).  
   **Fix:** None for the cited population, archive-check, duplicate and broadcasting defects.

9. **Round-4 #32 — RESOLVED for the cited NaN-reduction defects — CORRECT — MINOR.**  
   `e1stats.num()` maps non-finite values to infinity (`e1stats.py:39–44`), and qualification uses it before maxima (`:142–144`). The three changed 3D distance routines use non-finite-safe maxima (`e1analyze3d.py:28–30,70`; `select3d.py:30–32,67`; `e2analyze3d.py:33–35,94`).

   **Repeated counterexample:** `tight=[0, NaN]` now returns `(False, False, 'ok')`, rather than qualifying.  
   **Fix:** None for that defect.

10. **Round-4 #33 — RESOLVED for required H200 E5 acceptance — CORRECT — MINOR.**  
    A7 moves the panel into registered job `smk5`; its acceptance invokes single-job E5 analysis with expected type H200 and requires successful completion (`smoke_accept.py:230–237`). It is therefore no longer silently omitted.

    **Remaining MINOR omission:** the single-job smoke report still does not include descriptive G-E5.2 (`e5analyze.py:716–746`).  
    **Fix:** Report G-E5.2 separately without making it an acceptance veto.

11. **Round-4 #34 — NOT RESOLVED overall — NEEDS-FIX — BLOCKING.**  
    **E5 portion resolved:** doubled-work subject/case/repetition coverage is checked exactly (`e5analyze.py:204–230`). Removing all but one doubled row now returns `ALL=False`.

    **smk3 portion incomplete:** both members of every rung pair are now checked, but only production/tight validity **lengths** are required (`smoke_accept.py:195–200`). Case identities and complete tighter evidence are not checked. The pilot predicate sets its expected tighter count from the supplied distance list (`:204–209`).

    **In-memory counterexample:** eight production cases with only **one** tighter case returned `own=True`; the smoke’s “label defined” check therefore passes.

    **Fix:** Require exact registered case identities and all eight pilot tighter cases, with matching saved-array mappings and lengths. Preserve the distinction between a legitimate `own=False` outcome and incomplete evidence.

12. **Round-4 #35 — RESOLVED for conflicting bank aliases — CORRECT — MINOR.**  
    Before replacing an existing bank registry entry, `install()` now requires matching resolved checkpoint path and hash (`e2score3.py:40–43`). The union also retains setting-to-bank/width consistency checks (`:20–30`).  
    **Fix:** None for the cited overwrite defect.

13. **NEW — E5 missing equivalence evidence can pass — WRONG — BLOCKING.**  
    `equivalence_ok` requires merely a nonempty list whose surviving entries pass (`e5analyze.py:135`). It does not require one record for each registered scored subject. The new population checks cover triplets and doubled rows, not equivalence records.

    **In-memory end-to-end counterexample:** an otherwise complete five-case/two-repetition fixture with only `equivalence=[{"passed": true}]` returned `status='ok', ALL=True`. No subject-specific equivalence evidence existed. The test fixture itself uses this anonymous record (`e5_tests.py:81–84`).

    **Fix:** Require exact subject coverage and uniqueness for equivalence evidence, including the normalization subject. Update positive fixtures to carry that evidence, then test deletion of each required record.

14. **NEW — 3D selection target certification passes empty validity lists — WRONG — BLOCKING.**  
    `select3d.py:76–79` evaluates `all(gl56_valid[:8])` and `all(gl64_valid)` without validating their lengths. Empty lists therefore certify a target when coefficient distances pass. That result reaches cross-family qualification and can turn a diagnostic pick into `selected` (`:124–133`).

    **In-memory predicate check:** eight zero check distances plus both validity lists empty returned `True`.

    **Fix:** Require exact registered target/check validity populations and mappings before certification. Missing flags must make target certification unavailable.

15. **NEW — smk1b can omit required arm checks and GPU evidence — WRONG — BLOCKING.**  
    - Acceptance loops over surviving 2D arms (`smoke_accept.py:65–74`). Analysis builds those arms from supplied configuration metadata (`e1analyze.py:361–375`). Removing the CN-R arm removes all its integrity, reproduction and target checks without adding a failure.
    - The 3D acceptance block never requires `backend == 'gpu'` (`smoke_accept.py:82–97`). E1 3D integrity checks completion and mismatches, but not backend (`e1analyze3d.py:232`).

    **Fix:** Require the exact registered BE/CN-R arm inventory before iterating, and explicitly check the 3D job’s GPU backend and completion evidence.

For the requested smoke assessment, “positive path” below means code-path inspection, not acceptance of actual smoke results:

| Smoke | Can complete, passing evidence reach ACCEPT? | Reject every missing/failed required control or gate? |
|---|---|---|
| **smk1b** | Yes; no remaining demonstrated positive-path blocker. | **No:** missing arms/GPU evidence and inadequate resource validation. |
| **smk2** | Yes; the subprocess bank-root failure is repaired, and A6.3’s selection branches are connected. | **No:** resource evidence is not fully validated. |
| **smk3** | Yes, conditional on archive availability and successful live accounting lookup. | **No:** tighter populations and case identities can be incomplete; resource validation is inadequate. |
| **smk4** | Yes for the in-memory E5 positive fixture; the full wrapper was not run. | **No:** subject-specific equivalence evidence can disappear. |
| **smk5** | The explicit H200 single-job path exists; no demonstrated positive-path blocker. | **No:** the same E5 equivalence and resource-validation holes apply. |

The pure E5 drift, empty-reduction and inventory tests passed. These repairs are substantive, but the demonstrated false acceptances keep the verdict **NEEDS-FIX**.