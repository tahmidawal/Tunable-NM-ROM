**Overall verdict: NEEDS-FIX. Acceptance code must not yet be used to accept smoke outputs or release production.**

Audited HEAD `2c91003cc`, including both round-three response commits. No files changed, scratch created, smoke outputs inspected, jobs submitted, or SSH/cluster commands run. CPU probes used in-memory fixtures with bytecode writing disabled.

References below are relative to `experiments/jcp-robust/`. Closed findings are marked MINOR to distinguish them from remaining blockers.

1. **Round-three #1 — RESOLVED — CORRECT — MINOR.**  
   Target-check disagreement now enters `mismatches` (`e1analyze3d.py:81`) and invalidates integrity (`:218`). The verdict consumer gives integrity precedence (`e1verdicts.py:87–94`).  
   **Fix:** None for the original disagreement defect. Incomplete target-check arrays remain covered by #2.

2. **Round-three #2 — NOT RESOLVED — NEEDS-FIX — BLOCKING.**  
   Production populations, expected certificate lengths, and unique 3D certificate case identities improved (`e1analyze.py:101–122`; `e1analyze3d.py:90–121`). The three-draw smoke no longer fails the production draw-count requirement.

   Remaining holes:
   - `e1analyze3d.py:81–83` uses `zip` and `all` without requiring `dist_check` and `check_valid` to contain every check case. Empty serialized lists can leave target certification true when recomputed distances pass.
   - Smoke cases and tighter-case populations still come from supplied output/configuration metadata (`e1analyze.py:71–76`; `e1analyze3d.py:78–93`), rather than independently registered smoke contracts.
   - E1 3D successor and tight coefficient arrays are compared positionally without checking their saved case mappings (`e1analyze3d.py:102–108`).
   - Selection stationarity still accepts the right *number* of certificate rows without checking unique case identities or expected certificate lengths (`select3d.py:83–85`).

   **Fix:** Validate exact identities, lengths, and shapes for every evidence role against independent production/smoke contracts before reducing.

3. **Round-three #4 — RESOLVED for the cited exceptions and replay-NaN defects — CORRECT — MINOR.**  
   The per-configuration handler now catches `OSError`, and E1 replay explicitly rejects non-finite new coefficients (`e1analyze3d.py:202–205,224–226`). Smoke replay checks both arrays for finiteness (`smoke_accept.py:184–187`).  
   **Fix:** None for those specific defects. Other non-finite reduction paths remain blocking under finding 32 below.

4. **Round-three #11 — NOT RESOLVED — WRONG — BLOCKING.**  
   The collected-bank override fixes in-process analysis, but does not reach selection:
   - `smoke_accept.py:109` sets the Python global `e2bank.OUTPUT_ROOT`.
   - `:125` launches a fresh interpreter for `select2d.py`.
   - That interpreter initializes `OUTPUT_ROOT=None` (`e2bank.py:20`), then asserts that it is present for `SMOKE-RUNTIME` banks (`:30–31`; `select2d.py:27`).

   An in-memory probe reproduced the assertion:  
   `runtime-hashed smoke bank needs the collected output and the recorded hashes`

   The 3D replay archive hash check is repaired (`smoke_accept.py:174–187`).  
   **Fix:** Pass the collected output root explicitly into the selection entry point and bank resolver.

5. **Round-three #12 — NOT RESOLVED — NEEDS-FIX — BLOCKING.**  
   E5 inventory support, production code-tree binding, and collection retention are implemented (`certify.py:124–151`; `cluster/stage.py:66–81`; `cluster/collect.py:63–65`).

   However, smoke staging explicitly generates **no certificate**, saves only inventory counts, and omits shell training steps (`cluster/stage.py:113–124`). This does not satisfy A4.2’s every-job inventory/certificate requirement. `smoke_accept.py` never validates these artifacts.

   **Fix:** Implement and validate the smoke resource contract, including training steps and execution identities. If initial smokes require a bootstrap reservation exception, record that explicitly in the design rather than silently substituting counts for the required artifacts.

6. **Round-three #23 — NOT RESOLVED — WRONG — BLOCKING.**  
   The added memory/count and invocation-identity checks help, but the complete timing contract is still not enforced:
   - Missing equivalence records pass: only explicitly failed records enter `eq_fail` (`e2analyze.py:104`).
   - Failed fused identities are ignored; the veto examines only `invocations` (`:105–111`), although `dtime.py:297–299` records fused identity failures.
   - Duplicate repetitions count toward the minimum (`e2analyze.py:115`; `e2analyze3d.py:156`).
   - 2D bank binding compares checkpoint hashes but omits rotation identity (`e2analyze.py:98`).

   **CPU counterexample:** a panel with duplicate `rep=0` rows, no equivalence evidence, a failed fused identity, and a mismatched rotation produced `state='member'`.

   **Fix:** Require complete subject equivalence coverage, all-path identity vetoes, unique registered repetition keys, finite positive ratios, and checkpoint-plus-rotation binding before assigning costs.

7. **Round-three #24 — NOT RESOLVED — NEEDS-FIX — BLOCKING.**  
   The finite-offline/`None`-serialized comparison is repaired (`select3d.py:97–102`; `e2analyze3d.py:108`). Selection also checks production/successor/tight coefficient case mappings (`select3d.py:68–94`).

   But the E2 H-acc-r path still compares production, tight, and gl80 arrays positionally without validating their saved mappings (`e2analyze3d.py:102–110`). Its serialized-distance comparison uses `zip` without requiring equal lengths (`:108`), so an empty `dist_adt` list produces no disagreement. Tight/tighter qualification remains based on serialized values (`:49–56`).

   **Fix:** Apply the complete mapping, shape, length, and recomputation checks to the H-acc-r path as well as selection.

8. **Round-three #26 — NOT RESOLVED — NEEDS-FIX — BLOCKING.**  
   Selection is now invoked and H-acc-N is calculated, but they remain disconnected:
   - Acceptance reads only selection’s outcome string (`smoke_accept.py:125–130`).
   - It then hard-codes `fib4181` and `sel` for H-acc-N (`:132–134`).
   - `configs/smk2u.json` still hard-codes the deployed evaluation rule.
   - `smk3` never invokes the selection procedure (`smoke_accept.py:140–188`).

   The pilot-bank population is correctly expanded to eight cases (`configs/smk3q.json:123–141`).

   **Fix:** Feed the actual selection result into downstream evaluation/verdict construction, including explicit none/unavailable handling. Exercise the 3D rung-selection path on the pilot bank.

9. **Round-three #27 — NOT RESOLVED overall — NEEDS-FIX — BLOCKING.**  
   The three specific implementation defects are repaired:
   - Single-type analysis avoids the H200 lookup (`e5analyze.py:539–540,612–624`).
   - Smoke mode explicitly checks job status, controls, and gates (`:658–682`).
   - Both entry points exit unsuccessfully when acceptance fails (`:703`; `smoke_accept.py:212`).

   An in-memory complete E5 fixture returned `ALL=True`; removing `G-FR` returned `ALL=False`. Nevertheless, the overall gate remains unreliable because of findings 4, 8, 33, and 34.

   **Fix:** Retain the new explicit checks and complete the missing acceptance coverage.

10. **Round-three #28 — RESOLVED for the missing-width decoder map — CORRECT — MINOR.**  
    Analysis installs both jobs’ settings (`e2analyze3d.py:83`), and `install_union()` retains the union (`e2score3.py:20–30`). This repairs the missing W768/W1024 lookup.  
    **Fix:** None for the original missing-width defect; bank-registry consistency remains a separate issue under finding 35.

11. **Round-three #29 — RESOLVED — CORRECT — MINOR.**  
    Only `outcome == 'selected'` becomes a definitive rule label. Diagnostic picks become `None` in both dimensions (`e2verdicts.py:50–52,70–73`).  
    **Fix:** None.

12. **Round-three #30 — NOT RESOLVED — NEEDS-FIX — BLOCKING.**  
    P-cross gates and G-E5.1 now govern job status before pooling (`e5analyze.py:118–133`). Missing `--p1ref` fails reproduction (`:448–450`).

    Remaining evidence holes:
    - G-E5.1 derives its supposedly registered subject/case population from the supplied panel configuration (`:451–454`).
    - Reference files have no owning-lane hash/provenance verification (`:465–483`).
    - Duplicate case identities are silently collapsed into dictionaries (`:471–472`).
    - Shapes are not checked before subtraction, allowing NumPy broadcasting (`:477–481`).

    An in-memory probe accepted an arbitrary, unpinned reference containing the supplied matching arrays.

    **Fix:** Bind reproduction to independently registered subjects/cases and pinned reference hashes; require unique identities and exact expected shapes.

13. **Round-three #31 — RESOLVED for the listed omissions — CORRECT — MINOR.**  
    The training record is mandatory, device memory must be finite, host headroom is checked, and checked rung validity lists have nonzero expected lengths (`smoke_accept.py:144–171`).  
    **Fix:** None for the original omissions. Prefer collected accounting evidence over live `sacct` during acceptance so the decision remains reproducible after accounting retention expires. Additional rung coverage is addressed below.

32. **NEW — WRONG — BLOCKING: non-finite distances can still qualify.**  
    `e1stats.draw_qualified()` uses Python `max` without rejecting non-finite elements (`e1stats.py:139–148`). A later NaN can be ignored.

    **CPU counterexample:** otherwise valid evidence with `tight=[0.0, NaN]` returned:
    ```text
    (True, True, 'ok')
    ```

    The 3D distance routines also use Python `max` over time states (`e1analyze3d.py:63–65`; `select3d.py:59–62`; `e2analyze3d.py:87–89`), and some agreement checks explicitly skip non-finite offline values.

    **Fix:** Validate every contributing coefficient/distance array for shape and finiteness before reductions. Convert non-finite evidence into failure/unavailability rather than relying on `max` ordering.

33. **NEW — WRONG — BLOCKING: `smk1` ignores its H200 E5 panel entirely.**  
    `jobs/smk1.json` now includes `e5time`, but `smoke_accept.smk1()` returns after E1 checks (`smoke_accept.py:34–81`). No H200 C-E5.1 or G-E5.1 check is performed. A missing or failed E5 panel therefore cannot change its acceptance result.

    G-E5.2 is also not reported through smoke acceptance. Under A2.4 it is **descriptive/non-blocking**; failure should be reported, not used as an acceptance veto.

    **Fix:** Invoke single-job E5 acceptance on the H200 panel and conjoin its required checks with E1 acceptance; report G-E5.2 separately.

34. **NEW — WRONG — BLOCKING: required control evidence can disappear without rejection.**
    - **E5:** `c_e5_1()` builds its subject population from surviving rows (`e5analyze.py:186–200`). It never requires the configured doubled subjects/cases/repetitions.  
      **End-to-end in-memory counterexample:** reducing an otherwise complete five-case/two-repetition fixture to **one doubled-work row** still produced `status='ok', ALL=True`.
    - **smk3:** acceptance checks only the first rung of each pair (`smoke_accept.py:168–172`). It does not require the gl80 heavy-cell partner, lat16384 successor, or pilot gl32 partner to have complete valid evidence, nor recompute the pilot selection decision.

    **Fix:** Validate explicit required evidence inventories before evaluating controls. Require both members of each smoke rung pair and the applicable anchor/selection evidence.

35. **NEW — NEEDS-FIX — BLOCKING: the decoder union validates aliases, not bank identity.**  
    `install_union()` checks only `(bank_name, width)` for repeated setting names (`e2score3.py:25–28`). Each `install()` overwrites `C3.BANKS[name]` with the next configuration’s path/hash (`:35–40`). Two jobs using the same bank alias with different checkpoint hashes are therefore not rejected; earlier coefficients can be decoded through the later bank.

    **Fix:** Validate a single consistent checkpoint identity across both jobs, selection evidence, and scores before mutating decoder registries.

**Smoke acceptance assessment**

| Smoke | Would accept complete, passing evidence? | Would reject every missing/failed required control? |
|---|---|---|
| **smk1** | E1 path’s previous three-draw obstacle is repaired. | **No.** H200 E5 checks are absent; incomplete evidence holes remain. |
| **smk2** | **No, as written.** Selection’s fresh process lacks the collected-bank root. | **No reliable end-to-end decision.** Selection does not drive evaluation, and resource evidence is unchecked. |
| **smk3** | Potentially, with available accounting/archive evidence. | **No.** Required partner rungs and the selection path are unchecked. |
| **smk4** | **Yes for the complete synthetic fixture tested.** The former single-type crash is fixed. | **No.** Removing almost all doubled-work evidence still yields acceptance; reproduction provenance remains unchecked. |

The pure E5 drift test and E5 inventory test passed. These checks establish useful repairs, but the demonstrated false acceptances and `smk2` execution failure keep the overall verdict **NEEDS-FIX**.