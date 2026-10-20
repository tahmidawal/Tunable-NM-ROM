**Overall verdict: NEEDS-RESTATEMENT.** A26 repairs round 17’s substantive traffic and memory-admission defects. The remaining issues concern extension accounting and explicit coverage consequences; they do not invalidate the corrected arithmetic.

References: `D` = `experiments/jcp-hr-baselines/DESIGN.md`; `F` = the supplied qualify worktree’s `experiments/jcp-qualify/runs/FRONT-2D.json`; `L` = the supplied speed3d worktree’s `experiments/jcp-speed3d/runs/GPU-LEDGER.json`.

1. **CORRECT — A26.1’s corrected arithmetic.**

   Recomputed using \(m=104448,\ N=32768,\ K=s=16384,\ I=19000\):

   | Quantity | Recomputed |
   |---|---:|
   | Append flops \(8mKs\) | \(2.24300372066304\times10^{14}\) |
   | Gradient flops \(2mNI\) | \(1.30056978432\times10^{14}\) |
   | Gradient bytes \(8mNI\) | \(5.20227913728\times10^{14}\) |
   | Append bytes + gradient bytes | \(1.417429401993216\times10^{15}\) |
   | Listed refactorisation flops | \(9.83066474446848\times10^{14}\) |
   | Listed total flops | \(1.337423824945152\times10^{15}\) |
   | Refactorisation share | 73.5045% |
   | Listed traffic / 4.8 TB/s | 4.9216 minutes |
   | Listed traffic / 2 TB/s | 11.8119 minutes |

   Thus “about 5 minutes,” “about 12 minutes,” and approximately 73% are defensible. These are partial-workload estimates, not runtime guarantees; A26 now says so and identifies 19,000 iterations as assumed rather than bounded.

   **Evidence:** `D:1393–1400`, `D:1459–1465`; actual iteration bound at `D:1233–1235`.

   **Fix:** No substantive correction required. Explicitly supersede A24’s remaining “1.2e14” gradient-flop figure with \(1.30057\times10^{14}\), and retain the exclusions beside every runtime projection.

2. **CORRECT — the revised memory acceptance predicate is conservative and sound.**

   Requiring the **entire process-cumulative peak** to be finite and at most 120 GB eliminates the previous false acceptance when an earlier phase had already established a higher peak. Missing telemetry and OOM fail acceptance.

   This certifies the executed path retrospectively. It does not establish beforehand that the path will fit, or certify an unvisited maximal-support shape. An inherited high peak can conservatively reject an otherwise feasible ECSW path.

   **Evidence:** `D:1465–1469`, replacing `D:1414–1416`; earlier pre-path language at `D:1351–1352`.

   **Fix:** Describe this explicitly as **post-path memory acceptance**, superseding A22.2’s pre-path pilot requirement. Distinguish “admission failed because cumulative telemetry exceeded the threshold” from proof that ECSW itself requires excessive memory.

3. **NEEDS-RESTATEMENT — extension membership is consistent, but its accounting trigger needs precision.**

   The permitted extensions agree: κ=8→16; 2D rank 384→512; no 3D rank extension; no p/r=8. A frozen screening-based decision inside the tuning job is compatible with Stage-A selection and later audited Stage-B membership. No separate job is scientifically necessary.

   However, A25 tests “elapsed C-search time (from the start of Stage A),” while A26 places other work between the HR grid and extension, including the limiting FOM comparator. Elapsed wall time since Stage A is therefore not automatically accumulated C-search. A25 also charges selection-relevant Stage-B work to C-search, although that future expenditure is absent from the extension trigger. The text does not clearly distinguish a provisional start threshold from a total allocation guarantee.

   **Evidence:** `D:657–659`, `D:716–718`, `D:1182–1188`, `D:1429–1445`, `D:1489–1492`; task-level accounting at `D:560–570`.

   **Fix:** Define the trigger using accumulated, classified C-search timers. State whether Stage-B expenditure is reserved before extension admission or whether the allocation is only a start threshold with subsequent overruns disclosed. Freeze and deduplicate extension requests from the original Stage-A front; do not recursively extend EXT members.

4. **CORRECT — the allocation magnitudes and category mapping are defensible as disclosed upper bounds.**

   Arithmetic gives:

   - 2D: \((13596+658)/3600=3.959444444\) GPU-h, or **1.979722222 per cell**.
   - speed3d g1–g3: \(1.7472+2.4092+3.6294=7.7858\) GPU-h.
   - 3D: \(15949/3600+7.7858=12.216077778\) GPU-h.

   Excluding `c3desc*` and quadrature escalation from C-search follows A2.5’s C-hr mapping. Whole-job totals containing inseparable construction, certification or overhead are upper bounds, not exact task-level search expenditure. A25 discloses that limitation. The independent V ledger and factor-1.25 wording still govern budget-comparability claims.

   **Evidence:** `D:560–570`, `D:676–685`, `D:788–791`, `D:1439–1445`; `L:26–43`, `L:56–64`.

   **Fix:** Print **3.959444 total / 1.979722 per cell**, or clearly label 3.960 as upward rounding. Retain “upper-bound allocation” and never equate these allocations with measured equal expenditure. I verified arithmetic only, not sacct provenance.

5. **CORRECT — the wall-limit bound protects the cap; it does not guarantee completion.**

   \[
   1.512+2+5+12+7.5=28.012\ \text{GPU-h},
   \]
   leaving **1.988 GPU-h** below the cap. This holds for the stated one-GPU allocations when Stage-B limits **sum** to at most 7.5 hours and additional attempts are charged before submission.

   A truncated mandatory item remains `not_run`; “never dropped” means it cannot be deliberately removed to manufacture completed coverage. Earlier governing provisions already require an **incomplete** cell when the registered core cannot fit. Missing verdict evidence cannot support a favorable conclusion.

   **Evidence:** `D:503–509`, `D:627–635`, `D:799–802`, `D:1475–1478`, `D:1494–1500`.

   **Fix:** Cross-reference those incomplete-cell consequences directly in A26.4, including Stage-B truncation. State that retries/requeues and extra batches consume the remaining ledger allowance. Also distinguish **execution order** from **first-dropped-first defer order** at `D:1490–1492`; those are different policies.

6. **CORRECT — picks after Stage B and expanded A–B–A coverage resolve the selection conflict.**

   A1.6 already permits selection on the full verdict cohort as an explicitly developmental, in-sample comparison. A26.5 expressly supersedes the earlier pre-verdict pick freeze while preserving audited membership freezing.

   Timing every verdict-Q1-qualified Stage-B member against every admitted X covers potential B_cost candidates outside the exploratory error-envelope filter. It preserves the uncapped absolute and X-specific screening fronts and provides timing for every possible role candidate.

   **Evidence:** `D:186–196`, `D:226–247`, `D:385–394`, `D:461–463`, `D:572–576`, `D:854–857`, `D:1502–1506`.

   **Fix:** No substantive change required. Preserve X-specific role selection, the six-case timing estimand, drift rejection, timed/scored equality and “developmental, in-sample, unadjusted” labels when implementing the post-job picks.

7. **CORRECT — A27 identifies exactly the admissible FRONT-2D members.**

   Filtering the actual **members** arrays by baseline mesh, LSPG, BE/CNR and Δt=0.005 yields precisely the four listed identities. Their rules are fib4181 and gauss128. The file’s SHA-256 matches A27’s prefix.

   The file’s embedded commit is `ef1a731…`; `2374fafe1` is the later freeze commit containing it. That is a provenance distinction, not a contradiction. Other configuration records appearing elsewhere in the JSON are not additional front members.

   **Evidence:** `D:1306–1312`, `D:1519–1527`; `F:4`, `F:2040`, `F:5397`, `F:23599`, `F:27630`.

   **Fix:** No membership correction required. Preserve the exact configuration identity and G-ours replay requirements; identify the embedded generation commit separately from the freeze commit. Retain the 3D diagnostic-only restriction.

8. **CORRECT — A26.7 repairs the design-audit precondition, but a clean design audit alone cannot release production.**

   The accepted-successor wording resolves A19’s stale reference to round 14. A26 also supplies appropriate design-level requirements for terminal evidence, quality propagation, saturation fallback, inventory checkpointing and staged comparator assets.

   Once this round’s restatements are applied, the **design component** can satisfy A19.4(a). A19.4(b–c) and A19.5 still independently require accepted updating-QR code/gate evidence, the regenerated final-algorithm resource artifact and amended smk2 acceptance. This read-only design audit does not establish those implementation conditions.

   **Evidence:** `D:1264–1269`, `D:1354–1358`, `D:1376–1378`, `D:1471–1500`, `D:1508–1513`.

   **Fix:** Record acceptance against exact design, code/gate, smoke-acceptance and resource-artifact revisions. Do not describe design acceptance as unconditional production readiness.

**Overall: NEEDS-RESTATEMENT, with no remaining substantive arithmetic or memory-predicate WRONG finding.** Clarify extension accounting and explicitly connect truncation to incomplete coverage before accepting this as A19’s clean successor audit.

No files changed, jobs submitted, GPU work performed, tests run or scratch created.