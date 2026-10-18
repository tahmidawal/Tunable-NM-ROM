**Independent audit: the reported arithmetic is sound, but the protocol and retraction claims need qualification.** All 15 sampled numbers match their summary entries and cited artifacts. No reported cost ratio crosses jobs. The main findings are post-data gate changes, development-cohort reuse, an ineffective verification check, and incomplete retraction disclosure.

This audit was read-only. No files were created, changed, or deleted; no GPU work or jobs were run.

**Verified numbers**

Paths below are relative to `experiments/lshape/`. `S[i]` means zero-based entry `i` in [reports/summary.json](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-lshape/experiments/lshape/reports/summary.json).

Artifact abbreviations:

| Alias | File | Job |
|---|---|---|
| T | `artifacts/lsh02/result.json` | 3783786 |
| D03 | `artifacts/lsh03/result.json` | 3784662 |
| D04 | `artifacts/lsh04/result.json` | 3784663 |
| D06 | `artifacts/lsh06/result.json` | 3784910 |
| D07 | `artifacts/lsh07/result.json` | 3789568 |

For invocation-based entries, errors are maxima of `same_grid_error`; costs are `1000 × median(total_seconds)`, filtered by the stated mesh and subject. Percentages multiply stored errors by 100.

| # | Reported number | Summary entry and metric | Raw artifact key / calculation | Verified value |
|---:|---|---|---|---:|
| 1 | N=64 discretisation error: **1.2945%** | S[57], `discretisation_delta_vs_1024_worst` | D03: max `references[intervals=64].discretisation_delta_vs_fine` | 0.012944868335093 → **match** |
| 2 | `smooth_R512`, N=512 floor: **0.7093%** | S[67], `bank_floor_dev_worst` | T: `bank_arms[1].floors[1].floor_dev.worst` | 0.007092762944276 → **match** |
| 3 | Selected bank’s common-cohort floor: **1.6155%** | S[74], `bank_floor_common_worst` | T: `bank_arms[3].floors[0].floor_common.worst` | 0.016154732631240 → **match** |
| 4 | Enriched bank, N=512 floor: **0.6650%** | S[79], `bank_floor_dev_worst` | T: `bank_arms[4].floors[1].floor_dev.worst` | 0.006649750617333 → **match** |
| 5 | Corner-enrichment improvement: **1.067×** | S[67].value / S[79].value | T: smooth/enriched N=512 worst floors | 1.066620893389 → **match** |
| 6 | K=32 primary best-found development error: **3.1802%** | S[97], `best_found_dev_worst` | T: `head_arms[4].best_found_development.worst` | 0.031802251878827 → **match** |
| 7 | That head’s best-found/floor ratio: **4.095×** | S[99], `head_floor_over_bank_floor` | T: `head_arms[4].head_floor_over_bank_floor`; also recomputed from its errors | 4.094928731703 → **match** |
| 8 | Prose: validation worst-error range ends at **16.7%** | S[55], `best_found_validation_worst_max` | T: max `head_arms[].best_found_validation.worst` | 0.166614176986808 → **match** |
| 9 | N=64 sparse-direct cost: **1.282 ms** | S[156], `median_total_ms` | D03: `invocations[intervals=64,name=fom_splu].total_seconds` | 1.281905570067 ms → **match** |
| 10 | N=128 `pod64` worst error: **7.7013%** | S[455], `worst_same_grid` | D03: `invocations[intervals=128,name=pod64].same_grid_error` | 0.077012692308622 → **match** |
| 11 | N=256, M=257 primary solved error: **3.1814%** | S[535], `worst_same_grid` | D04: `invocations[name=neural_q0@head_sdf_R512_K32].same_grid_error` | 0.031813906217927 → **match** |
| 12 | Same primary: **96/96 stationary** | S[540], `stationary` | D04: sum `stationary` for that subject; all 96 have `reason=4` | **96/96 — match** |
| 13 | N=512 direct/Q64 cost ratio: **7.60×** | S[882].value / S[964].value | D07: median costs for `fom_splu` / `neural_q64@head_sdf_R512_K16` | 36.504633375444 / 4.800847033039 = **7.603790148743 — match** |
| 14 | Free-bank worst error: **0.7791%** | S[692], `worst_same_grid` | D06: `invocations[name=freebank@head_sdf_R512_K16].same_grid_error` | 0.007790957054213 → **match** |
| 15 | Free bank versus POD-256 speed: **1.54×** | S[814].value / S[695].value | D06: median costs for `pod256` / `freebank@head_sdf_R512_K16` | 3.987448522821 / 2.583266003057 = **1.543568690991 — match** |

Beyond this sample, independent aggregation found **zero discrepancies across 834 summary metrics and all 976 checked cells in the 122 timed-subject rows**. Every summary row’s `source_sha` matches its referenced artifact JSON.

**Mismatches and unsupported wording**

1. **“The development cohort … selected nothing” is false for the added M=1024 experiment.**  
   The report says this at lines 3 and 90. However, `checks/free_rung_M_sweep.py:19` explicitly loads `config.cohorts.development`, and DESIGN §A6 selects M=1024 after inspecting that cohort’s errors. For `sdf_R512`, `checks/free_rung_M_sweep.json` records:
   - M=513: `free_worst=0.06278163696244682`;
   - M=1024: `free_worst=0.007790957054212477`.

   The bank-selection rule remains separate and correctly followed. The **test-mode choice** used development results, so the blanket untouched-cohort claim needs narrowing.

2. **The report gives the wrong primary bank rank.**  
   Report line 432 says “both primaries have R=514.” T’s `selection.bank.R` and both primary heads’ `R_total` are **512**. The free-bank table also correctly says 512. Non-constructibility at M=257 remains true.

3. **The retracted attempt’s completion count is wrong, and its rerun did not reproduce the same selection.**  
   Report line 437 and DESIGN §A3 say seven of eight heads completed. `artifacts/lsh01/result.json.gz` contains **six completed `head_arms`**, and its log shows failure during the seventh. More substantively:
   - `lsh01`: `selection.bank.selected="smooth_R512"`, common worst **0.016413566587053962**;
   - `lsh02`: `selection.bank.selected="sdf_R512"`, common worst **0.01615473263124043**.

   All six banks’ final weight hashes differ between attempts. This contradicts §A3’s assertion that the rerun reproduces the completed arms. The cause is not established by this audit.

4. **The existing raw-number verification check passes without checking anything.**  
   In [checks/verify_report_2026-09-17.py](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-lshape/experiments/lshape/checks/verify_report_2026-09-17.py:44), `job` is an integer while summary `job_id` values are strings. The condition `r['job_id'] != job` skips **all 244 intended comparisons**. Consequently, the saved `report_values_recomputed_from_raw_invocations` result’s zero difference is not evidence of successful checking. My independent recomputation did check those values and found no discrepancies.

5. **The promised frontier output is incomplete.**  
   DESIGN §8 requires frontiers both with and without full-order models, using both cost definitions. The report displays the full-order-inclusive frontiers. The summary includes `nondominated_reduced_only` for complete-query cost, but there is no corresponding reduced-only secondary-cost output.

**Gate audit**

The scientific thresholds—5% solved error, 1% bank floor, 1.2× head/floor, and the falsification thresholds—were not lowered. Several numerical acceptance gates were changed after results arrived.

| Report verdict | DESIGN requirement | Audit finding |
|---|---|---|
| Reduced-model frontier membership | §8: worst same-grid error versus median cost, within mesh/job | **Matches raw data.** Reduced-member counts are 0, 3, 6, 6 at N=64,128,256,512 for M=257; 4 at N=256/M=1024. Both displayed cost definitions reproduce. |
| Selected bank; development ranking disagrees | §4: minimize worst common-cohort floor | **Correct for lsh02.** `sdf_R512` wins the common cohort; development ranking differs. The earlier attempt selected a different bank, as noted above. |
| Bank target passes | §4: N=512 development floor <1% | **Pass.** Best arm is 0.6650%; even the selected bank passes at 0.7697574711%. |
| Every head misses its target | §5: best-found/floor ≤1.2 | **Correct.** T’s seven `head_floor_over_bank_floor` values range from 2.240956 to 5.768152. |
| Falsification clause not met | §8: every smooth floor >1.5%, **and** enrichment improvement >1.5× | **Correct.** `smooth_R512` reaches 0.7093%; enrichment improves only 1.066621×. |
| Cell-success clauses 1–3 pass | §8: primary <5%, stationary, better than POD-K | **Pass.** At M=257: 3.1813906%, 96/96 reason-4 exits, versus POD-32 at 18.9497%. The additional M=1024 block also passes these clauses. |
| G-FOM-1/2/3 pass | §2: symmetry, independent assembly, positive eigenvalue and accuracy check | **Pass on recorded evaluation meshes.** Symmetry and assembly differences are zero; eigenvalue checks pass. No N=1024 gate record is supplied. |
| G-FOM-4 / “every G-FOM gate passes” | Originally residual ≤1e-12 on every case/mesh | **Pass only under amended §A7 for N=512; not under the original gate.** Details below. |
| G-FOM-5 pass | Tight iterative solutions agree within 1e-8 | **Pass.** Maximum recorded disagreement is 3.26352e-11. §A2 changed ILU-PCG to IC(0)-PCG before production solve results. |
| G-FOM-6 square fidelity | Relative floor agreement ≤1e-9 | **Pass.** `checks/gate-square.json`: JAX path 5.30004e-11; NumPy path 5.08386e-11. |
| Training audit passes | §9 independent numerical checks, as amended | Stored 19 checks pass, but basis and floor acceptance depend on **post-data §A3/§A4 changes**. |
| Solve audits pass | §9 saved-field/error checks | All four stored audits report 13 passes. Reference acceptance is subject to §A7 and the fine-reference coverage gap below. |
| Free rung is linear, head-independent to round-off, near its floor | §6 plus §A6/§A9 | **Supported.** Zero iterations; worst/floor 1.00318097; saved check reports field disagreement 4.061408e-14. Byte identity was correctly withdrawn. |
| Free bank dominates POD-256 | §8 dominance on worst error and complete cost | **Correct within lsh06:** 0.7790957% versus 0.9303368%, and 2.583266 versus 3.987449 ms. This is an adaptive M=1024 extension, with unequal model ranks. |

The changes requiring explicit flags are:

| Amendment | When changed | Original → revised gate | Affected verdict |
|---|---|---|---|
| **A3** | After lsh01 aborted | Absolute basis orthogonality error <1e-8 → error/√q <1e-8 | Accepted correction bases. In T, `head_arms[6].basis.orthogonality_error=2.1243365e-8` **fails the original**; scaled error `1.8776659e-9` passes. |
| **A4** | After auditing lsh02 | Relative floor difference ≤1e-8 → absolute difference ≤1e-6 | `bank_floors_reproduced`. Audit relative difference `5.2360169e-6` fails the old rule; absolute difference `4.7233144e-8` passes the new one. |
| **A7** | After lsh05 aborted | Reference residual ≤1e-12 → maximum of 1e-12 and a calculated round-off bound | G-FOM-4 and therefore cell-success clause 4. D07 cases **8, 12 and 14** fail the original threshold. |
| **A6** | After interim results and a development-cohort sweep | Added free-rung experiment at selected M=1024 | An adaptive configuration extension, rather than a changed numerical pass threshold. Its cohort reuse must be disclosed. |

For a concrete A7 example, D07 `references[case=12,intervals=512]` records:

- `same_grid_residual = 1.1827457296152875e-12`;
- `same_grid_residual_limit = 7.948451603138954e-12`.

The numerical rationale may be reasonable; it does not make this an unchanged preregistered pass.

There is also an **unamended verification-coverage gap at N=1024**. All solve artifacts record maximum `fine_residual=4.719972229429103e-12`; **31 of 32** fine references exceed the original 1e-12 threshold. `lsh_solve.py:207` records these residuals without asserting the gate, and `lsh_audit_np.py:326` discards the fine-reference residual when rebuilding the fields. In lsh07 they fit within the recorded round-off floors, but neither a fine-reference acceptance result nor the complete N=1024 operator-gate record is present. “Every gate on every mesh” is therefore too broad.

**Cross-job and GPU check**

**No cross-job cost ratio found.** The five explicit performance ratios recompute as follows:

| Ratio in report | Numerator / denominator, complete ms | Job | Recomputed |
|---|---:|---:|---:|
| N=128 direct / POD-64 | 2.475291956 / 2.293698955 | 3784662 | 1.079170× → 1.08× |
| N=256 direct / neural Q64 | 8.386203437 / 3.028258565 | 3784663 | 2.769316× → 2.77× |
| N=512 direct / neural Q64 | 36.504633375 / 4.800847033 | 3789568 | 7.603790× → 7.60× |
| N=256/M=1024 direct / free bank | 8.350772550 / 2.583266003 | 3784910 | 3.232641× → 3.23× |
| POD-256 / free bank | 3.987448523 / 2.583266003 | 3784910 | 1.543569× → 1.54× |

All jobs identify an **NVIDIA A100 80GB PCIe**. Logs place 3784662/3789568 on `pax050` and 3784663/3784910 on `pax049`; each ratio stays within its allocation. Sparse-direct subjects run on that node’s CPU, while reduced subjects run on its GPU—this is explicitly disclosed and follows the common host-input/host-output contract.

The M=257 and M=1024 timing blocks are kept separate. All subject/case groups retain three repetitions with matching field hashes.

**Retraction completeness**

**Incomplete against DESIGN; comparison against `checks/retractions.md` is unavailable because that file does not exist.**

| Amendment / correction | Coverage in the report |
|---|---|
| A1/A5/A8: independent-review substitution | Disclosed: self-audit substituted for unavailable Codex review. |
| A2: invalid ILU-PCG comparator replaced | Current comparator is correctly named IC(0), but the rejected ILU method and its failure are absent from the deviations record. |
| A3: lsh01 withdrawn and basis gate changed | Withdrawal disclosed. Completion count is wrong; the changed bank selection and failure to reproduce the first attempt are omitted. |
| A4: bank-floor audit threshold changed after failure | Disclosed in the audit discussion, though not consolidated with the retractions. |
| A6: free-rung M choice and separate timing block | Disclosed. Consequent development-cohort reuse is not reconciled with “selected nothing.” |
| **A7: lsh05 failed; G-FOM-4 changed; lsh07 substituted** | **Missing from the report’s deviations record.** This is the most consequential omission because the report labels clause 4 an unqualified pass. |
| A9: byte-identical free-rung claim withdrawn | Clearly corrected in the free-rung discussion. |
| A10: job/block indexing correction | Separate tables and job provenance are implemented; the potential overwrite is documented in DESIGN. |

No summary row cites withdrawn job **3780148** or failed job **3784664**. Their numerical results have not leaked into the reported tables.

**Three weakest claims and likely reviewer questions**

1. **“The development cohort selected nothing.”**  
   M=1024 was chosen using that cohort, and the nominally identical training rerun changed the winning bank.  
   **Reviewer:** Which conclusions survive on a fresh cohort after fixing M and the checkpoint, and why did the same-seed rerun change bank selection?

2. **“Pre-registered cell success: every G-FOM gate passes.”**  
   Clauses 1–3 hold, but clause 4 relies on a post-failure relaxation; the N=1024 reference is outside the implemented gate coverage.  
   **Reviewer:** Show original and amended outcomes side by side, justify the new bound independently, and explicitly certify the fine reference before calling the overall result a preregistered pass.

3. **“The learned bank beats the classical linear baseline on both axes.”**  
   The measured comparison is valid but narrow: free bank **R=512** versus POD **256**, with no POD-512 control. It concerns worst error; POD-256’s median error is **0.0185929%**, versus **0.3169820%** for the free bank. Some other frontier distinctions are also tiny: the N=256 smooth head’s median cost beats POD-128 by only **0.002190 ms**, with their order reversing in two of three repetition-level medians.  
   **Reviewer:** Does the win survive a matched-rank POD comparison, both worst and median error, and uncertainty estimates for frontier membership?