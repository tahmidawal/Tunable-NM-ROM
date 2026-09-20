**Audit result:** All 15 sampled values are numerically consistent with their available raw sources. Two have incomplete JSON provenance. The H-ORACLE failures and exploratory R-LADDER failure stand. No cost ratio crosses jobs or GPUs. Retraction disclosure is incomplete, and Phase-1 certification depends on a gate changed after its measurements were observed.

No files were changed; no GPU work or jobs were run.

Paths below are relative to `experiments/ns2d/`:

- **R** = [reports/2026-09-17-ns2d.md](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-ns2d/experiments/ns2d/reports/2026-09-17-ns2d.md)
- **S** = [reports/summary.json](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-ns2d/experiments/ns2d/reports/summary.json); array indices are zero-based.
- **A(nsXYZ)** = `artifacts/nsXYZ/result.json`.
- **D** = [DESIGN.md](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-ns2d/experiments/ns2d/DESIGN.md).

**Verified numbers**

Raw values below are shortened for readability. Comparisons used full stored precision and the report’s rounding.

| # | Report location and value | Summary entry or calculation | Artifact key and raw value | Result |
|---|---|---|---|---|
| 1 | R:23, Laplacian error `3.516e-15` | `S[0].value` | A(ns101), job **3780151**: `gates["F-LAP_N64"].rel = 3.51629027354e-15` | Match |
| 2 | R:56, TG continuum error `1.584e-04` | `S[37].value` | A(ns101): `gates["F-TG_N128"].cont_rel = 0.000158379271261` | Match |
| 3 | R:59, TG order `2.004` | `log2(S[37].value / S[45].value)` | A(ns101): `gates["F-TG-ORDER"].orders[1] = 2.00448962986` | Match; derived summary value |
| 4 | R:82, independent implementation discrepancy `9.136e-16` | `S[78].value` | A(ns101): `gates["F-INDEP"].worst_rel = 9.13561806313e-16` | Match |
| 5 | R:94, training CFL `1.23` | `S[106].value` | A(ns101): `data.train_N256.max_cfl = 1.22595741470` | Match |
| 6 | R:139, oracle median `2.024e-01` | `S[155].value` | A(ns203), job **3787319**: `oracle["256"].oracle_median = 0.202417228629` | Match |
| 7 | R:3, K=32 POD/oracle ratio `1.15` | `S[288].value / S[285].value` | A(ns204), job **3787320**: `gates["H-ORACLE_N256"].ratio_podK_over_oracle = 1.15301738157` | Match |
| 8 | R:290, cohort value discrepancy `3.602e-16` | **Absent** | A(ns302), job **3808495**: `gates["B-DATA_dev_N256"].value_worst_rel = 3.60181660586e-16` | Numerically matches; summary gap |
| 9 | R:364, three-mode oracle median `5.282e-02` | `S[429].value` | A(ns303), job **3808498**: `oracle["256"].oracle_median = 0.0528230841330` | Match |
| 10 | R:490, retracted bank rank `128` | `S[596].value` | A(ns202), job **3783797**: `gates["B-ORTH_N256"].rank = 128` | Match |
| 11 | R:590, plain data slope `−0.236` | `S[793].value` | A(ns301), job **3808493**: `summary.plain.loglog_slope = −0.235785849470` | Match; independently refitted |
| 12 | R:605, q=32 worst evolved error `7.077e-01` | `S[811].value` | A(ns304), job **3808502**: `aggregates.neural_q32.worst_evolved = 0.707701237608` | Match |
| 13 | R:698, top/q0 cost `3.93×` | `S[1037].value` | A(ns304): `52.5923382271 / 13.3875380225 = 3.92845481662`, using `aggregates.neural_q{512,0}.median_seconds` | Match |
| 14 | R:617, matched solved/manifold ratio `2.51` | `S[859].value` | A(ns304): `aggregates.neural_q0.median_evolved / median(max_t>0(decomposition.q0.manifold_per_state)) = 2.51186326449` | Match |
| 15 | R:641, q=0 final enstrophy ratio `1.043` | `S[873].value = 1.04261050566` | **No corresponding result-JSON field.** Independently recomputed from the saved NPZ fields: `1.04261050566` | Numerically matches; JSON provenance gap |

For row 15, the sources are `runs/ns304/archive/experiments/ns2d/output/rom_fields_N256.npz`, keys `neural_q0__case0` through `neural_q0__case7`, and `reference_N256.npz`, key `U`, using the final output time.

All nine report source hashes match the actual result files. Every summary row carrying one of those jobs also has the correct result-file SHA256.

**Mismatches and disclosure problems**

1. **Incomplete requested trace.** Row 8 is missing from S. Row 15 requires NPZ files although its summary provenance records only the result-JSON hash. Neither is an arithmetic error, but neither has the complete requested report → summary → artifact-JSON chain.

2. **“No Phase-3 job was ever submitted” is literally incorrect.** R:3 says this, while A(ns304) records `phase = 3`, `job_id = "3808502"`, `complete = true`. R:596 correctly identifies it as exploratory. The accurate statement is that no **confirmatory** Phase-3 job ran on a passing head.

3. **Zero budget exits excludes a failed cold start.** R:608 reports zero for q=256. That correctly counts evolution steps, but A(ns304) `invocations[553]` records:
   - `subject = "neural_q256"`, `case = 1`, `rep = 3`;
   - `ic_iterations = 400`, `ic_reason = 0`;
   - `budget_exits = 0`.

   The same cold-start cap occurs in all three timed repetitions. The column needs an explicit “evolution steps only” qualification. This does not overturn the defined per-step R-CONV verdict.

4. **Existing audit mismatches are disclosed, not resolved.** R:725–729 correctly reports six failed checks for ns201, eight for ns202, and three for ns301. They concern the retracted oracle/projection calculations and SciPy fits outperforming the reported best-found oracle. An audit reporting `all_match = false` should not be read as clean numerical certification.

**Gate audit**

All **133 displayed gate-table verdicts** agree with their stored `passed` flags. Checking those flags against the design reveals these qualifications:

| Verdict or gate | Evidence and assessment |
|---|---|
| **Phase 1 certified / F-JAC** | **Post-observation gate change.** D §A2 changed the centred-control requirement from `≥1e-3` to control/Arakawa `≥1e6` after observing job 3780151’s measurements. The original controls were `9.63917e-4`, `2.36617e-4`, `5.89289e-5`: all fail the original rule. `artifacts/ns101/audit.json`, checks `F-JAC_N{64,128,256}.passed_A2`, records amended passes with ratios `1.52567e14`, `6.40887e13`, `2.40819e13`. R retains the original failures without explaining this certification dependency beside them. |
| **F-TG** | Passes §A1: discrete errors below `1e-10`, semi-discrete errors below `1e-6`, prediction discrepancies below 2%, and orders within `2±0.05`. §A1 was committed **before** the job; it is not a post-data rescue. |
| **F-LAP, F-JAC order, F-MMS, F-BUDGET, F-MESH** | Stored measurements satisfy their numerical rules, subject to the documented successive-difference order estimator. MMS spatial orders are `1.997496/1.998010`; temporal order `2.000006`; flipped-sign discrepancy `0.0520992`. |
| **F-INDEP** | The `≤1e-10` numerical test passes at `9.13562e-16`. Coverage is **100 steps**, versus the family’s 500-step trajectory. R states the shorter coverage; D’s “one dev trajectory” wording should be qualified accordingly. |
| **S0, F-CFL, F-NEWTON, F-DATA** | All nine logs contain `jax_backend=gpu`; result metadata records x64 and `highest`. Phase-1 maximum CFL is `1.22596`, worst cohort Newton residual `9.83707e-15 < 1e-11`, maximum Newton iterations two. Cohort hashes are recorded; the driver asserts disjoint draws. |
| **B-DATA / R-DATA** | **Changed after ns201 data**, through §A4, from hash equality to hash-or-value, including inferred training-cohort acceptance. The change preceded the later runs using it. Ns201 remains failed/retracted. Ns302’s value check and ns304’s reference check satisfy `≤1e-8`; their training-cohort passes are inferred, not direct equality checks. |
| **B-RANKCAP, B-ORTH, B-FLOOR, H-TRAIN** | Retractions and rerun verdicts agree with the data. At N=256, bank/POD-R floor ratios are `1.08069` ns203, `1.26107` ns204, `1.19560` ns302, `1.02767` ns303, all below two. Ns201/ns202 have rank 128 and correctly fail B-ORTH. |
| **H-ORACLE** | **All four retained heads fail the unchanged 2× bar:** ns203 `1.186751`, ns204 `1.153017`, ns302 `1.455552`, ns303 `1.394286`. The 1.5 marker introduced in §A9 is diagnostic, not a replacement gate. |
| **H-SOLVED** | **Unamended statistic discrepancy:** D:172 specifies the initializer at `t=0`; `ns2d_phase2.py:396` tests medians across all evaluated times. Recomputing the intended t=0 comparison still passes every recorded mesh. N=256 ratios for ns203/ns204/ns302/ns303 are `1.150724/1.000000/1.329898/1.000000`, all below 1.5. |
| **Ns301 diagnosis** | §A9’s least-squares slopes independently recompute to `−0.235786/−0.214660/−0.147891`: all **ambiguous**. Regularisation does not reach the specified 10% improvement. The report preserves these verdicts. |
| **R-TB, R-TFFT, R-TQ, R-LIN, R-CONV** | Pass their thresholds: respectively `0`, `7.06507e-16`, `2.39370e-14`, `3.54774e-13`, and zero evolution budget exits. The cold-start qualification above remains. R-FOM lacks a named gate entry, but its reference invocations show maximum residual `9.33696e-16` and at most two Newton iterations. |
| **R-LADDER and frontier** | Correctly **FAILS**, because q=0→32 worsens worst evolved error from `0.687654` to `0.707701`. The worst-error gain is `11.4343×`; median error is monotone. Independently recomputed frontier contains no neural rung. |
| **Amendments §A9 and §A14** | §A9 changed scope after the original heads failed, but was committed before its new jobs and explicitly preserves the failures. Its ladder uses different rungs and fixed M, so it cannot establish the original confirmatory verdict. §A14 corrects the layer statistic after data; it does **not** change R-LADDER’s gate. |

Thus, “the bar was never lowered” is defensible specifically for **H-ORACLE**, not as a blanket description of every gate.

**Cross-job check**

**Pass.** All reported online timing comparisons use ns304, job **3808502**, host **pax105**, **NVIDIA A100 80GB PCIe**.

I recomputed all **20 subjects’ timing medians** from their retained arrays: 24 timed invocations each, totalling 480. All agree with `aggregates.*.median_seconds`. Accuracy is identical across timed repetitions, and the non-dominated set matches the report.

Two qualifications:

- **FOM-versus-ROM order was not balanced.** A(ns304) `invocations[0:224]` contains all FOM invocations; ROM/POD follows. The AB/BA ordering in `ns2d_phase3.py:399–403` applies within the reduced-model block. This departs from D’s broad balanced-order timing contract, although burn-in is present.
- **DESIGN’s matched-POD cost conclusion is false at the top rung.** `aggregates.pod_k544.median_seconds = 53.902544790`, versus `neural_q512 = 52.592338227`: POD is **2.49% slower**, while more accurate. POD is cheaper at **five of six** matched dimensions. The current report’s tables are correct; D §A11/§A13/§A14 retain the overstatement.

**Retraction completeness**

**Incomplete.** There is no dedicated retractions section in R. Neither `checks/` nor `checks/retractions.md` exists in this experiment directory, so completeness against that requested ledger cannot be certified.

| Item | Disclosure status |
|---|---|
| §A4 rank-capped ns201/ns202 and contaminated oracle formula | Disclosed in R’s introduction, explanations and audit failures. |
| §A8 erroneous “no budget exits” for Phase-2 oracles | Disclosed and corrected with per-mesh counts. |
| §A14 mismatched three-layer statistics | Clearly corrected at R:611–624; old values retained as secondary. |
| §A1 TG gate replacement | Current rule reflected; historical replacement not explained in R. |
| **§A2 post-observation F-JAC re-gating** | **Missing from R’s certification explanation.** |
| §A5 substitution of self-audits for unavailable independent-model audits | Not disclosed in R. |
| §A10 three SciPy/oracle discrepancies | Failed check names disclosed, but the amendment’s impact explanation is omitted. |
| **“POD better and cheaper at every matched dimension”** | **Still needs correction in DESIGN**, as quantified above. |
| **“FOM ntol=1e-3 dominates every reduced model”**, D:630 | False: `pod_k32` costs `212.080 ms`, versus FOM `421.269 ms`. R correctly retains POD-32 on the frontier. |
| **“Solved median error falls … 11×”**, D:714 | Statistic mismatch remains: median gain is `0.413921506 / 0.024376675 = 16.9802×`; **11.4343× is the worst-error gain**. |

**Three weakest claims and reviewer questions**

1. **“A ratio near 1 means the head is capacity-limited” — R:320 and repeated elsewhere.**  
   The compared quantities use different normalisations: training reconstruction divides by each snapshot’s norm (`ns2d_decoder.py:124–125`), whereas held-out oracle error divides by the initial-state norm (`ns2d_phase2.py:355–371`). They also use different time sampling and fitting procedures. Ns302 changes data exposure per training example.  
   **Reviewer question:** Does the diagnosis survive matched normalisation, matched times, train-oracle versus dev-oracle, and matched training exposure?

2. **“Phase 1 certified” — R:3.**  
   The certification depends on §A2’s post-observation replacement, while the report prints original failures and `all gates passed = no`.  
   **Reviewer question:** Which exact amended criteria define certification, when were they adopted, and why is that history absent beside the verdict?

3. **The interpretation of the exploratory ladder as evidence for the intended nonlinear correction trade.**  
   R:698 correctly reports a worst-case monotonicity failure. Reaching the bank span at q=R removes the manifold restriction by construction; no neural rung reaches the cost/error frontier. Some decomposition fits also hit their budgets: A(ns304) `decomposition.q0.reasons["0"] = 4` of 48 states.  
   **Reviewer question:** What does this establish beyond enlargement to the full linear bank, and which claimed mechanism survives the failed monotonicity gate and matched POD controls?