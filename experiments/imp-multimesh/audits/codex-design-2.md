**Overall verdict: WRONG as a production-ready design.** A1 fixes the DST normalization and the failed A100 extrapolation check, and it correctly expands our Q2 population. It does not close the head-refit dependency, casewise ranking, full-cost accounting, or several control and verdict defects.

This audit was read-only. I changed no files, submitted no jobs, and created no scratch files or large arrays.

1. **Decision rule — WRONG.**

   Joint enumeration with shared costs is the right correction. Explicit treatment of solver failures and reference-ineligible cases is also correct.

   However, A1’s “casewise” rule still compares **two aggregate maxima**, potentially attained on different cases:
   \[
   W_{\rm CP}\le W_X+2\varepsilon_{\rm ref}(c^\dagger).
   \]
   This is not the COMMON-BRIEF rule for ranking two arms **on the same case**. For example, with negligible reference uncertainty, errors \(X=(0.10,0.01)\) and \(\mathrm{CP}=(0.09,0.09)\) satisfy the aggregate rule despite CP being much worse on the second case. An aggregate worst-error comparison can be registered separately, but cannot be called compliance with casewise matching.

   The median rule is undefined: “under the same tie rule” identifies no reference uncertainty for an even-sized pooled median, whose central observations can belong to different cases.

   Missing-data semantics also conflict: A1.7 makes an affected workload not evaluable, whereas A1.10 limits that consequence to the deferred arm. Use explicit three-valued logic:
   
   - A measured matching CP combination proves a match, regardless of other missing arms.
   - Exhaustive measured nonmatches prove no match.
   - Otherwise the comparison is unresolved.
   
   Likewise, missing qualification evidence for X must not be treated as demonstrated disqualification and removed from “us.”

   Finally, “comparative multi-mesh cost advantage survives” remains too strong. Nonmatching can result solely from accuracy: a cheaper CP arm with worse accuracy establishes no cost advantage for X at matched accuracy. Report **“the stop rule did not fire”** and the specific unresolved trade-off.

2. **Cost model and bank-fit projection — WRONG overall; A100 numerical check CORRECT.**

   From the [A100 log](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-jcp-consolidated/experiments/separable-decoder/runs/push_r3a/logs/2835788.out:16):
   \[
   \rho_{\rm sub}=\frac{409-144}{10000}=0.0265,\qquad
   \rho_{\rm full}=\frac{10681-7683}{10000}=0.2998.
   \]
   Using the rounded log timestamps, \(t_1=12\) and \(t_{\rm done}=10685-10681=4\):
   \[
   T_{\rm fit}=12+289999(0.0265)+10000(0.2998)+4
              =10698.9735\ {\rm s}.
   \]
   This is **0.1341% above 10,684.65 s**, comfortably passing 5%. The previous unexplained late slowdown is explained by the separate full-batch kernel in [sep_solvers.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/separable-decoder/sep_solvers.py:478).

   Remaining defects:

   - **The head-refit exclusion is factually wrong.** `Model.trust_linear` evaluates the checkpoint head on `Z_tr`, rotates those outputs, and derives the trust radius. `Bank.cold_trust` passes that radius into LSPG. Thus the deployed solver depends on both the head and training codes, even though its trial space is linear. See [qcore.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/jcp-p1-scaling/vendor/quad2d/qcore.py:152) and [p1common.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/jcp-p1-scaling/p1common.py:169). Cold setup also evaluates them, although the linear initializer ignores the candidate-related entries of the resulting tuple. Charge the refit dependency, or register and certify a replacement that removes it.
   - **J2 stops before measuring \(t_{\rm done}\).** Specify its H200 measurement or an explicit bounded approximation.
   - The H200 full-batch window includes first-use compilation. Multiplying its 5,000-step average by 10,000 charges that transition overhead twice. Measure transition compilation once and steady full-batch throughput separately.
   - Rotation and rule selection remain zero-cost omissions. Calling the resulting number “full cost” is unsupported; omission labels alone do not establish that the verdict is robust.
   - The ledger needs basis-qualified keys. `contrib[L,r]`, `nnls[L,r]`, and `codes[L]` depend on the source basis as well as mesh/rank. Conversely, reusable raw FOM states can be shared. Record the dependency graph explicitly.
   - A1 charges source `armdata` and compilation whenever CPU uses a source rule, even when the source mesh is absent from the workload. Rule construction does not inherently require a deployed source-query executable. Separate construction kernels from source-query preparation to avoid charging unused deployment work.
   - Distinguish compilation already included in timed setup/build phases from the separately charged query compilation.

3. **Resampling, coefficients and rule mapping — CORRECT, with implementation details to freeze.**

   The revised formulas agree with the repository’s orthonormal DST-I:
   \[
   \Phi_L=U_LEU_s\Phi_s,\qquad c_L=(L/s)c_s.
   \]
   Under restriction, if \(\widetilde\Phi_L=QR\),
   \[
   Qc_L=(L/s)\widetilde\Phi_Lc_s
   \quad\Longrightarrow\quad c_L=(L/s)Rc_s.
   \]
   For row-oriented snapshot codes, this is
   \[
   C_L=(L/s)C_sR^\top.
   \]
   Under restriction, “same physical field” means the **low-pass projected field**, as A1 otherwise acknowledges.

   Freeze two implementation requirements:

   - With two sign-fixed QR passes, \(R\) must be the **combined reconstruction factor**, including sign changes, not merely the second pass’s factor.
   - Transform codes consistently for each deployed rank. Transforming full-rank codes and then taking a prefix is not generally equivalent to transforming the original rank-\(r\) reconstruction.

   The node mapping and removal of assumed reuse accuracy are correct. The diagonal-ratio rank check is a heuristic, not a reliable conditioning certificate; use a small-\(R\) singular-value check if numerical rank is the acceptance criterion.

4. **Qualification, inherited evidence and J5 — NEEDS-RESTATEMENT.**

   **`p1acc` supports `q2_cases='all38'`.** It selects this value through `P.select`, intersects it with the accuracy population, and runs successor-rule checks for those cases. See [p1acc.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/jcp-p1-scaling/p1acc.py:129) and [p1common.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/jcp-p1-scaling/p1common.py:91). This closes the previous Q2 population defect, provided the field is placed in **each configuration**, with the required successor rules.

   A1’s historical statement that Q1 covered all 38 cases is still too broad: the previously identified 4096² R128/R256 jobs had only 16 accuracy cases. The new J5 repairs the coverage prospectively.

   The competitor labels are now appropriately “Q1-qualified.” However, agreement between two support sizes is still not own-state Q2 with the \(\rho\) bar. Keep the sensitivity explicitly labelled a **rollout-agreement sensitivity**, not numerical qualification.

   J6’s six output coefficient vectors do not establish identity of all internal states, iterations, or exits. Compare the solver/configuration manifest and step statistics as well; alternatively certify the timed invocation through a verified equivalent recording path.

5. **Scoring contract — WRONG overall.**

   Padded-field restriction, per-mesh reporting, eligibility handling, and explicit pooled-median interpretation close those parts of the previous finding.

   The casewise ranking defect and undefined median uncertainty remain as described in item 1. Reporting a secondary median verdict does not make the primary comparison satisfy the lane brief’s “accuracy-matched (worst and median)” requirement. Separate the verbatim worst-only stop rule from the claim requiring both metrics.

6. **Must-fail controls — WRONG overall.**

   The revised individual controls are assessed below. None was executed on a GPU during this read-only audit.

   | Control | Verdict | Does A1 close it? |
   |---|---|---|
   | 1. CPU preflight | **CORRECT** | Explicit CPU execution should produce exit 42. Demonstration remains a smoke requirement. |
   | 2. Resampling and trust radii | **NEEDS-RESTATEMENT** | Physical-field and trust-radius checks are valuable. But restriction without QR need not fail: a basis with sufficiently little discarded spectral energy can remain orthonormal within tolerance. Pin a real fixture with a measured raw defect above the threshold. Add reconstruction checks for the combined QR factor and transformed codes. |
   | 3. Rule mapping | **CORRECT** | Deterministic coordinates and sampled basis values replace the invalid accuracy-factor requirement. Use a strict source-to-finer transfer and a frozen nondegenerate basis fixture. |
   | 4. LM budget 0 | **NEEDS-RESTATEMENT** | Much stronger than budget 1, but still empirical: an unsolved predictor is not logically guaranteed to exceed the stationarity threshold on more than 1% of steps. Freeze the case/configuration and demonstrate the intended stationarity failure. |
   | 5. Blocked contributions | **CORRECT** | Independent candidate generation, kept-state masks, and \(G,b\) equality address the former omissions. The registered advection mutation still needs its measured separation in S1. |
   | 6. Wrong-case scoring | **WRONG** | The specific mutation is detectable, but a 10% projected-IC proximity test does not bind case identity. Nearby cases can pass; changing viscosity alone leaves the IC identical. Bind the complete descriptor, cohort/index, and run identity to the scored output. |
   | 7. Timing injection | **NEEDS-RESTATEMENT** | Scaling the injection is better. Define whether it affects every observation contributing to the A2 median. Sleeping once in a 38-case block need not move its median enough to fail. |
   | 8. Decision logic | **WRONG** | Removing one competitor measurement need not make a comparison unresolved if another measured combination already proves a match. Also, multiplying costs by \(10^6\) is not a mathematical guarantee without checking the fixture’s cost ratios. Test witness-present and witness-missing cases separately. |
   | 9. Replication mismatch | **NEEDS-RESTATEMENT** | The smoke-sized fixture fixes scheduling. Still register the actual R128/R256 separation and a known failing case; different ranks alone do not guarantee an error difference above \(10^{-8}\). |

   For control 6, I reconstructed the registered dev6 descriptors and Gaussian ICs using NumPy only. Case 0 versus case 1 has relative IC distance **1.377256**, normalized by case 1. If the projected case-0 IC satisfies its own 0.1 bound, its distance from case 1 remains at least **1.277020**. Thus **this particular swap can fire**; it is the claimed general identity protection that is inadequate.

7. **Feasibility, budget and timing — NEEDS-RESTATEMENT.**

   Sequential residency, measured memory checks, sentinels, and production gates are substantial improvements. Remaining gaps:

   - S1 needs real source bases, rules, codes and states before J0 constructs them. Identify the read-only input artifacts and hashes, or include their construction in S1’s budget.
   - A 12-state NNLS pilot cannot automatically certify the runtime of the full 120-state path through linear scaling. Active-set behaviour and termination can change with the training matrix. State a conservative projection and how early termination affects attainable support sizes.
   - “5% = the timing drift band” is false: the registered ratio interval is \([1/1.1,1.1]\), approximately −9.1% to +10%. Neither threshold bounds bank-cost projection uncertainty.
   - One observation per competitor case is not a repetition array. Specify saved observations, medians, outlier counts, and repeated timing sufficient to assess relevant near-ties.
   - Disabling persistent compilation caching does not clear in-process JIT caches. Specify fresh-process or equivalent cold-cache boundaries.
   - Define the common host/device input, output-transfer, and storage boundaries. “Full-field input/output” alone does not establish what is charged.
   - The 5.9 GPU-hour sum is arithmetically correct, but remains an estimate pending phase certificates.

8. **PM at 4096² — WRONG.**

   Sampling actual CNR snapshot trajectories and acknowledging out-of-core construction fixes two earlier defects.

   But **“PM ≥ estimated snapshot cost” still asserts an unsupported lower bound**. A sample mean can exceed the population mean; the sample min–max is neither a guaranteed population range nor a confidence interval. Use “estimated snapshot-only cost; remaining setup and accuracy unmeasured,” with an explicitly descriptive sample range or a justified uncertainty calculation. Do not infer that PM loses regardless of accuracy.

9. **Interpretation and other threats — NEEDS-RESTATEMENT.**

   Implementation-specific scope, the stronger possible POD alternative, and lower-envelope break-even reporting are correct.

   The remaining claim language must reflect the unresolved accounting, ranking and uncertainty issues. An unfired stop rule, a vacuous match with no qualified X, and a measured comparative cost advantage are three different outcomes and must remain separate.

| Previous item | Second-round verdict |
|---|---|
| 1. Stop rule and operationalisation | **WRONG** |
| 2. Cost model and H200 projection | **WRONG** |
| 3. Resampling, coefficients and reuse | **CORRECT** |
| 4. Tuning, qualification and J5 | **NEEDS-RESTATEMENT** |
| 5. Scoring contract | **WRONG** |
| 6. Must-fail controls | **WRONG** |
| 7. Feasibility, budget and timing | **NEEDS-RESTATEMENT** |
| 8. PM cost estimate | **WRONG** |
| 9. Interpretation | **NEEDS-RESTATEMENT** |

**Remaining required changes:**

- Correct the head/`Z_tr` dependency and charge its creation, or register and certify a replacement.
- Separate full-batch compilation, steady training and terminal work; complete the artifact dependency ledger and resolve zero-cost omissions.
- Replace the purported casewise rule, define median matching, and register consistent three-valued verdict logic.
- Freeze combined-QR reconstruction and rank-specific code transformations.
- Correct historical qualification coverage and strengthen timing-run identity checks.
- Repair controls 2, 4, 6–9 and demonstrate all controls on frozen real fixtures.
- Specify S1 inputs, production cost projections, timing repetitions, cache boundaries and uncertainty treatment.
- Remove the PM lower-bound inequality and the claim that an unfired stop rule establishes a cost advantage.