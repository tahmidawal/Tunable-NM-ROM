**Overall verdict: WRONG as a production-ready design; not CLEAN.** A3 closes most decision-logic defects, but still omits required head-production work, does not establish a reliable query-compilation cost, and leaves parts of the feasibility certificate and verdict wording incomplete.

This audit was read-only. No files changed, no jobs submitted, no scratch files created, and no writes to main or `LAB-LOG.md`. GPU controls were inspected, not executed. Later amendments govern throughout.

1. **Decision rule — NEEDS-RESTATEMENT.**

   A3.1 closes the two principal findings: timeouts and deferrals remain unresolved, and each accuracy criterion gets its own combination search. Demonstrated construction failures and measured Q1 failures can now eliminate candidates without timing them.

   However, its supposedly exhaustive terminal-state list does not explicitly cover actual NNLS exits such as `max_iter`, `incomplete (KKT violated)`, or unusable checkpoints following a numerical-quality failure. These exist in [nnls.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/jcp-hr-baselines/nnls.py:345).

   Add a default: **every unsuccessful construction without a registered impossibility certificate is UNRESOLVED**. Do not inherit the older builder’s blanket “construction failure” interpretation for unreached targets.

2. **Cost model and bank-fit projection — WRONG overall; bank formula CORRECT.**

   I recomputed the formula from the specified [A100 log](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-jcp-consolidated/experiments/separable-decoder/runs/push_r3a/logs/2835788.out:16):

   $$
   \rho_{\rm sub}=\frac{409-144}{10000}=0.0265\ {\rm s/step},
   \qquad
   \rho_{\rm full}=\frac{10681-9190}{5000}=0.2982\ {\rm s/step},
   $$

   $$
   \tau_{\rm tr}=(9190-7683)-5000(0.2982)=16\ {\rm s},
   $$

   $$
   T_{\rm fit}
   =12+289999(0.0265)+16+10000(0.2982)+4
   =10698.9735\ {\rm s}.
   $$

   This is **0.1340568% above 10,684.65 s**, passing the registered 5% check. The training loop confirms that `STEPS=25000 FULL_LAST=10000` switches to full-batch work at step 15,001. The transition residual is empirical overhead, not an isolated compilation measurement.

   **Head provenance and H200 remeasurement are corrected.** The deployed `Z_tr` is refitted by the head job. Measuring extraction and the producing arm on H200 removes the unsupported cross-GPU throughput conversion.

   **But A3 still omits shared preparation and checkpoint emission.** In [sep_hfit_run.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/separable-decoder/sep_hfit_run.py:146):

   - Input loading, whitening, Cholesky/triangular solves and setup occur before the arm timer.
   - The arm timer ends before conversion back to deployed head parameters and checkpoint emission.
   - `mid.seconds` therefore cannot represent the complete producing dependency.

   Recomputed from the historical JSON:

   | Component | Seconds |
   |---|---:|
   | `full.total_seconds` | 2675.561637 |
   | `mid.seconds` | 360.853482 |
   | `wide.seconds + k32.seconds` | 2305.507155 |
   | Outside all arm timers | **9.201000** |

   The last row contains shared preparation and other overhead; it is not all necessarily dependency work, but A3 charges none of it. For the single-arm H200 rerun, use the complete producing invocation or explicitly partition preparation, fitting/evaluation and emission. Apply the same boundary to A_100.

   **Compilation is only partially repaired.** Including construction compilation in build timers and canonicalizing PM/CP source artifacts are correct. However, disabling persistent caching does not prevent an in-process executable-cache hit. Calling `lower().compile()` after the same executable has already been compiled can measure retrieval rather than compilation. Multiplying that near-zero result by three does not recover the missing cost.

   Specify a fresh-process measurement or demonstrate that the measured executable has not already been compiled. Otherwise, describe the accounting as conditional on pre-existing compilation caches; the ×0/×3 scenario is not a bound on cold deployment cost.

   **A_sel remains incomplete.** A3.4 specifies nondeployed dev6 queries, but not construction work unique to rejected configurations—such as another rank’s contributions, NNLS path, arm preparation and compilation. It also adds historical nonproducing head experiments without specifying their H200 measurement or conversion, despite withdrawing the cross-GPU projection. Define the exact selection-artifact union and its hardware provenance.

3. **Resampling, coefficients and rule mapping — CORRECT.**

   The orthonormal DST-I in [engines.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/mr-burgers2d/engines.py:54) supports

   $$
   \Phi_L=U_LEU_s\Phi_s,\qquad c_L=(L/s)c_s.
   $$

   The prolonged basis’s nodal amplitudes are scaled by $s/L$ relative to the physical sine interpolant; reciprocal coefficient scaling restores the field.

   Under restriction,

   $$
   \widetilde\Phi_L=U_LTU_s\Phi_s=QR,
   \qquad c_L=(L/s)Rc_s.
   $$

   This reconstructs the **low-pass projected field**, not generally the source field evaluated at coarse nodes. For row-oriented codes and a deployed prefix,

   $$
   C_{L,r}=(L/s)C_s[:, :r]R[:r,:r]^\top.
   $$

   The combined sign-fixed factor and rank-specific transformation are correct. Small NumPy checks gave maximum absolute discrepancies of approximately $7.8\times10^{-16}$ for prolongation and $2.7\times10^{-16}$ for restriction.

   The draft `mmcore.transform_codes` agrees. Its older diagonal-ratio acceptance path remains a later code-audit issue, not a defect in A2/A3’s singular-value requirement.

4. **Qualification, inherited evidence and J5 — CORRECT.**

   **LSPG depends indirectly on both the head and `Z_tr`.** Its evolution is linear-span, but [qcore.Model.trust_linear](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/jcp-p1-scaling/vendor/quad2d/qcore.py:152) evaluates the head on stored codes and derives the trust radius. `Bank.cold_trust` supplies that radius to the solver.

   **`p1acc` supports `q2_cases='all38'`.** [p1common.select](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/jcp-p1-scaling/p1common.py:91) implements it; [p1acc](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/jcp-p1-scaling/p1acc.py:129) reads it per configuration and intersects it with the accuracy population. Registered successor rules must also be supplied.

   Fresh qualification, Q1-only competitor labels, and internal-state/iteration/exit identity checks close the previous findings.

5. **Scoring contract — CORRECT.**

   The primary strict worst-error stop rule, worst-plus-median claim, casewise reference-uncertainty comparison and rollout-agreement sensitivity are distinguished. A3’s independent searches close the remaining selection defect. Eligibility, padded restriction and pooled-median interpretation remain appropriate.

6. **Must-fail controls — CORRECT in design, subject to successful smoke demonstration.**

   | Control | Verdict | Assessment under A3 |
   |---|---|---|
   | **1. CPU preflight** | **CORRECT** | Explicit CPU execution must exit 42. |
   | **2. Resampling/trust radii** | **CORRECT** | Physical-node, QR reconstruction and rank-specific code checks address the relevant errors. Scaling a normalized column by $1+10^{-6}$ produces a Gram defect near $2\times10^{-6}$, above the threshold. Shift/transpose separation remains empirical. |
   | **3. Rule mapping** | **CORRECT** | Coordinate and sampled-value checks replace an unjustified accuracy guarantee. Use a strict source-to-finer fixture. |
   | **4. Premature solver** | **CORRECT** | A3 explicitly blocks production if neither mutation fails Q1. Record the failing Q1 predicate; do not infer stationarity failure merely from moving dynamics. |
   | **5. Blocked contributions** | **CORRECT** | Independent candidates, kept-state masks and $G,b$ comparisons are appropriate. The advection mutation must exhibit the registered separation. |
   | **6. Case identity** | **CORRECT** | Full descriptor and cohort/index binding detects the specified swap, including distinctions invisible in the initial field alone. |
   | **7. Timing injection** | **CORRECT** | Injection into every observation fixes the median problem. The unmutated paired block must pass first. |
   | **8. Decision logic** | **CORRECT** | Exhaustive fixture universes and the two global-outcome cases repair the logical errors. Details below. |
   | **9. Replication mismatch** | **CORRECT** | The frozen R128/R256 separation exceeds the $10^{-8}$ comparison threshold. |

   Control 8’s individual cases now have defensible expectations:

   | Subcontrol | Verdict | Required fixture interpretation |
   |---|---|---|
   | **8(a)** Errors $+\infty$ | **CORRECT** | Exhaustive universe; qualified X with finite errors. |
   | **8(b)** Costs $1.5C_X+1$ | **CORRECT** | Mutate the complete comparator cost used by the decision predicate. |
   | **8(c)** Q1 false | **CORRECT** | Measured failures, not missing qualification evidence. |
   | **8(d)** Remove one of two witnesses | **CORRECT** | A measured admissible witness must exist before duplication. |
   | **8(e)** Remove all witnesses; unresolved candidate remains | **CORRECT** | The unresolved candidate must remain in the declared universe. |
   | **8(f)** Missing X with/without another NO-MATCH | **CORRECT** | Correctly distinguishes NOT EVALUABLE from NOT FIRED. |
   | **8(g)** All X disqualified | **CORRECT** | Must report vacuity and its reason. |

   These are controlled decision fixtures derived from real measurements. In particular, duplicating a witness tests logic; it does not establish two independently measured scientific alternatives. If the smoke supplies no admissible witness, 8(d) has not passed.

7. **Feasibility, budget and timing — NEEDS-RESTATEMENT.**

   A3 fixes the contribution-matrix size:

   $$
   120\cdot384\cdot32768\cdot8
   =12{,}079{,}595{,}520\ {\rm bytes}
   =11.25\ {\rm GiB}.
   $$

   Recording its peak with $\Phi_{384}$ resident is materially stronger than the old 12-state pilot.

   **It still does not automatically certify the entire NNLS path.** The implementation allocates its fixed-capacity QR buffers once a candidate is accepted, but active-support refactorization and deletion workspaces depend on the reached support and execution path. Tiling states does not guarantee those paths are exercised. See [UpdQR.refactor](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/jcp-hr-baselines/nnls.py:277).

   Require either observed coverage of the maximum relevant workspace shapes or an explicit allocation certificate for unexercised paths, retries and host memory. A successful tiled pilot alone is insufficient.

   The inherited unreduced-POD rollout diagnostics at 256² and 1024² also remain required unless explicitly withdrawn. A3 still does not identify their runtime/memory allowance.

   The runtime projection is now correctly labelled a planning estimate. Timeouts remain unresolved, and the approximately 6.4 GPU-hour total is a projection pending certificates, not demonstrated feasibility.

8. **PM at 4096² — CORRECT.**

   A2.8 remains governing: estimated snapshot-only cost, descriptive sample range, remaining setup and accuracy unmeasured. A3 introduces no regression. This estimate cannot decide the CP stop rule.

9. **Interpretation — NEEDS-RESTATEMENT.**

   Vacuous outcomes are now explained correctly.

   However, A3.7 weakens the earlier necessary condition by saying the least-cost admissible competitor is named **“if any.”** If every competitor is inaccurate or disqualified, NO-MATCH can be proven without any measured cost advantage.

   Retain the explicit requirement:

   **Claim comparative cost advantage only if an accuracy-admissible competitor combination exists, the relevant search is resolved, and its minimum cost exceeds X’s cost.**

   If none exists, report an accuracy/qualification separation with cost advantage undetermined. If the intent is that A2’s existence condition survives, state that explicitly rather than relying on reconciliation between amendments.

| Round-three item | Fourth-round verdict |
|---|---|
| 1. Decision rule | **NEEDS-RESTATEMENT** |
| 2. Cost model and bank projection | **WRONG**; bank formula **CORRECT** |
| 3. Resampling, coefficients and mapping | **CORRECT** |
| 4. Qualification and J5 | **CORRECT** |
| 5. Scoring contract | **CORRECT** |
| 6. Controls | **CORRECT**, pending real smoke |
| 7. Feasibility, budget and timing | **NEEDS-RESTATEMENT** |
| 8. PM estimate | **CORRECT** |
| 9. Interpretation | **NEEDS-RESTATEMENT** |

**Remaining required changes:**

- Include head preparation and checkpoint-production overhead with consistent H200/A_100 boundaries.
- Establish a genuine query-compilation measurement; cached retrieval multiplied by three is insufficient.
- Define complete A_sel artifact accounting and hardware provenance.
- Map every noncertifying NNLS termination to UNRESOLVED.
- Certify maximum NNLS workspace paths and account for, or explicitly withdraw, unreduced-POD diagnostics.
- Require an existing admissible competitor before claiming comparative cost advantage.

**Not CLEAN.**