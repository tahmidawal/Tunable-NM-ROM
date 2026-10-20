**Overall verdict: NEEDS-RESTATEMENT.** A2 resolves much of audit 2, but leaves interaction coverage incomplete, reintroduces a missing-component dominance problem, and omits an FD reference needed by its new \(T=1\) cross-check.

Read-only audit: no files changed, jobs submitted, or arrays copied; no scratch used.

Below, **D** means [DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/DESIGN.md). **Q**, **C**, and **E** denote the supplied qualify, combined, and references experiment directories.

1. **Lane coverage — NEEDS-RESTATEMENT**

   B7, long-horizon Q1/Q2, spectral checks, B8 diagnostics, controls, and the defer order are covered. The remaining problem is the binding requirement to measure interactions for at least the two largest components.

   A2 explicitly excludes M from the contest and exempts the 3D F–S pair, calling this a “lane decision; recorded for the coordinator” (D:589–601). Recording a scope reduction is not coordinator agreement. The [lane brief](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/campaign/briefs/jcp-budget.md:18) still requires the broader coverage.

   Moreover, the proposed ingredient–dynamics calculation does not measure interaction with the **ideal-dynamics component**; see item 2. Thus audit-2 required fix 1 remains open.

   Resolve this either through a coordinator-approved restricted estimand, or additional measurements. Do not claim the brief’s unrestricted “two largest components” contract is satisfied.

2. **B2 mathematics — WRONG in A2’s interaction interpretation; core identities CORRECT**

   The Möbius effects and full-factorial vector telescoping are correct (D:179–182). A1 correctly distinguishes the GAL-deployed endpoint, common ideal, and 3D closure path (D:432–441).

   The per-time floor identity is also correct: project onto the sampled bank span, use the same normalizer throughout, and include only zero walls outside the 2D interior. The relevant interiors are \(255^2\) and \(63^3\). C/budget2d.py:219–231 implements this convention. Separately maximized terms do **not** inherit the per-time identity.

   **Remaining mathematical misidentification:** A2 defines

   \[
   \chi_x=2\langle I_x,u_\varnothing-\Pi u_{\rm ref}\rangle.
   \]

   Its displayed identity is true, but this is interaction with the **total deployed in-span error**, not with \(u_\star-\Pi u_{\rm ref}\), the candidate called ideal dynamics. Indeed,

   \[
   \chi_x=
   2\langle I_x,u_\star-\Pi u_{\rm ref}\rangle+
   2\langle I_x,u_\varnothing-u_\star\rangle.
   \]

   The second term mixes numerical displacement into the purported dynamics interaction. D:594–597 therefore does not close audit 2’s ingredient-specific dynamics finding. Report the first term separately, with an explicit background for the 3D tight-end F effect.

   \(U_\star\) remains a correctly specified **indicator**, not a bound.

3. **Ingredient switches — NEEDS-RESTATEMENT**

   The form and time packages are now honestly defined; preserving the initialization-derived stopping scale isolates the IC replacement. Q/p1core.py:346–350 supplies that scale, and :192 uses it. The 3D IC switch is structurally zero on the mesh path: Q/s3common.py:135–143 calls `TB.project`, whose triangular solves implement mesh LS. The projection **error** remains nonzero.

   **Remaining LS issue:** A2’s purported TSQR computes local projected right-hand sides using

   \[
   R_b^{-\mathsf T}G_b^{\mathsf T}u_b
   \]

   rather than applying \(Q_b^{\mathsf T}\) directly (D:666–669). The identity requires invertible local \(R_b\); global full column rank does not guarantee full rank of every row block. Even when invertible, forming \(G_b^{\mathsf T}u_b\) and solving through an ill-conditioned local \(R_b\) does not provide the claimed QR stability merely because the final \(R\) is acceptable.

   Use the block QR reflectors to apply \(Q_b^{\mathsf T}\), streaming all required IC right-hand sides, or specify local-rank/conditioning checks and a stable fallback. Merely reporting \(\operatorname{cond}(R)>10^{10}\) leaves the earlier conditioning acceptance issue unresolved.

   The blocked bank structure makes full-grid LS feasible in memory in principle; that is separate from this numerical issue.

4. **Decision rules — NEEDS-RESTATEMENT**

   A2 resolves the missing minimum populations, adjacent-window growth screening, common front populations, Kendall ties, and named zero denominators (D:639–650). The thresholds are generally falsifiable.

   **Direct contradiction:** D:640–642 requires a complete candidate set before awarding a largest component. D:657–660 then permits dominance to be computed **without Q** after its convergence check fails.

   “Q withheld” does not prevent a false dominance conclusion: the withheld Q component could exceed every retained component. Such results must be “largest among available” descriptively, with registered dominance inconclusive.

   Failure propagation must also cover the ideal dynamics and other contrasts using an unverified \(q_\infty\) trajectory. Withholding only the Q main effect does not establish that the remaining “ideal” anchor is quadrature-converged.

5. **Reference generation — NEEDS-RESTATEMENT**

   The lattice strides are correct:

   - \(1025^2\): 2, 4, 8 for FD \(L=2048,4096,8192\).
   - \(127^3\): 1, 2, 4 for \(L=128,256,512\), using interior indices \(js-1\).
   - Refinement: 12 and 6.

   E/refcore.py:225–236 confirms the indexing. Horizon generalization is feasible but must cover all hard-coded clipping, stage-count, shape, metadata and judge-loop assumptions—not just buffers.

   A2 adequately specifies retry selection, route pairing, G-R2 consequences, case 43’s \(T=0.5\) FD ladder, and spectral escalation. Cross-GPU bars are acceptance criteria requiring demonstration, not guaranteed reproducibility. Restricting refinement at \(T=1\) is defensible with non-rankable failures.

   **New concrete omission:** D:636–637 adds \(T=1\) PS checks on val32 **{0,19}**, but FD references at \(T=1\) remain **sub16 = dev6 + val32[0:10]** (D:89–90,134). Case 19 therefore has no matching \(T=1\) FD ladder.

   Add that FD run and its eligibility/replay records and cost. A PS trajectory alone cannot validate the \(W_3\) FD reference. These are legitimate validation-cohort cases; cohort provenance is not the problem.

6. **Must-fail controls — NEEDS-RESTATEMENT, principally for applicability**

   | Control | Verdict | Remaining assessment |
   |---|---|---|
   | C-A | **NEEDS-RESTATEMENT** | The new amplitude exceeds the uncertainty-aware inconsistency threshold. But A1.4 excludes coarse reference-limited cases before classification; injection leaves their coarse error unchanged. Freeze a resolved fixture/population, or distinguish the raw inequality test from the production classifier. |
   | C-E | **CORRECT** | Testing the raw balance resolves normalization; the absolute floor resolves the zero-comparison issue. Demonstration of the retained nonzero predicate remains a smoke requirement. |
   | C-F | **CORRECT** | Orthogonalization, uncertainty-aware amplitude, residual-zero handling and seed fallback address the prior defects, for complete available cases. |
   | C-I | **CORRECT** | Direct perturbed-minus-unperturbed decoded IC distance removes the previous cancellation ambiguity. |
   | C-S | **CORRECT, smoke-conditional** | Explicit 2D and 3D fixtures now exist. Budget zero need not freeze the state—it can return a predictor—so the registered distance must actually be demonstrated. |
   | C-T | **CORRECT, smoke-conditional** | Both dimensions now have explicit fixtures. Prior evidence supports, but does not prove, the new predicates. |
   | C-Q | **CORRECT, smoke-conditional** | A2 explicitly acknowledges the changed Gauss comparator; it no longer treats the predecessor result as proof. |
   | C-R1 | **CORRECT, smoke-conditional** | Named dimension-specific first-order ladders, prerequisites and explicit order rejection are specified. Order failure at \(T=0.5\) still requires demonstration. |
   | C-R2/C-score | **CORRECT** | Identity rejection is a guaranteed plumbing control when applied to the actual scoring path; numerical time-shift differences remain descriptive. |

   Evidence: D:603–617; reference-limited exclusion at D:472–474. No real-size smoke was run during this audit.

7. **Scoring contract — CORRECT in its main specification**

   Development/validation-only populations, ST primary, restricted S use, conditional uncertainty labels, per-case-kind eligibility, and GAL diagnostic treatment are retained. The FRONT-2D and EXPORT hashes match, as do the listed front members and four 3D configurations.

   A2 now separates validated verdicts, unvalidated provisional comparisons without verdicts, and suspended descriptive scores (D:634–637). It also excludes designated cross-check cases unless their own check passes.

   The missing \(T=1\) FD case in item 5 prevents the intended \(W_3\) validation. The incomplete-candidate problem in item 4 separately blocks dominance claims.

8. **Compute, memory and storage — NEEDS-RESTATEMENT only where the inventory changes**

   I accept the supplied coordinator smoke allowance: **one smoke at a time, each ≤2 GPU-hours**. This audit does not reopen that resolved issue.

   A2’s listed estimates sum to **33.5 GPU-hours**, or **41.875** with 25% contingency. Job caps, outstanding reservations, production release, cluster reference retention, and local large-array restrictions are appropriately specified.

   The memory pilot covers the relevant combined paths. Approximate table storage alone remains 1.76/7.03 GiB for 2D Gauss320/640 at \(r=384,M=1536\), and 4.02/6.00 GiB for 3D Gauss56/64 at \(r=512,M\approx2048\); construction peaks require the registered measurements.

   Reprice the missing case-19 FD ladder and the stable LS implementation. The current total is an estimate, not a demonstrated resource certificate.

9. **Other validity hole — NEEDS-RESTATEMENT**

   A2 requires **every check-rule run** to satisfy `stats.nfail == 0` using its own stopping test (D:652–656). But G-B4 replays Q-containing corners with their solver unchanged; in 3D some are fixed-sweep or QN runs.

   Those solver paths do not expose the adaptive `stats.nfail` contract. Q/s3common.py:223–234 distinguishes their validity rules; Q/s3ctx.py:155–159 separately certifies their stationarity.

   Specify solver-specific acceptance for these check-rule runs. Silently replacing a fixed-sweep/QN check with adaptive LM changes the solver as well as quadrature and invalidates the intended contrast. Adaptive tight/ideal anchors can and should retain the strict verified-convergence requirement.

Audit-2 required-fix closure:

| Audit-2 fix | Round-3 status |
|---|---|
| 1. Largest-component interactions | **Open** |
| 2. C-A/C-F/C-E constructions | **Mostly resolved; C-A applicability remains** |
| 3. Per-dimension fixtures/comparators | **Resolved at design stage; smoke pending** |
| 4. Case 43 and PS escalation | **Resolved** |
| 5. Reference pairing/eligibility/\(W_3\) | **Partial; new case-19 FD omission** |
| 6. Decision/population rules | **Partial; Q-withheld dominance contradicts completeness** |
| 7. Anchor convergence/quadrature claims | **Partial; solver-specific check acceptance unresolved** |
| 8. M, LS and logical switches | **Partial; LS stability/rank handling unresolved** |
| 9. Smoke authority/compute inventory | **Authority resolved; inventory needs the added FD run** |

**Overall verdict: NEEDS-RESTATEMENT. Required fixes:**

1. Obtain coordinator agreement for narrowed interaction coverage or meet the binding scope; separately compute interaction with the actual ideal-dynamics vector.
2. Make unavailable Q force registered dominance inconclusive, and propagate failed quadrature verification to dependent ideal-anchor claims.
3. Add val32 case 19’s \(T=1\) FD ladder and reprice it.
4. Replace local inverse-based TSQR right-hand-side construction with stable QR application, with explicit rank/conditioning failure handling.
5. Define acceptance for fixed-sweep/QN check-rule runs without changing the solver being compared.
6. Restrict C-A’s classifier assertion to explicitly resolved cases, while retaining its raw alias-detection test.