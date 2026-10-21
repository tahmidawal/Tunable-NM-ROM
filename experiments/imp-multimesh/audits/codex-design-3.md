**Overall verdict: WRONG as a production-ready design.** A2 closes most mathematical and scoring defects, but introduces incorrect head-cost accounting and leaves compilation, timeout semantics, and decision controls insufficiently specified.

This audit was read-only: no files changed, no jobs submitted, no scratch arrays created, and no writes to main or `LAB-LOG.md`. GPU controls were inspected, not executed. A2 takes precedence throughout.

1. **Decision rule — NEEDS-RESTATEMENT.**

   A2 correctly separates aggregate worst-error matching, casewise comparison with reference uncertainty, and strict median matching. Joint enumeration and witness-based MATCH also close the principal round-two defects. Missing qualification evidence for X is correctly distinguished from failure.

   Two gaps remain:

   - **Construction outcomes need explicit semantics.** A2.7 calls a time-limited NNLS configuration “not constructible,” while A2.1 requires every registered combination to be measured before NO-MATCH. A timeout is missing evidence about the intended configuration, not proof that it cannot match. Conversely, a demonstrated rank failure or measured Q1 failure can eliminate a combination without timing it. Define these terminal states separately.
   - **Different verdicts require separate searches.** The worst-plus-median verdict must enumerate combinations satisfying both metrics. Testing the median only on the cheapest worst-admissible witness can miss another valid witness. Apply the same principle to casewise and rollout-agreement sensitivities.

   The global NOT FIRED rule is logically sound: one demonstrated NO-MATCH suffices even if another X remains unresolved. This matters for control 8(f).

2. **Cost model and bank-fit projection — WRONG overall; phase-wise bank calculation CORRECT.**

   From the specified [A100 log](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-jcp-consolidated/experiments/separable-decoder/runs/push_r3a/logs/2835788.out):

   $$
   \rho_{\rm sub}=\frac{409-144}{10000}=0.0265\ {\rm s/step},
   $$

   $$
   \rho_{\rm full}=\frac{10681-9190}{5000}=0.2982\ {\rm s/step},
   $$

   $$
   \tau_{\rm tr}=(9190-7683)-5000(0.2982)=16\ {\rm s}.
   $$

   With rounded timestamps \(t_1=12\) and \(t_{\rm done}=10685-10681=4\),

   $$
   T_{\rm fit}
   =12+289999(0.0265)+16+10000(0.2982)+4
   =10698.9735\ {\rm s}.
   $$

   This is **0.13406% above 10,684.65 s**, passing the registered 5% check. A2 measures terminal work and counts the transition once. The training loop also confirms that `STEPS=25000 FULL_LAST=10000` makes steps 15,001–25,000 full-batch. The 16-second residual is an empirical transition estimate; it does not isolate compilation from throughput variation.

   **The head dependency is correctly acknowledged, but its provenance and cost allocation are wrong.**

   - [`Model.trust_linear`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/jcp-p1-scaling/vendor/quad2d/qcore.py:152) evaluates the head on checkpoint `Z_tr`, rotates its outputs, and derives the trust radius.
   - [`Bank.cold_trust`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/jcp-p1-scaling/p1common.py:169) supplies that radius to LSPG. Thus linear evolution still depends on both head and codes.
   - **The deployed `Z_tr` is refitted by the head job.** [`sep_hfit_run.py`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/separable-decoder/sep_hfit_run.py:378) receives the fitted `Z` and writes it as `Z_tr` at line 411. “`Z_tr` comes from the bank job” is false for the deployed checkpoint.

   More seriously, **2,675.56 s is the entire `full` experiment, not just the producing head arm**. Its [JSON](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-jcp-consolidated/experiments/separable-decoder/runs/dn256b/out/hfit_dn256b_full.json) contains:

   | Arm | Recorded arm seconds | Produces deployed checkpoint? |
   |---|---:|---|
   | `mid` | 360.85348 | Yes |
   | `wide` | 1,136.40957 | No |
   | `k32` | 1,169.09759 | No |

   The `mid` training timer is 338.59354 s; its arm timer also includes evaluation. The two other arms contribute **2,305.50716 s** and belong in selection work under A2’s dependency-versus-selection distinction. Shared preparation must be allocated separately. Adding coefficient extraction to the entire 2,675.56-second experiment therefore overcharges dependency cost and biases the comparison against ours.

   **The H200 head projection remains unsupported.** A bank-training throughput ratio measured between H200 and A100-40GB is not a measured conversion for coefficient extraction and head fitting on A100-80GB. Excluding the head or retaining its unscaled time does not bound that error. It also conflicts with the lane brief’s instruction to remeasure offline costs when the GPU type differs. Measure those phases on H200, or keep this accounting explicitly provisional and outside a definitive same-device full-cost verdict.

   **Ledger improvements are substantial but incomplete.** Basis-qualified keys, shared raw states, and removal of unused source-query compilation are correct. However:

   - A fresh `jax.jit` wrapper does not guarantee cold compilation; JAX has shared caches. Register a fresh-process boundary with persistent caching disabled, or a verified equivalent.
   - Specify where construction-kernel compilation is charged. Excluding compilation from every build timer while recording only query compilation would omit it.
   - Canonicalize identical source artifacts: PM-s and CP-s at mesh s must reference the same basis/rule artifacts, not duplicate costs merely because their labels differ.
   - The A_sel formula must explicitly include the nonproducing arms inside `full`, the other head experiments, competitor selection, and the stand-in for unmeasured work, each once.

   Renaming the result “full cost of the measured items” correctly discloses omissions. The selection stand-in is a scenario, not an uncertainty bound on rotation or rule-selection cost.

3. **Resampling, coefficients and rule mapping — CORRECT.**

   The orthonormal DST-I in [`engines.py`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/mr-burgers2d/engines.py:54) supports the registered formulas:

   $$
   \Phi_L=U_LEU_s\Phi_s,\qquad c_L=(L/s)c_s.
   $$

   The transformed basis has nodal amplitudes scaled by \(s/L\); the reciprocal coefficient scaling preserves the physical interpolant.

   Under restriction,

   $$
   \widetilde\Phi_L=QR,\qquad
   Qc_L=(L/s)\widetilde\Phi_Lc_s,
   $$

   hence

   $$
   c_L=(L/s)Rc_s,\qquad
   C_{L,r}=(L/s)C_s[:, :r]R[:r,:r]^\top.
   $$

   This represents the **low-pass projected field**, not generally the source field evaluated at coarse nodes.

   The combined sign-fixed factor \(R=S_2R_2S_1R_1\), rank-specific transformation, and singular-value condition-number test close round two. Upper-triangularity justifies using the prefix factorization.

   The draft `mmcore.transform_codes` agrees with these formulas. Its separate older `orth_ok` diagonal-ratio check is an implementation issue for the later code audit; it does not invalidate A2’s design requirement.

4. **Qualification, inherited evidence and J5 — CORRECT.**

   A2 corrects historical coverage and requires fresh qualification for all registered meshes and cases.

   **`p1acc` supports `q2_cases='all38'`.** [`p1common.select`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/jcp-p1-scaling/p1common.py:91) implements `all38`; [`p1acc.py`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/jcp-p1-scaling/p1acc.py:129) reads it from each configuration and intersects it with the accuracy population. Successor rules must also be present, as A2 requires.

   Calling the competitor comparison a rollout-agreement sensitivity closes the Q2-label defect.

   The recording path supports the proposed identity check: `p1core` records internal states, per-step iteration counts, and exit reasons; `p1acc` persists them. Comparing these with timed outputs is materially stronger than comparing six output vectors alone.

5. **Scoring contract — CORRECT.**

   A2 closes the round-two scoring defects: aggregate worst matching is identified honestly, median matching has a defined strict predicate, and the worst-plus-median claim is separated from the verbatim stop rule.

   Casewise comparisons retain the conditional \(2\varepsilon_{\rm ref}\) allowance. Padded lattice restriction, per-mesh reporting, reference eligibility, and the stated pooled-median interpretation remain appropriate.

   The separate verdicts must use their own admissible-combination searches, as noted in item 1.

6. **Must-fail controls — WRONG overall.**

   These are design assessments; passing still requires the registered real-data smoke.

   | Control | Verdict | Does A2 close round two? |
   |---|---|---|
   | **1. CPU preflight** | **CORRECT** | Unchanged and appropriate: CPU backend must exit 42. |
   | **2. Resampling and trust radii** | **CORRECT** | Scaling one normalized column by \(1+10^{-6}\) creates a diagonal Gram defect approximately \(2\times10^{-6}\), safely above \(10^{-10}\). Combined-QR reconstruction and rank-specific code checks close the missing checks. Shift/transpose separation still needs demonstration on the real fixture. |
   | **3. Rule mapping** | **CORRECT** | Coordinate and sampled-value checks are appropriate for a strict source-to-finer transfer. Reuse accuracy remains an empirical result. |
   | **4. Premature solver** | **NEEDS-RESTATEMENT** | The frozen fixture and fallback improve it. But moving dynamics do not mathematically imply \(\gamma>10^{-3}\): a nonzero residual can have a small projected gradient. Require measured failure on more than 1% of steps; if neither mutation achieves it, S1 fails and production remains blocked. |
   | **5. Blocked contributions** | **CORRECT** | Independent candidate construction, identical kept-state masks, and \(G,b\) comparisons address round two. The mutation’s separation remains a smoke requirement. |
   | **6. Case identity** | **CORRECT** | Full descriptor plus cohort/index checks detect the specified wrong-case presentation, including viscosity differences that an IC-distance test misses. |
   | **7. Timing injection** | **CORRECT** | Injecting into every observation fixes the median defect. Require the unmutated paired block to pass first; otherwise unrelated drift can obscure the control. |
   | **8. Decision logic** | **WRONG** | Expected outcomes lack necessary fixture preconditions, and 8(f) contradicts the global rule in a valid situation. Details below. |
   | **9. Replication mismatch** | **CORRECT** | The stored errors confirm the registered separation. The smallest dev6 absolute gap is **\(1.9084649524669234\times10^{-5}\)**, well above \(10^{-8}\). |

   **Control 8 needs explicit, separately frozen fixtures.**

   - For (a)–(c), NO-MATCH requires exhaustive resolution of the registered candidate universe. Missing combinations otherwise remain UNRESOLVED. The smoke does not automatically provide a complete production universe.
   - For (d), “two witnesses present” is not guaranteed by the smoke’s real results. Define a controlled fixture derived from measured records that demonstrably contains two admissible witnesses.
   - For (f), suppose one qualified X already has a proven NO-MATCH. Removing another X’s status leaves the global outcome **NOT FIRED**, under A2.1—not NOT EVALUABLE. To expect NOT EVALUABLE, require that no other pair proves NO-MATCH. Also test the opposite situation explicitly.

7. **Feasibility, budget and timing — NEEDS-RESTATEMENT.**

   S1 now identifies its source construction, the NNLS runtime projection is explicit, and repetitions, stored observations, outlier reporting, and device input/output boundaries close several round-two gaps. The revised budget sums to 6.0 GPU-hours; it remains a projection pending certificates.

   Three issues remain:

   - **The 12-state NNLS memory pilot does not certify the production allocation.** At \(r=384\) and 32,768 candidates, a dense contribution matrix with 120 retained states occupies **11.25 GiB**; the 12-state version occupies **1.125 GiB**, before NNLS workspaces. Require a production-shaped memory pilot or a complete allocation certificate covering that difference.
   - **Equal dimensions do not bound NNLS runtime.** Target-mesh conditioning, support growth, bans, retries, and termination can change. The ×1.5 projection is a planning estimate, not a demonstrated upper bound. Time limits protect the GPU budget, but unfinished configurations must remain unresolved for comparative conclusions.
   - **Cold-cache boundaries remain unspecified adequately**, as discussed in item 2.

   If the inherited unreduced-POD rollout diagnostics remain required, include their memory and runtime in the certificates explicitly; A2 only relabels the support-size comparison.

8. **PM at \(4096^2\) — CORRECT.**

   A2 removes the unsupported lower-bound inequality and “loses regardless” conclusion. An estimated snapshot-only cost with a descriptive sample range, unmeasured remaining setup, and unmeasured accuracy is the correct interpretation. It cannot decide the CP stop rule.

9. **Interpretation and remaining threats — NEEDS-RESTATEMENT.**

   A2 closes the central language error: an unfired stop rule no longer automatically establishes comparative cost advantage. Implementation-specific scope, strongest-tested-competitor language, and lower-envelope break-even reporting remain appropriate.

   Keep two qualifications explicit:

   - **A vacuous FIRES result is not an empirical CP match.** If every X is disqualified, the reason to abandon the comparative claim is lack of a qualified configuration of ours. State that reason beside the outcome.
   - **A cost advantage requires the least-cost admissible competitor**, exhaustive resolution of relevant candidates, and the stated accounting variant. Merely exhibiting one admissible competitor that costs more is insufficient. A2’s “exists and costs more” sentence is safe only in the context of an already established NO-MATCH.

   Until the head accounting and hardware projection are repaired, conclusions concern a provisional partial-cost model.

| Round-two item | Third-round verdict |
|---|---|
| 1. Decision rule | **NEEDS-RESTATEMENT** |
| 2. Cost model and bank projection | **WRONG** — bank formula itself **CORRECT** |
| 3. Resampling, coefficients and reuse | **CORRECT** |
| 4. Qualification and J5 | **CORRECT** |
| 5. Scoring contract | **CORRECT** |
| 6. Must-fail controls | **WRONG** |
| 7. Feasibility, budget and timing | **NEEDS-RESTATEMENT** |
| 8. PM cost estimate | **CORRECT** |
| 9. Interpretation | **NEEDS-RESTATEMENT** |

**Remaining required changes:**

- Correct deployed `Z_tr` provenance; split `full` head-job dependency work from its nonproducing selection arms.
- Measure head/extraction costs on H200, or explicitly withhold a definitive same-device full-cost verdict.
- Specify cold compilation boundaries and complete, nonduplicated compilation/selection accounting.
- Distinguish timeouts and deferrals from demonstrated construction or qualification failures.
- Enumerate admissible combinations independently for every matching sensitivity.
- Repair control 8’s fixture preconditions and global-outcome tests; make control 4’s empirical failure gate explicit.
- Certify production-shaped NNLS memory and carry runtime-limit failures into unresolved verdicts.
- State vacuous outcomes and comparative-cost claims with the qualifications above.

**Not CLEAN.**