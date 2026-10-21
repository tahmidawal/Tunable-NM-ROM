**Overall verdict: WRONG as a production-ready design; not CLEAN.** A4 closes the terminal-state default, head-production boundary and comparative-claim wording. Compilation accounting, selection-cost coverage and the NNLS memory certificate still need changes.

This audit was read-only: no files changed, no jobs submitted, no scratch files created, and no writes to main or `LAB-LOG.md`. GPU controls were inspected, not executed.

1. **Decision rule — CORRECT.**

   A4.1 closes round four’s missing default: unsuccessful construction without a registered certificate remains UNRESOLVED, including `max_iter`, KKT failure, crashes and missing checkpoints. A KKT-certified optimum that supplies a usable target remains constructible.

   A3’s independent searches and witness-based MATCH semantics survive unchanged. A timeout cannot establish NO-MATCH.

2. **Cost model and bank-fit projection — WRONG overall; bank formula and head boundary CORRECT.**

   I recomputed the phase-wise formula directly from the specified [A100 log](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-jcp-consolidated/experiments/separable-decoder/runs/push_r3a/logs/2835788.out:16):

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

   This is **0.1340568% above 10,684.65 s**, passing the registered 5% check. The training loop switches to full-batch work at step 15,001 for the registered probe. The transition residual remains an empirical overhead estimate, not an isolated compilation measurement.

   **Head boundary — CORRECT.** A4.2 charges the complete H200 extraction and producing-head processes, including preparation and checkpoint emission. The historical A_100 reconstruction is explicitly an approximation; it should retain that label because historical internal timers do not measure process startup and exit.

   **Query compilation — NEEDS-RESTATEMENT.** [A4.3](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/DESIGN.md:569) still asserts that creating a new JIT wrapper proves its executable has never been compiled. That implication is false.

   The listener is useful, but “at least one backend-compile event during `lower().compile()`” must be bound to the **deployed query executable**, not an incidental compilation during lowering. In the installed JAX source, the backend-duration event surrounds `compile_or_get_cached`; the persistent-cache-disabled requirement therefore matters as well.

   More decisively, the fallback still allows a suspected cache hit to produce verdicts “with and without” its measured time. If that measurement is retrieval time, both variants omit the unknown compilation cost. Neither establishes the intended deployment cost.

   **Required:** obtain a verified compile measurement, preferably in a fresh process with persistent caching disabled, or make affected cost verdicts explicitly conditional on pre-existing executable caches. A missing valid measurement cannot support an unconditional deployment-cost verdict.

   **A_sel — NEEDS-RESTATEMENT.** A4.4 now identifies qualification runs and historical hardware provenance, but does not fully close the artifact-union finding. Its explicit construction list names other ranks’ contributions and NNLS paths; it leaves rejected configurations’ `armdata`, trust-code preparation, query compilation and qualification-specific setup/compilation unspecified.

   This matters in the inherited implementation: [Cell.run](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/jcp-hr-baselines/pipeline.py:226) times the query separately from certification. A field named `run_seconds` does not by itself establish complete qualification cost.

   Define selection cost as the measured union of all required selection artifacts and execution work, **minus artifacts already charged in A_dep**. Specify timer boundaries so compilation is neither omitted nor counted again inside run totals. Mixed A100/H200 historical additions may remain explicitly descriptive; they are not a same-device cost measurement.

3. **Resampling, coefficients and rule mapping — CORRECT.**

   The orthonormal DST-I convention gives
   $$
   \Phi_L=U_LEU_s\Phi_s,\qquad c_L=(L/s)c_s.
   $$
   For restriction,
   $$
   \widetilde\Phi_L=U_LTU_s\Phi_s=QR,\qquad c_L=(L/s)Rc_s.
   $$
   The latter represents the **low-pass projected field**. With row-oriented, rank-specific codes,
   $$
   C_{L,r}=(L/s)C_s[:, :r]R[:r,:r]^\top.
   $$

   Independent small NumPy checks produced maximum discrepancies of \(4.16\times10^{-16}\) for prolongation at common nodes and \(3.82\times10^{-17}\) for restriction reconstruction. The draft [transform_codes](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/mmcore.py:139) agrees. A4 introduces no regression.

4. **Qualification, inherited evidence and J5 — CORRECT.**

   **LSPG depends indirectly on both `h` and `Z_tr`.** [Model.trust_linear](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/jcp-p1-scaling/vendor/quad2d/qcore.py:152) evaluates the head on stored codes to determine the trust radius supplied through `Bank.cold_trust`. Linear-span evolution does not remove that dependency. The head-producing job emits the refitted `Z_tr`.

   **`p1acc` supports `q2_cases='all38'`.** [p1acc.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/jcp-p1-scaling/p1acc.py:129) reads it per configuration through `P.select`, which implements `all38`. Both the accuracy population and registered successor/check rules must also be supplied, as already required.

   Fresh qualification and trajectory/iteration/exit identity requirements remain intact.

5. **Scoring contract — CORRECT.**

   A4 preserves separate searches for the strict worst-error stop rule, worst-plus-median claim, casewise uncertainty comparison and rollout-agreement sensitivity. Eligibility and failure handling remain unchanged. No new blocking defect.

6. **Must-fail controls — CORRECT in design, pending successful real-data smoke.**

   A4 changes none of the previously accepted controls. Their acceptance remains conditional on observed failures under the registered mutations.

   | Control | Verdict | Reason |
   |---|---|---|
   | 1. CPU preflight | **CORRECT** | Explicit CPU execution must exit 42. |
   | 2. Resampling/trust radii | **CORRECT** | Physical-value, QR, rank-specific code and deliberate column-scaling checks remain valid. Shift/transpose separation must be demonstrated. |
   | 3. Rule mapping | **CORRECT** | Coordinates and sampled values test the mapping directly; use a strict source-to-finer fixture. |
   | 4. Premature solver | **CORRECT** | Production stops unless a registered mutation actually fails Q1. |
   | 5. Blocked contributions | **CORRECT** | Independent candidates, kept-state masks and \(G,b\) comparisons remain required. |
   | 6. Case identity | **CORRECT** | Full descriptor and cohort/index binding detect the specified swap. |
   | 7. Timing injection | **CORRECT** | Injection affects every observation; the unmutated block must first pass. |
   | 8. Decision logic | **CORRECT** | Exhaustive fixture universes and explicit unresolved states remain valid. |
   | 9. Replication mismatch | **CORRECT** | The registered R128/R256 separation exceeds the comparison tolerance. |

   | Control 8 subcase | Verdict | Reason |
   |---|---|---|
   | (a) Errors \(+\infty\) | **CORRECT** | NO-MATCH in the exhaustive fixture with finite-error qualified X. |
   | (b) Costs \(1.5C_X+1\) | **CORRECT** | Mutating complete comparator costs defeats the cost predicate. |
   | (c) Q1 false | **CORRECT** | Measured failures eliminate witnesses. |
   | (d) Remove one duplicated witness | **CORRECT** | The remaining witness preserves MATCH. |
   | (e) Remove all witnesses; unresolved candidate remains | **CORRECT** | UNRESOLVED follows. |
   | (f) Missing X, with/without another NO-MATCH | **CORRECT** | Correctly distinguishes NOT FIRED from NOT EVALUABLE. |
   | (g) All X disqualified | **CORRECT** | Vacuity must be reported explicitly. |

   Control 8(d) still requires an actual measured admissible witness before duplication. Duplication tests decision logic, not independent scientific alternatives.

7. **Feasibility, budget and timing — WRONG as a complete memory certificate.**

   A4.5 explicitly withdraws the unreduced-POD diagnostics. That closes their unresolved runtime/memory allowance.

   The remaining memory calculation correctly gives:
   $$
   |\Phi|=51.5144448\ {\rm GB},\quad
   |G|=12.07959552\ {\rm GB},
   $$
   $$
   |Q|=3.01989888\ {\rm GB},\quad
   |R|=0.536870912\ {\rm GB}.
   $$
   Thus its stipulated \(\Phi+G+2(Q+R)\) allocation totals **70.707579904 GB**.

   **The unsupported step is treating \(2(Q+R)\) as a bound on all NNLS workspace.** [UpdQR.refactor](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/jcp-hr-baselines/nnls.py:277) calls `_qr_active`, which gathers active columns and performs Householder QR while persistent buffers remain live, then performs non-donated buffer updates. Deletion also constructs and factors a window and multiplies the corresponding \(Q\) block. Fixed buffer capacity alone does not bound those temporaries, backend QR workspace or their simultaneous lifetimes.

   The statement “certificate holds if the observed peak is below the bound” does not repair this: a tiled pilot can avoid the largest support-dependent paths. Observing a smaller peak on that path cannot certify unexecuted paths.

   **Required:** cover the maximum relevant refactor/deletion shapes and retries through measured allocation evidence or a conservative complete allocation bound, including host copies and backend workspace. This is an incomplete feasibility certificate, **not evidence that the job necessarily exceeds H200 memory**.

8. **PM at \(4096^2\) — CORRECT.**

   A2.8 still governs: estimated snapshot-only cost, descriptive sample range, remaining setup and accuracy unmeasured. It cannot decide the CP stop rule. A4 introduces no regression.

9. **Interpretation — CORRECT.**

   A4.6 explicitly requires an existing accuracy-admissible, Q1-qualified competitor, a resolved search and a higher minimum admissible cost before claiming comparative cost advantage. NO-MATCH without an admissible competitor is correctly labelled an accuracy/qualification separation with cost advantage undetermined.

| Round-four item | Fifth-round verdict |
|---|---|
| 1. Decision rule | **CORRECT** |
| 2. Cost model and bank projection | **WRONG overall**; bank formula and head boundary **CORRECT** |
| 3. Resampling, coefficients and mapping | **CORRECT** |
| 4. Qualification and J5 | **CORRECT** |
| 5. Scoring contract | **CORRECT** |
| 6. Controls, including 8(a–g) | **CORRECT**, pending real smoke |
| 7. Feasibility, budget and timing | **WRONG** memory certificate; diagnostic withdrawal **CORRECT** |
| 8. PM estimate | **CORRECT** |
| 9. Interpretation | **CORRECT** |

Remaining required changes:

- Verify compilation of the deployed query; replace the suspected-cache-hit fallback with a valid measurement or explicitly cache-conditional verdict.
- Define A_sel through a complete artifact union and explicit timing boundaries, subtracting costs already charged in A_dep.
- Replace the NNLS two-copy assertion and partial-pilot acceptance rule with coverage of maximum relevant workspace allocations.

**Not CLEAN.**