**Do not launch unchanged.** A0 closes several findings, but not all 12. A1 is feasible in principle; its main defects are misleading comparator ranks, weak acceptance criteria, incomplete resource checks, and an incorrect reference-uncertainty claim. No files were modified.

**(a) Previous audit: closure**

Closure means resolved in the design, not verified in an implementation. References are to [DESIGN.md](experiments/jcp-wide-bank/DESIGN.md).

| # | Verdict | Reason / remaining fix |
|---|---|---|
| 1 | **CLOSED** | A0 explicitly studies the linear-span setting and evaluates H3 using the deployed rule’s accuracy and cost. |
| 2 | **CLOSED** | A0 separates the control outcomes and restores injected faults through the numerical audit’s acceptance path. |
| 3 | **CLOSED** | Eligibility covers candidates and reference rollouts, and invalid targets make primary selection unavailable rather than silently weakening it. |
| 4 | **CLOSED** | A0 specifies the field metric, changed state population, aggregation, denominator check, and limited interpretation of sampled $\rho$. |
| 5 | **CLOSED** | All selection/certification cases are checked, with eligibility, tighter tolerances, and withdrawal/escalation rules. |
| 6 | **NOT-CLOSED** | Unpivoted thin QR with a diagonal cutoff is not a reliable rank-revealing procedure; use pivoted QR/SVD and verify projection residuals. |
| 7 | **NOT-CLOSED** | H1’s improvement and “within 0.05 pp” outcomes can overlap, and unavailable deployed arms lack an explicit outcome; specify decision precedence and an unavailable category. |
| 8 | **NOT-CLOSED** | “All subjects → FOM/decode → all subjects” does not specify paired baseline–trim–baseline comparisons, and choosing the family from the final panel compromises that panel’s independence. |
| 9 | **CLOSED** | Conclusions are restricted to the tested meshes; historical invariance is explicitly only motivation. |
| 10 | **NOT-CLOSED** | Singular values of $A$ check the linear component, not the reached-state residual Jacobian; the original claim that $M\ge R'$ determines coefficients also needs explicit correction. |
| 11 | **NOT-CLOSED** | Sequential settings help, but rule/target residency, state chunks, executable-cache release, and peak-memory accounting remain unspecified. |
| 12 | **NOT-CLOSED** | Solver pins and Richardson removal are fixed, but complete workloads, raw-data retention, and measured budgets remain incomplete; A1 adds further unsupported costs. |

**(b) A1 adversarial review**

Source-code line numbers below refer to `train2.py` at `58d83d09b`.

1. **Baseline-recipe fairness — NEEDS-RESTATEMENT.**  
   Width 2048 preserves the original width/rank ratio; 2048 POD modes is a reasonable choice, and disabling heads cannot change the already-trained bank. However, changing rank, width, POD truncation, Fourier-feature seed, initialization, and minibatch sequence does **not isolate rank**. Call this a **single-seed capacity-scaled baseline recipe**. Within-bank prefix comparisons isolate deployment width; old/new-bank comparisons do not. Report dropped POD energy and the effective whitening penalty. “Modes must exceed rank” is a safeguard against a truncated target being representable, not a proof that 2048 is sufficient.

2. **Unchanged executable and dependencies — NEEDS-RESTATEMENT.**  
   There is **no hard-coded 512 rank limit** in `train2.py`. However, copying just `train2.py` and `common.py` is insufficient: `common.py` imports sibling `paper-b3d/vendor/b3d_common.py`. Stage and hash the transitive dependencies at the expected paths. Also, `head_variants=[]` skips head optimization but **still computes head targets and writes `head_data.npz`**; account for that work or explicitly amend the code.

3. **`compare_bank` with extended ladder — WRONG.**  
   Lines 444–445 pass the new ladder to the old bank; `floors_full` line 313 uses `X[:, :Rp]`, so ranks 640, 768, 896 and 1024 silently return the **same 512-column floor under false rank labels**. This will not crash. Filter each bank’s ladder to its actual column count and assert requested ranks are valid; mark larger old-bank ranks unavailable.

4. **POD and host-memory feasibility — NEEDS-RESTATEMENT.**  
   The $18432^2$ Gram is **2.72 GB**, unchanged by increasing bank rank; chunked assembly avoids the original giant GEMM allocation, but `eigh` still needs eigenvectors and workspace. Three full $18432\times250047$ arrays would be **110.61 GB**, but the actual 33-node group has only $31^3$ points, so raw training snapshots total **78.13 GB**. The important peak is lines 102–103: raw snapshots + concatenated `allu` + `allu**2` are already about **234.40 GB**, before validation fields, other arrays, and runtime overhead. Thus 320 GB is plausible, not established. Record host MaxRSS and device peaks through generation, scaling, POD, training, ordering, and full-grid floors; preferably compute the scale by streamed sums.

5. **Ordering SVD — CORRECT.**  
   The ordering uses **all three groups’** training coefficients expressed in the 65-grid metric—not only 65-grid snapshots. At rank 1024, `Arows` is $55296\times1024$, about **0.453 GB**; `full_matrices=False` avoids a catastrophic square left factor. Preserve this construction and check numerical rank plus $T^\top R_G^\top R_GT\approx I$.

6. **B1: finite training/checkpoint selection — CORRECT.**  
   It is a legitimate failure-capable integrity gate, not evidence of useful accuracy. Require finite saved parameters, logs, floors and artifacts, and record the selected checkpoint. Note that the current code checks objective finiteness at checkpoints rather than every step.

7. **B2: inverse check/conditioning — NEEDS-RESTATEMENT.**  
   The inverse residual can fail, but merely recording a condition number permits an unusably ill-conditioned bank to pass. Add explicit rank/conditioning criteria or a tested numerical-stability criterion covering orthogonality, reconstruction, and projection residuals. A small inverse residual alone is insufficient.

8. **B3: equal-width comparison — CORRECT.**  
   A 10% permitted degradation is a meaningful, failure-capable comparison on identical fields. Compute the baseline afresh from `compare_bank`, using its unrounded floor; the metadata value is approximately **3.1216%**. Clarify that B3 is a comparison flag, since failure does not reject the bank, and that the cohort has already selected the checkpoint.

9. **B4: strict floor decrease and stopping 1c — WRONG.**  
   Nested least-squares floors are non-increasing by construction; strict decrease can fail through equality, but arbitrarily tiny improvements pass. This is a weak test of a useful dial. More seriously, a flat floor against native-grid first-order training fields does **not** rule out improved off-mesh dynamics against the refined reference. Register a minimum material floor improvement and numerical tolerance; retain the 1c accuracy experiment regardless, or label any stop as a resource decision.

10. **$R'=768$ prefix — CORRECT.**  
    This legitimately tests deployment of one ordered bank without retraining. It does not establish the performance of a separately trained rank-768 bank. Training-energy ordering also does not guarantee validation-optimal prefixes.

11. **Training smoke — WRONG as currently specified.**  
    Changing only the listed fields leaves `white_group="65"` with a single 17-node group, `order_mesh=65`, and ladder entries exceeding rank 64; the latter causes `energy[r-1]` to fail. Specify the entire smoke config: consistent white/order mesh, valid ladder/POD ranks, and retained snapshot indices needed by `(0,10,20,30,40,50)`. The tiny smoke must also exercise comparison against a narrower bank; it cannot establish production memory feasibility.

12. **1c rules and converged target — NEEDS-RESTATEMENT.**  
    Gauss $56^3$ versus $48^3$, plus the $80^3/64^3$ target check, is a reasonable **empirical convergence test**, not an established converged reference for the new network. Retain withdrawal on failure and specify a successor to $56^3$—A0’s escalation to $56^3$ is now a no-op. Apply checks at every mesh/setting and pin the newly introduced rules. Reuse the Gauss-48 result as both candidate and check without double-counting its execution.

13. **Meshes, tensors, and inherited gates — NEEDS-RESTATEMENT.**  
    Nodes 65/129 are correct. However, H4 passing at both meshes does not itself demonstrate mesh invariance: add paired shared-node field distances and mesh-specific $\rho$ results. Tensor omission above 512 is reasonable, but describe it as the registered resource policy—34 GB of tensor storage alone does not prove it cannot fit on H200. Define replacements or scope for K-eval G2/G4, whose vendor implementations require tensors, at the larger ranks. The optional 257-node extension also needs a memory gate: its full rank-1024 mesh bank alone is about **136 GB**.

14. **H4 decision rule — NEEDS-RESTATEMENT.**  
    A 20% improvement at both meshes is a sensible development bar. Specify the deployed-family selection rule, freeze it before confirmation timing, give unavailable outcomes for failed gates/eligibility, and resolve overlap between “20% improvement” and “within 0.05 pp.” Interpret the result as the joint $(R',M)$ dial, not a rank-only effect.

15. **“Possibly reference-limited below 1.7%” — WRONG.**  
    The [prior report](</home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-01-burgers3d-offmesh-quadrature.md:41>) identifies **1.70% on one probe case**, not a cohort-wide upper bound; it also reports **2.28%** on held-out cases. Neither is a continuum-error bound. Remove the threshold-based implication that larger gains escape reference uncertainty. Label every physical-accuracy interpretation reference-limited until matched reference convergence is established; distinguish relative percentage improvement from percentage-point error change.

16. **GPU-hour estimates — NEEDS-RESTATEMENT.**  
    **J3 ≈3 H200-hours is plausible as a planning estimate**, but QR backpropagation, full-grid validation, CPU ordering, floors and memory pressure need measurement. The step-50 `seconds` field includes POD preparation, compilation and validation; dividing it by 50 would badly overestimate steady-state training time. Use synchronized warm-step timing or checkpoint differences with overhead accounted for.  
    **J4’s 1.5-TFLOP Jacobian arithmetic is correct**, but 25 ms assumes about 60 TFLOP/s sustained, and 28 Jacobians/query ignores potentially substantial adaptive-start work. Measure the full largest query and enumerate compilation, checks, certification, timing repetitions, floors and optional trims. The quoted 3–4 hours is unvalidated, not demonstrably impossible.

| Area | Verdict | Required disposition |
|---|---|---|
| A0 closure | **6 CLOSED / 6 NOT-CLOSED** | Fix rank checks, verdict precedence, timing, memory and workload contracts |
| Baseline recipe | **NEEDS-RESTATEMENT** | Capacity-scaled, single-seed experiment |
| Rank-1024 training path | **NEEDS-RESTATEMENT** | Complete dependencies and production resource checks |
| Comparator ladder | **WRONG** | Reject/filter ranks above each bank’s width |
| Ordering / 768 prefix | **CORRECT** | Preserve training-only ordering and prefix interpretation |
| B1 / B3 | **CORRECT** | Integrity gate / benchmark comparison flag |
| B2 / B4 | **NEEDS-RESTATEMENT / WRONG** | Numerical-stability gate / material gain without invalid stopping inference |
| Tiny smoke | **WRONG** | Supply a fully consistent config |
| 1c quadrature / meshes / H4 | **NEEDS-RESTATEMENT** | Convergence, wide-rank gates, cross-mesh checks, explicit decisions |
| Reference threshold | **WRONG** | Remove the purported 1.7% uncertainty boundary |
| GPU-hour estimates | **NEEDS-RESTATEMENT** | Plausible placeholders pending measured workloads |