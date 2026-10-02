**Do not launch the production jobs yet.** The core quadrature algebra is sound; the acceptance and measurement protocol has material gaps. Audit was read-only and offline, with no JAX execution. All ten hashes in `vendor/PROVENANCE.json` match.

Paths below are relative to `experiments/quadrature-study/` unless stated otherwise.

1. **BLOCKER — Failed or missing references do not stop evaluation.**  
   `refjob.py:62` records `accepted=False` but still saves the reference and exits successfully; `qstudy.py:101` ignores acceptance metadata and silently permits missing references. Thus `afterok` does not enforce G5, and `complete=True` can accompany missing primary errors. **Fix:** require accepted reference manifests, matching case descriptors/configuration/hashes, and complete ST/S coverage before any rollout; exit nonzero on rejection.

2. **BLOCKER — G8 is neither implemented nor fully supported by saved evidence.**  
   `DESIGN.md:197` requires `audit_qs.py`, which is absent. At meshes above 256, `qstudy.py:243` and `qstudy.py:293` save restricted fields only: these cannot independently reproduce the **full-mesh** same-grid errors and distances. The claimed NumPy/JAX gradient parity and injected-control checks also lack supplied evidence. **Fix:** implement the audit before production; retain full truth fields for audit cases or independently accumulated full-mesh evidence. Define perturbations and detection tolerances explicitly.

3. **MAJOR — B4 confounds mesh scaling with GPU hardware.**  
   `DESIGN.md:210` compares 4096/256 solve times, while `DESIGN.md:227` assigns A100 to 256 and H200 to 4096. Decode subtraction cannot remove that confound. **Fix:** measure both endpoints within one H200 allocation, preferably on the same physical GPU, with interleaved timing.

4. **MAJOR — Timing warm-up is order-dependent; burn-in includes compilation.**  
   At `qstudy.py:393`, warming occurs only when randomized case `x==0` arrives. Other decode cases can therefore be timed before compilation. Separately, `experiments/mr-burgers2d/engines.py:271` creates a new jitted burn kernel each call and starts its clock before compilation: “0.1 s burn” need not mean 0.1 s of GPU work. **Fix:** precompile/warm every signature before randomization; cache and warm the burn kernel before starting its duration.

5. **MAJOR — Timed outputs are discarded, disconnecting cost from accuracy.**  
   `qstudy.py:397` records only elapsed time. Accuracy comes from phase 1; the loose timed FOM has no recorded accuracy or convergence result at all. This violates the repository’s same-invocation measurement rule. **Fix:** retain each timed result and compute errors, convergence, iterations and/or verified output hashes after stopping its timer.

6. **MAJOR — One selectable rollout arm has no B5 measurement.**  
   `make_configs.py:18` includes Gauss-192, but `make_configs.py:35` omits it from the rho ladder. Consequently `DESIGN.md:213` cannot select or reject that candidate using its specified B5 criterion. **Fix:** add Gauss-192 to the rho list and assert that every recommendation candidate has every required metric.

7. **MAJOR — G3 changes the truth solver underneath the parity comparison.**  
   `DESIGN.md:192` / `make_configs.py:73` quote historical errors measured against **`fft_tight`**; `qstudy.py:210` uses **`lean_tight`**. I checked the archived source results: the quoted values and gtol choices are correct, including the 256 parity arms, but the truth implementation differs. **Fix:** replay the historical truth path for G3, or compare saved ROM fields directly and separately gate truth-path equivalence. Scalar worst-error agreement alone is weak solver parity.

8. **MAJOR — Cohort identity and the test freeze are conventions, not enforced gates.**  
   `qstudy.py:91` records hashes but never checks the expected test64 hash or pairwise cohort disjointness. `make_configs.py:99` generates test/reference-test jobs without a frozen-selection prerequisite; `cluster/stage.py:39` does not enforce one. **Fix:** require a committed selection manifest and expected cohort hashes before staging either `reft` or `t*`; record an irreversible “opened” status outside the numerical selection process. Describe test64 as a reused historical cohort, not newly unseen data.

9. **MAJOR — G6 validates the rho target on lat64 states, not the reference rollout used by B3.**  
   `qstudy.py:289` collects only lat64 states; `qstudy.py:332` checks Gauss-640/768 there. This does not establish convergence along `gref`’s own trajectory. The supplied calibration covers initial fitted linear states, not all three settings’ reference trajectories. **Fix:** check the reference’s own reached states and gate Gauss-640 versus Gauss-768 rollout distance against a preregistered fraction of B3.

10. **MAJOR — Resource feasibility is plausible but unestablished; rollout references are not streamed.**  
    `qstudy.py:164` caches full rule blocks for every setting/arm; `qcore.py:196` materializes full continuum-test tables. Static counts give **90.0 GB** for the 4096 bank plus rollout caches alone, excluding Smolyak and workspaces; **25.6 GB** at 1024. Each 4096 dense 16-tangent field costs **2.15 GB**, with multiple intermediates required. Gauss-640’s accurate-setting host test table is **5.03 GB**, and its four trigonometric arrays total **20.13 GB**. **Fix:** bound cache lifetimes, stream reference rollout evaluations/Jacobians, and establish peak-memory and runtime budgets before the full cohort. H200/A100 fit is not disproved, but neither memory headroom nor the 12-hour runtime is demonstrated.

11. **MINOR — “Exact mesh Jacobian” needs a nondifferentiability qualification.**  
    `qcore.py:235` differentiates `vendor/hops.py:115`, whose upwind branch changes at central value zero. At a generic zero crossing, left and right derivatives differ; JAX returns the selected branch derivative. **Fix:** state “exact away from switching surfaces; JAX branch derivative at zero,” and verify directional derivatives away from crossings, including nonidentity `Tm`.

12. **MINOR — Smolyak’s retained weights do not generally sum to one.**  
    `qcore.py:84` / `DESIGN.md:75` assert unit weight sum, but `vendor/hari_quadrature.py:76` removes boundary nodes and their weights. This is valid for these boundary-vanishing integrands. **Fix:** document the exception; **do not renormalize** the retained weights.

The driver currently enforces backend/precision and same-grid truth convergence, and records G6. It does not implement the complete G1–G8 acceptance decision before marking a job complete (`qstudy.py:409`). Add a separate explicit acceptance result; distinguish development-only, cross-job and per-job gates.

| Claim | Verdict | Assessment |
|---|---|---|
| **C1** | **CORRECT** | Since \(\Phi=\psi/L\), the mesh sum is \(L[h^2\sum\psi a_h]\to L\int\psi u(u_x+u_y)\). The positive advection sign matches `engines.residual`. This is a continuum counterpart, not finite-mesh equality. |
| **C2** | **CORRECT** | Integration by parts gives the minus sign. The sine tests vanish on every boundary edge; the decoder’s Dirichlet factor also makes \(u=0\) there. Equality requires exact integration. |
| **C3** | **NEEDS-RESTATEMENT** | Point and flux Jacobians, including multiplication by `Tm`, are exact. Mesh is exact branchwise, with the zero-crossing qualification in finding 11. |
| **C4** | **CORRECT** | With matched options and `exact_steps=0`, mesh reproduces BK’s arithmetic; dense reproduces `res_d/evalJ_d`. S1 proves zero field difference and identical iterations/exits for **one fast/128 case**, not universal bitwise parity. |
| **C5** | **CORRECT** | HF constructs the evaluator synchronously; `finally` restores the monkeypatch. Dense calls HF directly. The off-mesh replacement preserves initialization, linear terms and solver policy. |
| **C6** | **CORRECT** | `[1:]` selects steps 1–50; head states pass through the folded head. Both targets use those coefficients. Point chunks retain original weights and sum contributions without renormalization (`qstudy.py:313`). |
| **C7** | **CORRECT** | For the registered meshes, both restrictions sample coordinates \(i/256,j/256\), including boundaries. Normalization uses the initial field on those same nodes. Reference identity/acceptance remains unverified by the loader. |
| **C8** | **WRONG** | The complete evaluability/acceptance claim fails: missing Gauss-192 rho, G8 evidence, reference validation and valid B4 hardware comparison. **Controls are structurally fireable:** B3 thresholds are dimensionless \(0.0005,0.002,0.005\), and rho cancels the factor \(L\). Nothing forces them to pass; actual failure is not yet demonstrated. Specify whether G7 requires both controls to fail separately in every setting. |
| **C9** | **NEEDS-RESTATEMENT** | No test-outcome access occurs in a normal development configuration, but “cannot leak” is unsupported without a freeze gate. Historical gtol values match; G3’s truth implementation and some ROM code paths differ. |
| **C10** | **WRONG** | Randomization and blocking dict outputs are correct. Warm-up/burn-in and same-invocation evidence are defective. Query-minus-decode is an operational estimate, not an exact decomposition of a fused executable; specify the median/subtraction order. |
| **C11** | **NEEDS-RESTATEMENT** | Nested bank blocks and tangent chunks avoid obvious monolithic allocations. Static storage suggests 1024/A100 is plausible and 4096/H200 potentially feasible, but full caches, compiler workspaces and runtime remain unvalidated. |

No files were changed, including the lab log.