VERDICT: NOT-CLEAN

`I` = `reports/2026-10-20-improvement-ideation`. Paths below are repository-relative.

**Round-1 disposition:** resolved at ideation-report scope: **1, 3–16, 18–21**. Partially resolved: **2, 17, 22, 23**. Remaining defects and new regressions follow.

**Verification:** all **232 `VALUES.json` entries reproduce**, and rebuilding the report in memory matches `report.md` exactly. All **seven CPU analyses reproduce** within numerical tolerance. Their file-writing calls were intercepted in memory; bytecode writes were disabled. No files, Git state, GPU jobs or cluster jobs were changed.

The bank/sine comparisons use matching lattices, normalization and reference fields. Exact matching counts **533/1308** reproduce. The stability proxy’s common normalization cancels correctly; its continuum operators, sampled states and heuristic thresholds remain limitations, now disclosed. OOD scoring correctly retains **153 cases/configuration**, uses **52 calibration cases**, excludes rejected references and includes the two invalid runs as alarms.

The arithmetic also reproduces:

- ETD estimates: **0.835842/0.463826 ms**, assumptions rather than performance bounds.
- Advection work: **75,497,472 → 37,748,736 flops**.
- Tensor storage: **1.811939/0.452985 GB**, or **0.908329/0.227082 GB** with symmetric packing.
- ECSW point ratios: **5.333/10.667**, descriptive rather than support guarantees.
- Decode ratios: **0.809/1.98/1.73**, appropriately caveated cross-job estimates. ROM decode is inside `jcp-combined/t2core.py:265–273`; spectral synthesis is inside `jcp-fom2d/psjax2d.py:61–68`.

1. **[major] F5 still defines inconsistent dynamics, and its cost estimate belongs to the old method.**

   **Evidence:** `I/report.md:229` defines both
   $R_m\dot c=-f(c)$ and $f=R_m^{-1}Q^\top F$, but its update
   $(I+hJ_f/2)\delta=-hf$
   discretizes **$\dot c=-f$**. The mass matrix has effectively been applied twice in the stated ODE. Production’s definition is unambiguous in `worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/t2core.py:9,144–145`.

   Counterexample: $R_m=2$, $f(c)=c$, $c(0)=1$. At $T=0.25$, the stated ODE gives $e^{-0.125}$; the update converges to $e^{-0.25}$—a persistent difference of **0.10369612**.

   Moreover, `I/values.py:85–88` still sums rectangular LSPG Jacobian, normal-matrix and Cholesky kernels to **1.305869 ms**. This is not an established lower bound for the newly proposed square Galerkin solve. Its generally nonsymmetric matrix needs a specified solver; old normal-equation costs do not transfer automatically.

   **Required fix:** write $\dot c=-f$ with the already mass-inverted $f$, or retain the mass consistently. Specify the linear solve and startup; rebuild the operation inventory. Define the order test against the same reduced ODE using a converged reference or three refinement levels, rather than an unspecified two-level “observed order” against E0 (`report.md:233`).

2. **[major] Replace the median-level “in-span” proxies with the available actual decomposition.**

   **Evidence:** `I/values.py:326–329` computes square-root differences of independently aggregated medians. `report.md:105` admits this is not the decomposition, but lines **9,107,275** still use it substantively; line 275 even retains “non-span.”

   I decoded `jcp-regimes/runs/rom1024/output/qacc_rom1024/R_R384_LSPG_CNR_1.npz`, projected each corresponding `store/2d/<regime>/ST/c*.npz` reference onto the lattice bank, and computed
   $\max_{t>0}\|v(t)-Pu(t)\|/\|u_0\|$ before taking medians:

   | Regime | Report proxy | Actual median |
   |---|---:|---:|
   | N0s, 8 cases | 0.844% | **0.755188%** |
   | C1, 16 cases | 4.5% | **4.606862%** |
   | C2, 16 cases | 13.3% | **13.593435%** |

   The corresponding total-error and floor medians reproduce the lane records.

   **Required fix:** generate these actual quantities from paired fields. Preserve the distinction between reduced-model, initialization and numerical errors. C1/C2 still support the broad diagnosis; the invalid aggregation is unnecessary.

3. **[major] The boundary hypothesis is still stated as an established cause.**

   **Evidence:** `I/report.md:73` says the envelope *makes* wall curvature nonzero; line **246** calls the tail a “symptom” of that envelope. Neither follows from `jcp-regimes/vendor/quad2d/npbank.py:39–42`.

   Writing $u=16x(1-x)y(1-y)n$ gives
   $$
   u_{xx}(0,y)=32y(1-y)\,[n_x(0,y)-n(0,y)].
   $$
   This can be **zero**. The envelope permits incompatible curvature; it does not force it. Nor does smooth odd extension imply finite sine bandwidth.

   **Required fix:** change these claims to hypotheses. Measure curvature of relevant decoded trajectories, not just individual bank columns, and test spectral-tail convergence. The revised parity-constrained construction in F16 is mathematically sensible; its claimed causal justification remains unproved.

4. **[major] W2 has contradictory stop rules.**

   **Evidence:** `I/report.md:282` stops the boundary arm if the quadrature improvement fails **OR** the tail improvement fails. The W2 table at **355** uses **AND**.

   An arm giving a **2× quadrature reduction but no tail reduction** therefore fails the prose and passes the table.

   **Required fix:** choose one Boolean rule and generate both presentations from the same definition. Freeze the population, aggregation, rank, reference convention and qualifying-rule search for each endpoint. Also use matched initialization/training seeds or an incumbent-recipe retraining control: comparing new fits only against the archived bank does not isolate the architectural/data change from retraining variation.

5. **[major] F12’s bootstrap acceptance rule can certify false confidence.**

   **Evidence:** `I/report.md:304` requires a bootstrap upper bound on false alarms below **0.1**. With **zero false alarms among 32 good cases**, ordinary empirical bootstrap samples contain zero false alarms exclusively, producing an upper bound of **zero**.

   The exact two-sided 95% binomial upper endpoint is instead
   $1-0.025^{1/32}=\mathbf{0.108881}$; the one-sided 95% endpoint is **0.089368**. Thus interval convention changes the verdict. The proposed 32 in-range cases do not guarantee 32 *good* cases under either error target.

   **Required fix:** specify boundary-valid binomial intervals for detection/false alarms, sidedness, minimum good/bad counts and inconclusive handling. Use paired, appropriately stratified uncertainty for the AUC difference. Freeze actual new-family distributions rather than “such as multi-bump and signed.” Saved-data leakage is now disclosed adequately; the fresh acceptance gate is the remaining defect.

6. **[major] F13 is still an unspecified adaptive method and an ambiguous combined experiment.**

   **Evidence:** `I/report.md:308` names “ETDRK4 against a third-order companion” without specifying an embedded pair, coefficients or controller. A companion is not automatically available with valid stiff-order/error-estimation behavior. Relevant missing prior art explicitly constructs robust exponential **(4,3)** pairs: [Zoto–Bowman, *Robust Exponential Runge-Kutta Embedded Pairs*](https://arxiv.org/abs/2303.12139).

   The same stop rule combines in-range time-adaptivity speed with C1 quadrature qualification. Either useful mechanism can be rejected because the other fails. The fixed-step comparator for the **1.5×** threshold is unspecified.

   **Required fix:** select a concrete embedded method and controller; specify tolerances, rejected-step handling and a tuned fixed-step comparator. Separate time-only, quadrature-only and combined arms. Define full-rollout Q2 for the adaptive policy—local rule agreement alone is insufficient.

7. **[minor] The endpoint and closure prose still compares unlike error quantities.**

   **Evidence:** `I/report.md:8,88` acknowledges the mesh mismatch but still declares that the **256²** endpoint “does not reach” the **1024²** POD pick. This is descriptive evidence, not an established same-problem comparison.

   At **131,342**, the “gap between $e_{\rm dyn}^*$ and the floor” remains mathematically misleading. The budget summary gives **1.538526%** and **1.287380%** for separately maximized, different components. Their difference, **0.251146 percentage points**, is not projection-dynamics error. The revised F17 distance-based kill criterion is otherwise much better.

   **Required fix:** remove the cross-mesh conclusion pending a matched total-error comparison. Describe F17 as reducing distance to the projected reference, whose ideal target is zero—not the projection floor. Add total-reference-error reporting before claiming improved worst accuracy.

8. **[minor] The F6 ranking omits a relevant inexpensive control, and W4 confuses advantage with capability.**

   **Evidence:** `jcp-dims/report.md:79–81` includes **r=384, M=1152, m*=16384**. Compared with the deployed **M=1536, m*=16384**, this reduces $mM$ from **25,165,824 to 18,874,368**, exactly **25%**, without increasing quadrature size. `I/report.md:87,237–240` omits this intermediate row while dismissing fewer tests. This does not prove a runtime gain, but supports a bounded timing control.

   Separately, `report.md:365` says a successful continuous-POD control should reduce “mesh-free” to “continuous evaluation.” Losing a cost comparison does not invalidate mesh-independent evolution or mesh-free evaluation.

   **Required fix:** retain the M=1152 candidate as a cheap same-integrator control, or give measured reasons to defer it. Restrict W4’s consequence to withdrawal of a **comparative cost advantage**, not a demonstrated architectural property.

9. **[minor] F7’s “CPU kill” requires unavailable evidence and hardware timing.**

   **Evidence:** `I/report.md:247` requires scoring against references on **513²**, then selecting a truncation whose synthesis beats dense decode. The inspected regime reference `store/2d/C1/ST/c000.npz` contains **six 257² fields and a 257² initial condition**, not 513² truth. CPU timing also cannot establish the H200 decode crossover.

   **Required fix:** separate the cheap CPU approximation screen from the GPU timing gate. Identify genuine denser reference fields or budget their generation; interpolating the 257² truth cannot independently validate denser-lattice accuracy. Specify what happens if no truncation beats dense decode.

10. **[minor] Some screening gates can pass without supporting their stated interpretation.**

    **Evidence:** `I/report.md:338` lets F15 pass by improving over a smaller bank while increasing total unknowns to **1024**. On C1 this could be entirely a sine-only capacity effect. The existing in-range **896-dimensional hybrid versus 2304-sine** comparison is useful, but does not establish the steep-regime mechanism.

    At **328**, F14f compares “oracle” errors without specifying nonlinear latent-solve restarts, stopping tolerances or optimization-failure accounting. At **261**, “after F1 removes T” still contradicts the revised claim that F1 only reduces time error.

    **Required fix:** add equal-total-dimension sine-only and bank controls to F15. Define a reproducible nonlinear-oracle optimization budget and failure policy for F14f. Replace “removes T” with “reduces T.”

11. **[minor] Report-rule compliance is improved but incomplete.**

    **Evidence:** title/status, LaTeX, coloured Mermaid and the expanded glossary are present; the filename departure is explicitly explained. However, `I/review_record.md:14,22` still directly embeds measured **0/108** and **153** results, bypassing existing value placeholders. These are not design constants, contrary to the blanket provenance claim at `report.md:3`.

    `report.md:88` also inserts the **1.05% r=512 `refdv` floor** into an otherwise E0 discussion without a local reference warning. `VALUES.json`’s `nonspan_*` source descriptions retain the incorrect terminology.

    **Required fix:** generate the remaining measured review-record counts, correct the provenance descriptions, and attach the first-order-reference qualification beside the r=512 comparison. Preserve the current filename exception explanation rather than claiming literal filename-rule compliance.