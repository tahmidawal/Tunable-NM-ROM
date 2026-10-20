VERDICT: NOT-CLEAN

Audit scope: all **172 generated value entries reproduce**, and regenerating the report **in memory** matches `report.md`. All six CPU analyses reproduce exactly or to roundoff; all **38 reference-array hashes** match `refdv`. These checks establish reproducibility, not validity of the interpretations below. No files were written and no GPU or cluster jobs were launched.

Paths below are repository-relative. `I` denotes `reports/2026-10-20-improvement-ideation`.

1. **[blocker] The new analyses’ reference scheme is mislabelled.**  
   **Evidence:** `I/report.md:3` calls `refdv` second-order in space and time. `worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/vendor/quad2d/refjob.py:3–9` specifies **sign-upwind + backward Euler**, \(L=8192\), \(\Delta t=0.0003125\). All 38 analysed arrays match that run’s recorded hashes. `jcp-combined/DESIGN.md:342` explicitly calls these references first-order.  
   **Fix:** Retract “second order.” Label these as projection errors against first-order numerical solutions. Recompute against E0 before using their tiny sine floors, compression ratios or spectral tails as evidence about continuum solutions. Similar bank medians do not validate the references’ high-frequency content.

2. **[blocker] “Ideal-dynamics error” is not a lower bound over solvers or dynamics.**  
   **Evidence:** `I/report.md:8,81,327` treats **1.54%** as an unavoidable error within the bank span. `worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/report.md:86,448` defines it as the distance of **one converged Galerkin trajectory** from the projected reference. The actual projection floor quoted here is **1.29%**, below ECSW’s **1.35%**. Moreover, `values.py:182–195` takes budget numbers from **256²**, while the ECSW comparison is **1024²**.  
   **Fix:** Remove the impossibility claim and glossary definition “a floor for any solver change.” State the particular dynamics, mesh and reference population. Only the projection floor bounds every coefficient trajectory in that span.

3. **[major] The decode “ceiling” is a cross-job estimate, not a universal bound.**  
   **Evidence:** `I/values.py:58–65,127–146` combines P1 component medians with fom2d query medians. It even uses **R384 initialization at 1024² for R128 at 4096²**: \(0.197302\) ms instead of \(0.168408\) ms. Decode really is timed: `jcp-combined/t2core.py:265–273`; PS synthesis also is timed: `worktrees/2026-10-18-jcp-fom2d/experiments/jcp-fom2d/psjax2d.py:61–68`. The timing boundary is therefore sound; the bound is not.  
   **Fix:** Label **0.809/1.97/1.72** as illustrative estimates for the current dense decoder and workload. Remove “no box ROM” and “r ≥ 384” impossibility claims (`report.md:64,314`). A changed decoder or improved accuracy requiring a different matched FOM invalidates that extrapolation. Establish any implementation-specific ceiling within one job.

4. **[major] The explicit-stability calculation cannot “pass” the proposed timestep.**  
   **Evidence:** `I/analysis/explicit_stability.py:34–61` uses continuum eigenvalues, quadrature-built \(A\), projected reference states and only three times. Production uses mesh-projected \(A\) and discrete eigenvalues (`jcp-combined/vendor/quad2d/vendor/hops.py:35–57`). Recomputed continuum/discrete selected-set symmetric differences are **2 modes at M=512**, and **4 at M=1536 on L=256**; the maximum eigenvalue discrepancy there is **2.405%**. The common normalization factor cancels consistently—there is no missing-factor-\(L\) error—but the operators differ.  
   Also, \(2.8/\rho(J_a)\) is not a stability bound for general complex spectra, non-normal systems or ETDRK4 splitting. Positive-real eigenvalues are not explained by “non-normal effects” alone.  
   **Fix:** Replace “all states allow” and “already passed” (`report.md:136,155`) with “proxy thresholds exceed.” Check the actual frozen operators and reached trajectories, including stiff regimes, and test timestep convergence. Use \(\varphi(-\Delta t\,\nu D)\), correcting the positive argument in line 133.

5. **[major] F2 incorrectly says the bank is not Dirichlet-exact.**  
   **Evidence:** `I/report.md:168`. The evaluator explicitly multiplies every feature by \(16x(1-x)y(1-y)\): `worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/vendor/quad2d/npbank.py:39–43`. Linear rotation preserves zero boundary values.  
   **Fix:** Remove this assertion. Check mass conditioning separately; lack of orthonormality does not establish ill-conditioning. Mass whitening should precede a scale-dependent condition-number kill criterion.

6. **[major] The ETD cost model assumes its principal performance gain.**  
   **Evidence:** `I/values.py:89–99` substitutes the **r=32** advection timing for **r=384 bank-tested Galerkin**, producing **0.464 ms**, **9.6×** and **2.9×**. There is no measurement supporting that substitution. At fixed \(m=16384\), direct advection evaluation changes approximately from
   \[
   2m(2r+M)=75{,}497{,}472
   \quad\text{to}\quad
   6mr=37{,}748{,}736
   \]
   flops: **2×**, not the Jacobian’s 4×. ETD does not form that Jacobian. Stage transforms, combinations and query-dependent coefficient preparation are not fully modelled.  
   **Fix:** Separate operation counts from latency assumptions and present a range. Crucially, benchmark **offline precontraction of \(R_m^{-1}Q^\top\Psi^\top\)**: it removes the online \(M\)-projection from the existing Galerkin RHS without changing the test formulation. This cheaper idea is missing.

7. **[major] The sine-decode rejection does not follow from its measured tail.**  
   **Evidence:** `I/report.md:221`; `analysis/decode_spectral.py:32–39` measures truncation of **projected reference fields**, not deployed trajectories or full-resolution fields. A bank tail is not necessarily added error against truth. For orthogonal sine projection \(P\),
   \[
   \|P\hat u-u\|^2
   \le \|\hat u-u\|^2+\|(I-P)u\|^2.
   \]
   The stored values bound the worst truncated projection error by **1.22291%**, versus **1.21759%** before truncation—not an obligatory additional **0.237%**.  
   **Fix:** Move F7 to a cheap diagnostic: directly score truncated deployed trajectories, sweep truncation size, validate on a denser lattice, then benchmark synthesis. A finite-grid tail also does not prove unavoidable aliasing under adequately oversampled projection.

8. **[minor] “Modes needed” are rounded-up search-grid results, not minima.**  
   **Evidence:** `I/analysis/bank_vs_sine.py:43,60–77` searches strides of 16 or 64. Recomputing every prefix gives worst-case matching counts **533**, not 576, for r=128; **1308**, not 1344, for r=384. Per-case median ratios become **1.01172** and **0.638021**, rather than 1.1 and 0.67.  
   **Fix:** Say “smallest tested prefix” or use exact prefix searches. The bank/sine comparison itself uses the same lattice, references and normalization correctly; its limitation is the first-order reference and finite-lattice scope.

9. **[minor] Two discretization counts are wrong.**  
   **Evidence:** `I/report.md:36,42` counts PS nonlinear points as \(36^2,72^2\). `jcp-fom2d/pscore.py:31–42,64–82` evaluates interior arrays of **35²=1225** and **71²=5041** points. The latter ratio is **16384/5041=3.25015**, not 3.16049.  
   `analysis/floor_vs_upwind.py:22` uses \(h=1/256\), whereas `worktrees/2026-10-08-jcp-smooth-bank/experiments/jcp-smooth-bank/DESIGN.md:57` specifies **256 nodes including boundaries**, \(h=1/255\). Corrected viscosity ratios are **0.0666984 median**, **0.344400** for the worst-floor case.  
   **Fix:** Correct both counts. Label the `L256` FOM error as a nearby generator comparator, not the identical training discretization.

10. **[major] F11 does not isolate its proposed cause.**  
    **Evidence:** `I/report.md:98,242,254–255` infers “more width will not fix it” from prefixes of one frozen bank, then changes **data fidelity and regime coverage simultaneously**. Better fit to the training generator can result from distribution matching; it does not isolate upwind smearing from temporal error. The kill condition uses **AND**, permitting continuation when only one desired improvement occurs.  
    **Fix:** Limit the width conclusion to the tested prefixes. Use matched fidelity-only and coverage-only controls, consistent \(t_0\) normalization, and fresh evaluation cases. Specify whether both floor targets are required; if so, failure of either must stop the experiment. Failure of the smearing diagnostic must not automatically kill the separate coverage hypothesis.

11. **[major] The head-only oracle target is impossible with the existing bank.**  
    **Evidence:** `I/report.md:298` demands a C2 oracle error below **9.88181%/3=3.29394%**. The full r=512 span already has median C2 floor **9.25480%** (`jcp-regimes/checks/analysis.json`, `projection_floor["C2|R512"]`). The existing head maps into that frozen span: `worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/DESIGN.md:30–37`.  
    **Fix:** Make F14f explicitly dependent on a new/enriched bank or a decoder outside the existing span. Compare head and linear oracle errors within the same new representation, on held-out states. Head retraining alone cannot pass.

12. **[major] The OOD headline changes targets across configurations and omits failures.**  
    **Evidence:** The AUC range **0.876671–0.986601** reproduces, but `analysis/ood_indicator.py:38` excludes missing-error/invalid cases. GAL has **156**, not 158, observations; C2 cases **8 and 13** are excluded. “Bad” thresholds vary from **0.103835%** for GAL to **2.03554%** for R128-BE. For the saved common **1%** target, AUC is **0.687856–0.996439** and detection **0.413793–0.981481**.  
    **Fix:** Report denominators, excluded failures, both targets and per-configuration results. Keep failed queries in deployment accounting as mandatory fallbacks. The disclosed in-sample threshold reuse is real; additionally disclose exploratory selection among four indicators. Freeze both the alarm threshold and error target before fresh evaluation.

13. **[major] F12’s “pursue” ranking conflicts with its own economic kill criterion.**  
    **Evidence:** `I/report.md:275` requires ROM + indicator + fallback to beat FOM alone. For the cited r=384 full-field workload, the ROM alone costs **97.1/11.7≈8.30×** the FOM; doubling cannot rescue it. Rule agreement also cannot detect shared representation error.  
    **Fix:** Apply the cost screen before allocating W3. Pursue only under a specified workload where the first-stage ROM is cheaper, or explicitly pursue accuracy monitoring without a speed claim. Include a cheap parameter/steepness classifier control, per-regime miss rates and a fixed mixture distribution.

14. **[major] F2’s rollout-distance kill tests agreement with a different model, not correctness.**  
    **Evidence:** `I/report.md:173` rejects bank-tested Galerkin if it differs from sine-tested GAL-CN by \(2.5\times10^{-4}\), calling that Q1. Changing the projection changes the reduced ODE; an accurate improvement can legitimately exceed that distance.  
    **Fix:** Apply solver convergence checks within each formulation, quadrature convergence within each formulation, and compare both against the same references. Define a field-space projection norm and normalize/whiten the basis before conditioning checks.

15. **[major] F3’s point ratio and kill test do not establish qualified hyper-reduction.**  
    **Evidence:** The arithmetic **16384/3072=5.333** and **16384/1536=10.667** is correct, but ECSW uses stencil gathers and a different projected operator. `I/report.md:189` tests \(\rho\) only on saved states. `jcp-regimes/DESIGN.md:185–186` requires successor-rollout distance, reached-state \(\rho\), successor validity and check-rule convergence.  
    **Fix:** Treat node ratios as descriptive, not a feasible support prediction. Explicitly exclude dev6 from fitting if calling it held-out; test own reached trajectories and the complete Q2 contract. Account for stencil storage and contributions.

16. **[major] F4’s tensor times are bandwidth lower bounds; its kill test targets the wrong quantity for F1.**  
    **Evidence:** `I/values.py:294–301` correctly obtains **1.81194 GB / 4.8 TB/s = 0.377487 ms**, and **0.452985 GB / 4.8 TB/s = 0.0943718 ms**. These omit attainable bandwidth, contraction arithmetic and additional traffic. `report.md:203` tests only Jacobian speed, although ETD needs nonlinear evaluations.  
    **Fix:** Benchmark nonlinear value, Jacobian and complete query separately. Include symmetric tensor storage/precontraction. Exact assembly preserves the polynomial mathematically; a CP approximation requires new accuracy, stability and quadrature-equivalence checks. An \(m\)-term representation establishes CP rank **at most** \(m\), not exactly \(m\).

17. **[major] F5 is not yet a defined numerical method.**  
    **Evidence:** `I/report.md:209` names a “Rosenbrock-W-type” one-pass method but supplies no coefficients, order conditions or acceptance/error mechanism. `values.py:85–88` sums kernels to **1.31 ms**; the current component model is **2.11 ms**, versus measured **1.775 ms**, already about **19%** discrepant.  
    **Fix:** Specify the actual update and its order before quoting gains. Count gradient formation, solves and certification explicitly. Q1 status alone cannot establish equal accuracy for a changed integrator; require reference errors, refinement and end-to-end timing.

18. **[major] The test-space diagnosis reverses what the data establish.**  
    **Evidence:** `I/report.md:165` attributes quadrature demand to bandwidth because \(m^*\) falls as \(M\) grows. In `worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/report.md:75–81`, r=384 needs **36864** points at M=512/768 but **9216** at M=2048/3072. That is not evidence that lower test bandwidth reduces quadrature demand; conditioning and rollout qualification intervene. Lines 119 and 217 also say **0/108 fail**, whereas the source says **0/108 pass**.  
    **Fix:** Correct the failure count to **108/108**. Separate bandwidth diagnostics from conditioning/qualification. Square-test failure does not by itself reject every intermediate \(r<M<4r\) choice.

19. **[major] W4 is not a complete break-even experiment.**  
    **Evidence:** `I/report.md:289–291,310` compares selected construction figures and tests interpolated-POD online behavior at one endpoint. The baseline report explicitly says comparable bank per-mesh setup costs are missing (`worktrees/2026-10-06-jcp-hr-baselines/experiments/jcp-hr-baselines/report.md:533`). Full-field decode also grows with mesh; H-cost-N concerns evolution (`final/report.md:25`).  
    **Fix:** Register total cost versus both mesh count and queries per mesh: data, training, validation, rule selection, operator preparation, compilation, memory and queries. Include a continuous/interpolated POD rule reused across meshes where feasible. The endpoint 1.25-band test neither establishes break-even nor falsifies continuous evaluation as a property.

20. **[major] Several rankings overreach the campaign and cited literature.**  
    **Evidence:** `I/report.md:82,133` turns an observed **7.2× time-package effect** into a general LSPG penalty; the cited paper instead establishes conditional equivalences and potentially non-monotone timestep/error behavior. [Carlberg–Barone–Antil](https://arxiv.org/abs/1504.03749).  
    Line 233 dismisses NiTROM as “linear-ish”; its examples include nonlinear CGL and lid-driven cavity flow at **Re=8300**. [NiTROM manuscript](https://albertopadovan.github.io/manuscripts/PadovanVollmerBodony_SIADS2024.pdf). Fixed geometry does not exclude transport or useful coordinate dilation; non-box geometry does not eliminate every fast solver. The NS conclusion extrapolates from one limited pilot.  
    **Fix:** Replace categorical drops with scoped priorities. Consider local banks/local hyper-reduction, coordinate dilation, dynamics-aware projection and sine enrichment—the latter already lowers the stored worst floor from **1.21759% to 0.636408%**. Localized hyper-reduction has direct prior art: [Peherstorfer et al.](https://kiwi.oden.utexas.edu/papers/Localized-DEIM-Peherstorfer.pdf).

21. **[major] Qualification and developmental caveats do not travel with the quoted results.**  
    **Evidence:** `I/report.md:70–84` omits H-base’s **developmental, in-sample, unadjusted, unresolved-budget** restrictions stated in `reports/2026-10-17-full-experiment-plan/final/report.md:62–65,82`. C1/C2 rollout errors in lines 95–98 are unqualified. The **3.94×** batching figure concerns a speed3d winner later found quadrature-unqualified; it is throughput, not single-query latency (`jcp-speed3d/report.md:195–212`; `LAB-LOG.md`, qualify completion entry).  
    **Fix:** Attach these restrictions directly to each affected table and ranking. Preserve the distinction between descriptive performance, numerical qualification and held-out confirmation.

22. **[major] The remaining kill criteria are not fully pre-registrable or decisive.**  
    **Evidence:** `I/report.md:155,255,275,279,286,293,307–310`. W1 conflates integrator and projection changes; F13 has no explicit criterion; F14d has no batched FOM success threshold; F14a’s “finest practical mesh” is discretionary. W2’s AND condition and W3’s “detection <0.7 at false alarm ≤0.1” do not unambiguously reject either failed target.  
    **Fix:** Freeze method arms, comparator selection, accuracy tolerance, failure handling, sample sizes and timing summaries. Test F1/F2 separately before combining them. Require **detection ≥0.7 AND false alarm ≤0.1**, with uncertainty. Use accuracy-matched FOM ladders and both throughput and latency for batching; geometry pilots must include optimized FOM and POD controls.

23. **[minor] Report-rule compliance is partial.**  
    **Evidence:** `CLAUDE.md:142–164`. The title, status, LaTeX displays, coloured Mermaid and closing glossary are present. However, `report.src.md` directly embeds numerous numerical settings, budgets and thresholds; the claim that **every number** comes through `values.py` is false. The glossary omits E0, dev6/val32, NNLS and several named methods; it also contradicts itself about N versus N−1 sine modes (`report.md:328–329`). The standalone filename is `report.md`, contrary to the literal naming rule.  
    **Fix:** Distinguish sourced measurements from registered design constants and trace both. Add inline provisional labels and missing definitions, correct the mode convention, and either adopt the required filename or document the directory-layout exception.