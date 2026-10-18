**Overall verdict: NEEDS-RESTATEMENT before this becomes an executable pre-registration.** Most headline measurements match the evidence, but several mathematical claims are overstated, some controls cannot discriminate the claimed failure, and the shared sealed-cohort policy permits cross-lane leakage. The main plan also contains experiments missing from appendix 04.

I made no file changes and ran no jobs, including no lab-log update.

References below use [Plan](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-06-jcp-offmesh-paper-plan.md), [3D](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-01-burgers3d-offmesh-quadrature.md), [2D](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-01-burgers2d-offmesh-quadrature.md), and [Hari](/home/tahmid/Downloads/quadrature-study-2026-09-30/quadrature/results/SUMMARY.md).

**1. Numbers and their context in Plan §§2–3 and 7**

| Item | Verdict | Source check and correction |
|---|---|---|
| §2: mesh invariance across **64³–256³** | **CORRECT** | 3D §4: selected-rule worst-error ratios are 1.003 and 1.001 for the two widths. This is approximate invariance against the stated first-order reference, not identical solutions. |
| §2: lattice beats Gauss at equal point count; **lattice-256** and Smolyak fail; **one random shift** | **CORRECT** | Matches 3D §§3 and 6. Crucial population distinction: the quadrature ladders in §3 are from **24 certification cases in validation jobs**, not the 32-case held-out rollout cohort. The “solid” label should not inherit the preceding row’s held-out description. |
| §2: **4.30 GB → 0.34 GB**, `gl24`, **R′=512** | **CORRECT** | Exact displayed values in 3D §5. These are advection-data storage, not total process/GPU memory. |
| §2: query faster in **all six cells** | **CORRECT** | Selected rule versus tensor: 34.1<47.9, 11.1<12.2, 40.6<52.0, 12.1<14.1, 76.7<90.1, 33.0<35.1 ms. Same-allocation comparisons within each cell. |
| §2: 2D per-case spread **≤0.011 pp** | **CORRECT** | Exact test-cohort maximum for the reported linear-setting checks: continuum rollout, Gauss 64², and Fibonacci 6765. It is **spread of ST error**, not a bound on cross-mesh field distance or every off-mesh rule. |
| §2: 2D solve time flat in mesh size | **CORRECT** | 2D test result: ratios 0.972–1.040 across 40 arms on one H200. Full-query time is not flat. |
| §2: off-mesh is more accurate than tensor **and same-mesh FOM** | **NEEDS-RESTATEMENT** | This needs mesh/width qualifiers, not just “fragile.” In 3D the best FOM is more accurate at 256³. The selected-rule/tensor difference at 256³, R′=256 is explicitly unresolved at the reference’s uncertainty scale. |
| §2: slower at **64³** | **CORRECT** | Correct under the report’s same-grid matching rule. Refined-accuracy speedup is undefined there because no tested same-mesh FOM matches the selected off-mesh accuracy. |
| §2: **2.21× → 2.59× at 256³** | **NEEDS-RESTATEMENT** | Exact numbers, but identify **R′=512, tensor→selected `gl24`, same-grid matching rule**. These are not generic matched-continuum-accuracy speedups. |
| §2: **6.45×** | **CORRECT** | Exact 3D §5 value for selected `gl24`, R′=512, using the refined-reference matching rule. The stated fragility is appropriate. |
| §2: Hari’s other PDEs use **R=128**, CPU | **CORRECT** | Supported by Hari’s setup/limitations. The CPU qualification is appropriate for these reduced-model experiments; Hari separately used a GPU for the 3D FOM. |
| §3: **23 objections** | **CORRECT** | Appendix 05 enumerates 23. |
| §3 F1: **513³ reference** | **NEEDS-RESTATEMENT** | Source says **513 nodes per axis, 512³ cells**. The plan elsewhere labels meshes by intervals/cells. Use “513-node-per-axis reference” to avoid mixing conventions. |
| §3 F2: bank saw **129³ data** | **NEEDS-RESTATEMENT** | Source says fields **up to 129 nodes per axis**. This supports the training-resolution confound, but not the causal assertion that super-resolution explains the result. |
| §7: **8.3× / 9.1× / 12.8×** per doubling | **NEEDS-RESTATEMENT** | Reproduced by dividing **rounded table entries**, specifically 3D §3, **64³, R′=512**: 0.083/0.010, 0.010/0.0011, 0.0011/0.000086. They are not exact copied measurements and do not summarize all meshes. Generate ratios from JSON. |
| §7: those ratios establish **≈m⁻³** | **WRONG** | Three finite-range ratios from one shift and one mesh do not establish an asymptotic rate. Their effective exponents are approximately 3.05, 3.18, and 3.68. They are compatible with several explanations, including pre-asymptotic behavior. |
| §7: **C1–C11 already mapped into E1/E2/E3/E7** | **WRONG** | Several checks are missing or materially altered. C7’s synthetic perturbations/stability measurements are absent; C5’s diagonal-versus-anisotropic test analysis is absent; C6’s explicit leading-coefficient comparison is absent. C9 belongs in **E4**, and its R′=1024 sweep is absent. C11’s convergence-slope test is replaced by an invariance threshold. |

The remaining numerals in these sections are experiment identifiers, section references, or proposed method orders—not historical measurements.

**2. Theory in [03-theory.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-06-jcp-offmesh-paper-plan/03-theory.md), §§1–2**

| Claim | Verdict | Assessment |
|---|---|---|
| Lemma 1.1: a fixed SiLU bank has some complex neighborhood of analyticity around the real compact domain | **CORRECT** | Real preactivations avoid SiLU’s complex poles; compactness and continuity give a nonzero neighborhood for a fixed finite network. |
| Lemma 1.1: the **displayed explicit strip-width lower bound** | **WRONG** | The proposed induction does not establish it. Real-axis preactivation bounds do not control complex excursions. A product over all layer weights does not enforce every intermediate layer’s pole-avoidance condition: small downstream weights cannot undo a pole encountered upstream. |
| “The strip shrinks like inverse RFF scale” | **NEEDS-RESTATEMENT** | A possible conservative dependence with other quantities fixed, not an established law across independently trained networks. Weights, biases, composition, and cancellations matter. |
| SIREN, Gaussian RBF, fixed-sine banks are entire | **CORRECT** | For the stated finite architectures with entire operations and polynomial masks. Entire does not mean band-limited or imply one universal growth bound. |
| Theorem 1.2: geometric tensor-Gauss convergence from analyticity on Bernstein ellipses | **CORRECT** | The telescoping argument with positive weights is sound. |
| Theorem 1.2: the **displayed constant and exponent** | **WRONG** | The cited theorem uses **n+1 points**. For p points, its factor is proportional to ρ⁻²ᵖ⁺²/(ρ²−1), not the displayed ρ⁻²ᵖ/(ρ²−1). The bound on normalized tests also drops their 2^(d/2) factor. Further, the strip argument only covers real coordinates inside [0,1], while a Bernstein ellipse extends beyond the endpoints. [Trefethen, Theorem 19.3](https://people.math.ethz.ch/~hiptmair/Seminars/RAP_22/TRE13.pdf). |
| Corollary 1.3(i): entire banks have super-geometric asymptotic Gauss convergence | **CORRECT** | As an asymptotic statement for fixed entire integrands. |
| Corollary 1.3(i): the particular frequency/onset formula applies to every entire bank | **WRONG** | That optimization needs a specified complex-growth bound. Finite trigonometric sums, Gaussian RBFs, and composed sine networks have different growth. “Bank’s own growth rate” cannot universally be replaced by one additive frequency. |
| Corollary 1.3(ii): infer the actual analyticity strip from the observed quadrature slope | **NEEDS-RESTATEMENT** | The quoted arithmetic is approximately consistent, but it gives an **effective fitted convergence parameter**, not a certified strip width. An upper bound need not be sharp; cancellation, changing worst states, and pre-asymptotic behavior can alter slopes. |
| `gl24` at ρ≈10⁻² is “converged” | **NEEDS-RESTATEMENT** | It may meet an operational rollout criterion. It is not a converged continuum quadrature target. |
| Lemma 1.5: integrand vanishes to **exactly** second order on every face | **NEEDS-RESTATEMENT** | It vanishes to **at least** second order. The leading coefficient can vanish for particular states/tests; opposite-face derivatives can also match. |
| Generic periodic extension is C¹, with second-derivative jumps | **CORRECT** | For nonzero, unmatched leading face coefficients. This explains why analyticity inside the box does not automatically give periodic analyticity. |
| Product Fourier decay bounded by ∏ max(1,|hⱼ|)⁻³ | **NEEDS-RESTATEMENT** | A plausible provable bound for this analytic, boundary-vanishing class, with the mixed-derivative/face recursion completed. Define the Fourier coefficient class precisely; coefficient decay is not automatically membership in a same-index Korobov Hilbert space under every convention. |
| Theorem 1.4: a suitable lattice family can obtain O(m⁻³⁺ε) from that bound | **NEEDS-RESTATEMENT** | Defensible as a **conditional upper bound** after specifying the lattice construction and its dual-lattice bound. It does not establish an exact rate for every bank, shift, or finite ladder. |
| “A fixed sine bank is still **only** m⁻³” | **WRONG** | An upper bound is not a lower bound. Additional boundary matching and Fourier cancellation can give faster convergence or exact integration for suitable finite spectra and rules. |
| Corollary 1.6: universal Gauss/lattice onset ratio and crossover | **WRONG** | These are heuristics, not a theorem. Lattice resolution depends on the generating vector and dual lattice; analyticity does not supply a hard frequency box. The crossover cannot be inferred from ρ alone with an unspecified additive constant. |
| Proposition 1.7: tent reflection makes this integrand smoother | **CORRECT** | Because the first normal derivative vanishes at both walls, the reflected first derivative matches; even second derivatives match automatically. The generic periodic extension improves from C¹ to C². Hari’s blanket “adds a kink” explanation is inappropriate here. |
| Proposition 1.7: therefore the deployed tent lattice converges like m⁻⁴ | **NEEDS-RESTATEMENT** | Fourth-order Fourier coefficient decay is a reasonable next bound under the stated regularity, but the quadrature rate still needs the relevant function-space and lattice-quality argument. Tent-rule theory explicitly depends on these choices. [Dick–Nuyens–Pillichshammer](https://arxiv.org/abs/1211.3799). |
| Tent transform necessarily multiplies onset m by 2ᵈ and explains the measured loss | **WRONG** | Local coordinate compression does not establish that universal point-count penalty. Reflection changes Fourier content and its interaction with the particular lattice. The numerical loss remains unexplained. |
| Proposition 1.8(a): tensor polynomial exactness requires at least nᵈ nodes | **CORRECT** | The polynomial-vanishing-at-the-nodes, then squaring, argument is valid. |
| Proposition 1.8(b): that lower bound explains an O(1) Smolyak error for these integrands | **NEEDS-RESTATEMENT** | Failure of polynomial exactness does not imply a nonzero error lower bound for a particular sine integrand. Cancellation matters. The claimed Bessel-coefficient description is also too uniform: parity, zeros, and the turning-point region matter. |
| Proposition 1.8(c): negative weights explain observed LM failures | **NEEDS-RESTATEMENT** | They can amplify perturbations. Causation is not established by the current experiments; positive weights do not by themselves guarantee nonlinear-solver stability either. |
| Scrambled Sobol RMSE and MC rates | **NEEDS-RESTATEMENT** | MC’s m⁻¹ᐟ² scaling and smooth-integrand scrambled-net RMSE bounds are appropriate with their assumptions. A single scramble’s observed ≈m⁻¹ slope is not a universal deterministic rate. |
| Proposition 2.1: local sign-upwind expansion | **CORRECT** | The leading term is −h|u|uₓₖₓₖ/2. The stated sign is correct. |
| Proposition 2.1: projected expansion with an unqualified O(h²) remainder | **NEEDS-RESTATEMENT** | Fix the normalization: N and E₁ include L^(d/2), so absolute remainder scaling must include it or use normalized quantities. Also justify composite quadrature error using the actual piecewise-smooth structure; Lipschitz continuity alone does not give the stated generic second-order quadrature bound. |
| Corollary 2.2: different continuum and stencil limits produce two plateaus | **CORRECT** | For a fixed smooth state, convergent rule families, and nonzero denominators. |
| Plateau necessarily halves under mesh doubling | **NEEDS-RESTATEMENT** | Asymptotically, with a nonzero leading term and comparable states. The reports use states reached by different mesh solves, so exact halving is not implied. |
| Neither rule family can ever go below the other target’s floor | **WRONG** | The plateau is a limiting discrepancy. At finite m, quadrature error can cancel stencil error and temporarily fall below it. |
| Off-mesh rollout is “therefore” O(h²) mesh-invariant | **WRONG** | O(h²) perturbations of residual ingredients imply O(h²) solution perturbations only with stability, consistent initialization/normalization, and the same solution branch. The failed head setting demonstrates this distinction. Nₘ as defined also explicitly contains L^(d/2); mesh independence requires normalized coordinates/operators. |
| Beating the FOM is possible **only because** the bank saw finer training data | **WRONG** | Finer training data are one possible explanation for a small representation error, not a mathematical necessity. The proposed coarse-trained-bank experiment exists precisely because causation remains unresolved. |

The defensible replacement for the central rate claim is:

> Generic boundary behavior suggests algebraic Fourier decay and potentially near-third-order lattice upper bounds for suitable rules. The observed ladders do not yet establish an asymptotic rate. Tent reflection improves boundary regularity here, but its measured finite-resolution penalty remains unexplained.

**3. Kill criteria K0–K6**

The criteria are not yet a single consistent decision procedure. [Appendix 04 §11](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-06-jcp-offmesh-paper-plan/04-experiment-design.md:453) differs materially from the main table.

| Criterion | Verdict | Decidability and connection to claim |
|---|---|---|
| **K0** | **NEEDS-RESTATEMENT** | Main plan triggers on reference failure; 04 permits order failures on up to 10% of cases, while E0 says each case must pass. Specify what happens to any failed case without silently excluding hard cases. The fallback to two fourth-order resolutions cannot provide the stated three-level second-order verification. |
| **K1** | **NEEDS-RESTATEMENT** | Main includes analytic-gradient trapezoid; 04 only includes the second-order tensor and requires ≥2 meshes. Define whether “matches” means paired field distance, paired error difference, or difference of cohort maxima. Matching a better comparator undermines superiority over that comparator; it does not erase the measured difference from the first-order incumbent. |
| **K2** | **NEEDS-RESTATEMENT** | Main omits 04’s “every linear-rung setting.” Fix the largest required mesh rather than allowing “largest affordable” to move after results. A speedup between 1× and 2× is still faster: this is a threshold for a **substantial-speedup claim**, not a truth test for “faster.” Memory and mesh independence need independent evidence. |
| **K3** | **NEEDS-RESTATEMENT** | Main has no mesh quantifier; 04 requires every mesh. Define matched accuracy and timing uncertainty. POD-ECSW matching performance does not prove that the decoder offers nothing: transfer, storage, and offline cost remain separate questions. |
| **K4** | **WRONG** | Main uses **OR**, 04 uses **AND**. E7 requires no effectivity below 0.5, but K4 tolerates such failures on 5% of states. Most seriously, failure of the estimator cannot be repaired by simply requiring the same unvalidated doubling estimator online. |
| **K5** | **NEEDS-RESTATEMENT** | “Unexplained” is not a performance criterion. E3c’s ≥80% explanation rule is not sufficiently defined, and explaining a failure does not repair it. Scope should depend on independent accuracy/invariance confirmation, with explanation reported separately. |
| **K6** | **NEEDS-RESTATEMENT** | The main plan’s non-polynomial reaction and 04’s Allen–Cahn/HJ are different experiments. “Reproduce Burgers behavior” needs PDE-specific predictions: Allen–Cahn is intentionally expected **not** to show the Burgers stencil gap. Scope claims separately by nonlinearity, convergence, accuracy, and cost. |

The assertion that only simultaneous K1/K2/K3 can derail the JCP article is editorial speculation. Reference or certification failures can independently invalidate central claims.

**4. Coverage of fatal objections from [05-red-team.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-06-jcp-offmesh-paper-plan/05-red-team.md)**

**Every fatal objection is named in the main plan. Several are not covered by a concrete matching experiment in 04.**

| Red-team objection | Verdict | Coverage |
|---|---|---|
| **1: reference/FOM bias** | **NEEDS-RESTATEMENT** | E0 is substantial coverage. The main plan adds a second-order timed FOM, but 04’s E4 lists solver families without explicitly fixing that spatial/time discretization. Reference fallback and uncertainty problems remain. |
| **2: finer-training-data confound** | **WRONG — missing detailed experiment** | Main E2c promises a coarse-only trained bank. It is absent from 04’s E2 arms, training design, and detailed budget. |
| **3: analytic gradient versus point placement** | **WRONG — missing detailed experiment** | Main E2b promises mesh-node quadrature with analytic gradients. That arm is absent from 04’s E2 list. A central-difference tensor is not a substitute. |
| **4: standard hyper-reduced baselines** | **NEEDS-RESTATEMENT** | E5 covers ECSW/DEIM/GNAT broadly. The promised **POD + exact tensor** is absent from its detailed baseline table; “tensor incumbent” uses the neural bank. Baseline-native solvers also conflict with the universal LM requirement. |
| **5: nonlinear-manifold framing** | **CORRECT** | The recommended title avoids the claim, and documenting the head failure is sufficient. Repair is optional if scope remains linear-span. |
| **6: quadratic-only evidence** | **WRONG — main/appendix mismatch** | Main requires a full-scale genuinely non-polynomial case. 04 instead trains Allen–Cahn and HJ, with steady Bratu only an appendix check. Allen–Cahn is cubic and the continuum HJ nonlinearity is quadratic; neither supplies the promised experiment. |
| **7: efficient-FOM speedup**, conditional fatal | **NEEDS-RESTATEMENT** | E4 addresses mesh/tolerance/time-step matching, but interpolation does not create an executable FOM configuration. Freeze an actual selection policy and substantiate matched points with runs. |
| **10: a-priori control versus empirical selection** | **NEEDS-RESTATEMENT** | E7 is relevant, but current theory and three-level checks do not yield a deterministic certificate. Either prove the required assumptions or make the claim explicitly empirical/probabilistic. |

Additional major objections still lack clear experiments: independent bank seeds, a non-box geometry, and longer-horizon stability. Those can be scoped out, but should not be described as covered.

**5. Controls, leakage, and other design failures**

| Design item | Verdict | Failure and needed correction |
|---|---|---|
| **5% injected sleep versus 10% drift gate** | **WRONG** | A one-sided 5% slowdown gives 1.05, which passes the 1.10 threshold. If applied equally to both repetitions, it produces no drift. Inject a fault exceeding the gate into a specified timed segment, and verify timer sensitivity separately. |
| **Bad quadrature must break mesh invariance** | **WRONG** | A fixed inaccurate rule can produce the same wrong solution at every mesh. For an all-continuum implementation, this passes invariance by construction. Test quadrature failure against an independently converged target, not mesh spread. |
| **All-continuum H3 “confirms” the source of head failure** | **WRONG** | Removing every mesh dependence necessarily removes mesh variation in a deterministic computation. It does not isolate linear terms from initialization, normalization, or branch selection. Change one ingredient at a time. |
| **Three-level doubling protects against every unresolved alias** | **WRONG** | Several levels can share an unresolved dual-lattice component. A common hidden error can coexist with a decaying component that makes the observed difference ratios look asymptotic. Agreement is not proof of saturation. |
| **Bad vector must be detected by doubling** | **WRONG** | For z=(1,1,…), increasing m still samples the same shifted diagonal. Both resolutions can agree accurately on the wrong lower-dimensional integral. Include an independent rule family/randomization and define the certificate’s failure policy. |
| **Eight-shift standard error certifies a single shifted rule** | **WRONG** | Standard error estimates uncertainty of the **shift average**, not a single production shift. Specify the deployed estimator, confidence multiplier, vector norm, and repeated/sequential-testing policy. |
| **Effectivity in [1,10] for 95% means “certificate”** | **NEEDS-RESTATEMENT** | This is empirical coverage on a state population. Raw doubling differences can underestimate even under benign contraction; use a justified safety factor and distinguish probabilistic coverage from deterministic bounds. |
| **Gauss exactly integrates a sine bank at predicted p\*** | **WRONG** | Gauss has polynomial exactness, not general trigonometric exactness. Hari’s own sine-bank table gives 2.3×10⁻⁸ at 48², despite calling that “exact.” Use a polynomial test with known Gauss exactness, or an analytic integral with a justified tolerance bound. |
| **MC4096, POD-r4, loose ECSW must always fail** | **NEEDS-RESTATEMENT** | These methods may legitimately pass on an easy state/family. Higher-rank POD need not improve a nonlinear rollout monotonically either. Distinguish empirical stress arms from deliberately injected faults with guaranteed observable consequences. |
| **Error budget: solver/test-space error is “the remainder”** | **WRONG** | Defining the last component by subtraction makes budget closure tautological. Norms do not generally add in quadrature or equal a triangle bound within 10%. Use independently measured vector differences/ablations; report interactions and bounds. |
| **Tensor and sign-upwind parity at 10⁻¹²** | **WRONG if treated as the same operator** | The incumbent tensor uses a fixed backward difference; sign-upwind is state-dependent and is not globally one quadratic tensor. 3D §3 reports tensor-versus-sign-upwind discrepancies around 10⁻⁵–10⁻⁴. Register separate exact-operator parity and sign-switch discrepancy tests. |
| **“Opened once per lane” for one shared sealed cohort** | **WRONG as a sealing guarantee** | A later lane can adapt after an earlier lane reveals the same cases. Per-lane hashes do not prevent this. Freeze all dependent choices globally before any headline results are exposed, or use independent final cohorts. |
| **E7 raises rule size and updates cost tables after p95 failure** | **NEEDS-RESTATEMENT** | Explicitly restrict this adaptation to validation. On sealed data, report failure without replacement; any revised method needs a fresh final cohort. |
| **FOM Pareto selection at the ROM’s sealed-cohort error** | **NEEDS-RESTATEMENT** | A predeclared envelope over frozen configurations can be a descriptive test result. Choosing a deployable comparator/configuration from it is test-based selection and conflicts with “every FOM-setting choice on validation.” Separate these uses. |
| **Read-only shared reference store** | **NEEDS-RESTATEMENT** | Write protection preserves integrity, not secrecy. Separate reference generation/QC from access to final-case solutions and outcome summaries. |
| **Uniform timing across one GPU model** | **NEEDS-RESTATEMENT** | Same model does not mean same device/allocation. Cost comparisons still need paired runs within a job, as the repository rules require. |
| **Interpolated “continuous” FOM Pareto front** | **NEEDS-RESTATEMENT** | Log-log interpolation is a plotting estimate, not measured solver performance. It can create an imaginary matched-accuracy setting and manufacture a threshold crossing. |
| **Same solver family for all E5 baselines** | **NEEDS-RESTATEMENT** | POD-Galerkin, LSPG, and GNAT have distinct formulations. Forcing every baseline through the same LM algorithm can handicap them. Include faithful baseline-native methods plus controlled common-solver ablations. |

Three reference-design problems also need resolution:

- **WRONG:** replacing three second-order levels with two fourth-order levels while retaining the [1.8,2.2] order gate.
- **NEEDS-RESTATEMENT:** Richardson differences are conditional uncertainty estimates, not automatically rigorous bounds. If each arm’s error relative to truth can move by ε_ref, their error difference can move by up to **2ε_ref**.
- **WRONG as a universal statement:** the scoring lattice is not shared by every proposed FOM mesh. For example, a 257² scoring lattice is not a subset of a 128² mesh; the 3D 48³ comparator also breaks the claimed shared-node construction. Specify and audit interpolation or use a compatible scoring design.

**6. GPU-hour consistency**

| Quantity | Verdict | Arithmetic |
|---|---|---|
| Main experiment table total | **CORRECT** | 730 GPU-h, including 50 GPU-h for stretch NS. |
| Main lane table total | **CORRECT** | Also 730 GPU-h. |
| Appendix 04 experiment/lane totals | **CORRECT** | Both sum to 715 GPU-h. E3’s 95 includes the separately listed 15 for E3c; no extra double count is needed. |
| Main versus 04 | **NEEDS-RESTATEMENT** | Difference is exactly **15 GPU-h**: E2 is 50 in the main plan, 35 in 04. Consequently E1+E2 is 65 versus 50. |
| Scope behind that difference | **NEEDS-RESTATEMENT** | The main adds coarse-bank training and analytic-gradient mesh quadrature, but 04 does not specify or cost them. E6a’s non-polynomial substitution also lacks an updated detailed budget. |
| Totals excluding NS stretch | **CORRECT after explicit separation** | Main **680**, appendix **665** GPU-h. |
| Approximately 900 with contingency | **CORRECT as rounding** | 715×1.25=893.75; 730×1.25=912.5 GPU-h. |
| Main **5–7 days at four jobs** | **WRONG** | 730/(4×24)=**7.60 days minimum**, before dependencies, queueing, or idle capacity. |
| Appendix **4–6 days at four jobs** | **WRONG** | 715/(4×24)=**7.45 days minimum**. |
| Contingency schedule | **NEEDS-RESTATEMENT** | 900 GPU-h requires **9.38 days minimum** at four continuously occupied GPUs. |

**The ten most important corrections, ranked**

1. **Make sealing global across dependent lanes.** Freeze every rule, baseline, FOM policy, and estimator before exposing the shared final cohort; prohibit E7 retuning on it.
2. **Withdraw the unconditional certificate claim.** Doubling and three-level agreement cannot establish saturation; estimator failure cannot be repaired by mandating that estimator.
3. **Repair the reference protocol.** Resolve failed-case handling, fourth-order fallback, scoring-grid compatibility, and uncertainty propagation before accuracy ranking.
4. **Synchronize the fatal-objection experiments into 04.** Add the coarse-only bank, mesh-node analytic-gradient arm, POD exact tensor, and full-scale non-polynomial case with concrete protocols and budgets.
5. **Replace “lattices converge like m⁻³” with a conditional upper-bound claim.** Keep observed finite-range slopes separate; do not assert a universal exact rate or ceiling.
6. **Rebuild the Gauss strip argument.** Correct the layerwise pole-avoidance proof, ellipse domain, normalization, point-count exponent, and entire-function growth assumptions.
7. **Replace non-discriminating controls.** Fix the 5%/10% timing contradiction; test inaccurate rules against truth rather than invariance; replace sine “exactness” with a valid exactness test.
8. **Rewrite K0–K6 as one unambiguous table.** Specify populations, mesh quantifiers, metrics, uncertainty, OR/AND logic, and separate explanation from successful confirmation.
9. **Make performance and error-budget evidence independent.** Run actual matched FOM configurations, preserve baseline-native methods, and measure error components instead of defining a remainder.
10. **Correct provenance and scheduling.** Label the lattice ratios as rounded, validation-derived, 64³/R′=512 observations; distinguish nodes from cells; reconcile 715/730 GPU-h and budget at least 7.5–7.6 fully occupied four-GPU days before contingency.