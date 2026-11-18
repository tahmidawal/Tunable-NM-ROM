# Building a convincing Tunable NM-ROM paper

This November 18, 2026 literature assessment recommends a JCP paper that explains **when the nonlinear head, ordered bank and quadrature each help, what limits their accuracy, and what a complete solution costs**. The figure designs and theoretical statements below are proposals, not new experimental results; existing project findings are cited qualitatively to their current records.

The strongest next step is to connect your existing evidence into a numerical-methods argument, then fill specific gaps. More PDE examples alone will not establish the contribution. The literature already contains nonlinear-manifold projection, continuous neural fields, nonlinear compression inside a larger linear basis, and hyper-reduction. Your paper should establish what the **particular combination of a reusable ordered bank, a nonlinear head, fixed weak tests and efficient integration** makes possible. This is a recommendation based on closely related papers, not an official JCP acceptance checklist.

## Published papers connect approximation, dynamics and cost

The papers below were selected for methodological proximity. Publication metadata was verified against primary records; detailed analyses were inspected in journal HTML or author manuscripts. For most JCP papers, the accessible full text was an author version, so its figure numbering can differ from the final publication. The source notes retain those access distinctions and section references. No published experiment was independently reproduced for this review.

| Published paper | Venue | Analysis worth carrying into your paper |
|---|---|---|
| Lee and Carlberg, *Model reduction of dynamical systems on nonlinear manifolds using deep convolutional autoencoders* (2020) | [JCP](https://doi.org/10.1016/j.jcp.2019.108973) | Derives manifold Galerkin/LSPG properties and discrete error bounds; separates best-approximation diagnostics from evolved ROM error. This is the central precedent for explaining whether a failure comes from representation or dynamics. |
| Kim, Choi, Widemann and Zohdi, *A fast and accurate physics-informed neural network reduced order model with shallow masked autoencoder* (2022) | [JCP](https://doi.org/10.1016/j.jcp.2021.110841) | Studies decoder/Jacobian sampling cost, hyper-reduced error, reduced dimension, training data and extrapolation. Includes cases where a nonlinear ROM without effective hyper-reduction provides no speedup. |
| Barnett and Farhat, *Quadratic approximation manifold for mitigating the Kolmogorov barrier in nonlinear projection-based model order reduction* (2022) | [JCP](https://doi.org/10.1016/j.jcp.2022.111348) | Compares affine and quadratic manifolds using flow fields, force histories and offline/online resources. A useful precedent for testing whether a neural nonlinearity is needed. |
| Chen, Chiaramonte, Grinspun and Carlberg, *Model reduction for the material point method via an implicit neural representation of the deformation map* (2023) | [JCP](https://doi.org/10.1016/j.jcp.2023.111908) | Separately varies gradient regularization, latent dimension and sampling; tests resolution transfer. Its hardware allocation differs between full and reduced methods, so its speedup is not a portable benchmark target. |
| Romor, Stabile and Rozza, *Explicable hyper-reduced order models on nonlinearly approximated solution manifolds of compressible and incompressible Navier–Stokes equations* (2025) | [JCP](https://doi.org/10.1016/j.jcp.2025.113729) | Compares reconstruction, evolved fields, sampling choices and parameter generalization; counts adaptive overhead and reports regimes with unfavorable speed. |
| Puri, Prakash, Kara and Zhang, *SNF-ROM: Projection-based nonlinear reduced order modeling with smooth neural fields* (2025) | [JCP](https://doi.org/10.1016/j.jcp.2025.113957) | Closely related continuous-field projection method. Studies time step, sampling, memory, derivatives and generalization; discusses cancellation between temporal and sampling errors. |
| Fresca, Dedè and Manzoni, *A Comprehensive Deep Learning-Based Approach to Reduced Order Modeling of Nonlinear Time-Dependent Parametrized PDEs* (2021) | [Journal of Scientific Computing](https://doi.org/10.1007/s10915-021-01462-7) | Varies reduced dimension, data quantity, architecture and training loss. Its parameter/time predictor differs from your online physics solve; its ablation design remains useful. |
| Fresca and Manzoni, *POD-DL-ROM: Enhancing deep learning-based reduced order models for nonlinear parametrized PDEs by proper orthogonal decomposition* (2022) | [CMAME](https://doi.org/10.1016/j.cma.2021.114181) | Uses nonlinear compression within a linear outer basis; compares pretraining costs and physical outputs such as flow rate. Establishes relevant prior art for the head-plus-bank structure. |
| Romor, Stabile and Rozza, *Non-linear Manifold Reduced-Order Models with Convolutional Autoencoders and Reduced Over-Collocation Method* (2023) | [Journal of Scientific Computing](https://doi.org/10.1007/s10915-023-02128-2) | Reports reconstruction versus POD, sampling ladders, mean and maximum time error, compressed-decoder costs and solver failures. |
| Geelen, Wright and Willcox, *Operator inference for non-intrusive model reduction with quadratic manifolds* (2023) | [CMAME](https://doi.org/10.1016/j.cma.2022.115717) | Separates quadratic reconstruction from inferred reduced dynamics; compares equal-coordinate-dimension dynamics and time histories. |
| Diaz, Choi and Heinkenschloss, *A fast and accurate domain decomposition nonlinear manifold reduced order model* (2024) | [CMAME](https://doi.org/10.1016/j.cma.2024.116943) | Analyzes the nonlinear solver and conditional error bounds; compares linear/nonlinear and full/hyper-reduced variants. Its inverse-residual assumption is particularly relevant to your weak-test analysis. |

**CROM and CoLoRA are also close comparators, but they are conference papers:** CROM appeared at ICLR and CoLoRA at ICML. They should be discussed alongside the journal literature without being relabelled as journal publications. For CoLoRA, distinguish equation-driven integration from its data-driven variant. You do not need to implement every paper in this table; one well-matched neural-field comparator, strong classical/POD controls, and claim-specific ablations are more useful than many weak comparisons. A quadratic-manifold comparison becomes especially valuable if you claim an advantage specific to neural nonlinear compression. ([CROM](https://crom-pde.github.io/), [CoLoRA proceedings](https://proceedings.mlr.press/v235/berman24b.html))

## Your architecture makes the scientific questions precise

Write the two deployed choices explicitly:

$$
u(z)=b+G h(z),\qquad z\in\mathbb R^k,
\qquad
u(c)=b+\widehat G_{R'}c,\qquad c\in\mathbb R^{R'}.
$$

Here $b$ supplies prescribed boundary data; $G$ evaluates the boundary-compatible spatial bank; $h$ maps latent coordinates to bank coefficients; and $\widehat G$ is the ordered bank. The current manuscript keeps the nonlinear head unchanged and changes the **linear-span prefix** at runtime. It does not establish a nested family of nonlinear heads or a freely adjustable latent dimension after training. ([Current implementation description](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-11-13-manuscript-rebuttal/paper_revision/main.tex:341), [ordering argument](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-11-13-manuscript-rebuttal/paper_revision/main.tex:1363))

| Symbol | Meaning in the proposed paper | What changing it tests |
|---|---|---|
| $N$ | Total requested field degrees of freedom, not points per coordinate | Output resolution and input/output work |
| $R$ | Full frozen bank width | Representation capacity and stored/evaluated bank cost |
| $R'$ | Active ordered span width | Retraining-free span tunability |
| $k$ | Nonlinear head's latent dimension | Compression; generally a separately trained architectural choice |
| $M$ | Number of smooth weak tests | How much of the equation residual the solve observes |
| $m$ | Quadrature sites | Integration accuracy and nonlinear evaluation cost |
| $m^+$ | Distinct evaluated sites including stencil neighbors | Actual sampled spatial work for a stencil-based operator |
| $\Delta t$, $\tau_s$ | Time step and the implemented solver stopping tolerance | Temporal and solver sensitivity, studied independently |

**A mathematical fact, not an empirical hypothesis:** at the same full bank width,

$$
\{b+Gh(z):z\in\mathbb R^k\}
\subseteq b+\operatorname{range}(G).
$$

Thus the head cannot have lower globally optimal reconstruction error than the unrestricted full bank in the same norm. Its benefit must concern fewer unknowns, dynamics, conditioning, generalization or cost. A head can still outperform a much smaller linear space, and can evolve more accurately than a solver operating in the full bank. Those are different comparisons. Similarly, nested span prefixes guarantee monotone **best approximation** as width grows, not monotone rollout accuracy or runtime. Do not describe a best-found nonlinear fit as a certified manifold-distance minimum.

This distinction matters for your current evidence: the harder Navier–Stokes manuscript comparison uses mesh-specific POD banks with nonlinear heads. It supports a head/dynamics mechanism, separately from the claim that a learned continuous bank is superior. The cylinder example uses a different Galerkin rollout; label that variation rather than implying that it validates the complete fixed-test least-squares algorithm on that geometry. ([Current results](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-11-13-manuscript-rebuttal/paper_revision/main.tex:640))

## A concrete figure plan for your paper

The following are proposed main-paper figures, with supplementary panels where appropriate. Their captions describe questions and controls; they deliberately contain no invented trends. Use disjoint whole trajectories for training, development and final testing. Choose settings on development cases, then freeze them before final evaluation. Report case medians, upper tails or worst errors, and failures alongside average behavior.

### Figure 1 — Separate representation quality from reduced dynamics

**Layout and axes.** Place reconstruction error and evolved solution error in adjacent panels. The horizontal axis is online dimension: $k$ for the head and $R'$ for the span. Label the underlying full width $R$ separately. The vertical axis is relative physical-field error in a common mass-weighted norm; include a derivative or flux-error panel where transport/diffusion makes it relevant. Add time histories for a representative difficult trajectory.

**Controls.** On the same snapshots and frozen bank, compare full-bank least-squares projection, ordered-prefix projection, carefully optimized head reconstruction, dense reduced trajectories and hyper-reduced trajectories. Include matched-data POD and, for the head mechanism, POD plus a head. Compute linear projection with a stable factorization; use multiple starts and convergence checks for nonlinear reconstruction, labelling it best-found.

**Interpretation.** If the head has a larger reconstruction error but smaller evolved error than its full span, the evidence supports constrained dynamics or regularization. If both reconstruction and rollout are poor, improve the bank before changing the solver. If dense evolution works and sparse evolution fails, the bottleneck is hyper-reduction. These are conditional interpretations, not forecast results. This follows the approximation/dynamics separation in Lee–Carlberg and Kim et al. ([Lee–Carlberg full text](https://arxiv.org/html/1812.08373v3), [Kim et al. full text](https://arxiv.org/html/2009.11990))

*Proposed caption:* “Approximation and evolution errors for identical banks and training data. Best-found head fits are distinguished from exact span projections; failed trajectories remain visible.”

### Figure 2 — Measure what tunability buys from one training run

**Layout and axes.** Plot prefix width $R'$ against reconstruction error and rollout error, then plot complete-query time against error. Use a common symbol for settings from one frozen bank. A companion table lists the width, tests, quadrature, setup requirement and final-test outcome selected for each requested tolerance.

**Controls.** Compare ordered prefixes with unordered prefixes, POD prefixes and separately trained models at selected widths. Keep the data and scoring norm matched. Hold the other numerical settings fixed for isolation, and separately show the qualified settings where more tests or points are necessary. Count per-prefix operator construction or quadrature fitting even though neural weights stay frozen.

**Interpretation.** A useful menu of accuracy/cost choices would support the tunability claim. A gap to separately trained models quantifies the price of reuse. An irregular rollout curve despite monotone projection error signals dynamics or numerical errors, not a failure of the nesting theorem. A $k$ sweep belongs to a separate training study unless an explicitly implemented architecture allows $k$ to change without retraining.

*Proposed caption:* “Accuracy and cost of ordered span prefixes from one frozen model, compared with independently constructed models. Setup required for each setting is reported separately.”

### Figure 3 — Show that the weak tests see the important errors

**Layout and axes.** Use a heatmap with test count $M$ on one axis and actual online dimension on the other. Plot physical error, solver failures and a scaled Jacobian condition diagnostic. Add a scatter plot of physical error against the minimized tested residual, with an independently expanded weak-test residual as a check.

**Controls.** Freeze the bank, quadrature, time step and inputs. For the head, analyze

$$
J_r(z)=SPF'(u(z))GJ_h(z),
$$

where $F$ is the chosen discrete PDE residual, $P$ contains the tests and $S$ is the actual fixed row scaling. Compare $M$ with $k$ for heads and with $R'$ for spans. Keep tests comfortably above the unknown count for production; the project's square-test failures are diagnostics, not a recommended operating point. Record iterations, rejected steps, initialization sensitivity and failures.

**Interpretation.** A small minimized residual alongside large physical error exposes directions the tests do not detect. To separate physical sensitivity from arbitrary latent scaling, orthonormalize the decoder tangent in the mass norm before inspecting the tested operator on it. Also inspect the bank's mass-weighted Gram matrix. A positive smallest singular value at sampled states is local diagnostic evidence, not a proof of uniform stability. The linear skip can receive a matched ablation if its conditioning benefit is a central claim. Keep the weak formulation; this analysis does not require strong-form collocation.

*Proposed caption:* “Accuracy and solver behavior as the weak-test space grows. Physical tangent scaling separates residual observability from the chosen latent coordinates.”

### Figure 4 — Identify the source of each accuracy plateau

**Layout and axes.** Use separate panels for error against $\Delta t$, $M$, $m$ and nonlinear tolerance, with reference uncertainty and reconstruction error marked. For quadrature, add dense-versus-sparse residual error and Jacobian-action error on states reached by the deployed solver. Plot those errors and query cost against $m$.

**Controls.** Tighten one ingredient at a time and check interactions where they matter. Compare sparse and sufficiently integrated versions of the **same operator**. Where applicable include exact tensor contraction, classical rules and the existing nonnegative fitted rule. Requalify the rule when the relevant bank, tests or integrand changes. The local rule of thumb $m\approx4M$ is an empirical starting point, not a universal convergence theorem.

Changing an upwind stencil to analytical derivatives changes the target operator. Keep dense discrete, sampled discrete, dense continuous-weak and sparse continuous-weak comparisons explicitly distinct. A quadrature lattice is not a coarse physics mesh. Add field-gradient or flux error against a qualified reference; low field error alone can conceal derivative error. SNF-ROM's discussion of temporal/sampling cancellation is a direct reason to avoid attributing an improvement to one approximation without these controls. ([SNF-ROM full text](https://arxiv.org/html/2405.14890v2))

**Interpretation.** Saturation with $m$ only identifies a quadrature plateau once the other errors are controlled. If changing derivatives, rather than nodes, explains an improvement, the paper should report that mechanism. Do not make an additive percentage “error budget” from unrelated trajectories: error contributions can cancel.

*Proposed caption:* “Controlled refinement of time stepping, test space, quadrature and solve tolerance for a fixed representation. Discrete and continuous operator targets are identified separately.”

### Figure 5 — Test mesh transfer and account for full-field output

**Layout and axes.** Plot total output degrees of freedom $N$ against error, reduced evolution time, input preparation time, field reconstruction time and memory. Use separate panels rather than hiding these quantities in one speedup. Add a diagram or table showing which objects remain frozen and which are rebuilt.

**Controls.** Transfer the same learned weights across meshes, disclose normalization, ordering, test, operator and quadrature changes, and compare with continuous/interpolated POD. Independently refine the truth calculation. For stencil evaluation count $m^+$, including neighbors. A head's dense field reconstruction generally costs $O(NR)$ and a span's $O(NR')$; neither a dense input nor a dense output is free.

**Interpretation.** Flat reduced-evolution cost with growing output cost supports a mesh-independent reduced calculation, not constant-cost complete field delivery. If POD transfers equally well, characterize continuous evaluation as a shared capability and identify the additional benefit your bank provides. Chen et al. and SNF-ROM establish that resolution transfer itself is not new. ([Chen et al. full text](https://arxiv.org/html/2109.12390), [SNF-ROM](https://doi.org/10.1016/j.jcp.2025.113957))

*Proposed caption:* “Transfer of one frozen representation across output meshes, with input, reduced evolution, output and setup costs separated.”

### Figure 6 — Compare the cost of answers at the same accuracy

**Layout and axes.** Plot complete-query time against median error and worst-case error in separate panels. Mark failed configurations. Add stored-operator bytes and peak device memory where storage is a contribution; these are different metrics. Show cold and repeated-query contracts separately.

**Controls.** Use a tuned eligible full-order solver, POD with appropriate hyper-reduction, your span and your head, and a close intrusive neural-field comparator. Classical controls should include relevant coarse-grid, second-order and spectral choices. Give each method an accuracy-based selection procedure, matched input/output requirements and a transparent development budget. Neural operators predicting from parameters or initial fields answer a different online task in some settings; distinguish that comparison.

Measure cost and accuracy from the same returned solution on the same hardware allocation. Include initial projection, nonlinear iterations, field output and required failure recovery. Persist repeated timings and use medians and outlier counts. Separate compilation, data generation, training and setup from online work. For repeated queries, incremental break-even is

$$
Q_{\mathrm{break}}=
\frac{C_{\mathrm{off,ROM}}-C_{\mathrm{off,base}}}
{C_{\mathrm{on,base}}-C_{\mathrm{on,ROM}}}.
$$

This applies when the denominator is positive and both methods use the same cost contract. With a positive denominator, a nonpositive incremental offline cost means there is no extra offline investment to recover; a positive crossover is rounded up to an integer query count. If the ROM has no per-query saving, additional queries do not recover an extra offline investment.

**Interpretation.** A speed advantage restricted to one tolerance or regime can support a precise claim. If the strongest baseline wins, retain that result and center any demonstrated storage, reuse or dynamics advantage. A universal speedup claim is unnecessary and currently unsupported by the project's closeout.

*Proposed caption:* “Accuracy–cost frontiers for identical requested outputs and qualified reference errors. The timed invocation supplies the scored field; offline costs are reported separately.”

### Figure 7 — Establish a useful operating region and physical behavior

**Layout and axes.** Plot error and relevant physical quantities against time; add parameter-versus-error/failure panels separating interpolation from extrapolation. For heat/Burgers, examine energy or dissipation under the applicable boundary conditions. For incompressible flow, include divergence and kinetic energy; for cylinder flow, include lift, drag and phase.

**Controls.** Use untouched trajectories, longer horizons with independently qualified references and quadrature, initial-guess perturbations, and several training seeds when conclusions depend on learned weights. A supplementary training-data-size study can reveal whether the claimed mechanism survives data scarcity. Check exact boundary enforcement under the mask/lift; divergence-free tests do not by themselves make the trial field divergence-free.

**Interpretation.** Stable rollout inside a parameter range establishes an operating region, not unconditional stability. Long-time drift or extrapolation failures define the method's limits. The paper need not add adaptive trees or a new geometry architecture to answer this question.

*Proposed caption:* “Physical behavior and failure boundaries on held-out trajectories, distinguishing interpolation, extrapolation and extended integration.”

## The theory should explain the actual weak-residual solver

Begin with accessible propositions: boundary enforcement under stated mask/lift assumptions; nested-span best approximation; ordering optimality for the specified weighted training objective; exact preassembly of linear terms; and operation/storage counts including output. These are useful, method-specific statements. A proof for standard manifold LSPG cannot simply be reused for a residual tested against a finite fixed space.

**A candidate conditional result to formalize** is the following. Let $u_*$ solve the selected discrete PDE, let $r(u)=SPF(u)$, and choose an admissible trial-state set $\mathcal A$ in one specified solution basin. Assume $r$ is $L$-Lipschitz on a domain containing $u_*$ and $\mathcal A$; on $\mathcal A$ assume

$$
\|r(v)-r(w)\|\ge\beta\|v-w\|,\qquad\beta>0.
$$

Assume also a uniform quadrature defect $\sup_{v\in\mathcal A}\|\widetilde r(v)-r(v)\|\le\varepsilon_q$ and a returned state $\widetilde u\in\mathcal A$ satisfying

$$
\|\widetilde r(\widetilde u)\|
\le\inf_{v\in\mathcal A}\|\widetilde r(v)\|+\varepsilon_s.
$$

A triangle-inequality argument then gives

$$
\|u_*-\widetilde u\|
\le\left(1+\frac{2L}{\beta}\right)
\inf_{v\in\mathcal A}\|u_*-v\|
+\frac{2\varepsilon_q+\varepsilon_s}{\beta}.
$$

This is a proposed analysis framework, **not a proved stability property of the implementation**. The lower bound is a substantial assumption because weak tests can miss state differences. Residual near-minimization is stronger than Levenberg–Marquardt stationarity: the implemented stopping tolerance $\tau_s$ does not certify the theorem's residual-norm suboptimality $\varepsilon_s$. For time evolution, previous-step errors require an additional propagation argument; the displayed estimate is not a complete trajectory bound. Start with the linear elliptic span, where restricted rank and least squares are tractable, then state the nonlinear extension conditionally. Diaz et al. provide a relevant precedent for making inverse-residual assumptions explicit. ([Author manuscript](https://arxiv.org/pdf/2305.15163))

The empirical counterpart is Figure 3's tangent sensitivity and Figure 4's quadrature/solver diagnostics. Sampled singular values cannot certify a uniform $\beta$, but they can reveal why an error guarantee would become weak. That connection between assumptions and observed behavior is more persuasive than an unrelated general neural-network approximation theorem.

## Reuse the completed evidence, then fill the decisive gaps

| Analysis area | What the project already establishes | What still needs attention for this paper |
|---|---|---|
| Head versus span | Manuscript and budget studies contain same-bank contrasts; harder Navier–Stokes supports the head mechanism. | Assemble projection and dynamics for identical banks, distinguish POD-bank from neural-bank evidence, and avoid universal head superiority. |
| Tunability | The ordering appendix establishes the training-data span argument; dimension/cost sweeps exist. | Present a frozen-model tolerance-selection story and identify per-setting setup. Add only missing ordering/retraining controls. |
| Tests and solver | Square-test configurations failed qualification in the earlier campaign. | Explain visibility/conditioning and report qualified oversampling behavior; a failed control cannot establish a transition curve. |
| Quadrature and consistency | Extensive point-count, derivative and refinement studies exist. The final primary criterion passes, while the two-dimensional secondary-reference criterion fails. | Preserve both reference definitions and decisions; consolidate dense/sparse residual and Jacobian checks if absent. |
| Baselines and output | Stronger classical and continuous-POD controls exist. The final spectral comparator wins on timing-qualified prepared-input panels. | Complete missing input/output accounting only if an end-to-end claim needs it; do not manufacture ratios across allocations. |
| Robustness | Longer-time and harder-regime studies exist with reference or quadrature restrictions. | Reuse qualified portions, clearly label provisional parts, and fill only the missing controls needed by retained claims. |

These assessments come from the [baseline campaign](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/final/report.md), [improvement campaign](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-21-improvement-campaign/final/report.md), [current manuscript](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-11-13-manuscript-rebuttal/paper_revision/main.tex) and [final closeout](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-15-experiment-closeout.md). Quantitative paper tables must be regenerated from their run records rather than copied into this planning report.

**Work order.** First freeze the paper's claims and assemble existing qualified results into the figure plan. Next close the most consequential explanation gap: approximation versus dynamics and weak-test visibility. Then fill any missing complete-query or frozen-model control essential to the central claim. Freeze those choices before new final testing. Put broad architecture sweeps, additional PDE replications and training-size details in supplements where they support rather than obscure the mechanism.

A compact example set can use a linear elliptic/parabolic problem for correctness, Burgers for transport/quadrature limits, and a harder flow for the nonlinear-head mechanism. Not every figure needs every PDE. Release configuration choices, checkpoints, case-level errors, repeated timings, seeds, code revisions and regeneration commands, and independently rebuild the main figures from those records.

The strongest manuscript sequence is **precise method and novelty → conditional analysis → representation and dynamics → tunability and numerical consistency → cost and physical limitations**. If the authors choose the narrower quadrature article instead, emphasize Figures 4–6 and the rule-selection/storage question. The earlier [submission-readiness assessment](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-15-jcp-submission-readiness.md) leaves the relationship to the conference manuscript as a coauthor decision; settle that scope before deciding which results belong in the journal article. Hierarchical tree embeddings remain a separate extension.

## Plain-language glossary

**Table columns.** “Published paper” identifies a prior article; “Venue” is where it appeared; “Analysis worth carrying into your paper” identifies a methodological lesson, not a claim that every study is mandatory. “Symbol,” “Meaning” and “What changing it tests” define the experiment controls. “Analysis area,” “What the project already establishes” and “What still needs attention” distinguish current evidence from proposed work.

**ROM, NM-ROM and FOM.** A reduced-order model solves for fewer unknowns than a full-order model. A nonlinear-manifold ROM restricts solutions to a curved family described by a nonlinear decoder. A PDE is a partial differential equation, such as heat, Burgers or Navier–Stokes.

**Publication and method names.** JCP is the *Journal of Computational Physics*; JSC is the *Journal of Scientific Computing*; CMAME is *Computer Methods in Applied Mechanics and Engineering*. ICLR is the International Conference on Learning Representations; ICML is the International Conference on Machine Learning. CROM names a continuous reduced-order modeling method using implicit neural representations. CoLoRA uses continuous low-rank adaptation of such representations.

**Bank, head, span, prefix and latent coordinate.** A bank is a collection of spatial functions; its width counts them. A head converts a small latent state into bank coefficients. The span allows arbitrary linear combinations. An ordered prefix keeps the first selected functions. “Frozen” means learned weights are unchanged. A mask vanishes on prescribed boundaries; a lift supplies nonzero boundary values.

**Projection, reconstruction and rollout.** Projection seeks the closest representable reference state. Reconstruction evaluates that representation. A rollout evolves it by solving reduced equations over time. A best-found nonlinear fit is an achieved error, not proof of the smallest achievable error. A representation floor is the minimum error possible within a chosen family.

**POD, PCA and quadratic manifold.** Proper orthogonal decomposition and principal component analysis identify useful linear directions from example fields. Continuous/interpolated POD makes those directions evaluable across spatial locations. A quadratic manifold adds products of reduced coordinates. Neural-field methods describe a spatial field as a function of coordinates. A neural operator predicts fields from input functions or parameters.

**Weak test, residual and hyper-reduction.** A residual measures failure to satisfy an equation; smooth tests measure selected weighted averages of that failure. Quadrature approximates integrals with weighted points. Hyper-reduction reduces equation-evaluation work. Empirical quadrature fits points/weights from data; NNLS means nonnegative least squares. A stencil uses neighboring spatial values. A tensor contraction evaluates a preassembled multilinear operator.

**Galerkin, Petrov–Galerkin and LSPG.** These project equations onto chosen test directions; Petrov–Galerkin allows tests to differ from trial directions. Least-squares Petrov–Galerkin minimizes a residual-based objective. Levenberg–Marquardt is a damped nonlinear least-squares algorithm. Stationarity means its local gradient is small, not that the solution is globally best or physically accurate.

**Jacobian, tangent, Gram matrix and conditioning.** A Jacobian describes local sensitivity. Decoder tangents are field changes reachable by small coordinate changes. A Gram matrix records inner products between basis functions. Small singular values identify weakly detected directions; conditioning measures sensitivity to perturbations. A mass-weighted norm accounts for physical spatial volume; an $H^1$ measure includes first derivatives. Flux is transported quantity per area and time.

**Consistency, reference uncertainty and stability.** Consistency concerns approximating the intended equation. Reference uncertainty is the estimated error in the comparison solution. Stability concerns growth of perturbations or time-evolution errors. An inverse-Lipschitz assumption says different trial states must produce sufficiently different tested residuals. Its constant $\beta$ measures that separation; $L$ bounds forward sensitivity. The theorem's $\varepsilon_q$ and $\varepsilon_s$ bound quadrature error and residual-norm optimization error on the stated admissible set.

**Held-out, extrapolation, seed and ablation.** Held-out cases were excluded from fitting/selection. Extrapolation tests inputs outside the training range. A seed fixes randomized choices. An ablation changes one component to identify its contribution. A Pareto frontier contains settings for which neither accuracy nor cost can improve without sacrificing the other. Offline costs precede queries; online costs occur per query. Break-even is the query count at which savings repay additional offline cost. Divergence measures local expansion of a velocity field; lift, drag, phase and energy are physical quantities of interest.
