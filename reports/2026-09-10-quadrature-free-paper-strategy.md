# A paper strategy for nonlinear manifold ROMs with precomputed operators

This document assesses how to rebuild the older submission around the separable decoder and empirical-quadrature-free reduced operators. It is a research and writing proposal, not a new numerical result or an authorization to start experiments; historical measurements retain the qualifications in their source reports.

The initial recommendation was one substantially rewritten paper. The user's subsequent preferred split assigns tunability to the CP/EQ paper and mesh-independent cached reduced computation to the precomputed-operator paper. The separate-paper plan below follows that emphasis; each paper still needs independent evidence, and no new experiment campaign has been authorized by this discussion. The common scientific question is whether a learned nonlinear map in a modest spatial bank gives a useful accuracy–cost tradeoff while preserving efficient projection of suitable discrete PDE operators.

## What was read

The [older manuscript PDF](../Older%20Paper%20/neurips26__Copy_%20%283%29.pdf) and the `main.tex` inside its [source archive](../Older%20Paper%20/neurips26__Copy_%20%281%29.zip) were compared with the user-supplied OpenReview comments, the canonical lab log, the [tensor/Poisson tables](2026-09-03-burgers-poisson-tensor-tables.md), their underlying method reports, and the [historical cost audit](2026-09-10-historical-poisson-and-burgers-cost-audit.md). The Poisson quadrature-free and Burgers tensor source implementations were inspected. This review did not independently rerun the numerical experiments or verify each rebuttal table against its original artifacts.

The current lab log also records an approved historical replay in the existing Burgers and Poisson worktrees. Its [protocol](../worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/HISTORICAL-COMPARISON-REPLAY.md) is the immediate reproduction dependency; this proposal does not duplicate or redirect that work. Earlier wave evidence remains excluded under the user's evidence reset.

## The mathematical contribution to establish

Write the frozen decoder as

$$
u(z)=u_g+G h_\theta(z),\qquad
G\in\mathbb R^{n\times R},\quad
h_\theta:\mathbb R^k\rightarrow\mathbb R^R.
$$

Here the spatial bank and network weights are fixed during a query; only the latent coordinates are solved. The nonlinear image lies inside the bank's linear span, and its local dimension is at most the latent dimension. Full local dimension requires an appropriate Jacobian rank condition; a linear skip alone does not prove it.

Let $P\in\mathbb R^{M\times n}$ represent the selected smooth weak tests, including the discrete integration weights and declared row scaling. For a fixed linear discrete operator $A$,

$$
r_w(z)=P\big(Au(z)-f\big)
       =B h_\theta(z)-b,
\qquad B=PAG,\quad b=P(f-Au_g).
$$

The matrix $B$ can be assembled offline. For a polynomial field operator, its projected coefficients can similarly be assembled into reduced tensors. For a quadratic term, after absorbing fixed-lift contributions into the linear and constant terms,

$$
[r_w(z)]_i=[B h_\theta(z)-b]_i+
\sum_{a,b}T_{iab}[h_\theta(z)]_a[h_\theta(z)]_b.
$$

The coefficient map may be a non-polynomial neural network; polynomial structure is required of the relevant discrete field operator for this tensor construction. JAX still differentiates the nonlinear head and reduced contractions with respect to the latent coordinates. Empirical quadrature disappears from this repeated operator evaluation. This does not eliminate offline spatial sums, source processing, initial-field fitting, dense reconstruction, or transfers required by the chosen query contract.

The quadratic tensor has storage and contraction work scaling as $O(MR^2)$. Thus bank size matters independently of latent dimension. Fixed reduced dimensions remove explicit mesh-size loops from cached reduced evaluations; they do not prove mesh-independent iteration counts or complete-query cost. Rebuilding the mesh-dependent bank and operators is setup, even when neural weights transfer unchanged. Parameter-varying operators also need a declared treatment: a finite affine decomposition permits preassembling its terms, whereas an arbitrary new coefficient field does not automatically inherit the same offline/online separation.

### The old decoder already contains the relevant factorization

The original manuscript's CP spatial expansion also has the form $u_g+G h(z)$ after incorporating its fixed output bias and boundary mask. Therefore precomputability is not exclusive to the newer coordinate-network bank. For the original tangent-projected linear equation, one can likewise write

$$
J_u(z)^\top(Au(z)-f)
=Dh(z)^\top\left[C h(z)-d\right],
\qquad C=G^\top A G,\quad d=G^\top(f-Au_g).
$$

This is an algebraic inference from the printed architecture, not a measured speedup of its implementation. It also does not make tangent projection equivalent to the newer weak least-squares objective. The newer spatial network supplies a coordinate-based bank and a different training approach; frozen-weight mesh transfer must be demonstrated separately. No new older-ViT/CP training campaign is proposed here.

## Repairs needed in the mathematical presentation

| Issue in the manuscript or discussion | Required treatment in the new paper |
| --- | --- |
| Tangent Galerkin, full residual least squares, and weak residual least squares are used interchangeably | Define the exact implemented residual, tests, row scaling, norm, and stationarity condition. For the inspected tensor/QF path, describe minimization of the overdetermined weak residual. |
| The printed heat update treats the nonlinear decoder like a homogeneous linear map | Start from the fully discrete field residual with the decoded previous state, then apply the selected weak projection and nonlinear solver. |
| The Poisson update is presented as Gauss–Newton without identifying its objective or neglected curvature | Explain whether it is an energy-based approximate Newton step or a residual least-squares step; these have different derivatives. |
| The rebuttal says residual least squares and Galerkin coincide for a linear decoder | State any actual equivalence assumptions. Linear trial spaces alone do not imply equivalence. |
| A small projected residual is presented as residual consistency sufficient for accuracy | Distinguish weak stationarity, untested residual components, representation error, and physical solution error. |
| Quadrature count and tighter tolerances are presented as a continuous guaranteed accuracy frontier | Identify discrete precomputed quadrature choices in the EQ control. For the tensor path, retain only implemented solver-budget knobs and measure their error–cost behavior; a smaller residual need not improve physical error. |

For example, with a linear trial basis $V$ and a field residual $AVz-b$, ordinary residual least squares imposes

$$
V^\top A^\top(AVz-b)=0,
$$

whereas Galerkin imposes $V^\top(AVz-b)=0$. Similarly, differentiating the original nonlinear tangent residual gives

$$
D_z\!\left[J_u^\top(Au-f)\right]
=J_u^\top A J_u+\sum_j(Au-f)_j\nabla_z^2u_j.
$$

Dropping the second term may define a useful approximation, but it must be identified. Calling every such solve Gauss–Newton obscures which equations were actually solved.

## What the retained evidence supports

Numerical values should be imported from the existing generators into the eventual manuscript, rather than recopied into a second table here.

| Evidence | Supported interpretation | Remaining limitation |
| --- | --- | --- |
| Poisson quadrature-free parity checks | Exact preassembly reproduces the full-grid weak residual and its derivatives within the recorded floating-point checks | Exact evaluation of selected equations does not certify the field; historical stopping limitations remain visible |
| Burgers tensor/full-grid comparisons | Precomputed polynomial advection closely tracks the tested full-upwind ROM trajectories | The sign-dependent stencil is not globally polynomial; decoded negative undershoots produce nonzero mismatch |
| Burgers paired FOM comparison | A historical device-query advantage exists for the largest tested small-bank configuration against the named same-grid solver | It is not a win at every mesh, bank, error target, or FOM implementation |
| Tensor versus tuned EQ | Eliminates empirical node fitting and node selection while retaining similar observed online accuracy and cost | The historical FOM advantage was already largely present with tuned EQ |
| Poisson versus CG | A crossover exists against the recorded iterative comparator | The same archive includes a faster direct spectral comparator; untested nonseparable problems cannot inherit the CG speedup claim |
| Newer multiresolution campaign | Evaluates different banks, query accounting, and FOM choices | Its Burgers query does not retest the archived small-bank tensor configuration |

The sign restriction belongs beside every Burgers exactness claim. Nonnegative truth is insufficient: decoded states and solver trial states must satisfy the condition too. Present exactness for the fixed polynomial stencil and measure the discrepancy from the original sign-upwind operator. Do not silently change the FOM discretization to obtain polynomial algebra. Extending the method to sign-changing fields would require a separately verified treatment; neither pointwise clipping nor an arbitrary positivity transform automatically preserves the existing precomputation.

## Novelty relative to existing work

Precomputed tensors for polynomial reduced operators are established in [tensorial POD](https://arxiv.org/abs/1402.2018). Nonlinear manifold Galerkin and time-discrete minimum-residual projection are established in [Lee and Carlberg](https://arxiv.org/abs/1812.08373). [Weder, Schwerdtner, and Peherstorfer](https://arxiv.org/abs/2412.17695) also establish online-efficient quadratic-manifold Neural Galerkin models for linear PDEs. These are primary sources checked during this review, not an exhaustive novelty search.

The candidate contribution is the learned spatial bank plus a flexible nonlinear coefficient map, combined with explicit weak discrete operators and evidence about when its compression pays. Demonstrating that coefficient nonlinearity helps at matched cost is essential. Exact tensor algebra alone does not establish originality, and the guarantees of the quadratic-manifold literature do not automatically transfer to a general neural head or a different projection.

## The bounded path to a submission

**Reproduce the configuration that motivated the idea.** Use the already specified historical replay, including its initializer and device-resident supplied input/full-field output. Keep the original same-grid FOMs explicitly named and preserve the separately named Poisson direct control. The approved replay is not replaced by the newer coarse-mesh/host-output comparison. For the eventual paper, show an efficient same-grid solver alongside the historical comparator; a replay against one solver supports a claim about that solver.

**Isolate precomputation.** At one frozen checkpoint, compare full-grid weak projection, corrected EQ, and exact precomputed algebra. Keep tests, row scaling, initialization, time discretization, stopping, and output contract fixed. Check residuals, Jacobians, accepted and rejected solver states, physical trajectories, and sign violations. Charge and report offline operator construction separately. This tests whether removing EQ changes fidelity, setup burden, and online work.

**Isolate the learned nonlinear map.** Compare the nonlinear head with a linear map at matched latent dimension, a quadratic map at matched latent dimension, and unrestricted bank coefficients. Include a properly solved linear POD comparison, rather than relying on instability of a different Galerkin rollout as evidence for decoder nonlinearity. Keep the weak objective and time solver comparable where mathematically applicable; test an adequately overdetermined system for each solved dimension. Report bank projection and best-recorded nonlinear fitting errors separately from rollout errors. Unrestricted coefficients reveal what accuracy is lost to compression and what solver work it saves.

**Freeze and confirm a compact scope.** Make Poisson the algebraic correctness and limitation case; make Burgers the nonlinear performance case. Heat is useful only with verified recursion and current matched timings. Wave and further dimensional/PDE expansion need not delay this paper. Select settings on development data, repeat training seeds, then evaluate an unopened cohort with frozen choices. Report physical error distributions, worst cases, nonstationary/budget exits, timing medians and outliers, saved repetitions, complete device-query cost, and separate setup/input/output costs. Measure tolerance sweeps at fixed weights; do not assume a frontier exists because a solver accepts a tolerance argument.

A suitable working title is **Nonlinear Manifold Reduced Models with Precomputed Weak Operators**. A potential abstract claim is: “We investigate separable learned decoders whose nonlinear latent coordinates admit exact preassembly of linear and polynomial weak operators, and characterize the accuracy, cost, and discretization limits of eliminating empirical quadrature.” Replace “investigate” with stronger result language only when the controlled comparisons support it.

## Alternative: separate CP/EQ and precomputed-operator papers

Separate papers are plausible if each answers a distinct question and has enough independent evidence. Merely replacing the residual evaluator while repeating the same architecture, tunability claim and headline experiments would leave the second contribution narrow. The following separation is a proposed research structure, not a conclusion that both papers are already ready for submission.

| Dimension | CP/EQ paper | Precomputed-operator paper |
| --- | --- | --- |
| Central question | Can a compact CP decoder support reliable intrusive solving and useful runtime accuracy–cost choices from fixed weights? | Which decoder/operator structures permit exact reduced preassembly, and when is it preferable to sampled evaluation? |
| Main mechanism | Efficient evaluation at selected stencil nodes, a nonlinear coefficient map, corrected EQ and a precisely defined latent solve | Linear reduced matrices and polynomial tensors acting on nonlinear coefficients, with explicit operator and rank restrictions |
| Main evidence | Cold-start and rollout reliability, CP/skip ablations, fixed-checkpoint solver/EQ curves, tuned neural and classical baselines | Full-grid/EQ/precomputed comparisons at fixed checkpoints, residual and Jacobian parity, bank-size/time/memory tradeoffs, nonlinear versus simpler coefficient maps |
| Most useful distinctive scope | Operators whose sampled discrete evaluation remains practical when an exact small tensor is unavailable or expensive | Fixed or suitably parameter-separated linear operators and affordable polynomial nonlinear operators |
| Main scientific risk | Corrected results may not establish enough benefit or novelty beyond existing NM-ROM combinations | Precomputability already exists in tensorial POD and quadratic-manifold work; neural compression needs a demonstrated benefit |

For the CP/EQ paper, a possible title is **Tunable CP-Decoder Manifold ROMs with Empirical Quadrature**. The original manuscript supplies its starting material, but its results must pass the existing corrections: discrete truth and boundary consistency, correct nonlinear time stepping, actual solver equations, matched timings, fixed checkpoints along each curve, and properly tuned baselines. Correction work restores validity; it does not by itself supply a new scientific contribution. Do not retain the blanket claim that EQ is necessary for CP decoders on linear PDEs. A sign-dependent upwind operator or a bank too large for an economical dense tensor offers a concrete motivation to investigate sampled evaluation. Its value still needs measurement.

For the precomputed-operator paper, retain the working title **Nonlinear Manifold Reduced Models with Precomputed Weak Operators**. Its scope should include the conditions under which the reduction is exact, the cost dependence on bank size, the benefit and accuracy loss of the nonlinear coefficient map, and the limitations of transferring weights between meshes. Strong evidence would combine the existing operator-parity tests with controlled linear/quadratic/neural head comparisons, a cost and memory crossover against EQ, and fresh confirmation. It must retain the Burgers sign qualification and avoid claiming that the existing modest tensor-versus-EQ online difference alone establishes a broad new acceleration method.

The architectures and evaluation strategies are separate choices: the original CP decoder also admits precomputation in suitable cases. This is a bridge between the papers, not a reason to conceal that overlap. A small frozen-checkpoint CP control in the operator paper could establish that preassembly works beyond the newer coordinate-network bank; this is a proposed comparison only, not a restart of an older-decoder training campaign. If affordable preassembly dominates every CP/EQ case, that finding would weaken a separate EQ-centered paper and should change its scope or the decision to split.

Use the corrected CP/EQ method as an explicitly identified baseline in the operator paper. Shared datasets, checkpoints and reproduction tables should be identified and cross-referenced; each paper's new evidence should support its own main claim. One shared benchmark infrastructure is sufficient. The split does not require disjoint PDEs or new implementations of truth generation, timers and metric code.

The next concrete planning artifact is a pair of abstracts and a claim-to-figure allocation: assign each central claim its decisive experiment and acceptance criterion, then assess whether either paper depends on the other's claimed novelty. Prepare the CP/EQ correction inventory first while the already approved historical replay continues. Launching additional CP or operator experiments remains a separate action requiring the repository's normal base/worktree handling. The recommendation is to develop both outlines now and retain two submissions only if both independently earn their claims.

## Selected emphasis: tunability and mesh-independent reduced computation

The user clarified that the CP/EQ paper should own the tunability story and the precomputed-operator paper should emphasize cost staying constant with mesh refinement. Adopt that division of headline claims. The earlier recommendation to place “Tunable” in both titles is withdrawn as the preferred publication framing; no mathematical or measured result is withdrawn.

For the CP/EQ paper, hold a trained checkpoint fixed and vary solver budgets/tolerances and precomputed quadrature choices. Its principal figure is physical error against query cost. Demonstrate useful operating points and disclose saturation, stalled solves and non-monotone relationships between weak residual and field error. More optimization cannot escape the frozen decoder's representation limit.

For the precomputed-operator paper, hold reduced dimensions and solver policy fixed and vary spatial resolution. Its principal figure is cached reduced cost against mesh size, accompanied by physical accuracy, iteration counts and complete device-query timings. A fixed-size reduced residual/Jacobian evaluation has no explicit full-grid loop. Total latent solve cost is also independent of resolution in operation count when iteration and time-step counts are fixed, and can be approximately flat empirically when those counts stay comparable. Maintaining accuracy may require larger banks/tests or more work; that dependence must be measured. Use the working title **Nonlinear Manifold Reduced Models with Precomputed Weak Operators** and state the mesh-independent cached-cost claim precisely in the abstract.

This constant-cost claim is with respect to mesh size, not reduced dimensions or accuracy requirements. Offline bank/operator assembly remains mesh-dependent. Reading arbitrary dense input and reconstructing an arbitrary requested full field must touch its values, even with GPU-resident input and output. Full device-query timings may appear nearly flat over a finite measured range, but cached operator algebra alone does not prove asymptotically constant full-query cost. Preserve the user's historical device-query protocol and report its initialization/decoding costs alongside cached solving; do not silently replace it with another query contract.

The properties are not mathematically exclusive: the precomputed solver still has iteration/tolerance controls, and a fixed-size EQ rule can also remove explicit mesh dependence from repeated evaluations. The second paper's distinction must therefore include exact preassembly without fitted empirical nodes for the stated operator class, together with the bank/head and fidelity evidence described above. The contribution cannot be only the observation of a flat timing curve. No new tuning study for the second paper is proposed by this clarification.

## Plain-language glossary

- **NM-ROM / ROM / FOM:** nonlinear manifold reduced-order model / a solver using fewer coordinates / the full discretized solver.
- **PDE:** partial differential equation, the physical equation being approximated.
- **Spatial bank, coefficient map, latent coordinates:** fixed spatial functions / a neural or simpler map producing their weights / the unknown reduced variables solved for each query.
- **CP:** canonical-polyadic factorization, expressing a spatial array as a sum of products of one-dimensional factors.
- **$n$, $R$, $k$, $M$:** full field size, bank size, latent dimension, and number of weak equations.
- **Weak tests and row scaling:** functions used to average the PDE residual, and weights setting the relative importance of those averaged equations.
- **Residual, Jacobian, stationarity, curvature:** equation mismatch, its first derivatives, a vanishing optimization gradient, and second-derivative effects.
- **Galerkin / residual least squares:** setting residual projections to zero / minimizing the size of a specified residual. Their equations depend on the selected test space and norm.
- **Empirical quadrature / EQ:** fitting sparse spatial points and weights to approximate integrals or discrete sums.
- **Precomputed operator / tensor / quadrature-free:** cached reduced matrix or multidimensional coefficient array / an array encoding a polynomial operation / eliminating fitted empirical spatial sampling from the repeated reduced operator, while still allowing offline spatial integration.
- **Upwind, sign restriction, undershoot:** choosing a difference stencil according to transport direction, the state condition making a fixed stencil valid, and a decoded field taking unwanted negative values.
- **Projection error, fitting error, rollout:** error from restricting fields to a spatial bank, error from fitting the nonlinear representation, and the sequence of computed time steps.
- **POD:** proper orthogonal decomposition, a data-based construction of a linear spatial basis.
- **CG / direct spectral solver:** conjugate gradient, an iterative linear solver / a transform-based direct solver available for the particular separable operators.
- **Checkpoint, seed, cohort, frozen transfer:** saved weights / recorded randomness / a collection of inputs / evaluating unchanged neural weights on another mesh.
- **Device query, offline setup, paired timing:** a timed computation beginning and ending in GPU memory / preparation outside each query / alternating methods within the same hardware allocation.
- **Table columns:** “Issue” identifies a statement needing repair; “Required treatment” describes the correction; “Evidence” names the retained study; “Supported interpretation” states its warranted conclusion; “Remaining limitation” states what it does not establish. In the paper comparison, “Dimension” names the planning question and each paper column specifies its proposed scope.
