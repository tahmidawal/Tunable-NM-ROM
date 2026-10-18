# 07 — Novelty check for the JCP off-mesh paper (literature sweep, 2026-10-06)

This note follows up `02-venue-literature.md` and does not repeat it. It covers three things: a full read of the three arXiv preprints and the LiCROM cubature section that note 02 left unread; a new 2024–2026 sweep; and a verdict on each of the five claimed contributions. The status of each verdict is **provisional**. The sweep used web search and arXiv abstract and HTML pages only. Google Scholar, ScienceDirect and Wiley full texts were not accessible, so a paywalled 2025–26 journal paper with no arXiv twin could have been missed.

Verification marks:
- **R**: full text or the relevant section was read this session.
- **A**: only the arXiv abstract page was read.
- **S**: known only from a search snippet. These are **unverified**.

---

## 1. The four items left open by note 02

### 1a. arXiv:2505.14595 (R, HTML v2)
**N. Hosseini Dashtbayaz, H. Salehipour, A. Butscher, N. Morris, "Physics-informed Reduced Order Modeling of Time-dependent PDEs via Differentiable Solvers" (Φ-ROM).** NeurIPS 2025, according to the arXiv comment; v1 is from 20 May 2025 and v2 from 25 Oct 2025.

**What it does.** Φ-ROM uses a conditional INR decoder (auto-decoding, DINo-style). Training couples the decoder to a *differentiable numerical solver*. The target latent velocity is $\dot\alpha = J_D^{\dagger}(\alpha)\,\dot{\hat u}$, and $\dot{\hat u}$ comes from the solver, for example $(S[\hat u]-\hat u)/\Delta t_S$. During training it uses "stochastic hyper-reduction": a random $\gamma N$ subset of the **solver grid** ($\gamma = 0.1$), solved by QR least squares. Online it integrates a learned latent ODE with Bogacki–Shampine. It claims robustness to the solver's discretisation and grid.

**Pre-emption: no.**
- The physics is the *discrete solver's*, so the ROM inherits that solver's stencil error. This is the opposite design to our continuum target.
- It has no test functions, no quadrature weights and no online projection solve.
- It has no ordering of the latent dimensions.

**Cite it as** a 2025 INR ROM that ties the latent dynamics to the mesh solver. It is the clearest contrast for our continuum-vs-mesh claim.

### 1b. arXiv:2507.07830 (R, HTML v3)
**S. N. Rodriguez, S. L. Brunton, L. K. Magargal, P. Khodabakhshi, J. W. Jaworski, N. A. Apetre, J. C. Steuben, J. G. Michopoulos, A. Iliopoulos, "Meshless projection model-order reduction via reference spaces for smoothed-particle hydrodynamics".** arXiv only, v3 of 30 Oct 2025; no venue is listed.

**What it does.** SPH snapshots are interpolated onto a fixed meshless reference space (§3.1, Shepard-filtered SPH interpolation), and POD is applied there. Polyharmonic-spline interpolation (§3.2) is the trial map back to the moving particles. Projection is Galerkin or adjoint Petrov–Galerkin (§3.3). §1.3 says explicitly: "hyper-reduction ... is not included in this work but is currently under development."

**Pre-emption: no.** The basis is linear POD, and the paper has no hyper-reduction, no quadrature, no neural field and no ordering beyond POD's own. It is relevant only as "meshless ROM" background.

### 1c. arXiv:2508.21279 (R, HTML v2)
**C. Vales, S. W. Cheung, D. M. Copeland, Y. Choi, "Machine-precision energy conservative reduced models for Lagrangian hydrodynamics by quadrature methods".** arXiv v2 of 5 Mar 2026, LLNL-JRNL-2010206. It is a journal submission and is not yet published.

**What it does.** The model is a linear POD ROM of FEM Lagrangian hydrodynamics. Hyper-reduction is by EQP, posed as NNLS, which picks a sparse subset of the **existing FEM quadrature points** (§3.1: EQP "does not attempt to identify new quadrature points"). The new element is an energy-conservative EQP variant (§4).

**Pre-emption: no.** It is the opposite of our approach: snapshot-fitted, mesh-point-bound and linear. **Use it as** a 2026 example of the fitted-EQP incumbent, and as evidence that the LLNL group (libROM) is still investing in fitted quadrature.

### 1d. LiCROM cubature section (R, arXiv HTML 2310.15907)
**Y. Chang, P. Y. Chen, Z. Wang, M. M. Chiaramonte, K. Carlberg, E. Grinspun, "LiCROM: Linear-Subspace Continuous Reduced Order Modeling with Neural Fields", SIGGRAPH Asia 2023 Conf. Papers.**

**Cubature (§5).** LiCROM uses a self-described "naïve cubature scheme". It selects $m$ **mesh vertices at random**, adds all vertices incident to them, and gives the result **equal weights**. The points are precomputed per mesh, including for every mesh in a remeshing sequence. The authors note that data-aware sampling could reduce the point count. The subspace is linear, $u(X,t) = W(X)\,q(t)$, with **no ordering of the basis**.

**Pre-emption: partly, and only on the "linear neural-field subspace" part of our bank** (already credited). Its cubature is random, equal-weight and mesh-vertex-based. That makes it a contrast case for classical off-mesh rules, not a precedent.

---

## 2. New 2024–2026 sweep

### 2a. Neural-field / INR ROMs and their integration points

| Citation | Mark | Relevance | Pre-empts? |
|---|---|---|---|
| M. Liu, Y. Chang, Z. Wang, P. Y. Chen, E. Grinspun, "Precise Gradient Discontinuities in Neural Fields for Subspace Physics", SIGGRAPH Asia 2025 / ACM TOG (arXiv:2505.20421) | R | A neural-field subspace solved by energy minimisation. Integrals use "stochastic cubature with uniform sampling". This is the second graphics paper, after Simplicits, to use **non-fitted MC** cubature. | No. It strengthens our point that this community uses MC, not characterised rules. |
| Y. Liu, Z. Fang, S. Darkner, N. Aigerman, K. Erleben, P. Kry, T. Schneider, "Neural Kinematic Bases for Fluids", arXiv:2504.15657 (2025) | A | An MLP kinematic basis trained with orthogonality, divergence-free and smoothness losses, giving mesh-free fluid animation. The integration scheme was not visible in the abstract. **Not fully read.** | Probably not. Check the full text before submission for any ordering or cubature. |
| A. Prakash, M. L. Klasky, "Structure-preserving variational neural fields: uncertainty-quantified ROM of nonlinear conservation laws", arXiv:2607.10965 (Jul 2026) | A | A latent neural-field ROM from the SNF-ROM co-author, with a conservation-law manifold. Its projection and quadrature are unknown from the abstract. | Unknown, likely no. **Read before submission.** |
| N. Ning, B. Peherstorfer, "Filtered Neural Galerkin model reduction schemes...", arXiv:2511.00670 (2025) | R | A Neural Galerkin ROM whose collocation uses "N=1000 equidistant points" (Burgers) and a 128×128 equidistant grid. | No. It confirms the Peherstorfer line still uses unweighted equidistant points. |
| J. S. Hesthaven, B. Peherstorfer, B. Unger, "Nonlinear model reduction for transport-dominated problems", arXiv:2602.01397 (Feb 2026, survey) | A | A survey organised around parametrisations, reduced dynamics and online solvers. Its treatment of sampling and quadrature was not visible. | No, being a survey. **Cite it as the up-to-date review, and read §"online solvers" to check whether it already frames off-grid quadrature as a category.** |
| J. Berman, B. Peherstorfer, "Randomized Sparse Neural Galerkin Schemes...", NeurIPS 2023 | S | Randomised sparse *parameter* subsets per step as an accuracy/cost lever. | No. It is a different dial, over parameters rather than an ordered basis width. |
| N. Sibuet, S. Ares de Parga, J. R. Bravo, R. Rossi, "A discrete physics-informed training for projection-based ROMs with neural networks", arXiv:2504.13875 (2025) | A | PROM-ANN with an FEM-discrete residual loss. | No. The physics is mesh-discrete. |

### 2b. Hyper-reduction (2024–2026)

| Citation | Mark | Relevance | Pre-empts? |
|---|---|---|---|
| J. A. Hernández, S. Ares de Parga, R. Rossi, "Dimensional hyperreduction of nonlinear FE models via empirical cubature with manifold-adaptive weights", arXiv:2609.03068 (Sep 2026) | A | MAW-ECM: cubature weights that vary with the latent coordinates on a nonlinear (non-neural) manifold, built by greedy pruning from snapshots. It is the newest fitted-cubature state of the art for nonlinear manifolds. | No. It is snapshot-fitted and uses FE Gauss points. **Cite it as the current incumbent.** |
| A. Larsson, M. Kim, C. Vales, S. Adriaenssens, D. M. Copeland, Y. Choi, S. W. Cheung, "Hyper-reduction methods for accelerating nonlinear FE simulations: open source implementation and reproducible benchmarks", arXiv:2602.23551 (2026) | A | A gappy-POD vs EQP benchmark, open source. | No. **It is a ready-made faithful EQP baseline**, which addresses red-team item 3 in note 02. |
| R. Humphry et al., "Efficient Hyperreduction for Large-Scale Problems: Exploiting Reducible Constraint Manifolds in EQP", *IJNME* 2025, doi:10.1002/nme.70204 | S | Lowers the cost of the EQP fit. | No. **Authors other than the first were not verified.** |
| B. Liljegren-Sailer, arXiv:2512.14416 | (in 02) | — | — |
| Ingimarson, Rebholz, Iliescu, "Full and reduced order model consistency of the nonlinearity discretization in incompressible flows", *CMAME* 2022, doi:10.1016/j.cma.2022.115620 (arXiv:2111.06749) | A | Argues that the ROM **should match** the FOM's nonlinear discretisation, because consistency gives optimal bounds *with respect to the FOM*. | No. It is the **counter-position** to our continuum-target framing, and a referee may cite it. We must state that our error target is the continuum solution, not the FOM, and explain why that changes the analysis. (Older than 2024, but essential.) |

**Searches with no ROM hit.** None of these queries found a ROM or hyper-reduction paper using these rules: "hyper-reduction" + "lattice rule" / "Fibonacci lattice"; "hyper-reduction" + Smolyak / sparse grid (the only hits used sparse grids in *parameter* space); "hyper-reduction" + Gauss quadrature + neural decoder.

### 2c. QMC, lattice and classical quadrature for neural PDE residuals (full-order or training-time)

| Citation | Mark | Relevance | Pre-empts? |
|---|---|---|---|
| S. Badia, K. Nori, "Adaptive anisotropic composite quadratures for residual minimisation in neural PDE approximations", arXiv:2605.00308 (May 2026) | A | An error split into approximation, quadrature and optimisation, with **quadrature error controlled against a richer reference rule**. | Partly, for the *error-indicator* idea, but in a training context and not for ROMs. |
| J. M. Taylor, D. Pardo, "Stochastic Quadrature Rules for Solving PDEs using Neural Networks", arXiv:2504.11976 (2025, v. Oct 2026) | A | Deterministic or biased rules can give wrong Deep Ritz solutions; unbiased high-order stochastic rules help. | No. It is a useful *caveat* to cite: we use a fixed rule online, not in training. |
| T. Yu, I. Oseledets, "Quasi-Random Physics-informed Neural Networks", *Neurocomputing* 2026 (arXiv:2507.08121) | A | Low-discrepancy collocation for PINNs. | No, it is training-time. Add it to the Matsubara–Yaguchi line. |
| Y. Yang, P. He, X. Peng, Q. He, "A novel number-theoretic sampling method for neural network solutions of PDEs", arXiv:2411.17039 (2024/25) | A | Good-lattice-point PINN collocation with error bounds. | No, it is training-time. Same line. |
| C. Smaragdakis, "Learning Geometric-Aware Quadrature Rules for Functional Minimization" (QuadrANN), arXiv:2508.05445 (2025) | A | Learned quadrature weights on point clouds, compared against QMC. | No. |
| J. Dölz, F. Henríquez, "Fully discrete analysis of the Galerkin POD neural network approximation...", arXiv:2502.01859 (2025) | A | QMC is used in *parameter* space for POD-NN. | No. |

### 2d. Ordered, nested or tunable bases

| Citation | Mark | Relevance | Pre-empts? |
|---|---|---|---|
| O. Rippel, M. Gelbart, R. Adams, "Learning ordered representations with nested dropout", ICML 2014 | S/K | The origin of ordering by nested dropout; it reduces to PCA in the linear case. | **Must cite.** It is the ordering mechanism (or its ancestor) for any "importance-ordered" neural bank. It does not involve ROMs or a deployment dial. |
| N. Aretz, K. Willcox, "Nested Operator Inference for Adaptive Data-Driven Learning of ROMs", arXiv:2508.11542 (2025) | A | Uses the POD hierarchy to warm-start OpInf for each target dimension. | No. The ordering is POD's, and each dimension gets its own learned model. |
| "Sparse POD Mode Selection and Manifold Dimensionality Reduction with Neural Networks", arXiv:2605.27756 (2026) | S (authors **not verified**) | Selects modes beyond energy ordering. | No. |
| L. J. Shikhman, "Operator Boosting Produces Pareto-Efficient PDE Surrogates", arXiv:2606.17460 (accepted KDD 2027 AI4Sciences per arXiv) | A | Stagewise residual stacks of small operators; truncating the stack would be a dial, but the paper does not evaluate one. | No. **Nearest "one model, many costs" neighbour in operator learning; cite and distinguish.** |
| Slimmable / once-for-all networks (Yu & Huang 2019, arXiv:1903.05134) | S | One set of weights runs at several widths. This is the generic ML ancestor of a width dial. | No ROM use found. Cite it as the ML ancestor. |
| Matryoshka representation and SAE work (e.g. arXiv:2503.17547) | S | Nested prefixes of a representation. | No PDE or ROM use found. |
| F. A. B. Silva, J. C. Ragusa, T. Guo, R. Geelen, "Residual-Driven Lifting Identification for NM-ROMs of Parametrized Linear PDEs", arXiv:2607.27471 (Jul 2026) | A | A **snapshot-free** NM-ROM trained by minimising a computable residual-based error bound (linear, affine PDEs). | Partly pre-empts any "data-free" wording; see claim 5. Its quadrature for the dual norm was not visible. **Read before submission.** |

**What the ordering sweep found.** Searches combined "nested dropout", "matryoshka", "slimmable", "adaptive rank" and "tunable" with "reduced-order", "neural field", "neural operator" and "deployment". None found a ROM or neural operator in which **one trained neural basis is truncated at deployment** to trade accuracy against cost. The closest items are POD itself (trivially ordered, linear), Nested OpInf (a POD hierarchy with one model per dimension) and Operator Boosting (stackable, but not evaluated as a dial).

**Not accessed.**
- Google Scholar directly.
- ScienceDirect, Wiley and SIAM full texts.
- Non-English literature.
- The full texts of arXiv:2504.15657, 2607.10965, 2602.01397 and 2607.27471.

---

## 3. Verdict per claimed contribution

### C1. Tunable, importance-ordered neural-field bank (width $R'$, optional head, as a deployment dial)
**Verdict: NOVEL in the ROM setting, with ancestry that must be credited.**

- Ordering by nested dropout is Rippel et al. 2014.
- Width-adaptive single networks are the slimmable / once-for-all line.
- Linear neural-field subspaces are LiCROM and Simplicits.
- Ordered linear bases are POD itself.

No 2023–2026 ROM, INR-ROM or neural-operator paper found offers a truncatable neural trial basis as an online accuracy/cost dial.

**Safe wording:** "a coordinate-network trial bank trained so that its leading $R'$ functions form a usable reduced basis for every $R'$, giving a deployment-time accuracy/cost dial from a single trained model, in the spirit of POD truncation and nested-dropout ordering."

**Avoid:** "first ordered neural basis" (Rippel) and "first tunable ROM" (POD truncation is a tunable ROM).

### C2. Fixed classical quadrature (tensor Gauss–Legendre, randomly shifted rank-1 lattices) replacing snapshot-fitted hyper-reduction in an LSPG neural-field ROM
**Verdict: NOVEL as a combination. Each ingredient is PARTLY PRE-EMPTED.**

| Ingredient | Already done by |
|---|---|
| Non-fitted points for neural-field ROMs | Simplicits 2024, Liu et al. 2025 (MC); LiCROM 2023 (random equal-weight mesh vertices); Weder et al. 2025 and Ning–Peherstorfer 2025 (equidistant collocation); SNF-ROM (uniform grid subsample) |
| Lattice / QMC for neural residuals | Matsubara–Yaguchi 2025, Yang et al. 2024, Yu–Oseledets 2026, all training-time |
| Gauss quadrature with neural trial and fixed tests | VPINN / hp-VPINN, full-order |
| Off-mesh hyper-reduction points | CECM 2024, MAW-ECM 2026, both fitted |

Nobody found uses **weighted, convergence-characterised classical rules for the online hyper-reduced Petrov–Galerkin residual of a ROM**. The three newly read preprints do not change this. Φ-ROM uses a random grid subset of a discrete solver. Vales et al. use fitted EQP on FEM points. Rodriguez et al. have no hyper-reduction at all.

**Safe wording:** "we replace snapshot-fitted mesh-point hyper-reduction (ECSW/EQP/ECM) with fixed, state-independent classical rules, whose error is governed by integrand regularity rather than by training snapshots."

**Must also cite** Ryckelynck 2005 (the "a priori hyperreduction" name) and avoid that phrase.

**Storage claim.** The $m(2R'+M)$ vs $MR'^2$ comparison is an arithmetic consequence of the formulation, not a literature claim. It is safe as long as it is stated as a comparison against the precomputed quadratic tensor (Carlberg 2011 compressive tensors; standard Galerkin-POD tensors).

### C3. Continuum-vs-mesh-target decoupling (ROM error no longer inherits the stencil's $O(h)$ error)
**Verdict: PARTLY PRE-EMPTED (mechanism by SNF-ROM; opposing position by Ingimarson–Rebholz–Iliescu 2022). The measured decoupling is new.**

- SNF-ROM already motivates AD derivatives by avoiding low-order FD stencil error.
- Φ-ROM (2025) is a recent example of the opposite choice, tying the ROM to the discrete solver.
- Ingimarson et al. argue for FOM–ROM consistency.

No paper found measures ROM error against mesh $h$ and shows it becoming $h$-invariant once the tested residual is integrated on the continuum. Nor does any paper reframe hyper-reduction certification against the continuum target.

**Safe wording:** "we quantify ... ROM error that is invariant to the training mesh at fixed bank, whereas the stencil-tested ROM tracks the stencil's $O(h)$ error."

- Credit SNF-ROM for the mechanism.
- Discuss Ingimarson et al. explicitly: they measure error to the FOM, and we measure error to the continuum.
- Keep the note-02 caveat: the claim needs a high-order or exact reference.

### C4. Rule-family characterisation for tensor sine tests (Gauss geometric; lattice from boundary vanishing; Smolyak fails on mixed frequencies; Sobol $\sim 1/m$; MC useless)
**Verdict: NOVEL as a ROM-context characterisation. Every individual fact is classical.**

The individual facts are:
- Gauss convergence for analytic integrands: Trefethen.
- Lattice and trapezoid behaviour for effectively periodic integrands: Sloan–Joe; Trefethen–Weideman; Dick–Nuyens–Pillichshammer.
- Smolyak exactness classes: Novak–Ritter; Bungartz–Griebel.

No ROM, VPINN or PINN paper found compares Gauss, lattice, Sobol, Smolyak and MC on the tested residual of a reduced model. Berrone–Canuto–Pintore 2022 is the closest *theory*, and Taylor–Pardo 2025 is a caveat about deterministic rules in training.

**Safe wording:** "we show that classical rule theory predicts and explains ... in this setting". Present it as an application of known theory with a new consequence, not as new quadrature theory. Restrict the Smolyak statement to "tensor-product tests with high mixed frequencies".

### C5. Data-free error indicator
**Verdict: PARTLY PRE-EMPTED in principle. Wording is the risk.**

Snapshot-free error estimation already exists in three forms:
- Classical reduced-basis dual-norm residual estimators.
- Error control of a quadrature against a richer reference rule: Badia–Nori 2026, and Gauss–Kronrod practice generally.
- Standard-error estimates from random-shift replicates of lattice rules: Dick–Kuo–Sloan 2013, a textbook QMC tool.

Silva et al. 2026 train an NM-ROM snapshot-free by minimising a residual error bound. EQP (Yano–Patera 2019) has a-posteriori quadrature-error control, but it is fitted.

What appears new is the use of these indicators to **certify the online hyper-reduction of an NM-ROM without held-out snapshots**, which is precisely what fitted EQ cannot do.

**Safe wording:** "a snapshot-free indicator of the hyper-reduction (quadrature) error, obtained from rule refinement / independent random shifts."

**Avoid:**
- "error estimator" or "certified" unless it is shown to be a bound.
- "first data-free".

---

## 4. Actions before submission

1. **Read the full texts** of arXiv:2607.27471 (how its residual dual norm is integrated), 2607.10965, 2602.01397 (whether the survey already names off-grid quadrature) and 2504.15657.
2. **Add citations:**
   - New: Φ-ROM, Vales et al., Liu et al. 2025, MAW-ECM 2026, Larsson et al. 2026, Ingimarson et al. 2022, Rippel et al. 2014, Badia–Nori 2026, Taylor–Pardo 2025, Yu–Oseledets 2026, Aretz–Willcox 2025, Hesthaven–Peherstorfer–Unger 2026.
   - Optional: Operator Boosting and slimmable networks.
3. **Baseline:** use the open-source EQP from Larsson et al. 2026 (libROM) as the faithful fitted-EQP baseline.
4. **Verify the S-marked items** before citing: Humphry et al. authors, arXiv:2605.27756 authors, Rippel et al. venue details.

## Glossary

- **Pre-empts / partly pre-empts:** whether an earlier paper already makes the same claim (all of it, or one ingredient of it).
- **R / A / S:** how a citation was checked — R = full text or relevant section read; A = arXiv abstract only; S = search snippet only (unverified).
- **INR / neural field / coordinate network:** a neural network that takes a spatial coordinate $x$ and returns a field value, so it can be evaluated anywhere, not only on mesh nodes.
- **Bank $G(x)$, width $R'$:** our set of $R'$ neural basis functions; the solution is $u(x)=G(x)c$.
- **LSPG / Petrov–Galerkin:** the reduced solve minimises the PDE residual measured against a set of $M$ test functions (here tensor sines), rather than against the trial basis itself.
- **Hyper-reduction:** making the nonlinear residual cheap to evaluate by computing it at only $m$ points with weights.
- **ECSW / EQP / ECM / CECM / NNLS:** families of hyper-reduction that *fit* points and weights to training snapshots (NNLS = non-negative least squares).
- **Classical / fixed rule:** a quadrature rule built without data (Gauss–Legendre, lattice, Sobol, Smolyak, Monte Carlo).
- **Rank-1 lattice, CBC, random shift:** a QMC point set generated by one integer vector (built component-by-component); randomly shifting it gives independent replicates from which an error estimate can be computed.
- **Smolyak / sparse grid:** a combination of low-order tensor rules that is exact only for "low mixed-degree" functions.
- **Stencil $O(h)$ error:** the error a finite-difference formula on mesh spacing $h$ makes against the exact derivative.
- **Continuum target:** measuring the ROM against the exact PDE solution, not against the discrete FOM.
- **Nested dropout / slimmable / Matryoshka:** ML training tricks that make the leading units of a network usable on their own.
