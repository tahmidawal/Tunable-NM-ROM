# 02 — JCP venue expectations, literature map, novelty positioning

Agent 2 of 5, 2026-10-06. Inputs read: Hari's `LITERATURE.md` and `SUMMARY.md` (quadrature-study-2026-09-30), `reports/2026-10-01-burgers3d-offmesh-quadrature.md`, and web sources listed inline. The ScienceDirect Guide for Authors returned HTTP 403 to every fetch route (WebFetch, curl, browser pane), so the JCP policy statements below come from search-engine extracts of that page. They are marked **[GfA-snippet]** and must be re-checked by a human in a browser before submission.

---

## 1. What JCP expects

### Scope
JCP "focuses on the computational aspects of physical problems" and "encourages original scientific contributions in advanced mathematical and numerical modeling ... interdisciplinary in nature", with editors who "seek to emphasize methods that cross disciplinary boundaries" (aims and scope, via [search extract](https://www.sciencedirect.com/journal/journal-of-computational-physics)). The editorial board has dedicated scientific-ML associate editors, and the Editor-in-Chief, Dongbin Xiu, works in ML for scientific computing ([editorial board](https://www.sciencedirect.com/journal/journal-of-computational-physics/about/editorial-board)). A neural-manifold ROM is within scope. JCP has already published our closest neighbours: Lee–Carlberg 2020, Kim–Choi–Widemann–Zohdi 2022, Romor–Stabile–Rozza 2025, Weder–Schwerdtner–Peherstorfer 2025, and **SNF-ROM (Puri et al., JCP 532, 2025)**. The editors will route the paper to referees from those groups.

### Article types and length [GfA-snippet]
- **Regular article.** No formal page limit; work should be "clear and concise". Typical JCP ROM papers run 25–40 pages in the Elsevier preprint format, with 6–12 figures and 4–8 tables. Lee–Carlberg runs about 30 pages and SNF-ROM about 35.
- **Short Note.** At most 4 pages, no abstract. It does not fit this paper.
- Highlights (3–5 bullets) are encouraged. The abstract is a single paragraph; aim for 150–250 words.

### Data and code policy [GfA-snippet]
- JCP applies Elsevier research-data **Option C**. Authors must deposit research data in a repository and cite and link it, or state why they cannot share it. A data-availability statement is required at submission.
- JCP "encourages authors to provide a link to their code and data ... in a public git repository, an open repository with DOI ... or supplementary materials", and **the link is published prominently with the paper**.
- What we need to provide: a Zenodo DOI snapshot of the code-only mirror, plus the run JSONs, the frozen `selection.json` with its SHA, and the scripts that build the tables. Our existing "tables generated from JSON" discipline is a selling point and should be said explicitly. The 4.3 GB tensors and the trained banks need either a DOI deposit or a regeneration recipe with seeds.

### Theory versus computation
JCP is computation-led, but referees expect each claim of a method to be backed by an error statement. For this paper a full a-priori theory is not required. What is required:
- a short proposition bounding the quadrature error of the tested nonlinear term by known rule-error estimates, given the integrand's regularity (analytic decoder, boundary vanishing, test bandwidth);
- an explanation of why Smolyak fails, in terms of mixed-frequency exactness;
- a consistency argument for the $O(h)$ continuum-versus-stencil gap;
- numerical convergence studies that confirm each of these.

About one theory section (4–6 pages) plus 3–4 computational sections is the right balance.

### What JCP referees demand of ROM papers
This list combines the GfA, the reviewing culture visible in the cited JCP ROM papers, and Carlberg-school norms.

1. **A FOM baseline that is efficient and fairly tuned.** The comparison must be wall-clock against the best same-accuracy full-order setting on the same hardware, not against a deliberately slow FOM. *We are exposed here.* In 3D, the off-mesh ROM is **slower** than tuned Newton–BiCGStab at 64³ and 128³ (same-grid speedups of 0.11–0.54× at $R'=512$). It wins only at 256³ (2.6–7.8×), and at 256³ the best FOM setting is more accurate than either ROM. The paper must say this. It cannot be framed as a speed paper.
2. **A full cost account.** Separate offline cost (bank training, tensor or EQ fit, rule construction) from online cost. Report flops, bytes and memory per Jacobian, and state which online pieces still scale with $N$: the projection of $u_0$ and the output decoding do.
3. **Hyper-reduction baselines.** Compare against at least one data-fitted method (ECSW or EQP-style NNLS), DEIM/GNAT-type collocation where applicable, the exact quadratic tensor, and dense evaluation. Hari's re-implemented EQ needs to become a faithful ECSW/EQP baseline, or be labelled honestly as a re-implementation.
4. **Convergence studies.** Show rule error ρ against point count $m$ for every rule family, ROM error against $R'$ and $M$, against mesh $h$, and against $\Delta t$. Also decompose the error budget into manifold (projection) error, test-space error, quadrature error, time error and solver error. The 3D report explicitly did **not** decompose the 2.83 %, and a referee will ask for it.
5. **A reference solution that converges in the quantity being claimed.** *This is the biggest risk.* The "continuum-target removes $O(h)$" claim is currently measured against a **first-order upwind** 513³ reference. That reference is biased, and the Richardson diagnostic was post hoc. JCP referees will ask for a high-order or spectral reference, or a manufactured or exact solution such as the Cole–Hopf solution of viscous Burgers in 1D or 2D.
6. **Generalisation and held-out testing.** Report the parameter or initial-condition distribution, a validation/held-out split, and at least one extrapolation case.
7. **More than one PDE, and harder physics.** 2D and 3D viscous Burgers alone will look thin. Hari's Allen–Cahn, Hamilton–Jacobi and Bratu cases help. Referees will still ask about non-smooth solutions (shocks at small ν) and about non-box or non-tensor geometry, which is where tensor sine tests and lattices stop working.
8. **Reproducibility details.** Precision, hardware, timing protocol (A–B–A), and seeds.

---

## 2. Literature map

The "Verified" column records how each citation was checked:
- **W**: confirmed this session by web search (title, venue, volume).
- **H**: read in full text by Hari's agent.
- **K**: standard citation known with high confidence but not re-fetched.
- **?**: not verified.

### 2a. LSPG and nonlinear-manifold ROMs

| Citation | Verified | Relevance |
|---|---|---|
| K. Carlberg, C. Bou-Mosleh, C. Farhat, "Efficient non-linear model reduction via a least-squares Petrov–Galerkin projection and compressive tensor approximations", *IJNME* 86(2):155–181, 2011 | K | Origin of LSPG and of compressive tensor approximation. Our solve is LSPG-type. |
| K. Carlberg, M. Barone, H. Antil, "Galerkin v. least-squares Petrov–Galerkin projection in nonlinear model reduction", *JCP* 330:693–734, 2017 | K | Time-discrete optimality of LSPG with backward Euler. This is the justification for our LM-per-step formulation. |
| K. Lee, K. Carlberg, "Model reduction of dynamical systems on nonlinear manifolds using deep convolutional autoencoders", *JCP* 404:108973, 2020 | H,K | Manifold Galerkin and LSPG. The decoder is mesh-bound and must be decoded on the full grid. |
| Y. Kim, Y. Choi, D. Widemann, T. Zohdi, "A fast and accurate physics-informed neural network ROM with shallow masked autoencoder", *JCP* 451:110841, 2022 | W | Masked sparse decoder plus DEIM/gappy mesh-node hyper-reduction. This is the NM-ROM hyper-reduction incumbent. |
| F. Romor, G. Stabile, G. Rozza, "Non-linear manifold ROMs with convolutional autoencoders and reduced over-collocation method", *J. Sci. Comput.* 94:74, 2023 | W | Greedy magic-point plus submesh hyper-reduction for CAE NM-ROMs. |
| F. Romor, G. Stabile, G. Rozza, "Explicable hyper-reduced order models on nonlinearly approximated solution manifolds of compressible and incompressible Navier–Stokes equations", *JCP* 524:113729, 2025 | W | Recent JCP NM-ROM hyper-reduction on the mesh. |
| A. N. Diaz, Y. Choi, M. Heinkenschloss, "A fast and accurate domain decomposition nonlinear manifold ROM", *CMAME* 425:116943, 2024 | W | DD-NM-ROM with sparse decoders and mesh hyper-reduction. |
| G. Cocola, J. Tencer, F. Rizzi, E. Parish, P. Blonigan, "Hyper-reduced autoencoders for efficient and accurate nonlinear model reductions", arXiv:2303.09630, 2023 | H | Hyper-decoder trained on a Q-DEIM sample mesh. |
| J. Barnett, C. Farhat, "Quadratic approximation manifold for mitigating the Kolmogorov barrier...", *JCP* 464:111348, 2022 | W | Quadratic manifolds with ECSW. |

### 2b. Hyper-reduction

| Citation | Verified | Relevance |
|---|---|---|
| M. Barrault, Y. Maday, N. C. Nguyen, A. T. Patera, "An 'empirical interpolation' method", *C. R. Math.* 339:667–672, 2004 | K | EIM, the root of interpolation-based hyper-reduction. |
| S. Chaturantabut, D. C. Sorensen, "Nonlinear model reduction via discrete empirical interpolation", *SISC* 32(5):2737–2764, 2010 | K | DEIM. |
| Z. Drmač, S. Gugercin, "A new selection operator for the discrete empirical interpolation method", *SISC* 38(2):A631–A648, 2016 | K | Q-DEIM. |
| K. Carlberg, C. Farhat, J. Cortial, D. Amsallem, "The GNAT method...", *JCP* 242:623–647, 2013 | K | GNAT, the canonical LSPG plus gappy hyper-reduction. |
| D. Ryckelynck, "A priori hyperreduction method", *JCP* 202:346–366, 2005 | K | The "a priori" wording is already taken. Cite it and avoid the bare phrase "a-priori hyper-reduction". |
| C. Farhat, P. Avery, T. Chapman, J. Cortial, *IJNME* 98(9):625–662, 2014; C. Farhat, T. Chapman, P. Avery, *IJNME* 102(5):1077–1110, 2015 | K | ECSW, the NNLS-weighted mesh-element sampling we replace. |
| S. Grimberg, C. Farhat, R. Tezaur, C. Bou-Mosleh, "Mesh sampling and weighting for the hyperreduction of nonlinear Petrov–Galerkin ROMs with local reduced-order bases", *IJNME* 122(7):1846–1874, 2021 | W | ECSW for LSPG. **Hari's file lists the wrong author (Youkilis); use these four.** |
| S. S. An, T. Kim, D. L. James, "Optimizing cubature for efficient integration of subspace deformations", *ACM TOG* 27(5):165, 2008 | K | NNLS cubature, the template for ECSW, ECM, EQP and CROM's sampler. |
| M. Yano, A. T. Patera, "An LP empirical quadrature procedure for reduced basis treatment of parametrized nonlinear PDEs", *CMAME* 344:1104–1123, 2019 | W | EQP, with a-posteriori-controlled quadrature error. This is the most rigorous empirical rival. |
| J. A. Hernández, M. A. Caicedo, A. Ferrer, "Dimensional hyper-reduction of nonlinear FE models via empirical cubature", *CMAME* 313:687–722, 2017 | K | ECM. |
| J. A. Hernández, J. R. Bravo, S. Ares de Parga, "CECM: a continuous empirical cubature method...", *CMAME* 418:116552, 2024 | W | **2024 continuous-point hyper-reduction.** Point locations become continuous design variables, but they are still fitted to snapshots. This is the closest "off-mesh" rival from the hyper-reduction side. |
| B. Liljegren-Sailer, "Reducing training complexity in empirical quadrature-based model reduction via structured compression", arXiv:2512.14416, 2025 | W | Current EQ work still targets offline fitting cost. That is the cost our approach removes. |

### 2c. INR and continuous decoders in ROMs

| Citation | Verified | Relevance |
|---|---|---|
| P. Y. Chen, J. Xiang, D. H. Cho, Y. Chang, G. A. Pershing, H. T. Maia, M. M. Chiaramonte, K. Carlberg, E. Grinspun, "CROM: Continuous reduced-order modeling of PDEs using implicit neural representations", ICLR 2023 (arXiv:2206.02607) | W,H | **Closest prior work.** Off-mesh integration samples, autodiff gradients, and the mesh-independent cost claim. Samples are greedy-selected from mesh nodes, the update is unweighted collocation, and there are no test functions. |
| P. Y. Chen, M. Chiaramonte, E. Grinspun, K. Carlberg, "Model reduction for the material point method via an implicit neural representation of the deformation map", *JCP* 478:111908, 2023 | H | INR ROM in JCP, integrated over a particle subset. |
| Y. Chang, P. Y. Chen, Z. Wang, M. M. Chiaramonte, K. Carlberg, E. Grinspun, "LiCROM: Linear-subspace continuous reduced order modeling with neural fields", SIGGRAPH Asia 2023 Conf. Papers (arXiv:2310.15907) | W | Linear neural-field subspace, the analogue of our bank. Cubature details were not checked (**?**). |
| Z. Zong, X. Li, M. Li, M. M. Chiaramonte, W. Matusik, E. Grinspun, K. Carlberg, C. Jiang, P. Y. Chen, "Neural stress fields for reduced-order elastoplasticity and fracture", SIGGRAPH Asia 2023 (arXiv:2310.17790) | W | INR evaluated at arbitrary points on sampled particles. No quadrature theory. |
| V. Modi, N. Sharp, O. Perel, S. Sueda, D. I. W. Levin, "Simplicits: mesh-free, geometry-agnostic elastic simulation", *ACM TOG* 43(4):117, 2024 | W | **Fixed, non-fitted cubature for a neural-field subspace already exists**, as uniform Monte Carlo. Our point is that MC is the worst rule; see Hari's ladders. |
| V. Puri, A. Prakash, L. B. Kara, Y. J. Zhang, "SNF-ROM: Projection-based nonlinear reduced order modeling with smooth neural fields", *JCP* 532:113957, 2025 | W (full text read) | **Critical JCP precedent.** Galerkin projection with a neural-field decoder and forward-mode AD spatial derivatives, explicitly to avoid "low-order finite difference stencil" error in CAE-ROMs. Hyper-reduction points "can be sampled anywhere in Ω", but in practice they are a uniform subsample of the FOM grid with no weights. It partly pre-empts the continuum-target argument. |
| P. Weder, P. Schwerdtner, B. Peherstorfer, "Nonlinear model reduction with Neural Galerkin schemes on quadratic manifolds", *JCP* 539:114249, 2025 (arXiv:2412.17695) | W,H | Collocation points decoupled from the FOM grid, "hyper-reduction baked in", N-independent cost. Points are uniform or equidistant, with no weights and no test functions. |
| J. Bruna, B. Peherstorfer, E. Vanden-Eijnden, "Neural Galerkin schemes with active learning for high-dimensional evolution equations", *JCP* 496:112588, 2024 | H | Monte Carlo estimation of Galerkin integrals with adaptive measures. It notes that in low dimension the integrals "can be performed by quadrature on a grid". |
| J. Berman, B. Peherstorfer, "CoLoRA: Continuous low-rank adaptation for reduced implicit neural modeling of parameterized PDEs", ICML 2024 (arXiv:2402.14646) | W | Neural field with an equation-driven Neural Galerkin variant at sample points. |
| Z. Wen, K. Lee, Y. Choi, arXiv:2311.16410, 2023; Y. Kim, K. Lee, Z. Wen, Y. Choi, "Physics-informed ROM with conditional neural fields", arXiv:2412.05233, 2024 | W (titles) | CNF-ROM. The PDE residual is a training loss only, with a latent-ODE online stage. |

### 2d. Quadrature theory

| Citation | Verified | Relevance |
|---|---|---|
| I. H. Sloan, S. Joe, *Lattice Methods for Multiple Integration*, Oxford Univ. Press, 1994 | K | Rank-1 lattice rules; exponential convergence for periodic analytic integrands. |
| J. Dick, F. Y. Kuo, I. H. Sloan, "High-dimensional integration: the quasi-Monte Carlo way", *Acta Numerica* 22:133–288, 2013 | W | QMC, lattice and CBC survey; the error theory for lattice and Sobol rules. |
| D. Nuyens, R. Cools, "Fast algorithms for component-by-component construction of rank-1 lattice rules in shift-invariant RKHS", *Math. Comp.* 75:903–920, 2006 | W | The CBC construction used in 3D. |
| J. Dick, D. Nuyens, F. Pillichshammer, "Lattice rules for nonperiodic smooth integrands", *Numer. Math.* 126:259–291, 2014 (arXiv:1211.3799) | W (arXiv) | Background for tent transforms. Explains why the tent transform hurts *our* boundary-vanishing integrand. |
| L. N. Trefethen, J. A. C. Weideman, "The exponentially convergent trapezoidal rule", *SIAM Rev.* 56(3):385–458, 2014 | K | **Key mechanism.** Our integrand vanishes on the boundary, so lattice and trapezoid rules behave as if it were periodic. |
| L. N. Trefethen, "Is Gauss quadrature better than Clenshaw–Curtis?", *SIAM Rev.* 50(1):67–87, 2008; and *Approximation Theory and Approximation Practice*, SIAM, 2013/2019 | W / K | Geometric convergence of Gauss quadrature for analytic integrands. |
| T. Gerstner, M. Griebel, "Numerical integration using sparse grids", *Numer. Algorithms* 18:209–232, 1998 | K | Smolyak cubature. |
| H.-J. Bungartz, M. Griebel, "Sparse grids", *Acta Numerica* 13:147–269, 2004 | K | Exactness only on mixed-order or hyperbolic-cross spaces, which is why tensor sines with $a,b$ up to ~25 defeat Smolyak. |
| E. Novak, K. Ritter, "Simple cubature formulas with high polynomial exactness", *Constr. Approx.* 15:499–522, 1999 | K (pages **?**) | Smolyak exactness classes. |

### 2e. VPINN and neural Petrov–Galerkin quadrature

| Citation | Verified | Relevance |
|---|---|---|
| E. Kharazmi, Z. Zhang, G. E. Karniadakis, "Variational PINNs", arXiv:1912.00873, 2019; and "hp-VPINNs", *CMAME* 374:113547, 2021 | W | NN trial space, polynomial or trigonometric tests and Gauss quadrature, at training time with a global network. |
| S. Berrone, C. Canuto, M. Pintore, "Variational physics informed neural networks: the role of quadratures and test functions", *J. Sci. Comput.* 92:100, 2022 | W | An a-priori analysis of how quadrature precision and test degree interact. It is the closest *theory* precedent: the best strategy is low-degree tests with high-precision quadrature, which supports our design. |
| T. Matsubara, T. Yaguchi, "Number theoretic accelerated learning of physics-informed neural networks", AAAI 39(1):595–603, 2025 | W | Good-lattice training for PINN collocation. **Lattice rules for NN residual integrals are already done** (for training losses). |
| J. Chen, R. Du, P. Li, L. Lyu, "Quasi-Monte Carlo sampling for solving PDEs by deep neural networks", *Numer. Math. Theor. Meth. Appl.* 14(2), 2021 | H | QMC for deep Ritz. |
| M. Ainsworth, J. Dong, "Galerkin neural networks", *SISC* 2021 (arXiv:2105.14094) | H | Neural Galerkin full-order solver with classical quadrature. |
| Y. Shang, F. Wang, J. Sun, "Randomized neural network with Petrov–Galerkin methods", *CNSNS* 127:107518, 2023 | H (**?** on volume) | Random-feature trial space, Legendre tests and Gauss–Legendre quadrature, as a full-order method. |

### 2f. 2024–2026 mesh-free or continuous hyper-reduction (the "is it already done?" sweep)

The sweep found **CECM (2024)**, **SNF-ROM (2025)**, **Weder et al. (2025)**, **Simplicits (2024)**, Liljegren-Sailer (2025), energy-conserving EQP (arXiv:2508.21279, **?** on authors), meshless SPH projection ROM (arXiv:2507.07830, **?** not read), and a differentiable-solver PI-ROM with INR decoder (arXiv:2505.14595, **?** not read). None replaces fitted hyper-reduction with classical deterministic rules of characterised convergence for a tested PG residual. Two of them are near misses. SNF-ROM supplies off-grid AD derivatives with uniform points. Simplicits supplies non-fitted cubature, but it is Monte Carlo. The sweep was English-only and limited by paywalls, so arXiv:2505.14595 and 2507.07830 must be read before submission.

---

## 3. Novelty assessment

### Already done, and by whom (must be credited, not claimed)
- **Decoding an INR at arbitrary points with exact autodiff gradients, inside an online reduced solve:** CROM 2023, Chen et al. MPM 2023, neural stress fields 2023, LiCROM 2023, SNF-ROM 2025, Weder et al. 2025.
- **"Online cost independent of the mesh":** claimed by CROM, SNF-ROM and Weder et al.
- **Fixed, non-fitted cubature for a neural-field subspace:** Simplicits 2024 (uniform MC) and Weder et al. (uniform or equidistant collocation).
- **AD derivatives of a smooth neural field avoiding the low-order FD stencil error:** argued explicitly in SNF-ROM.
- **Lattice or QMC points for NN residual integrals:** Matsubara–Yaguchi 2025; Chen et al. 2021.
- **Neural or random-feature trial space, fixed polynomial or trig tests, and Gauss quadrature:** VPINN, hp-VPINN, RNN-PG and Galerkin NNs (full-order or training-time).
- **Hyper-reduction points off the mesh:** CECM 2024, though it uses fitted, continuous point locations.
- **Precomputed quadratic tensors for polynomial nonlinearities:** standard in Galerkin POD, and the compressive-tensor idea in Carlberg 2011.

### Genuinely new, with reasonable confidence
1. **A characterised classical rule replacing data-fitted hyper-reduction in an LSPG NM-ROM.** The tested nonlinear term of a Petrov–Galerkin NM-ROM is evaluated with a deterministic, state-independent, mesh-independent rule (tensor Gauss, CBC rank-1 lattice). The rule's error is controlled by integrand regularity rather than certified statistically on held-out states, and there is no offline fit. No ROM paper found does this with weighted, convergence-characterised rules. The prior "fixed point" ROMs use unweighted uniform or MC points.
2. **Characterisation of rule families for tensor-product spectral tests.** The paper establishes that:
   - Gauss and lattice rules converge super-algebraically because the boundary-vanishing integrand is effectively periodic;
   - the tent transform hurts;
   - Smolyak fails structurally because of mixed high frequencies;
   - lattices beat Gauss in 3D at equal $m$;
   - Sobol converges as $1/m$ and MC is useless.

   Each fact is classical in isolation. The combination, demonstrated end to end in a ROM with must-fail controls, was not found anywhere.
3. **Quantifying the continuum-target effect in an NM-ROM.** With the stencil-tested ROM, error tracks the mesh's $O(h)$ upwind error (10.46 → 6.09 → 3.87 %). With the off-mesh rule it is mesh-invariant at 2.83 %, and it can beat the same-mesh FOM. The *mechanism* (AD derivatives avoid stencil error) is in SNF-ROM. The *measured decoupling of ROM error from the training mesh*, and the reframing of hyper-reduction certification against a continuum target rather than the mesh target, are new.
4. **Memory and data:** the advection storage falls from 4.3 GB ($M R'^2$ tensor) to 0.05–0.34 GB, with zero offline fitting. This is a secondary contribution.

### Claims that would get the paper rejected if overstated
- "First to evaluate a neural decoder off-mesh" or "first mesh-independent online cost". CROM, SNF-ROM and Weder et al. refute both.
- "First non-data-fitted hyper-reduction" or "a priori hyper-reduction". Ryckelynck 2005 owns the phrase, and Simplicits and Weder use non-fitted points.
- "First to use AD derivatives to avoid stencil error". SNF-ROM refutes this.
- "Lattice rules for neural PDE residuals are new". Matsubara–Yaguchi refutes this.
- "The ROM is more accurate than the FOM" stated generally. This holds only against a first-order reference, only at 64³ and 128³, and only because the bank was trained on 129-node data. At 256³ the FOM is more accurate.
- Any speed-up headline. The off-mesh ROM is slower than tuned Newton–BiCGStab at 64³ and 128³.
- "A-priori error bounds" unless the proposition is actually proved. Today the evidence is empirical ρ ladders.
- "Smolyak/sparse grids are bad for ROMs" in general. The failure is specific to tensor-product high-mixed-frequency tests.
- Generality claims beyond the box domain, smooth solutions, homogeneous Dirichlet conditions and sine tests.

---

## 4. Candidate titles, abstract skeleton, and claims we must not make

### Titles
1. *Classical quadrature in place of empirical hyper-reduction for neural-field reduced-order models*
2. *Off-mesh quadrature for Petrov–Galerkin reduced models on coordinate-network manifolds: lattice rules, Gauss rules, and the continuum target*
3. *Mesh-independent hyper-reduction of nonlinear-manifold ROMs by fixed lattice and Gauss quadrature*

Title 1 is the safest. Title 3 risks the "mesh-independent was already claimed" objection unless the abstract qualifies it at once.

### Abstract skeleton (≤200 words)
> Nonlinear-manifold reduced-order models (NM-ROMs) need hyper-reduction of their nonlinear terms. Standard approaches fit sample points and weights to snapshots on the full-order mesh, which requires certifying them on held-out states [*gap*]. We consider an NM-ROM whose trial manifold is a frozen coordinate-network bank $u(x)=G(x)c$, tested against $M$ tensor-product sine functions and advanced by Levenberg–Marquardt with backward Euler. Because the bank and its exact gradient can be evaluated anywhere, we replace data-fitted quadrature with fixed classical rules on the continuum, which requires no offline fit [*method*]. We compare tensor Gauss–Legendre, rank-1 (Fibonacci/CBC) lattice, Sobol, Smolyak, and Monte Carlo rules [*study*]. Gauss and lattice rules converge super-algebraically because the tested integrand is smooth and vanishes on the boundary. Lattices outperform Gauss in three dimensions. Smolyak grids fail because they discard the mixed high frequencies of tensor tests [*finding 1*]. Because the rule integrates the continuum term rather than a first-order mesh stencil, the reduced solution's error becomes independent of the training mesh: for 3D viscous Burgers it is X % at 64³–256³, against Y–Z % for an exact stencil tensor [*finding 2; numbers from JSON, against a reference still to be confirmed*]. The advection storage drops by an order of magnitude [*cost*]. We state the limits: smooth solutions, box domains, and online cost that is not below a tuned full-order solver on coarse meshes [*honesty*].

### Claims we must NOT make
1. "First off-mesh / mesh-free / continuous ROM". The firsts belong to CROM (2023) and SNF-ROM (2025).
2. "Mesh-independent online cost" as our contribution. We can call it a *consequence* and cite the prior claims. Also, the projection of $u_0$ and the output decoding remain $O(N)$.
3. "Speed-up over the FOM" in the abstract or title.
4. "More accurate than the full-order model" without the qualifier: against a first-order 513³ reference, on meshes coarser than the bank's training resolution.
5. "Error-certified" or "a-priori error bound" unless a theorem is in the paper.
6. "Replaces all hyper-reduction". Mesh-bound decoders (CAE, masked AE, POD with interpolation) lose 2–4 orders of magnitude (Hari §8), so the method needs an analytic decoder.
7. "Sparse grids fail" without the qualifier "for tensor-product high-frequency tests".
8. "Removes the $O(h)$ error" as an established fact before it is checked against a high-order or exact reference, and before the 2.83 % has been decomposed into its error sources.
9. Any number typed by hand. Every number must come from the run JSONs (project rule).

### Pre-submission checklist from this agent
- Re-verify the GfA items in a browser: page limits, highlights, data-statement wording, code-link field.
- Read arXiv:2505.14595, 2507.07830, 2508.21279 and the LiCROM cubature section.
- Fix the Grimberg et al. authorship.
- Obtain a high-order or exact reference for the continuum-target claim.
- Add a faithful ECSW or EQP baseline.
- Rewrite the cost section around storage and offline cost rather than wall-clock wins.

### Sources fetched or searched
[JCP journal page](https://www.sciencedirect.com/journal/journal-of-computational-physics) · [JCP Guide for Authors (403; via search extracts)](https://www.sciencedirect.com/journal/journal-of-computational-physics/publish/guide-for-authors) · [JCP editorial board](https://www.sciencedirect.com/journal/journal-of-computational-physics/about/editorial-board) · [SNF-ROM arXiv](https://arxiv.org/abs/2405.14890) / [JCP](https://www.sciencedirect.com/science/article/pii/S0021999125002402) · [Weder et al.](https://www.sciencedirect.com/science/article/abs/pii/S0021999125005327) · [CECM](https://www.sciencedirect.com/science/article/pii/S004578252300676X) · [Simplicits](https://dl.acm.org/doi/10.1145/3658184) · [LiCROM](https://arxiv.org/pdf/2310.15907) · [Neural stress fields](https://arxiv.org/pdf/2310.17790) · [Romor et al. 2023](https://link.springer.com/article/10.1007/s10915-023-02128-2) · [Romor et al. 2025](https://arxiv.org/abs/2308.03396) · [Diaz et al.](https://www.sciencedirect.com/science/article/abs/pii/S0045782524001993) · [Grimberg et al.](https://arxiv.org/pdf/2008.02891) · [Kim et al. 2022](https://www.osti.gov/pages/servlets/purl/1843130) · [Berrone et al.](https://link.springer.com/article/10.1007/s10915-022-01950-4) · [hp-VPINNs](https://www.osti.gov/pages/biblio/1778822) · [Matsubara–Yaguchi](https://ojs.aaai.org/index.php/AAAI/article/view/32040) · [Yano–Patera](http://arrow.utias.utoronto.ca/~myano/papers/yp_2018_rb_eqp.pdf) · [Trefethen 2008](https://people.maths.ox.ac.uk/trefethen/publication/PDF/2008_128.pdf) · [Dick–Kuo–Sloan](https://www.cambridge.org/core/journals/acta-numerica/article/abs/highdimensional-integration-the-quasimonte-carlo-way/03F126DDF465F915B22D5D709CD28946) · [Liljegren-Sailer](https://arxiv.org/abs/2512.14416) · [CROM (ICLR 2023 list)](https://sites.google.com/view/neural-fields/accepted-papers)
