# Literature search: quadrature-based hyper-reduction for coordinate-network NM-ROMs

Compiled 2026-09-22 by a web-search agent (~45 searches over arXiv, Springer, ScienceDirect,
OpenReview, ACM; full-text reads of CROM, Kim–Lee–Wen–Choi 2024, Bruna–Peherstorfer–Vanden-Eijnden,
CoLoRA, Romor–Stabile–Rozza, Cocola et al., Weder–Schwerdtner–Peherstorfer). Items read only as
abstracts are flagged in Section 4. Verify specifics before quoting.

## 1. Closest prior work, by theme

### Theme 1: Hyper-reduction for nonlinear-manifold / autoencoder ROMs

1. Lee & Carlberg, "Model reduction of dynamical systems on nonlinear manifolds using deep
   convolutional autoencoders", JCP 404:108973, 2020.
   https://www.sciencedirect.com/science/article/pii/S0021999119306783
   Manifold Galerkin / LSPG with a conv decoder; the decoder must be evaluated on the whole grid,
   which is the bottleneck later works attack. Mesh-bound decoder, no off-mesh evaluation.
2. Kim, Choi, Widemann, Zohdi, "A fast and accurate physics-informed neural network reduced order
   model with shallow masked autoencoder", JCP 451:110841, 2022. arXiv:2009.11990 (precursor
   arXiv:2011.07727). Shallow decoder with a sparsity mask mirroring the FD stencil so a subset of
   output nodes needs a subset of hidden units; hyper-reduction is DEIM/gappy-style sampling on
   mesh nodes. Origin of "sparse/masked decoder" hyper-reduction. Sample points are mesh nodes
   selected by DEIM-type fitting.
3. Romor, Stabile, Rozza, "Non-linear manifold ROMs with convolutional autoencoders and reduced
   over-collocation method", J. Sci. Comput. 94:74, 2023. arXiv:2203.00360. LSPG solved only at
   greedily chosen "magic points" plus stencil submesh; a compressed decoder is trained
   teacher–student to output only those nodes. Mesh DOFs chosen by greedy fitting.
4. Cocola, Tencer, Rizzi, Parish, Blonigan, "Hyper-reduced autoencoders for efficient and accurate
   nonlinear model reductions", arXiv:2303.09630, 2023. Hyper-decoder trained on snapshots
   restricted to a Q-DEIM sample/stencil mesh; collocation LSPG + gappy POD. Mesh-node sampling.
5. Barnett & Farhat, "Quadratic approximation manifold for mitigating the Kolmogorov barrier in
   nonlinear projection-based MOR", JCP 464:111348, 2022; Barnett, Farhat, Maday,
   "Neural-network-augmented projection-based MOR", JCP 2023. Quadratic / NN-augmented manifolds
   with ECSW (NNLS-fitted mesh weights).
6. Fulton, Modi, Duvenaud, Levin, Jacobson, "Latent-space dynamics for reduced deformable
   simulation", CGF 38, 2019. Autoencoder subspace + An–Kim–James cubature (discrete fit on tets).
7. Diaz, Choi, Heinkenschloss, "A fast and accurate domain-decomposition nonlinear manifold ROM",
   arXiv:2305.15163 / CMAME 2024; "Scalable NM-ROM for dynamical systems", arXiv:2412.00507.
   DD-NM-ROM with sparse shallow decoders and mesh-node hyper-reduction. Not read in full.
8. Weder, Schwerdtner, Peherstorfer, "Nonlinear model reduction with Neural Galerkin schemes on
   quadratic manifolds", JCP 539, 2025. arXiv:2412.17695. Near miss: they state that the FOM grid
   points and the collocation points "can be different ... a form of hyper-reduction baked in" and
   that online costs "scale independently of the state dimension". But the decoder is a
   spline-interpolated quadratic map of the discrete state, the collocation points are equidistant
   or uniformly sampled, and no quadrature weights, test functions or quadrature theory are used
   (plain least-squares collocation).

### Theme 2: Neural-field / INR decoders for ROMs and PDE solvers

9. Chen, Xiang, Cho, Chang, Pershing, Maia, Chiaramonte, Carlberg, Grinspun, "CROM: Continuous
   reduced-order modeling of PDEs using implicit neural representations", ICLR 2023.
   arXiv:2206.02607, https://openreview.net/forum?id=FUORz1tG8Og. CLOSEST PRIOR WORK. From the
   full text: the online scheme works at "integration samples" M that "need not coincide with the
   ... full-order finite element discretization samples"; gradients come from differentiating the
   network. Step 2 is explicit strong-form collocation time-stepping at each sample; step 3 is an
   unweighted nonlinear least-squares "network inversion" solved by Gauss–Newton. No test
   functions, no weak form, no quadrature weights. Sample selection (Sec. 4.4, App. B): uniform
   sampling is reported to fail; a greedy residual-driven algorithm inspired by An et al. (2008)
   selects the fewest samples from the discrete full-order samples, so the points are
   training-mesh nodes chosen by data-dependent fitting. They do state the mesh-independence claim
   ("requires only |M| samples"). No Gauss/CC/QMC/lattice/sparse-grid rules, no a-priori error.
10. Chen, Chiaramonte, Grinspun, Carlberg, "Model reduction for the material point method via an
    implicit neural representation of the deformation map", JCP 478:111908, 2023. arXiv:2109.12390.
    INR deformation map with analytic gradients; forces integrated on a particle subset.
11. Zong et al., "Neural stress fields for reduced-order elastoplasticity and fracture", SIGGRAPH
    Asia 2023. arXiv:2310.17790. INR stress evaluated "at arbitrary spatial locations"; sampled
    particles; no quadrature theory.
12. Kim, Lee, Wen, Choi, "Physics-informed reduced order model with conditional neural fields"
    (CNF-ROM), arXiv:2412.05233, 2024; Wen, Lee, Choi, arXiv:2311.16410, 2023. Coordinate-network
    decoder + parametric neural ODE; the PDE residual is only a training / fine-tuning loss at
    collocation points; online is a latent ODE rollout, no projection, no quadrature.
13. Berman & Peherstorfer, "CoLoRA", ICML 2024. arXiv:2402.14646. Neural field with low-rank
    time-adapted weights; equation-driven variant = Neural Galerkin least squares at sample points,
    no quadrature weights or classical rule.
14. Bruna, Peherstorfer, Vanden-Eijnden, "Neural Galerkin schemes with active learning for
    high-dimensional evolution equations", JCP 496:112588, 2024. arXiv:2203.01360. Gram and
    right-hand-side integrals estimated by Monte Carlo with adaptive measures; they explicitly
    note that in low dimensions "this calculation can be performed by quadrature on a grid" and
    set classical quadrature aside for high dimension. Survey: Berman, Schwerdtner, Peherstorfer,
    Handbook of Numerical Analysis 25, 2024. Also Wen, Vanden-Eijnden, Peherstorfer,
    arXiv:2306.15630.
15. Pan, Brunton, Kutz, "Neural Implicit Flow", JMLR 24, 2023 (arXiv:2204.03216); Yin et al.,
    "DINo", ICLR 2023 (arXiv:2209.14855); Serrano et al., "CORAL", NeurIPS 2023 (arXiv:2306.07266).
    Data-driven latent dynamics with coordinate-network decoders; no online residual.
16. Cho, Lee, Park, Rim, Welper, arXiv:2510.25123, 2025. Low-rank neural representations of
    hyperbolic waves; data-driven.

### Theme 3: QMC / sparse grids / lattices for NN-parametrised PDE residuals

17. Chen, Du, Li, Lyu, "Quasi-Monte Carlo sampling for solving PDEs by deep neural networks",
    NMTMA 14(2), 2021. arXiv:1911.01612. Deep Ritz with QMC point sets; convergence at the QMC
    rate. Training-loss integration of a global NN.
18. Longo, Mishra, Rusch, Schwab, "Higher-order quasi-Monte Carlo training of deep neural
    networks", SISC 43(6), 2021. arXiv:2009.02713. Parametric-input integration.
19. Matsubara & Yaguchi, "Number theoretic accelerated learning of physics-informed neural
    networks", AAAI 2025. arXiv:2307.13869. "Good lattice training" (rank-1 lattice points) with
    periodization tricks for PINN collocation. Closest to the Fibonacci-lattice + tent findings,
    but for PINN training, no ROM, no test functions.
20. "A novel number-theoretic sampling method for neural network solutions of PDEs", Neural
    Networks 2025. arXiv:2411.17039 (authors not verified).
21. Caradot et al., "Provably accurate adaptive sampling for collocation points in PINNs",
    ECML-PKDD 2025. arXiv:2504.00910.
22. Wu & Sun, "Enhancing the accuracy of PINN surrogates in flash calculations using sparse grid
    guidance", Fluid Phase Equilibria 2023. Only sparse-grid + PINN item found; not Galerkin.
23. Quadrature theory to cite: Sloan & Joe, Lattice Methods for Multiple Integration, OUP 1994;
    Dick, Nuyens, Pillichshammer, "Lattice rules for nonperiodic smooth integrands", Numer. Math.
    126, 2014 (arXiv:1211.3799); Cools, Kuo, Nuyens, Suryanarayana, "Tent-transformed lattice
    rules ...", J. Complexity 2016; Hickernell, "Obtaining O(N^{-2+eps}) convergence for lattice
    quadrature rules", MCQMC 2002; Hinrichs & Oettershagen (authorship unverified),
    arXiv:1409.5894 (Fibonacci optimality for bivariate periodic classes); Cools et al.,
    "Integration and approximation with Fibonacci lattice points" (DRNA); Trefethen & Weideman,
    "The exponentially convergent trapezoidal rule", SIAM Review 56, 2014; Gerstner & Griebel,
    Numer. Algorithms 18, 1998; Bungartz & Griebel, "Sparse grids", Acta Numerica 13, 2004;
    Novak & Ritter, Constr. Approx. 15, 1999 (Smolyak exactness is total-degree-like, so products
    of high-frequency sines in each coordinate are exactly the class where Smolyak degrades).
    No paper was found stating, in a Galerkin/ROM context, that tensor-product test functions
    defeat sparse grids.

### Theme 4: RB/ROM hyper-reduction via quadrature

24. An, Kim, James, "Optimizing cubature for efficient integration of subspace deformations",
    ACM TOG 27(5), 2008. NNLS-fitted cubature; template for ECSW/ECM/EQP and CROM's sampler.
25. Farhat, Avery, Chapman, Cortial, IJNME 98, 2014; Farhat, Chapman, Avery, IJNME 102, 2015
    (ECSW); Grimberg, Farhat, Youkilis, Tezaur, IJNME 2021 (arXiv:2008.02891, ECSW for LSPG).
26. Hernández, Caicedo, Ferrer, "Dimensional hyper-reduction of nonlinear FE models via empirical
    cubature", CMAME 313, 2017.
27. Hernández, Bravo, Ares de Parga, "CECM: A continuous empirical cubature method", CMAME 418,
    2024. arXiv:2308.03877. Point coordinates become continuous design variables, but still
    fitted to training snapshots and needing integrand values at FE Gauss points.
28. Yano & Patera, "An LP empirical quadrature procedure for reduced basis treatment of
    parametrized nonlinear PDEs", CMAME 344, 2019; Yano, ACOM 45, 2019. Follow-ups:
    arXiv:2508.21279 (energy-conserving EQP), arXiv:2512.14416.
29. Carlberg, Farhat, Cortial, Amsallem, GNAT, JCP 242, 2013; Chaturantabut & Sorensen, DEIM,
    SISC 32, 2010; Drmač & Gugercin, Q-DEIM, SISC 38, 2016; Lauzon et al., S-OPT, SISC 46, 2024;
    Ryckelynck, "A priori hyperreduction method", JCP 202, 2005.
30. "Element boundary terms in ROMs for flow problems: domain decomposition and adaptive coarse
    mesh hyper-reduction", CMAME 2020. Coarse-mesh integration, still mesh-based and adapted.
    Finding: no RB/ROM paper integrates the projected nonlinear term with a state-independent
    classical rule on the continuum using a smooth learned decoder; all quadrature-based
    hyper-reduction in the ROM literature is empirical (fitted to snapshots) and mesh-bound.

### Theme 5: Petrov–Galerkin with fixed spectral tests + quadrature and neural / random-feature trial spaces

31. Kharazmi, Zhang, Karniadakis, VPINNs, arXiv:1912.00873, 2019; hp-VPINNs, CMAME 374, 2021.
    NN trial space, Legendre (and trigonometric) test functions, Gauss quadrature; global NN,
    training time only.
32. Berrone, Canuto, Pintore, "VPINNs: the role of quadratures and test functions", J. Sci.
    Comput. 92, 2022. arXiv:2109.02035. A-priori analysis; recommends low-degree test functions
    with high-precision quadrature, which supports the low-frequency-test + high-order-rule design.
33. Shang, Wang, Sun, "Deep Petrov–Galerkin method", arXiv:2201.12995, 2022; "Randomized neural
    network with Petrov–Galerkin methods", CNSNS 127:107518, 2023. Random-feature trial space +
    FE/Legendre tests + Gauss–Legendre quadrature; full-order solver, no latent manifold.
34. Chen, Chi, E, Yang, random feature method, JML 2022 (arXiv:2207.13380, collocation); Zhang,
    Bao, Ju, Zhang, TransNet, arXiv:2301.11701; Xu, Wang, Wang, Weak TransNet, J. Sci. Comput. 2026
    (arXiv:2506.14812); Evo-GTransNet, arXiv:2608.19615, 2026. Fixed-feature Galerkin with
    quadrature; none is a ROM with a latent nonlinear head.
35. Ainsworth & Dong, "Galerkin neural networks", SISC 2021 (arXiv:2105.14094); Meuris, Qadeer,
    Stinis, Sci. Rep. 13:1739, 2023 (arXiv:2111.05307); Marcondes, arXiv:2605.03542, 2026 (random
    test functions).
36. Odd-DC (arXiv:2511.18241), PTPI-DL-ROMs (arXiv:2405.08558): neural-manifold ROMs with physics
    losses; not read.

## 2. Novelty assessment

Not novel (phrase carefully):
- Evaluating an INR / coordinate-network decoder and its analytic derivatives at sparse points
  that are not mesh nodes is established: CROM (2023), Zong et al. (2023), Weder et al. (2025),
  CNF-ROM / CoLoRA.
- The "online cost independent of N" argument has been made by CROM and Weder et al.
- QMC / rank-1 lattice points with periodization for NN residual integrals exist for PINN and
  deep-Ritz training (Chen et al. 2021; Matsubara & Yaguchi 2025; arXiv:2411.17039).
- Fixed spectral or polynomial test functions + Gauss quadrature with a neural or random-feature
  trial space exists as a full-order solver (VPINN, DPGM, RNN-PG, Weak TransNet).

Appears novel (no prior instance found):
1. Replacing data-fitted hyper-reduction (ECSW / EQP / ECM / greedy cubature / DEIM, including
   CROM's residual-driven greedy sampler on training-mesh nodes) with a fixed, state-independent,
   mesh-independent classical quadrature rule on the continuum (tensor Gauss–Legendre, rank-1
   Fibonacci lattices) for the tested nonlinear term of an LSPG NM-ROM, so that the
   hyper-reduction error is an a-priori property of a smooth integrand rather than something
   certified statistically on held-out states. CROM's online step is unweighted collocation plus
   network inversion, not a weighted quadrature of a PG residual.
2. The architecture combination: neural-field spatial bank × small latent head + nested linear
   corrections, LSPG onto a fixed low-frequency sine test space with all linear terms
   pre-assembled exactly, and quadrature only for the tested nonlinear term. No NM-ROM with fixed
   spectral test functions was found.
3. The finding that Smolyak sparse grids fail for tensor-product sine test functions because of
   mixed high frequencies, whereas lattice rules and tensor Gauss converge super-algebraically and
   Sobol/Halton give only about 1/m. Consistent with classical theory but not stated in a
   Galerkin/ROM context.
4. The integrated-by-parts (flux) form to avoid the decoder's gradient inside the hyper-reduced
   integral in this setting.

Honest framing: "partial decoding" as a capability is CROM's; the contribution is turning it into
classical quadrature with a-priori error control replacing empirical quadrature, and
characterising which rules work for spectral test functions.

## 3. Suggested citations for the write-up

Core comparators: CROM [9]; Weder–Schwerdtner–Peherstorfer [8]; Kim et al. 2022 [2]; Lee &
Carlberg [1]; Romor et al. [3]; Cocola et al. [4]; Barnett & Farhat [5]; CNF-ROM [12]; CoLoRA
[13]; Neural Galerkin [14]. Neural-field lineage: [10, 11, 15]. Hyper-reduction lineage: [24–29].
QMC/PINN: [17–20]. Quadrature theory: [23]. Spectral PG with neural trial spaces: [31–34].

## 4. Caveats

- Read in full text: CROM, CNF-ROM, Bruna et al., CoLoRA, Romor et al., Cocola et al.; Weder et
  al. via arXiv HTML. Others are based on abstracts or search snippets.
- Authorship of arXiv:2411.17039 and arXiv:1409.5894 not verified.
- No "Chen & Ghattas" hyper-reduction-for-NM-ROM paper surfaced; nothing found on Sobol/Halton
  O(1/m) specifically for NN integrands (the rate is standard QMC theory).
- No paper using the phrase "partial decoding" was found.
- English-language web results only; paywalls prevented reading some Springer/Elsevier texts.
