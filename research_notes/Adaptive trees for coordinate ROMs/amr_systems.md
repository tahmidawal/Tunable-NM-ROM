# AMR systems: complete researcher notes

Source: Astra AMR-systems researcher, 2026-10-26. Read-only research; no code, runs, installs or files changed. Notes retained as research evidence; quantitative repository statements below are source notes, not newly generated results.

## Part 1 of 3

Recommendation: the best first practical experiment is an ordered coordinate-bank warm start for repeated adaptive heterogeneous diffusion solves, followed by a conventional validated FOM. Local bank/learned coarse correction should be later arms contingent on measurements. This is a plausible application, not evidence of superiority over POD. Continuous POD can equally operate on AMR meshes, supply initial guesses and coarse spaces; merely adding a tree does not overcome current disadvantage.

SOURCE/REPOSITORY FINDINGS

1. Repository final report /home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-21-improvement-campaign/final/report.md (line 32 overview; 156–166 multi-mesh): continuous POD matched bank on all investigated multi-mesh workloads at lower cost; irregular geometry/tested variable coefficient did not establish bank advantage; steep-state failure remained time-step certification. Do not transfer older same-grid speed claims.
2. Hou, Chen, Xia, A reduced basis warm-start iterative solver for parameterized linear systems, https://arxiv.org/html/2311.13862v3 (§3.2/4.2). RB initial guess then PCG, including MG-PCG, reaches high-fidelity solution. A solution-based multispace RB preconditioner loses efficiency once iterative accuracy exceeds RB approximation accuracy. Distinguish good solution approximation from good repeated error correction. Paper uses reduced over-collocation: NOT recommendation to import its collocation recipe here. Their harder example has much worse amortization than simple one. Avoid claiming numbers transfer.
3. Nguyen/Chen, RB iterative solution of parameterized SPD systems, https://arxiv.org/pdf/1804.06363. RB iterations and RB-preconditioned CG target truth-discretization accuracy beyond reduced approximation. Prior art for hybrid role, not speed evidence for our bank/AMR.
4. Abdulle/Henning RB-LOD https://arxiv.org/abs/1410.3253. Parameterized multiscale elliptic problems; local basis from decoupled local problems, offline representative parameters, sparse online systems. Heterogeneous diffusion already has sophisticated classical localized-ROM competition.
5. Smetana/Taddei https://arxiv.org/html/2202.09872v1. PUM global assembly, global Galerkin, local residual enrichment; analyzed exponential convergence is in-sample linear-coercive setting, not generic nonlinear certificate. §3.3 explicitly says residual assembly over all components is expensive and specialized hyper-reduction is not addressed; paper supports architecture, not completed fast online implementation. §3.3 sparse Jacobian uses bounded patch-neighbor count.

RANKING/INFERENCES

1. Bank warm start → tuned AMR FOM: closest to existing solution-trained bank; FOM corrects representation error; easy to disable. Killed if predict/decode/setup exceeds iterations saved or previous/POD starts cheaper.
2. Direct global-bank ROM + adaptive integration: smallest method change; attractive for sparse requested output/qualified approximate answer. Killed by cheaper continuous POD, mesh-scale coefficient integration, full-output cost.
3. Adaptive local corrections vs adaptive local POD: active patches can avoid globally paying for detail. Fixed local space has no training-projection advantage over equal-dimension weighted global POD. Possible benefits are union of state-dependent spaces, sparse evaluation, generalization/component reuse. Disclose full dictionary size and active-set selection cost.
4. Bank-enriched coarse/preconditioner: may remove difficult global error modes; risks solution basis failing iterative-error manifold, dense restriction/prolongation, AMG/recycled Krylov doing same cheaper.
5. Independent local learned/domain-decomposition solvers: component reuse possible; interface/global coupling remains; local POD offers same structure.

Last: moving steep Burgers then NS. Tree does not repair time-step certification; NS adds pressure/divergence/transfer obligations.

FIRST WORKLOAD PROPOSAL

Repeated quasi-steady heterogeneous conduction/scalar diffusion $-\nabla\cdot(a(x;\mu)\nabla u)=f(x;\mu)$, $u=g$ on boundary; localized sources, few inclusions/channels; parameter variations move/change only part of difficult structure. 2D fixed geometry first; vary source and material separately then jointly. Motivating feature is repeated local structure with expensive best FOM, not irregular geometry/high contrast alone. Prior variable-coefficient negative result requires new strongest-FOM cost gate first.

Fixed operator/many RHS: mandatory reused sparse factorization, reused AMG setup, block solves, Krylov recycling controls; may be bad neural opportunity. Changing operator diminishes reuse but ROM also pays reduced-operator update/certification. Local forcing does not imply localized elliptic response; keep global coarse coupling. Linear diffusion plus linear prefix is coordinate-bank acceleration, not evidence for nonlinear head.

No reviewed source/result proves current bank beats continuous/local POD on AMR, warm starts, preconditioning or DD. No engineering-time estimate before compiler/backend/interface profiling.

## Part 2 of 3: backend and integration

TAKEAWAY: MFEM nonconforming FE is best first elliptic correctness/FOM backend. Keep bank evaluation/reduced kernels in JAX via explicit interface. AMReX if targeting conservative FV evolution. p4est only when custom distributed forest needed; not itself PDE discretization.

VERIFIED OFFICIAL SOURCES

6. p4est forest workflow https://p4est.github.io/api/p4est-latest/forest.html: distributed topology, refinement/coarsening/partitioning; mesh changes distinct from partition changes. Does not supply app diffusion stencil, BCs, estimator, solver.
7. MFEM https://mfem.org/howto/ncmesh/: nonconforming quad/hex refinement, derefinement, load balancing. Hanging constraints via P; true system $P^TAPx=P^Tb$, recovery Px. Optional nc_limit controls level difference; default not bounded.
8. MFEM examples https://mfem.org/examples/: ex6 Poisson AMR supports partial assembly/GPU; ex15 refinement/derefinement/transfer/rebalance is prescribed sequence, NOT validated time integration. Ex6 ZZ estimator is demonstration, not automatic proof for heterogeneous new PDE.
9. AMReX https://amrex-codes.github.io/amrex/docs_html/AmrCore.html: block-structured AMR; app implements tagging/create/remake. Flux registers accumulate coarse/fine area-time-weighted mismatch; reflux corrects adjacent coarse cells.
10. AMReX https://amrex-codes.github.io/amrex/docs_html/LinearSolvers.html: MLMG geometric multigrid incl multi-level composite operators, cell-centered variable coefficient/nodal variants, hypre/PETSc, embedded boundaries. Credible baselines. Rolling docs showed development version; pin/test actual release.
11. DG reduced-basis elements https://www.numdam.org/articles/10.1051/m2an/2015045/: local independent bases weakly coupled for flux/solution continuity; global coarse FE space in two-level preconditioner. Local models do not remove interfaces.
12. JAX https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html: transformed/compiled intermediate/output shapes static; changed argument shape can retrace. https://docs.jax.dev/en/latest/jax.experimental.sparse.html explicitly calls experimental sparse reference implementation, not recommended performance-critical. (13 URLs total including both JAX pages, roughly 12 relevant sources.)

IMPLEMENTATION PROPOSAL, NOT DEMONSTRATED

Separate (i) FOM reference/final-correction mesh, (ii) weak integration partition, (iii) optional active local trial hierarchy. May share index; refinement of one need not refine others. Continuous global bank needs no tree decoder just because AMR observations.

Modules:

- mesh_adapter: leaf bounds/topology/maps/BC labels/true constraints/volumes/stable IDs.
- field_transfer: constrained FE projection/interpolation or conservative cell averages; boundary lifts; refine/coarsen/history transfer.
- weak_assembly: physical quadrature, values/gradients/tests, mass/stiffness/load, coefficient integration.
- bank_catalog: global bank, later windows/local ranks/support graph.
- adapt_controller: separate integration/representation/time/FOM-estimator decisions.
- hybrid_solver: predict, true FOM residual check, conventional correction/fallback.
- audit_metrics: independent integration, conservation/flux, same-invocation cost+accuracy, setup/compile/transfer.

Current reuse: /home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-geometry/experiments/imp-geometry/bank.py has weighted QR/SVD, AD physical gradients, general M/K assembly. bank_space starting point, not tree support. Train assumes common lattice; ragged AMR data needs weighted interface.

MATH/CONVENTIONS

For $x=T_K(\widehat x)$, integrate $\sum_q\widehat w_q|\det J_K|F(T_K(\widehat x_q))$; $\nabla_x b(T_K^{-1}x)=J_K^{-T}\nabla_{\widehat x}b$. Window $\psi_K=\chi_Kb\circ T_K^{-1}$ has $\nabla\psi_K=(\nabla\chi_K)b+\chi_KJ_K^{-T}\nabla b$; dropping window term wrong. Rescaling coordinates changes physical relative diffusion/advection/reaction scales; does not establish shared-bank transfer across levels.

FV snapshots: $\overline u_{\theta,K}=|K|^{-1}\int_Ku_\theta$; train physical weighted loss $\sum_K|K||\overline u_{\theta,K}-\overline u_K|^2/\sum_K|K||\overline u_K|^2$. FE data integrate FE and decoder consistently. Uniform leaf sampling overweights refined regions unless corrected; equal-depth averages not physical L2.

Two different targets: continuous weak-form ROM integrates continuous field directly (no hanging-node decoder constraint if conforming continuous global trial); ROM inside FE maps prediction into constrained FE space then uses exact FOM operator. Independently evaluating network at all mesh nodes may violate hanging interpolation. Project/sample true DOFs then apply P (or constrained projection); FE interpolation differs from direct network values between nodes. Preserve target distinction/provenance.

FV: one shared oriented flux per fine subface, coarse contribution=sum fine; time-integrate matching substeps; exclude covered coarse cells in domain integrals. Reflux fixes mismatch, NOT inconsistent flux accuracy. Center-to-center two-point flux at T junction can miss tangential offset; use validated backend reconstruction and linear patch/manufactured/flux-jump tests.

Historical /home/tahmid/Dev/Octree-Refine-Net/QuadtreeAE-NMROM/final/PROGRESS.md documents T-junction inaccuracy and regularization-dependent fine-only masking. Fine-leaf-only residual not whole-domain PDE. Reuse topology/data lessons, not 'better than FOM' claim.

JAX PLAN

Tree mutation/connectivity outside compiled inner solve. Flat arrays bucketed by level/type/quadrature/rank, modest padded capacities/masks; freeze topology/basis/rule during solve. Cache by geometry/bank/rule ID, invalidate affected blocks. Measure padding/memory vs recompiles. No assumed sparse-array or tiny-patch GPU speed. Local support only gives sparse reduced coupling with local tests/assembly; global tests or dense QR/SVD destroys it. Keep global coarse modes separately coupled, local ordering/overlap dependencies.

NS DERIVED REQUIREMENTS

Localize potential before curl: $u_K=\nabla\times(\chi_KA_K)$. Windowing already divergence-free u gives $\nabla\cdot(\chi u)=\nabla\chi\cdot u$, generally nonzero. Pointwise div-free not automatically discrete flux-free after AMR interpolation. General map needs appropriate vector transformation, not scalar coordinate scaling. Pressure global; maintain gauge, compatible velocity-pressure spaces/constraints, BCs and divergence under mesh/history transfer. Smooth div-free velocity not necessarily wall-compatible. NS after scalar validated prototype. No NS-AMR implementation demonstrated; GPU array exchange/backend availability untested.

## Part 3 of 3: decisive gates, cost and no-go

All proposed, no experiments. Start approximation/cost diagnostics BEFORE neural retraining. Preserve canonical lab-log rules: best matched-accuracy FOM, same allocation/device comparisons, f64/highest, repetitions, iteration/outlier reporting, qualified time integration.

GATE A CORRECTNESS ONLY

Manufactured smooth diffusion uniform + fixed graded quadtrees, then piecewise coefficient with interface-compatible reference. Check volumes sum/domain no double-count covered cells; constant/linear reproduction and physical derivative scaling; FE constraints/FV flux consistency; convergence with refinement; unchanged-bank integration converges independently. Parent retained plus zero child coefficients preserves field; quantify coarsening/history projection physical error. No speed/application claims yet.

GATE B ROOM TO WIN

Small accurate AMR dataset + strongest FOM profile. Compute global weighted POD, fixed local POD at matched total dimension, oracle adaptive local POD, inherited/global-bank floors where meaningful. Oracle sees truth ONLY approximation diagnostic, never deployable/timing comparator. Full dictionary footprint. Stop if oracle localization offers little, most patches active, or FOM cheaper than prediction/decode lower bound. Fixed local linear basis cannot beat weighted global POD optimum training MSE at same total dimension; union/activation/sparsity/generalization are hypotheses.

GATE C CHEAPEST HYBRID

Identical physical cases/FOM paths: tuned zero-start FOM; previous/nearest-parameter solution start; reused factorization/AMG/Krylov; global continuous-POD start; local/adaptive-POD start; coordinate-bank start. Same final FOM algebraic + discretization acceptance. Freeze mesh paths first to isolate solve; then end-to-end actual adaptation/estimator cost, no oracle mesh gift. Enforce BC/constraints, measure true FOM residual, reject nonfinite/unhelpful guess; FOM finishes beyond bank floor. Saved iterations alone insufficient.

GATE D DIRECT/LOCAL

Compare global bank uniform reference integration; same bank tree quadrature; global+local corrections; continuous global/local POD with equal integration/adaptivity/integrator/qualification options. Separate FOM discretization, representation, quadrature, algebraic/time errors. Active-set selector vs oracle: selection cost, false negatives, fallback. Include local polynomial/hierarchical FE corrections non-neural control.

GATE E COARSE CORRECTION CONDITIONAL

SPD A, constrained bank matrix Z: $E=Z^TAZ$, $Q=ZE^{-1}Z^T$. Q low rank, NOT standalone full-rank preconditioner. Proposed symmetric candidate with SPD standard $M_0^{-1}$:

$$M^{-1}=Q+(I-QA)M_0^{-1}(I-AQ).$$

This is our design candidate, not algorithm copied from either cited paper. Verify symmetry/definiteness/solver; changing basis/rank/inner solve during Krylov needs compatible flexible method rather than assume ordinary PCG. Same arm with POD/local POD/recycled Krylov. Test iterative error/correction snapshots because solution-trained bank need not cover late residuals.

GATE F TRANSIENT THEN 3D

Moving local scalar transient + independently verified adaptive time integrator; charge basis/history/regrid. Steep Burgers only if time qualification passes. 3D after real 2D end-to-end advantage survives POD/FOM. Dimensionality not proof.

COST MODEL

$$T_{\rm query}=T_{\rm input}+T_{\rm tree}+T_{\rm transfer}+T_{\rm assembly}+T_{\rm reduced}+T_{\rm checks}+T_{\rm output}+T_{\rm fallback}.$$

$$T_{\rm warm}=T_{\rm predict}+T_{\rm decode/project}+T_{\rm residualcheck}+T_{\rm remainingFOM}.$$

Positive only if overhead < FOM work saved. Dense Z restriction/prolongation ~$nr$/application; $Z^TAZ$ updates may cost more. Count setup and repeated correction; fewer iterations not necessarily faster. Sparse local overlap reduces blocks but global solve/coarse coupling remains; independent batched patch solves are smoothers/local corrections, not complete elliptic solve.

$$C_{\rm total}=C_{\rm offline}+\sum_{j=1}^{Q_{\rm workload}}T_j.$$

$$Q_{\rm break}=\frac{C_{\rm offline,bank}-C_{\rm offline,baseline}}{\operatorname{mean}T_{\rm baseline}-\operatorname{mean}T_{\rm bank}},$$

only positive denominator and sufficiently stationary workload. Also distributions/median/tails/failure/fallback sensitivity, not just mean.

FAIRNESS

Physical accuracy + same requested output, not equal leaves/rank. Full-field and sparse outputs only actual intended workloads, all comparators same output. Charge arbitrary sampled coefficient/forcing read. Include data/train/order/rule/compile/cache/regrid/transfer/check/fallback. Separate cold setup/amortized reuse. Burn in/interleave one device/allocation/save repetitions. Tune AMR FOM polynomial degree, mesh tol, algebraic tol, integrator, preconditioner with declared budget. Held-out material/source combos then separate distribution shift; outliers/fallback counts.

NO-GO

Stop application if continuous/local POD equal accuracy cheaper; saved FOM work doesn't pay predict/decode; amortization beyond realistic queries; selector misses local errors or full scan erases savings; active rank tracks leaves; local coupling dense/ill-conditioned; quadrature successor check substituted for unproved physical certificate; refinement fails convergent accuracy; fine-cell residual masking needed; transient only passes relaxed time criteria; speed requires overtuned tolerance/undertuned comparator/different hardware.

Unknown deciding entire direction: whether coordinate bank has any advantage once continuous/local POD gets identical AMR and hybrid machinery. Pilot may yield useful adaptive ROM interface with NO neural advantage, which is valid outcome. No new measured numbers. ALL NOTES DELIVERED.
