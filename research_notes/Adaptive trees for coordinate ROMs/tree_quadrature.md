# Tree quadrature: complete researcher notes

Source: Astra tree-quadrature researcher, 2026-10-26. No files/jobs/installs. Notes retained as research evidence; quantitative repository statements below are source notes, not newly generated results.

## Part 1 of 4

Core conclusion: tree quadrature offers selective auditable integration and diagnostics; no speed advantage is established, and current Burgers evidence argues against assuming one. It cannot improve an unchanged bank's best-approximation floor. Strong first target: repeated heterogeneous diffusion, reduced warm start with independently checked FE correction/fallback, compared with global continuous POD and local POD.

VERIFIED REPOSITORY EVIDENCE (absolute base /home/tahmid/Dev/Tunable-NM-ROM-Claude):

- worktrees/2026-10-20-jcp-consolidated/experiments/jcp-combined/vendor/quad2d/qcore.py:209 offmesh_data caches values, gradients and tests; :238 tested_value and :252 tested_jac contract caches. Insert tree rule layer here. :120 Model and :141 values_grads expose frozen bank and physical AD gradients. :185 operators uses sine tested mass/lambda. :230 mesh_data contains actual stencil path.
- same combined/vendor/quad3d/offmesh.py:73 offmesh_tables caches B,D,P; :84 contract_offmesh computes J_N(c)=P^T[diag(Dc)B+diag(Bc)D]. Fewer points only reduces part of cost.
- combined/t2core.py:115 residual histories; :121 diffusion row scaling; :133 evalJ uses sine-eigenvalue simplification; :165 evolves using A*c_n,A*c_(n-1),F(c_n); :259 sampled cold initialization; :265 full-field decode. :35 BDF2 fixed-step coefficients. :247 a_conditioning does SVD rank/cond check.
- worktrees/2026-10-21-imp-geometry/experiments/imp-geometry/bank.py:152 weighted QR/SVD ordering using frozen reference rule; :171 bank_space; :183 mass M=P^T W P; :184 stiffness K=Gx^T W Gx+Gy^T W Gy. rom.py:127 make_query, :137 Mt selection, :145 Mtr/Ktr, :151 input fit inverse mass. This provides general assembly precedent; do not retain nu*lambda*A outside box eigenfunction setting.
- reports/2026-10-21-improvement-campaign/final/report.md:128 E6; :144 says doubling/quadrupling points costs 1.37x/2.14x step-Jacobian microbenchmark; fitted-weight arms fail Q2. :208 E10; :211 max rho 0.0386 across 17600 accepted states, trigger never fires. Numbers should be source-generated for report, not copied by hand.
- CRUCIAL NUANCE: worktrees/2026-10-21-imp-fronts/experiments/imp-fronts/E10-DESIGN.md:8-28 explicitly states STEP-UNCONVERGED establishes MISSING STEP CERTIFICATION, NOT THE MAGNITUDE OF TEMPORAL ERROR. :150-160 every-step evidence, 171 runs / 17771 states including initial; threshold 0.116, maximum 0.0386 overall, 0.0148 FC1, output-state recompute match 2.5e-9 relative. Do not assert that measured time error explains all steep failure. Every-step rho checks show quadrature escalation does not fire under that registered criterion, not a universal proof that quadrature is harmless.

BENEFIT INFERENCES: best candidates are localized/moving difficult integrands, localized coefficients, cut geometry or future local basis supports. Global sine tests and global smooth neural features can force global refinement. Compare with higher-order fixed Gauss; a tree is not automatically best. For fixed geometry and affine kappa(x;mu)=sum_l theta_l(mu)kappa_l(x), preassemble reduced stiffness blocks; tree benefit is then principally offline. Nonaffine online coefficients/source/geometry may justify online integration. Representation adaptation and quadrature adaptation must be separate. Local dictionary activation is a union-of-spaces; disclose total dictionary and selection cost. Positive weights do not automatically confer ECSW energy guarantees on this weak least-squares method.

PRIMARY/OFFICIAL SOURCES (all fetched/verified, exact scopes; keep derivations/proposals labelled):

1. Steven G. Johnson Cubature documentation https://github.com/stevengj/cubature : vector-valued batched h-adaptive integration and p-adaptive Clenshaw-Curtis; p can be better for smooth <=3D. References Genz-Malik 1980 and Berntsen-Espelid-Genz 1991. Not a JAX performance claim.
2. Pedro Gonnet, A Review of Error Estimation in Adaptive Quadrature, ACM Computing Surveys 44(4),2012, DOI 10.1145/2333112.2333117, https://arxiv.org/html/1003.4629v2 , esp §5.3: rule differences fail by cancellation/aliasing; no unconditional bound from agreement.
3. Patera & Yano, An LP Empirical Quadrature Procedure for Parametrized Functions (2017), https://arrow.utias.utoronto.ca/~myano/papers/py_2017_eqp.pdf : sparse positive rules, training integral constraints, regularity/coverage-dependent guarantee; assumes accurate truth rule and constant integration.
4. Grimberg, Farhat, Tezaur, Bou-Mosleh, Mesh sampling and weighting for the hyperreduction of nonlinear Petrov-Galerkin reduced-order models with local reduced-order bases, https://arxiv.org/abs/2008.02891 : ECSW extension to PG and FE/FV/FD; their local bases are piecewise-affine approximation subspaces, not automatically our spatial cell banks. Do not transfer industrial speed claims.
5. Smetana & Taddei, Localized model reduction for nonlinear elliptic PDEs: localized training, partition of unity, and adaptive enrichment, https://arxiv.org/pdf/2202.09872 : §5 local Riesz residuals, overlap/stable-decomposition constants; nonlinear bound requires stability/proximity (Prop 5.1, Remark 5.1); exponential enrichment convergence is for linear coercive problems. Continued next message.

## Part 2 of 4

SOURCES CONTINUED:

6. Tommaso Taddei, An offline/online procedure for dual norm calculations of parameterized functionals: empirical quadrature and empirical test spaces (2019), https://arxiv.org/html/1805.08100v2 . §2.1 $r^T G^{-1}r$ dual norm; §2.5 Eq24 splits quadrature error and missing empirical-test-space content. Training constraints can fail on unseen parameters. Only finite dimensional restricted test residual is available cheaply without further assumptions.
7. JAX official https://docs.jax.dev/en/latest/control-flow.html : jit specializes on shape/dtype and new shape triggers recompilation; internal data-dependent shapes restricted. Supports proposed buckets/padding, not a performance guarantee.
8. SciPy official https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cubature.html : array-valued cubature, global adaptive subdivision, abs+relative tolerances, explicit nonconvergence; convergence not guaranteed; tensor GK has exponential dimension node count. Good independent CPU reference, not inner GPU controller.
9. Ohlberger, Rave, Schindler, True Error Control for the Localized Reduced Basis Method for Parabolic Problems, https://arxiv.org/pdf/1606.09216 : elliptic reconstruction estimates cover discretization+reduction for scalar parabolic equations. Example rigorous separation, not directly current nonlinear Burgers LSPG.
10. Hernández, Ares de Parga, Rossi, Dimensional hyperreduction ... manifold-adaptive weights (Sept2026 PREPRINT), https://arxiv.org/abs/2609.03068 : feasible fixed ECM rule, greedy pruning, positive weight redistribution and latent-dependent weights. Prior art only, do not transfer benchmark savings or treat as established guarantee.

IMPLEMENTATION PROPOSAL (unimplemented):

A. General assembly. $u_r(x;c)=u_{\rm bc}(x)+B_r(x)c$. Smooth solve tests $\psi_i$ with M comfortably > active solved dimension. $A_{ij}=\int\psi_i b_j$, $K_{ij}(\mu)=\int\nabla\psi_i^T\kappa(x;\mu)\nabla b_j$; lifting/source/boundary explicit. Generic LMM residual $R=A(a_0c+a_1c_n+a_2c_{n-1})+\Delta t\{b_0F(c;\mu)+b_1F(c_n;\mu)\}$; $J_R=a_0A+\Delta t b_0J_F$. Replace box $\nu\lambda A$ scaling with declared general physical/test normalization/preconditioning.

OPERATOR FIDELITY: qcore has mesh/upwind and continuum point/flux variants. Tree must integrate chosen operator. For FOM-exact FV/FD target keep actual local stencil/face dependencies; do not silently replace discrete upwind by continuum derivatives. Continuum weak model is a separately labelled change. Both are tested weak residuals, not global pointwise residual minimization.

B. Tree/cache. Each leaf $K=T_K(\widehat K)$ stores stable Morton/cell ID, level/bounds, mapped positive Gauss nodes, $w_{Kq}=|\det DT_K|\widehat w_q$, B/physical gradients, tests/gradients, static local operator blocks, version hashes, optional audit-rule caches. Integration leaves disjoint: replace parent contribution by children; never add parent and children integrals together. (Trial hierarchy may retain parent basis.) For globally smooth bank/tests, pure integration-tree faces are bookkeeping, no artificial PDE flux/penalty needed. Cut geometry/discontinuous coefficients/FV or discontinuous local trial spaces need actual interface treatment. Naive masked box integration introduces geometry error. Update cached $A_{\rm new}=A_{\rm old}-A_K+\sum_{\rm child}A_{\rm child}$, similarly K, with periodic full f64 sum checks. Keep bank ordering and physical normalization reference fixed while quadrature adapts.

C. Residual AND Jacobian integration estimator. Let $g_i(x;c,H)$ be complete weak time-discrete residual integrand, H accepted histories. Parent rule $Q_K^p$, child composite $Q_K^+=\sum_{\rm child}Q_{\rm child}^p$. At SAME candidate and histories $d_K^R=(Q_K^+-Q_K^p)g$; $d_K^J=(Q_K^+-Q_K^p)\partial_c g$. Frozen independently accurate test Gram $G_V=[(\psi_i,\psi_j)_V]=L_VL_V^T$, $W=L_V^{-1}$; $\|Wr\|^2=r^TG_V^{-1}r$ is RESTRICTED test-space dual norm. Coefficient metric $G_c=L_cL_c^T$, $S_c=L_c^{-T}$ converts unit normalized coefficient directions. Proposed score $\eta_K^Q=\|Wd_K^R\|_2+\Delta_{\rm trust}\|Wd_K^JS_c\|_F$. Frobenius controls operator norm but may be costly. $\eta_{\rm sum}^Q=\sum_K\eta_K^Q$ avoids intercell cancellation IN THE RULE-DIFFERENCE ESTIMATE, not proof of true integral bound. Compare abs+relative tolerance to stable mass/history/forcing scales; do not divide only by near-zero residual/nonlinearity. State dt convention: residual-integral vs residual-rate tolerance differs by time factor.

For cheap screen use a small fixed set of normalized Jv directions including current update; full Jacobian offline and periodic audit. Label directional screen heuristic. No universal Richardson denominator $2^p-1$ without verified smoothness/asymptotic order. Narrow features can evade parent and child: minimum coverage and independent/higher-order rule audits. Continued.

## Part 3 of 4

IMPLEMENTATION CONTINUED:

D. Independent local weak probes. Quadrature convergence means existing tested equations integrated well, not missing physics observed. On overlapping patches $\omega_K$ choose smooth $\zeta_{K\ell}=\chi_K(x)p_\ell(T_K^{-1}x)$, outside solve test span. Include neighbor-spanning tests; cell-interior bubbles can miss interface imbalance. Evaluate FULL weak PDE/time-discrete defect with a separately qualified audit rule: $r_{K\ell}=R(u_r;\zeta_{K\ell})$; $\eta_K^V=\sqrt{r_K^TG_K^{-1}r_K}$. $G_K$ is physical declared H1-type test Gram, removing arbitrary test amplitude/support-volume bias. Use rank revealing factorization if dependent; diagonal-only normalization misses correlations. These are INDICATORS: finite probes miss directions. Root-sum-square patch norms needs overlap/stable decomposition constants before claiming a global bound. Initial probes should use classical quadrature; don't immediately fit another residual empirical rule against repository output-snapshot-only requirement.

E. Solve-retry state machine:

OFFLINE fix bank, ordering metric, solve/probe test spaces; independently qualify positive classical reference rule; preassemble affine/static operators; precompile small capacity ladder; validate residual AND Jacobian on decoder-output snapshots, active rank prefixes, forecast/trust perturbations and held-out states; save fallback rule, input rule, provenance.

ONLINE sampled volume-weighted initial fit. Before each step save accepted coefficients/history/time metadata. Freeze basis/tests/metric/rule/parameters inside nonlinear solve. Solve and verify fresh-Jacobian stationarity/root criterion. Audit parent-child residual+Jacobian at candidate.

If integration fails: mark high eta cells, refine subject to point/depth/overhead limits, rebuild affected tables AND history terms under new rule, restart SAME step from saved accepted state; if budget exhausted fixed qualified rule or full fallback.

If nonlinear solve fails: solver/fallback policy, don't presume quadrature cause.

Once both pass: independent probes and separate temporal check. If temporal fails, valid step policy/retry. If spatial probes remain large, report/enrich trial representation under separate policy or FE fallback. Accept only after relevant checks pass. Coarsen only between accepted steps with hysteresis/audit.

Quadrature-only adaptation leaves physical coefficients $c_n,c_{n-1}$ unchanged; recompute $A_Qc_n$, $A_Qc_{n-1}$, $F_Q(c_n)$, scales/cache affected by rule. Nested trial enrichment append ZERO coordinates to EVERY stored history, retaining parent fields. Coarsening/non-nested replacement/re-rotation needs weighted physical projection of ALL required histories and projection-error check; reset damping/extrapolation if needed. Existing BDF2 constant-step coefficients cannot simply receive a different dt; use valid variable-step coefficients/history or deliberate verified startup.

F. Empirical rules later only if profiling supports. Fit positive local rules on DECODER-OUTPUT snapshots as required here; constant/volume reproduction and declared output moments; validate weak operator+Jacobian on unseen reachable states+perturbations. Revalidate/refit after bank/support/rank/coefficient/test changes. Don't split old learned weights into children and assume validity. Classical fallback on failing cells. Start fixed weights per solve. If $w=w(c)$ later, exact Jacobian includes $dw/dc$; otherwise explicitly an outer iteration. Recent MAW-ECM preprint relevant prior art but not evidence for our benefit.

CERTIFICATION HIERARCHY:

1. Fresh solver tolerance = convergence of declared frozen-rule optimization/root, not PDE accuracy.
2. Independent rule agreement = conditional quadrature qualification unless analytic derivative/validated integration bounds.
3. Independent finite local weak probes = residual indicators, not full dual residual bound.
4. Linear coercive: if $a(v,v)\ge\alpha\|v\|_V^2$ and known $\alpha_{\rm LB}>0$, $\|u-u_r\|_V\le\|R(u_r)\|_{V'}/\alpha_{\rm LB}$. Sparse/probe residual needs omitted-component and integration-error bound before substitution. Scalar diffusion coefficient lower bound may give coercivity; high contrast may make pessimistic. Trusted FE residual/correction can decide final acceptance while probes decide optional enrichment.
5. FE-relative certificate is relative to discrete FE solution; continuum needs FE/geometry/data error too.
6. Nonlinear/transient needs stability/proximity and propagation analysis. Adding guessed error budget pieces isn't a theorem.

Conditioning: monitor $\sigma_{\min}(WJ_RS_c)$ and rank of tested mass/tangent. M>r is necessary design rule not observability guarantee. Use independently accurate metric; block/local orthogonalization if support sparsity matters. Constrained FE transfer must honor boundary/hanging constraints. Physical-volume weights in initialization, errors, history transfer. Continued.

## Part 4 of 4: final delivery

JAX/COST PROPOSAL: host tree controller only between solves, packed device arrays, finite ladder of point/rank/(optional level) capacities. Zero padded weights and FINITE padded values; 0*NaN isn't valid masking. Depth buckets can group mapping/support work but many tiny launches can lose to one packed batch. Padded capacity costs capacity, not active count: record both. Pass big arrays explicitly, don't capture in closures. Cache bank/rank/geometry/cell/rule versions. First new-cell evaluation includes neural gradients, allocations and transfers: adaptation cost, not free setup.

Approximate current cost $C_{\rm iter}\sim C_{\rm nonlinear}(m,r,M)+C_{\rm normal}(Mr^2)+C_{\rm solve}(r^3)$; explicit Jacobian contractions can have $O(mMr)$. Tree only reduces m, leaving dense reduced costs. Local bank savings require actual sparse/local contractions, not zero-filled dense matrices. If measured reducible integration fraction f of total query cost, optimistic complete-elimination ceiling $S_{\max}\le1/(1-f)$ BEFORE adaptation overhead. Must MEASURE f, not infer it from nodes.

DECISIVE TESTS:

1. Correctness fixtures: constant/polynomial integration, physical volume, parent replacement, gradient map, manufactured weak diffusion, source/lifting, coefficient discontinuity/cut geometry, rank/conditioning, finite difference Jacobian; seed narrow feature missed initially.
2. Static saved-state replay: rank prefixes/coefficient contrast, compare uniform Gauss, higher-order fixed Gauss, classical tree, any empirical rule; residual AND Jacobian errors; trust/forecast perturbations. If refinement global or audit costs exceed fixed rule, stop online tree.
3. Offline-online split: repeated diffusion affine preassembly first. Tree that only helps setup yields assembly/portability benefit, not evolution speed.
4. Probe fault test: accurate integration with missing rank/local trial directions -> probes must detect region. Adequate rank with underintegration -> integration estimator fires without falsely asking for representation. Test independent temporal defects separately.
5. Warm-start benchmark: same final trusted FE acceptance/output; compare normal FE start, global continuous POD, local POD, coordinate bank fixed quadrature, coordinate bank tree. Include transfers, certificates, correction/fallback and bad starts. Cost and final accuracy from SAME invocation.
6. Only then moving-localized diffusion/reaction transient with qualified time method; compare fixed offline UNION tree. If union is similarly cheap, prefer fixed rule.
7. Burgers only on NEW evidence: E10 does not justify another quadrature escalation campaign. Show case with independently material integration error AND qualified solver/reference/time checks.

STOP ONLINE TREE (keep useful offline code) if: integration already below budget yet physical error unchanged; trial/probe/time failure persists after quadrature passes; nearly uniform refinement; dense solve/launch costs dominate theoretical saving; padding/cache/audits/retries erase gain; output-fitted rules fail Jacobian/trajectory validation; probes miss seeded errors or cost nearly full order; certified warm start slower than ordinary FE/local POD at same accuracy; gain depends on omitting initialization/rule construction/adaptation/certificate/output.

Use same-device within-job timing, warm GPU, medians/outliers/failure rates, persist repetitions, f64 highest precision. No cross-job speed ratios. Disclose dictionary size and selection cost for adaptive trial activation. Generic online mesh-independent cost claim must include input sampling, feature rebuild, scans and output; full output itself scales with output count.

UNRESOLVED: no measured tree-rule speed, overhead, estimator reliability or total gain. No implemented nonlinear stability lower bound/probe-completeness estimate; no proof tree cures continuous-POD cost disadvantage. Local quadrature order, probe coverage, cadence and tolerance allocation need experiments. m~4M project heuristic is not universal theorem; preserve conservative policy and inspect actual rank/conditioning/qualification.

Full original notes were finalized in my prior final answer to coordinator, now completely delivered here across parts 1–4. No new research/writes occurred during transfer. Ten primary/official substantive sources above; CECM 2023 also surfaced but not used because final full-content fetch timed out, so no unsupported source-specific claim needed.
