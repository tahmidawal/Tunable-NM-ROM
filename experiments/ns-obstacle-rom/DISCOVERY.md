# Obstacle-flow ROM, quadrature and POD discovery

This is a prospective implementation recommendation for the fixed-geometry Navier–Stokes campaign, written before its data, neural fitting or numerical production. It contains no new measured results; the proposed thresholds require independent design review before use.

**Implementation precedence:** the coordinator's `experiments/ns-obstacle/DESIGN.md` and dated amendments supersede this initial recommendation. The reviewable implementation is described in `API.md`; in particular it uses the actual inlet coordinate in both dimensions, a viscosity-scaled Stokes pressure lift, mandatory mixed Galerkin and PCHIP controls, constant-first pressure augmentation, and the final central qualification thresholds. No runtime physical gate has passed.

## Evidence recovered

The current source fork is `93f399faad67f94ea2b56fe25f13003143e64694`, with code-only mirror base `5a64899fdcfbeb6cfe5b3371af1a8d8a3d2b568f`. The lane is `exp/2026-10-27-ns-obstacle-rom`, its worktree is `/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-27-ns-obstacle-rom`, and its reserved cluster namespace is `/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/`. No jobs have been submitted.

The canonical lab log and the completed Burgers, improvement and refinement reports establish no general bank advantage over strong FOM or POD competitors. The geometry improvement lane did not establish that leaving a box creates an advantage. The continuous-POD comparison withdrew the comparative multi-mesh cost claim. The refinement report separates a failed sampled POD rule from an admitted unsampled POD comparator. These limitations govern this campaign too; their numerical tables remain in their source-generated reports rather than being transcribed here.

| Component | Exact source at the fork, unless stated | Status and disposition |
|---|---|---|
| Periodic NS2D FOM | `experiments/ns2d/ns2d_fom.py`, `ns2d_indep.py` | Previously verified only for its periodic vorticity problem. Its Poisson inversion and periodic operator are inappropriate for channel walls and obstacle forces. |
| NS2D neural bank | `experiments/ns2d/ns2d_decoder.py` | Periodic scalar vorticity output, integer Fourier features, grid mean removal. Reuse training-program structure only after geometry-specific replacement. |
| NS2D ROM/POD | `experiments/ns2d/ns2d_rom.py` | Weak least squares with Fourier tests and an exact Arakawa quadratic tensor; no off-mesh rule. Reuse analytic quadratic contraction and Jacobian ideas, not PDE operators or test functions. |
| Periodic NS3D FOM | `experiments/ns3d/ns3d_fom.py`, `ns3d_independent.py`, `ns3d_verify.py` | Periodic spectral Leray projection and rotational advection. No channel pressure/outlet/obstacle treatment. Not an obstacle-flow reference. |
| NS3D coordinate bank | `experiments/ns3d-coordnet/coordnet.py` | Continuous curl of periodic vector potential. Autodiff curl, gradients, weighted projection loss and QR/SVD ordering are useful primitives. Periodic features, grid normalization, Leray cleanup and mesh-dependent orthogonalization are not portable unchanged. |
| Co-moving NS3D model | `experiments/ns3d-shift/shift_rom.py`, `ns3d-shift-head/head_rom.py` | Translation invariance and Fourier tests are tied to the torus. Freeze the geometry; do not move the obstacle or reference frame with these routines. |
| Audits and timing | `experiments/ns3d-coordnet/rom_mesh.py`, `verify_coordnet.py`; `experiments/ns3d-shift-head/head_ladder.py` | Reusable provenance, same-job timing and independent reconstruction patterns. Adapt dtype and output contract; inherited allowance for stored single-precision comparator fields is not adopted. |
| Positive-weight fit and POD workflow | `experiments/jcp-hr-baselines/{nnls.py,hrcore.py,build.py,rom.py}` | Source for fit diagnostics and comparator accounting, not a qualified obstacle rule. Retain all failed fit outcomes. |
| Curved geometry scalar work | `worktrees/2026-10-21-imp-geometry/experiments/imp-geometry/DESIGN.md` and implementation | Demonstrates why box envelopes and sine tests need replacement. Scalar Burgers boundary treatment is not an incompressible mixed formulation. |

Read evidence: root `reports/2026-10-17-full-experiment-plan/final/report.md`, `reports/2026-10-21-improvement-campaign/final/report.md`, `reports/2026-10-26-burgers3d-refinement/2026-10-27-burgers3d-refinement.md`, `reports/2026-10-20-spectral-investigation/{why-spectral,where-we-win}/report.md`; and `worktrees/2026-10-01-ns3d-coordnet-bank/reports/2026-10-01-ns3d-coordnet-bank.md`. The last report explicitly distinguishes the continuum curl from sampled/projected bank derivatives and records the historical pilot-floor failure. No old sealed cohort is read.

## First construction: mixed steady hybrid model

Keep the inlet mean speed fixed and vary viscosity. The coordinator's prospective family uses training Reynolds numbers $15+1.25j$ for $j=0,\ldots,8$, development values $16.875,23.125$, and frozen validation values $16,18,22,24$. Only steady solutions are learned in this pilot. A steady FOM root solve is the matched comparator, not a long transient. The separate FOM transient verification remains useful but is not charged as a mandatory per-query steady operation.

Let $\Gamma_D$ contain inlet, solid walls and obstacle, and let $\Gamma_o$ be the outlet. Obtain a unit-inlet Stokes velocity lifting $\ell$ from the admitted body-fitted FOM with zero wall/obstacle trace and the exact selected outlet condition. The fixed inlet makes it reusable across viscosities. Retain its discrete divergence and boundary error as part of the total error budget; it is not asserted exactly solenoidal. Store the lifting's continuous FE evaluator and count its lookup/reconstruction costs. Do not invent a componentwise masked inlet profile and call it divergence-free.

For 2D channel height $H$, circle centre $(c_x,c_y)$ and radius $a$, use a smooth, dimensionless product envelope

$$b(x,y)=\frac{x}{L}\frac{y(H-y)}{H^2}\frac{(x-c_x)^2+(y-c_y)^2-a^2}{L^2}.$$

Each neural velocity column is $g_j=b\,f_{\theta,j}$, with two vector outputs per column. The factor vanishes on every essential boundary and deliberately has no outlet factor. Its small near-wall values can worsen conditioning; derivative and conditioning checks are mandatory. It enforces homogeneous velocity trace, not incompressibility. Do not divide snapshots by $b$ during training.

For a straight pipe with axial coordinate $x$, pipe radius $R$ and sphere centre $c$, the analogous preliminary envelope is

$$b(x,y,z)=\frac{x}{L}\frac{R^2-y^2-z^2}{R^2}\frac{\lVert (x,y,z)-c\rVert^2-a^2}{L^2}.$$

This is only infrastructure for the later three-component model. It does not qualify the 3D representation, and it must use the same exact geometry as the boundary evaluators.

The first trial field and pressure are

$$u_r=\ell+\sum_{j=1}^{r}a_j g_j+\sum_{j=1}^{s}\eta_j s_j,\qquad p_r=\sum_{j=1}^{p}d_j\chi_j.$$

Here $\chi_j$ are continuous FE pressure POD modes from the same training states, and $s_j$ are FE velocity supremizers. This is explicitly a **hybrid learned-velocity model**, not a fully mesh-free velocity–pressure representation. The lifting, pressure and supremizer evaluation costs, storage and point-location work remain in every applicable query contract. The linear span is the first ordered-bank model; a nonlinear latent head is deferred to avoid confounding representation, pressure stability and solver changes.

Use $r\in\{2,4,8\}$, $p=\min(r,\text{numerical pressure-snapshot rank})$, and one uncompressed supremizer per pressure mode, $s=p$. Total solved unknowns are $r+2p$, at most $24$. Disclose every dimension separately. A rank-deficient snapshot prefix is omitted with its reason, not padded with numerical noise. Use one small tanh neural architecture and one recorded seed initially; fit the largest bank once, then freeze the full QR/SVD ordering from training data before taking nested prefixes. Tanh permits an independent NumPy evaluator with analytic first derivatives for same-CPU online comparisons. Any transform is computed using the common integration measure and frozen across output meshes. Do not perform an undisclosed per-output-mesh Löwdin rotation.

Train on the admitted velocity snapshots minus the lifting with physical-volume weights, retaining a derivative-sensitive diagnostic. Evaluate unconstrained weighted least-squares representation error of each deployed trial space before solving the PDE. Compare to mass-weighted POD with the same data and number of velocity unknowns; a learned linear subspace cannot improve the optimal training mean-square POD projection at equal rank. A POD floor from velocity-only columns and a hybrid floor including extra supremizers must not be presented as an equal-dimension comparison.

## Weak residual, pressure convention and stability

Adopt the reference lane's exact viscous weak form and outlet condition. Gradient-form diffusion with $(\nu\nabla u-pI)n=0$ and symmetric-stress traction $(-pI+2\nu\epsilon(u))n=0$ are distinct boundary-value problems. Their equivalence for divergence-free interior PDEs does not permit silently switching outlet conditions. Physical surface force always uses the appropriate Cauchy stress with the declared kinematic/dynamic pressure convention.

In the gradient-form case, the stationary residual rows are

$$R_i(u,p)=\int_\Omega (u\cdot\nabla)u\cdot v_i+\nu\nabla u:\nabla v_i-p\,\nabla\cdot v_i\,dx-\int_{\Gamma_o}t_o\cdot v_i\,ds,$$

$$C_j(u)=\int_\Omega q_j\nabla\cdot u\,dx.$$

Velocity tests vanish on $\Gamma_D$ but remain free on the outlet. Use a fixed test bank across widths: $64$ momentum test functions and $16$ scalar continuity tests. Include the maximum-prefix supremizers, smooth low FE vector-Laplacian modes and appropriate pressure/scalar modes, remove dependent columns with recorded rank tolerance, and fail if the intended independent counts cannot be obtained. This gives a total of $80$ rows for at most $24$ unknowns. Test functions are geometry-compatible; box sines and old Fourier modes are excluded.

Minimize the overdetermined weak residual using fixed nondimensional Riesz weights,

$$\min_{a,\eta,d}\ \|K_V^{-1/2}R\|_2^2+\|M_Q^{-1/2}C\|_2^2,$$

where $K_V$ is the test-space $H^1$ Gram and $M_Q$ its scalar $L^2$ Gram. The exact fixed normalization is part of configuration and does not change with viscosity, quadrature setting or iteration. This is **weak residual least squares**, not Galerkin. Its stationarity alone does not establish a small physical or continuity error. Preserve the returned fields and both residual blocks.

Define each supremizer variationally using the same essential-boundary space, $(s_j,v)_{H^1}=\int_\Omega\chi_j\nabla\cdot v\,dx$. At an open outlet the supremizer must not be forced to zero on all boundaries: that could remove the constant-pressure response. A prescribed traction ordinarily fixes the pressure level. Do not additionally impose mean-zero pressure or remove the constant pressure mode unless the chosen reference formulation actually has a nullspace. Pressure comparisons retain the selected physical gauge; any aligned-pressure diagnostic is separate.

For each prefix compute

$$\beta_r^2=\lambda_{\min}(M_p^{-1/2}B_rK_r^{-1}B_r^TM_p^{-1/2}),$$

with velocity $H^1$ Gram $K_r$, pressure $L^2$ Gram $M_p$ and divergence coupling $B_r$. The prospective gates are $\beta_r\ge10^{-3}$ in the frozen nondimensionalization and $\beta_r/\beta_{h,Q_r}\ge0.5$, where $\beta_{h,Q_r}$ is obtained from full-FE velocity supremizers for the same reduced pressure space. Verify full column rank and condition number at most $10^{10}$ for the whitened coupled steady Jacobian at initialization and every returned solution. Use QR/SVD solves for diagnostics; do not conceal singularity with an unreported ridge. These checks are finite-space numerical conditions, not a universal nonlinear stability theorem. A failed pair stops that prefix or requires a recorded pre-result design amendment.

## POD controls and nonlinear evaluation

Give POD the same admitted training states, lifting, pressure space, supremizer construction, scalar test set, quadrature controls and solver tolerances. Include both a conventional stable mixed POD-Galerkin root solve and the identical overdetermined weak least-squares solve, named separately. Evaluate floors before dynamics and retain the fastest accuracy-admissible POD policy.

For fixed trial/test functions, the convection term is exactly quadratic in coefficients. Build the unsampled reduced tensor using converged mapped-element integration, including constant and linear lifting contributions, and independently compare its residual and analytic Jacobian with direct full-rule assembly. It removes online full-mesh evaluation without any fitted rule; its cost scales with reduced dimensions. At these pilot ranks it is a mandatory strong control for **both** POD and the learned bank. Do not force POD to pay full-grid assembly when the exact tensor is small.

Boundary force is linear in pressure coefficients and velocity derivatives for a stationary geometry, so converged obstacle-boundary force moments may also be precomputed. Include the lifting contribution and physical normal orientation. With $n_f$ the outward fluid normal on the obstacle, force exerted on the body is $-\int_{\Gamma_b}\sigma n_f\,ds$. Compare direct traction integration and precomputed force moments. A pressure/velocity field error does not automatically bound force error.

## Off-mesh quadrature qualification

Start with positive mapped Gauss quadrature on the actual curved fluid elements, using volume Jacobians, and separate curved boundary quadrature. Its points need not be FOM nodes. This is geometry-conforming, mesh-associated integration; it is not a claim that no geometric mesh exists. Establish geometry area/volume, boundary length/area, first moments and divergence-theorem controls before evaluating a model. If the available mesh is polygonal, document and converge the geometry approximation separately rather than using exact-circle assertions for chordal cells.

The first compact candidate is a positive training-output moment fit on admissible fluid points, with support caps $m\in\{320,640,1280,2560\}$ for the fixed test bank. Fit functions evaluated on reconstructed decoder/POD training states, including relevant output/gradient products and fixed geometric moments; do not fit nearly-zero converged PDE residual snapshots. Selection may use development states only. This is described as fitted positive quadrature unless the implemented equations meet a documented ECSW definition; positivity does not prove inlet-driven Navier–Stokes energy conservation. Permit one fit per support, preserve failed fits and their NNLS optimality diagnostics, and stop when the cap is reached. Refit when the actual test/trial/integration problem changes.

Prospective common qualification thresholds, to freeze in the integrated design: reference-rule operator relative difference at most $10^{-4}$, normalized Jacobian-vector difference at most $10^{-4}$, returned-state velocity difference at most $10^{-3}$, pressure difference at most $2\times10^{-3}$, drag/lift coefficient difference at most $10^{-3}$, and continuity/mass-balance gates no worse than the common reference/ROM acceptance requirement. Denominators must be bounded by declared physical scales, not the almost-zero converged residual. Compare at training, development and returned solution states; the validation rule is frozen before its first evaluation. A successor rule alone is not an independent reference unless its own integration convergence is established. Keep geometry, nonlinear-operator, solution and force differences in separate columns.

The mapped rule and exact tensor remain admitted controls if every compact fit fails. Failed fitted POD sampling is not evidence for a learned-bank advantage. The heuristic $m\approx4M$ selects an initial candidate, not a certificate for curved geometry.

## The solenoidal alternative and why it is deferred

In 2D, $u'=\nabla^\perp\psi$ is divergence-free when differentiated consistently. No slip on a solid boundary needs both constant streamfunction trace and zero normal derivative; setting only $\psi=0$ ensures no penetration but can leave tangential slip. A squared homogeneous envelope can enforce zero first derivatives, but assigning the same zero streamfunction on all disconnected solid boundaries can overconstrain admissible circulation/flux topology. The inlet lifting and outlet conditions still need a compatible construction. In 3D, $u'=\nabla\times A$ is solenoidal, but $A=0$ alone does not generally imply no-slip velocity. A sufficiently vanishing potential can enforce it at the expense of second derivatives for weak viscous terms and additional representation restrictions. Pressure and forces still require a justified recovery solve. These are later controlled alternatives, not assumptions inherited from the periodic curl bank.

## Resources, stops and dependency handoff

Reference/data gates precede all fitting. Proposed first ROM allocation ceiling is two GPU-hours inclusive of a calibration fit, one largest-bank fit, prefix/POD/operator assembly and a small paired evaluation; one extra GPU-hour is reserved only for a documented implementation repair. These are planning ceilings, not measured estimates. No architecture search or multi-seed claim follows from this pilot. Store peak host/device memory, compile/calibration time and all failed allocation cost. Cap a single assembled candidate at $16$ GiB and avoid materializing point-by-test-by-trial-by-trial arrays; chunk tensor assembly. At $r=8,p=s=8,M=64$ the coefficient tensor is small, but pointwise derivative and FE-evaluation memory still must be measured before 3D work.

Use the common campaign acceptance criteria for velocity, pressure, divergence, mass balance and forces. Report $\|u_r-u_{ref}\|_{L^2}/(U\sqrt{|\Omega|})$, $\|p_r-p_{ref}\|_{L^2}/(U^2\sqrt{|\Omega|})$, $D\|\nabla\cdot u_r\|_{L^2}/(U\sqrt{|\Omega|})$, and net flux divided by inlet flux, with the coordinator's fixed characteristic $U,D$. Also report velocity relative to reference-field norm and the actual pressure gauge. Near-zero lift/transverse force uses absolute coefficient error, not division by its small reference value. A representation failure stops the corresponding rank; a reference failure stops training; a pressure-stability or quadrature failure excludes the configuration from ranking. A no-speedup result is an admissible outcome and ends a speed claim, not a reason to weaken a comparator. No large 3D bank before admitted reference data, small projection floors, measured derivative/evaluation memory and an audited resource projection.

Timed outputs are (i) full velocity and pressure on a declared common output set, (ii) fixed probes, and (iii) integrated forces, each with identical FOM/POD/bank contracts. Primary online comparisons run NGSolve FOM and NumPy bank/POD reduced algebra on the same allocated CPU cores with the same thread limit, randomized paired repetitions, CPU burn-in and no overlapping GPU training. GPU fitting is offline. A JAX/GPU online result is an optional heterogeneous-device diagnostic, never the primary same-device speed claim. Save initialization, nonlinear solve, output reconstruction, total query and repetition arrays from the same invocation that yields scored outputs. Include a tuned coarse/high-order FOM mesh ladder and reusable setup, Stokes lifting and preconditioner policies under explicit cold and amortized contracts. Account for offline data, mesh generation, lifting, pressure/supremizers, fitting, ordering, integration selection and tensor assembly. Break-even is undefined when the ROM has no admitted online saving. Hold training bank, tests, integration rule and output contract fixed when varying output mesh; continuous FE POD also supports evaluation away from training nodes.

Missing infrastructure is substantial: admitted curved mixed FOM, pressure/traction convention, common-point FE interpolation with derivatives, lifting and supremizer solvers, geometry quadrature, weighted snapshot writer, neural boundary checks, coupled weak solver, and force/mass/balance output. None is supplied by the periodic NS bank. The next implementation decision depends on the reference agent's verified library availability and exact formulation.

## Primary sources consulted

- Ballarin, Manzoni, Quarteroni and Rozza, *Supremizer stabilization of POD-Galerkin approximation of parametrized steady incompressible Navier–Stokes equations*: [author preprint](https://www.mate.polimi.it/biblioteca/add/qmox/13-2014.pdf). Supports pressure-space enrichment and stable steady mixed POD; it does not validate this new neural trial/test pair.
- Stabile and Rozza, *Finite volume POD-Galerkin stabilised reduced order methods for the parametrised incompressible Navier–Stokes equations*: [paper](https://arxiv.org/abs/1710.11580), [full text](https://arxiv.org/pdf/1710.11580). Compares supremizer and pressure-Poisson strategies and explains inlet lifting; formulation details must still match the selected FE benchmark.
- Stabile et al., *POD-Galerkin Reduced Order Methods for CFD Using Finite Volume Discretisation: Vortex Shedding Around a Circular Cylinder*: [paper](https://arxiv.org/abs/1701.03424). Pressure recovery matters for drag/lift; a velocity-only result is insufficient.
- Pressio developers, [ECSW documentation](https://pressio.github.io/rom-tools-and-workflows/generated/romtools.hyper_reduction.ecsw.html). Describes projected sampling/weighting and explicitly limits general energy-conservation claims. We adopt no conservation theorem from the method's name.

## Glossary

- **FOM:** the full-order numerical PDE solver used as reference and comparator.
- **POD:** the volume-weighted linear subspace extracted from the same training fields.
- **Bank/prefix/rank $r$:** continuous learned velocity columns and the retained first columns after a frozen ordering.
- **Lifting:** a fixed field that supplies the nonzero inlet boundary condition before homogeneous corrections are added.
- **Pressure gauge:** the reference level of pressure fixed by the boundary-value problem or an actual nullspace constraint.
- **Supremizer:** a velocity function selected to make a pressure mode visible through the divergence coupling.
- **Inf-sup constant $\beta_r$:** the smallest normalized pressure–velocity coupling in the finite reduced spaces.
- **$p,s$:** pressure and supremizer dimensions; both count toward the unknown and storage budget.
- **$M,m$:** number of weak residual tests and number of volume quadrature points, respectively.
- **Weak residual least squares:** minimization of integrated PDE residuals against test functions; it is not a pointwise residual fit or a Galerkin root solve.
- **Riesz whitening:** scaling residual rows using the fixed function-space Gram matrix so arbitrary test scaling does not change the objective.
- **Representation floor:** least-squares distance to the chosen trial span on a declared integration/output measure, not a realizable PDE solution.
- **Mapped Gauss rule:** weighted integration points transformed through the geometry's element maps.
- **NNLS:** nonnegative least squares, used to fit positive quadrature weights.
- **ECSW:** energy-conserving sampling and weighting; its name does not certify conservation for this inlet-driven problem.
- **Exact tensor control:** full-rule preintegration of every coefficient in the quadratic reduced convection operator.
- **Matched output:** the same fields, probes or forces required from every timed method.
- **Development/validation:** cases allowed for model choice and cases reserved for evaluation after the choice is frozen.
