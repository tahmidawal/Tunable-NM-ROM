# Burgers tree embedding development plan

Version 1 — November 18, 2026. This is the complete protocol for a new execution session. It specifies proposed experiments, not results. The present session writes the plan only; it does not start training, create an execution worker, or resume the earlier campaign.

The objective is to solve a PDE with the same physical accuracy and requested output as a strong NM-ROM, using less total online time. The first problem is scalar two-dimensional viscous Burgers with a compact pulse on a smooth background. The proposed representation has a global embedding and independently adjustable local child embeddings, decoded by shared frozen neural networks.

## 1. The question to answer

With fixed decoder weights, retained parents, unchanged windows and zero-output child codes, adding children contains the old representation as a special case. Therefore the **best achievable reconstruction error cannot increase**. Strict improvement depends on useful new directions; an optimizer can still fail to find the better fit. This nesting statement does not compare separately retrained checkpoints or equal-budget global and local models.

The experiment asks:

> Does adding local child variables reduce error more efficiently than adding global variables, and does that advantage survive the complete PDE solve?

Distinguish three findings:

1. Representation benefit: lower approximation error for a given number of adjustable variables and measured evaluation cost.
2. Physics benefit: the learned representation supports an accurate, stable trajectory solve.
3. Runtime benefit: the complete adaptive method is faster at the same physical error and output contract.

A reconstruction result establishes only the first. Lower latent dimension alone establishes none of the three.

## 2. Scope, authorization and ownership

The user will start a new session using [START-TREE-EMBEDDING-SESSION.md](START-TREE-EMBEDDING-SESSION.md). That launch instruction authorizes the new study. Reading this file alone does not resume any work.

The November 17 campaign remains paused under [EXECUTION-OWNER.md](EXECUTION-OWNER.md). Its assigned worker, worktree, namespace, unfinished code and results must be preserved. Its guard applies to that existing execution lane. A user-launched session for this new protocol should create a separate lane; it does not need to take over the old worker or edit the old guard.

Before implementation, the new session must:

1. Read this plan, its adjacent JSON, the launch instruction, `/home/tahmid/Dev/Tunable-NM-ROM-Claude/LAB-LOG.md`, and applicable `AGENTS.md` and `CLAUDE.md`.
2. Claim this study through an exclusively created directory `/home/tahmid/Dev/QuadtreeNMROM/execution/tree-embeddings-owner.lock/`. Record session identity, UTC time, branch, worktree, namespace and status in `owner.json`. If a claim already exists, inspect it and resolve ownership before creating duplicate work. A stale-looking claim is not permission to overwrite it.
3. Create one new sparse worktree under `/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/YYYY-MM-DD-burgers-tree-embeddings`, with branch `exp/YYYY-MM-DD-burgers-tree-embeddings`. Add a unique suffix if needed.
4. Use the current fork point in the parent's `CLAUDE.md`. At plan creation it is `93f399faad67f94ea2b56fe25f13003143e64694`, with code-only mirror base `5a64899fdcfbeb6cfe5b3371af1a8d8a3d2b568f`. Verify these before creating the worktree; do not branch from frozen `main` by accident.
5. Put implementation under `experiments/burgers-tree-embeddings/` in that new worktree. Use a new cluster namespace `/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-<date>-<unique-id>/`.
6. Copy this protocol and JSON into the lane, record their hashes, and append the new ownership/status to the canonical lab log. The new lane owns implementation writes. Other sessions may read it.

Do not merge, overwrite old checkpoints, clean unrelated jobs, or start 3D work. This plan includes implementation, training, evaluation and bounded repairs for the stated 2D problem. The diagonal ridge, two-pulse family and octree extension are later proposals, not substitutes for an unsuccessful primary experiment.

## 3. Exact PDE and initial conditions

Use

$$u_t+u(u_x+u_y)=\nu\Delta u,\qquad (x,y)\in[0,1]^2,$$

with homogeneous Dirichlet boundaries, viscosity $\nu=0.005$, and $0\leq t\leq0.25$.

Define

$$B(x,y)=\sin^2(\pi x)\sin^2(\pi y).$$

The pulse family is

$$u_0(x,y)=B(x,y)\left[0.15+A\exp\left(-\frac{(x-c_x)^2+(y-c_y)^2}{2\sigma^2}\right)\right].$$

Draw $c_x,c_y$ independently uniformly from $[0.25,0.65]$, $\sigma$ log-uniformly from $[0.035,0.075]$, and $A$ uniformly from $[0.8,1.4]$. The coefficient $A$ is inside the boundary envelope and is not the peak of $u_0$.

The smooth control family is $u_0=aB$, with $a$ uniform on $[0.4,0.8]$. Its purpose is to measure unnecessary adaptation and overhead; do not assume in advance that its evolved fields need no local corrections.

The previously illustrated pulse $(c_x,c_y,\sigma,A)=(0.35,0.35,0.045,1.2)$ and smooth field $0.8B$ are visible development fixtures. They are not unopened test cases.

### Data splits

| Cohort | Trajectories | Composition | Seed | Use |
|---|---:|---|---:|---|
| Training | 64 | 48 pulse, 16 smooth | 2026111801 | Fit weights, features, codes and training-only quadrature |
| Development | 16 | 12 pulse, 4 smooth | 2026111802 | Choose models, thresholds, solver settings and repairs |
| Final test | 64 reserved | 48 pulse, 16 smooth | 2026111803 | Evaluate the frozen method once |

For prefix-stable generation, case index $i$ is smooth iff `i % 4 == 3`. Create its RNG with `numpy.random.default_rng(numpy.random.SeedSequence([cohort_seed, i]))`. For a pulse, draw `cx`, `cy`, `log_sigma`, then `A`, in that order; for a smooth case, draw `a`. Save exact parameter arrays and hashes. Split whole trajectories before extracting patches or choosing snapshots.

Train using 11 snapshots at $t=0,0.025,\ldots,0.25$, with equal trajectory/time weight. The physical trajectory accuracy contract uses $t=0.05,0.10,0.15,0.20,0.25$, with initial reconstruction reported separately. Development representation checks cover all 11 snapshots.

Reserve final-test seeds at startup, but do not generate or inspect their solution fields before model/policy freeze. An initial 16-case final-test block can be evaluated for logistics; keep the model frozen while finishing the remaining 48. A result on only 16 cases must be labelled as such. No tuning based on any of these final outcomes.

Add a separate 12-case development alignment panel: for each $\sigma\in\{0.035,0.055\}$ and $\delta\in\{-0.01,0,0.01\}$, use an edge center $(0.5+\delta,0.35)$ and a corner center $(0.5+\delta,0.5-\delta)$, with $A=1.2$. The 12-case final alignment panel replaces these centers with $(0.375+\delta,0.58)$ and $(0.625+\delta,0.625-\delta)$. Keep stress results separate from the random cohort, but include every case when claiming coverage of that panel.

If sample coverage is the diagnosed limitation, expand training to 128 and development to 32 by the same prefix-stable generator. Register this before generating the additional solutions. The final cohort stays unopened.

## 4. Accuracy and output contracts

For case $i$, use the current-time reference norm:

$$e_i=\max_{t\in\{0.05,0.10,0.15,0.20,0.25\}}\frac{\|\widehat u_i(t)-u_{i,\mathrm{ref}}(t)\|_{L^2}}{\|u_{i,\mathrm{ref}}(t)\|_{L^2}}.$$

The final gate is $e_i+\epsilon_{i,\mathrm{ref}}\leq0.01$ for every case in a claimed passing panel. Here $\epsilon_{i,\mathrm{ref}}$ is an empirical reference-convergence allowance, not a rigorous error certificate. This protocol deliberately uses the current-time norm; the older campaign used the initial-field norm. Preserve old results and never mix their percentages with the new ones without recomputation.

Report maximum pointwise error normalized by the reference maximum magnitude, its location, and signed/absolute spatial error maps. These are companion diagnostics, not an unstated pointwise 1% guarantee. Report all failures, median, 90th percentile and worst case.

The primary online input is a callable for the known initial field, plus viscosity, boundary condition and requested times. Every method may query the same callable; initial sampling and fitting are timed. The tree receives no privileged pulse center/width descriptor. A source-aware or moving-frame method given those descriptors must have its own labelled input contract and matching controls. Add a secondary dense-array input contract if needed; count scan and transfer costs consistently.

The primary output is one device-resident final field at $t=0.25$ on exactly $4096\times4096$ nodes, $x_j=y_j=j/4095$. A second contract returns all five evolved fields. Host transfer and file writing are separate, explicitly timed contracts, never silently excluded from a claim that includes them.

Internal PDE mesh size, quadrature size, latent dimension and output sample count are distinct. The inherited code defines `L` as intervals, so `L=4096` means 4097 nodes including the boundaries. Preserve and label that convention internally while producing exactly 4096 output samples per axis. Rendering a coarse approximation on this output grid does not by itself qualify its accuracy.

## 5. Reference qualification and reusable sources

Reuse verified utilities rather than restarting the old experiments. Read-only source locations under the parent repository include:

- `worktrees/2026-11-17-burgers-quadtree/experiments/burgers-quadtree/continuous.py`: sine-space reference integration; its continuous weak ROM is a distinct operator from the inherited upwind ROM.
- The same directory's `family_b.py`, `local_bank.py`, `c1_family_b.py`, and `c1_b_bank.py`: initial-field, window, derivative and training ideas. The old mixture of one/two anisotropic pulses and varying viscosity is **not** this new dataset.
- The same directory's `vendor/tree/experiments/{mr-burgers2d,separable-decoder,quadrature-study,jcp-references}/`: incumbent kernels and numerical reference utilities.
- `worktrees/2026-11-13-manuscript-rebuttal/paper_revision/main.tex`: method distinctions and comparator limitations.
- `reports/2026-11-17-burgers-quadtree-pilot.md` and `reports/2026-11-18-burgers-continuation.md`: historical findings; do not treat them as results for this new study.

Vendor only needed code into the new lane and record paths, commits, content hashes, modifications and licenses. Do not import mutable scripts from another worktree at runtime. Do not reuse old trained checkpoints as matched-data baselines without retraining.

Generate fresh references for the chosen family using the verified sine-pseudospectral solver. Start with space levels 128/256/512 intervals and time steps $0.000625/0.0003125/0.00015625$. Refine space at fixed fine time step and time at fixed fine space separately. Retain all levels. Continue to larger spaces/smaller steps when convergence is irregular or insufficient.

Use the sum of the last observed spatial change, temporal change and reference-extension/scoring change as a documented empirical allowance, and inspect the preceding refinement differences. Require allowance at most 0.1% in the current-time norm at every evaluated time. This rule estimates convergence; it is not a theorem. Compare with an independently discretized second-order reference on the visible fixtures, all development alignment cases and cases with the largest uncertainty. Disagreement above the same tolerance blocks reference admission and triggers further refinement.

Score on independent increasingly dense grids, including actual 4096 output for finalists. Require an error change below 0.02 percentage points between the final two scoring resolutions and include it in the allowance. Spectral coefficients may be evaluated directly at output points. Any interpolation/extension used by another reference needs a separate accuracy check.

A failed reference case remains in the cohort as unresolved coverage. Do not drop it or replace it with an easier case. Time-stepping, derivative and field reconstruction fixtures must pass before training is interpreted.

## 6. Models to build first

Build one common coordinate-feature and solver interface. Every learned model uses smooth activations, explicit physical boundary handling and float64.

| Model | Representation | Purpose |
|---|---|---|
| Global direct bank | $G_R(x)c$ | Efficient learned global linear-space control |
| Global nonlinear Tunable | $G_R(x)h(z)$ | Primary existing NM-ROM comparator |
| Tree with direct local coefficients | Same global parent plus shared local features times local coefficients | Isolate spatial allocation from nonlinear local compression |
| Tree with nonlinear child embeddings | Same global parent plus shared local features times a small nonlinear head | Test the proposed tree of independent embeddings |

Also evaluate uniform and prescribed-tree allocation using the **same local checkpoint**. These do not need another decoder architecture. Include a training-only POD/polynomial local diagnostic to identify feature-learning failures, but it must not replace the neural experiment.

### Initial architecture

Use a global coordinate bank with $R=128$ outputs, three tanh hidden layers of width 128, and the same coordinate encoding across relevant global arms. Start with coordinates plus sine/cosine features at frequencies 1, 2, 4 and 8; record the convention. Use ordered-prefix training for direct ranks 16/32/64/128. Train nonlinear global heads separately at latent dimensions 16/32/64; a 128-dimensional head is a conditional capacity check, not mandatory compression. Each head has two tanh hidden layers of width 64 and a nonzero linear skip into the bank coefficients.

The tree's parent is the matched-data global nonlinear model with $k_0=16$, shared identically by the two local variants. If it already solves the task efficiently, record that result; the adaptive policy should be able to select no children.

For local features, use one shared coordinate MLP with two tanh hidden layers of width 64 and $R_{\mathrm{local}}=16$ outputs. Inputs are centered normalized patch coordinates and patch scale $\log_2h$. The direct local arm uses properly trained feature prefixes $r_P\in\{4,8,16\}$. The nonlinear local arm starts with $z_P\in\mathbb R^4$ and a shared head with two tanh hidden layers of width 32, mapping into the same 16 feature channels.

Define the local head by

$$h_\psi(z)=Lz+f_\psi(z)-f_\psi(0),$$

with nonzero full-column-rank initialization of $L$. This gives $h_\psi(0)=0$ while allowing nonzero derivatives with respect to child codes. Verify useful independent field directions at zero; initializing every weight and both factors to zero is invalid.

The assembled field is

$$\widehat u_{\mathcal T}(x)=G_0(x)h_0(z_0)+\sum_{P\in\mathcal A(\mathcal T)}w_P(x)b_\theta(\xi_P;h_P)^T h_\psi(z_P).$$

For the direct local arm, replace $h_\psi(z_P)$ with local coefficients. The active set includes retained internal nodes as well as leaves. A root split activates four local codes; a deeper split retains the existing parent and adds four more. Children are independent online variables, even if an optional initializer predicts them from their parent.

Use fixed $C^2$ tensor-product windows: one on a cell core, quintic taper to zero over a halo of 0.25 cell widths, multiplied by a smooth zero-boundary factor. Do not renormalize existing windows when adding children. Start at maximum depth 3; depth 4 is a documented repair if representation evidence warrants it.

Compare total active-variable budgets 32/64/128, counting the global code and every retained local code. Also show root-only 16. Natural tree sizes may fall below a cap; report actual counts. Match cost/error curves, not nominal counts alone. Count stored feature channels, network parameters, test count, quadrature points, padded computations and cache memory separately.

Equal latent dimension does not mean equal feature-span dimension: four nonlinear local variables can act through 16 features. Include the 16-direct-coefficient control to measure whether that nonlinear compression is useful.

## 7. Training the hierarchy

### Parent training

Fit the global bank and heads using training trajectories only. A training-only POD basis may initialize coordinate features; its fitting metric and regression error must be recorded. Optimize per-snapshot codes rather than relying solely on an encoder's output. Validate optimized projection with multiple initializations so an encoder failure is not mistaken for a manifold limitation.

Freeze the selected parent weights for the first local comparison. Keep refitting parent **codes** jointly with child codes. Both local variants use exactly this parent checkpoint.

### Residual initialization

For each training snapshot and current parent approximation compute

$$e_i=u_i^{\mathrm{ref}}-\widehat u_i^{\mathrm{parent}}.$$

Train shared local features on these missing fields. Patch extraction can initialize a local POD teacher, but the deciding objective is the **assembled physical field**, including all overlapping windows. Independent unweighted patch errors are insufficient.

Initialize direct local coefficients by weighted least squares. Then fit the nonlinear local head and snapshot-specific child codes to these corrections, using the physical feature metric rather than arbitrary coefficient Euclidean distance. Do not label this preliminary coefficient fit as an actual PDE result.

### Assembled training and deeper levels

Optimize a loss over snapshots and several valid tree configurations:

$$\mathcal L=\sum_{i,q}\alpha_q\left[\frac{\|\widehat u_{i,q}-u_i^{\mathrm{ref}}\|_{L^2}^2}{\|u_i^{\mathrm{ref}}\|_{L^2}^2}+\lambda\frac{\|\nabla\widehat u_{i,q}-\nabla u_i^{\mathrm{ref}}\|_{L^2}^2}{\|u_i^{\mathrm{ref}}\|_{L^2}^2}\right].$$

Here $q$ identifies a tree/prefix and its own fitted codes. Refit parent and child codes for every configuration; simply truncating a large-code optimum is not a best-fit smaller model. Begin with equal weights across sampled configurations and equal trajectory/time weight. On the nondimensional unit domain start $\lambda=0$; a preregistered derivative-loss repair uses $10^{-4}$, then $10^{-3}$ only if warranted. Include coordinate-map and window derivatives. Physical-volume factors must survive patch normalization.

Use a balanced mixture of uniform depth-1/depth-2 trees, connected random refinements within the variable cap, and training-reference-guided refinements. Half the minibatches should include a shallower truncation of a sampled tree. Score every truncation after code refitting. Train across depths with shared weights; after changing shared weights, recompute/refit parent predictions rather than treating stale residual targets as immutable.

Training-reference-guided allocation is permitted offline. Oracle allocation on development snapshots is only a labelled upper-bound diagnostic. It cannot select the online tree or appear in deployment timing.

### Initial optimization schedule

- Use Adam at $10^{-3}$ with cosine decay to $10^{-5}$ for weights, up to 20,000 updates for the global stage, 20,000 for local feature initialization, and 20,000 for nonlinear-head/assembled fitting. These are ceilings, not compulsory iteration counts.
- Refit codes with damped Gauss–Newton/LM, initially up to 25 iterations per refit, and record stationarity and restart sensitivity. Use small trajectory/time minibatches and pass large arrays as runtime arguments.
- Start physical training integration at 64 by 64 points, with finer patch-aware integration where needed. Validate losses at doubled resolution before trusting them. Use correct weights for any nonuniform sampling.
- Save checkpoints and development metrics every 1,000 weight updates. Select by worst optimized development projection error, with median error as a tie-breaker, using the same evaluation configurations across checkpoints. Early stop after five checks without a 1% relative improvement in that objective; retain the best and final checkpoints and all failures. An unconverged code fit must be diagnosed before declaring a feature-learning plateau.
- Use training seed 2026111851 for the first complete comparison; repeat finalists with 2026111852 and 2026111853. Select the deployment seed/checkpoint using development data only and report seed sensitivity.
- Freeze local features for the first head-compression comparison. Any later joint feature/head fine-tuning is a separately labelled recipe with its corresponding direct-feature control.

At most 12 architecture/training recipes in the first round. Evaluating trained prefixes is not a new architecture recipe, but every fit and its cost must still be logged. Change one diagnostic hypothesis at a time. Two unsuccessful distinct repairs to the same bottleneck trigger a written stop/amendment decision, not a silent threshold change.

Choose repairs from the evidence rather than a full Cartesian sweep:

| Observed limitation | Bounded next change |
|---|---|
| A training-only local POD span works but the neural bank does not | Improve coordinate-feature fitting; try width 128 or 32 local feature outputs as separate recipes |
| Direct local coefficients work but optimized four-dimensional child codes fail | Train an eight-dimensional local head offline, or retain the direct local method and record the nonlinear-compression failure |
| Both local variants need more spatial resolution | Test depth 4; separately consider halo width 0.125 or 0.5 while auditing derivatives and cost |
| Field fits are good but tested operators are poor | Check differentiation/integration first, then the registered derivative-loss repair |
| Different code starts give materially different fits | Repair fitting/initialization and stationarity checks before increasing decoder size |
| Accuracy passes but runtime fails | Profile actual assembly, Jacobian, solve, selection and output costs; change only the measured bottleneck |

Each repair gets its own configuration hash and comparable baseline. An eight-dimensional child head is selected as a different frozen recipe before its evaluation; it does not authorize padding a running four-dimensional nonlinear code.

## 8. Representation and implementation checks

Required checks before interpreting training:

- Zero-initialized children preserve field values and relevant derivatives to float64 numerical tolerance, with old embeddings unchanged.
- Child-code Jacobians at zero are nonzero and sufficiently independent in the measured physical metric. Record singular values and parent-child redundancy.
- Boundary values, support windows, coordinate scaling and AD derivatives pass analytic/finite-difference fixtures away from nonsmooth numerical branches.
- Training, validation and scoring weights recover known integrals. Refining a patch does not multiply its physical weight.
- Direct coefficient fitting agrees with independent QR/SVD reference fits. Conditional orthogonalization in measurement space must not turn compact physical supports into global ones.
- Nonlinear projection is checked with temporal-warm and multiple cold initializations. Report the best found error and stationarity; do not claim a certified global nonlinear minimum.
- Timed field evaluation and Jacobian products return the same numerical values as an untimed reference implementation.

Representation advancement gate for a tree candidate: at most 0.5% relative $L^2$ projection error at **every** development snapshot and development alignment case. This leaves room for numerical/dynamics error but is not an additive guarantee of the 1% trajectory target. Keep global controls and their accuracy-cost curves even if they need more variables or do not meet this representation screening target; their final eligibility is determined by the physical trajectory contract.

Measure field evaluation, code-Jacobian products, derivative/tested-operator discrepancy, peak memory and active-variable count. Compare at equal variable budget and at equal achieved error. Include the cost of evaluating all ancestor contributions and any padded inactive entries.

A tree may advance to fixed-tree PDE diagnostics even without a preliminary microbenchmark speed advantage, to identify a credible bottleneck; label this diagnostic. Do not launch a large architecture sweep without either representation benefit or a specific falsifiable cost hypothesis.

## 9. PDE solve on the learned representation

Start with fixed trees and frozen network weights. Only the embeddings are online unknowns. The global and local models must use the same verified spatial operator, time integrator, input contract and error scoring for the mechanism comparison.

Use the incumbent **tested weak residual**, retaining its exact fine-grid upwind advection stencil and linear diffusion treatment. Do not replace it with pointwise strong-form collocation or a coarse PDE mesh. Quadrature points are locations where fine physics is evaluated; stencil neighbors retain spacing $1/L$.

For illustration, the backward-Euler discrete residual is

$$R_h(Z)=D_{\mathcal T}(Z)-u^n+\Delta t\,[a_h(D_{\mathcal T}(Z))-\nu\Delta_hD_{\mathcal T}(Z)].$$

Minimize its scaled, tested form $\|S\Phi^T R_h(Z)\|_2^2$ with the appropriate quadrature approximation for the nonlinear tested term. Preserve the incumbent normalization/scaling convention and verify it on a small dense fixture. Dense and sampled implementations must agree as integration is refined.

Start $L=1024$ for numerical development, then compare 2048/4096 before claiming the fine-grid result. Start time steps $0.005,0.0025,0.00125$ and refine as needed. All divide the registered output intervals. For a fair decoder comparison, qualify a common operator/time setting before separately optimizing each method's cost to the same accuracy.

Use damped Gauss–Newton/LM with temporal warm starts and parent-child zero transfer. Begin with at most 30 nonlinear iterations, a scaled stationarity tolerance of $10^{-8}$, and a normalized update tolerance of $10^{-10}$; verify returned fields against a solve tightened by 100 times before loosening tolerances for speed. Do not confuse successful termination with physical accuracy.

Any latent trust region must include zero for new children and the previous feasible parent state. Record trust-region activations. Do not silently clip decoded fields to improve reported errors.

Start with test count $M\geq4K$ and quadrature count $m\geq4M$, where $K$ is active code dimension. These are initial heuristics, not convergence evidence. Enrich tests and quadrature independently, include global and localized audit tests, and monitor the normalized tested Jacobian's singular values. A test/quadrature increase that materially changes the field must be resolved before treating a large residual as a decoder-capacity failure.

If global sine tests miss local directions, add a predeclared family of localized smooth tests and normalize their Gram metric. Keep common validation tests when comparing candidate actions; raw sums from different test sets are not comparable. Count test changes, integration changes and cache rebuilds in online cost.

Cache linear tested operators and frozen coordinate-feature values when feasible. The nonlinear head and its Jacobian still depend on embeddings. Preserve exact spatial locality when applying local banks; a dense solve or dense assembly that ignores it can erase the benefit. Do not assume overlapping patches yield independent matrix blocks.

The previously developed continuous-flux/Galerkin operator is an optional separately labelled solver experiment after the primary comparison. Apply any operator change to global controls too, qualify it against physical references, and report that it is a different method. It cannot silently replace the fine-upwind comparison to create an apparent decoder win.

Fixed-tree advancement gate: all development trajectories satisfy the physical 1% gate, with reference allowance; halving the time step and refining integration each change the evaluated solution by at most 0.1% in the target norm. Report projection error, dynamics error and numerical sensitivities separately. If the root/global method is faster and equally accurate, keep that result visible.

## 10. Online adaptation

Begin with a simple explicit policy; do not train a complex selector before the decoder and fixed-tree dynamics work.

At the initial-condition fit and before accepting each time step:

1. Solve on the current tree, keeping topology fixed inside the nonlinear solve.
2. Evaluate independent normalized local and global physics probes using integration distinct from the solve rule. Initial-condition mismatch is also observable at $t=0$.
3. Distinguish incomplete nonlinear convergence or quadrature error from missing representation. Tighten solve/integration first when the diagnostic supports it.
4. Rank candidate leaves by a probe-based estimated error reduction per estimated added cost. Consider the top two locations, one sibling-group split at each, a local-rank increase for the direct arm, and a registered global-capacity fallback. A learned indicator is a later optional recipe trained only on training/development outcomes.
5. Trial each allowed action within a cap of four candidate solves per adaptation cycle. Fit all active embeddings jointly. Include all rejected trials in the timer.
6. Compare on the same validation metric; accept an action only if it reduces that metric by at least 20% and its predicted cost stays within the registered cap. The 20% threshold is an initial development setting, to be calibrated rather than treated as an error theorem.
7. Re-solve the current time step after an accepted refinement. Do not advance an inaccurate step and assume a later split repairs its history. A step retry or time-step reduction is recorded and timed.

Start with at most two accepted adaptation cycles per time step, maximum depth 3 and the selected active-variable cap. If probes remain unacceptable, use the registered validated fallback or mark the solve failed. A successful fallback is a tree-with-fallback result, not a pure-tree success.

A direct-bank prefix can grow by appending zero coefficients. Separately trained nonlinear heads are not automatically nested: padding a code from dimension 16 to 32 is not a valid transfer rule. Initially use a qualified global fallback that recomputes the trajectory from the known initial condition, with all prior failed work charged. A later switch between nonlinear heads requires physical state projection, transfer-error auditing and complete cost accounting. Keep the initial nonlinear local dimension fixed at four.

Initially retain children for the remainder of a trajectory. Test coarsening only after refinement works; it requires field transfer/projection, a changed-space audit, and timed overhead. Growth under the fixed cap is itself a result, especially for a moving pulse.

Calibrate probe thresholds on development data against actual physical error. Include false-negative counts and cases with small probes but large true errors. Probe tests do not certify 1% accuracy in deployment. Final accuracy is established by independent evaluation against qualified references.

The policy must be able to leave the tree at its root. Also compare a prescribed initial-field-based allocation and a uniform allocation from the same checkpoint, so adaptive selection has to justify its cost.

## 11. NM-ROM competitors and controls

The primary competitors are other NM-ROMs. Classical solvers supply reference and efficiency context; they do not replace that comparison.

| Priority | Comparator | Required treatment |
|---|---|---|
| First | Matched-data global Tunable nonlinear head and direct-bank mode | Both need tuning to the physical target, including hyper-reduction and cold start |
| First | Uniform/prescribed local allocation and direct local coefficients | Isolate hierarchy, adaptation and nonlinear child compression |
| Confirmation | CROM | Verify the author's implementation and original dynamics; distinguish faithful reproduction from a CROM-style common-solver ablation |
| Confirmation | SNF-ROM | Use its smooth-neural-field training and native dynamics; qualify spatial/time accuracy independently |
| Confirmation | Shallow masked-autoencoder NM-ROM | Include hyper-reduction; an unmasked dense autoencoder is not a fair substitute |
| Conditional extension | Equation-driven CoLoRA | Keep equation-driven and direct data-driven versions separate; record adaptation costs |

Primary sources: [CROM project and code links](https://crom-pde.github.io/), [SNF-ROM](https://arxiv.org/html/2405.14890v2), [masked autoencoder NM-ROM](https://arxiv.org/abs/2009.11990), [CoLoRA](https://proceedings.mlr.press/v235/berman24b.html).

Train on the same reference trajectories and provide comparable development tuning effort. Respect each method's intended algorithm in end-to-end benchmarks. Common-solver decoder ablations answer a different question and must be labelled. Record code revisions, modifications, tuning budget, convergence failures and reproduction gaps. An unavailable or failed reproduction limits the claim; it is not evidence that the proposed method beats that competitor.

For classical context, include independently tuned coarse-grid, second-order and spectral solvers, and a POD-based ROM. Measure them at the same physical tolerance and output contract. A coarse/fine solver used to manufacture reference data is not automatically a fair speed baseline.

Use the exact translating-pulse control from `discussion/2026-11-18-case-selection/case_specification.json` if spatial localization appears to be the sole source of the gain. Compare with a moving-coordinate representation and charge localization/coordinate-estimation costs. This is a separate linear-advection diagnostic, not Burgers evidence.

## 12. Timing and success criteria

Benchmark the complete query: initial sampling and code fit, tree selection, all accepted/rejected trials, feature and operator assembly, nonlinear solving, integration checks, state transfer, fallback and requested dense reconstruction. Report compiler/JIT work, offline training and reusable cache setup separately, with setup amortization versus number of queries. Include any query-specific compilation in the query cost.

Do not cache only tree subsets observed in development. A precomputed cache must cover a declared finite universe and have its construction/storage charged, or new branches incur measured cache misses. Do not cache whole solutions for benchmark parameters.

Paired timing must run methods on the same physical GPU in the same allocation, with burn-in, synchronized results, balanced randomized method order and saved repetition arrays. Use five repetitions per case plus a second independent block for finalists. Accuracy and timing come from the same returned solver invocation. Instrumented component timings are diagnostics, not a substitute for an uninstrumented total timer.

Choose the fastest development-qualified configuration of each method before final testing. Do not pick a different best method per test case using reference errors. Freeze architecture, training seed/checkpoint, data transforms, integration, stopping rules, tree policy, fallback and output contract.

For the requested objective, all cases in the claimed panel must pass 1% accuracy and the median paired tree/global runtime ratio must be below 1, with the upper endpoint of a trajectory-level bootstrap 95% interval below 1. A substantial improvement target is at least 2 times faster, not an assumption. Report casewise ratios, worst slowdown and the fraction of cases actually faster. Resample trajectories, not correlated snapshots or repetitions as independent observations.

A claim of beating NM-ROM competitors requires eligible reproduced comparator results, not just beating the global direct bank. If only Tunable is successfully compared, state that narrower conclusion.

Repeat at output sizes 512/1024/2048/4096 after the main configuration is frozen. This identifies when dense output dominates even if the reduced solve is cheap. No assertion that producing a dense field is independent of its number of entries.

## 13. Stages and decisions

| Stage | Build/run | Gate and resulting decision |
|---|---|---|
| 0 | New lane, provenance, parameters, reference and algebra fixtures | Trustworthy data, norms, derivatives and ownership before interpreting experiments |
| 1 | Global models and shared direct local features; residual training; nonlinear child head | All-case 0.5% development representation gate; measure error-versus-cost and useful independent child directions |
| 2 | Fixed-tree coupled weak PDE solves | All-case 1% development trajectory gate and numerical refinement checks |
| 3 | Deployable indicator, candidate actions, temporal reuse and no-child option | Accurate complete queries; quantify adaptation cost against prescribed/uniform controls |
| 4 | Bounded repairs, three training seeds, faithful NM-ROM baselines | Freeze a development-qualified method and comparators; otherwise record a scoped negative |
| 5 | Final 64 trajectories plus 12 alignment cases, repeated paired timing, actual 4096 outputs | Accept only the measured accuracy/runtime conclusion supported by all cases |

Do not require every stage to succeed to produce a useful deliverable. A negative result must identify whether the limitation is training, representation, dynamics, selection or complete cost. Retain checkpoints and evidence. Do not close the entire broader architectural idea based on one failed recipe, and do not continue an unfalsifiable search for a positive result.

## 14. Compute and execution rules

When the user launches the new session with this plan, start with a finite reservation of **8 allocated GPU-hours** for Stage 0 and the first representation comparison. This is a starting reservation, not a forecast that every stage will finish within it. Stop submissions at its boundary unless another finite batch is registered first.

The launch instruction allows evidence-based expansion within this 2D study. Use cumulative review checkpoints at 8, 32 and 80 allocated GPU-hours. These are accounting/decision checkpoints, not automatic spending targets. Before each additional batch, record the specific unresolved question, configurations, maximum allocation, expected artifacts and decision gate. Do not increase compute merely because a desired outcome has not appeared. Beyond the planned confirmation effort, present a revised scope instead of drifting into unrelated work.

Use at most two concurrent owned GPU jobs and at most four hours per job. Prefer Tufts A100/H100, through `tufts-login`, on partition `gpu`. Follow current parent rules for any fallback. Local numerical work is limited to sub-minute smoke tests under `jaxrun`. Every real GPU job requires successful GPU preflight, float64, and `JAX_DEFAULT_MATMUL_PRECISION=highest`.

Use `/home/tahmid/Dev/.venv/bin/python` locally and `/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python` on Tufts. Do not install new dependencies unless necessary; use `--no-deps` under the parent rules. Do not assume the login-node Python has the environment.

One unique submit/output directory per job. Check queue state before and after submission. Regenerate data deterministically in cluster jobs or verify an explicitly registered in-namespace cache; never silently use an arbitrary local data directory. Record failed allocations and compilation costs in the ledger. No account-wide cancellations.

Use `git -c gc.auto=0` and propagate that setting through mirror operations. Commit source/configuration regularly and back up through the parent's code-only mirror workflow, not a direct push of archive-heavy experiment branches. Track large data/checkpoints with manifests and checksums. Preserve at least the verified local and designated backup copies before removing only owned remote duplicates. Do not alter prior archives.

Append to the canonical lab log after each completed batch and whenever status changes. Keep the old campaign's pause intact while recording the new study separately. Never claim a process/job is running without its returned ID and verified state.

## 15. Implementation contract and deliverables

The new lane should contain the following modules or equivalent clearly documented interfaces. These filenames are a build specification; they do not yet exist.

| Component | Responsibilities |
|---|---|
| `data.py`, `references.py` | Prefix-stable cohorts, PDE references, independent scoring and convergence allowances |
| `geometry.py`, `decoder.py` | Tree topology, fixed windows, shared features/heads, exact zero-child transfer |
| `fit.py`, `train.py` | Code fitting, residual initialization, assembled multi-depth training and checkpoints |
| `operators.py`, `solve.py` | Verified fine-grid weak physics, cached terms, nonlinear embedding solves |
| `adapt.py` | Fixed-tree solves, probe scores, bounded trial actions and temporal transfer |
| `baselines/` | Matched-data global models, uniform/direct controls and reproduced competitors |
| `evaluate.py`, `benchmark.py` | Same-call accuracy/timing, all-case reporting, frozen-test guard and bootstrap |
| `cluster/`, `manifests/` | Unique jobs, resource ledger, source/data/model hashes and retention |
| `report.py` | Tables and figures generated from saved arrays |

Required figures:

1. Parent reconstruction, parent error, summed child correction and remaining error for the same held-out snapshot.
2. Representation error versus total active variables, and versus field/Jacobian evaluation time.
3. Evolved reference field, actual ROM error and selected patches at several times.
4. Error and active-variable count over time, with accepted/rejected refinement actions.
5. Physical trajectory error versus complete runtime, including NM-ROM competitors.
6. Total runtime versus output resolution, plus measured adaptation and cache overhead.

Save per-case parameters, full fitting/solver status, selected/rejected trees, rank/conditioning diagnostics, trained checkpoints, reference levels, returned-state evidence, raw timing arrays and all resource costs. Produce a generated report with its claim status and a glossary. Do not manually type experimental numbers into prose tables.

The **first execution batch** is: establish ownership and provenance; implement numerical fixtures and the new data generator; qualify visible/development reference data; train the global baseline and first shared local decoder; fit the nonlinear local head; produce the representation comparison and the zero-child/Jacobian tests. Begin real work in the new session rather than returning this plan again. Proceed to later stages only through their gates or a clearly labelled diagnostic exception.

## 16. Decoder literature and limits of the analogy

[ROAD](https://proceedings.mlr.press/v205/zakharov23a.html), [OctField](https://arxiv.org/abs/2111.01067) and [ACORN](https://arxiv.org/abs/2105.02788) provide precedents for hierarchical latent representations or adaptive spatial decoding. Their cited image/shape results do not establish fast PDE dynamics for this proposal. The implementation here retains the Tunable separable structure initially so that spatial features can be cached while small nonlinear heads supply solution-dependent coefficients.

[Lee and Carlberg](https://arxiv.org/abs/1812.08373) provide the nonlinear-manifold projection/minimum-residual framework. The proposed adaptive tree still needs its own conditioning, state-transfer, physics and runtime validation.

## Glossary

- **Embedding / latent code:** the solution-dependent numbers adjusted by the reduced solver.
- **Feature bank:** trained coordinate functions used to construct a field; its output width differs from code dimension.
- **Decoder weights:** parameters trained offline and frozen during the online PDE solve.
- **Child correction:** an additive field supported near a child cell, controlled by independent variables.
- **Projection error:** the best fit found with the reference field available; it is not a PDE trajectory result.
- **Nested representation:** a larger model can reproduce the smaller one exactly by setting added corrections to zero.
- **Weak residual:** PDE equations integrated/tested against smooth functions, with a specified metric.
- **Hyper-reduction:** evaluating the reduced equations through selected integration/stencil locations.
- **Reference allowance:** an empirical estimate of remaining numerical ambiguity in the comparison solution.
- **Oracle:** a diagnostic that uses reference information unavailable during deployment.
- **Development:** cases used to choose settings; they are not final tests.
- **Active variables:** all adjustable parent/child numbers, including retained internal-node codes.
- **Paired runtime ratio:** tree time divided by comparator time on the same case and physical GPU.
- **Fallback:** a validated alternative solve used after the proposed method fails; its full cost is included.
