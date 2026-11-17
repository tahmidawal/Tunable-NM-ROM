# Burgers quadtree NM-ROM experiment plan

Draft protocol, November 17, 2026. The objective is a faster PDE solve at the same physical accuracy and requested fine-grid output, achieved by adding representation capacity only where needed. These are proposed settings and decision rules, not measured results. The user has authorized a separate development session to begin and to expand compute as needed.

This protocol changes the immediate development priority from the October Poisson-first sequence to Burgers. The architecture and numerical cautions in PLAN.md still apply. Poisson may be used as a small algebra/debugging fixture; it is not a prerequisite research campaign. The October proposal remains a historical proposal.

**Execution ownership:** Before implementation or submissions, read [EXECUTION-OWNER.md](EXECUTION-OWNER.md). A desktop task-creation request stalled on November 17, so a separately delegated worker owns execution. A delayed duplicate task must respect that ownership and must not start another campaign.

## 1. Target and what counts as success

Primary problem: the existing two-dimensional scalar viscous Burgers equation on the unit square,

$$u_t+u(u_x+u_y)=\nu\Delta u,\qquad u|_{\partial\Omega}=0,\qquad 0<t\leq0.25.$$

The initial field is prescribed, coefficients are advanced by solving the PDE, and the output is a physical field. Reconstruction from known future snapshots and truth-assisted tree selection are diagnostics only.

Primary accuracy target: at every registered evolved output time, relative physical L2 error at most 1%, normalized by the reference initial-field norm. For case i,

$$e_i=\max_{t\in\{0.05,0.10,0.15,0.20,0.25\}}
\frac{\|u_i(t)-u_{i,\mathrm{ref}}(t)\|_{L^2}}{\|u_{i,\mathrm{ref}}(0)\|_{L^2}}.$$

Require the conservative numerical assessment e_i + epsilon_ref,i <= 0.01 for every test case in a claimed passing panel. Reference uncertainty is a numerical convergence assessment, not a rigorous analytical bound. Report raw error, uncertainty, failures, median, 90th percentile and worst case. A 1% median is not a 1% per-case result. This is agreement with a converged fine reference to the requested tolerance; it does not mean reproducing every fine-solver digit.

The primary output contract is one final field at t=0.25 on exactly 4096 by 4096 nodes including boundaries, x_j=j/4095. Also benchmark an output-heavy contract returning all five evolved fields on that grid. State explicitly which contract each timing uses. Inherited code calls L the number of intervals and returns L+1 nodes; do not silently equate its L=4096 with exactly 4096 output nodes. Internal solver mesh, output grid, and integration points are separate objects.

Primary speed statistic: median across cases of paired total-time ratios T_tree/T_baseline, using per-case medians over repetitions. A useful speedup requires its case-bootstrap 95% upper confidence limit below 1, at least 75% of cases faster, and a 90th-percentile paired ratio <=1.10. A substantial speedup target is median ratio <=0.50. These thresholds do not replace the all-case accuracy requirement. Publish full ratios even when a threshold fails.

First demonstrate 1%; subsequently repeat frozen accuracy-cost curves at 0.5% and 0.1% where references are sufficiently accurate. Tight targets are separate panels, not replacements selected after seeing results.

## 2. Information and timing contract

Primary input: the same physical initial-field array on the requested output grid, already resident on the GPU, plus viscosity and boundary/domain metadata. No method receives future fields, reference norms, or the generative Gaussian parameters during the solve. Every method pays for its own restriction, projection, initialization and input-dependent setup. A descriptor-input panel is optional and must grant the identical descriptor access to all methods.

Primary total time starts from this common input and ends with the requested dense output resident on the GPU. Include all time steps, candidate trials, rejected actions, integration/test updates, nonlinear solves, state transfer, checking, fallback, output evaluation and synchronization. Benchmark host-input/host-output transfer-inclusive time separately. Initial-array generation is common preparation and is reported separately. Ground-truth error scoring is an offline benchmark operation; online acceptance checks remain charged.

For the final-field contract, retain reduced states or solver states at all five times for offline scoring. Avoid accidentally timing five dense decodes while labelling the measurement final-only, or excluding decodes from the five-output contract. Retention overhead must be the same in paired calls or separately quantified.

Report cold first-query time including compilation and required cache construction, and warm repeated-query time separately. Charge new tree-specific compilation/cache misses on the query that causes them. Global reusable caches may be amortized, with their size, preparation cost, and reuse assumptions reported. Report offline training and break-even query counts separately.

## 3. Data families and splits

### A. Existing Burgers family: primary continuity benchmark

Reuse the verified generator semantics: interior Gaussian initial condition with center coordinates uniform in [0.15,0.85], width uniform in [0.05,0.20], amplitude uniform in [0.5,2], and viscosity log-uniform in [0.01,0.1]. Boundary samples are set to zero exactly. Preserve the inherited boundary incompatibility at t=0; do not silently replace it by a tapered Gaussian. Initial projection error and early-time convergence are separate diagnostics.

Reproduce existing global models on exposed historical development cases first. Historical tests already used in earlier reports are development evidence here, not fresh confirmation. Do not open other projects' still-sealed populations.

For new matched-data training, propose 256 training trajectories, 64 development trajectories and 64 untouched final-test trajectories, with independent seeds 2026111701, 2026111702 and 2026111703. Use the inherited sequential parameter-draw implementation and save parameter arrays with hashes. Verify actual parameter disjointness against inherited training/validation data before assigning final-test status. Generate final-test solutions only after the selected methods and evaluation code are frozen.

Use the first 32 training and first 8 development trajectories for the initial batch. Increase to 128/32 and then 256/64 as needed. The order is fixed before viewing results. Sample 11 training times, 0, 0.025, ..., 0.25, with equal trajectory/time weighting; dense internal stepping must be reference-qualified. Correlated frames are never independent test cases.

### B. Localized moving and interacting structures: separate mechanism benchmark

Use the same PDE and boundaries with a smooth boundary-compatible initial field

$$u_0(x,y)=\sin^2(\pi x)\sin^2(\pi y)
\left[b+\sum_{j=1}^{J}a_j\exp\left(-\tfrac12\left[(\xi_j/\sigma_{j,1})^2+(\eta_j/\sigma_{j,2})^2\right]\right)\right],$$

where (xi_j,eta_j) are coordinates rotated by theta_j about center c_j. Draw J=1 or 2 with equal probability; b uniform in [0.1,0.3], a_j uniform in [0.5,1.5], centers uniform in [0.2,0.8]^2, widths log-uniform in [0.03,0.12], theta_j uniform in [0,pi), and viscosity log-uniform in [0.0025,0.02]. Two structures must have center separation in [0.08,0.35], using rejection sampling with recorded draws. This creates candidates for steepening and interaction; inspect development trajectories before claiming that interactions actually occur in the time interval.

Use separate 256/64/64 splits with seeds 2026111711/12/13, the same progressive subsets, and the same output times. Train matched global baselines on this family; an out-of-distribution frozen global bank is not the primary comparator. Begin this family after the family-A reference/solver plumbing passes. It may proceed after an A-localization failure, because the scientific question differs; keep the A failure visible.

Add a separate 24-case stress panel after method freeze: 8 patch-edge cases, 8 dyadic-corner cases, and 8 boundary-near cases. Fix the generator before computing solutions: for edge/corner cases place centers at dyadic coordinates with offsets {-0.01,0,+0.01}; boundary-near centers have one coordinate in {0.05,0.95}. Remaining parameters use the in-family distribution. Publish the exact table and seed 2026111721 before the panel runs. These are structured stress tests, not a representative random cohort. Lower viscosities and 3D are later extensions, not substitutes for failing primary cases.

## 4. Reference and scoring qualification

Use the parent project's space-and-time-refined reference methodology, with source hashes recorded. First validate the inherited solver on smooth manufactured fixtures and reproduce the historical development reference checks. For new cases, start a spatial ladder of 128,256,512 intervals and a time ladder dt=0.0025,0.00125,0.000625; refine space and time separately, then jointly. Continue to 1024 and finer or halve dt when needed. Initial boundary layers require the inherited startup treatment or a separately validated equivalent.

Use at least three asymptotic levels before a Richardson estimate; if convergence is irregular, report it and refine rather than extrapolating a reassuring bound. At the 1% target require reference uncertainty <=0.1% of the initial-field norm on every admitted case. Use <=one tenth of the target for tighter panels. Compare against a separately discretized sine-pseudospectral solver on the fixed 8 development cases and on cases with the largest uncertainty; disagreement prevents reference admission.

Score on increasingly dense independent grids until the change in error is <=0.02 percentage points for the 1% target; use one fiftieth of the target at tighter tolerances. Final confirmation must include the actual requested 4096 by 4096 grid, evaluated in chunks if necessary. An interpolated coarse reference is acceptable only after its extension error is qualified. A 4096-point output alone does not make a reference converged.

Retain all preselected cases in coverage counts. A reference failure is unresolved coverage, not permission to replace a difficult case. No whole-panel success claim while coverage remains unresolved.

## 5. Architecture and shared decoder training

Use an additive frozen-feature decoder with online coefficients:

$$u(x,t)=B_0(x)c_0(t)+\sum_{P\in\mathcal A(t)}w_P(x)B_\theta(T_P^{-1}x)c_P(t).$$

Start with global ranks r0 in {16,32,64,128}, local prefixes k in {4,8,16,32}, tree depths 1 through 4, and an initial total active coefficient cap of 512. A planned extension allows depth 5 and cap 1024 if representation evidence warrants it. Count global rank, local ranks, total active coefficients, stored features, and network parameters separately.

Use fixed tensor-product C2 windows: value one on the cell core, a quintic smoothstep taper to zero over a halo of width alpha times the cell width, alpha=0.25 initially. Boundary-touching columns must vanish on the physical wall through a fixed smooth boundary factor; verify its derivatives. Retain parent columns when adding children; child coefficients start at zero. Never renormalize existing windows. Check overlap and parent-child dependence without replacing physical compact-support columns by globally supported orthogonalized functions.

Initial shared network: coordinates in two dimensions, three tanh hidden layers of width 128, linear output of 32 scalar features. Neural weights are trained offline and frozen during PDE solves. Local coefficients are direct coefficients, not nonlinear latent coordinates. Larger width 256, a small bank per depth, and k_max=64 are explicit alternatives. Material/viscosity conditioning is deferred; sharing does not assume scale invariance of the PDE.

Train on the detail missed by the coarse bank for initialization, then optimize the assembled projection objective. Freeze the coarse bank during local training, but refit its coefficients jointly with local coefficients for every configuration and every prefix. Use physical-volume-weighted field error normalized by the trajectory initial norm. Optional derivative loss is lambda times the similarly normalized squared physical gradient error, with lambda in {0,1e-4,1e-3} on the unit square; report its units and scaling. Include derivatives of windows and coordinate maps. Do not replace this with an unweighted sum over extracted patches.

Sample configurations with equal probability from uniform depth-1/2 trees, random connected parent-child trees, and training-reference-guided trees. Within each class, sample prefixes uniformly and enforce the active coefficient cap. Weight whole trajectories equally. Reference-guided training configurations use training data only and do not grant the online selector reference access.

Initial optimizer proposal: Adam, 20,000 updates, learning rate 1e-3 decaying to 1e-5, batch 8 trajectory-time/configuration pairs, physical integration by chunked reference samples. Evaluate development projection/rollout criteria every 1,000 updates; retain the best eligible checkpoint. Permit 40,000 updates only if development improvement is still material. Use one seed to screen recipes and three independent seeds for finalists. First measure a 200-update throughput pilot to resize batches without changing the objective. Use differentiable stable least-squares fits with explicit rank tolerance, initially 1e-10 relative singular cutoff, and report removed directions; do not quietly add a large ridge bias.

Local POD and windowed polynomials are early controls, not the final neural deliverable. They use the same training data, physical metric, supports, and coefficient fitting machinery. Approximate POD accuracy must be checked rather than assumed optimal.

## 6. PDE solve, more k versus splitting, and time transfer

First reproduce the inherited weak residual and FOM-exact upwind advection inside it. Adaptations of the newer off-mesh path must preserve their documented operator, not silently mix a discrete upwind baseline with a different continuous operator. A later continuous weak-form variant is a labelled method change, independently checked against the converged PDE reference.

Use the existing nonlinear solver as the first controlled implementation. Linear coefficients do not make Burgers linear. Start time steps at 0.005 with candidate halvings to 0.0025 and 0.00125. Add BDF2 only as a separately validated alternative with correct history transfer. Choose the fastest qualifying setting per method on development data. A tenfold tighter nonlinear tolerance and a halved step must each change the returned trajectory by <=0.1% in initial-norm units at the primary target; adapt these checks to one tenth of tighter targets. Log stationarity, iteration budgets and failures; small parameter updates alone are not convergence.

Begin with at least four normalized solve tests per active coefficient as a diagnostic starting point, combining global and local smooth tests. Test-count ratios are not stability proofs. Use a defined test Gram metric and independent integration refinement; local quadrature must resolve the finest features, windows and nonlinear products. Doubling the integration resolution must change the trajectory by <=0.1% at the primary target. Avoid double-counting overlapping integration regions. Jacobian singular diagnostics, directions removed, and failed fine-scale tests are recorded.

Adapt at initialization and every five time steps initially; also compare every step and every ten steps. At an adaptation event:

1. Solve on the current configuration and evaluate multiscale selection probes.
2. Shortlist the two highest-indicator patches. Test the next local prefix and a four-child group with k=4 for each; also test one global-rank increase. This gives at most five trial solves per event. Later individual-child actions are an ablation, not an unbounded combinatorial search.
3. Refit all active coefficients for every candidate with suitably resolving tests. Compare on a common union of selection probes, independently integrated and normalized; its extra cost is charged.
4. Accept only a candidate with at least 10% selection-score reduction and reduction larger than three times the score's integration-change estimate. Among eligible candidates choose the largest benefit per estimated added cost over the next adaptation interval. Fit the cost model on development timings and freeze it. This is a heuristic; final physical accuracy is assessed independently.
5. Allow at most two accepted actions per event. If no action helps, keep the field and record stagnation. If solver/online audit checks fail, use the frozen fallback policy and charge it.

Calibrate an online stop/check threshold on development data so accepted development trajectories have <=0.8% physical error, leaving a margin below 1%. If no probe threshold separates failures reliably, report that limitation and test a fixed conservative tree or stronger checks. Finite probes are not error certificates. The final audit probe set is frozen separately; it cannot drive candidate ranking. If changed after observing results, the affected cases become development evidence.

Carry the tree and coefficients forward between steps; never rebuild from scratch by default. First use refinement only. Then test coarsening after three consecutive low-indicator events. Permit removal only if independent projection changes the current field by <=0.05% of the known input norm and all required history states meet that transfer bound. Recheck physics after transfer. For backward Euler carry the previous physical field or its required moments; for BDF2 transfer both history states consistently. Truth-based optimal trees are labelled representation diagnostics and never timed as deployable selectors.

## 7. Required benchmarks

Select configurations on development data, then freeze one policy per method/target/output contract. Do not choose the best method separately for each test case using truth. All deployable methods receive identical input information and solve the PDE.

| Method | Purpose and starting sweep |
|---|---|
| Existing nonlinear-head NM-ROM, u=B h(z) | Primary incumbent nonlinear manifold. Latent sizes 16,32,64,128 where existing models support them; verify rather than invent checkpoint availability. |
| Existing ordered direct-coefficient bank, u=B_r c | Strong incumbent coordinate-bank solver. r=64,128,256,384,512; preserve existing optimized quadratures and warm starts. |
| Matched-data global neural banks/heads | Retrain the global alternatives on the same new trajectories and physical target as the tree, especially for family B. Distinguish architecture effects from better training data. |
| Nonlinear POD-autoencoder NM-ROM | POD lift dimensions 128,256,512; latent sizes 8,16,32,64; two 128-wide tanh layers in encoder/decoder initially. Solve online by least-squares weak residual on the decoded manifold. Compare a validated hyper-reduced path and a full-evaluation small-problem control; do not handicap it by requiring full-grid work if hyper-reduction is available. Report POD truncation separately. |
| POD ROM | Linear POD trial space with the same weak least-squares solver; r=16,32,64,128,256,512. Add a stable Galerkin implementation when available. Same data and initial projection accounting. |
| Global coarse plus uniform local correction | Same shared decoder, without adaptive allocation. Shows whether selecting locations adds value. |
| Fixed local tree and simple front-based selector | Same neural features and solver; choose patches from initial/current gradient indicators without candidate solves. Measures whether trial-selection overhead is worthwhile. |
| Adaptive local POD and polynomial corrections | Same tree/selection rules. Tests whether neural features improve the local architecture. |
| Tuned classical PDE solvers | Existing upwind solver, central second-order solver and sine-pseudospectral solver. Allow a coarser internal mesh if it meets the target on the common output grid. Charge interpolation/synthesis and input preparation. |

An optional quadratic-manifold ROM is a second-round comparator if the required nonlinear baselines are complete. A direct neural operator is optional and reported separately because it does not perform the same online physics solve. No claim of superiority to all NM-ROM methods follows from this finite suite.

Compare against the fastest eligible NM-ROM chosen on development data, and report each individual comparator too. If a baseline cannot meet 1%, report its accuracy-cost curve and failure count; a speed ratio against that inaccurate setting is not a matched-accuracy win. Keep tree-with-fallback and pure-tree results separate. Fallback may provide a robust deployed method, but it cannot hide failures of the proposed decoder.

## 8. Experiment stages and decisions

| Stage | Concrete work | Evidence to advance |
|---|---|---|
| 0: Reproduce and qualify | Inventory current code/checkpoints; reproduce historical dev8-scale results; qualify references and identical input/output timing. | Correct outputs, physical error scoring and timing witnesses. Establish an eligible baseline or document the extra tuning needed. |
| 1: Representation | Train32/dev8, then train128/dev32 as needed; compare global projection with local POD, polynomial and shared-neural projections. | Seek <=0.5% reconstruction error on at least 90% of development cases plus useful measured assembly/decode headroom. All remaining cases stay in the record. This is a screening target, not final acceptance. |
| 2: Fixed-tree dynamics | Coupled solves on fixed learned/classical spaces; no truth-informed online tree. | Per-case trajectory error, solve-to-projection gap, time/integration checks and stable solves. Determine whether failures are representation or dynamics. |
| 3: Adaptive dynamics | More-k/split/global trials, simple selectors, temporal reuse and transfer; measure overhead. | Full development target and lower total cost, or a clear bottleneck suitable for a bounded repair. |
| 4: Controlled modifications | Screen the listed alternatives on dev32; finalists on dev64 and three training seeds. | A frozen feasible configuration with accuracy and measured cost advantage, including adaptive overhead. |
| 5: Final confirmation | Fresh test64 per family, output grids 512,1024,2048,4096; actual 4096 output is mandatory for the headline. | All-case accuracy qualification plus the registered speed test; otherwise a scoped negative or partial result. |

Stage 1 must not consume the whole effort in classical diagnostics: after the first valid classical control, run at least one shared neural recipe. A poor greedy tree search alone does not disprove the architecture. If a stage fails, allow two distinct, documented repair hypotheses before closing that branch or writing a prospective amendment. Do not silently weaken gates.

Screen at most 12 recipe configurations in the first modification round, changing one group at a time: coarse/local ranks; depth; overlap {0.125,0.25,0.5}; width {128,256}; shared versus depth-specific weights; derivative penalty; adaptation frequency; fixed/front/trial selector; and time integrator. Rank by feasibility first, then measured complete runtime. Keep every tried recipe, including failures. Expand to a second round of at most 12 only when the first round identifies a concrete unresolved bottleneck. Network parameter and rank sweeps are not a full Cartesian product.

For final confirmation select the checkpoint/policy using development outcomes only. Repeat finalist training with three seeds to report sensitivity; do not select a seed on final-test results. If final results motivate a change, preserve them and use a newly registered test cohort for the revised method.

## 9. Compute expansion and stopping

The user explicitly permits compute expansion. Begin with an eight allocated-GPU-hour batch for setup verification, reference pilot, baseline reproduction, and the first representation comparison. This is a starting allocation, not a campaign cap. Reference qualification may consume the batch; do not skip it to reach training.

Plan around cumulative checkpoints of 8,32,80 and 160 GPU-hours, with costs estimated again from observed throughput. These are accounting checkpoints, not automatic spending targets or required permission requests. Expand autonomously when the next finite batch addresses a documented uncertainty and is likely to change the decision. Further reference refinement or a justified confirmation can exceed an estimate, with the reason recorded. No open-ended parameter search until a favorable result appears.

Before each batch record configurations, maximum allocated GPU-hours, expected artifacts and pass/stop criteria. Use jobs of at most four hours initially and at most two concurrent owned GPU jobs; training continuations checkpoint and resume. Do not submit more work than the batch allowance. Count compilation, failed allocations, data generation, training, selection, comparisons and audits. Keep GPU models and CPU-hours separate; GPU-hours across hardware are resource accounting, not a performance comparison.

Two failed repair hypotheses on the same bottleneck trigger a written decision to stop that branch or make a specific protocol amendment. Lack of a cost advantage on the simple family can justify testing the separate localized family, but does not erase the negative result. Full confirmation can legitimately conclude no speed advantage. 3D and unrelated PDE campaigns require a separate scoped plan.

## 10. Timing, evidence and reproducibility

Paired timing uses the same allocation and physical GPU, float64 and highest matrix-product precision, GPU preflight, and burn-in before timed blocks. Randomize balanced method order. Keep five repetitions per case and a second independent timing block for final selected methods. Save every repetition and score the actual returned states from those calls. No cross-job wall-clock ratios, cherry-picked fastest repetitions or truth-based selection of timed cases.

On all final cases measure the primary 4096 final-field contract; also measure the five-field contract and 512/1024/2048 scaling. Store per-case component diagnostics without assuming separately instrumented component times sum to the production timer. Report time for initialization, selection, assembly, nonlinear iterations, transfer, checks, decoding, total and fallback; distinguish directly measured production total from diagnostic decompositions.

Save per-case parameters, reference levels, errors, probe scores, selected actions, active coefficients by time, patch occupancy, solver failures, rejected candidates, cache misses, memory peaks and timing arrays. Record exact software/config/model hashes, train/dev/test identities, device, job ID, precision and cumulative resource ledger. Generate plots and tables from saved arrays. Retain reconstructible timed-output evidence and independent checks on physical fields; checksums alone do not validate the numerical result.

Use one experiment worktree/branch and isolated cluster namespace per writer, following applicable parent operational rules. The QuadtreeNMROM directory currently contains planning artifacts; the execution session must establish versioned source and backups before long jobs. Read other branches as scientific context; never change archived results or other jobs. Preserve and checksum evidence before deleting only owned remote duplicates. No automatic merges.

## 11. Deliverables and first execution batch

Deliver a reproducible experiment runner, frozen configuration/data manifests, initial baseline reproduction, qualified references, classical and neural local decoder checkpoints, NM-ROM comparator checkpoints/configurations, a cost ledger, and an evidence-generated report. The key plots are physical error versus complete runtime, speedup versus output size, patch/rank maps over a trajectory, error maps, and selection overhead versus saved solve work. Report neural improvement and spatial-adaptivity improvement separately.

The first separate session should: read this protocol and applicable operating rules; create its isolated versioned implementation lane; inspect and vendor verified Burgers/reference/baseline sources with provenance; register exact pilot configurations; implement numerical fixtures; run tiny smoke checks; then submit and monitor the initial finite cluster batch. Begin real implementation and experiments rather than returning another general plan. Preserve the 4096 final objective even while using smaller pilot grids.

## 12. Existing evidence to carry forward

- [Architecture plan](PLAN.md) and [reconciled reviews](reviews/2026-10-28/coordinator-assessment.txt): smooth additive supports, coupled solve, correct physical weighting, stability and total-cost accounting.
- [November experiment closeout](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-15-experiment-closeout.md): current Burgers quadrature/reference results, classical comparators and timing limitations. Historical near-1% medians do not establish the all-case target here.
- [Earlier reconstruction protocol](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-27-burgers-reconstruction/DESIGN.md): same-input controls, trajectory-level splits, physical error normalization and reference qualification. Reconstruction is not a trajectory PDE solve.
- Historical shared/separate latent-hierarchy attempts in the canonical [lab log](/home/tahmid/Dev/Tunable-NM-ROM-Claude/LAB-LOG.md) did not pass their cost gates. Locality must earn its overhead rather than assuming that a coarse stage saves time.
- Existing source inspected for problem definition: `worktrees/2026-11-05-burgers-quadrature-plateau/experiments/burgers-quadrature-plateau/vendor/tree/experiments/mr-burgers2d/engines.py` under the parent repository. Verify hashes in the execution manifest.

## Glossary

- NM-ROM: a reduced model using a learned nonlinear solution representation; this campaign also compares the direct-coefficient learned-bank variant.
- Local rank k: number of active coefficients in a patch. Global rank, latent size, stored bank size and network width are different quantities.
- POD: a linear representation fitted to solution snapshots. A POD-autoencoder adds a nonlinear map in POD coordinates.
- Weak least squares: solving coefficients by minimizing tested/integrated PDE equations rather than a pointwise residual.
- Reference: a numerically qualified fine solution; uncertainty records its remaining numerical ambiguity.
- Projection error: best fit with the answer known; a diagnostic lower target for the actual PDE solver, not a deployable solver result.
- Development/test: cases used to select settings versus untouched cases used to confirm the frozen settings.
- Hyper-reduction: reducing the work needed to evaluate reduced equations, typically through selected integration points.
- Paired ratio: tree runtime divided by comparator runtime on the same case and GPU. Below one means faster.
- Fallback: completing a failed reduced solve with an independently validated method, with its full cost charged.
