# B8 fixed-tree neural PDE diagnostic

## Scope and prospective decision

Run one diagnostic-only fixed-tree solve using the retained primary-trained B8 nonlinear child checkpoint `b8childprimary`, on primary development case `dev003` (smooth control). This is an explicit exception to representation-stage admission for debugging the actual embedding-space PDE path. The selected model has not passed all-case representation qualification. A successful smooth trajectory cannot qualify the model, establish narrow-pulse/ridge/two-pulse accuracy, or support superiority over CROM/SNF-ROM.

Reserve at most30 allocated GPU-minutes inside the existing B8 accounting tranche. Use a unique owned job directory, successful GPU preflight, float64/highest, and maximum two owned pending/running jobs. Retain failures and all allocated time. No final-test fields, old-campaign writes, merges or adaptive oracle.

## Fixed configuration

- Driver: `solve_residual_tree.py --model-job b8childprimary --diagnostic --case-label dev003 --layout uniform --nodes 4 --L 16384 --Q 128 --M 512 --dt 0.0005`.
- Frozen B7 parent weights with32 parent variables; four independently adjustable16-variable nonlinear children,96 total variables. Truncate the retained trained tree to its first four children. Fixed windows and network weights throughout the solve.
- Original scalar viscous Burgers equation, viscosity0.005, homogeneous Dirichlet boundaries and final time0.25.
- The physical upwind stencil and discrete diffusion spectrum remain at `L=16384`. `Q=128` specifies weak integration only, not replacement by coarse-grid physics. Use512 smooth test modes, comfortably more than96 variables.
- Crank–Nicolson weighting `theta=0.5`, step0.0005,60 LM iterations per PDE step, stationarity tolerance1e-8 and physical-update tolerance1e-10. Never relax tolerances to make this diagnostic finish.
- Requested outputs at0.05,0.10,0.15,0.20,0.25: actual4096-by4096 device arrays, with retained prediction, reference and error arrays. Stop and record a failed initial fit or failed/nonfinite step rather than silently returning an unconverged trajectory.

## Inputs and truth separation

Initialization samples the common initial-field callable. Case parameters exist only inside that callable; they do not initialize neural embeddings directly. Parent starts use the retained training sampled-field/code library, followed by nonlinear field fitting. Child starts use the initial residual and zero-tangent backtracking, followed by coupled parent/child fitting. No retained development embedding is loaded.

The online evolution uses only its initial field, model, fixed tree and PDE. Load future reference solution coefficients only after the corresponding model output has been produced, solely for post-output scoring. Error is relative physical L2 divided by the current-time reference norm. Record the inherited qualified primary-reference allowance and error-plus-allowance against the unchanged1% physical target. These per-output comparisons remain diagnostic even if all pass.

## Timing and interpretation

Record source/model loading, initial sampling/allocation, feature-cache assembly, cold fitting, explicit compilation, each attempted PDE step and all requested reconstructions. A complete cold-query wall timer includes Python control and mandatory acceptance/status transfers. Exclude only explicitly timed GPU conditioning, reference scoring, archival I/O and the separately labelled optional output host transfer. Persist those exclusions and a timing snapshot boundary so repeated status writes are not counted twice. Also retain the sum of instrumented components as a separately labelled incomplete decomposition, not a substitute for total online wall time.

The weak feature cache currently assembles dense child features once. Packed output restricts child network evaluation to window support; the parent still evaluates every requested output point. Report both facts. Cache assembly, compilation and actual dense output costs remain charged. This single unpaired run cannot establish a speedup or a fair comparator ranking.

## Follow-up decisions

If initialization or stepping fails, diagnose the recorded stationarity/update/residual evidence before another configuration. If the trajectory completes, retain all step statuses and all five physical errors, regardless of accuracy. Passing outputs justify subsequent time-step/integration/tolerance checks, not method admission. Required later evidence includes time-step halving, integration refinement, tighter stopping tolerances, harder primary and expanded cases, full representation qualification, deployable adaptation, and accuracy-matched same-hardware CROM/SNF-ROM comparisons. No automatic advancement claim follows from this diagnostic.
