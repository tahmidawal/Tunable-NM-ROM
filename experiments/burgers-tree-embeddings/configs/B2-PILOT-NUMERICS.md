# B2: qualify the direct tree against the stronger global control

This prospective configuration is conditional on completed B1 retention/accounting and the independent Q1024 integration check. It does not activate a new allocation ceiling until those records are written. Reserve at most **90 additional allocated GPU-minutes**, including setup, compilation, unsuccessful jobs and smoke failures. No trained recipes are added. Keep at most two owned jobs pending/running, A100-80G float64/highest with GPU preflight, and no job longer than30 minutes. The cumulative eight-hour review remains a decision checkpoint; if B1 plus this finite reservation would exceed eight hours, review the completed evidence before submitting B2.

Evidence motivating this batch: direct24/52-node/1376-variable representation passes every expanded development snapshot; all three exposed CN trajectories pass physical accuracy after increasing weak tests. The44032-test comparison admits22016 tests for that earlier setting, and B1 independently checks Q512 against dense integration and Q1024. The stable training-POD512 plus prescribed Fourier completion has better representation error with960 variables. Its earlier nonlinear-arm global solves are stationary, but those do not qualify the new common numerical setting. The nonlinear child16 checkpoint remains stopped after two failed online optimizer repairs; this batch does not continue that search.

Freeze the same direct24 checkpoint,128-variable parent,52-node initial-callable prescribed tree and stable global960 control. Both use the same known initial-field callable and complete fine sign-upwind weak operator. Start a new, explicitly labelled common baseline at **L8192, CN dt0.00125, Q512,22016 tests**. Choosing the finer spatial baseline follows the visible first-order upwind sensitivity at L1024/2048/4096; it is not an isolated attribution to any one changed setting. The new global basis also prevents treating an old/new baseline difference as a numerical refinement. Use the same three cases dev000,dev003,align002.

From that baseline perform five independent comparisons, each changing only the named numerical control:

| Check | Changed setting | Maximum job reservation |
|---|---|---:|
| Common baseline | As above; no previous-setting comparison |30min|
| Fine spatial operator | L16384; Q and fine-stencil definition otherwise unchanged |25min|
| Weak integration | Q1024; stencil spacing remains1/8192 |30min|
| Weak test enrichment |44032 common tests |25min|
| Time integration | dt0.000625 |30min|
| Solver precision | Outer stationarity1e-10 and update1e-12; inner CG relative1e-8/absolute1e-14; initial-fit CG relative1e-11/absolute1e-15 |25min|

All other solver settings remain fixed, including the30-iteration LM cap,80 inner iterations and the physical objective. Each reservation is additionally bounded by the remaining batch allowance; observed completed job times may justify a shorter reservation, but never an unrecorded larger batch. The sum of these maximum individual reservations is not permission to spend their sum: actual terminal allocations plus full outstanding reservations must fit the90-minute batch guard.

Require every returned step to satisfy the existing complete stopping rule: finite diagnostics, stationarity at tolerance, and either physical update at tolerance or stationarity100 times smaller. Require all five evolved physical errors plus the current-time reference allowance at most1%, and an actual4096-by4096 final device field whose error also passes. For each refinement require current-reference-normalized state change at every specified output time at most0.1%. Retain initial-fit residuals and use the same frozen representation hashes when comparing coefficient states. There is no clipping, truth-assisted selection or reference access during stepping.

If all pilot checks pass, this batch admits only the **three-case numerical setting**. It justifies an explicit subsequent finite batch for all44 development trajectories and their time/integration checks; it does not establish the all-development gate, deployable adaptation, a runtime improvement, or native NM-ROM confirmation. A failed check triggers a written diagnosis. Do not launch an unlimited spatial/test/integration ladder, loosen tolerances or perform a third nonlinear optimizer repair. No final-test solution fields, old campaign writes, automatic merges or unrelated3D work.

Expected artifacts: six complete result records with actual output arrays, full returned-convergence diagnostics, separate field-change comparisons, scheduler accounting including failures, checksum-verified local and designated archive copies, numerical-sensitivity plots and a scoped advancement decision. Pilot clocks remain diagnostic because they charge joint setup; no method-specific speed ratio is inferred from them.
