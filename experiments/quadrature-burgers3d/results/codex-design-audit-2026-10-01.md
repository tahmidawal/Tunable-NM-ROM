**Verdict: do not launch the production experiment yet.** The core off-mesh mathematics is correct. The blockers are reference-job memory, reference acceptance, and an incomplete selection/audit pipeline.

I made no file changes, submitted no jobs, and used no network. Checks were short CPU-only runs with bytecode writing disabled. The model, rules archive, and vendored-file hashes match their recorded hashes.

**1. Off-mesh mathematics — CORRECT.**

The DST uses
\[
\Phi_{jm}=(2/N)^{3/2}\prod_a\sin(\pi k_{ma}j_a/N)
=N^{-3/2}\psi_m(x_j).
\]
Thus, with mesh-node weights \(w_j=N^{-3}\),
\[
N^{3/2}w_j\psi_m(x_j)=\Phi_{jm}.
\]
The coordinates and normalization agree: [common.py:109](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/vendor/burgers3d-span/common.py:109), particularly lines 117–123 and 175–180; [offmesh.py:58](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/offmesh.py:58).

For \(a=P^\top[(Bc)\odot(Dc)]\), the implemented Jacobian is correct:
\[
J_u=P^\top[\operatorname{diag}(Dc)B+\operatorname{diag}(Bc)D].
\]
Consequently, \(J_uc=2a\), and \(J_u\) is linear in \(c\). Extrapolating cached Jacobians therefore gives the exact Jacobian at the extrapolated coefficient vector—not an approximation. Evidence: `offmesh.py:84–92,147–159`.

This establishes normalization consistency; it does **not** make continuum differentiation equal to mesh upwinding.

**2. Solver equivalence and G4 — CORRECT implementation; NEEDS-RESTATEMENT of coverage.**

Comparing [offmesh.py:115](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/offmesh.py:115) with [tables.py:151](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/vendor/burgers3d-retry/tables.py:151), I found no additional numerical change beyond the injected contract and dense predictor branch. The wrapper returns only the query function, whereas the vendor also returns an identity coefficient function; that is an interface difference.

A synthetic CPU rollout produced **exactly identical fields and coefficients**.

G4 is a real comparison against a separate implementation. However, [qpanel.py:224](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/qpanel.py:224) tests one case at the **smallest** configured rank. It records coefficient disagreement but asserts only field disagreement. It does not check iteration counts, reasons, or gradients. Assert both advertised quantities and preferably exercise both deployed ranks. G4 cannot detect a bug shared through the common LM implementation.

**3. G2 — CORRECT; not tautological.**

[qpanel.py:157](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/qpanel.py:157) assembles the mesh-node Jacobian through analytic sine evaluations and `contract_offmesh`; the tensor was built through FFT-based DST projection in `tables.py:74–85`. Those are independent routes for normalization and test ordering.

My synthetic check gave relative disagreement **\(3.56\times10^{-16}\)**. Doubling \(P\) made disagreement **1.0**, demonstrating that an incorrect global scaling fails.

Its limitation is coverage: only the first 32 bank columns, first 64 tests, and one coefficient vector. Errors confined to higher test modes or columns could escape. G3, unlike G2, shares \(P\) between its two sides and cannot independently validate scaling.

**4. Dense sign-upwind Jacobian — CORRECT, with a nonsmoothness qualification.**

[offmesh.py:95](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/offmesh.py:95) computes JVPs of the actual sign-upwind operator along bank columns. The branch condition and stencil match [b3d_common.py:278](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/vendor/paper-b3d/vendor/b3d_common.py:278).

Within a fixed sign pattern, the operator is quadratic and positively homogeneous of degree two, so \(J_uc=2a\). Recomputing candidate Jacobians is necessary because the Jacobian is not globally linear across sign changes.

At exactly zero nodal values, a classical Jacobian need not exist; JAX supplies the derivative of the selected branch. The radial identity still holds. CPU checks on mixed-sign fields gave Jacobian disagreement **\(3.29\times10^{-16}\)** and half-\(Jc\) disagreement **\(3.76\times10^{-16}\)**.

**5. Rho study — CORRECT comparisons; NEEDS-RESTATEMENT of certification strength.**

[qpanel.py:243](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/qpanel.py:243) correctly uses:

- Gauss \(80^3\) with analytic continuum derivatives as the continuum target.
- Sign-upwind advection on every mesh node as the mesh target.
- The same tensor-reached coefficient states, \(R'\), and test set for every rule within a mesh/rank comparison.
- Separate evolved-state and \(k=0\) summaries.

`continuum_adv` correctly truncates the rotation to the coefficient width. Dense mesh rho is identically zero by construction; that is a target identity, not an independent dense correctness test.

Qualifications:

- This certifies quadrature on **tensor trajectories**, not trajectories reached by each off-mesh solver.
- The Gauss convergence statistic is recorded but its acceptance/unresolved logic is absent (`qpanel.py:273–289`).
- Certification rollouts’ convergence reasons and finiteness are not explicitly accepted before their states become the certification set.
- Test spaces change slightly across meshes. The implemented shell completion gives \(M=(2052,1027)\) at 65 nodes and \((2049,1024)\) at 129 nodes for ranks \((512,256)\). Rule comparisons remain fair within each mesh, but cross-mesh changes are not solely quadrature effects.

**6. Full-grid distances and time alignment — CORRECT.**

Since \(G^\top G=LL^\top\),
\[
\|G\delta c\|_2^2=\delta c^\top LL^\top\delta c=\|L^\top\delta c\|_2^2.
\]
For coefficient rows, the code correctly multiplies by \(L\): [qpanel.py:373](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/qpanel.py:373).

`internal` contains the projected initial coefficients followed by every accepted step. With \(\Delta t=0.01\), `[::5]` gives precisely the six output times (`offmesh.py:185–186`, `qpanel.py:348–350`). Normalization uses the full-grid initial reference norm, consistent with this metric.

Two bookkeeping qualifications: dense distances use only overlapping cases, and the converged arm’s self-distance is omitted rather than stored as zero. Selection code must handle that explicitly.

**7. Refined error arithmetic — CORRECT; reference validation — WRONG/incomplete.**

The index formula in [qpanel.py:44](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/qpanel.py:44) maps exactly to \(x=k/64\). CPU checks at \(n=65,129,257,513\) found zero coordinate discrepancy.

`G65` evaluates the same frozen features and rotation at those nodes; the mesh bank has no extra mesh-dependent normalization. Coefficient decoding, FOM extraction, and division by the refined initial norm are correct (`qpanel.py:404–405,534–558`).

However:

- [refjob.py:104](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/refjob.py:104) accepts a cohort based only on finite restricted fields. Newton residuals and iteration caps are recorded but **do not prevent a `.done` marker**.
- The consumer records the archive hash and marker contents but never compares them. It checks seed/count, but not reference mesh, timestep, tolerances, or a reference acceptance record (`qpanel.py:532–537`).
- Initial-field agreement is recorded without an assertion.

A finite, unconverged or incorrectly configured reference can therefore become the primary accuracy target.

**8. Timing — NEEDS-RESTATEMENT; query comparison is broadly fair, enforcement is insufficient.**

I checked the retry implementation at the pinned commit. The A–B–A burn-in, randomized ordering, synchronization, drift and neighbor calculations are substantially copied faithfully: [qpanel.py:422](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/qpanel.py:422).

Off-mesh and tensor queries have the same resident-input/full-output contract, initialization and decoding. Compilation is excluded and query repetitions are saved.

Problems:

- **Empty neighbor groups yield ratio 1.0 and pass.** This is not evidence of no neighbor effect (`478,488`). At large meshes, common grid projection/decoding costs may compress arm timing ratios so that no predecessor satisfies the fourfold threshold.
- Timing failures are recorded without a downstream verdict mechanism that excludes their results.
- Timed outputs are compared only on the restricted lattice, using an absolute tolerance. Full-grid accuracy belongs to a different invocation; sparse agreement does not establish full-output equality.
- Microbenchmarks have no per-arm burn-in, fixed ordering, and discard the 50 repetition arrays (`496–510`).
- Dense arms are excluded from both timing and Jacobian microbenchmarks.

Also, total query cost is **not mesh-independent**: projection and six dense decodes remain grid-dependent. Only the off-mesh nonlinear evaluation has that property.

**9. Selection, criteria and controls — WRONG/incomplete as an executable preregistration.**

The main problems are in [DESIGN.md:150](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/DESIGN.md:150):

- **Unconditional fallback:** if no arm qualifies, selection becomes `lat32768`, even if it is nonfinite, nonstationary, fails rho, or fails the convergence comparison. “No qualifying setting” must be a possible outcome.
- The converged reference requires only agreement between two rollouts. Both could be unconverged or similarly wrong. Require acceptable solver behavior and resolved quadrature evidence.
- “Non-stationary exits” is ambiguous. Vendor LM also returns reason 1 for residual tolerance, 2 for tiny steps, and 3 for damping exhaustion—not just nonfiniteness (`common.py:403–440`). Specify which count toward the 1% limit.
- No selection implementation, `selection.json` validation, or held-out staging guard exists. `controls` is configured but unused. The promised `audit_q.py` and `refine_q.py` are absent.
- **(i)** is an explicit, falsifiable inequality, but no rule handles comparisons below the continuum target’s resolution.
- **(ii)** tests constancy of worst error, not constancy of fields. Different fields can have equal error. Moreover, the selected rule may differ by mesh: the criterion concerns a selection policy, not necessarily one fixed arm.
- **(iii)** defines measurements, not a binary cost-success claim. That is acceptable if reported descriptively.

The must-fail thresholds are numerically capable of firing, but failure is not mathematically guaranteed. Poor quadrature can have small rollout impact in a diffusion-dominated problem; negative weights do not guarantee divergence. An unexpectedly successful control is not automatically proof that the measurement apparatus is non-discriminating.

Similarly, a continuum/upwind gap below 0.116 does not disprove an \(O(h)\) consistency gap. That numerical threshold needs a narrower interpretation.

I found **no intrinsic \(N^{3/2}\)-driven breakdown** in the relative rho or normalized-distance thresholds at \(256^3\); their scaling cancels appropriately. The concrete threshold that can become unable to fire is the empty-group neighbor timing test.

**10. Memory and feasibility — NEEDS-RESTATEMENT for panels; BLOCKING problem for the refined-reference path.**

Calculated storage at 257 nodes, in decimal GB:

| Array | GB |
|---|---:|
| Bank rows, \(512\times255^3\) f64 | 67.92 |
| Full tensor, \(2052\times512^2\) | 4.30 |
| Rank-256 tensor slice | 0.54 |
| All rules’ B/D/P, both deployed ranks | approximately 5.45 |
| 64 resident initial fields | 8.49 |
| Host same-grid REF, 64 cases × six fields | 50.94 |

The panel’s persistent storage is plausible on an H200, and REF alone fits comfortably within `--mem 240G`. But approximately **86.7 GB device storage precedes compilation, FFT workspaces, temporary copies and solver buffers**. Table assembly has substantial transient storage; “~80 GB” is not a peak-memory estimate. Evidence: `tables.py:32–85`, `qpanel.py:144–214,304–310`, [stage.py:70](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/cluster/stage.py:70).

The larger problem is the **513-node reference**:

- `common.make_fom` scans all 100 steps and stacks every full state before selecting outputs: [common.py:251](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/vendor/burgers3d-span/common.py:251).
- That stack alone is **106.75 GB**, before Krylov vectors, FFT workspaces and outputs. It already exceeds an A100-80G.
- The FOM closes over `spec`, a **1.067 GB** array at 513 nodes (`common.py:229–237`). At 257 nodes it is still about 133 MB.

A small CPU JAX trace confirmed both the captured spectral constant and the 100-state scan output. Small compiled CPU memory checks also retained timestep-dependent temporary storage. Exact GPU peak usage remains unmeasured, but H200 feasibility is not established.

The new off-mesh routines generally pass large arrays explicitly; the inherited FOM integration defeats the blanket “all large arrays are explicit arguments” claim. A lane-local equivalent FOM wrapper should retain only required outputs and pass spectral data explicitly, preserving the frozen vendor files.

**11. Cohort integrity — NEEDS-RESTATEMENT, acceptable with restricted claims.**

[DESIGN.md:183](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/DESIGN.md:183) honestly discloses both reuses. The canonical lab log corroborates prior tensor evaluation.

Reusing validation for new rule selection is acceptable. Reusing 923901 supports a **previously evaluated comparison cohort**, with off-mesh choices frozen before its new results. It is not a fresh, globally unopened confirmation set. Replace “opened once” with “evaluated once by this lane after selection is frozen,” retaining the disclosure.

Also, matching cases does not reproduce every published incumbent setting: this lane fixes \(\Delta t=0.01\) and ranks 512/256, whereas the retry selections included \(\Delta t=0.005\) and rank 192 at smaller meshes. Distinguish the same-job tensor baseline from the historical published row.

**12. Other findings — NEEDS-RESTATEMENT, with several substantive omissions.**

- **Reference self-error:** the 257-versus-513 difference on one case is a refinement discrepancy, not a demonstrated error bound for the 513 reference across 96 cases. Both space and time change, permitting cancellation. This needs qualification beside close accuracy comparisons. Evidence: `DESIGN.md:97–104`, `refjob.py:79–94`.
- **Smolyak weight sum:** the claim that every rule sums to one is false. The committed `smol8` has 2559 points, 1016 negative weights, and sum **0.997526143099771**. Boundary removal explains this (`rules.py:86–90`). It is harmless for this boundary-vanishing integrand; **do not renormalize it**.
- **Single shifts:** seed 0 is consistently reproduced, but these results establish performance of those particular shifted/scrambled rules, not variability across shifts. A single Sobol size cannot establish a \(1/m\) convergence rate.
- **Tensor versus FOM:** the tensor is fixed-backward, whereas the FOM is sign-upwind. Negative decoded regions introduce another distinction beyond continuum-versus-mesh differentiation. The recorded tensor mesh rho and minimum field value help diagnose it.
- **Audit recovery:** only sampled states for one designated rule are saved for independent rho checking (`qpanel.py:282–285`). That is insufficient by itself to independently reproduce every rule’s worst rho.
- **Job preparation:** staging correctly verifies committed bytes, uses the GPU partition and explicit venv, sets f64/highest precision, and hashes staged content. But it does not bind held-out jobs to a frozen selection. `make_configs.py:30–36` generates validation/smoke/probe configurations only, despite its “every job config” description.
- **Probe timing:** `refjob.py:87` launches a warm-up without blocking before starting the next timing interval; probe timing also lacks the prescribed burn-in.

**BLOCKERS — must fix before running**

- Replace or safely wrap the FOM’s full-state scan and large captured spectral constant; establish bounded memory for the 513-node reference.
- Require converged, finite reference solves before publishing `.done`; validate reference configuration and checksum when consuming it.
- Resolve the unconditional selection fallback and implement explicit gate/eligibility/control outcomes, including “no selection.”
- Complete the promised audit/refinement and selection-artifact checks before launching the production campaign. Held-out staging must require the frozen selection.

**MAJORS**

- Treat empty timing-neighbor groups as untested; retain microbenchmark repetitions and burn in each block.
- Strengthen timed-output verification beyond a sparse lattice.
- Qualify reference uncertainty, tensor-trajectory rho coverage, reused-cohort claims, and mesh-invariance claims.
- Define all solver exit classifications and unresolved-rho comparisons.
- Separate empirical “bad-rule” expectations from actual fault-injection controls.
- Assert G4 coefficient equivalence and refined initial-field agreement.

**MINORS**

- Correct the Smolyak weight-sum statement without changing its weights.
- Record actual test counts and changing test spaces across meshes.
- State that dense timing is omitted and total queries remain grid-dependent.
- Correct config-generator scope, define selection ties/self-distance, and synchronize probe warm-ups.