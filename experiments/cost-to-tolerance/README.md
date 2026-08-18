# Cost to tolerance: the (k, N) surface and the iso-error Pareto frontier

**Question.** How does the cost of the latent solve depend on the latent dimension `k`, and
does that `k` dependence change with mesh resolution `N`? The project had only two 1-D slices
through this surface — cost vs `k` at `N`=64 and cost vs `N` at `k`=8 — crossing at a single
point. This cell measures the full surface for **Poisson-2D** and **Burgers-2D**, and derives
from it the **iso-error Pareto frontier** the committed tables cannot produce.

Everything here reuses the reference ROM implementations. `blat_common.make_weak_ops` /
`blat_common`'s own `lm_step_jit` are the Burgers ROM; `pro_common` + `followup/fu_eq` are the
Poisson ROM. Nothing about the model, the objective, the hyper-reduction or the solver is
re-derived — only the stopping rule, the measurement protocol and the sweep are new.

---

## 1. The two protocol defects this cell exists to fix

**(a) Iterations to termination, not to a tolerance.** The committed cost tables report
iterations to *solver termination* (relative decrease `< 1e-12`, step size, or budget), so
different `k` stop at different accuracy and "cost vs `k`" conflates work with target.

**(b) Cost and accuracy from different runs.** The committed timing used a mean-init solve on
a single source; the committed accuracy used a nearest-init average over 16 sources. The
cross-check column in the timing JSON disagrees with the accuracy table by up to ~7x at some
`k`. Two numbers that never came from the same solve cannot define a Pareto point.

Both are fixed here by construction, and the fix is checked in the harness rather than
asserted (see §3 and the Codex audits).

## 2. The tolerance

The solver stops on the **relative reduction of the objective it is actually minimising**,
measured from the run's own initial guess:

```
||r(z_j)||  <=  tau * ||r(z_0)||          tau in {1e-1, 1e-2, 1e-3}
```

read as *"good enough" / "converged" / "at the manifold floor"*. This needs no oracle and no
held-out field, so it is computable at deployment.

A tolerance on the true discrete residual `||A u - f|| / ||f||` is **unreachable and was
deliberately not used**: at the weak-form solution that quantity is ~2e-1 while the field
error is ~8e-3. That amplification — the discrete Laplacian multiplying grid-scale decoder
error by ~(N-1)^2 — is the entire reason the weak form exists. The achieved
`||A u - f|| / ||f||` (Poisson) and the FOM's own backward-Euler residual along the ROM
trajectory (Burgers) are reported per cell **for reference only**.

Where the rule attaches:

| PDE | objective the tolerance is applied to | reference value `||r(z_0)||` |
|---|---|---|
| Poisson | `|| Lambda^-1 Phi_M^T (A u(z) - f) ||` on the m EQ nodes | at the mean training latent / mean training POD coefficient |
| Burgers, per time step | the same weak residual for step `n` | at that step's own warm start `z_n` |
| Burgers, cold start | the hyper-reduced misfit to the known `u0` on the same m EQ nodes | at the better of the two cold-start inits |

**Censoring.** A cell that stops for any reason other than reaching `tau` is *censored*: it is
reported with `censored: true`, its `censored_frac`, its achieved `||r||/||r(z_0)||` and the
error it did reach. Censored cells are never dropped. A POD arm that converges to its own
exact minimiser while still above `tau` is censored — correctly, because for that model the
tolerance is genuinely unreachable.

A censored cell's cost is the cost of running to *termination*, not the cost to the tolerance,
so admitting it to a "cost at tau" frontier would reintroduce the very defect (a) that this
cell exists to remove. Two frontiers are therefore reported: the **strict** frontier
(uncensored, blow-up-free cells only) is the primary one, and the **as-deployed** frontier
(set the knob, take whatever the solver reaches) is tabulated and drawn dashed beside it.
A Burgers cell in which any trajectory blows up has a non-finite `err_rel_l2` — it can never
enter either frontier — with the finite-only mean retained as a labelled diagnostic.

## 3. What one cell is, and why cost and accuracy come from the same run

A cell is `(pde, method, N, k, M, m, tau)` with `method` in `{coord, pod}`. For each cell the
NNLS-EQ quadrature is refit, one tau-stopped LM is built (tau is a *runtime* argument, so the
three tolerances share one compilation and one kernel), and then, **for every one of the 16
held-out sources**:

```
2 warm-ups  ->  7 timed, block_until_ready-synchronised repetitions
time  = median over the 7 reps, then median over the 16 sources
error = graded on the latent returned by the LAST TIMED REPETITION
```

The solve is deterministic, so every repetition returns the same latent; the reported
`time_ms` and `err_rel_l2` therefore come from the same invocation, the same initial guess and
the same source set.

**Time accounting.** There is exactly **one timed quantity per cell**: the whole online path,
compiled into a single jitted function and synchronised once. `time_ms` in the shared Pareto
schema is that number, because it is what the FOM baseline delivers:

* Poisson: input preprocessing (`Lambda^-1 Phi_M^T f`) + latent solve + decode of the interior
  field. The FOM baseline is the testbed's own jitted CG returning exactly that interior field,
  timed on **every** test source with the same warm-up/median-of-7 protocol and summarised with
  the same across-source statistic (CG iteration counts are source dependent, so timing one
  source would compare different workloads).
* Burgers: hyper-reduced cold start + 50 tau-stopped latent steps + the 51-slice full-field
  decode. The FOM baseline is the testbed's own jitted implicit rollout at batch 1, likewise
  timed on every test trajectory.

### The FOM denominator is a ladder, not a point

The archived Poisson baseline is `cg(..., tol=mp.CG_TOL)` with `CG_TOL = 1e-13`, and the
archived Burgers baseline is a **fixed 8-iteration Newton scan** — in both cases *the setting
that manufactured the truth data*, not one any consumer of the solution would ask for. A
"speedup vs the FOM" against those numbers divides a 1e-2-accurate ROM by a fully converged
solve: two different questions, with an over-convergence factor that is itself mesh dependent,
so it changes the **slope** of an N-ladder and not merely its level.

So the FOM is measured as an **iso-accuracy ladder** on the same axes as the ROMs:

* Poisson: the same `jax.scipy.sparse.linalg.cg` from `x0 = 0` at tolerances
  1e-2 … 1e-13 (the archived 1e-13 is always a rung). Each rung's **true** residual
  `||Au-f||/||f||` is recomputed and recorded, never the recursive one.
* Burgers: the testbed's own implicit rollout at Newton scan lengths 1, 2, 3, 4, 6, 8 — its own
  module-level knob, so the operator is untouched.

Every rung is graded against the exact rung and timed with the ROM's own protocol, so the FOM
appears as a **curve** in the Pareto figure; the exact rung keeps the vertical-line "price of
exactness" role. Every ROM row carries `fom_iso_accuracy_ms` (the cheapest FOM rung that
actually reaches that row's error) alongside the exact-rung speedup, and each mesh reports an
`overconvergence_factor`. The **iso-accuracy rung is the headline denominator**; the 1e-13
number is kept as a clearly labelled secondary.

Field names (`t_fom_testbed_ms`, `fom_testbed_cg_tol`, `fom_testbed_true_rel_res`,
`t_fom_baseline_native_ms`, `overconvergence_factor`) and the matched-tolerance rule — pick the
cheapest rung that actually *delivers* at least what the archived baseline *achieved*, rather
than what it nominally asked for — are shared verbatim with the `rom-warmstart-fom` cell, so
the two cells cannot publish two different Poisson denominators. Both compare
`jax.scipy`-at-`CG_TOL` against `jax.scipy`-at-tau (like-for-like in the solver); an
instrumented CG is ~15% cheaper per iteration and mixing the two understates the correction.

**Anchoring the correction — and it is per-mesh, not global.** A measured over-convergence
ratio may only be applied to the archived ladders if re-timing the *archived function itself*
reproduces the archived number. Each mesh records `anchor_vs_archived`: this run's re-timed
`cg(tol=CG_TOL)` against `pt_n/timing_n.json`'s `fom_cg_s`, with both achieved true residuals.

Two **independent** re-timings — this cell and `rom-warmstart-fom` — agree that the anchor
holds at the fine meshes and *fails at the coarse end*: ~2% at N=128, 3–5% at N=256/512, but
**15% (this cell) and 37% (the peer) at N=32**. A coarse solve is dominated by per-iteration
kernel-launch overhead, which does not transfer between environments or driver versions. So the
rule, shared with that cell, is per-row: **time multiplier where the mesh's anchor is
single-digit percent, iteration multiplier where it is not** — and it is published on every row
of the anchor table rather than asserted once.

**N=32 is uncorrected, not corrected.** There both instruments are weak *simultaneously*: the
anchor fails, and the achieved-residual over-convergence factor is exactly 1.00 because the
1e-13 solve genuinely reaches ~9e-14 at that mesh. A number neither instrument supports is not
an estimate, so N=32 is reported uncorrected.

**Do not read agreement between two within-run multipliers as validation.** A time multiplier
and an iteration multiplier computed inside the *same* run share precisely the environment that
the anchor failure is about, so their agreeing closely is not evidence that either is
transferable. (The peer cell measured 0.865 vs 0.862 at N=32 — the mesh where the anchor fails
worst.)

At N=64 the two re-timings agree with **each other** to 0.3% (7.144 ms here, 7.124 ms there)
while both sit ~8% below the archived 7.786 ms. At that mesh the archive is the outlier, not the
re-timings.

Note also that the archived baseline does not reach its nominal 1e-13 at fine meshes (6.96e-11
at N=512), so two over-convergence factors are reported and they answer different questions: one
against what the archived run *achieved*, one against the tolerance a consumer would actually
*ask for* (1e-6, 1e-8). The first degenerates to 1.00 at coarse meshes; the second is the one
that bites.

### The scalar asymmetry — a rule you can act on

If you are correcting an archived table, the most useful thing here is **not** either
over-convergence factor on its own; it is that the two PDEs behave differently:

> **Poisson: a single scalar is defensible — but only per assumed tolerance.** The factor is
> nearly constant across the mesh ladder (a 2.8% spread), so one number per tolerance changes an
> archived ladder's level and leaves its **slope** intact. It is *not* constant across
> tolerances: it varies by 1.34x from end to end.
>
> **Burgers: a single scalar is not defensible at all.** The factor has a 28% spread across N
> and is genuinely mesh-dependent, so a scalar bends the slope of the N-ladder — the very
> quantity a mesh-independence claim rests on. Correct per mesh, or not at all.

**Always name the reference tolerance next to a Poisson factor.** Measured, at every rung
(`rom-warmstart-fom`'s consolidated run, corroborated by this cell at N=64/128):

| N | vs `tau`=1e-6 | vs `tau`=1e-8 | vs `tau`=1e-10 |
|---|---|---|---|
| 32 | 1.56x | 1.32x | 1.16x |
| 64 | 1.56x | 1.32x | 1.16x |
| 128 | 1.57x | 1.31x | 1.15x |
| 256 | 1.53x | 1.31x | 1.16x |
| 512 | 1.53x | 1.31x | 1.16x |

This cell independently measures **1.56x at N=64 and 1.57x at N=128 against `tau`=1e-6**, on the
nose. An unqualified "the Poisson factor is 1.16x" is the `tau`=1e-10 column quoted as though it
were the whole story — the tightest rung, and the one that flatters the archived numbers most.
That mislabelling is what produced an apparent disagreement between the two cells when there was
never a measurement discrepancy at all. **1e-6 is the column to quote for a deployment**, and it
carries a sign change rather than a rescaling: at 1e-6 the archived Poisson ROM no longer beats
the FOM at N=256 (0.91x, against 1.20x as previously published at 1e-10).

**This cell's factor is a FLOOR, not a point estimate.** The `rom-warmstart-fom` `tau` is a
continuous stopping test on the recomputed true residual, so its 1e-6 solve lands within 1.01x
of target. This cell's ladder must pick the cheapest *rung* clearing a target, which necessarily
overshoots — so the FOM is charged for a little more accuracy than the comparison needs, and the
resulting factor is an underestimate. The half-decade refinement (§ above) shrinks the overshoot
but cannot remove it.

The asymmetry sets the stakes for reconciling definitions between cells: a mismatched Burgers
denominator lands on a ~4x correction, a mismatched Poisson one on a 1.16–1.57x correction
*depending entirely on which tolerance is meant*.

### Cross-checking the two Burgers ladders

This cell and `rom-warmstart-fom` ladder **different knobs** on Burgers, and that is stated
rather than glossed: this cell varies the testbed's own fixed-length `NEWTON_ITERS` over
{1,2,3,4,6,8}; that cell replaces the fixed scan with a tolerance-based Newton loop and ladders
`tau` in `||R(u, u_prev, nu)|| <= tau*||u_prev||`. They are not comparable rung-for-rung.

They do reconcile, because a tolerance-based solve reports how many steps it actually took: that
cell measures **1.97–1.99 Newton steps per time step** at `tau`=1e-6, against the testbed's fixed
8. So the ~4x over-convergence is one fact seen from two directions.

Comparing their time against this cell's integer `NEWTON_ITERS`=2 rung would be sloppy, because
**two known biases act in opposite directions** and would silently partly cancel:

* their solve takes 1.97–1.99 steps per time step, **not** 2.00, so an integer-2 rung here does
  1–3% more work than theirs;
* their loop performs an **outer tolerance test** — one residual evaluation per time step — that
  a fixed-length loop does not, so a few percent of their time has no counterpart here.

Rather than net these into a vague band, the first is *removed* and the second is *reported*.
This cell's ladder {1,2,3,4,6,8} pins a linear cost model `t(k) = a + b·k` (`a` the fixed
per-rollout overhead, `b` the marginal cost of one Newton step per time step), which is evaluated
at their **exact measured step count** instead of at 2. The residual-evaluation asymmetry then
has a known sign: after interpolation this cell should read **slightly low**. A deviation of a
few percent in that direction is expected; a materially larger one, or one in the other
direction, is the signal worth chasing.

**Pinned at `tau`=1e-6, and only there.** The peer's per-step Newton count *rises with tighter
tolerance* — 1.97 at `tau`=1e-6 but up to 2.40 across its full ladder. The interpolation below
may only be evaluated at the `tau`=1e-6 counts, because those are the counts at which the quoted
times were measured. Reusing this interpolation against any other rung of their ladder would
evaluate this cell's cost model at a step count that does not correspond to the time it is
being compared against.

Pre-registered targets (peer's `tau`=1e-6, mean over 4 held-out trajectories, A100 80GB PCIe,
Slurm job 2511371, burn-in on, paired timing, median of 7 with 2 warm-ups), recorded **before**
this cell's Burgers panels landed:

| N | peer ms | peer steps/step | peer achieved residual |
|---|---|---|---|
| 32 | 47.62 | 1.97 | 9.71e-07 |
| 64 | 85.56 | 1.97 | 9.83e-07 |
| 128 | 228.1 | 1.98 | 9.94e-07 |
| 256 | 449.9 | 1.99 | 9.57e-07 |

**The expected sign is only valid if neither cell carries per-step Python overhead — checked in
advance, not assumed.** A Python-side per-step cost on this side would push these numbers
*high*, which is the same direction as a genuine disagreement and the opposite of the
residual-test effect, so it could mask effect 2 or flip the sign outright. Confirmed by reading
the code before the panels ran:

* the peer's chain is a single jitted `lax.scan`, timed median-of-7 with a burn-in;
* this cell's rungs call `burgers2d_film.make_rollout`'s `@jax.jit` `rollout`, whose body is
  `jax.lax.scan(body, U0_b, None, length=NUM_STEPS)` — the **whole 50-step chain is one jitted
  call**, and the timed lambda is a single invocation with a single `block_until_ready`;
* the only Python loop is over **sources**, entirely outside the timed region.

So per-step Python overhead is zero on both sides, effect 2 is the only asymmetry left, and the
expected reading is *slightly low*. Two residual differences are recorded rather than corrected
for: the peer takes the **mean** over 4 trajectories while this cell reports mean and median over
16 (the comparison below uses the **mean**, matching theirs), and the burn-in is 3 s there
against 1.5 s here.

<!-- BEGIN GENERATED: burgers_denominator -->
| N   | peer tau=1e-6 ms | peer achieved res | this cell rungs bracketing it | this cell ms | this cell achieved res | deviation % | verdict                                           |
|-----|------------------|-------------------|-------------------------------|--------------|------------------------|-------------|---------------------------------------------------|
| 32  | 47.6             | 9.71e-07          | 2-2                           | 41.1         | 9.94e-07               | -13.7       | low by a lot                                      |
| 64  | 85.6             | 9.83e-07          | 2-3                           | 65.4-109.0   | 1.4e-05 / 1.1e-10      | --          | INCONCLUSIVE (target falls between integer rungs) |
| 128 | 228.1            | 9.94e-07          | 2-3                           | 194.4-324.5  | 8.2e-05 / 2.0e-09      | --          | INCONCLUSIVE (target falls between integer rungs) |
| 256 | 449.9            | 9.57e-07          | 2-3                           | 354.7-587.1  | 1.6e-04 / 2.5e-09      | --          | INCONCLUSIVE (target falls between integer rungs) |
<!-- END GENERATED: burgers_denominator -->

Agreement confirms the two denominators are the same quantity; disagreement means one of the two
ladders is not measuring what it claims, and that must be resolved before either Burgers speedup
is quoted. The generated table also reports the ladder fit `R2` — if the linear cost model does
not hold, the interpolation is not trustworthy either and the comparison should not be made.

<!-- BEGIN GENERATED: anchor -->
| pde       | N   | archived ms | re-timed (this cell) | re-timed (peer cell) | vs archive: this % | vs archive: peer % | CROSS-INSTRUMENT % | over-conv (achieved) | over-conv (engineering)  | verdict                                                                         |
|-----------|-----|-------------|----------------------|----------------------|--------------------|--------------------|--------------------|----------------------|--------------------------|---------------------------------------------------------------------------------|
| poisson2d | 32  | 5.59        | 3.61                 | 3.55                 | 35.4               | 36.5               | 1.7                | 1.00                 | 1e-06:1.53x  1e-08:1.31x | archive stale -> use the re-timed baseline [nothing to correct: over-conv 1.00] |
| poisson2d | 64  | 7.79        | 7.17                 | 7.12                 | 7.9                | 8.5                | 0.7                | 1.00                 | 1e-06:1.55x  1e-08:1.31x | time [nothing to correct: over-conv 1.00]                                       |
| poisson2d | 128 | 15.14       | 14.74                | 14.86                | 2.7                | 1.9                | 0.8                | 1.00                 | 1e-06:1.53x  1e-08:1.27x | time [nothing to correct: over-conv 1.00]                                       |
| poisson2d | 256 | 31.14       | 29.75                | 29.61                | 4.5                | 4.9                | 0.5                | 1.00                 | 1e-06:1.51x  1e-08:1.30x | time [nothing to correct: over-conv 1.00]                                       |
| poisson2d | 512 | 96.01       | 93.06                | 93.07                | 3.1                | 3.1                | 0.0                | 1.00                 | 1e-06:1.51x  1e-08:1.29x | time [nothing to correct: over-conv 1.00]                                       |
| poisson2d | 128 | 15.14       | 15.05                | 14.86                | 0.7                | 1.9                | 1.3                | 1.00                 | 1e-06:1.57x  1e-08:1.31x | time [nothing to correct: over-conv 1.00]                                       |
| poisson2d | 256 | 31.14       | 31.13                | 29.61                | 0.0                | 4.9                | 5.1                | 1.00                 | 1e-06:1.51x  1e-08:1.30x | time [nothing to correct: over-conv 1.00]                                       |
| poisson2d | 32  | 5.59        | 3.66                 | 3.55                 | 34.6               | 36.5               | 3.1                | 1.00                 | 1e-06:1.52x  1e-08:1.30x | archive stale -> use the re-timed baseline [nothing to correct: over-conv 1.00] |
| poisson2d | 512 | 96.01       | 97.15                | 93.07                | 1.2                | 3.1                | 4.4                | 1.00                 | 1e-06:1.69x  1e-08:1.46x | time [nothing to correct: over-conv 1.00]                                       |
| poisson2d | 64  | 7.79        | 7.18                 | 7.12                 | 7.8                | 8.5                | 0.8                | 1.00                 | 1e-06:1.56x  1e-08:1.31x | time [nothing to correct: over-conv 1.00]                                       |
<!-- END GENERATED: anchor -->

**GPU clock ramp.** The NNLS-EQ fits run on the host for minutes, and a device coming out of
that idle stretch is still ramping — the `rom-warmstart-fom` cell measured a 17% swing at N=512
between identical work timed early versus late, in both an instrumented and a library solver.
Every timed block here is preceded by a short GPU burn-in (`ctol_tol.burn_in`).

The preprocess and decode stages are also measured in isolation. They are value-independent and
`k`-independent, so `time_ms_solve` — the latent-solve component the cost(`k`) question asks
about — is **derived** by subtracting them from the timed pipeline, and every row records that
derivation explicitly. Nothing is a sum of independently measured medians.

The input preprocessing is timed as the **deployable** operation: the selected-mode contraction
`Phi_sel (M' x n_i^2)` against the source, `O(M' N^2)`. The reference splitter's grid branch
performs the full dense sine transform `S^T f S` and then gathers the `M'` retained modes, which
is `O(N^3)` and at N=512 would charge the ROM for work no deployment needs. Both are measured;
the selected-mode result is asserted equal to `pro_common.weak_source_term` to 1e-10 relative.

## 4. Grid and settings

`k` in {2, 4, 6, 8, 12, 16, 24, 32}; `N` in {32, 64, 128, 256, 512} (Poisson) and
{32, 64, 128, 256} (Burgers). `M`=64, `m`=256 fixed, **except `M`=256 whenever `k` >= 32** —
the weak form collapses when `M <= k` (heat `M`=16,`K`=16 gave 9.0e-2 against a 6.3e-3
ceiling; Burgers POD `k`=64,`M`=64 diverged).

**Correction to the grid spec.** The brief said `M`=256 whenever `k` >= 32 *while holding
`m`=256*. That lands exactly on the `m` = `M` corner and violates this project's own operating
rule that `m ~ 4M` is the knee (HANDOFF rule 4). It is a spec error, and the measurement below
shows it is a large one: at `k`=8 with `M` raised to 256 and `m` held at 256, the NNLS-EQ fit
degrades from 1.5e-3 to **7.5e-2** — a 50x worse quadrature — and the solve blows up to 22.5 ms.
The fixed-`k` isolator reproduces that collapse with `k` frozen, so the "k=32 cliff" is the
hyper-reduction corner, **not** the latent dimension.

Accordingly the `m ~ 4M` setting (`m`=1024 at `M`=256) is **primary** at `k` >= 32, and the
`m` = `M` = 256 run is kept as `artefact_m_eq_M`, labelled as an artefact of the original spec.
A spec error of ours must not masquerade as a property of the method at high `k`. Two
supplementary arms keep the decomposition measurable:

* `artefact_m_eq_M` at `k` >= 32 — the original spec's corner;
* an **isolator** at fixed `k`=8 with (`M`, `m`) = (256, 256) — the same change with `k` frozen.

All of these are valid configurations and all enter the Pareto scatter and frontier.

<!-- BEGIN GENERATED: configuration -->
| pde       | k                   | N                 | tau               | M                      | m   | EQ pool cap | EQ snapshots | EQ rows | time_ms =                                         |
|-----------|---------------------|-------------------|-------------------|------------------------|-----|-------------|--------------|---------|---------------------------------------------------|
| burgers2d | 2,4,6,8,12,16,24,32 | 32,64,128,256     | 1e-01,1e-02,1e-03 | 64 (k<32), 256 (k>=32) | 256 | 4096        | 64           | 3072    | cold start + latent rollout + 51-slice decode     |
| poisson2d | 2,4,6,8,12,16,24,32 | 32,64,128,256,512 | 1e-01,1e-02,1e-03 | 64 (k<32), 256 (k>=32) | 256 | 4096        | 64           | 3072    | preprocess + latent solve + interior-field decode |
<!-- END GENERATED: configuration -->

**No retraining.** The coordinate decoder is meshfree, so the existing `k`-ladder checkpoints
(trained at N=64, one per `k`, same data draw — fingerprint-checked) evaluate natively at
every `N`. The NNLS-EQ weights are refit whenever `N`, `M`, the method or `k` changes; the POD
basis and the FOM truth are rebuilt at every mesh.

**POD comparability.** The POD arm uses the same weak objective, the same test modes, the same
NNLS-EQ hyper-reduction (fitted on POD-output snapshots), the same LM solver, the same
tau-stopped cold start, and the same error metric. Because a POD decoder is only defined at
grid nodes, the EQ candidate pool is the **interior grid for both arms**; the headline Poisson
recipe used a meshfree pool for the coordinate decoder, so a `pool_control` arm re-measures the
coordinate cell at `k`=8 with that meshfree pool at every mesh. The exact linear POD minimiser
(one precomputed pseudo-inverse matvec, `pod_direct`) is measured as well — POD is given its
strongest implementation, not a handicapped one.

**EQ candidate-pool cap.** The reference Burgers fit uses every interior grid node as an ECSW
candidate column, which is 260 100 columns at N=512 (a 17 GB design matrix at `M`=64 and 4x
that at `M`=256). This cell caps the pool at 4096 candidates — the Poisson recipe's own pool
size — drawn once from a fixed stream, for both PDEs, both arms and every mesh. At N <= 64 the
interior has 900 / 3844 nodes, i.e. fewer than the cap, so the pool *is* the full interior and
the fit is identical to the reference. The NNLS targets remain the **exact full-grid**
projections at every mesh, and the support padding rule and the (coordinate-whitened) latent
perturbation are the reference's. Per-cell fit quality is reported below.

Because a capped random pool could in principle create an artificial error floor at fine
meshes, a **`cap_control` arm** re-fits `k`=8 for **both** methods with the pool *uncapped*
(every interior node) at every mesh where the design matrix still fits, and re-measures. That
bounds what the cap costs by measurement rather than by assumption.

**Burgers POD training set.** The POD basis is built from **all 512 training trajectories** —
the same training set the coordinate decoder was trained on — retaining every 4th time slice
so the snapshot matrix is 3.5 GB rather than 13.7 GB at N=256. The parameter spread, which is
the axis a POD basis needs, is complete. The projection floor of each `V_k` on the held-out
test set is reported per mesh as an oracle bound on the POD arm.

**Cold start (Burgers).** Both inits are the reference's: the mean t=0 training latent, and the
t=0 latent of the training trajectory whose initial field is nearest to the known `u0`. The
reference performs that nearest search on the full grid *outside* the timed region, which is an
`O(N_train N^2)` online operation and would quietly reintroduce mesh dependence. Here it is
done on the **same m EQ nodes** the ROM already samples — `O(N_train m)`, mesh independent —
and **inside** the timed, jitted cold start.

## 5. How this was run

One job per **(PDE, mesh) panel**, all submitted simultaneously, each in its own directory
under `/cluster/tufts/paralab/tawal01/ctol/`, all requesting the same GPU type (A100). Inside
a panel every timing shares one GPU, so the per-(PDE, N) Pareto frontier — whose dominance is
computed *within* a panel — is valid as measured.

The **scaling figure compares timings across meshes**, which panels on different physical GPUs
cannot support. So after the panels land, `ctol_pick_configs.py` selects, from the panel
*accuracies* (which are GPU-independent), the **whole non-dominated frontier** of every
(PDE, method, N) plus the target-reaching, most-accurate and fastest configurations, and a
single **consolidation job re-times exactly those, sequentially, in one process on one GPU**.
That consolidation run is the only timing source the scaling figure uses; the cross-check
table below puts panel and consolidated timings for the same configurations side by side, so
the size of the cross-GPU hazard is visible rather than assumed away.

The local GB10 box was used only for sub-minute smoke tests of the harness; no reported number
comes from it.

```bash
cd experiments/cost-to-tolerance
./cluster/preflight.sh panels          # MANDATORY: run the real code path locally first
./cluster/make_cells.sh panels && for c in ctol_p_n32 ... ctol_b_n256; do ./cluster/launch.sh $c; done
./cluster/pull.sh
python ctol_pick_configs.py
./cluster/make_cells.sh consolidate && ./cluster/launch.sh ctol_consol_p && ./cluster/launch.sh ctol_consol_b
./cluster/pull.sh
python ctol_tables.py && python ctol_figs.py
```

### Preflight — exercise the code path before submitting

Two cluster jobs in this cell died to defects a 30-second local run would have caught, and
**neither was in the measurement**: an unchunked `jacfwd` in the ceiling that OOMed at 16.87 GiB
50 minutes in, and a `dict() got multiple values for keyword argument 'k'` that killed a job
after 24 s having done all of its arithmetic correctly. Both were in the bookkeeping *around*
the measurement — which is where every defect in this cell has been.

`cluster/preflight.sh <panels|fom|ceiling|recover>` therefore runs the mode's real code path
end-to-end locally at N=32 with 2 sources, and **fails loudly if the JSON is absent, unparseable,
or contains no record of the kind that mode is supposed to write**. Checking that the arithmetic
is right never reaches the record-append path; this does. Run it before every submission.

### Surface integrity

`ctol_tables.py` refuses to build these tables unless every panel finished, every expected
primary cell is present exactly once, and every panel ran on GPU in f64 at matmul precision
`highest`. Its verdict:

<!-- BEGIN GENERATED: integrity -->
**PROVISIONAL -- the surface is incomplete:**

* poisson2d: missing primary cell N=512 k=32 M=256 m=1024 pod tau=1e-01
* poisson2d: missing primary cell N=512 k=32 M=256 m=1024 pod tau=1e-02
* poisson2d: missing primary cell N=512 k=32 M=256 m=1024 pod tau=1e-03

Non-fatal, published rather than hidden:

* poisson2d: panel ctol_p_n512/ctol_poisson_n512.json did not write `complete: true` (crashed or was cancelled); its cells are covered by the union below, otherwise the coverage check would have failed
<!-- END GENERATED: integrity -->

### Provenance

The executed bundle is assembled from four worktrees, so this tree's commit alone does not
identify it. Every panel records the commit and dirty state of all four source trees, the
sha256 of every module and checkpoint it actually loaded, and the staged manifest hash.

<!-- BEGIN GENERATED: provenance -->
| pde       | panel (N)         | commit  | gpu                   | node   | jax_backend | slurm job | matmul precision | f64 | seed | sources | time reps | warm-ups | complete | file                                         |
|-----------|-------------------|---------|-----------------------|--------|-------------|-----------|------------------|-----|------|---------|-----------|----------|----------|----------------------------------------------|
| burgers2d | 32                | 9741ff0 | NVIDIA A100-PCIE-40GB | pax051 | gpu         | 2511809   | highest          | yes | 0    | 16      | 7         | 2        | yes      | ctol_b_n32/ctol_burgers_n32.json             |
| burgers2d | 64                | 9741ff0 | NVIDIA A100 80GB PCIe | pax106 | gpu         | 2511810   | highest          | yes | 0    | 16      | 7         | 2        | yes      | ctol_b_n64/ctol_burgers_n64.json             |
| burgers2d | 128               | 9741ff0 | NVIDIA A100 80GB PCIe | pax106 | gpu         | 2511812   | highest          | yes | 0    | 16      | 7         | 2        | yes      | ctol_b_n128/ctol_burgers_n128.json           |
| burgers2d | 256               | 9741ff0 | NVIDIA A100 80GB PCIe | pax007 | gpu         | 2511813   | highest          | yes | 0    | 16      | 7         | 2        | yes      | ctol_b_n256/ctol_burgers_n256.json           |
| poisson2d | 32,64,128,256,512 | b61ba89 | NVIDIA A100 80GB PCIe | pax007 | gpu         | 2513071   | highest          | yes | 0    | 16      | 7         | 2        | yes      | ctol_ceil_all/ctol_poisson_ceiling_all.json  |
| poisson2d | 32,64,128,256,512 | 68ac248 | NVIDIA A100-PCIE-40GB | pax051 | gpu         | 2512420   | highest          | yes | 0    | 16      | 7         | 2        | yes      | ctol_fom_p/ctol_poisson_fomladder.json       |
| poisson2d | 32,64,128,256,512 | 68ac248 | NVIDIA A100-PCIE-40GB | pax051 | gpu         | 2512420   | highest          | yes | 0    | 16      | 7         | 2        | yes      | ctol_fom_p/ctol_poisson_fomladder.json       |
| poisson2d | 32                | 9741ff0 | NVIDIA A100 80GB PCIe | pax007 | gpu         | 2511802   | highest          | yes | 0    | 16      | 7         | 2        | yes      | ctol_p_n32/ctol_poisson_n32.json             |
| poisson2d | 64                | 9741ff0 | NVIDIA A100 80GB PCIe | pax106 | gpu         | 2511803   | highest          | yes | 0    | 16      | 7         | 2        | yes      | ctol_p_n64/ctol_poisson_n64.json             |
| poisson2d | 128               | 9741ff0 | NVIDIA A100 80GB PCIe | pax106 | gpu         | 2511805   | highest          | yes | 0    | 16      | 7         | 2        | yes      | ctol_p_n128/ctol_poisson_n128.json           |
| poisson2d | 256               | 9741ff0 | NVIDIA A100-PCIE-40GB | pax051 | gpu         | 2511806   | highest          | yes | 0    | 16      | 7         | 2        | yes      | ctol_p_n256/ctol_poisson_n256.json           |
| poisson2d | 512               | 9741ff0 | NVIDIA A100-PCIE-40GB | pax051 | gpu         | 2511808   | highest          | yes | 0    | 16      | 7         | 2        | NO       | ctol_p_n512/ctol_poisson_n512.json           |
| poisson2d | consolidation     | 42d1781 | NVIDIA A100-PCIE-40GB | pax051 | gpu         | 2512661   | highest          | yes | 0    | 16      | 7         | 2        | yes      | ctol_rec_n512/ctol_poisson_n512_recover.json |
<!-- END GENERATED: provenance -->

### FOM baselines

<!-- BEGIN GENERATED: fom -->
| pde       | N   | interior DOF | accuracy knob | FOM ms  | err vs exact | achieved residual | truth-manufacturing |
|-----------|-----|--------------|---------------|---------|--------------|-------------------|---------------------|
| burgers2d | 32  | 900          | Newton=8      | 193.24  | 0.000e+00    | 9.9e-13           | yes                 |
| burgers2d | 32  | 900          | Newton=6      | 141.80  | 0.000e+00    | 9.9e-13           |                     |
| burgers2d | 32  | 900          | Newton=4      | 91.62   | 0.000e+00    | 9.9e-13           |                     |
| burgers2d | 32  | 900          | Newton=3      | 66.26   | 0.000e+00    | 9.9e-13           |                     |
| burgers2d | 32  | 900          | Newton=2      | 42.07   | 1.783e-08    | 9.9e-07           |                     |
| burgers2d | 32  | 900          | Newton=1      | 18.68   | 2.864e-04    | 1.7e-03           |                     |
| burgers2d | 64  | 3844         | Newton=8      | 335.75  | 0.000e+00    | 1.0e-12           | yes                 |
| burgers2d | 64  | 3844         | Newton=6      | 246.36  | 0.000e+00    | 1.0e-12           |                     |
| burgers2d | 64  | 3844         | Newton=4      | 157.80  | 0.000e+00    | 1.0e-12           |                     |
| burgers2d | 64  | 3844         | Newton=3      | 111.48  | 3.463e-13    | 1.1e-10           |                     |
| burgers2d | 64  | 3844         | Newton=2      | 66.23   | 8.697e-08    | 1.4e-05           |                     |
| burgers2d | 64  | 3844         | Newton=1      | 28.77   | 2.617e-04    | 6.2e-03           |                     |
| burgers2d | 128 | 15876        | Newton=8      | 1037.32 | 0.000e+00    | 1.0e-12           | yes                 |
| burgers2d | 128 | 15876        | Newton=6      | 758.68  | 0.000e+00    | 1.0e-12           |                     |
| burgers2d | 128 | 15876        | Newton=4      | 476.54  | 0.000e+00    | 1.0e-12           |                     |
| burgers2d | 128 | 15876        | Newton=3      | 337.56  | 2.264e-12    | 2.0e-09           |                     |
| burgers2d | 128 | 15876        | Newton=2      | 202.43  | 1.714e-07    | 8.2e-05           |                     |
| burgers2d | 128 | 15876        | Newton=1      | 77.54   | 2.586e-04    | 1.9e-02           |                     |
| burgers2d | 256 | 64516        | Newton=8      | 1844.33 | 7.229e-16    | 9.8e-13           | yes                 |
| burgers2d | 256 | 64516        | Newton=6      | 1350.99 | 7.229e-16    | 9.8e-13           |                     |
| burgers2d | 256 | 64516        | Newton=4      | 857.15  | 7.229e-16    | 9.8e-13           |                     |
| burgers2d | 256 | 64516        | Newton=3      | 610.35  | 7.616e-13    | 2.5e-09           |                     |
| burgers2d | 256 | 64516        | Newton=2      | 369.16  | 1.340e-07    | 1.6e-04           |                     |
| burgers2d | 256 | 64516        | Newton=1      | 158.89  | 2.734e-04    | 4.6e-02           |                     |
| poisson2d | 32  | 900          | tol=1e-13     | 3.66    | 0.000e+00    | 1.0e-13           | yes                 |
| poisson2d | 32  | 900          | tol=1e-13     | 3.61    | 0.000e+00    | 1.0e-13           | yes                 |
| poisson2d | 32  | 900          | tol=1e-10     | 3.15    | 8.576e-12    | 9.7e-11           |                     |
| poisson2d | 32  | 900          | tol=1e-10     | 3.13    | 8.576e-12    | 9.7e-11           |                     |
| poisson2d | 32  | 900          | tol=1e-08     | 2.81    | 1.181e-09    | 1.0e-08           |                     |
| poisson2d | 32  | 900          | tol=1e-08     | 2.76    | 1.181e-09    | 1.0e-08           |                     |
| poisson2d | 32  | 900          | tol=1e-06     | 2.40    | 1.419e-07    | 1.0e-06           |                     |
| poisson2d | 32  | 900          | tol=1e-06     | 2.36    | 1.419e-07    | 1.0e-06           |                     |
| poisson2d | 32  | 900          | tol=1e-05     | 2.22    | 1.077e-06    | 9.9e-06           |                     |
| poisson2d | 32  | 900          | tol=1e-05     | 2.18    | 1.077e-06    | 9.9e-06           |                     |
| poisson2d | 32  | 900          | tol=3e-05     | 2.08    | 3.637e-06    | 2.9e-05           |                     |
| poisson2d | 32  | 900          | tol=1e-04     | 1.97    | 1.739e-05    | 1.0e-04           |                     |
| poisson2d | 32  | 900          | tol=1e-04     | 1.91    | 1.739e-05    | 1.0e-04           |                     |
| poisson2d | 32  | 900          | tol=3e-04     | 1.78    | 6.600e-05    | 3.0e-04           |                     |
| poisson2d | 32  | 900          | tol=1e-03     | 1.65    | 2.585e-04    | 1.0e-03           |                     |
| poisson2d | 32  | 900          | tol=1e-03     | 1.61    | 2.585e-04    | 1.0e-03           |                     |
| poisson2d | 32  | 900          | tol=3e-03     | 1.45    | 7.141e-04    | 2.9e-03           |                     |
| poisson2d | 32  | 900          | tol=1e-02     | 1.33    | 2.579e-03    | 9.9e-03           |                     |
| poisson2d | 32  | 900          | tol=1e-02     | 1.29    | 2.579e-03    | 9.9e-03           |                     |
| poisson2d | 32  | 900          | tol=3e-02     | 1.14    | 8.091e-03    | 3.0e-02           |                     |
| poisson2d | 32  | 900          | tol=1e-01     | 0.93    | 5.895e-02    | 1.0e-01           |                     |
| poisson2d | 32  | 900          | tol=1e-01     | 0.90    | 5.895e-02    | 1.0e-01           |                     |
| poisson2d | 32  | 900          | tol=3e-01     | 0.56    | 2.759e-01    | 3.0e-01           |                     |
| poisson2d | 32  | 900          | tol=3e-01     | 0.54    | 2.759e-01    | 3.0e-01           |                     |
| poisson2d | 64  | 3844         | tol=1e-13     | 7.18    | 0.000e+00    | 3.6e-13           | yes                 |
| poisson2d | 64  | 3844         | tol=1e-13     | 7.17    | 0.000e+00    | 3.6e-13           | yes                 |
| poisson2d | 64  | 3844         | tol=1e-10     | 6.22    | 5.890e-12    | 1.0e-10           |                     |
| poisson2d | 64  | 3844         | tol=1e-10     | 6.18    | 5.890e-12    | 1.0e-10           |                     |
| poisson2d | 64  | 3844         | tol=1e-08     | 5.49    | 7.536e-10    | 9.5e-09           |                     |
| poisson2d | 64  | 3844         | tol=1e-08     | 5.49    | 7.536e-10    | 9.5e-09           |                     |
| poisson2d | 64  | 3844         | tol=1e-06     | 4.64    | 1.053e-07    | 1.0e-06           |                     |
| poisson2d | 64  | 3844         | tol=1e-06     | 4.60    | 1.053e-07    | 1.0e-06           |                     |
| poisson2d | 64  | 3844         | tol=1e-05     | 4.24    | 8.294e-07    | 9.8e-06           |                     |
| poisson2d | 64  | 3844         | tol=1e-05     | 4.24    | 8.294e-07    | 9.8e-06           |                     |
| poisson2d | 64  | 3844         | tol=3e-05     | 4.07    | 2.534e-06    | 3.0e-05           |                     |
| poisson2d | 64  | 3844         | tol=1e-04     | 3.73    | 1.183e-05    | 1.0e-04           |                     |
| poisson2d | 64  | 3844         | tol=1e-04     | 3.72    | 1.183e-05    | 1.0e-04           |                     |
| poisson2d | 64  | 3844         | tol=3e-04     | 3.47    | 4.140e-05    | 2.9e-04           |                     |
| poisson2d | 64  | 3844         | tol=1e-03     | 3.21    | 1.732e-04    | 1.0e-03           |                     |
| poisson2d | 64  | 3844         | tol=1e-03     | 3.19    | 1.732e-04    | 1.0e-03           |                     |
| poisson2d | 64  | 3844         | tol=3e-03     | 2.87    | 5.433e-04    | 3.0e-03           |                     |
| poisson2d | 64  | 3844         | tol=1e-02     | 2.52    | 1.743e-03    | 1.0e-02           |                     |
| poisson2d | 64  | 3844         | tol=1e-02     | 2.51    | 1.743e-03    | 1.0e-02           |                     |
| poisson2d | 64  | 3844         | tol=3e-02     | 2.21    | 5.761e-03    | 3.0e-02           |                     |
| poisson2d | 64  | 3844         | tol=1e-01     | 1.80    | 3.235e-02    | 1.0e-01           |                     |
| poisson2d | 64  | 3844         | tol=1e-01     | 1.80    | 3.235e-02    | 1.0e-01           |                     |
| poisson2d | 64  | 3844         | tol=3e-01     | 1.16    | 1.761e-01    | 3.0e-01           |                     |
| poisson2d | 64  | 3844         | tol=3e-01     | 1.16    | 1.761e-01    | 3.0e-01           |                     |
| poisson2d | 128 | 15876        | tol=1e-13     | 15.05   | 0.000e+00    | 2.2e-12           | yes                 |
| poisson2d | 128 | 15876        | tol=1e-13     | 14.74   | 0.000e+00    | 2.2e-12           | yes                 |
| poisson2d | 128 | 15876        | tol=1e-10     | 13.03   | 4.295e-12    | 1.0e-10           |                     |
| poisson2d | 128 | 15876        | tol=1e-10     | 12.96   | 4.295e-12    | 1.0e-10           |                     |
| poisson2d | 128 | 15876        | tol=1e-08     | 11.60   | 5.602e-10    | 9.9e-09           |                     |
| poisson2d | 128 | 15876        | tol=1e-08     | 11.52   | 5.602e-10    | 9.9e-09           |                     |
| poisson2d | 128 | 15876        | tol=1e-06     | 9.61    | 7.746e-08    | 9.9e-07           |                     |
| poisson2d | 128 | 15876        | tol=1e-06     | 9.60    | 7.746e-08    | 9.9e-07           |                     |
| poisson2d | 128 | 15876        | tol=1e-05     | 8.52    | 6.857e-07    | 1.0e-05           |                     |
| poisson2d | 128 | 15876        | tol=1e-05     | 8.51    | 6.857e-07    | 1.0e-05           |                     |
| poisson2d | 128 | 15876        | tol=3e-05     | 8.15    | 1.782e-06    | 3.0e-05           |                     |
| poisson2d | 128 | 15876        | tol=1e-04     | 7.67    | 7.798e-06    | 9.9e-05           |                     |
| poisson2d | 128 | 15876        | tol=1e-04     | 7.67    | 7.798e-06    | 9.9e-05           |                     |
| poisson2d | 128 | 15876        | tol=3e-04     | 7.17    | 2.871e-05    | 3.0e-04           |                     |
| poisson2d | 128 | 15876        | tol=1e-03     | 6.53    | 1.166e-04    | 9.9e-04           |                     |
| poisson2d | 128 | 15876        | tol=1e-03     | 6.51    | 1.166e-04    | 9.9e-04           |                     |
| poisson2d | 128 | 15876        | tol=3e-03     | 5.73    | 4.162e-04    | 3.0e-03           |                     |
| poisson2d | 128 | 15876        | tol=1e-02     | 5.18    | 1.238e-03    | 1.0e-02           |                     |
| poisson2d | 128 | 15876        | tol=1e-02     | 5.17    | 1.238e-03    | 1.0e-02           |                     |
| poisson2d | 128 | 15876        | tol=3e-02     | 4.53    | 4.063e-03    | 3.0e-02           |                     |
| poisson2d | 128 | 15876        | tol=1e-01     | 3.74    | 1.726e-02    | 1.0e-01           |                     |
| poisson2d | 128 | 15876        | tol=1e-01     | 3.74    | 1.726e-02    | 1.0e-01           |                     |
| poisson2d | 128 | 15876        | tol=3e-01     | 2.76    | 1.151e-01    | 3.0e-01           |                     |
| poisson2d | 128 | 15876        | tol=3e-01     | 2.74    | 1.151e-01    | 3.0e-01           |                     |
| poisson2d | 256 | 64516        | tol=1e-13     | 31.13   | 0.000e+00    | 1.2e-11           | yes                 |
| poisson2d | 256 | 64516        | tol=1e-13     | 29.75   | 0.000e+00    | 1.2e-11           | yes                 |
| poisson2d | 256 | 64516        | tol=1e-10     | 26.93   | 3.157e-12    | 1.0e-10           |                     |
| poisson2d | 256 | 64516        | tol=1e-10     | 25.80   | 3.157e-12    | 1.0e-10           |                     |
| poisson2d | 256 | 64516        | tol=1e-08     | 23.88   | 4.102e-10    | 9.9e-09           |                     |
| poisson2d | 256 | 64516        | tol=1e-08     | 22.89   | 4.102e-10    | 9.9e-09           |                     |
| poisson2d | 256 | 64516        | tol=1e-06     | 20.64   | 5.188e-08    | 1.0e-06           |                     |
| poisson2d | 256 | 64516        | tol=1e-06     | 19.73   | 5.188e-08    | 1.0e-06           |                     |
| poisson2d | 256 | 64516        | tol=1e-05     | 18.64   | 4.615e-07    | 1.0e-05           |                     |
| poisson2d | 256 | 64516        | tol=1e-05     | 17.81   | 4.615e-07    | 1.0e-05           |                     |
| poisson2d | 256 | 64516        | tol=3e-05     | 16.91   | 1.264e-06    | 3.0e-05           |                     |
| poisson2d | 256 | 64516        | tol=1e-04     | 16.83   | 5.194e-06    | 1.0e-04           |                     |
| poisson2d | 256 | 64516        | tol=1e-04     | 15.99   | 5.194e-06    | 1.0e-04           |                     |
| poisson2d | 256 | 64516        | tol=3e-04     | 15.05   | 1.958e-05    | 3.0e-04           |                     |
| poisson2d | 256 | 64516        | tol=1e-03     | 14.47   | 8.042e-05    | 1.0e-03           |                     |
| poisson2d | 256 | 64516        | tol=1e-03     | 13.89   | 8.042e-05    | 1.0e-03           |                     |
| poisson2d | 256 | 64516        | tol=3e-03     | 12.35   | 2.923e-04    | 3.0e-03           |                     |
| poisson2d | 256 | 64516        | tol=1e-02     | 11.61   | 8.878e-04    | 1.0e-02           |                     |
| poisson2d | 256 | 64516        | tol=1e-02     | 11.09   | 8.878e-04    | 1.0e-02           |                     |
| poisson2d | 256 | 64516        | tol=3e-02     | 9.49    | 2.979e-03    | 3.0e-02           |                     |
| poisson2d | 256 | 64516        | tol=1e-01     | 8.59    | 1.065e-02    | 1.0e-01           |                     |
| poisson2d | 256 | 64516        | tol=1e-01     | 7.85    | 1.065e-02    | 1.0e-01           |                     |
| poisson2d | 256 | 64516        | tol=3e-01     | 6.85    | 6.581e-02    | 3.0e-01           |                     |
| poisson2d | 256 | 64516        | tol=3e-01     | 6.24    | 6.581e-02    | 3.0e-01           |                     |
| poisson2d | 512 | 260100       | tol=1e-13     | 97.15   | 0.000e+00    | 7.0e-11           | yes                 |
| poisson2d | 512 | 260100       | tol=1e-13     | 93.06   | 0.000e+00    | 7.0e-11           | yes                 |
| poisson2d | 512 | 260100       | tol=1e-10     | 84.39   | 2.290e-12    | 1.2e-10           |                     |
| poisson2d | 512 | 260100       | tol=1e-10     | 80.41   | 2.290e-12    | 1.2e-10           |                     |
| poisson2d | 512 | 260100       | tol=1e-08     | 72.23   | 2.924e-10    | 1.0e-08           |                     |
| poisson2d | 512 | 260100       | tol=1e-08     | 66.70   | 2.924e-10    | 1.0e-08           |                     |
| poisson2d | 512 | 260100       | tol=1e-06     | 61.62   | 3.291e-08    | 1.0e-06           |                     |
| poisson2d | 512 | 260100       | tol=1e-05     | 58.08   | 3.443e-07    | 1.0e-05           |                     |
| poisson2d | 512 | 260100       | tol=1e-06     | 57.38   | 3.291e-08    | 1.0e-06           |                     |
| poisson2d | 512 | 260100       | tol=1e-05     | 55.57   | 3.443e-07    | 1.0e-05           |                     |
| poisson2d | 512 | 260100       | tol=3e-05     | 53.17   | 9.381e-07    | 3.0e-05           |                     |
| poisson2d | 512 | 260100       | tol=1e-04     | 52.73   | 3.356e-06    | 1.0e-04           |                     |
| poisson2d | 512 | 260100       | tol=1e-04     | 50.28   | 3.356e-06    | 1.0e-04           |                     |
| poisson2d | 512 | 260100       | tol=3e-04     | 47.35   | 1.308e-05    | 3.0e-04           |                     |
| poisson2d | 512 | 260100       | tol=1e-03     | 45.72   | 5.396e-05    | 1.0e-03           |                     |
| poisson2d | 512 | 260100       | tol=1e-03     | 43.68   | 5.396e-05    | 1.0e-03           |                     |
| poisson2d | 512 | 260100       | tol=3e-03     | 39.18   | 1.951e-04    | 3.0e-03           |                     |
| poisson2d | 512 | 260100       | tol=1e-02     | 36.49   | 6.526e-04    | 1.0e-02           |                     |
| poisson2d | 512 | 260100       | tol=1e-02     | 34.98   | 6.526e-04    | 1.0e-02           |                     |
| poisson2d | 512 | 260100       | tol=3e-02     | 31.04   | 1.965e-03    | 3.0e-02           |                     |
| poisson2d | 512 | 260100       | tol=1e-01     | 27.38   | 7.104e-03    | 1.0e-01           |                     |
| poisson2d | 512 | 260100       | tol=1e-01     | 26.54   | 7.104e-03    | 1.0e-01           |                     |
| poisson2d | 512 | 260100       | tol=3e-01     | 22.06   | 3.677e-02    | 3.0e-01           |                     |
| poisson2d | 512 | 260100       | tol=3e-01     | 20.36   | 3.677e-02    | 3.0e-01           |                     |
<!-- END GENERATED: fom -->

---

## 6. Results

### 6.1 Cost of the latent solve vs k, overlaid across meshes

Figures: `figs/ctol_cost_vs_k_poisson2d.{png,pdf}`, `figs/ctol_cost_vs_k_burgers2d.{png,pdf}`.
The bottom row of each figure normalises every curve at `k`=8: if the `k` dependence is
mesh-independent the normalised curves collapse onto one another.

**Poisson, latent solve (ms), tau = 1e-2**

<!-- BEGIN GENERATED: cost_k_poisson2d_1e-02 -->
| method | N   | k=2  | k=4       | k=6  | k=8       | k=12 | k=16      | k=24 | k=32 |
|--------|-----|------|-----------|------|-----------|------|-----------|------|------|
| coord  | 32  | 7.16 | 3.07      | 3.29 | 2.68      | 3.00 | 3.01      | 3.66 | 9.30 |
| coord  | 64  | 6.73 | 3.01      | 3.03 | 2.60      | 3.03 | 3.08      | 3.37 | 9.92 |
| coord  | 128 | 6.60 | 3.19      | 3.25 | 2.83      | 3.23 | 3.09      | 3.49 | 9.32 |
| coord  | 256 | 6.13 | 2.21      | 2.46 | 1.83      | 2.39 | 2.38      | 2.89 | 8.34 |
| coord  | 512 | 4.08 | below res | 1.73 | below res | 1.51 | below res | 1.50 | 7.83 |
| pod    | 32  | 0.88 | 1.12      | 2.41 | 0.86      | 1.03 | 1.09      | 0.93 | 0.91 |
| pod    | 64  | 2.27 | 0.55      | 1.26 | 1.91      | 1.13 | 0.97      | 0.93 | 0.57 |
| pod    | 128 | 1.81 | 1.43      | 0.94 | 0.97      | 0.85 | 1.17      | 0.95 | 0.64 |
| pod    | 256 | 1.51 | 1.01      | 2.36 | 1.07      | 0.85 | 1.09      | 1.26 | 0.95 |
| pod    | 512 | 0.68 | 0.82      | 1.15 | 0.92      | 0.81 | 0.77      | 1.04 | --   |
<!-- END GENERATED: cost_k_poisson2d_1e-02 -->

**Burgers, latent solve (ms), tau = 1e-2**

<!-- BEGIN GENERATED: cost_k_burgers2d_1e-02 -->
| method | N   | k=2    | k=4    | k=6    | k=8    | k=12   | k=16   | k=24    | k=32    |
|--------|-----|--------|--------|--------|--------|--------|--------|---------|---------|
| coord  | 32  | 239.02 | 227.91 | 266.72 | 329.24 | 438.87 | 614.45 | 1168.09 | 6097.61 |
| coord  | 64  | 223.66 | 219.11 | 254.22 | 305.86 | 424.62 | 582.02 | 1014.81 | 6408.55 |
| coord  | 128 | 228.36 | 227.01 | 257.66 | 301.02 | 425.48 | 578.19 | 1114.94 | 6214.97 |
| coord  | 256 | 254.08 | 271.16 | 301.42 | 351.67 | 500.51 | 655.26 | 1163.00 | 7232.79 |
| pod    | 32  | 34.08  | 32.93  | 33.59  | 30.83  | 37.61  | 35.83  | 37.79   | 45.32   |
| pod    | 64  | 32.64  | 32.29  | 34.40  | 32.16  | 33.24  | 34.14  | 39.72   | 45.12   |
| pod    | 128 | 34.47  | 31.40  | 33.54  | 30.98  | 37.72  | 35.52  | 38.46   | 47.57   |
| pod    | 256 | 33.22  | 31.70  | 33.42  | 31.96  | 35.73  | 36.54  | 44.60   | 44.92   |
<!-- END GENERATED: cost_k_burgers2d_1e-02 -->

**Shape of the k dependence, normalised at k=8** (a row of all-equal numbers across `N` is
mesh independence)

<!-- BEGIN GENERATED: kshape_poisson2d -->
_normalised at k=8; quantity: latent solve (derived)_

| tau   | method | N   | k=2  | k=4  | k=6  | k=8  | k=12      | k=16      | k=24      | k=32 |
|-------|--------|-----|------|------|------|------|-----------|-----------|-----------|------|
| 1e-01 | coord  | 32  | 4.08 | 0.82 | 1.08 | 1.00 | 1.12      | 1.23      | 1.32      | 3.43 |
| 1e-01 | coord  | 64  | 3.63 | 0.81 | 1.04 | 1.00 | 1.01      | 1.17      | 1.22      | 3.34 |
| 1e-01 | coord  | 128 | 3.52 | 0.82 | 1.04 | 1.00 | 1.01      | 1.20      | 1.29      | 3.10 |
| 1e-01 | coord  | 256 | 6.23 | 0.71 | 1.12 | 1.00 | 1.04      | 1.32      | 1.46      | 5.51 |
| 1e-01 | pod    | 32  | 1.43 | 1.61 | 1.26 | 1.00 | 0.21      | 0.21      | 0.24      | 0.29 |
| 1e-01 | pod    | 64  | 1.23 | 0.21 | 0.47 | 1.00 | 0.08      | 0.08      | 0.09      | 0.09 |
| 1e-01 | pod    | 128 | 2.85 | 2.02 | 1.46 | 1.00 | 0.31      | 0.28      | 0.29      | 0.35 |
| 1e-01 | pod    | 256 | 2.44 | 1.38 | 1.73 | 1.00 | 0.27      | 0.18      | 0.21      | 0.54 |
| 1e-01 | pod    | 512 | 1.62 | 1.69 | 2.28 | 1.00 | below res | below res | below res | --   |
| 1e-02 | coord  | 32  | 2.67 | 1.15 | 1.23 | 1.00 | 1.12      | 1.12      | 1.36      | 3.46 |
| 1e-02 | coord  | 64  | 2.59 | 1.16 | 1.16 | 1.00 | 1.17      | 1.19      | 1.30      | 3.81 |
| 1e-02 | coord  | 128 | 2.33 | 1.13 | 1.15 | 1.00 | 1.14      | 1.09      | 1.24      | 3.30 |
| 1e-02 | coord  | 256 | 3.35 | 1.21 | 1.34 | 1.00 | 1.31      | 1.30      | 1.58      | 4.56 |
| 1e-02 | pod    | 32  | 1.03 | 1.30 | 2.81 | 1.00 | 1.20      | 1.27      | 1.08      | 1.06 |
| 1e-02 | pod    | 64  | 1.19 | 0.29 | 0.66 | 1.00 | 0.59      | 0.51      | 0.49      | 0.30 |
| 1e-02 | pod    | 128 | 1.86 | 1.47 | 0.97 | 1.00 | 0.87      | 1.20      | 0.98      | 0.65 |
| 1e-02 | pod    | 256 | 1.41 | 0.95 | 2.21 | 1.00 | 0.80      | 1.02      | 1.18      | 0.88 |
| 1e-02 | pod    | 512 | 0.74 | 0.89 | 1.24 | 1.00 | 0.88      | 0.84      | 1.13      | --   |
| 1e-03 | coord  | 32  | 0.81 | 0.46 | 0.85 | 1.00 | 1.38      | 1.26      | 2.05      | 3.86 |
| 1e-03 | coord  | 64  | 0.93 | 0.54 | 1.12 | 1.00 | 1.53      | 1.41      | 2.23      | 5.07 |
| 1e-03 | coord  | 128 | 0.79 | 0.46 | 0.89 | 1.00 | 1.33      | 1.25      | 1.84      | 4.39 |
| 1e-03 | coord  | 256 | 0.96 | 0.54 | 1.02 | 1.00 | 1.66      | 1.45      | 2.63      | 5.79 |
| 1e-03 | coord  | 512 | 0.68 | 0.35 | 0.76 | 1.00 | 1.59      | 1.28      | 2.30      | 5.97 |
| 1e-03 | pod    | 32  | 1.02 | 1.29 | 2.79 | 1.00 | 1.21      | 1.25      | 1.09      | 1.05 |
| 1e-03 | pod    | 64  | 1.21 | 0.30 | 0.67 | 1.00 | 0.61      | 0.51      | 0.50      | 0.30 |
| 1e-03 | pod    | 128 | 1.74 | 1.38 | 0.92 | 1.00 | 0.81      | 1.12      | 0.93      | 0.62 |
| 1e-03 | pod    | 256 | 1.55 | 0.99 | 2.34 | 1.00 | 0.84      | 1.09      | 1.24      | 1.25 |
| 1e-03 | pod    | 512 | 0.66 | 0.80 | 1.10 | 1.00 | 0.78      | 0.75      | 0.98      | --   |
<!-- END GENERATED: kshape_poisson2d -->

<!-- BEGIN GENERATED: kshape_burgers2d -->
_normalised at k=8; quantity: latent solve (derived)_

| tau   | method | N   | k=2  | k=4  | k=6  | k=8  | k=12 | k=16 | k=24 | k=32  |
|-------|--------|-----|------|------|------|------|------|------|------|-------|
| 1e-01 | coord  | 32  | 2.27 | 2.02 | 0.94 | 1.00 | 1.30 | 1.63 | 2.27 | 14.72 |
| 1e-01 | coord  | 64  | 2.10 | 1.99 | 0.90 | 1.00 | 1.20 | 1.50 | 2.04 | 16.68 |
| 1e-01 | coord  | 128 | 2.17 | 2.12 | 0.92 | 1.00 | 1.23 | 1.52 | 2.10 | 14.45 |
| 1e-01 | coord  | 256 | 2.05 | 2.06 | 0.88 | 1.00 | 1.26 | 1.58 | 2.58 | 18.62 |
| 1e-01 | pod    | 32  | 1.12 | 1.07 | 1.09 | 1.00 | 1.22 | 1.16 | 1.15 | 1.36  |
| 1e-01 | pod    | 64  | 1.01 | 1.00 | 1.07 | 1.00 | 1.03 | 1.07 | 1.15 | 1.34  |
| 1e-01 | pod    | 128 | 1.11 | 1.02 | 1.08 | 1.00 | 1.21 | 1.14 | 1.20 | 1.33  |
| 1e-01 | pod    | 256 | 1.04 | 0.98 | 1.04 | 1.00 | 1.12 | 1.13 | 1.21 | 1.40  |
| 1e-02 | coord  | 32  | 0.73 | 0.69 | 0.81 | 1.00 | 1.33 | 1.87 | 3.55 | 18.52 |
| 1e-02 | coord  | 64  | 0.73 | 0.72 | 0.83 | 1.00 | 1.39 | 1.90 | 3.32 | 20.95 |
| 1e-02 | coord  | 128 | 0.76 | 0.75 | 0.86 | 1.00 | 1.41 | 1.92 | 3.70 | 20.65 |
| 1e-02 | coord  | 256 | 0.72 | 0.77 | 0.86 | 1.00 | 1.42 | 1.86 | 3.31 | 20.57 |
| 1e-02 | pod    | 32  | 1.11 | 1.07 | 1.09 | 1.00 | 1.22 | 1.16 | 1.23 | 1.47  |
| 1e-02 | pod    | 64  | 1.02 | 1.00 | 1.07 | 1.00 | 1.03 | 1.06 | 1.24 | 1.40  |
| 1e-02 | pod    | 128 | 1.11 | 1.01 | 1.08 | 1.00 | 1.22 | 1.15 | 1.24 | 1.54  |
| 1e-02 | pod    | 256 | 1.04 | 0.99 | 1.05 | 1.00 | 1.12 | 1.14 | 1.40 | 1.41  |
| 1e-03 | coord  | 32  | 0.72 | 0.69 | 0.81 | 1.00 | 1.34 | 1.86 | 3.54 | 18.43 |
| 1e-03 | coord  | 64  | 0.73 | 0.71 | 0.83 | 1.00 | 1.39 | 1.90 | 3.38 | 20.86 |
| 1e-03 | coord  | 128 | 0.75 | 0.75 | 0.85 | 1.00 | 1.40 | 1.91 | 3.67 | 20.45 |
| 1e-03 | coord  | 256 | 0.73 | 0.76 | 0.86 | 1.00 | 1.41 | 1.82 | 3.35 | 19.96 |
| 1e-03 | pod    | 32  | 1.10 | 1.04 | 1.09 | 1.00 | 1.22 | 1.15 | 1.23 | 1.47  |
| 1e-03 | pod    | 64  | 1.01 | 1.00 | 1.07 | 1.00 | 1.03 | 1.06 | 1.23 | 1.40  |
| 1e-03 | pod    | 128 | 1.07 | 0.98 | 1.04 | 1.00 | 1.17 | 1.10 | 1.20 | 1.48  |
| 1e-03 | pod    | 256 | 1.05 | 1.00 | 1.05 | 1.00 | 1.13 | 1.15 | 1.41 | 1.43  |
<!-- END GENERATED: kshape_burgers2d -->

The same shape on the **end-to-end** online cost, which is directly measured and therefore
well defined at every mesh (the derived latent-solve time is a small difference of large
numbers once the O(n) decode dominates, and is suppressed as `below res` above):

<!-- BEGIN GENERATED: kshapee2e_poisson2d -->
_normalised at k=8; quantity: end-to-end online_

| tau   | method | N   | k=2  | k=4  | k=6  | k=8  | k=12 | k=16 | k=24 | k=32 |
|-------|--------|-----|------|------|------|------|------|------|------|------|
| 1e-01 | coord  | 32  | 3.64 | 0.85 | 1.07 | 1.00 | 1.10 | 1.20 | 1.27 | 3.09 |
| 1e-01 | coord  | 64  | 3.23 | 0.84 | 1.04 | 1.00 | 1.01 | 1.15 | 1.19 | 2.99 |
| 1e-01 | coord  | 128 | 2.94 | 0.86 | 1.03 | 1.00 | 1.01 | 1.15 | 1.22 | 2.63 |
| 1e-01 | coord  | 256 | 2.60 | 0.92 | 1.04 | 1.00 | 1.02 | 1.11 | 1.15 | 2.40 |
| 1e-01 | coord  | 512 | 1.70 | 0.94 | 1.03 | 1.00 | 1.05 | 1.03 | 1.04 | 1.60 |
| 1e-01 | pod    | 32  | 1.32 | 1.45 | 1.20 | 1.00 | 0.41 | 0.42 | 0.44 | 0.46 |
| 1e-01 | pod    | 64  | 1.20 | 0.29 | 0.52 | 1.00 | 0.16 | 0.17 | 0.17 | 0.18 |
| 1e-01 | pod    | 128 | 2.35 | 1.74 | 1.32 | 1.00 | 0.49 | 0.47 | 0.48 | 0.53 |
| 1e-01 | pod    | 256 | 1.96 | 1.27 | 1.50 | 1.00 | 0.52 | 0.49 | 0.51 | 0.80 |
| 1e-01 | pod    | 512 | 1.21 | 1.26 | 1.49 | 1.00 | 0.54 | 0.64 | 0.70 | --   |
| 1e-02 | coord  | 32  | 2.51 | 1.13 | 1.21 | 1.00 | 1.11 | 1.11 | 1.33 | 3.23 |
| 1e-02 | coord  | 64  | 2.41 | 1.14 | 1.14 | 1.00 | 1.15 | 1.17 | 1.26 | 3.50 |
| 1e-02 | coord  | 128 | 2.12 | 1.11 | 1.12 | 1.00 | 1.12 | 1.08 | 1.20 | 2.93 |
| 1e-02 | coord  | 256 | 2.02 | 1.09 | 1.15 | 1.00 | 1.14 | 1.14 | 1.26 | 2.56 |
| 1e-02 | coord  | 512 | 1.40 | 1.06 | 1.12 | 1.00 | 1.10 | 1.05 | 1.09 | 1.90 |
| 1e-02 | pod    | 32  | 1.03 | 1.24 | 2.45 | 1.00 | 1.16 | 1.22 | 1.07 | 1.05 |
| 1e-02 | pod    | 64  | 1.17 | 0.35 | 0.69 | 1.00 | 0.63 | 0.55 | 0.53 | 0.37 |
| 1e-02 | pod    | 128 | 1.69 | 1.37 | 0.97 | 1.00 | 0.90 | 1.16 | 0.98 | 0.72 |
| 1e-02 | pod    | 256 | 1.33 | 0.97 | 1.94 | 1.00 | 0.84 | 1.03 | 1.15 | 0.98 |
| 1e-02 | pod    | 512 | 0.83 | 0.94 | 1.14 | 1.00 | 0.93 | 0.92 | 1.10 | --   |
| 1e-03 | coord  | 32  | 0.82 | 0.48 | 0.85 | 1.00 | 1.36 | 1.25 | 2.02 | 3.77 |
| 1e-03 | coord  | 64  | 0.94 | 0.56 | 1.12 | 1.00 | 1.51 | 1.39 | 2.18 | 4.90 |
| 1e-03 | coord  | 128 | 0.80 | 0.49 | 0.90 | 1.00 | 1.30 | 1.24 | 1.79 | 4.18 |
| 1e-03 | coord  | 256 | 0.97 | 0.67 | 1.01 | 1.00 | 1.48 | 1.33 | 2.19 | 4.49 |
| 1e-03 | coord  | 512 | 0.86 | 0.72 | 0.90 | 1.00 | 1.26 | 1.12 | 1.57 | 3.22 |
| 1e-03 | pod    | 32  | 1.02 | 1.23 | 2.44 | 1.00 | 1.17 | 1.21 | 1.07 | 1.04 |
| 1e-03 | pod    | 64  | 1.19 | 0.36 | 0.70 | 1.00 | 0.65 | 0.56 | 0.54 | 0.37 |
| 1e-03 | pod    | 128 | 1.60 | 1.31 | 0.93 | 1.00 | 0.85 | 1.10 | 0.94 | 0.69 |
| 1e-03 | pod    | 256 | 1.43 | 1.00 | 2.04 | 1.00 | 0.88 | 1.09 | 1.20 | 1.26 |
| 1e-03 | pod    | 512 | 0.78 | 0.87 | 1.06 | 1.00 | 0.87 | 0.86 | 1.02 | --   |
<!-- END GENERATED: kshapee2e_poisson2d -->

<!-- BEGIN GENERATED: kshapee2e_burgers2d -->
_normalised at k=8; quantity: end-to-end online_

| tau   | method | N   | k=2  | k=4  | k=6  | k=8  | k=12 | k=16 | k=24 | k=32  |
|-------|--------|-----|------|------|------|------|------|------|------|-------|
| 1e-01 | coord  | 32  | 2.22 | 1.98 | 0.94 | 1.00 | 1.29 | 1.60 | 2.22 | 14.16 |
| 1e-01 | coord  | 64  | 2.01 | 1.90 | 0.91 | 1.00 | 1.19 | 1.46 | 1.96 | 15.30 |
| 1e-01 | coord  | 128 | 1.84 | 1.80 | 0.95 | 1.00 | 1.16 | 1.38 | 1.79 | 10.63 |
| 1e-01 | coord  | 256 | 1.44 | 1.44 | 0.95 | 1.00 | 1.10 | 1.24 | 1.66 | 8.32  |
| 1e-01 | pod    | 32  | 1.12 | 1.07 | 1.09 | 1.00 | 1.22 | 1.16 | 1.15 | 1.36  |
| 1e-01 | pod    | 64  | 1.01 | 1.00 | 1.07 | 1.00 | 1.03 | 1.07 | 1.15 | 1.34  |
| 1e-01 | pod    | 128 | 1.11 | 1.02 | 1.08 | 1.00 | 1.21 | 1.14 | 1.20 | 1.33  |
| 1e-01 | pod    | 256 | 1.04 | 0.98 | 1.04 | 1.00 | 1.12 | 1.13 | 1.21 | 1.40  |
| 1e-02 | coord  | 32  | 0.73 | 0.70 | 0.81 | 1.00 | 1.33 | 1.85 | 3.51 | 18.29 |
| 1e-02 | coord  | 64  | 0.74 | 0.73 | 0.84 | 1.00 | 1.38 | 1.87 | 3.24 | 20.30 |
| 1e-02 | coord  | 128 | 0.79 | 0.78 | 0.87 | 1.00 | 1.36 | 1.81 | 3.38 | 18.26 |
| 1e-02 | coord  | 256 | 0.81 | 0.84 | 0.90 | 1.00 | 1.28 | 1.58 | 2.55 | 14.20 |
| 1e-02 | pod    | 32  | 1.11 | 1.07 | 1.09 | 1.00 | 1.22 | 1.16 | 1.22 | 1.47  |
| 1e-02 | pod    | 64  | 1.01 | 1.00 | 1.07 | 1.00 | 1.03 | 1.06 | 1.23 | 1.40  |
| 1e-02 | pod    | 128 | 1.11 | 1.01 | 1.08 | 1.00 | 1.22 | 1.15 | 1.24 | 1.54  |
| 1e-02 | pod    | 256 | 1.04 | 0.99 | 1.05 | 1.00 | 1.12 | 1.14 | 1.40 | 1.41  |
| 1e-03 | coord  | 32  | 0.72 | 0.70 | 0.81 | 1.00 | 1.34 | 1.85 | 3.50 | 18.20 |
| 1e-03 | coord  | 64  | 0.74 | 0.72 | 0.84 | 1.00 | 1.37 | 1.87 | 3.30 | 20.22 |
| 1e-03 | coord  | 128 | 0.78 | 0.78 | 0.87 | 1.00 | 1.35 | 1.80 | 3.35 | 18.11 |
| 1e-03 | coord  | 256 | 0.82 | 0.83 | 0.91 | 1.00 | 1.28 | 1.55 | 2.60 | 13.90 |
| 1e-03 | pod    | 32  | 1.10 | 1.04 | 1.09 | 1.00 | 1.22 | 1.15 | 1.23 | 1.47  |
| 1e-03 | pod    | 64  | 1.01 | 1.00 | 1.07 | 1.00 | 1.03 | 1.06 | 1.23 | 1.40  |
| 1e-03 | pod    | 128 | 1.07 | 0.98 | 1.04 | 1.00 | 1.17 | 1.10 | 1.20 | 1.48  |
| 1e-03 | pod    | 256 | 1.05 | 1.00 | 1.05 | 1.00 | 1.13 | 1.15 | 1.41 | 1.43  |
<!-- END GENERATED: kshapee2e_burgers2d -->

Full cost tables at the other tolerances:

<!-- BEGIN GENERATED: cost_k_poisson2d_1e-01 -->
| method | N   | k=2  | k=4       | k=6       | k=8       | k=12      | k=16      | k=24      | k=32 |
|--------|-----|------|-----------|-----------|-----------|-----------|-----------|-----------|------|
| coord  | 32  | 7.05 | 1.42      | 1.88      | 1.73      | 1.94      | 2.13      | 2.28      | 5.94 |
| coord  | 64  | 6.70 | 1.49      | 1.93      | 1.85      | 1.86      | 2.16      | 2.26      | 6.17 |
| coord  | 128 | 6.52 | 1.51      | 1.93      | 1.85      | 1.88      | 2.22      | 2.38      | 5.74 |
| coord  | 256 | 6.57 | 0.75      | 1.19      | 1.05      | 1.10      | 1.39      | 1.54      | 5.80 |
| coord  | 512 | 5.48 | below res | below res | below res | below res | below res | below res | 4.32 |
| pod    | 32  | 0.88 | 0.99      | 0.78      | 0.62      | 0.13      | 0.13      | 0.15      | 0.18 |
| pod    | 64  | 2.27 | 0.39      | 0.88      | 1.85      | 0.14      | 0.15      | 0.16      | 0.17 |
| pod    | 128 | 1.81 | 1.28      | 0.92      | 0.63      | 0.19      | 0.18      | 0.18      | 0.22 |
| pod    | 256 | 1.50 | 0.85      | 1.06      | 0.62      | 0.17      | 0.11      | 0.13      | 0.33 |
| pod    | 512 | 0.68 | 0.71      | 0.95      | 0.42      | below res | below res | below res | --   |
<!-- END GENERATED: cost_k_poisson2d_1e-01 -->

<!-- BEGIN GENERATED: cost_k_poisson2d_1e-03 -->
| method | N   | k=2  | k=4  | k=6  | k=8  | k=12  | k=16  | k=24  | k=32  |
|--------|-----|------|------|------|------|-------|-------|-------|-------|
| coord  | 32  | 7.16 | 4.08 | 7.47 | 8.82 | 12.13 | 11.12 | 18.11 | 34.03 |
| coord  | 64  | 6.71 | 3.91 | 8.07 | 7.20 | 11.01 | 10.14 | 16.04 | 36.49 |
| coord  | 128 | 6.52 | 3.77 | 7.40 | 8.27 | 10.96 | 10.36 | 15.22 | 36.34 |
| coord  | 256 | 6.13 | 3.45 | 6.53 | 6.39 | 10.58 | 9.28  | 16.81 | 37.01 |
| coord  | 512 | 4.05 | 2.10 | 4.55 | 5.99 | 9.51  | 7.65  | 13.76 | 35.74 |
| pod    | 32  | 0.87 | 1.11 | 2.40 | 0.86 | 1.04  | 1.08  | 0.94  | 0.91  |
| pod    | 64  | 2.28 | 0.56 | 1.26 | 1.89 | 1.15  | 0.97  | 0.94  | 0.56  |
| pod    | 128 | 1.79 | 1.42 | 0.95 | 1.02 | 0.83  | 1.15  | 0.95  | 0.63  |
| pod    | 256 | 1.58 | 1.01 | 2.39 | 1.02 | 0.86  | 1.11  | 1.27  | 1.28  |
| pod    | 512 | 0.69 | 0.83 | 1.15 | 1.04 | 0.81  | 0.78  | 1.03  | --    |
<!-- END GENERATED: cost_k_poisson2d_1e-03 -->

<!-- BEGIN GENERATED: cost_k_burgers2d_1e-01 -->
| method | N   | k=2    | k=4    | k=6    | k=8    | k=12   | k=16   | k=24   | k=32    |
|--------|-----|--------|--------|--------|--------|--------|--------|--------|---------|
| coord  | 32  | 238.92 | 212.01 | 98.89  | 105.09 | 137.12 | 171.00 | 238.38 | 1546.58 |
| coord  | 64  | 223.97 | 211.70 | 95.98  | 106.46 | 128.27 | 159.94 | 217.70 | 1775.37 |
| coord  | 128 | 228.24 | 222.19 | 97.08  | 105.02 | 129.18 | 159.95 | 220.20 | 1517.16 |
| coord  | 256 | 247.90 | 248.68 | 106.68 | 120.77 | 151.94 | 191.01 | 312.16 | 2248.67 |
| pod    | 32  | 34.52  | 32.93  | 33.54  | 30.75  | 37.60  | 35.60  | 35.42  | 41.77   |
| pod    | 64  | 32.61  | 32.29  | 34.47  | 32.14  | 33.25  | 34.42  | 37.11  | 42.93   |
| pod    | 128 | 34.43  | 31.47  | 33.53  | 30.94  | 37.57  | 35.33  | 36.98  | 41.06   |
| pod    | 256 | 33.34  | 31.66  | 33.40  | 32.19  | 35.98  | 36.50  | 39.06  | 45.02   |
<!-- END GENERATED: cost_k_burgers2d_1e-01 -->

<!-- BEGIN GENERATED: cost_k_burgers2d_1e-03 -->
| method | N   | k=2    | k=4    | k=6    | k=8    | k=12   | k=16   | k=24    | k=32    |
|--------|-----|--------|--------|--------|--------|--------|--------|---------|---------|
| coord  | 32  | 237.79 | 229.04 | 268.02 | 331.05 | 445.18 | 617.36 | 1170.39 | 6100.12 |
| coord  | 64  | 224.04 | 218.96 | 254.72 | 307.01 | 425.91 | 583.80 | 1037.02 | 6405.31 |
| coord  | 128 | 228.96 | 226.68 | 257.65 | 303.93 | 426.30 | 580.34 | 1114.87 | 6214.83 |
| coord  | 256 | 264.89 | 273.92 | 312.81 | 361.82 | 510.19 | 657.17 | 1212.03 | 7220.16 |
| pod    | 32  | 33.78  | 32.18  | 33.59  | 30.81  | 37.59  | 35.58  | 37.82   | 45.23   |
| pod    | 64  | 32.68  | 32.22  | 34.47  | 32.26  | 33.24  | 34.19  | 39.76   | 45.05   |
| pod    | 128 | 34.44  | 31.52  | 33.55  | 32.20  | 37.63  | 35.57  | 38.51   | 47.58   |
| pod    | 256 | 33.13  | 31.62  | 33.39  | 31.70  | 35.77  | 36.50  | 44.62   | 45.26   |
<!-- END GENERATED: cost_k_burgers2d_1e-03 -->

### 6.2 Work to reach the tolerance vs k

Figures: `figs/ctol_iters_vs_k_*.{png,pdf}`. Jacobian evaluations (accepted LM steps; the
Burgers numbers are summed over the cold start and all 50 time steps).

<!-- BEGIN GENERATED: iters_k_poisson2d_1e-02 -->
| method | N   | k=2  | k=4  | k=6  | k=8  | k=12 | k=16 | k=24 | k=32 |
|--------|-----|------|------|------|------|------|------|------|------|
| coord  | 32  | 24.4 | 13.1 | 14.6 | 10.1 | 15.5 | 7.3  | 10.8 | 18.1 |
| coord  | 64  | 23.4 | 12.3 | 15.6 | 10.4 | 14.1 | 7.5  | 10.2 | 18.2 |
| coord  | 128 | 23.9 | 14.8 | 14.5 | 10.3 | 15.7 | 7.5  | 10.1 | 18.2 |
| coord  | 256 | 24.0 | 12.8 | 15.2 | 10.3 | 16.1 | 7.5  | 11.0 | 18.1 |
| coord  | 512 | 23.7 | 12.5 | 14.5 | 9.4  | 14.7 | 7.4  | 10.7 | 18.1 |
| pod    | 32  | 3.5  | 3.6  | 3.4  | 3.8  | 3.8  | 3.8  | 3.9  | 3.6  |
| pod    | 64  | 3.2  | 3.6  | 3.6  | 3.5  | 3.8  | 3.9  | 3.8  | 3.7  |
| pod    | 128 | 3.3  | 3.5  | 3.8  | 3.6  | 3.9  | 3.7  | 3.9  | 3.6  |
| pod    | 256 | 3.3  | 3.6  | 3.4  | 3.9  | 3.8  | 3.7  | 3.7  | 3.4  |
| pod    | 512 | 3.6  | 3.6  | 3.6  | 3.6  | 3.7  | 3.7  | 4.0  | --   |
<!-- END GENERATED: iters_k_poisson2d_1e-02 -->

<!-- BEGIN GENERATED: iters_k_burgers2d_1e-02 -->
| method | N   | k=2   | k=4   | k=6   | k=8   | k=12  | k=16  | k=24  | k=32  |
|--------|-----|-------|-------|-------|-------|-------|-------|-------|-------|
| coord  | 32  | 448.2 | 323.8 | 304.2 | 320.2 | 353.8 | 413.6 | 602.3 | 640.0 |
| coord  | 64  | 425.6 | 322.2 | 302.9 | 325.5 | 356.2 | 414.3 | 645.6 | 693.9 |
| coord  | 128 | 445.4 | 323.5 | 306.8 | 330.4 | 365.4 | 422.1 | 644.9 | 694.9 |
| coord  | 256 | 414.0 | 333.6 | 302.8 | 327.2 | 367.4 | 416.9 | 621.3 | 719.4 |
| pod    | 32  | 193.5 | 197.4 | 199.8 | 201.4 | 202.7 | 206.3 | 207.1 | 208.1 |
| pod    | 64  | 192.5 | 197.2 | 199.2 | 201.1 | 202.8 | 206.7 | 208.4 | 208.5 |
| pod    | 128 | 193.2 | 197.6 | 200.8 | 202.4 | 203.8 | 205.4 | 207.9 | 207.5 |
| pod    | 256 | 194.6 | 198.2 | 199.9 | 201.1 | 203.6 | 206.0 | 206.3 | 208.4 |
<!-- END GENERATED: iters_k_burgers2d_1e-02 -->

At the other two tolerances:

<!-- BEGIN GENERATED: iters_k_poisson2d_1e-01 -->
| method | N   | k=2  | k=4 | k=6 | k=8 | k=12 | k=16 | k=24 | k=32 |
|--------|-----|------|-----|-----|-----|------|------|------|------|
| coord  | 32  | 24.4 | 4.4 | 7.6 | 4.8 | 7.8  | 4.8  | 5.1  | 10.5 |
| coord  | 64  | 23.4 | 4.5 | 9.2 | 4.9 | 7.1  | 4.7  | 5.1  | 9.7  |
| coord  | 128 | 23.9 | 4.5 | 7.1 | 4.9 | 8.6  | 4.6  | 4.9  | 10.4 |
| coord  | 256 | 24.0 | 4.5 | 9.6 | 4.8 | 10.2 | 4.7  | 5.1  | 10.8 |
| coord  | 512 | 22.9 | 4.4 | 8.6 | 4.9 | 7.5  | 4.8  | 4.8  | 9.6  |
| pod    | 32  | 3.5  | 3.4 | 3.2 | 3.4 | 2.8  | 2.5  | 2.0  | 2.0  |
| pod    | 64  | 3.2  | 3.4 | 3.3 | 3.0 | 2.8  | 2.4  | 2.0  | 2.0  |
| pod    | 128 | 3.3  | 3.3 | 3.4 | 3.2 | 2.8  | 2.4  | 2.0  | 2.0  |
| pod    | 256 | 3.3  | 3.4 | 3.2 | 3.4 | 2.8  | 2.4  | 2.0  | 2.0  |
| pod    | 512 | 3.6  | 3.4 | 3.2 | 3.2 | 2.8  | 2.4  | 2.0  | --   |
<!-- END GENERATED: iters_k_poisson2d_1e-01 -->

<!-- BEGIN GENERATED: iters_k_poisson2d_1e-03 -->
| method | N   | k=2  | k=4  | k=6  | k=8  | k=12 | k=16 | k=24 | k=32 |
|--------|-----|------|------|------|------|------|------|------|------|
| coord  | 32  | 24.4 | 16.9 | 27.9 | 30.1 | 32.9 | 30.9 | 38.7 | 40.0 |
| coord  | 64  | 23.4 | 15.9 | 29.2 | 28.2 | 34.7 | 29.0 | 37.4 | 39.6 |
| coord  | 128 | 23.9 | 17.7 | 27.4 | 27.4 | 33.8 | 30.3 | 35.2 | 40.1 |
| coord  | 256 | 24.0 | 17.1 | 30.6 | 30.6 | 36.2 | 28.8 | 38.4 | 41.4 |
| coord  | 512 | 23.7 | 16.2 | 26.0 | 31.3 | 35.1 | 30.4 | 37.3 | 40.8 |
| pod    | 32  | 3.5  | 3.6  | 3.4  | 3.8  | 3.8  | 3.8  | 3.9  | 3.8  |
| pod    | 64  | 3.2  | 3.6  | 3.6  | 3.5  | 3.8  | 3.9  | 3.8  | 3.9  |
| pod    | 128 | 3.3  | 3.5  | 3.8  | 3.6  | 3.9  | 3.7  | 3.9  | 3.9  |
| pod    | 256 | 3.3  | 3.6  | 3.4  | 3.9  | 3.8  | 3.7  | 3.7  | 3.7  |
| pod    | 512 | 3.6  | 3.6  | 3.6  | 3.6  | 3.7  | 3.7  | 4.0  | --   |
<!-- END GENERATED: iters_k_poisson2d_1e-03 -->

<!-- BEGIN GENERATED: iters_k_burgers2d_1e-01 -->
| method | N   | k=2   | k=4   | k=6   | k=8   | k=12  | k=16  | k=24  | k=32  |
|--------|-----|-------|-------|-------|-------|-------|-------|-------|-------|
| coord  | 32  | 448.2 | 273.1 | 129.9 | 137.9 | 118.6 | 183.8 | 201.6 | 250.8 |
| coord  | 64  | 425.6 | 279.0 | 133.3 | 130.4 | 127.1 | 169.1 | 249.6 | 257.6 |
| coord  | 128 | 445.4 | 279.6 | 136.6 | 127.9 | 128.6 | 181.0 | 232.4 | 322.8 |
| coord  | 256 | 414.0 | 290.5 | 134.1 | 136.6 | 134.4 | 190.8 | 301.6 | 317.1 |
| pod    | 32  | 193.5 | 197.4 | 199.8 | 201.4 | 202.7 | 204.5 | 194.8 | 196.3 |
| pod    | 64  | 192.5 | 197.2 | 199.2 | 201.1 | 202.8 | 205.8 | 198.2 | 196.9 |
| pod    | 128 | 193.2 | 197.6 | 200.8 | 202.4 | 203.8 | 205.1 | 197.9 | 196.6 |
| pod    | 256 | 194.6 | 198.2 | 199.9 | 201.1 | 203.6 | 204.9 | 197.4 | 198.0 |
<!-- END GENERATED: iters_k_burgers2d_1e-01 -->

<!-- BEGIN GENERATED: iters_k_burgers2d_1e-03 -->
| method | N   | k=2   | k=4   | k=6   | k=8   | k=12  | k=16  | k=24  | k=32  |
|--------|-----|-------|-------|-------|-------|-------|-------|-------|-------|
| coord  | 32  | 448.2 | 323.8 | 304.5 | 321.0 | 356.6 | 416.3 | 599.2 | 648.2 |
| coord  | 64  | 425.6 | 322.2 | 302.9 | 326.6 | 358.9 | 417.3 | 657.4 | 703.0 |
| coord  | 128 | 445.4 | 323.5 | 306.8 | 330.9 | 368.9 | 426.0 | 643.1 | 700.4 |
| coord  | 256 | 414.0 | 333.6 | 302.8 | 327.6 | 369.0 | 423.1 | 669.1 | 724.1 |
| pod    | 32  | 193.5 | 197.4 | 199.8 | 201.4 | 202.7 | 206.3 | 207.1 | 208.1 |
| pod    | 64  | 192.5 | 197.2 | 199.2 | 201.1 | 202.8 | 206.7 | 208.4 | 208.5 |
| pod    | 128 | 193.2 | 197.6 | 200.8 | 202.4 | 203.8 | 205.4 | 207.9 | 207.5 |
| pod    | 256 | 194.6 | 198.2 | 199.9 | 201.1 | 203.6 | 206.0 | 206.3 | 208.4 |
<!-- END GENERATED: iters_k_burgers2d_1e-03 -->

### 6.3 The knob -> accuracy map: field error actually achieved at each tau

**Accuracy is non-monotone in `k` — and the ceiling arm shows it is NOT the checkpoints.**
Each `k` is a *separately trained* decoder, so the natural first reading is that decoder quality
varies along the `k` axis. **The measurement rejects that.** Each checkpoint's own **oracle
ceiling** — the error of the best latent obtainable by LM on the data misfit to the held-out
field, which no solver can beat — is *cleanly monotone decreasing* in `k` at every mesh
(N=64: 1.24e-1 → 3.28e-3 from `k`=2 to 32, no reversals). The decoders get uniformly better.

What oscillates is the **ROM/ceiling ratio** (N=64, `tau`=1e-3: 1.1 at `k`=4, 6.6 at `k`=6,
1.2 at `k`=8, 7.7 at `k`=12, 1.6 at `k`=16). So the latent solve, not the decoder, is what
fails at particular `k`. Two further candidates are ruled out by the same table:

* **not hyper-reduction quality** — the NNLS-EQ fit is flat at ~1.3–1.7e-3 across all of them,
  and `k`=32's fit is the *best* of the set while its ratio is the *worst*;
* **not `M`/`k` headroom** — among the `M`=64 rows, `k`=6 has `M/k` = 10.7 and fails (6.6) while
  `k`=16 has `M/k` = 4.0 and works (1.6). More test modes per latent dimension is not the
  discriminator, despite "`M` > `k` comfortably" being a standing project rule.

**Two live hypotheses remain, and this cell does not yet decide between them.** A monotone
ceiling establishes that the decoder can *represent* the field at every `k`; it says nothing
about whether the manifold parameterisation is well *conditioned* there. So either:

1. **conditioning** — a checkpoint with near-degenerate or unused latent directions gives
   exactly this signature (good ceiling, ill-conditioned Gauss–Newton, high ratio). This is
   checkpoint-specific but is *not* "checkpoint quality" in the sense rejected above; it would
   be a training/architecture finding.
2. **a defect in the weak-form solve** at those dimensions — a solver finding.

The decoder-Jacobian spectrum at the ceiling latent (condition number and numerical rank per
(N, `k`), recorded free from the `H` = `Jᵀ J` the chunked ceiling already accumulates)
discriminates them, and is reported below. Whichever it is, this is a limitation of **our
method**, not of the approach, and it belongs in the verdict as such.

**The `k`=32 row is not a pure `k` step.** Under the promoted grid it runs at (`M`, `m`) =
(256, 1024) while `k` ∈ {2…24} run at (64, 256), so its ratio confounds the latent dimension
with a 4× change in both test modes and quadrature size. The isolator and artefact arms give
the comparison at **identical** (`M`, `m`) = (256, 256), and it survives: `k`=32 is ~7× worse
than `k`=8 at matched settings, consistently at every mesh (ratio 14.1 / 16.1 / 15.6 / 18.0 at
N=32/64/128/256, against 3.5 / 2.1 / 2.5 / 2.2 for `k`=8). So the `k`=32 degradation is real
and about `k`, not an artefact of the (`M`, `m`) change — but the raw ratio table's `k`=32
entry must still be read as measured under different settings.

<!-- BEGIN GENERATED: matched_Mm -->
| pde       | N   | k  | M   | m   | ROM err   | ceiling   | ROM/ceiling | EQ rel fit |
|-----------|-----|----|-----|-----|-----------|-----------|-------------|------------|
| burgers2d | 32  | 8  | 256 | 256 | 6.775e-02 | 1.232e-02 | 5.5         | 1.39e-01   |
| burgers2d | 32  | 32 | 256 | 256 | 8.507e-02 | 7.972e-03 | 10.7        | 1.46e-01   |
| burgers2d | 64  | 8  | 256 | 256 | 6.018e-02 | 1.235e-02 | 4.9         | 1.23e-01   |
| burgers2d | 64  | 32 | 256 | 256 | 8.358e-02 | 7.772e-03 | 10.8        | 1.32e-01   |
| burgers2d | 128 | 8  | 256 | 256 | 7.025e-02 | 1.456e-02 | 4.8         | 1.33e-01   |
| burgers2d | 128 | 32 | 256 | 256 | 8.472e-02 | 8.928e-03 | 9.5         | 1.35e-01   |
| burgers2d | 256 | 8  | 256 | 256 | 6.970e-02 | 1.611e-02 | 4.3         | 1.34e-01   |
| burgers2d | 256 | 32 | 256 | 256 | 9.062e-02 | 1.063e-02 | 8.5         | 1.32e-01   |
| poisson2d | 32  | 8  | 256 | 256 | 2.791e-02 | 7.966e-03 | 3.5         | 7.09e-02   |
| poisson2d | 32  | 32 | 256 | 256 | 5.074e-02 | 3.596e-03 | 14.1        | 6.62e-02   |
| poisson2d | 64  | 8  | 256 | 256 | 1.508e-02 | 7.043e-03 | 2.1         | 7.49e-02   |
| poisson2d | 64  | 32 | 256 | 256 | 5.273e-02 | 3.280e-03 | 16.1        | 6.75e-02   |
| poisson2d | 128 | 8  | 256 | 256 | 1.761e-02 | 6.951e-03 | 2.5         | 7.33e-02   |
| poisson2d | 128 | 32 | 256 | 256 | 5.045e-02 | 3.226e-03 | 15.6        | 7.04e-02   |
| poisson2d | 256 | 8  | 256 | 256 | 1.552e-02 | 6.933e-03 | 2.2         | 6.70e-02   |
| poisson2d | 256 | 32 | 256 | 256 | 5.796e-02 | 3.220e-03 | 18.0        | 7.06e-02   |
| poisson2d | 512 | 8  | 256 | 256 | 1.584e-02 | 6.929e-03 | 2.3         | 6.76e-02   |
| poisson2d | 512 | 32 | 256 | 256 | 5.162e-02 | 3.219e-03 | 16.0        | 6.65e-02   |
<!-- END GENERATED: matched_Mm -->

<!-- BEGIN GENERATED: jaccond_poisson2d -->
| quantity           | N   | k=2 | k=4 | k=6 | k=8 | k=12 | k=16 | k=24 | k=32     |
|--------------------|-----|-----|-----|-----|-----|------|------|------|----------|
| cond(J) median     | 512 | --  | --  | --  | --  | --   | --   | --   | 4.37e+03 |
| numerical rank / k | 512 | --  | --  | --  | --  | --   | --   | --   | 32/32    |
<!-- END GENERATED: jaccond_poisson2d -->

<!-- BEGIN GENERATED: ceiling_poisson2d -->
| quantity                            | N   | k=2       | k=4       | k=6       | k=8       | k=12      | k=16      | k=24      | k=32       |
|-------------------------------------|-----|-----------|-----------|-----------|-----------|-----------|-----------|-----------|------------|
| (M, m) measured at                  | 32  | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (256,900)  |
| ceiling                             | 32  | 1.250e-01 | 1.724e-02 | 1.047e-02 | 7.966e-03 | 7.088e-03 | 4.877e-03 | 4.424e-03 | 3.596e-03  |
| ROM tau=1e-01                       | 32  | 5.761e-01 | 8.405e-02 | 1.036e-01 | 9.539e-02 | 1.247e-01 | 7.673e-02 | 9.535e-02 | 1.073e-01  |
| ROM tau=1e-02                       | 32  | 5.761e-01 | 2.082e-02 | 6.209e-02 | 1.453e-02 | 5.395e-02 | 1.301e-02 | 1.983e-02 | 4.472e-02  |
| ROM tau=1e-03                       | 32  | 5.761e-01 | 1.958e-02 | 6.029e-02 | 1.076e-02 | 5.131e-02 | 7.647e-03 | 1.584e-02 | 4.185e-02  |
| ROM / ceiling (tau=1e-03)           | 32  | 4.61      | 1.14      | 5.76      | 1.35      | 7.24      | 1.57      | 3.58      | 11.64      |
| ceiling valid (ROM did not beat it) | 32  | yes       | yes       | yes       | yes       | yes       | yes       | yes       | yes        |
| (M, m) measured at                  | 64  | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (256,1024) |
| ceiling                             | 64  | 1.236e-01 | 1.551e-02 | 8.835e-03 | 7.043e-03 | 6.236e-03 | 4.133e-03 | 3.976e-03 | 3.280e-03  |
| ROM tau=1e-01                       | 64  | 5.455e-01 | 8.372e-02 | 1.113e-01 | 8.928e-02 | 1.173e-01 | 8.971e-02 | 8.770e-02 | 1.037e-01  |
| ROM tau=1e-02                       | 64  | 5.455e-01 | 2.075e-02 | 6.036e-02 | 1.322e-02 | 5.218e-02 | 1.155e-02 | 1.968e-02 | 4.329e-02  |
| ROM tau=1e-03                       | 64  | 5.455e-01 | 1.742e-02 | 5.845e-02 | 8.482e-03 | 4.789e-02 | 6.542e-03 | 1.491e-02 | 4.022e-02  |
| ROM / ceiling (tau=1e-03)           | 64  | 4.41      | 1.12      | 6.61      | 1.20      | 7.68      | 1.58      | 3.75      | 12.26      |
| ceiling valid (ROM did not beat it) | 64  | yes       | yes       | yes       | yes       | yes       | yes       | yes       | yes        |
| (M, m) measured at                  | 128 | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (256,1024) |
| ceiling                             | 128 | 1.233e-01 | 1.525e-02 | 8.637e-03 | 6.951e-03 | 6.137e-03 | 4.072e-03 | 3.930e-03 | 3.226e-03  |
| ROM tau=1e-01                       | 128 | 4.925e-01 | 8.514e-02 | 1.183e-01 | 9.486e-02 | 1.037e-01 | 1.011e-01 | 8.790e-02 | 1.053e-01  |
| ROM tau=1e-02                       | 128 | 4.925e-01 | 2.051e-02 | 6.077e-02 | 1.411e-02 | 4.925e-02 | 1.150e-02 | 1.886e-02 | 4.689e-02  |
| ROM tau=1e-03                       | 128 | 4.925e-01 | 1.704e-02 | 5.847e-02 | 8.397e-03 | 4.533e-02 | 6.234e-03 | 1.435e-02 | 4.376e-02  |
| ROM / ceiling (tau=1e-03)           | 128 | 3.99      | 1.12      | 6.77      | 1.21      | 7.39      | 1.53      | 3.65      | 13.56      |
| ceiling valid (ROM did not beat it) | 128 | yes       | yes       | yes       | yes       | yes       | yes       | yes       | yes        |
| (M, m) measured at                  | 256 | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (256,1024) |
| ceiling                             | 256 | 1.233e-01 | 1.519e-02 | 8.596e-03 | 6.933e-03 | 6.117e-03 | 4.062e-03 | 3.922e-03 | 3.220e-03  |
| ROM tau=1e-01                       | 256 | 4.925e-01 | 8.300e-02 | 1.350e-01 | 1.013e-01 | 1.117e-01 | 9.984e-02 | 9.974e-02 | 1.216e-01  |
| ROM tau=1e-02                       | 256 | 4.925e-01 | 2.003e-02 | 7.199e-02 | 1.406e-02 | 6.235e-02 | 1.127e-02 | 1.929e-02 | 6.089e-02  |
| ROM tau=1e-03                       | 256 | 4.925e-01 | 1.717e-02 | 7.027e-02 | 8.378e-03 | 6.007e-02 | 6.733e-03 | 1.563e-02 | 5.790e-02  |
| ROM / ceiling (tau=1e-03)           | 256 | 4.00      | 1.13      | 8.18      | 1.21      | 9.82      | 1.66      | 3.98      | 17.98      |
| ceiling valid (ROM did not beat it) | 256 | yes       | yes       | yes       | yes       | yes       | yes       | yes       | yes        |
| (M, m) measured at                  | 512 | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (256,1024) |
| ceiling                             | 512 | 1.233e-01 | 1.518e-02 | 8.586e-03 | 6.929e-03 | 6.112e-03 | 4.059e-03 | 3.920e-03 | 3.219e-03  |
| ROM tau=1e-01                       | 512 | 3.972e-01 | 9.325e-02 | 1.557e-01 | 8.127e-02 | 1.057e-01 | 8.246e-02 | 9.533e-02 | 1.063e-01  |
| ROM tau=1e-02                       | 512 | 3.960e-01 | 1.994e-02 | 9.640e-02 | 1.288e-02 | 2.915e-02 | 1.178e-02 | 1.893e-02 | 4.621e-02  |
| ROM tau=1e-03                       | 512 | 3.960e-01 | 1.695e-02 | 9.442e-02 | 8.614e-03 | 2.634e-02 | 6.388e-03 | 1.472e-02 | 4.315e-02  |
| ROM / ceiling (tau=1e-03)           | 512 | 3.21      | 1.12      | 11.00     | 1.24      | 4.31      | 1.57      | 3.76      | 13.41      |
| ceiling valid (ROM did not beat it) | 512 | yes       | yes       | yes       | yes       | yes       | yes       | yes       | yes        |
<!-- END GENERATED: ceiling_poisson2d -->

<!-- BEGIN GENERATED: ceiling_burgers2d -->
| quantity                            | N   | k=2       | k=4       | k=6       | k=8       | k=12      | k=16      | k=24      | k=32       |
|-------------------------------------|-----|-----------|-----------|-----------|-----------|-----------|-----------|-----------|------------|
| (M, m) measured at                  | 32  | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (256,900)  |
| ceiling                             | 32  | 6.949e-01 | 5.097e-02 | 1.476e-02 | 1.232e-02 | 9.831e-03 | 8.529e-03 | 7.661e-03 | 7.972e-03  |
| ROM tau=1e-01                       | 32  | 4.626e-01 | 7.327e-02 | 2.099e-02 | 2.321e-02 | 2.182e-02 | 3.666e-02 | 3.687e-02 | 2.380e-02  |
| ROM tau=1e-02                       | 32  | 4.626e-01 | 7.064e-02 | 1.692e-02 | 1.623e-02 | 1.172e-02 | 1.602e-02 | 2.235e-02 | 7.679e-03  |
| ROM tau=1e-03                       | 32  | 4.626e-01 | 7.064e-02 | 1.692e-02 | 1.623e-02 | 1.172e-02 | 1.588e-02 | 1.883e-02 | 7.568e-03  |
| ROM / ceiling (tau=1e-03)           | 32  | 0.67      | 1.39      | 1.15      | 1.32      | 1.19      | 1.86      | 2.46      | 0.95       |
| ceiling valid (ROM did not beat it) | 32  | no        | yes       | yes       | yes       | yes       | yes       | yes       | no         |
| (M, m) measured at                  | 64  | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (256,1024) |
| ceiling                             | 64  | 6.752e-01 | 5.541e-02 | 1.517e-02 | 1.235e-02 | 1.050e-02 | 8.975e-03 | 7.483e-03 | 7.772e-03  |
| ROM tau=1e-01                       | 64  | 4.314e-01 | 7.691e-02 | 2.238e-02 | 2.071e-02 | 1.936e-02 | 2.660e-02 | 4.497e-02 | 2.083e-02  |
| ROM tau=1e-02                       | 64  | 4.314e-01 | 7.456e-02 | 1.783e-02 | 1.579e-02 | 1.265e-02 | 1.180e-02 | 2.892e-02 | 9.383e-03  |
| ROM tau=1e-03                       | 64  | 4.314e-01 | 7.456e-02 | 1.783e-02 | 1.579e-02 | 1.265e-02 | 1.177e-02 | 2.428e-02 | 9.383e-03  |
| ROM / ceiling (tau=1e-03)           | 64  | 0.64      | 1.35      | 1.18      | 1.28      | 1.21      | 1.31      | 3.24      | 1.21       |
| ceiling valid (ROM did not beat it) | 64  | no        | yes       | yes       | yes       | yes       | yes       | yes       | yes        |
| (M, m) measured at                  | 128 | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (256,1024) |
| ceiling                             | 128 | 6.803e-01 | 6.057e-02 | 1.741e-02 | 1.456e-02 | 1.237e-02 | 1.197e-02 | 9.736e-03 | 8.928e-03  |
| ROM tau=1e-01                       | 128 | 4.489e-01 | 8.072e-02 | 2.390e-02 | 2.386e-02 | 2.574e-02 | 4.569e-02 | 5.337e-02 | 2.421e-02  |
| ROM tau=1e-02                       | 128 | 4.489e-01 | 7.871e-02 | 1.898e-02 | 1.672e-02 | 1.426e-02 | 1.578e-02 | 2.322e-02 | 9.640e-03  |
| ROM tau=1e-03                       | 128 | 4.489e-01 | 7.871e-02 | 1.898e-02 | 1.672e-02 | 1.426e-02 | 1.546e-02 | 2.192e-02 | 9.640e-03  |
| ROM / ceiling (tau=1e-03)           | 128 | 0.66      | 1.30      | 1.09      | 1.15      | 1.15      | 1.29      | 2.25      | 1.08       |
| ceiling valid (ROM did not beat it) | 128 | no        | yes       | yes       | yes       | yes       | yes       | yes       | yes        |
| (M, m) measured at                  | 256 | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (64,256)  | (256,1024) |
| ceiling                             | 256 | 6.858e-01 | 6.127e-02 | 1.888e-02 | 1.611e-02 | 1.372e-02 | 1.362e-02 | 1.161e-02 | 1.063e-02  |
| ROM tau=1e-01                       | 256 | 4.201e-01 | 8.435e-02 | 2.468e-02 | 2.541e-02 | 2.133e-02 | 4.135e-02 | 9.278e-02 | 2.427e-02  |
| ROM tau=1e-02                       | 256 | 4.201e-01 | 8.235e-02 | 2.000e-02 | 1.761e-02 | 1.489e-02 | 1.451e-02 | 2.079e-02 | 1.041e-02  |
| ROM tau=1e-03                       | 256 | 4.201e-01 | 8.235e-02 | 2.000e-02 | 1.761e-02 | 1.489e-02 | 1.444e-02 | 2.702e-02 | 1.037e-02  |
| ROM / ceiling (tau=1e-03)           | 256 | 0.61      | 1.34      | 1.06      | 1.09      | 1.09      | 1.06      | 2.33      | 0.98       |
| ceiling valid (ROM did not beat it) | 256 | no        | yes       | yes       | yes       | yes       | yes       | yes       | no         |
<!-- END GENERATED: ceiling_burgers2d -->


<!-- BEGIN GENERATED: err_k_poisson2d_1e-01 -->
| method | N   | k=2       | k=4       | k=6       | k=8       | k=12      | k=16      | k=24      | k=32      |
|--------|-----|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| coord  | 32  | 5.761e-01 | 8.405e-02 | 1.036e-01 | 9.539e-02 | 1.247e-01 | 7.673e-02 | 9.535e-02 | 1.073e-01 |
| coord  | 64  | 5.455e-01 | 8.372e-02 | 1.113e-01 | 8.928e-02 | 1.173e-01 | 8.971e-02 | 8.770e-02 | 1.037e-01 |
| coord  | 128 | 4.925e-01 | 8.514e-02 | 1.183e-01 | 9.486e-02 | 1.037e-01 | 1.011e-01 | 8.790e-02 | 1.053e-01 |
| coord  | 256 | 4.925e-01 | 8.300e-02 | 1.350e-01 | 1.013e-01 | 1.117e-01 | 9.984e-02 | 9.974e-02 | 1.216e-01 |
| coord  | 512 | 3.972e-01 | 9.325e-02 | 1.557e-01 | 8.127e-02 | 1.057e-01 | 8.246e-02 | 9.533e-02 | 1.063e-01 |
| pod    | 32  | 4.586e-01 | 2.936e-01 | 2.225e-01 | 1.795e-01 | 1.317e-01 | 1.036e-01 | 6.946e-02 | 5.367e-02 |
| pod    | 64  | 4.570e-01 | 2.913e-01 | 2.200e-01 | 1.769e-01 | 1.290e-01 | 1.008e-01 | 6.678e-02 | 5.105e-02 |
| pod    | 128 | 4.566e-01 | 2.908e-01 | 2.195e-01 | 1.763e-01 | 1.284e-01 | 1.002e-01 | 6.620e-02 | 5.049e-02 |
| pod    | 256 | 4.565e-01 | 2.907e-01 | 2.193e-01 | 1.761e-01 | 1.282e-01 | 1.000e-01 | 6.605e-02 | 5.035e-02 |
| pod    | 512 | 4.565e-01 | 2.906e-01 | 2.193e-01 | 1.761e-01 | 1.282e-01 | 1.000e-01 | 6.602e-02 | --        |
<!-- END GENERATED: err_k_poisson2d_1e-01 -->

<!-- BEGIN GENERATED: err_k_poisson2d_1e-02 -->
| method | N   | k=2       | k=4       | k=6       | k=8       | k=12      | k=16      | k=24      | k=32      |
|--------|-----|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| coord  | 32  | 5.761e-01 | 2.082e-02 | 6.209e-02 | 1.453e-02 | 5.395e-02 | 1.301e-02 | 1.983e-02 | 4.472e-02 |
| coord  | 64  | 5.455e-01 | 2.075e-02 | 6.036e-02 | 1.322e-02 | 5.218e-02 | 1.155e-02 | 1.968e-02 | 4.329e-02 |
| coord  | 128 | 4.925e-01 | 2.051e-02 | 6.077e-02 | 1.411e-02 | 4.925e-02 | 1.150e-02 | 1.886e-02 | 4.689e-02 |
| coord  | 256 | 4.925e-01 | 2.003e-02 | 7.199e-02 | 1.406e-02 | 6.235e-02 | 1.127e-02 | 1.929e-02 | 6.089e-02 |
| coord  | 512 | 3.960e-01 | 1.994e-02 | 9.640e-02 | 1.288e-02 | 2.915e-02 | 1.178e-02 | 1.893e-02 | 4.621e-02 |
| pod    | 32  | 4.586e-01 | 2.936e-01 | 2.225e-01 | 1.795e-01 | 1.317e-01 | 1.036e-01 | 6.946e-02 | 5.367e-02 |
| pod    | 64  | 4.570e-01 | 2.913e-01 | 2.200e-01 | 1.769e-01 | 1.290e-01 | 1.008e-01 | 6.678e-02 | 5.105e-02 |
| pod    | 128 | 4.566e-01 | 2.908e-01 | 2.195e-01 | 1.763e-01 | 1.284e-01 | 1.002e-01 | 6.620e-02 | 5.049e-02 |
| pod    | 256 | 4.565e-01 | 2.907e-01 | 2.193e-01 | 1.761e-01 | 1.282e-01 | 1.000e-01 | 6.605e-02 | 5.035e-02 |
| pod    | 512 | 4.565e-01 | 2.906e-01 | 2.193e-01 | 1.761e-01 | 1.282e-01 | 1.000e-01 | 6.602e-02 | --        |
<!-- END GENERATED: err_k_poisson2d_1e-02 -->

<!-- BEGIN GENERATED: err_k_poisson2d_1e-03 -->
| method | N   | k=2       | k=4       | k=6       | k=8       | k=12      | k=16      | k=24      | k=32      |
|--------|-----|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| coord  | 32  | 5.761e-01 | 1.958e-02 | 6.029e-02 | 1.076e-02 | 5.131e-02 | 7.647e-03 | 1.584e-02 | 4.185e-02 |
| coord  | 64  | 5.455e-01 | 1.742e-02 | 5.845e-02 | 8.482e-03 | 4.789e-02 | 6.542e-03 | 1.491e-02 | 4.022e-02 |
| coord  | 128 | 4.925e-01 | 1.704e-02 | 5.847e-02 | 8.397e-03 | 4.533e-02 | 6.234e-03 | 1.435e-02 | 4.376e-02 |
| coord  | 256 | 4.925e-01 | 1.717e-02 | 7.027e-02 | 8.378e-03 | 6.007e-02 | 6.733e-03 | 1.563e-02 | 5.790e-02 |
| coord  | 512 | 3.960e-01 | 1.695e-02 | 9.442e-02 | 8.614e-03 | 2.634e-02 | 6.388e-03 | 1.472e-02 | 4.315e-02 |
| pod    | 32  | 4.586e-01 | 2.936e-01 | 2.225e-01 | 1.795e-01 | 1.317e-01 | 1.036e-01 | 6.946e-02 | 5.367e-02 |
| pod    | 64  | 4.570e-01 | 2.913e-01 | 2.200e-01 | 1.769e-01 | 1.290e-01 | 1.008e-01 | 6.678e-02 | 5.105e-02 |
| pod    | 128 | 4.566e-01 | 2.908e-01 | 2.195e-01 | 1.763e-01 | 1.284e-01 | 1.002e-01 | 6.620e-02 | 5.049e-02 |
| pod    | 256 | 4.565e-01 | 2.907e-01 | 2.193e-01 | 1.761e-01 | 1.282e-01 | 1.000e-01 | 6.605e-02 | 5.035e-02 |
| pod    | 512 | 4.565e-01 | 2.906e-01 | 2.193e-01 | 1.761e-01 | 1.282e-01 | 1.000e-01 | 6.602e-02 | --        |
<!-- END GENERATED: err_k_poisson2d_1e-03 -->

<!-- BEGIN GENERATED: err_k_burgers2d_1e-01 -->
| method | N   | k=2       | k=4       | k=6       | k=8       | k=12      | k=16      | k=24      | k=32      |
|--------|-----|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| coord  | 32  | 4.626e-01 | 7.327e-02 | 2.099e-02 | 2.321e-02 | 2.182e-02 | 3.666e-02 | 3.687e-02 | 2.380e-02 |
| coord  | 64  | 4.314e-01 | 7.691e-02 | 2.238e-02 | 2.071e-02 | 1.936e-02 | 2.660e-02 | 4.497e-02 | 2.083e-02 |
| coord  | 128 | 4.489e-01 | 8.072e-02 | 2.390e-02 | 2.386e-02 | 2.574e-02 | 4.569e-02 | 5.337e-02 | 2.421e-02 |
| coord  | 256 | 4.201e-01 | 8.435e-02 | 2.468e-02 | 2.541e-02 | 2.133e-02 | 4.135e-02 | 9.278e-02 | 2.427e-02 |
| pod    | 32  | 6.003e-01 | 3.850e-01 | 2.617e-01 | 2.015e-01 | 1.280e-01 | 9.043e-02 | 5.462e-02 | 3.778e-02 |
| pod    | 64  | 6.062e-01 | 3.934e-01 | 2.703e-01 | 2.103e-01 | 1.368e-01 | 9.667e-02 | 5.937e-02 | 4.221e-02 |
| pod    | 128 | 6.098e-01 | 3.984e-01 | 2.757e-01 | 2.158e-01 | 1.425e-01 | 1.010e-01 | 6.273e-02 | 4.553e-02 |
| pod    | 256 | 6.117e-01 | 4.012e-01 | 2.787e-01 | 2.190e-01 | 1.459e-01 | 1.035e-01 | 6.497e-02 | 4.748e-02 |
<!-- END GENERATED: err_k_burgers2d_1e-01 -->

<!-- BEGIN GENERATED: err_k_burgers2d_1e-02 -->
| method | N   | k=2       | k=4       | k=6       | k=8       | k=12      | k=16      | k=24      | k=32      |
|--------|-----|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| coord  | 32  | 4.626e-01 | 7.064e-02 | 1.692e-02 | 1.623e-02 | 1.172e-02 | 1.602e-02 | 2.235e-02 | 7.679e-03 |
| coord  | 64  | 4.314e-01 | 7.456e-02 | 1.783e-02 | 1.579e-02 | 1.265e-02 | 1.180e-02 | 2.892e-02 | 9.383e-03 |
| coord  | 128 | 4.489e-01 | 7.871e-02 | 1.898e-02 | 1.672e-02 | 1.426e-02 | 1.578e-02 | 2.322e-02 | 9.640e-03 |
| coord  | 256 | 4.201e-01 | 8.235e-02 | 2.000e-02 | 1.761e-02 | 1.489e-02 | 1.451e-02 | 2.079e-02 | 1.041e-02 |
| pod    | 32  | 6.003e-01 | 3.850e-01 | 2.617e-01 | 2.015e-01 | 1.280e-01 | 9.043e-02 | 5.462e-02 | 3.777e-02 |
| pod    | 64  | 6.062e-01 | 3.934e-01 | 2.703e-01 | 2.103e-01 | 1.368e-01 | 9.667e-02 | 5.937e-02 | 4.221e-02 |
| pod    | 128 | 6.098e-01 | 3.984e-01 | 2.757e-01 | 2.158e-01 | 1.425e-01 | 1.010e-01 | 6.273e-02 | 4.553e-02 |
| pod    | 256 | 6.117e-01 | 4.012e-01 | 2.787e-01 | 2.190e-01 | 1.459e-01 | 1.035e-01 | 6.497e-02 | 4.748e-02 |
<!-- END GENERATED: err_k_burgers2d_1e-02 -->

<!-- BEGIN GENERATED: err_k_burgers2d_1e-03 -->
| method | N   | k=2       | k=4       | k=6       | k=8       | k=12      | k=16      | k=24      | k=32      |
|--------|-----|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| coord  | 32  | 4.626e-01 | 7.064e-02 | 1.692e-02 | 1.623e-02 | 1.172e-02 | 1.588e-02 | 1.883e-02 | 7.568e-03 |
| coord  | 64  | 4.314e-01 | 7.456e-02 | 1.783e-02 | 1.579e-02 | 1.265e-02 | 1.177e-02 | 2.428e-02 | 9.383e-03 |
| coord  | 128 | 4.489e-01 | 7.871e-02 | 1.898e-02 | 1.672e-02 | 1.426e-02 | 1.546e-02 | 2.192e-02 | 9.640e-03 |
| coord  | 256 | 4.201e-01 | 8.235e-02 | 2.000e-02 | 1.761e-02 | 1.489e-02 | 1.444e-02 | 2.702e-02 | 1.037e-02 |
| pod    | 32  | 6.003e-01 | 3.850e-01 | 2.617e-01 | 2.015e-01 | 1.280e-01 | 9.043e-02 | 5.462e-02 | 3.777e-02 |
| pod    | 64  | 6.062e-01 | 3.934e-01 | 2.703e-01 | 2.103e-01 | 1.368e-01 | 9.667e-02 | 5.937e-02 | 4.221e-02 |
| pod    | 128 | 6.098e-01 | 3.984e-01 | 2.757e-01 | 2.158e-01 | 1.425e-01 | 1.010e-01 | 6.273e-02 | 4.553e-02 |
| pod    | 256 | 6.117e-01 | 4.012e-01 | 2.787e-01 | 2.190e-01 | 1.459e-01 | 1.035e-01 | 6.497e-02 | 4.748e-02 |
<!-- END GENERATED: err_k_burgers2d_1e-03 -->

### 6.4 Censoring: percentage of cells that never reached tau

Reported honestly, never dropped. For Poisson this is the fraction of the 16 sources; for
Burgers the fraction of the 51 tau-stopped solves (cold start + 50 time steps), averaged over
the 16 held-out trajectories.

<!-- BEGIN GENERATED: censor_poisson2d -->
| tau   | method | N   | k=2 | k=4 | k=6 | k=8 | k=12 | k=16 | k=24 | k=32 |
|-------|--------|-----|-----|-----|-----|-----|------|------|------|------|
| 1e-01 | coord  | 32  | 100 | 0   | 6   | 0   | 6    | 0    | 0    | 6    |
| 1e-01 | coord  | 64  | 100 | 0   | 6   | 0   | 6    | 0    | 0    | 6    |
| 1e-01 | coord  | 128 | 100 | 0   | 6   | 0   | 6    | 0    | 0    | 6    |
| 1e-01 | coord  | 256 | 100 | 0   | 6   | 0   | 12   | 0    | 0    | 12   |
| 1e-01 | coord  | 512 | 88  | 0   | 12  | 0   | 6    | 0    | 0    | 6    |
| 1e-01 | pod    | 32  | 100 | 88  | 81  | 75  | 44   | 25   | 0    | 0    |
| 1e-01 | pod    | 64  | 100 | 88  | 81  | 75  | 44   | 25   | 0    | 0    |
| 1e-01 | pod    | 128 | 100 | 88  | 81  | 75  | 44   | 25   | 0    | 0    |
| 1e-01 | pod    | 256 | 100 | 88  | 81  | 75  | 44   | 25   | 0    | 0    |
| 1e-01 | pod    | 512 | 100 | 88  | 81  | 75  | 44   | 25   | 0    | --   |
| 1e-02 | coord  | 32  | 100 | 56  | 19  | 6   | 25   | 0    | 6    | 25   |
| 1e-02 | coord  | 64  | 100 | 56  | 19  | 6   | 19   | 0    | 6    | 31   |
| 1e-02 | coord  | 128 | 100 | 62  | 19  | 12  | 25   | 0    | 6    | 31   |
| 1e-02 | coord  | 256 | 100 | 56  | 12  | 6   | 25   | 0    | 6    | 31   |
| 1e-02 | coord  | 512 | 100 | 56  | 19  | 6   | 19   | 0    | 6    | 31   |
| 1e-02 | pod    | 32  | 100 | 100 | 100 | 100 | 100  | 100  | 100  | 88   |
| 1e-02 | pod    | 64  | 100 | 100 | 100 | 100 | 100  | 100  | 100  | 88   |
| 1e-02 | pod    | 128 | 100 | 100 | 100 | 100 | 100  | 100  | 100  | 88   |
| 1e-02 | pod    | 256 | 100 | 100 | 100 | 100 | 100  | 100  | 100  | 88   |
| 1e-02 | pod    | 512 | 100 | 100 | 100 | 100 | 100  | 100  | 100  | --   |
| 1e-03 | coord  | 32  | 100 | 100 | 100 | 100 | 100  | 94   | 94   | 100  |
| 1e-03 | coord  | 64  | 100 | 100 | 100 | 100 | 100  | 94   | 94   | 100  |
| 1e-03 | coord  | 128 | 100 | 100 | 100 | 100 | 100  | 94   | 94   | 100  |
| 1e-03 | coord  | 256 | 100 | 100 | 100 | 100 | 100  | 94   | 94   | 100  |
| 1e-03 | coord  | 512 | 100 | 100 | 100 | 100 | 100  | 94   | 94   | 100  |
| 1e-03 | pod    | 32  | 100 | 100 | 100 | 100 | 100  | 100  | 100  | 100  |
| 1e-03 | pod    | 64  | 100 | 100 | 100 | 100 | 100  | 100  | 100  | 100  |
| 1e-03 | pod    | 128 | 100 | 100 | 100 | 100 | 100  | 100  | 100  | 100  |
| 1e-03 | pod    | 256 | 100 | 100 | 100 | 100 | 100  | 100  | 100  | 100  |
| 1e-03 | pod    | 512 | 100 | 100 | 100 | 100 | 100  | 100  | 100  | --   |
<!-- END GENERATED: censor_poisson2d -->

<!-- BEGIN GENERATED: censor_burgers2d -->
| tau   | method | N   | k=2 | k=4 | k=6 | k=8 | k=12 | k=16 | k=24 | k=32 |
|-------|--------|-----|-----|-----|-----|-----|------|------|------|------|
| 1e-01 | coord  | 32  | 100 | 75  | 6   | 10  | 1    | 12   | 13   | 15   |
| 1e-01 | coord  | 64  | 100 | 79  | 8   | 6   | 1    | 10   | 15   | 17   |
| 1e-01 | coord  | 128 | 100 | 78  | 11  | 6   | 2    | 8    | 12   | 24   |
| 1e-01 | coord  | 256 | 100 | 80  | 10  | 9   | 4    | 12   | 18   | 26   |
| 1e-01 | pod    | 32  | 100 | 100 | 100 | 100 | 100  | 99   | 89   | 90   |
| 1e-01 | pod    | 64  | 100 | 100 | 100 | 100 | 100  | 99   | 91   | 90   |
| 1e-01 | pod    | 128 | 100 | 100 | 100 | 100 | 100  | 100  | 91   | 90   |
| 1e-01 | pod    | 256 | 100 | 100 | 100 | 100 | 100  | 100  | 92   | 91   |
| 1e-02 | coord  | 32  | 100 | 100 | 100 | 100 | 100  | 100  | 95   | 100  |
| 1e-02 | coord  | 64  | 100 | 100 | 100 | 100 | 100  | 100  | 96   | 100  |
| 1e-02 | coord  | 128 | 100 | 100 | 100 | 100 | 100  | 100  | 95   | 100  |
| 1e-02 | coord  | 256 | 100 | 100 | 100 | 100 | 100  | 99   | 96   | 100  |
| 1e-02 | pod    | 32  | 100 | 100 | 100 | 100 | 100  | 100  | 100  | 100  |
| 1e-02 | pod    | 64  | 100 | 100 | 100 | 100 | 100  | 100  | 100  | 100  |
| 1e-02 | pod    | 128 | 100 | 100 | 100 | 100 | 100  | 100  | 100  | 100  |
| 1e-02 | pod    | 256 | 100 | 100 | 100 | 100 | 100  | 100  | 100  | 100  |
| 1e-03 | coord  | 32  | 100 | 100 | 100 | 100 | 100  | 100  | 100  | 100  |
| 1e-03 | coord  | 64  | 100 | 100 | 100 | 100 | 100  | 100  | 100  | 100  |
| 1e-03 | coord  | 128 | 100 | 100 | 100 | 100 | 100  | 100  | 100  | 100  |
| 1e-03 | coord  | 256 | 100 | 100 | 100 | 100 | 100  | 100  | 100  | 100  |
| 1e-03 | pod    | 32  | 100 | 100 | 100 | 100 | 100  | 100  | 100  | 100  |
| 1e-03 | pod    | 64  | 100 | 100 | 100 | 100 | 100  | 100  | 100  | 100  |
| 1e-03 | pod    | 128 | 100 | 100 | 100 | 100 | 100  | 100  | 100  | 100  |
| 1e-03 | pod    | 256 | 100 | 100 | 100 | 100 | 100  | 100  | 100  | 100  |
<!-- END GENERATED: censor_burgers2d -->

### 6.5 Achieved discrete residual (reference only — never a stopping test)

Poisson: `||A u - f|| / ||f||` at the returned latent. Burgers: the FOM's own backward-Euler
residual `||R(u_{n+1}, u_n)|| / ||u_n||` along the ROM trajectory, averaged over the 50 steps.

<!-- BEGIN GENERATED: resid_poisson2d -->
| tau   | method | N   | k=2      | k=4      | k=6      | k=8      | k=12     | k=16     | k=24     | k=32     |
|-------|--------|-----|----------|----------|----------|----------|----------|----------|----------|----------|
| 1e-01 | coord  | 32  | 1.19e+00 | 5.31e-01 | 5.42e-01 | 6.57e-01 | 8.18e-01 | 3.61e-01 | 4.31e-01 | 6.13e-01 |
| 1e-01 | coord  | 64  | 1.30e+00 | 5.44e-01 | 6.38e-01 | 7.33e-01 | 8.12e-01 | 4.09e-01 | 5.09e-01 | 6.53e-01 |
| 1e-01 | coord  | 128 | 1.26e+00 | 5.63e-01 | 6.30e-01 | 8.22e-01 | 8.94e-01 | 4.01e-01 | 4.51e-01 | 6.84e-01 |
| 1e-01 | coord  | 256 | 1.27e+00 | 5.73e-01 | 7.15e-01 | 8.31e-01 | 8.79e-01 | 4.14e-01 | 4.91e-01 | 7.03e-01 |
| 1e-01 | coord  | 512 | 1.27e+00 | 6.11e-01 | 9.30e-01 | 7.43e-01 | 8.62e-01 | 4.07e-01 | 5.73e-01 | 7.15e-01 |
| 1e-01 | pod    | 32  | 9.19e-01 | 8.53e-01 | 8.14e-01 | 7.69e-01 | 7.09e-01 | 6.50e-01 | 5.60e-01 | 5.03e-01 |
| 1e-01 | pod    | 64  | 9.19e-01 | 8.53e-01 | 8.14e-01 | 7.69e-01 | 7.09e-01 | 6.49e-01 | 5.59e-01 | 5.03e-01 |
| 1e-01 | pod    | 128 | 9.19e-01 | 8.53e-01 | 8.14e-01 | 7.69e-01 | 7.09e-01 | 6.49e-01 | 5.59e-01 | 5.03e-01 |
| 1e-01 | pod    | 256 | 9.19e-01 | 8.53e-01 | 8.14e-01 | 7.69e-01 | 7.09e-01 | 6.49e-01 | 5.59e-01 | 5.03e-01 |
| 1e-01 | pod    | 512 | 9.19e-01 | 8.53e-01 | 8.14e-01 | 7.69e-01 | 7.09e-01 | 6.49e-01 | 5.59e-01 | --       |
| 1e-02 | coord  | 32  | 1.19e+00 | 3.14e-01 | 3.68e-01 | 1.96e-01 | 5.13e-01 | 1.62e-01 | 2.50e-01 | 3.76e-01 |
| 1e-02 | coord  | 64  | 1.30e+00 | 3.43e-01 | 4.28e-01 | 2.10e-01 | 4.94e-01 | 1.72e-01 | 2.85e-01 | 4.92e-01 |
| 1e-02 | coord  | 128 | 1.26e+00 | 3.58e-01 | 4.58e-01 | 2.12e-01 | 5.17e-01 | 1.72e-01 | 2.95e-01 | 5.42e-01 |
| 1e-02 | coord  | 256 | 1.27e+00 | 3.65e-01 | 5.09e-01 | 2.23e-01 | 5.30e-01 | 1.86e-01 | 3.03e-01 | 4.64e-01 |
| 1e-02 | coord  | 512 | 1.26e+00 | 3.65e-01 | 7.29e-01 | 2.18e-01 | 4.07e-01 | 1.87e-01 | 2.96e-01 | 5.31e-01 |
| 1e-02 | pod    | 32  | 9.19e-01 | 8.53e-01 | 8.14e-01 | 7.69e-01 | 7.09e-01 | 6.50e-01 | 5.60e-01 | 5.03e-01 |
| 1e-02 | pod    | 64  | 9.19e-01 | 8.53e-01 | 8.14e-01 | 7.69e-01 | 7.09e-01 | 6.49e-01 | 5.59e-01 | 5.03e-01 |
| 1e-02 | pod    | 128 | 9.19e-01 | 8.53e-01 | 8.14e-01 | 7.69e-01 | 7.09e-01 | 6.49e-01 | 5.59e-01 | 5.03e-01 |
| 1e-02 | pod    | 256 | 9.19e-01 | 8.53e-01 | 8.14e-01 | 7.69e-01 | 7.09e-01 | 6.49e-01 | 5.59e-01 | 5.03e-01 |
| 1e-02 | pod    | 512 | 9.19e-01 | 8.53e-01 | 8.14e-01 | 7.69e-01 | 7.09e-01 | 6.49e-01 | 5.59e-01 | --       |
| 1e-03 | coord  | 32  | 1.19e+00 | 3.08e-01 | 3.62e-01 | 1.94e-01 | 5.02e-01 | 1.48e-01 | 2.14e-01 | 3.62e-01 |
| 1e-03 | coord  | 64  | 1.30e+00 | 3.33e-01 | 4.22e-01 | 1.87e-01 | 4.43e-01 | 1.48e-01 | 2.44e-01 | 4.79e-01 |
| 1e-03 | coord  | 128 | 1.26e+00 | 3.46e-01 | 4.46e-01 | 1.99e-01 | 4.78e-01 | 1.51e-01 | 2.54e-01 | 5.25e-01 |
| 1e-03 | coord  | 256 | 1.27e+00 | 3.53e-01 | 5.04e-01 | 2.01e-01 | 5.17e-01 | 1.60e-01 | 2.74e-01 | 4.54e-01 |
| 1e-03 | coord  | 512 | 1.26e+00 | 3.53e-01 | 7.19e-01 | 2.09e-01 | 3.86e-01 | 1.64e-01 | 2.57e-01 | 5.19e-01 |
| 1e-03 | pod    | 32  | 9.19e-01 | 8.53e-01 | 8.14e-01 | 7.69e-01 | 7.09e-01 | 6.50e-01 | 5.60e-01 | 5.03e-01 |
| 1e-03 | pod    | 64  | 9.19e-01 | 8.53e-01 | 8.14e-01 | 7.69e-01 | 7.09e-01 | 6.49e-01 | 5.59e-01 | 5.03e-01 |
| 1e-03 | pod    | 128 | 9.19e-01 | 8.53e-01 | 8.14e-01 | 7.69e-01 | 7.09e-01 | 6.49e-01 | 5.59e-01 | 5.03e-01 |
| 1e-03 | pod    | 256 | 9.19e-01 | 8.53e-01 | 8.14e-01 | 7.69e-01 | 7.09e-01 | 6.49e-01 | 5.59e-01 | 5.03e-01 |
| 1e-03 | pod    | 512 | 9.19e-01 | 8.53e-01 | 8.14e-01 | 7.69e-01 | 7.09e-01 | 6.49e-01 | 5.59e-01 | --       |
<!-- END GENERATED: resid_poisson2d -->

<!-- BEGIN GENERATED: resid_burgers2d -->
| tau   | method | N   | k=2      | k=4      | k=6      | k=8      | k=12     | k=16     | k=24     | k=32     |
|-------|--------|-----|----------|----------|----------|----------|----------|----------|----------|----------|
| 1e-01 | coord  | 32  | 8.17e-02 | 7.26e-03 | 3.34e-03 | 3.51e-03 | 3.09e-03 | 1.51e-02 | 1.17e-02 | 7.70e-03 |
| 1e-01 | coord  | 64  | 6.22e-02 | 1.12e-02 | 5.74e-03 | 5.18e-03 | 4.94e-03 | 1.13e-02 | 2.48e-02 | 6.84e-03 |
| 1e-01 | coord  | 128 | 5.73e-01 | 1.59e-02 | 1.03e-02 | 9.21e-03 | 9.09e-03 | 4.80e-02 | 6.69e-02 | 1.37e-02 |
| 1e-01 | coord  | 256 | 1.42e+00 | 1.99e-02 | 1.45e-02 | 1.27e-02 | 1.20e-02 | 3.81e-02 | 2.31e-01 | 1.48e-02 |
| 1e-01 | pod    | 32  | 7.01e-03 | 8.62e-03 | 9.06e-03 | 9.56e-03 | 8.61e-03 | 8.14e-03 | 6.98e-03 | 5.70e-03 |
| 1e-01 | pod    | 64  | 1.07e-02 | 1.32e-02 | 1.43e-02 | 1.48e-02 | 1.39e-02 | 1.39e-02 | 1.25e-02 | 1.00e-02 |
| 1e-01 | pod    | 128 | 2.06e-02 | 2.50e-02 | 2.77e-02 | 2.82e-02 | 2.71e-02 | 2.83e-02 | 2.69e-02 | 2.19e-02 |
| 1e-01 | pod    | 256 | 5.16e-02 | 6.18e-02 | 6.89e-02 | 6.96e-02 | 6.74e-02 | 7.17e-02 | 6.99e-02 | 5.75e-02 |
| 1e-02 | coord  | 32  | 8.17e-02 | 7.23e-03 | 3.20e-03 | 3.15e-03 | 2.90e-03 | 5.29e-03 | 9.03e-03 | 3.39e-03 |
| 1e-02 | coord  | 64  | 6.22e-02 | 1.11e-02 | 5.47e-03 | 4.78e-03 | 4.87e-03 | 4.99e-03 | 1.52e-02 | 6.33e-03 |
| 1e-02 | coord  | 128 | 5.73e-01 | 1.59e-02 | 1.02e-02 | 8.91e-03 | 9.06e-03 | 1.04e-02 | 1.59e-02 | 1.02e-02 |
| 1e-02 | coord  | 256 | 1.42e+00 | 1.98e-02 | 1.43e-02 | 1.23e-02 | 1.17e-02 | 1.31e-02 | 1.83e-02 | 1.43e-02 |
| 1e-02 | pod    | 32  | 7.01e-03 | 8.62e-03 | 9.06e-03 | 9.56e-03 | 8.61e-03 | 8.14e-03 | 6.98e-03 | 5.70e-03 |
| 1e-02 | pod    | 64  | 1.07e-02 | 1.32e-02 | 1.43e-02 | 1.48e-02 | 1.39e-02 | 1.39e-02 | 1.25e-02 | 1.00e-02 |
| 1e-02 | pod    | 128 | 2.06e-02 | 2.50e-02 | 2.77e-02 | 2.82e-02 | 2.71e-02 | 2.83e-02 | 2.69e-02 | 2.19e-02 |
| 1e-02 | pod    | 256 | 5.16e-02 | 6.18e-02 | 6.89e-02 | 6.96e-02 | 6.74e-02 | 7.17e-02 | 6.99e-02 | 5.75e-02 |
| 1e-03 | coord  | 32  | 8.17e-02 | 7.23e-03 | 3.20e-03 | 3.15e-03 | 2.90e-03 | 5.29e-03 | 6.63e-03 | 3.38e-03 |
| 1e-03 | coord  | 64  | 6.22e-02 | 1.11e-02 | 5.47e-03 | 4.78e-03 | 4.87e-03 | 4.99e-03 | 1.41e-02 | 6.33e-03 |
| 1e-03 | coord  | 128 | 5.73e-01 | 1.59e-02 | 1.02e-02 | 8.91e-03 | 9.06e-03 | 1.04e-02 | 1.51e-02 | 1.02e-02 |
| 1e-03 | coord  | 256 | 1.42e+00 | 1.98e-02 | 1.43e-02 | 1.23e-02 | 1.17e-02 | 1.31e-02 | 2.80e-02 | 1.43e-02 |
| 1e-03 | pod    | 32  | 7.01e-03 | 8.62e-03 | 9.06e-03 | 9.56e-03 | 8.61e-03 | 8.14e-03 | 6.98e-03 | 5.70e-03 |
| 1e-03 | pod    | 64  | 1.07e-02 | 1.32e-02 | 1.43e-02 | 1.48e-02 | 1.39e-02 | 1.39e-02 | 1.25e-02 | 1.00e-02 |
| 1e-03 | pod    | 128 | 2.06e-02 | 2.50e-02 | 2.77e-02 | 2.82e-02 | 2.71e-02 | 2.83e-02 | 2.69e-02 | 2.19e-02 |
| 1e-03 | pod    | 256 | 5.16e-02 | 6.18e-02 | 6.89e-02 | 6.96e-02 | 6.74e-02 | 7.17e-02 | 6.99e-02 | 5.75e-02 |
<!-- END GENERATED: resid_burgers2d -->

### 6.6 The iso-error Pareto frontier

Figures: `figs/ctol_pareto_poisson2d.{png,pdf}`, `figs/ctol_pareto_burgers2d.{png,pdf}`.
x = online wall time (log), y = held-out rel-L2 (log); faint scatter = every configuration,
solid line = the non-dominated envelope per method, dots labelled with `k`; the FOM is a
vertical dashed line (it is the reference truth, so its error is 0 and it is off-plot — the
line reads *the price of exactness*).

Non-dominated configurations, per mesh:

<!-- BEGIN GENERATED: pareto_poisson2d -->
| N   | method | k  | M   | m    | tau   | time ms (e2e) | err rel-L2 | jac evals | censored % | x iso-accuracy FOM | x exact FOM |
|-----|--------|----|-----|------|-------|---------------|------------|-----------|------------|--------------------|-------------|
| 32  | coord  | 4  | 64  | 256  | 1e-01 | 1.71          | 8.405e-02  | 4.4       | 0          | 0.53               | 2.1         |
| 32  | coord  | 16 | 64  | 256  | 1e-01 | 2.42          | 7.673e-02  | 4.8       | 0          | 0.37               | 1.5         |
| 32  | coord  | 16 | 64  | 256  | 1e-02 | 3.29          | 1.301e-02  | 7.3       | 0          | 0.35               | 1.1         |
| 32  | pod    | 24 | 64  | 256  | 1e-01 | 0.37          | 6.946e-02  | 2.0       | 0          | 2.48               | 9.9         |
| 32  | pod    | 32 | 256 | 900  | 1e-01 | 0.39          | 5.367e-02  | 2.0       | 0          | 2.95               | 9.4         |
| 64  | coord  | 4  | 64  | 256  | 1e-01 | 1.82          | 8.372e-02  | 4.5       | 0          | 0.99               | 3.9         |
| 64  | coord  | 16 | 64  | 256  | 1e-02 | 3.42          | 1.155e-02  | 7.5       | 0          | 0.65               | 2.1         |
| 64  | pod    | 24 | 64  | 256  | 1e-01 | 0.35          | 6.678e-02  | 2.0       | 0          | 5.12               | 20.4        |
| 64  | pod    | 32 | 256 | 1024 | 1e-01 | 0.37          | 5.105e-02  | 2.0       | 0          | 4.82               | 19.2        |
| 128 | coord  | 4  | 64  | 256  | 1e-01 | 2.06          | 8.514e-02  | 4.5       | 0          | 1.81               | 7.1         |
| 128 | coord  | 16 | 64  | 256  | 1e-02 | 3.64          | 1.150e-02  | 7.5       | 0          | 1.25               | 4.0         |
| 128 | pod    | 24 | 64  | 256  | 1e-01 | 0.41          | 6.620e-02  | 2.0       | 0          | 9.04               | 35.6        |
| 128 | pod    | 32 | 256 | 1024 | 1e-01 | 0.46          | 5.049e-02  | 2.0       | 0          | 8.20               | 32.3        |
| 256 | coord  | 4  | 64  | 256  | 1e-01 | 3.16          | 8.300e-02  | 4.5       | 0          | 1.98               | 9.4         |
| 256 | coord  | 16 | 64  | 256  | 1e-02 | 4.80          | 1.127e-02  | 7.5       | 0          | 1.63               | 6.2         |
| 256 | pod    | 24 | 64  | 256  | 1e-01 | 0.47          | 6.605e-02  | 2.0       | 0          | 13.20              | 62.9        |
| 256 | pod    | 32 | 256 | 1024 | 1e-01 | 0.75          | 5.035e-02  | 2.0       | 0          | 10.44              | 39.6        |
| 512 | coord  | 4  | 64  | 256  | 1e-01 | 7.26          | 9.325e-02  | 4.4       | 0          | 2.80               | 12.8        |
| 512 | coord  | 8  | 64  | 256  | 1e-01 | 7.74          | 8.127e-02  | 4.9       | 0          | 2.63               | 12.0        |
| 512 | coord  | 16 | 64  | 256  | 1e-02 | 8.75          | 1.178e-02  | 7.4       | 0          | 3.03               | 10.6        |
| 512 | pod    | 24 | 64  | 256  | 1e-01 | 0.77          | 6.602e-02  | 2.0       | 0          | 26.50              | 121.1       |
<!-- END GENERATED: pareto_poisson2d -->

<!-- BEGIN GENERATED: pareto_burgers2d -->
| N | method | k | M | m | tau | time ms (e2e) | err rel-L2 | jac evals | censored % | x iso-accuracy FOM | x exact FOM |
|---|--------|---|---|---|-----|---------------|------------|-----------|------------|--------------------|-------------|
<!-- END GENERATED: pareto_burgers2d -->

The **as-deployed** frontier, censored cells included (set the knob, take whatever the solver
reaches). Where it extends past the strict frontier, the extra accuracy is bought by running to
termination rather than to the declared tolerance:

<!-- BEGIN GENERATED: paretodep_poisson2d -->
| N   | method | k  | M   | m    | tau   | time ms (e2e) | err rel-L2 | jac evals | censored % | x iso-accuracy FOM | x exact FOM |
|-----|--------|----|-----|------|-------|---------------|------------|-----------|------------|--------------------|-------------|
| 32  | coord  | 4  | 64  | 256  | 1e-01 | 1.71          | 8.405e-02  | 4.4       | 0          | 0.53               | 2.1         |
| 32  | coord  | 16 | 64  | 256  | 1e-01 | 2.42          | 7.673e-02  | 4.8       | 0          | 0.37               | 1.5         |
| 32  | coord  | 8  | 64  | 256  | 1e-02 | 2.97          | 1.453e-02  | 10.1      | 6          | 0.38               | 1.2         |
| 32  | coord  | 16 | 64  | 256  | 1e-02 | 3.29          | 1.301e-02  | 7.3       | 0          | 0.35               | 1.1         |
| 32  | coord  | 8  | 64  | 256  | 1e-03 | 9.10          | 1.076e-02  | 30.1      | 100        | 0.13               | 0.4         |
| 32  | coord  | 16 | 64  | 256  | 1e-03 | 11.40         | 7.647e-03  | 30.9      | 94         | 0.11               | 0.3         |
| 32  | pod    | 12 | 64  | 256  | 1e-01 | 0.34          | 1.317e-01  | 2.8       | 44         | 2.63               | 10.5        |
| 32  | pod    | 16 | 64  | 256  | 1e-01 | 0.35          | 1.036e-01  | 2.5       | 25         | 2.60               | 10.4        |
| 32  | pod    | 24 | 64  | 256  | 1e-01 | 0.37          | 6.946e-02  | 2.0       | 0          | 2.48               | 9.9         |
| 32  | pod    | 32 | 256 | 900  | 1e-01 | 0.39          | 5.367e-02  | 2.0       | 0          | 2.95               | 9.4         |
| 32  | pod    | 32 | 256 | 900  | 1e-03 | 1.12          | 5.367e-02  | 3.8       | 100        | 1.02               | 3.2         |
| 64  | coord  | 4  | 64  | 256  | 1e-01 | 1.82          | 8.372e-02  | 4.5       | 0          | 0.99               | 3.9         |
| 64  | coord  | 8  | 64  | 256  | 1e-02 | 2.93          | 1.322e-02  | 10.4      | 6          | 0.76               | 2.4         |
| 64  | coord  | 16 | 64  | 256  | 1e-02 | 3.42          | 1.155e-02  | 7.5       | 0          | 0.65               | 2.1         |
| 64  | coord  | 8  | 64  | 256  | 1e-03 | 7.52          | 8.482e-03  | 28.2      | 100        | 0.29               | 1.0         |
| 64  | coord  | 16 | 64  | 256  | 1e-03 | 10.47         | 6.542e-03  | 29.0      | 94         | 0.21               | 0.7         |
| 64  | pod    | 12 | 64  | 256  | 1e-01 | 0.34          | 1.290e-01  | 2.8       | 44         | 5.34               | 21.3        |
| 64  | pod    | 16 | 64  | 256  | 1e-01 | 0.35          | 1.008e-01  | 2.4       | 25         | 5.20               | 20.7        |
| 64  | pod    | 24 | 64  | 256  | 1e-01 | 0.35          | 6.678e-02  | 2.0       | 0          | 5.12               | 20.4        |
| 64  | pod    | 32 | 256 | 1024 | 1e-01 | 0.37          | 5.105e-02  | 2.0       | 0          | 4.82               | 19.2        |
| 64  | pod    | 32 | 256 | 1024 | 1e-03 | 0.76          | 5.105e-02  | 3.9       | 100        | 2.36               | 9.4         |
| 128 | coord  | 4  | 64  | 256  | 1e-01 | 2.06          | 8.514e-02  | 4.5       | 0          | 1.81               | 7.1         |
| 128 | coord  | 8  | 64  | 256  | 1e-02 | 3.38          | 1.411e-02  | 10.3      | 12         | 1.34               | 4.4         |
| 128 | coord  | 16 | 64  | 256  | 1e-02 | 3.64          | 1.150e-02  | 7.5       | 0          | 1.25               | 4.0         |
| 128 | coord  | 8  | 64  | 256  | 1e-03 | 8.82          | 8.397e-03  | 27.4      | 100        | 0.51               | 1.7         |
| 128 | coord  | 16 | 64  | 256  | 1e-03 | 10.91         | 6.234e-03  | 30.3      | 94         | 0.42               | 1.4         |
| 128 | pod    | 16 | 64  | 256  | 1e-01 | 0.41          | 1.002e-01  | 2.4       | 25         | 9.13               | 36.0        |
| 128 | pod    | 24 | 64  | 256  | 1e-01 | 0.41          | 6.620e-02  | 2.0       | 0          | 9.04               | 35.6        |
| 128 | pod    | 32 | 256 | 1024 | 1e-01 | 0.46          | 5.049e-02  | 2.0       | 0          | 8.20               | 32.3        |
| 128 | pod    | 32 | 256 | 1024 | 1e-03 | 0.87          | 5.049e-02  | 3.9       | 100        | 4.31               | 17.0        |
| 256 | coord  | 4  | 64  | 256  | 1e-01 | 3.16          | 8.300e-02  | 4.5       | 0          | 1.98               | 9.4         |
| 256 | coord  | 8  | 64  | 256  | 1e-02 | 4.22          | 1.406e-02  | 10.3      | 6          | 1.86               | 7.0         |
| 256 | coord  | 16 | 64  | 256  | 1e-02 | 4.80          | 1.127e-02  | 7.5       | 0          | 1.63               | 6.2         |
| 256 | coord  | 8  | 64  | 256  | 1e-03 | 8.79          | 8.378e-03  | 30.6      | 100        | 1.08               | 3.4         |
| 256 | coord  | 16 | 64  | 256  | 1e-03 | 11.70         | 6.733e-03  | 28.8      | 94         | 0.81               | 2.5         |
| 256 | pod    | 16 | 64  | 256  | 1e-01 | 0.46          | 1.000e-01  | 2.4       | 25         | 13.68              | 65.2        |
| 256 | pod    | 24 | 64  | 256  | 1e-01 | 0.47          | 6.605e-02  | 2.0       | 0          | 13.20              | 62.9        |
| 256 | pod    | 32 | 256 | 1024 | 1e-01 | 0.75          | 5.035e-02  | 2.0       | 0          | 10.44              | 39.6        |
| 256 | pod    | 32 | 256 | 1024 | 1e-02 | 1.36          | 5.035e-02  | 3.4       | 88         | 5.76               | 21.8        |
| 256 | pod    | 32 | 256 | 1024 | 1e-03 | 1.69          | 5.035e-02  | 3.7       | 100        | 4.64               | 17.6        |
| 512 | coord  | 4  | 64  | 256  | 1e-01 | 7.26          | 9.325e-02  | 4.4       | 0          | 2.80               | 12.8        |
| 512 | coord  | 8  | 64  | 256  | 1e-01 | 7.74          | 8.127e-02  | 4.9       | 0          | 2.63               | 12.0        |
| 512 | coord  | 8  | 64  | 256  | 1e-02 | 8.37          | 1.288e-02  | 9.4       | 6          | 3.17               | 11.1        |
| 512 | coord  | 16 | 64  | 256  | 1e-02 | 8.75          | 1.178e-02  | 7.4       | 0          | 3.03               | 10.6        |
| 512 | coord  | 8  | 64  | 256  | 1e-03 | 13.63         | 8.614e-03  | 31.3      | 100        | 1.95               | 6.8         |
| 512 | coord  | 16 | 64  | 256  | 1e-03 | 15.31         | 6.388e-03  | 30.4      | 94         | 2.03               | 6.1         |
| 512 | pod    | 12 | 64  | 256  | 1e-01 | 0.59          | 1.282e-01  | 2.8       | 44         | 34.35              | 157.0       |
| 512 | pod    | 16 | 64  | 256  | 1e-01 | 0.70          | 1.000e-01  | 2.4       | 25         | 29.13              | 133.2       |
| 512 | pod    | 24 | 64  | 256  | 1e-01 | 0.77          | 6.602e-02  | 2.0       | 0          | 26.50              | 121.1       |
| 512 | pod    | 24 | 64  | 256  | 1e-03 | 1.74          | 6.602e-02  | 4.0       | 100        | 11.69              | 53.4        |
<!-- END GENERATED: paretodep_poisson2d -->

<!-- BEGIN GENERATED: paretodep_burgers2d -->
| N   | method | k  | M   | m    | tau   | time ms (e2e) | err rel-L2 | jac evals | censored % | x iso-accuracy FOM | x exact FOM |
|-----|--------|----|-----|------|-------|---------------|------------|-----------|------------|--------------------|-------------|
| 32  | coord  | 6  | 64  | 256  | 1e-01 | 103.27        | 2.099e-02  | 129.9     | 6          | 0.18               | 1.9         |
| 32  | coord  | 6  | 64  | 256  | 1e-02 | 271.10        | 1.692e-02  | 304.2     | 100        | 0.07               | 0.7         |
| 32  | coord  | 8  | 64  | 256  | 1e-02 | 333.66        | 1.623e-02  | 320.2     | 100        | 0.06               | 0.6         |
| 32  | coord  | 12 | 64  | 256  | 1e-02 | 443.31        | 1.172e-02  | 353.8     | 100        | 0.04               | 0.4         |
| 32  | coord  | 32 | 256 | 900  | 1e-02 | 6102.02       | 7.679e-03  | 640.0     | 100        | 0.00               | 0.0         |
| 32  | coord  | 32 | 256 | 900  | 1e-03 | 6104.53       | 7.568e-03  | 648.2     | 100        | 0.00               | 0.0         |
| 32  | pod    | 8  | 64  | 256  | 1e-01 | 30.81         | 2.015e-01  | 201.4     | 100        | 0.61               | 6.3         |
| 32  | pod    | 24 | 64  | 256  | 1e-01 | 35.47         | 5.462e-02  | 194.8     | 89         | 0.53               | 5.4         |
| 32  | pod    | 24 | 64  | 256  | 1e-02 | 37.85         | 5.462e-02  | 207.1     | 100        | 0.49               | 5.1         |
| 32  | pod    | 32 | 256 | 900  | 1e-01 | 41.84         | 3.778e-02  | 196.3     | 90         | 0.45               | 4.6         |
| 32  | pod    | 32 | 256 | 900  | 1e-03 | 45.30         | 3.777e-02  | 208.1     | 100        | 0.41               | 4.3         |
| 64  | coord  | 6  | 64  | 256  | 1e-01 | 106.43        | 2.238e-02  | 133.3     | 8          | 0.27               | 3.2         |
| 64  | coord  | 8  | 64  | 256  | 1e-01 | 116.72        | 2.071e-02  | 130.4     | 6          | 0.25               | 2.9         |
| 64  | coord  | 12 | 64  | 256  | 1e-01 | 138.45        | 1.936e-02  | 127.1     | 1          | 0.21               | 2.4         |
| 64  | coord  | 6  | 64  | 256  | 1e-02 | 264.66        | 1.783e-02  | 302.9     | 100        | 0.11               | 1.3         |
| 64  | coord  | 8  | 64  | 256  | 1e-02 | 316.13        | 1.579e-02  | 325.5     | 100        | 0.09               | 1.1         |
| 64  | coord  | 8  | 64  | 256  | 1e-03 | 317.27        | 1.579e-02  | 326.6     | 100        | 0.09               | 1.1         |
| 64  | coord  | 12 | 64  | 256  | 1e-02 | 434.80        | 1.265e-02  | 356.2     | 100        | 0.07               | 0.8         |
| 64  | coord  | 16 | 64  | 256  | 1e-02 | 592.45        | 1.180e-02  | 414.3     | 100        | 0.05               | 0.6         |
| 64  | coord  | 16 | 64  | 256  | 1e-03 | 594.23        | 1.177e-02  | 417.3     | 100        | 0.05               | 0.6         |
| 64  | coord  | 32 | 256 | 1024 | 1e-03 | 6415.27       | 9.383e-03  | 703.0     | 100        | 0.00               | 0.1         |
| 64  | pod    | 8  | 64  | 256  | 1e-01 | 32.19         | 2.103e-01  | 201.1     | 100        | 0.89               | 10.4        |
| 64  | pod    | 12 | 64  | 256  | 1e-02 | 33.29         | 1.368e-01  | 202.8     | 100        | 0.86               | 10.1        |
| 64  | pod    | 16 | 64  | 256  | 1e-02 | 34.19         | 9.667e-02  | 206.7     | 100        | 0.84               | 9.8         |
| 64  | pod    | 24 | 64  | 256  | 1e-01 | 37.16         | 5.937e-02  | 198.2     | 91         | 0.77               | 9.0         |
| 64  | pod    | 24 | 64  | 256  | 1e-02 | 39.77         | 5.937e-02  | 208.4     | 100        | 0.72               | 8.4         |
| 64  | pod    | 32 | 256 | 1024 | 1e-01 | 42.97         | 4.221e-02  | 196.9     | 90         | 0.67               | 7.8         |
| 64  | pod    | 32 | 256 | 1024 | 1e-03 | 45.10         | 4.221e-02  | 208.5     | 100        | 0.64               | 7.4         |
| 128 | coord  | 6  | 64  | 256  | 1e-01 | 138.96        | 2.390e-02  | 136.6     | 11         | 0.56               | 7.5         |
| 128 | coord  | 8  | 64  | 256  | 1e-01 | 146.62        | 2.386e-02  | 127.9     | 6          | 0.53               | 7.1         |
| 128 | coord  | 6  | 64  | 256  | 1e-03 | 299.53        | 1.898e-02  | 306.8     | 100        | 0.26               | 3.5         |
| 128 | coord  | 8  | 64  | 256  | 1e-02 | 342.61        | 1.672e-02  | 330.4     | 100        | 0.23               | 3.0         |
| 128 | coord  | 12 | 64  | 256  | 1e-02 | 467.09        | 1.426e-02  | 365.4     | 100        | 0.17               | 2.2         |
| 128 | coord  | 12 | 64  | 256  | 1e-03 | 467.91        | 1.426e-02  | 368.9     | 100        | 0.17               | 2.2         |
| 128 | coord  | 32 | 256 | 1024 | 1e-03 | 6256.10       | 9.640e-03  | 700.4     | 100        | 0.01               | 0.2         |
| 128 | pod    | 8  | 64  | 256  | 1e-01 | 30.99         | 2.158e-01  | 202.4     | 100        | 2.50               | 33.5        |
| 128 | pod    | 16 | 64  | 256  | 1e-01 | 35.39         | 1.010e-01  | 205.1     | 100        | 2.19               | 29.3        |
| 128 | pod    | 24 | 64  | 256  | 1e-01 | 37.05         | 6.273e-02  | 197.9     | 91         | 2.09               | 28.0        |
| 128 | pod    | 24 | 64  | 256  | 1e-02 | 38.53         | 6.273e-02  | 207.9     | 100        | 2.01               | 26.9        |
| 128 | pod    | 32 | 256 | 1024 | 1e-01 | 41.12         | 4.553e-02  | 196.6     | 90         | 1.89               | 25.2        |
| 128 | pod    | 32 | 256 | 1024 | 1e-02 | 47.64         | 4.553e-02  | 207.5     | 100        | 1.63               | 21.8        |
| 256 | coord  | 6  | 64  | 256  | 1e-01 | 276.28        | 2.468e-02  | 134.1     | 10         | 0.58               | 6.7         |
| 256 | coord  | 12 | 64  | 256  | 1e-01 | 320.31        | 2.133e-02  | 134.4     | 4          | 0.50               | 5.8         |
| 256 | coord  | 6  | 64  | 256  | 1e-02 | 471.02        | 2.000e-02  | 302.8     | 100        | 0.34               | 3.9         |
| 256 | coord  | 8  | 64  | 256  | 1e-02 | 521.31        | 1.761e-02  | 327.2     | 100        | 0.30               | 3.5         |
| 256 | coord  | 12 | 64  | 256  | 1e-02 | 668.88        | 1.489e-02  | 367.4     | 100        | 0.24               | 2.8         |
| 256 | coord  | 12 | 64  | 256  | 1e-03 | 678.56        | 1.489e-02  | 369.0     | 100        | 0.23               | 2.7         |
| 256 | coord  | 16 | 64  | 256  | 1e-02 | 824.30        | 1.451e-02  | 416.9     | 99         | 0.19               | 2.2         |
| 256 | coord  | 16 | 64  | 256  | 1e-03 | 826.21        | 1.444e-02  | 423.1     | 100        | 0.19               | 2.2         |
| 256 | coord  | 32 | 256 | 1024 | 1e-03 | 7388.05       | 1.037e-02  | 724.1     | 100        | 0.02               | 0.2         |
| 256 | pod    | 4  | 64  | 256  | 1e-03 | 31.70         | 4.012e-01  | 198.2     | 100        | 5.01               | 58.2        |
| 256 | pod    | 8  | 64  | 256  | 1e-03 | 31.77         | 2.190e-01  | 201.1     | 100        | 5.00               | 58.0        |
| 256 | pod    | 12 | 64  | 256  | 1e-02 | 35.81         | 1.459e-01  | 203.6     | 100        | 4.44               | 51.5        |
| 256 | pod    | 16 | 64  | 256  | 1e-01 | 36.58         | 1.035e-01  | 204.9     | 100        | 4.34               | 50.4        |
| 256 | pod    | 24 | 64  | 256  | 1e-01 | 39.15         | 6.497e-02  | 197.4     | 92         | 4.06               | 47.1        |
| 256 | pod    | 24 | 64  | 256  | 1e-02 | 44.69         | 6.497e-02  | 206.3     | 100        | 3.56               | 41.3        |
| 256 | pod    | 32 | 256 | 1024 | 1e-02 | 45.02         | 4.748e-02  | 208.4     | 100        | 3.53               | 41.0        |
<!-- END GENERATED: paretodep_burgers2d -->

**A third view, and the operative one for Burgers.** `censored` is an ANY predicate — true if
*any* tau-stopped solve in the cell missed its tolerance. That is 16 solves for Poisson (one per
source) but **816** for Burgers (a cold start plus 50 time steps, for each of 16 trajectories),
so the two are not the same predicate and the Burgers strict frontier is empty *by construction
rather than by failure*: at tau=1e-1, k=8, N=64 only **6.5%** of the 816 solves miss. The
frontier below thresholds the censored **fraction** at 10% instead, which is comparable across
the two PDEs.

<!-- BEGIN GENERATED: paretofrac_poisson2d -->
| N   | method | k  | M   | m    | tau   | time ms (e2e) | err rel-L2 | jac evals | censored % | x iso-accuracy FOM | x exact FOM |
|-----|--------|----|-----|------|-------|---------------|------------|-----------|------------|--------------------|-------------|
| 32  | coord  | 4  | 64  | 256  | 1e-01 | 1.71          | 8.405e-02  | 4.4       | 0          | 0.53               | 2.1         |
| 32  | coord  | 16 | 64  | 256  | 1e-01 | 2.42          | 7.673e-02  | 4.8       | 0          | 0.37               | 1.5         |
| 32  | coord  | 8  | 64  | 256  | 1e-02 | 2.97          | 1.453e-02  | 10.1      | 6          | 0.38               | 1.2         |
| 32  | coord  | 16 | 64  | 256  | 1e-02 | 3.29          | 1.301e-02  | 7.3       | 0          | 0.35               | 1.1         |
| 32  | pod    | 24 | 64  | 256  | 1e-01 | 0.37          | 6.946e-02  | 2.0       | 0          | 2.48               | 9.9         |
| 32  | pod    | 32 | 256 | 900  | 1e-01 | 0.39          | 5.367e-02  | 2.0       | 0          | 2.95               | 9.4         |
| 64  | coord  | 4  | 64  | 256  | 1e-01 | 1.82          | 8.372e-02  | 4.5       | 0          | 0.99               | 3.9         |
| 64  | coord  | 8  | 64  | 256  | 1e-02 | 2.93          | 1.322e-02  | 10.4      | 6          | 0.76               | 2.4         |
| 64  | coord  | 16 | 64  | 256  | 1e-02 | 3.42          | 1.155e-02  | 7.5       | 0          | 0.65               | 2.1         |
| 64  | pod    | 24 | 64  | 256  | 1e-01 | 0.35          | 6.678e-02  | 2.0       | 0          | 5.12               | 20.4        |
| 64  | pod    | 32 | 256 | 1024 | 1e-01 | 0.37          | 5.105e-02  | 2.0       | 0          | 4.82               | 19.2        |
| 128 | coord  | 4  | 64  | 256  | 1e-01 | 2.06          | 8.514e-02  | 4.5       | 0          | 1.81               | 7.1         |
| 128 | coord  | 16 | 64  | 256  | 1e-02 | 3.64          | 1.150e-02  | 7.5       | 0          | 1.25               | 4.0         |
| 128 | pod    | 24 | 64  | 256  | 1e-01 | 0.41          | 6.620e-02  | 2.0       | 0          | 9.04               | 35.6        |
| 128 | pod    | 32 | 256 | 1024 | 1e-01 | 0.46          | 5.049e-02  | 2.0       | 0          | 8.20               | 32.3        |
| 256 | coord  | 4  | 64  | 256  | 1e-01 | 3.16          | 8.300e-02  | 4.5       | 0          | 1.98               | 9.4         |
| 256 | coord  | 8  | 64  | 256  | 1e-02 | 4.22          | 1.406e-02  | 10.3      | 6          | 1.86               | 7.0         |
| 256 | coord  | 16 | 64  | 256  | 1e-02 | 4.80          | 1.127e-02  | 7.5       | 0          | 1.63               | 6.2         |
| 256 | pod    | 24 | 64  | 256  | 1e-01 | 0.47          | 6.605e-02  | 2.0       | 0          | 13.20              | 62.9        |
| 256 | pod    | 32 | 256 | 1024 | 1e-01 | 0.75          | 5.035e-02  | 2.0       | 0          | 10.44              | 39.6        |
| 512 | coord  | 4  | 64  | 256  | 1e-01 | 7.26          | 9.325e-02  | 4.4       | 0          | 2.80               | 12.8        |
| 512 | coord  | 8  | 64  | 256  | 1e-01 | 7.74          | 8.127e-02  | 4.9       | 0          | 2.63               | 12.0        |
| 512 | coord  | 8  | 64  | 256  | 1e-02 | 8.37          | 1.288e-02  | 9.4       | 6          | 3.17               | 11.1        |
| 512 | coord  | 16 | 64  | 256  | 1e-02 | 8.75          | 1.178e-02  | 7.4       | 0          | 3.03               | 10.6        |
| 512 | pod    | 24 | 64  | 256  | 1e-01 | 0.77          | 6.602e-02  | 2.0       | 0          | 26.50              | 121.1       |
<!-- END GENERATED: paretofrac_poisson2d -->

<!-- BEGIN GENERATED: paretofrac_burgers2d -->
| N   | method | k  | M  | m   | tau   | time ms (e2e) | err rel-L2 | jac evals | censored % | x iso-accuracy FOM | x exact FOM |
|-----|--------|----|----|-----|-------|---------------|------------|-----------|------------|--------------------|-------------|
| 32  | coord  | 6  | 64 | 256 | 1e-01 | 103.27        | 2.099e-02  | 129.9     | 6          | 0.18               | 1.9         |
| 64  | coord  | 6  | 64 | 256 | 1e-01 | 106.43        | 2.238e-02  | 133.3     | 8          | 0.27               | 3.2         |
| 64  | coord  | 8  | 64 | 256 | 1e-01 | 116.72        | 2.071e-02  | 130.4     | 6          | 0.25               | 2.9         |
| 64  | coord  | 12 | 64 | 256 | 1e-01 | 138.45        | 1.936e-02  | 127.1     | 1          | 0.21               | 2.4         |
| 128 | coord  | 8  | 64 | 256 | 1e-01 | 146.62        | 2.386e-02  | 127.9     | 6          | 0.53               | 7.1         |
| 256 | coord  | 8  | 64 | 256 | 1e-01 | 290.41        | 2.541e-02  | 136.6     | 9          | 0.55               | 6.4         |
| 256 | coord  | 12 | 64 | 256 | 1e-01 | 320.31        | 2.133e-02  | 134.4     | 4          | 0.50               | 5.8         |
<!-- END GENERATED: paretofrac_burgers2d -->

**Who owns the frontier at each mesh**

<!-- BEGIN GENERATED: owner_poisson2d -->
| N   | coord points on frontier | pod points on frontier | owner | best coord err (uncensored) | at   | best pod err (uncensored) | at   |
|-----|--------------------------|------------------------|-------|-----------------------------|------|---------------------------|------|
| 32  | 1                        | 2                      | split | 1.301e-02                   | k=16 | 5.367e-02                 | k=32 |
| 64  | 1                        | 2                      | split | 1.155e-02                   | k=16 | 5.105e-02                 | k=32 |
| 128 | 1                        | 2                      | split | 1.150e-02                   | k=16 | 5.049e-02                 | k=32 |
| 256 | 1                        | 2                      | split | 1.127e-02                   | k=16 | 5.035e-02                 | k=32 |
| 512 | 1                        | 1                      | split | 1.178e-02                   | k=16 | 6.602e-02                 | k=24 |
<!-- END GENERATED: owner_poisson2d -->

<!-- BEGIN GENERATED: owner_burgers2d -->
| N   | coord points on frontier | pod points on frontier | owner | best coord err (uncensored) | at | best pod err (uncensored) | at |
|-----|--------------------------|------------------------|-------|-----------------------------|----|---------------------------|----|
| 32  | 0                        | 0                      | --    | --                          | -- | --                        | -- |
| 64  | 0                        | 0                      | --    | --                          | -- | --                        | -- |
| 128 | 0                        | 0                      | --    | --                          | -- | --                        | -- |
| 256 | 0                        | 0                      | --    | --                          | -- | --                        | -- |
<!-- END GENERATED: owner_burgers2d -->

### 6.7 Scaling: cheapest time reaching a fixed error target vs N

**Read the N=32 point with caution, in both arms.** At the coarsest mesh both the ROM and the
FOM are dominated by per-iteration kernel-launch overhead rather than by floating-point work —
which is exactly why the FOM anchor fails there (above). Mesh-independence is a claim about the
*flat* part of the curve; the leftmost point is the least informative, not the most.


Figures: `figs/ctol_scaling_poisson2d.{png,pdf}`, `figs/ctol_scaling_burgers2d.{png,pdf}`.
Mesh independence appears as a flat line while the FOM's rises. The configuration is chosen
from the panel accuracies; the time is the single-GPU consolidation timing.

<!-- BEGIN GENERATED: scaling_poisson2d -->
_timing source: single-GPU consolidation run_

| target rel-L2 | N   | coord ms                           | pod ms    | FOM ms (iso-accuracy) | FOM ms (exact) | best ROM speedup vs iso-accuracy FOM |
|---------------|-----|------------------------------------|-----------|-----------------------|----------------|--------------------------------------|
| 2e-02         | 32  | 3.29 (k=16, tau=1e-02, PANEL TIME) | unreached | 1.14                  | 3.61           | 0.35                                 |
| 2e-02         | 64  | 3.42 (k=16, tau=1e-02, PANEL TIME) | unreached | 2.21                  | 7.17           | 0.65                                 |
| 2e-02         | 128 | 3.64 (k=16, tau=1e-02, PANEL TIME) | unreached | 3.74                  | 14.74          | 1.03                                 |
| 2e-02         | 256 | 4.80 (k=16, tau=1e-02, PANEL TIME) | unreached | 7.85                  | 29.75          | 1.63                                 |
| 2e-02         | 512 | 8.75 (k=16, tau=1e-02, PANEL TIME) | unreached | 26.54                 | 93.06          | 3.03                                 |
| 1e-02         | 32  | unreached                          | unreached | 1.14                  | 3.61           | --                                   |
| 1e-02         | 64  | unreached                          | unreached | 2.21                  | 7.17           | --                                   |
| 1e-02         | 128 | unreached                          | unreached | 4.53                  | 14.74          | --                                   |
| 1e-02         | 256 | unreached                          | unreached | 9.49                  | 29.75          | --                                   |
| 1e-02         | 512 | unreached                          | unreached | 26.54                 | 93.06          | --                                   |
<!-- END GENERATED: scaling_poisson2d -->

<!-- BEGIN GENERATED: scaling_burgers2d -->
_timing source: PANEL JOBS (different GPUs) -- NOT cross-N comparable_

| target rel-L2 | N   | coord ms  | pod ms    | FOM ms (iso-accuracy) | FOM ms (exact) | best ROM speedup vs iso-accuracy FOM |
|---------------|-----|-----------|-----------|-----------------------|----------------|--------------------------------------|
| 5e-02         | 32  | unreached | unreached | 18.68                 | 193.24         | --                                   |
| 5e-02         | 64  | unreached | unreached | 28.77                 | 335.75         | --                                   |
| 5e-02         | 128 | unreached | unreached | 77.54                 | 1037.32        | --                                   |
| 5e-02         | 256 | unreached | unreached | 158.89                | 1844.33        | --                                   |
| 2e-02         | 32  | unreached | unreached | 18.68                 | 193.24         | --                                   |
| 2e-02         | 64  | unreached | unreached | 28.77                 | 335.75         | --                                   |
| 2e-02         | 128 | unreached | unreached | 77.54                 | 1037.32        | --                                   |
| 2e-02         | 256 | unreached | unreached | 158.89                | 1844.33        | --                                   |
<!-- END GENERATED: scaling_burgers2d -->

**Panel vs single-GPU consolidation cross-check** (the same configuration timed twice; a large
ratio means the panel GPUs differed materially and only the consolidated column may be read
across meshes)

<!-- BEGIN GENERATED: consolidation -->
| pde       | method | N   | k  | tau   | panel ms | consolidated ms | ratio | panel err | consolidated err | node   |
|-----------|--------|-----|----|-------|----------|-----------------|-------|-----------|------------------|--------|
| poisson2d | pod    | 512 | 32 | 1e-01 | --       | 1.12            | --    | --        | 5.032e-02        | pax051 |
| poisson2d | pod    | 512 | 32 | 1e-02 | --       | 1.35            | --    | --        | 5.032e-02        | pax051 |
| poisson2d | pod    | 512 | 32 | 1e-03 | --       | 1.35            | --    | --        | 5.032e-02        | pax051 |
<!-- END GENERATED: consolidation -->

### 6.8 Hyper-reduction fit quality and supplementary arms

NNLS-EQ relative fit per cell (at the first tau; the fit does not depend on tau):

<!-- BEGIN GENERATED: eqfit_poisson2d -->
| method | N   | k=2      | k=4      | k=6      | k=8      | k=12     | k=16     | k=24     | k=32     |
|--------|-----|----------|----------|----------|----------|----------|----------|----------|----------|
| coord  | 32  | 4.72e-03 | 1.93e-03 | 1.66e-03 | 1.50e-03 | 1.51e-03 | 1.34e-03 | 1.39e-03 | 2.61e-15 |
| coord  | 64  | 4.34e-03 | 1.70e-03 | 1.56e-03 | 1.31e-03 | 1.46e-03 | 1.37e-03 | 1.27e-03 | 1.02e-04 |
| coord  | 128 | 4.52e-03 | 1.90e-03 | 1.56e-03 | 1.49e-03 | 1.34e-03 | 1.30e-03 | 1.29e-03 | 1.26e-04 |
| coord  | 256 | 4.40e-03 | 1.83e-03 | 1.56e-03 | 1.54e-03 | 1.46e-03 | 1.26e-03 | 1.50e-03 | 1.20e-04 |
| coord  | 512 | 4.77e-03 | 1.60e-03 | 1.56e-03 | 1.45e-03 | 1.46e-03 | 1.27e-03 | 1.32e-03 | 1.36e-04 |
| pod    | 32  | 2.63e-07 | 3.17e-06 | 4.46e-06 | 6.98e-06 | 2.56e-05 | 3.93e-05 | 9.68e-05 | 1.53e-06 |
| pod    | 64  | 1.49e-06 | 5.59e-06 | 6.84e-06 | 8.63e-06 | 2.31e-05 | 4.15e-05 | 8.92e-05 | 1.40e-05 |
| pod    | 128 | 1.43e-06 | 5.24e-06 | 7.05e-06 | 7.93e-06 | 2.26e-05 | 4.26e-05 | 9.47e-05 | 1.59e-05 |
| pod    | 256 | 1.87e-06 | 5.23e-06 | 6.95e-06 | 7.21e-06 | 2.26e-05 | 4.85e-05 | 9.38e-05 | 1.31e-05 |
| pod    | 512 | 6.90e-07 | 5.24e-06 | 5.97e-06 | 8.26e-06 | 2.06e-05 | 4.19e-05 | 9.85e-05 | --       |
<!-- END GENERATED: eqfit_poisson2d -->

<!-- BEGIN GENERATED: eqfit_burgers2d -->
| method | N   | k=2      | k=4      | k=6      | k=8      | k=12     | k=16     | k=24     | k=32     |
|--------|-----|----------|----------|----------|----------|----------|----------|----------|----------|
| coord  | 32  | 3.71e-02 | 8.48e-03 | 6.52e-03 | 7.02e-03 | 6.96e-03 | 7.53e-03 | 7.72e-03 | 2.54e-15 |
| coord  | 64  | 3.75e-02 | 9.10e-03 | 7.64e-03 | 6.66e-03 | 7.41e-03 | 8.93e-03 | 9.03e-03 | 3.85e-03 |
| coord  | 128 | 4.76e-02 | 8.69e-03 | 7.54e-03 | 7.41e-03 | 7.96e-03 | 7.95e-03 | 8.89e-03 | 4.58e-03 |
| coord  | 256 | 5.49e-02 | 9.35e-03 | 6.83e-03 | 7.30e-03 | 8.29e-03 | 8.63e-03 | 9.51e-03 | 4.40e-03 |
| pod    | 32  | 5.49e-05 | 2.07e-04 | 3.60e-04 | 3.82e-04 | 5.64e-04 | 6.98e-04 | 1.29e-03 | 2.37e-15 |
| pod    | 64  | 3.69e-05 | 1.49e-04 | 2.29e-04 | 2.93e-04 | 4.96e-04 | 6.99e-04 | 1.33e-03 | 6.02e-05 |
| pod    | 128 | 2.28e-05 | 1.14e-04 | 1.73e-04 | 2.33e-04 | 4.30e-04 | 6.88e-04 | 1.17e-03 | 1.89e-04 |
| pod    | 256 | 1.42e-05 | 8.63e-05 | 1.41e-04 | 1.91e-04 | 4.23e-04 | 6.65e-04 | 1.47e-03 | 2.04e-04 |
<!-- END GENERATED: eqfit_burgers2d -->

POD projection floor on the held-out test set — the oracle bound no POD solver can beat
(Burgers only; the Poisson POD basis is built from the same 512 training sources):

<!-- BEGIN GENERATED: podfloor -->
| pde       | N   | snapshots | orthonorm dev | k=2       | k=4       | k=6       | k=8       | k=12      | k=16      | k=24      | k=32      |
|-----------|-----|-----------|---------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| burgers2d | 128 | 6656      | 2.0e-14       | 6.050e-01 | 3.805e-01 | 2.623e-01 | 2.029e-01 | 1.322e-01 | 9.462e-02 | 5.911e-02 | 4.205e-02 |
| burgers2d | 256 | 6656      | 1.8e-14       | 6.068e-01 | 3.830e-01 | 2.650e-01 | 2.057e-01 | 1.349e-01 | 9.679e-02 | 6.091e-02 | 4.368e-02 |
| burgers2d | 32  | 6656      | 2.2e-15       | 5.954e-01 | 3.683e-01 | 2.493e-01 | 1.901e-01 | 1.200e-01 | 8.515e-02 | 5.158e-02 | 3.558e-02 |
| burgers2d | 64  | 6656      | 1.8e-15       | 6.015e-01 | 3.760e-01 | 2.574e-01 | 1.981e-01 | 1.274e-01 | 9.092e-02 | 5.611e-02 | 3.929e-02 |
<!-- END GENERATED: podfloor -->

Supplementary arms: `pod_direct` = the exact linear POD minimiser as one pinv matvec;
`coord_meshfree_pool` = the headline meshfree EQ pool at `k`=8; `*_uncapped_pool` = the
`cap_control` arms that refit EQ with every interior node as a candidate:

<!-- BEGIN GENERATED: supplementary -->
| pde       | arm method           | N   | k  | M   | m    | tau   | e2e ms  | solve ms | err rel-L2 | censored % | EQ rel fit | arm             |
|-----------|----------------------|-----|----|-----|------|-------|---------|----------|------------|------------|------------|-----------------|
| burgers2d | pod_projection_floor | 128 | -- | --  | --   | --    | --      | --       | --         | --         | --         | oracle_floor    |
| burgers2d | oracle_ceiling       | 128 | 2  | 64  | 256  | --    | --      | --       | 6.803e-01  | --         | --         | ceiling         |
| burgers2d | oracle_ceiling       | 128 | 4  | 64  | 256  | --    | --      | --       | 6.057e-02  | --         | --         | ceiling         |
| burgers2d | oracle_ceiling       | 128 | 6  | 64  | 256  | --    | --      | --       | 1.741e-02  | --         | --         | ceiling         |
| burgers2d | oracle_ceiling       | 128 | 8  | 64  | 256  | --    | --      | --       | 1.456e-02  | --         | --         | ceiling         |
| burgers2d | oracle_ceiling       | 128 | 12 | 64  | 256  | --    | --      | --       | 1.237e-02  | --         | --         | ceiling         |
| burgers2d | oracle_ceiling       | 128 | 16 | 64  | 256  | --    | --      | --       | 1.197e-02  | --         | --         | ceiling         |
| burgers2d | oracle_ceiling       | 128 | 24 | 64  | 256  | --    | --      | --       | 9.736e-03  | --         | --         | ceiling         |
| burgers2d | oracle_ceiling       | 128 | 32 | 256 | 1024 | --    | --      | --       | 8.928e-03  | --         | --         | ceiling         |
| burgers2d | coord_uncapped_pool  | 128 | 8  | 64  | 256  | 1e-01 | 156.619 | 115.128  | 2.379e-02  | 7          | 7.39e-03   | cap_control     |
| burgers2d | pod_uncapped_pool    | 128 | 8  | 64  | 256  | 1e-01 | 30.775  | 30.729   | 2.158e-01  | 100        | 2.09e-04   | cap_control     |
| burgers2d | pod_projection_floor | 256 | -- | --  | --   | --    | --      | --       | --         | --         | --         | oracle_floor    |
| burgers2d | oracle_ceiling       | 256 | 2  | 64  | 256  | --    | --      | --       | 6.858e-01  | --         | --         | ceiling         |
| burgers2d | oracle_ceiling       | 256 | 4  | 64  | 256  | --    | --      | --       | 6.127e-02  | --         | --         | ceiling         |
| burgers2d | oracle_ceiling       | 256 | 6  | 64  | 256  | --    | --      | --       | 1.888e-02  | --         | --         | ceiling         |
| burgers2d | oracle_ceiling       | 256 | 8  | 64  | 256  | --    | --      | --       | 1.611e-02  | --         | --         | ceiling         |
| burgers2d | oracle_ceiling       | 256 | 12 | 64  | 256  | --    | --      | --       | 1.372e-02  | --         | --         | ceiling         |
| burgers2d | oracle_ceiling       | 256 | 16 | 64  | 256  | --    | --      | --       | 1.362e-02  | --         | --         | ceiling         |
| burgers2d | oracle_ceiling       | 256 | 24 | 64  | 256  | --    | --      | --       | 1.161e-02  | --         | --         | ceiling         |
| burgers2d | oracle_ceiling       | 256 | 32 | 256 | 1024 | --    | --      | --       | 1.063e-02  | --         | --         | ceiling         |
| burgers2d | pod_projection_floor | 32  | -- | --  | --   | --    | --      | --       | --         | --         | --         | oracle_floor    |
| burgers2d | oracle_ceiling       | 32  | 2  | 64  | 256  | --    | --      | --       | 6.949e-01  | --         | --         | ceiling         |
| burgers2d | oracle_ceiling       | 32  | 4  | 64  | 256  | --    | --      | --       | 5.097e-02  | --         | --         | ceiling         |
| burgers2d | oracle_ceiling       | 32  | 6  | 64  | 256  | --    | --      | --       | 1.476e-02  | --         | --         | ceiling         |
| burgers2d | oracle_ceiling       | 32  | 8  | 64  | 256  | --    | --      | --       | 1.232e-02  | --         | --         | ceiling         |
| burgers2d | oracle_ceiling       | 32  | 12 | 64  | 256  | --    | --      | --       | 9.831e-03  | --         | --         | ceiling         |
| burgers2d | oracle_ceiling       | 32  | 16 | 64  | 256  | --    | --      | --       | 8.529e-03  | --         | --         | ceiling         |
| burgers2d | oracle_ceiling       | 32  | 24 | 64  | 256  | --    | --      | --       | 7.661e-03  | --         | --         | ceiling         |
| burgers2d | oracle_ceiling       | 32  | 32 | 256 | 900  | --    | --      | --       | 7.972e-03  | --         | --         | ceiling         |
| burgers2d | pod_projection_floor | 64  | -- | --  | --   | --    | --      | --       | --         | --         | --         | oracle_floor    |
| burgers2d | oracle_ceiling       | 64  | 2  | 64  | 256  | --    | --      | --       | 6.752e-01  | --         | --         | ceiling         |
| burgers2d | oracle_ceiling       | 64  | 4  | 64  | 256  | --    | --      | --       | 5.541e-02  | --         | --         | ceiling         |
| burgers2d | oracle_ceiling       | 64  | 6  | 64  | 256  | --    | --      | --       | 1.517e-02  | --         | --         | ceiling         |
| burgers2d | oracle_ceiling       | 64  | 8  | 64  | 256  | --    | --      | --       | 1.235e-02  | --         | --         | ceiling         |
| burgers2d | oracle_ceiling       | 64  | 12 | 64  | 256  | --    | --      | --       | 1.050e-02  | --         | --         | ceiling         |
| burgers2d | oracle_ceiling       | 64  | 16 | 64  | 256  | --    | --      | --       | 8.975e-03  | --         | --         | ceiling         |
| burgers2d | oracle_ceiling       | 64  | 24 | 64  | 256  | --    | --      | --       | 7.483e-03  | --         | --         | ceiling         |
| burgers2d | oracle_ceiling       | 64  | 32 | 256 | 1024 | --    | --      | --       | 7.772e-03  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 32  | 2  | 64  | None | --    | --      | --       | 1.250e-01  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 32  | 4  | 64  | None | --    | --      | --       | 1.724e-02  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 32  | 6  | 64  | None | --    | --      | --       | 1.047e-02  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 32  | 8  | 64  | None | --    | --      | --       | 7.966e-03  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 32  | 12 | 64  | None | --    | --      | --       | 7.088e-03  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 32  | 16 | 64  | None | --    | --      | --       | 4.877e-03  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 32  | 24 | 64  | None | --    | --      | --       | 4.424e-03  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 32  | 32 | 256 | None | --    | --      | --       | 3.596e-03  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 64  | 2  | 64  | None | --    | --      | --       | 1.236e-01  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 64  | 4  | 64  | None | --    | --      | --       | 1.551e-02  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 64  | 6  | 64  | None | --    | --      | --       | 8.835e-03  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 64  | 8  | 64  | None | --    | --      | --       | 7.043e-03  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 64  | 12 | 64  | None | --    | --      | --       | 6.236e-03  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 64  | 16 | 64  | None | --    | --      | --       | 4.133e-03  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 64  | 24 | 64  | None | --    | --      | --       | 3.976e-03  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 64  | 32 | 256 | None | --    | --      | --       | 3.280e-03  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 128 | 2  | 64  | None | --    | --      | --       | 1.233e-01  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 128 | 4  | 64  | None | --    | --      | --       | 1.525e-02  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 128 | 6  | 64  | None | --    | --      | --       | 8.637e-03  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 128 | 8  | 64  | None | --    | --      | --       | 6.951e-03  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 128 | 12 | 64  | None | --    | --      | --       | 6.137e-03  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 128 | 16 | 64  | None | --    | --      | --       | 4.072e-03  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 128 | 24 | 64  | None | --    | --      | --       | 3.930e-03  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 128 | 32 | 256 | None | --    | --      | --       | 3.226e-03  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 256 | 2  | 64  | None | --    | --      | --       | 1.233e-01  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 256 | 4  | 64  | None | --    | --      | --       | 1.519e-02  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 256 | 6  | 64  | None | --    | --      | --       | 8.596e-03  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 256 | 8  | 64  | None | --    | --      | --       | 6.933e-03  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 256 | 12 | 64  | None | --    | --      | --       | 6.117e-03  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 256 | 16 | 64  | None | --    | --      | --       | 4.062e-03  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 256 | 24 | 64  | None | --    | --      | --       | 3.922e-03  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 256 | 32 | 256 | None | --    | --      | --       | 3.220e-03  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 512 | 2  | 64  | None | --    | --      | --       | 1.233e-01  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 512 | 4  | 64  | None | --    | --      | --       | 1.518e-02  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 512 | 6  | 64  | None | --    | --      | --       | 8.586e-03  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 512 | 8  | 64  | None | --    | --      | --       | 6.929e-03  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 512 | 12 | 64  | None | --    | --      | --       | 6.112e-03  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 512 | 16 | 64  | None | --    | --      | --       | 4.059e-03  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 512 | 24 | 64  | None | --    | --      | --       | 3.920e-03  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 512 | 32 | 256 | None | --    | --      | --       | 3.219e-03  | --         | --         | ceiling         |
| poisson2d | oracle_ceiling       | 128 | 2  | 64  | 256  | --    | --      | --       | 1.233e-01  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 128 | 2  | 64  | 256  | --    | 0.267   | 0.051    | 4.566e-01  | --         | --         | primary         |
| poisson2d | oracle_ceiling       | 128 | 4  | 64  | 256  | --    | --      | --       | 1.525e-02  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 128 | 4  | 64  | 256  | --    | 0.271   | 0.049    | 2.908e-01  | --         | --         | primary         |
| poisson2d | oracle_ceiling       | 128 | 6  | 64  | 256  | --    | --      | --       | 8.637e-03  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 128 | 6  | 64  | 256  | --    | 0.266   | 0.046    | 2.195e-01  | --         | --         | primary         |
| poisson2d | oracle_ceiling       | 128 | 8  | 64  | 256  | --    | --      | --       | 6.951e-03  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 128 | 8  | 64  | 256  | --    | 0.275   | 0.047    | 1.763e-01  | --         | --         | primary         |
| poisson2d | pod_direct           | 128 | 8  | 256 | 256  | --    | 0.295   | 0.048    | 1.763e-01  | --         | --         | supp_M256       |
| poisson2d | oracle_ceiling       | 128 | 12 | 64  | 256  | --    | --      | --       | 6.137e-03  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 128 | 12 | 64  | 256  | --    | 0.275   | 0.046    | 1.284e-01  | --         | --         | primary         |
| poisson2d | oracle_ceiling       | 128 | 16 | 64  | 256  | --    | --      | --       | 4.072e-03  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 128 | 16 | 64  | 256  | --    | 0.284   | 0.055    | 1.002e-01  | --         | --         | primary         |
| poisson2d | oracle_ceiling       | 128 | 24 | 64  | 256  | --    | --      | --       | 3.930e-03  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 128 | 24 | 64  | 256  | --    | 0.278   | 0.048    | 6.620e-02  | --         | --         | primary         |
| poisson2d | pod_direct           | 128 | 32 | 256 | 256  | --    | 0.286   | 0.053    | 5.183e-02  | --         | --         | artefact_m_eq_M |
| poisson2d | oracle_ceiling       | 128 | 32 | 256 | 1024 | --    | --      | --       | 3.226e-03  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 128 | 32 | 256 | 1024 | --    | 0.298   | 0.065    | 5.049e-02  | --         | --         | primary         |
| poisson2d | coord_meshfree_pool  | 128 | 8  | 64  | 256  | 1e-01 | 2.450   | --       | 9.083e-02  | 0          | 1.51e-03   | pool_control    |
| poisson2d | coord_meshfree_pool  | 128 | 8  | 64  | 256  | 1e-02 | 3.411   | --       | 1.311e-02  | 12         | 1.51e-03   | pool_control    |
| poisson2d | coord_meshfree_pool  | 128 | 8  | 64  | 256  | 1e-03 | 7.393   | --       | 8.551e-03  | 100        | 1.51e-03   | pool_control    |
| poisson2d | coord_uncapped_pool  | 128 | 8  | 64  | 256  | 1e-01 | 2.392   | --       | 9.791e-02  | 0          | 1.38e-03   | cap_control     |
| poisson2d | coord_uncapped_pool  | 128 | 8  | 64  | 256  | 1e-02 | 3.252   | --       | 1.313e-02  | 6          | 1.38e-03   | cap_control     |
| poisson2d | coord_uncapped_pool  | 128 | 8  | 64  | 256  | 1e-03 | 6.935   | --       | 8.568e-03  | 100        | 1.38e-03   | cap_control     |
| poisson2d | pod_uncapped_pool    | 128 | 8  | 64  | 256  | 1e-01 | 0.953   | --       | 1.763e-01  | 75         | 9.23e-06   | cap_control     |
| poisson2d | pod_uncapped_pool    | 128 | 8  | 64  | 256  | 1e-02 | 1.092   | --       | 1.763e-01  | 100        | 9.23e-06   | cap_control     |
| poisson2d | pod_uncapped_pool    | 128 | 8  | 64  | 256  | 1e-03 | 1.119   | --       | 1.763e-01  | 100        | 9.23e-06   | cap_control     |
| poisson2d | oracle_ceiling       | 256 | 2  | 64  | 256  | --    | --      | --       | 1.233e-01  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 256 | 2  | 64  | 256  | --    | 0.388   | 0.053    | 4.565e-01  | --         | --         | primary         |
| poisson2d | oracle_ceiling       | 256 | 4  | 64  | 256  | --    | --      | --       | 1.519e-02  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 256 | 4  | 64  | 256  | --    | 0.390   | 0.056    | 2.907e-01  | --         | --         | primary         |
| poisson2d | oracle_ceiling       | 256 | 6  | 64  | 256  | --    | --      | --       | 8.596e-03  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 256 | 6  | 64  | 256  | --    | 0.392   | 0.053    | 2.193e-01  | --         | --         | primary         |
| poisson2d | oracle_ceiling       | 256 | 8  | 64  | 256  | --    | --      | --       | 6.933e-03  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 256 | 8  | 64  | 256  | --    | 0.374   | 0.054    | 1.761e-01  | --         | --         | primary         |
| poisson2d | pod_direct           | 256 | 8  | 256 | 256  | --    | 0.446   | 0.056    | 1.761e-01  | --         | --         | supp_M256       |
| poisson2d | oracle_ceiling       | 256 | 12 | 64  | 256  | --    | --      | --       | 6.117e-03  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 256 | 12 | 64  | 256  | --    | 0.372   | 0.051    | 1.282e-01  | --         | --         | primary         |
| poisson2d | oracle_ceiling       | 256 | 16 | 64  | 256  | --    | --      | --       | 4.062e-03  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 256 | 16 | 64  | 256  | --    | 0.396   | 0.053    | 1.000e-01  | --         | --         | primary         |
| poisson2d | oracle_ceiling       | 256 | 24 | 64  | 256  | --    | --      | --       | 3.922e-03  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 256 | 24 | 64  | 256  | --    | 0.395   | 0.049    | 6.605e-02  | --         | --         | primary         |
| poisson2d | pod_direct           | 256 | 32 | 256 | 256  | --    | 0.471   | 0.053    | 5.165e-02  | --         | --         | artefact_m_eq_M |
| poisson2d | oracle_ceiling       | 256 | 32 | 256 | 1024 | --    | --      | --       | 3.220e-03  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 256 | 32 | 256 | 1024 | --    | 0.488   | 0.072    | 5.035e-02  | --         | --         | primary         |
| poisson2d | coord_meshfree_pool  | 256 | 8  | 64  | 256  | 1e-01 | 3.500   | --       | 9.079e-02  | 0          | 1.51e-03   | pool_control    |
| poisson2d | coord_meshfree_pool  | 256 | 8  | 64  | 256  | 1e-02 | 4.430   | --       | 1.309e-02  | 12         | 1.51e-03   | pool_control    |
| poisson2d | coord_meshfree_pool  | 256 | 8  | 64  | 256  | 1e-03 | 8.419   | --       | 8.511e-03  | 100        | 1.51e-03   | pool_control    |
| poisson2d | oracle_ceiling       | 32  | 2  | 64  | 256  | --    | --      | --       | 1.250e-01  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 32  | 2  | 64  | 256  | --    | 0.269   | 0.049    | 4.586e-01  | --         | --         | primary         |
| poisson2d | oracle_ceiling       | 32  | 4  | 64  | 256  | --    | --      | --       | 1.724e-02  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 32  | 4  | 64  | 256  | --    | 0.266   | 0.052    | 2.936e-01  | --         | --         | primary         |
| poisson2d | oracle_ceiling       | 32  | 6  | 64  | 256  | --    | --      | --       | 1.047e-02  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 32  | 6  | 64  | 256  | --    | 0.273   | 0.056    | 2.225e-01  | --         | --         | primary         |
| poisson2d | oracle_ceiling       | 32  | 8  | 64  | 256  | --    | --      | --       | 7.966e-03  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 32  | 8  | 64  | 256  | --    | 0.267   | 0.052    | 1.795e-01  | --         | --         | primary         |
| poisson2d | pod_direct           | 32  | 8  | 256 | 256  | --    | 0.265   | 0.056    | 1.795e-01  | --         | --         | supp_M256       |
| poisson2d | oracle_ceiling       | 32  | 12 | 64  | 256  | --    | --      | --       | 7.088e-03  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 32  | 12 | 64  | 256  | --    | 0.267   | 0.050    | 1.317e-01  | --         | --         | primary         |
| poisson2d | oracle_ceiling       | 32  | 16 | 64  | 256  | --    | --      | --       | 4.877e-03  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 32  | 16 | 64  | 256  | --    | 0.270   | 0.051    | 1.036e-01  | --         | --         | primary         |
| poisson2d | oracle_ceiling       | 32  | 24 | 64  | 256  | --    | --      | --       | 4.424e-03  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 32  | 24 | 64  | 256  | --    | 0.270   | 0.054    | 6.946e-02  | --         | --         | primary         |
| poisson2d | pod_direct           | 32  | 32 | 256 | 256  | --    | 0.260   | 0.051    | 5.540e-02  | --         | --         | artefact_m_eq_M |
| poisson2d | oracle_ceiling       | 32  | 32 | 256 | 900  | --    | --      | --       | 3.596e-03  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 32  | 32 | 256 | 900  | --    | 0.275   | 0.067    | 5.367e-02  | --         | --         | primary         |
| poisson2d | coord_meshfree_pool  | 32  | 8  | 64  | 256  | 1e-01 | 2.112   | --       | 9.150e-02  | 0          | 1.51e-03   | pool_control    |
| poisson2d | coord_meshfree_pool  | 32  | 8  | 64  | 256  | 1e-02 | 3.041   | --       | 1.481e-02  | 12         | 1.51e-03   | pool_control    |
| poisson2d | coord_meshfree_pool  | 32  | 8  | 64  | 256  | 1e-03 | 7.217   | --       | 1.133e-02  | 100        | 1.51e-03   | pool_control    |
| poisson2d | oracle_ceiling       | 512 | 2  | 64  | 256  | --    | --      | --       | 1.233e-01  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 512 | 2  | 64  | 256  | --    | 0.696   | 0.052    | 4.565e-01  | --         | --         | primary         |
| poisson2d | oracle_ceiling       | 512 | 4  | 64  | 256  | --    | --      | --       | 1.518e-02  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 512 | 4  | 64  | 256  | --    | 0.721   | 0.052    | 2.906e-01  | --         | --         | primary         |
| poisson2d | oracle_ceiling       | 512 | 6  | 64  | 256  | --    | --      | --       | 8.586e-03  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 512 | 6  | 64  | 256  | --    | 0.723   | 0.052    | 2.193e-01  | --         | --         | primary         |
| poisson2d | oracle_ceiling       | 512 | 8  | 64  | 256  | --    | --      | --       | 6.929e-03  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 512 | 8  | 64  | 256  | --    | 0.725   | 0.052    | 1.761e-01  | --         | --         | primary         |
| poisson2d | pod_direct           | 512 | 8  | 256 | 256  | --    | 1.107   | 0.051    | 1.761e-01  | --         | --         | supp_M256       |
| poisson2d | oracle_ceiling       | 512 | 12 | 64  | 256  | --    | --      | --       | 6.112e-03  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 512 | 12 | 64  | 256  | --    | 0.731   | 0.053    | 1.282e-01  | --         | --         | primary         |
| poisson2d | oracle_ceiling       | 512 | 16 | 64  | 256  | --    | --      | --       | 4.059e-03  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 512 | 16 | 64  | 256  | --    | 0.743   | 0.051    | 1.000e-01  | --         | --         | primary         |
| poisson2d | oracle_ceiling       | 512 | 24 | 64  | 256  | --    | --      | --       | 3.920e-03  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 512 | 24 | 64  | 256  | --    | 0.777   | 0.060    | 6.602e-02  | --         | --         | primary         |
| poisson2d | pod_direct           | 512 | 32 | 256 | 256  | --    | 1.179   | 0.060    | 5.139e-02  | --         | --         | artefact_m_eq_M |
| poisson2d | oracle_ceiling       | 64  | 2  | 64  | 256  | --    | --      | --       | 1.236e-01  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 64  | 2  | 64  | 256  | --    | 0.239   | 0.047    | 4.570e-01  | --         | --         | primary         |
| poisson2d | oracle_ceiling       | 64  | 4  | 64  | 256  | --    | --      | --       | 1.551e-02  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 64  | 4  | 64  | 256  | --    | 0.241   | 0.046    | 2.913e-01  | --         | --         | primary         |
| poisson2d | oracle_ceiling       | 64  | 6  | 64  | 256  | --    | --      | --       | 8.835e-03  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 64  | 6  | 64  | 256  | --    | 0.242   | 0.046    | 2.200e-01  | --         | --         | primary         |
| poisson2d | oracle_ceiling       | 64  | 8  | 64  | 256  | --    | --      | --       | 7.043e-03  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 64  | 8  | 64  | 256  | --    | 0.242   | 0.047    | 1.769e-01  | --         | --         | primary         |
| poisson2d | pod_direct           | 64  | 8  | 256 | 256  | --    | 0.247   | 0.051    | 1.769e-01  | --         | --         | supp_M256       |
| poisson2d | oracle_ceiling       | 64  | 12 | 64  | 256  | --    | --      | --       | 6.236e-03  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 64  | 12 | 64  | 256  | --    | 0.242   | 0.047    | 1.290e-01  | --         | --         | primary         |
| poisson2d | oracle_ceiling       | 64  | 16 | 64  | 256  | --    | --      | --       | 4.133e-03  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 64  | 16 | 64  | 256  | --    | 0.244   | 0.049    | 1.008e-01  | --         | --         | primary         |
| poisson2d | oracle_ceiling       | 64  | 24 | 64  | 256  | --    | --      | --       | 3.976e-03  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 64  | 24 | 64  | 256  | --    | 0.242   | 0.048    | 6.678e-02  | --         | --         | primary         |
| poisson2d | pod_direct           | 64  | 32 | 256 | 256  | --    | 0.243   | 0.046    | 5.214e-02  | --         | --         | artefact_m_eq_M |
| poisson2d | oracle_ceiling       | 64  | 32 | 256 | 1024 | --    | --      | --       | 3.280e-03  | --         | --         | ceiling         |
| poisson2d | pod_direct           | 64  | 32 | 256 | 1024 | --    | 0.261   | 0.063    | 5.105e-02  | --         | --         | primary         |
| poisson2d | coord_meshfree_pool  | 64  | 8  | 64  | 256  | 1e-01 | 2.227   | --       | 9.099e-02  | 0          | 1.51e-03   | pool_control    |
| poisson2d | coord_meshfree_pool  | 64  | 8  | 64  | 256  | 1e-02 | 3.153   | --       | 1.322e-02  | 12         | 1.51e-03   | pool_control    |
| poisson2d | coord_meshfree_pool  | 64  | 8  | 64  | 256  | 1e-03 | 7.430   | --       | 8.800e-03  | 100        | 1.51e-03   | pool_control    |
<!-- END GENERATED: supplementary -->

---

## 7. Verdict

_(written after the results land; every number quoted here is copied from a generated table
above and re-checked by the second Codex pass)_

## 8. Caveats

_(written after the results land)_

## 9. Files

```
ctol_tol.py          tau-stopped LM solvers; the Burgers rollout driver around
                     blat_common's own step kernel
ctol_eq.py           memory-capped NNLS-EQ quadrature fits (exact full-grid targets)
ctol_poisson.py      the Poisson (k x N x tau) surface driver
ctol_burgers.py      the Burgers (k x N x tau) surface driver
ctol_pick_configs.py chooses the configurations the consolidation job re-times
ctol_tables.py       builds runs/pareto_points.json and regenerates every table above
ctol_figs.py         the four figures
cluster/             make_cells.sh (panels | consolidate), launch.sh, pull.sh,
                     cancel.sh (the ONLY sanctioned cancel path -- numeric ids, ctol_* only)
runs/                per-panel JSONs, the consolidation JSON, pareto_points.json, Slurm logs
figs/                PNG + PDF (also copied to /home/tahmid/Dev/pod-ae-nmrom/Plots/)
CODEX-REVIEW-HARNESS.md   adversarial review of the harness, before the fan-out
CODEX-REVIEW-RESULTS.md   audit of every table and figure against the JSONs, after
```

## 10. Status

**Shared-account rule (2026-08-17 incident).** The `tawal01` cluster account is shared with
another agent. An ad-hoc `scancel --name=ctol_p_n32,ctol_p_n64,...` was issued from this cell
while retuning the supplementary arms. `scancel --name` takes ONE job name, not a
comma-separated list, so the value matched nothing — which left `scancel` with no effective
selector, and a `scancel` with no effective filter selects **every job belonging to the
invoking user**. All eleven of the other agent's jobs were killed at 00:00:00 elapsed. Nothing
was corrupted and both fleets resubmitted, but their queue position was lost along with ours.

From here on the only sanctioned way to cancel anything from this cell is
`cluster/cancel.sh <numeric job id> ...`, which refuses names, globs and user selectors,
verifies every id is currently queued under this account **and** named `ctol_*`, aborts the
whole call if any id fails, and re-prints `squeue` afterwards. Never run `scancel` by hand
here — at hour three of a multi-hour Burgers panel the same slip costs a working day.

**First submission, retuned.** The nine panels were first submitted with the supplementary
arms at `m`=1024. Timing the primary `M`=64, `m`=256 fits in the running jobs (15 s at 17 408
ECSW rows) calibrates the capped Lawson-Hanson refit at ~6 GFLOP/s, and its cost grows as
`n_rows * m^3 / 3`: an `M`=256, `m`=1024 Poisson fit is ~61 min, four per panel, i.e. **4.1 h
per panel of supplementary work against ~25 min for the whole primary grid**, and it would
have serialised the four Burgers panels behind it. The panels were cancelled ~14 minutes in
(before any supplementary fit completed, so nothing measured was discarded), the supplementary
arms retuned as described in §4, and everything resubmitted. The same pass caught a second
defect: the batch environment pinned `N_POD_TRAJ=128`, which would have silently overridden
the driver default and reinstated exactly the POD handicap the Codex audit had just removed.

The nine panel jobs (`ctol_p_n{32,64,128,256,512}`, `ctol_b_n{32,64,128,256}`) were submitted
together on 2026-08-17, one per directory under `/cluster/tufts/paralab/tawal01/ctol/`, all on
`--gres=gpu:a100:1`. To finish the cell:

```bash
./cluster/pull.sh                        # checksummed pull, deletes finished cluster dirs
python ctol_pick_configs.py              # frontier + target-reaching configs for consolidation
./cluster/make_cells.sh consolidate
./cluster/launch.sh ctol_consol_p && ./cluster/launch.sh ctol_consol_b
./cluster/pull.sh
python ctol_tables.py                    # aborts unless the surface is complete
python ctol_figs.py
```

`ctol_tables.py` will refuse to build the tables until every panel is complete, so a partial
surface cannot silently become a result. Sections 7 (Verdict) and 8 (Caveats) are written from
the generated tables once the surface is complete, and audited by a second Codex pass against
the JSONs (archived as `CODEX-REVIEW-RESULTS.md`).
