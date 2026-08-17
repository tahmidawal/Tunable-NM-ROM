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
./cluster/make_cells.sh panels && for c in ctol_p_n32 ... ctol_b_n256; do ./cluster/launch.sh $c; done
./cluster/pull.sh
python ctol_pick_configs.py
./cluster/make_cells.sh consolidate && ./cluster/launch.sh ctol_consol_p && ./cluster/launch.sh ctol_consol_b
./cluster/pull.sh
python ctol_tables.py && python ctol_figs.py
```

### Surface integrity

`ctol_tables.py` refuses to build these tables unless every panel finished, every expected
primary cell is present exactly once, and every panel ran on GPU in f64 at matmul precision
`highest`. Its verdict:

<!-- BEGIN GENERATED: integrity -->
<!-- END GENERATED: integrity -->

### Provenance

The executed bundle is assembled from four worktrees, so this tree's commit alone does not
identify it. Every panel records the commit and dirty state of all four source trees, the
sha256 of every module and checkpoint it actually loaded, and the staged manifest hash.

<!-- BEGIN GENERATED: provenance -->
<!-- END GENERATED: provenance -->

### FOM baselines

<!-- BEGIN GENERATED: fom -->
<!-- END GENERATED: fom -->

---

## 6. Results

### 6.1 Cost of the latent solve vs k, overlaid across meshes

Figures: `figs/ctol_cost_vs_k_poisson2d.{png,pdf}`, `figs/ctol_cost_vs_k_burgers2d.{png,pdf}`.
The bottom row of each figure normalises every curve at `k`=8: if the `k` dependence is
mesh-independent the normalised curves collapse onto one another.

**Poisson, latent solve (ms), tau = 1e-2**

<!-- BEGIN GENERATED: cost_k_poisson2d_1e-02 -->
<!-- END GENERATED: cost_k_poisson2d_1e-02 -->

**Burgers, latent solve (ms), tau = 1e-2**

<!-- BEGIN GENERATED: cost_k_burgers2d_1e-02 -->
<!-- END GENERATED: cost_k_burgers2d_1e-02 -->

**Shape of the k dependence, normalised at k=8** (a row of all-equal numbers across `N` is
mesh independence)

<!-- BEGIN GENERATED: kshape_poisson2d -->
<!-- END GENERATED: kshape_poisson2d -->

<!-- BEGIN GENERATED: kshape_burgers2d -->
<!-- END GENERATED: kshape_burgers2d -->

Full cost tables at the other tolerances:

<!-- BEGIN GENERATED: cost_k_poisson2d_1e-01 -->
<!-- END GENERATED: cost_k_poisson2d_1e-01 -->

<!-- BEGIN GENERATED: cost_k_poisson2d_1e-03 -->
<!-- END GENERATED: cost_k_poisson2d_1e-03 -->

<!-- BEGIN GENERATED: cost_k_burgers2d_1e-01 -->
<!-- END GENERATED: cost_k_burgers2d_1e-01 -->

<!-- BEGIN GENERATED: cost_k_burgers2d_1e-03 -->
<!-- END GENERATED: cost_k_burgers2d_1e-03 -->

### 6.2 Work to reach the tolerance vs k

Figures: `figs/ctol_iters_vs_k_*.{png,pdf}`. Jacobian evaluations (accepted LM steps; the
Burgers numbers are summed over the cold start and all 50 time steps).

<!-- BEGIN GENERATED: iters_k_poisson2d_1e-02 -->
<!-- END GENERATED: iters_k_poisson2d_1e-02 -->

<!-- BEGIN GENERATED: iters_k_burgers2d_1e-02 -->
<!-- END GENERATED: iters_k_burgers2d_1e-02 -->

At the other two tolerances:

<!-- BEGIN GENERATED: iters_k_poisson2d_1e-01 -->
<!-- END GENERATED: iters_k_poisson2d_1e-01 -->

<!-- BEGIN GENERATED: iters_k_poisson2d_1e-03 -->
<!-- END GENERATED: iters_k_poisson2d_1e-03 -->

<!-- BEGIN GENERATED: iters_k_burgers2d_1e-01 -->
<!-- END GENERATED: iters_k_burgers2d_1e-01 -->

<!-- BEGIN GENERATED: iters_k_burgers2d_1e-03 -->
<!-- END GENERATED: iters_k_burgers2d_1e-03 -->

### 6.3 The knob -> accuracy map: field error actually achieved at each tau

**Accuracy is non-monotone in `k`, and that is the checkpoints, not the solver.** Each `k` is a
*separately trained* decoder, so decoder quality varies along the `k` axis independently of
anything the ROM does (at N=64, `k`=6 and `k`=12 are worse than `k`=4 and `k`=8). Rather than
caveat this, each checkpoint's own **oracle ceiling** — the error of the best latent obtainable
by LM on the data misfit to the held-out field, which no solver can beat — is measured at every
(N, `k`) and reported next to the ROM error, together with the ROM/ceiling ratio. Read the
ratio, not the raw error, when judging the solver along `k`.

<!-- BEGIN GENERATED: ceiling_poisson2d -->
<!-- END GENERATED: ceiling_poisson2d -->

<!-- BEGIN GENERATED: ceiling_burgers2d -->
<!-- END GENERATED: ceiling_burgers2d -->


<!-- BEGIN GENERATED: err_k_poisson2d_1e-01 -->
<!-- END GENERATED: err_k_poisson2d_1e-01 -->

<!-- BEGIN GENERATED: err_k_poisson2d_1e-02 -->
<!-- END GENERATED: err_k_poisson2d_1e-02 -->

<!-- BEGIN GENERATED: err_k_poisson2d_1e-03 -->
<!-- END GENERATED: err_k_poisson2d_1e-03 -->

<!-- BEGIN GENERATED: err_k_burgers2d_1e-01 -->
<!-- END GENERATED: err_k_burgers2d_1e-01 -->

<!-- BEGIN GENERATED: err_k_burgers2d_1e-02 -->
<!-- END GENERATED: err_k_burgers2d_1e-02 -->

<!-- BEGIN GENERATED: err_k_burgers2d_1e-03 -->
<!-- END GENERATED: err_k_burgers2d_1e-03 -->

### 6.4 Censoring: percentage of cells that never reached tau

Reported honestly, never dropped. For Poisson this is the fraction of the 16 sources; for
Burgers the fraction of the 51 tau-stopped solves (cold start + 50 time steps), averaged over
the 16 held-out trajectories.

<!-- BEGIN GENERATED: censor_poisson2d -->
<!-- END GENERATED: censor_poisson2d -->

<!-- BEGIN GENERATED: censor_burgers2d -->
<!-- END GENERATED: censor_burgers2d -->

### 6.5 Achieved discrete residual (reference only — never a stopping test)

Poisson: `||A u - f|| / ||f||` at the returned latent. Burgers: the FOM's own backward-Euler
residual `||R(u_{n+1}, u_n)|| / ||u_n||` along the ROM trajectory, averaged over the 50 steps.

<!-- BEGIN GENERATED: resid_poisson2d -->
<!-- END GENERATED: resid_poisson2d -->

<!-- BEGIN GENERATED: resid_burgers2d -->
<!-- END GENERATED: resid_burgers2d -->

### 6.6 The iso-error Pareto frontier

Figures: `figs/ctol_pareto_poisson2d.{png,pdf}`, `figs/ctol_pareto_burgers2d.{png,pdf}`.
x = online wall time (log), y = held-out rel-L2 (log); faint scatter = every configuration,
solid line = the non-dominated envelope per method, dots labelled with `k`; the FOM is a
vertical dashed line (it is the reference truth, so its error is 0 and it is off-plot — the
line reads *the price of exactness*).

Non-dominated configurations, per mesh:

<!-- BEGIN GENERATED: pareto_poisson2d -->
<!-- END GENERATED: pareto_poisson2d -->

<!-- BEGIN GENERATED: pareto_burgers2d -->
<!-- END GENERATED: pareto_burgers2d -->

The **as-deployed** frontier, censored cells included (set the knob, take whatever the solver
reaches). Where it extends past the strict frontier, the extra accuracy is bought by running to
termination rather than to the declared tolerance:

<!-- BEGIN GENERATED: paretodep_poisson2d -->
<!-- END GENERATED: paretodep_poisson2d -->

<!-- BEGIN GENERATED: paretodep_burgers2d -->
<!-- END GENERATED: paretodep_burgers2d -->

**Who owns the frontier at each mesh**

<!-- BEGIN GENERATED: owner_poisson2d -->
<!-- END GENERATED: owner_poisson2d -->

<!-- BEGIN GENERATED: owner_burgers2d -->
<!-- END GENERATED: owner_burgers2d -->

### 6.7 Scaling: cheapest time reaching a fixed error target vs N

Figures: `figs/ctol_scaling_poisson2d.{png,pdf}`, `figs/ctol_scaling_burgers2d.{png,pdf}`.
Mesh independence appears as a flat line while the FOM's rises. The configuration is chosen
from the panel accuracies; the time is the single-GPU consolidation timing.

<!-- BEGIN GENERATED: scaling_poisson2d -->
<!-- END GENERATED: scaling_poisson2d -->

<!-- BEGIN GENERATED: scaling_burgers2d -->
<!-- END GENERATED: scaling_burgers2d -->

**Panel vs single-GPU consolidation cross-check** (the same configuration timed twice; a large
ratio means the panel GPUs differed materially and only the consolidated column may be read
across meshes)

<!-- BEGIN GENERATED: consolidation -->
<!-- END GENERATED: consolidation -->

### 6.8 Hyper-reduction fit quality and supplementary arms

NNLS-EQ relative fit per cell (at the first tau; the fit does not depend on tau):

<!-- BEGIN GENERATED: eqfit_poisson2d -->
<!-- END GENERATED: eqfit_poisson2d -->

<!-- BEGIN GENERATED: eqfit_burgers2d -->
<!-- END GENERATED: eqfit_burgers2d -->

POD projection floor on the held-out test set — the oracle bound no POD solver can beat
(Burgers only; the Poisson POD basis is built from the same 512 training sources):

<!-- BEGIN GENERATED: podfloor -->
<!-- END GENERATED: podfloor -->

Supplementary arms: `pod_direct` = the exact linear POD minimiser as one pinv matvec;
`coord_meshfree_pool` = the headline meshfree EQ pool at `k`=8; `*_uncapped_pool` = the
`cap_control` arms that refit EQ with every interior node as a candidate:

<!-- BEGIN GENERATED: supplementary -->
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
