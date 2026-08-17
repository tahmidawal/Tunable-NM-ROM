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

**Time accounting.** `time_ms` in the shared Pareto schema is the **end-to-end online cost**,
because that is what the FOM baseline delivers:

* Poisson: input preprocessing (`Lambda^-1 Phi^T f`, one matvec against a per-mesh table built
  offline) + latent solve + decode of the interior field. The FOM baseline is the testbed's own
  jitted CG returning exactly that interior field.
* Burgers: hyper-reduced cold start + 50 tau-stopped latent steps + the 51-slice full-field
  decode. The FOM baseline is the testbed's own jitted implicit rollout at batch 1.

`time_ms_solve` isolates the latent solve — the quantity the cost(`k`) question asks about —
and is what the cost(`k`) figure plots. Both are in every row.

## 4. Grid and settings

`k` in {2, 4, 6, 8, 12, 16, 24, 32}; `N` in {32, 64, 128, 256, 512} (Poisson) and
{32, 64, 128, 256} (Burgers). `M`=64, `m`=256 fixed, **except `M`=256 whenever `k` >= 32** —
the weak form collapses when `M <= k` (heat `M`=16,`K`=16 gave 9.0e-2 against a 6.3e-3
ceiling; Burgers POD `k`=64,`M`=64 diverged). A supplementary arm at `k`=32 adds `m`=1024 so
the `m ~ 4M` knee is not silently violated by the `m` = `M` = 256 corner.

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
projections at every mesh. Per-cell fit quality is reported below so the cap is auditable.

## 5. How this was run

One job per **(PDE, mesh) panel**, all submitted simultaneously, each in its own directory
under `/cluster/tufts/paralab/tawal01/ctol/`, all requesting the same GPU type (A100). Inside
a panel every timing shares one GPU, so the per-(PDE, N) Pareto frontier — whose dominance is
computed *within* a panel — is valid as measured.

The **scaling figure compares timings across meshes**, which panels on different physical GPUs
cannot support. So after the panels land, `ctol_pick_configs.py` selects the handful of
argmin configurations per (method, N) from the panel *accuracies* (which are
GPU-independent), and a single **consolidation job re-times only those, sequentially, on one
GPU**. That consolidation run is the only timing source the scaling figure uses; the
cross-check table below shows panel vs consolidated timings side by side.

```bash
cd experiments/cost-to-tolerance
./cluster/make_cells.sh panels && for c in ctol_p_n32 ... ctol_b_n256; do ./cluster/launch.sh $c; done
./cluster/pull.sh
python ctol_pick_configs.py
./cluster/make_cells.sh consolidate && ./cluster/launch.sh ctol_consol_p && ./cluster/launch.sh ctol_consol_b
./cluster/pull.sh
python ctol_tables.py && python ctol_figs.py
```

### Provenance

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

### 6.3 The knob -> accuracy map: field error actually achieved at each tau

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

Supplementary arms (`pod_direct` = the exact linear POD minimiser as one pinv matvec;
`coord_meshfree_pool` = the headline meshfree EQ pool at `k`=8):

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
cluster/             make_cells.sh (panels | consolidate), launch.sh, pull.sh
runs/                per-panel JSONs, the consolidation JSON, pareto_points.json, Slurm logs
figs/                PNG + PDF (also copied to /home/tahmid/Dev/pod-ae-nmrom/Plots/)
CODEX-REVIEW-HARNESS.md   adversarial review of the harness, before the fan-out
CODEX-REVIEW-RESULTS.md   audit of every table and figure against the JSONs, after
```
