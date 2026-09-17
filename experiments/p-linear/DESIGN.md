# p-linear — the Poisson correction ladder as the paper's linear-case figure

Pre-registered before the first cluster job. Amendments are appended as §A1, §A2, …
and never rewrite a criterion.

Worktree `worktrees/2026-09-17-p-linear`, branch `exp/2026-09-17-p-linear`, forked from
`exp/2026-09-16-p-bank-head` (commit `266dea9d`). Cluster namespace
`/cluster/tufts/paralab/tawal01/p_linear_20260917/`. Job cap: 8; planned: 3.

---

## 1. The question

The paper's structural claim is that the accuracy–cost trade from correction rank $q$
exists only where the projected residual is nonlinear in the bank coefficients (Burgers),
and that on a linear PDE the curve **degenerates**: the top rung is a linear reduced model
that is also (about) the cheapest point, and classical POD-LSPG or a direct solver is
non-dominated. For Poisson 2D that claim currently rests on the incumbent $R{=}128$/$K{=}16$
checkpoint's ladder to $q{=}64$ (`ccpoi01`: 6.09 → 4.18 % worst, cost span $\le 1.4\times$).
This cell nails it on the **best** Poisson checkpoint (`pbh02`'s `new_K32`: $R{=}512$,
$K{=}32$, bank floor 0.74 %, head 3.11 % worst at $1024^2$), with the ladder run to
$q{=}R$ and every comparator timed in the same job on the same GPU. It is paper table T11
and the linear-case figure.

A secondary question (job 3): *why* is the head dominated on Poisson? `p-bank-head` found
the head underfits its own training data (bank 1.93 % vs head 4.68 % on training sources,
codes converged). The one cheap remedy that does not change the architecture class — a
wider/deeper head and a larger $K$ trained with the same recipe on the same frozen bank —
is tested, and whether the head's best-found moves toward the bank floor is reported.

## 2. What is held fixed

The Poisson weak residual is exactly linear in the bank coefficients:

$$r(z, y) \;=\; B\,\bigl(h_\theta(z) + C_q\,y\bigr) - f_m,\qquad
B = \Phi^\top G \in \mathbb{R}^{M\times R},\quad f_m = \Lambda^{-1}\Phi^\top f,$$

with $\Phi$ the $M$ retained orthonormal sine test modes, $G$ the cached bank, $C_q$ the
first $q$ nested correction directions. The correction $y$ is eliminated **exactly** by the
retained `correction_core` path (thin QR of $BC_q$), so the nonlinear iteration is
$K$-dimensional at every $q$; at $q = R$ the projected operator is numerically zero and the
rung is the full-bank linear least-squares solve, reported as such.

Held fixed everywhere, unchanged from `pbh02` / `ccpoi01`: the two frozen checkpoints and
their retained 32-direction bases; the nearest-training-code initializer; LM budget 300;
stationarity tolerance $10^{-6}$; the `gj`/`lu` step rule at 64 unknowns; the output
contract (dense nodal field); `JAX_DEFAULT_MATMUL_PRECISION=highest`, float64; the twelve
opened development sources (`source_params(7090703, 6) ∪ source_params(7090732, 6)`); the
2048-interval reference chain. No new case is opened; the sealed cohorts stay sealed.

**Directions.** The retained 32 columns of each checkpoint's basis are used verbatim, so
every rung with $q \le 32$ is the parent's arm bit for bit. Columns $33..R$ extend them by
the rule the retained bases were built with (right singular vectors of the normalised
training residual at the stored training codes, in the bank's exact QR field metric at
the training mesh 255), applied to the component orthogonal to the retained 32; nested in
$q$. The `ccpoi01` extension rule (best-found codes on 192 residual snapshots at the query
mesh) differs; it is reproduced *separately* for the incumbent's $q{=}64$ gate only.

**Test count.** Two rules, both pre-registered in `cheap-corrections`: `m256` (256
requested modes, 257 retained, the parent's fixed rule — valid only while $M > K + q$, so
up to $q{=}128$ at $K{=}32$) and `m4` ($M = 4(K+q)$, the only rule valid at every rung to
$q = R$). POD-LSPG uses the same two rules with $K + q \to k'$. The free bank uses the top
rung's test count so that "$q = R$ equals the free bank" is an in-job identity check.

## 3. Subjects, per mesh (1024 primary; 256 second mesh), all in one job

| subject | family | unknowns | test rule | role |
|---|---|---:|---|---|
| `q{0,32,64,128}_m256@new_K32` | neural + eliminated linear | $K{+}q$ | m256 | ladder, fixed tests |
| `q{0,32,64,128,256,512}_m4@new_K32` | neural + eliminated linear | $K{+}q$ | m4 | **the ladder** |
| `a_neural@new_K32` | neural (parent kernel) | 32 | m256 | pbh02 gate |
| `d_freebank_m4@new_K32` | free bank, identity head | 512 | m4 at $K{+}R$ | top-rung identity |
| `q{0,32,64,128}_m256@incumbent`, `q{64}_m4@incumbent` | as above, $R{=}128$ | | | cross-job gates |
| `q64_{m256,m4}_ccrule@incumbent` | ccpoi01's direction rule | 80 | | ccpoi01 gate (1024 only) |
| `a_neural@incumbent`, `d_freebank_m256@incumbent` | parent kernel | 16 / 128 | m256 | pbh02 gates |
| `e_pod{32,64,128}_m256@trainset` | POD-LSPG, 3072 snapshots | $k'$ | m256 | pbh02 gates |
| `e_pod{32,64,128,256,512}_m4@trainset` | POD-LSPG | $k'$ | m4 | **the classical ladder** |
| `dst_direct` | full order, exact | — | — | the direct solver |
| `cg_{1e-2,1e-4,1e-6,1e-8}` | full order, unpreconditioned CG | — | — | iterative comparators |

Jacobi-PCG is **not** run: for the 5-point Laplacian with Dirichlet data the diagonal is
constant, so Jacobi preconditioning is a scalar rescaling and its iterates coincide with
CG's. This is stated rather than measured.

Reported per subject: worst and median **same-grid** error (against the same-mesh FD-DST
solution) and physical error (against the restricted 2048-interval reference); median
total query ms (host source in → dense field out) and median device ms; stationary and
solver-valid counts; iteration counts; 3 timed repetitions in randomised order with GPU
burn-in before every invocation, every repetition retained. Untimed per checkpoint: the
bank floor, the dense best-found at $q{=}0$ (parent path) and the **augmented best-found**
at every $q$ (multistart LM on $\|(I - V_qV_q^\top)(R_G h(z) - T)\|$, $V_q = R_G C_q$),
which brackets each rung: floor $\le$ best-found($q$) $\le$ solved($q$).

## 4. Pre-registered degenerate-curve criterion

Evaluated separately at each mesh on the `m4` ladder of `new_K32` (the only rule valid at
every rung). Error = worst same-grid over the 12 sources; cost = median total query ms in
the same job.

* **D1 — cost span.** $\max_q \mathrm{cost}(q) / \min_q \mathrm{cost}(q) < 2$.
* **D2 — the top rung.** $q = R$ has the lowest error of the ladder **and** its cost is
  within $1.1\times$ of the cheapest rung. The strict form — $q = R$ *is* the cheapest
  rung — is reported as pass/fail beside it.
* **D3 — classical wins.** A full-order solver (DST or a CG tolerance) is on the
  non-dominated set of all subjects, and POD-LSPG is on the non-dominated set of the
  *reduced* subjects (every subject except the full-order solvers).

The curve is **degenerate** (the paper's claim holds) if D1 ∧ D2 ∧ D3 at 1024. Partial
results are reported as partial.

**Falsification.** The claim is falsified — Poisson would carry a real trade like Burgers —
if D1 fails with error monotone non-increasing in $q$ (the ladder buys accuracy for
$\ge 2\times$ cost), **or** if a rung with $q < R$ sits on the reduced non-dominated set
strictly cheaper than every POD-LSPG rung of equal or better error. Either outcome is
reported with the same tables.

**Cross-job fidelity gates, all $\le 10^{-9}$ relative on per-case physical error, and on
same-grid error where the reference has it; any failure aborts the job before a verdict:**
`a_neural@incumbent`, `q32_m256@incumbent` (= pbh02 `a_neural_q32@incumbent`),
`d_freebank_m256@incumbent`, `a_neural@new_K32`, `q32_m256@new_K32`,
`e_pod{32,64,128}_m256@trainset`, `dst_direct` against pbh02 at 64/256/1024;
`q0_m256@incumbent`, `q32_m256@incumbent`, `q64_{m256,m4}_ccrule@incumbent`, `dst_direct`
against ccpoi01 at 1024. In-job consistency checks (reported, tolerance stated per pair):
`q0_m256` vs `a_neural` (same problem, two LM implementations, $10^{-6}$); `q512_m4` vs
`d_freebank_m4` (same least-squares problem, elimination vs identity-head LM, $10^{-6}$);
`q32_m4` vs `q32_m256` (identical arms under two names, $10^{-12}$ — a timing-noise
control).

## 5. Job 3 — head capacity on the frozen bank (pre-registered)

Bank: `bank_R512_S3072` frozen (the selected bank; its parameters are the `g` track of
`primary_K32.pkl`, identical bytes to `bankarm_head.pkl`). Recipe: `pbh_fit.train_head_phase`
exactly as `pbh02` — 150000 full-batch Adam updates, lr $10^{-3}$ cosine, seed
20260916201, $(\beta_w, \beta_s) = (0, 0)$, fresh codes — on the same 2611-source fit
split; the only knobs are the head width $w$, depth $L$, latent $K$, and one longer
schedule.

| arm | $K$ | $w$ | $L$ | updates | role |
|---|---:|---:|---:|---:|---|
| `K32_w128_L2` | 32 | 128 | 2 | 150000 | **control**: the incumbent recipe re-run |
| `K32_w256_L2` | 32 | 256 | 2 | 150000 | width only |
| `K32_w128_L3` | 32 | 128 | 3 | 150000 | depth only |
| `K32_w256_L3` | 32 | 256 | 3 | 150000 | 2× width, 3 layers (the coordinator's arm) |
| `K64_w128_L2` | 64 | 128 | 2 | 150000 | larger $K$ |
| `K64_w256_L3` | 64 | 256 | 3 | 150000 | larger $K$, 2× width, 3 layers |
| `K32_w128_L2_x3` | 32 | 128 | 2 | 450000 | optimisation-limited control |

Reported per arm at 255 intervals: head error at the stored codes on the fit split; best-found
on the internal-validation split and on the 12 development sources; the ratio best-found /
bank floor. Then, in the same job at 1024 intervals through the unchanged `poisson_ablation`
kernel with `dst_direct` interleaved: worst same-grid solved error and median total ms, so a
wider head's extra per-iteration cost is measured rather than argued.

* **H1 — reproducibility gate.** `K32_w128_L2` reproduces `pbh02`'s `head_K32_w0_s0`
  development best-found at 255 (3.1212 % worst) within 5 % relative. If it fails, every
  capacity conclusion is stated conditionally and the failure is the headline of job 3.
* **H2 — verdict.** Let $\rho = $ worst development best-found / worst bank floor at 255
  ($4.2\times$ for `pbh02`'s primary). If no capacity arm lowers $\rho$ by more than 20 %
  relative, the head is reported as **function-class-limited within the tested range**; if
  an arm brings $\rho$ under $2\times$, it is reported as a better anchor and its 1024
  solve enters the linear-case table beside `new_K32`. Anything between is reported as a
  partial movement with the numbers.

Selection, if any arm is carried forward, is by the internal-validation split; the
development sources select nothing.

## 6. Process

* Local: one 64-interval smoke through the shared slot lock, which must pass the same
  pbh02 gates at 64 to $\le 10^{-9}$ before anything is submitted.
* Cluster: `--gres=gpu:a100:1`, `--mem 180G`, `gpu` partition, `pax007` excluded, one job per
  attempt directory, `squeue` before and after every submit, `jax_backend=gpu` asserted,
  three attempts planned (`plin1024`, `plin256`, `plhead1`).
* After each job: checksum collection, an independent NumPy audit that imports neither the
  driver nor JAX and recomputes every reported error from the retained output fields against
  an independently rebuilt reference, bounded Git chunks, then deletion of the exact remote
  attempt directory.
* Codex (gpt-6-astra) audits this document before job 1 and the report against the raw
  JSONs after; accepted and rejected findings are recorded in the report.
* The report and `summary.json` are generated by `reports/generate_p_linear.py`; no number is
  typed by hand.
