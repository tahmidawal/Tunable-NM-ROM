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

---

### Amendments

**2026-09-17, §A1 — the top rung's identifiability rule, corrected before any job.** The
first local smoke crashed at `q512_m256@new_K32`: the driver skipped a rung only when
$M \le K + q$ *and* $q < R$, exempting the top rung, so it built a 512-column correction on
257 tests and the full-rank assertion fired. The rule in force: below the top rung every one
of the $K + q$ unknowns must be identifiable ($M > K + q$); at $q = R$ the head is redundant
and only the $R$-column linear part must have full rank ($M > R$). Under `m256` the top rung
is therefore skipped (257 tests), exactly as $q = 256$ is; under `m4` it runs with
$M = 4(K + R) = 2176$. No criterion changes; the `m4` ladder was always the pre-registered
one for D1–D3.

**2026-09-17, §A2 — Codex unavailable; substitute auditor.** The Codex CLI (gpt-6-astra)
refused every call with a usage-limit error until 19 September (`reports/codex-design-audit.md`
records the attempt). The design audit was performed instead by an independent fresh-context
Claude agent with the identical read-only prompt (`reports/codex-design-audit-substitute.md`).
This is a weaker independence guarantee than a second model family and is stated as such in
the report; the post-hoc report audit will retry Codex first.

**2026-09-17, §A3 — a recorded diagnostic, not a change.** Rebuilding the retained 32
directions from scratch on the local box reproduces their subspace only to a principal-angle
defect of $6\times10^{-6}$ although the QR metric matches the retained one to
$5.6\times10^{-17}$: the bank's condition number is $2.6\times10^{7}$, so the triangular solve
that forms $T = Q^\top u$ carries $\sim 3\times10^{-9}$ of round-off into the residual, and the
1 % singular-value gap at column 32 amplifies it. This is why the design uses the retained
prefix **verbatim** rather than rebuilding it, and it is reported as a property of the
construction. The smoke builds the extension from a 640-source prefix of the fit split
(mechanics only); the cluster jobs use the whole fit split.

**2026-09-17, §A4 — the top rung is evaluated by a direct linear solve; declared from the
smoke, before any job.** At $q = R$ the eliminated ladder path is mathematically inert in
$z$ (its projected operator is round-off) and the smoke showed it *chasing that round-off
for its whole budget*: 194 Jacobians and 106 ms against 5–7 Jacobians and 7 ms at $q = 256$,
while landing exactly on the bank floor (0.80923 % worst, both). Charging that iteration to
the top rung would fail D2 for an artefact, so the pre-registered top rung is implemented as
the rank-$R$ linear reduced model solved directly — thin QR of $B_M$ offline, one
projection, one triangular solve and one decode online (`d_linear_qr_m4@new_K32`, same $M$
as the `m4` top rung) — and D1–D2 use it as the $q = R$ point. The eliminated arm is kept
and reported as `q512_m4@new_K32` with this caveat; the in-job identity check becomes
`q512_m4` vs `d_linear_qr_m4` at $10^{-8}$ on the field. The identity-head free-bank LM
(`d_freebank_m4`) is kept too but is *not* the top rung: it solves the normal equations of
a $2.6\times10^{7}$-conditioned bank and stopped at its gradient tolerance 4 % above the
floor (0.845 % vs 0.809 %); it is compared to the direct solve at a $10^{-2}$ tolerance and
reported as a solver-tolerance finding.

**2026-09-17, §A5 — gate arithmetic and a threshold, from the same smoke.** (a) The
`dst_direct` same-grid gate compared two round-off-level numbers ($\sim10^{-16}$) as a ratio
and reported 0.26; a gate now also passes when the absolute difference is
$\le 10^{-12}$ — only an exact solver's same-grid error can ever use that branch, and every
other gate passed at $\le 10^{-12}$ relative. (b) The extension orthonormality assertion is
$10^{-7}$ on a $512\times512$ Gram (the parent's $10^{-8}$ was on $32\times32$); the smoke
measured $4.6\times10^{-10}$ with the full fit split. Nothing in D1–D3 or the gates' $10^{-9}$
relative tolerance changes.

**2026-09-17, §A6 — one consistency threshold, from the passing smoke.** The final smoke
passed all 23 fidelity gates (worst $7.8\times10^{-13}$ relative) and completed. The
`q512_m4` vs `d_linear_qr_m4` field check came out at $3.2\times10^{-7}$ against the
$10^{-8}$ I had declared: both solve the same full-rank least-squares problem, but the
eliminated path recovers $y$ through the QR of $BC$ after an inert $z$ iteration and the
direct arm through the QR of $B$, so they agree to the operator's conditioning times
round-off, not to $10^{-8}$. The threshold is set to $10^{-5}$; it is a reported
consistency pair, not a gate, and D1–D3 do not use it.

**2026-09-17, §A7 — the dense best-found oracle is capped by mesh, after an OOM on a 40 GB
A100.** `plin1024` (job `3780691`) died at 14m27s in the *untimed* dense best-found oracle:
`arms.make_reconstruction`'s `jacfwd` Jacobian is `f64[starts, (n-1)^2, K]` — at
$n = 1024$, $8\times1046529\times32\times8$ B = 2.0 GiB per array, with 32 GiB autotuner
variants — and the scheduler gave the job an **A100-PCIE-40GB** where the parent lane's
equivalent job had an A100 80GB. The log is complete and ends in a JAX
`RESOURCE_EXHAUSTED`; the share was at 91 %, so this is not the disk-full failure mode.
Nothing was timed, no gate ran, and no number from that job is used.

Fix, in the untimed diagnostic only: the dense oracle now runs only at
$n \le$ `dense_oracle_max_intervals` (256). It was always a **cross-check** of the
pre-registered `augmented best-found` at $q = 0$, which `plin_core.oracle_projected`
computes in the exact QR metric with an $R$-dimensional residual — the same quantity, and
the two agree to $\sim10^{-13}$ wherever both run (recorded per job as
`dense_vs_projected_oracle`). The resubmit also asks for an **H200** with `--mem 240G`, per
this repository's rule for memory-heavy runs. No pre-registered criterion, gate, arm or
timed path changes.

**2026-09-17, §A8 — the D1 clause and the falsification wording, disambiguated. Written
with the 256-interval numbers already in hand, and therefore POST-HOC for that mesh and
PRE-HOC for 1024.** This is the amendment a reader should be most sceptical of, so it is
stated in full.

At 256 intervals the `m4` ladder with the A4 top rung is

| rung | worst same-grid | median total ms |
|---|---:|---:|
| `q0_m4` | 3.1567 % | 9.214 |
| `q32_m4` | 2.4699 % | 9.537 |
| `q64_m4` | 2.0808 % | 9.819 |
| `q128_m4` | 1.5497 % | 10.219 |
| `q256_m4` | 0.9689 % | 9.658 |
| `d_linear_qr_m4` (top) | 0.7459 % | 3.221 |

so **D1 fails at 3.173×** — but it fails because the top rung is **3.17× cheaper** than the
dearest rung while also being the most accurate, not because any rung pays more for
accuracy. D2 passes in its strict form (the top rung *is* the cheapest and the most
accurate) and D3 passes.

The falsification clause reads: *falsified if D1 fails with error monotone non-increasing in
$q$ (the ladder buys accuracy for $\ge 2\times$ cost)*. Its two literal conjuncts are both
**met**; its parenthetical gloss is **not** — no rung costs $\ge 2\times$ the cheapest ladder
point while being more accurate than it. I did not anticipate a ladder whose top rung is
cheaper than its middle, so the clause as written is ambiguous here.

**I am not rewriting D1, and the literal verdict stands as reported.** Every report and both
independent implementations print, for every mesh: D1 as literally written (pass/fail),
`falsified_literal`, `falsified_intent` (the parenthetical, computed as "some rung costs
$\ge 2\times$ the cheapest ladder point *and* is strictly more accurate than it"), the list
of any rungs that satisfy it, and — clearly labelled post-hoc and not pre-registered — the
cost span over the $q < R$ rungs alone, which is the quantity D1 was designed to measure
(1.109× at 256). For the 1024 job, which had not run when this was written, the same six
quantities are reported and **`falsified_intent` is the clause I will treat as deciding the
paper's claim**, with the literal verdict printed beside it either way.
