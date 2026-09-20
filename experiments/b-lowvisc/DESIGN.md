# b-lowvisc — is there a viscosity at which the nonlinear manifold is worth having?

Pre-registered on 2026-09-17 before any cluster job was submitted. Every amendment is appended
to [§10 Amendments](#10-amendments) with its date and reason, never edited in place.

Worktree `worktrees/2026-09-17-b-lowvisc`, branch `exp/2026-09-17-b-lowvisc`, forked from
`exp/2026-09-17-b-seeds` at `cff8df73`. Cluster namespace
`/cluster/tufts/paralab/tawal01/b_lowvisc_20260917/`, one attempt directory per job. Nothing is
merged; nothing is pushed.

---

## 1. Why this lane exists

The adversarial review of the draft (`scratchpad/review-round1.md`, findings 10 and 12 and the
closing paragraph) says the paper currently characterises only **where the nonlinear manifold is
not worth having**. On Burgers $256^2$, job `3780638` put 27 reduced-order subjects and a tuned
full-order Newton grid in one allocation on one GPU and **zero reduced subjects were
non-dominated**: the cheapest full-order setting that beats the best reduced arm on both axes is
`nt1e-3_dt005` at 0.0489 % and 31.4 ms, against `q256_M1088_dense` at 0.5194 % and 4164.6 ms. On
the linear PDEs the top rung of the ladder collapses to a linear ROM and a direct transform wins
outright.

The review's single highest-value recommendation, which the coordinator shares, is to find **one
cell where the neural head plus the correction ladder is non-dominated**. That converts a
catalogue of losses into a thesis. The most plausible candidate it names is a lower-viscosity
Burgers cell.

## 2. The hypothesis, in two legs

The incumbent Burgers family draws (confirmed from the generator, not assumed — `engines.params_draw`
and `burgers2d_film.sample_params`, both `exp(U(log .01, log .1))`):

$$c_x,c_y\sim U(.15,.85),\quad w\sim U(.05,.20),\quad a\sim U(.5,2),\quad \nu\sim \exp U(\log 10^{-2},\log 10^{-1}).$$

So the incumbent viscosity range is $[10^{-2},10^{-1}]$, **one decade higher than the
$[10^{-3},10^{-2}]$ quoted in the lane brief**; the brief said to confirm it and this is the
confirmation.

At lower $\nu$ the solution is more advection-dominated. Two consequences would each favour a
reduced-order arm, and the hypothesis needs both:

* **Leg (a) — the Kolmogorov width decays more slowly**, so POD at every affordable rank degrades
  sharply. That is exactly the regime nonlinear manifolds are claimed for, and it is what would
  let a $K=16$ head plus a $q$-rank correction beat a POD basis of far higher rank.
* **Leg (b) — the full-order solve gets more expensive.** The backward-Euler Newton step is
  preconditioned by the exact Helmholtz operator $(I+\Delta t\,\nu\,\Lambda)^{-1}$ (`engines.helmholtz`),
  which is an *excellent* preconditioner when $\Delta t\,\nu\,\lambda_{\max}\gg1$ and degenerates to
  the identity as $\nu\to0$. If BiCGStab then needs many more iterations, full-order cost rises
  while the reduced query's cost — set by the reduced dimension and the test-space size — is far
  less sensitive.

**An honest prior, recorded before the measurement.** Leg (b) is the weaker of the two. At
$\nu=10^{-3}$, $\Delta t=0.005$, $L=256$ the stiff part is $\Delta t\,\nu\,\lambda_{\max}\approx2.6$
and the advective part $\Delta t\,|u|L\approx5$, so the unpreconditioned system is still
well-conditioned and the iteration count may rise only modestly. The gate job (§5) measures leg (b)
directly rather than assuming it, and reports the answer whichever way it falls.

## 3. The viscosity family, and why exactly this one

**The family: $\nu\sim\exp U(\log 10^{-3},\log 10^{-2})$, drawn by the identical sequential NumPy
generator with identical bounds on every other column** (`lv_common.params_draw`).

Because `Generator.uniform(lo, hi, n)` consumes the same underlying uniforms whatever the bounds,
and because both intervals are exactly one decade wide in log space, this choice has a property
no other candidate has:

* columns $(c_x,c_y,w,a)$ are **bit-identical** to the incumbent draw for the same seed and count
  — the initial conditions are literally the same six fields; and
* $\nu_{\text{low}}=\nu_{\text{incumbent}}/10$ case by case, verified to $3.6\times10^{-16}$
  relative.

So the low-viscosity cohort **is the incumbent cohort with the viscosity divided by ten, and
nothing else changed**. That is the cleanest controlled variable available, and it means the two
families can be compared case by case, not just in distribution. `gate.py` recomputes and gates
both statements (`cohorts_differ_only_in_viscosity`, `family_relation`); `audit_gate.py` recomputes
them again independently.

**What this costs, stated up front.** The spatial discretisation is first-order upwind
(`engines.spatial`), whose numerical viscosity is $|u|h/2$. On the six development cases at
$L=256$ that is $2.1$–$3.7\times10^{-3}$, i.e. a cell Reynolds number $a h/\nu$ of 0.13–0.42 for
the incumbent family but **1.3–4.2 for the low-viscosity family**: four of the six low-viscosity
cases have more numerical than physical viscosity on this grid. Consequences, all of which the
gate job measures and the report will state rather than hide:

* the $L=256$ discrete solution is a well-defined operator and the ROM is a surrogate *for it*, so
  the same-grid metric — the primary metric of every panel in this campaign — remains exactly as
  meaningful as it is for the incumbent cell;
* but the $L=256$ discretisation error against the 4096-interval reference will be larger for the
  low-viscosity family than the incumbent cell's 4.0265 %, and a reviewer is entitled to call the
  cell under-resolved. The gate job reports that number for both families;
* and leg (a) may be *muted*, because numerical diffusion floors how sharp the discrete solution
  can get. A full retrain at $L\ge512$ is out of this lane's compute budget (§8), so the gate job
  instead measures the full-order cost and accuracy at $L\in\{512,1024\}$ for both families, and
  the report states what a finer-mesh follow-up would need.

**Meshes.** Primary and only training/panel mesh: $L=256$ intervals, $\Delta t=0.005$, 50 steps to
$t=0.25$, outputs every 10 steps — the incumbent cell verbatim, so the only changed variable in
the whole lane is $\nu$. Mesh probe (full-order only, no reduced model): $L\in\{512,1024\}$.

## 4. Cohort discipline

* **Development cohort, opened:** the incumbent's own six cases — `params_draw(7090702, 4)` then
  `params_draw(911702, 2)` — at each family's viscosity bounds. Four are "opened development", two
  "fresh development", exactly as `bpn301`/`abl01` label them.
* **Training trajectories:** `params_draw(0, 128)` for the POD snapshots (b-panel's own setting),
  and for the bank/head retrain the incumbent recipe's own draws (seed 0 for 576 canonical, seed
  1000 for 4032 appended), all at the low-viscosity bounds. Disjointness from the development
  cohort is asserted in job.
* **Sealed cohort, untouched:** `params_draw(17092026, 6)` at the low-viscosity bounds — the same
  draw b-seeds sealed, so the two lanes' sealed sets are the same six parameter vectors up to the
  viscosity column. It is opened in **at most one job, the last one**, and only if a panel job has
  already produced a verdict on the development cohort. Every job before it records
  `final_cohort_unopened: true`. If the lane stops at the gate (§6, G-FAIL) the sealed cohort is
  never opened at all.
* No fit, selection, tuning or diagnosis ever touches the sealed cohort.

## 5. The three stages, and what each decides

### Stage 1 — the gate job `lvg01` (`gate.py`, `config-gate.json`). No trained model.

One allocation, one GPU, **both viscosity families side by side**, so every cross-family ratio it
reports is a same-job ratio.

1. **Data verification.** The 4096-interval reference is solved for all six cases of both families
   at Newton tolerance $10^{-11}$ and asserted below $2\times10^{-11}$; the 128 snapshot
   trajectories are asserted below the pre-registered `snapshot_residual_bar` $=10^{-8}$ (the
   inherited chain's requirement; b-seeds measured $10^{-12}$ against it).
2. **Leg (a).** For each family: method-of-snapshots POD of the 128 trajectories at stride 2, and
   the **static projection floor** of every rank $k\in\{16,32,64,128,256,512\}$ on the six
   development cases, in the same normalisation the panel's `best-found` column uses, plus the
   singular-value spectrum and the tail energy fraction.
3. **The validity check that must pass before leg (a) is read.** The incumbent family's floors are
   compared against job `3780638`'s own `best-found` column (61.6503 / 47.0681 / 19.8147 / 10.1189
   / 3.7686 / 0.6090 %, read from `comparators/bpn301-summary.json`, never typed) at $10^{-3}$
   relative. If that fails, the measurement is not the panel's measurement and no comparison is
   read from it.
4. **Leg (b).** The complete tuned Newton grid of job `3780638` — `fft_tight`, `dense_tight` and
   $\{10^{-2},10^{-3},10^{-4}\}\times\{\Delta t=0.005,0.01\}$ — timed for **both** families in one
   randomised order, three repetitions, 0.25 s burn-in, device sync, all repetitions retained, with
   Newton iteration counts and achieved linear residuals recorded. Plus the mesh probe at
   $L\in\{512,1024\}$.
5. **Output:** the full-order (cost, error) frontier of each family, the per-rank POD degradation
   ratio, the per-setting cost ratio, and the two go/no-go gates below.

### Stage 2 — the retrain `lvt01`. Only if the gate passes.

Stages A→B→C of the b-seeds chain (`sep_burgers_r3.py` → `sep_coeff_extract.py` →
`sep_hfit_run.py`), **the incumbent recipe unchanged** — $K=16$, $R=512$, `n_ff` 128, `g_hidden`
1024, `h_hidden` 256/512×2, 300000 + 200000 Adam steps, seed 0 — with the generator's viscosity
bounds as the only difference, so the sole changed variable remains the data regime. Reported: the
three layers (bank projection floor, best-found, solved) beside the incumbent cell's 0.39 / 2.54 /
2.56 %.

### Stage 3 — the panel `lvp01`. One allocation, one GPU.

The b-panel driver, restricted to what a fresh checkpoint supports: the correction ladder
$q\in\{0,16,32,64,128,256\}$ at **fixed test count $M$** (the `b-qxm` lane established fixed $M$ as
the honest control — `worktrees/2026-09-17-b-qxm/experiments/b-qxm/reports/2026-09-17-b-qxm.md`),
POD-LSPG at $k'\in\{16,\dots,512\}$ through the same weak objective and the same solver and budget,
the unrestricted $R=512$ bank, and the **tuned** full-order Newton controls at several tolerances
and time steps — tuned, not strawmen: the identical grid the incumbent panel used, which is the
grid that beats the incumbent ROM. Three timed repetitions, burn-in, device sync, randomised arm
order, exit reasons and stationarity recorded for every solve.

## 6. Pre-registered criteria

Let a *subject* be one timed arm and its two coordinates (median GPU ms over all repetitions and
cases; worst-over-evolved-times same-grid error against that job's own converged `fft_tight`).

**P — the pass criterion (primary).** **At least one reduced subject is non-dominated on
(median GPU ms, worst evolved %) against every same-job full-order control and every POD rank, with
every subject in the comparison converged** (zero budget exits, stationary or at an
attained-round-off initial fit under the rule `b-panel` DESIGN §5 pre-registered, which this lane
adopts verbatim). Non-dominance is computed over the admissible set of the panel job; the
reduced-only frontier is reported beside it but is **not** the criterion.

**G-a — leg (a) gate (gate job).** The low-viscosity POD-512 floor is at least $2\times$ the
incumbent POD-512 floor on the worst-all-times metric, in the same job. Reported for every rank.

**G-b — leg (b) gate (gate job).** At least one full-order setting costs $\ge1.5\times$ more on the
low-viscosity family than on the incumbent family, in the same job.

**Go/no-go.** Stage 2 is submitted if **G-a passes**. G-b is reported either way and is not by
itself blocking, because leg (a) alone can still move the reduced-vs-POD comparison even if it
cannot move the reduced-vs-full-order one; if G-a passes and G-b fails, the report says in advance
what that means: the cell will be reported as a **POD-degradation result**, not a speed result, and
P is expected to fail.

**Falsification — what would say the hypothesis is false.**

* **F1 (leg a dead).** G-a fails: POD does not degrade materially at $10\times$ lower viscosity.
  Then the regime is not harder in the sense the claim needs, **the lane stops at the gate**, no
  training job is submitted, and the negative is the deliverable. This is the branch the lane brief
  names explicitly.
* **F2 (the ROM degrades at least as much).** G-a passes, stage 2 runs, and the head's three-layer
  decomposition degrades by at least the factor POD does — i.e. the low-viscosity bank floor /
  best-found / solved are each $\ge$ the POD-512 degradation ratio times the incumbent's. Then the
  nonlinear manifold buys nothing from the harder regime and the hypothesis is false.
* **F3 (the full-order solve does not get more expensive).** G-b fails and P fails. Then the
  non-dominance loss is structural in this discretisation — an exactly preconditioned
  backward-Euler Newton step is simply cheap — and the report says so, with the measured cost
  ratios, as a closing negative.
* **F4 (under-resolution swallows the effect).** The $L=256$ discretisation error of the
  low-viscosity family against the 4096-interval reference exceeds the incumbent's 4.0265 % by
  more than $3\times$. Then every low-viscosity number is reported with that caveat attached
  inline, and the cell is not offered as a paper headline without a finer-mesh confirmation.

Any of F1–F4 is a real and publishable negative and closes the question the review raised.

## 7. Gates before any verdict

**In job** (`result.json`): `jax_backend=gpu`; float64; `JAX_DEFAULT_MATMUL_PRECISION=highest`;
`cohorts_differ_only_in_viscosity`; `incumbent_cohort_matches_abl01_to_one_ulp` (values to
$4\times10^{-16}$, with bitwise equality as a probe — `np.exp` differs by one ulp between the GB10
and the cluster, CLAUDE.md landmine and b-seeds A2); reference residual $<2\times10^{-11}$;
snapshot residual $\le10^{-8}$; POD orthonormality deviation $\le10^{-8}$; the two-route POD mode
parity; `timed_tight_matches_untimed_truth` (the timed `fft_tight` subject reproduces the untimed
same-grid truth bitwise); `incumbent_pod_floors_reproduce_bpn301`; training/evaluation disjointness;
`final_cohort_unopened`.

**Audit** (`audit_gate.py`, NumPy only, imports neither JAX nor the driver): every error recomputed
from the saved fields to $10^{-9}$; every field hash re-verified; the POD floors recomputed by a
**different route** — from the snapshot Gram's eigen-decomposition rather than from the modes; POD
orthonormality recomputed from the saved $G W$; all medians, frontiers and both legs' ratios
recomputed; every in-job gate replayed; the cohort redrawn from the seeds.

**Local smoke before submission:** `gate.py` runs end to end at 64 intervals on both families from
a staged tree, and `audit_gate.py` passes on its output. The smoke validates no number.

## 8. Compute plan and the job cap

| job | attempt | contents | GPU | expected |
|---|---|---|---|---|
| 1 | `lvg01` | the gate: both families, references, POD, the full-order grid, the mesh probe | a100 | ≈ 1 h |
| 2 | `lvt01` | stages A–C at the low-viscosity family, one seed | a100 | ≈ 4 h |
| 3 | `lvp01` | the panel: ladder + POD-LSPG + free bank + tuned full-order controls | a100 | ≈ 1 h |
| 4 | `lvs01` | sealed cohort, only if stage 3 produced a verdict | a100 | ≈ 1 h |

Cap 8; four planned, four reserved for reruns. `--gres=gpu:a100:1`, `--mem 180G`,
`--exclude pax007`, partition `gpu`, never `preempt`. Still pending after 3 h → `scancel` and
resubmit as h100, then h200, then l40s, never changing the science. `squeue -u tawal01` before and
after every submit, one job per attempt directory. After each job: checksum collection, local hash
verification, the NumPy audit, Git-chunked archive, then deletion of the exact remote attempt
directory (paralab is 91 % full).

## 9. Reporting

Every number in `reports/2026-09-17-b-lowvisc.md` is generated by `reports/generate_lowvisc.py`
from the run JSONs, with a machine-readable `reports/summary.json` (one object per row: subject,
metric, value, job id, source SHA). No cost ratio is ever formed across jobs or GPUs. Equations in
LaTeX, diagrams in mermaid, a plain-language glossary at the end.

## 10. Amendments

**A1 (2026-09-17, before the first submission) — the Codex audit is unavailable and is
substituted.** The protocol requires an independent Codex (`gpt-6-astra`, a different model family)
audit of this design before the first job. The coordinator's notice in `LANE-PROTOCOL.md` records,
verified, that `codex exec` returns "You've hit your usage limit … try again at Sep 19th, 2026
11:33 AM", and instructs lanes not to block on it. Substituted, and recorded here as a
substitution rather than presented as equivalent: a **written self-audit** at
`reports/self-audit-design.md`, listing each claim this design makes, the JSON field or source line
it rests on, and the check that was run. It is the same model family as the author and is therefore
strictly weaker than the required audit. After 2026-09-19 11:33, if this lane is still open, the
Codex audit of the finished report is run and appended as a dated addendum, retracting anything it
overturns.

**A2 (2026-09-17, before the first submission) — the brief's viscosity range was wrong and is
corrected here.** The lane brief said the incumbent cell "uses viscosity drawn from roughly
$\nu\in[10^{-3},10^{-2}]$" and instructed that the exact family be confirmed from the generator.
It is $[10^{-2},10^{-1}]$ (`engines.params_draw:25`, `burgers2d_film.sample_params:181`). The
low-viscosity family is therefore $[10^{-3},10^{-2}]$ — one decade below the incumbent, which is
coincidentally the range the brief believed the incumbent to have. Recorded so that no later reader
mistakes the incumbent cell for a low-viscosity one.

**A3 (2026-09-17, while the gate job `lvg01` was queued, before any of its numbers existed) — the
stage-3 panel is pre-registered here, from the `b-qxm` lane's finding.** Self-audit row 13 left
"fixed $M$ is the honest control" as an open item. `b-qxm`'s report, DESIGN and round-2 audit JSONs
were read; the relevant facts, and what this lane adopts from them:

* $M$ is the number of weak test equations (the first $M$ sine modes in ascending
  discrete-Laplacian order); a cell has $K+q$ unknowns and the gate `overdetermined_weak_system`
  requires $M>K+q$. The incumbent ladder scheduled $M=4(K+q)$, which **confounds** the ladder:
  raising $M$ alone at $q=0$ from 64 to 256 already buys 1.8890 → 1.2710 %, i.e. 31 % of the
  scheduled ladder's log-span. A corner-path decomposition puts the split at 69 % rank / 31 % test
  count, and rung by rung the test-count share falls 0.835 → 0.046: **$M$ buys the bottom of the
  ladder, $q$ buys the top.**
* Both fixed-$M$ columns: $M=256$ spans 1.220× in evolved error over $q\le128$ and **fails** the
  $\ge2\times$ knob bar; $M=1088=4(K+256)$ spans 2.437× in error and 5.163× in cost and **passes**.
  On the allegation of selective reporting (review finding 14): `b-qxm`'s DESIGN §3 names
  `fixed1088` and its arithmetic rationale before any job was submitted, §6 pre-registers the
  headline rule, and the report prints the failing $M=256$ column beside it. The fair caveat,
  which this lane repeats, is that $M=1088$ is the *only* fixed $M$ that can hold $q=256$ at all.
* Round-2 jobs, on disk and not yet in that lane's report: the whole $M=1088$ column in one job
  (error span 2.437×, **within-one-job** cost span 5.247×); $q=512$ does **not** reach stationarity
  at $M=1088$, 2112 or 3168 (worst joint stationarity 0.1774 against a $10^{-6}$ bar), so it is not
  test-starvation; and $q=256$ at $M=1088$ is **test-starved by 28 %** — $M^\star=2176$.

**Adopted for stage 3, fixed now.** Headline column: fixed $M=1088$, $q\in\{0,16,32,64,128,256\}$,
the whole column in one job. Control column, printed beside it and not dropped whatever it says:
fixed $M=256$, $q\le128$. Additional cell $(q,M)=(256,2176)$, because the headline $M$ costs the
top rung 28 % of its accuracy. **No $q=512$.** Dense (exact) quadrature everywhere; no EQ rule
enters any panel number, since this lane has no certified rule for a new checkpoint. Solver
`varpro.make_block_lm` at per-step budget 600, IC budget 400, `gtol` $10^{-6}$, residual tolerance
$10^{-9}\|u_{\rm gauss}\|\sqrt{n_g}$; `XLA_PYTHON_CLIENT_MEM_FRACTION=0.55`; the Gauss-Jordan→LU
control cell at $(32,1088)$ kept in and excluded from spans. The panel job requests an **80 GB**
A100 or an H100/H200 — `b-qxm` records that a panel with several large $M$ does not fit a 40 GB
card. POD-LSPG at $k'\in\{16,32,64,128,256,512\}$ through the same objective, solver and budget;
the unrestricted $R=512$ bank; and the tuned full-order Newton grid of §5, all in that one job.

**Consequence for criterion P, recorded before the panel runs.** A dense-quadrature reduced query
evaluates the residual at every node, so its cost per iteration is of the same order as a
full-order Newton step and it needs more iterations: in job `3780638` the cheapest dense reduced
arm is 284.8 ms against `fft_tight` at 89.3 ms. Without hyper-reduction, P therefore requires the
full-order grid to become several times more expensive — which is leg (b), the leg §2 already
records as the weaker one. **If leg (b) fails, P is expected to fail, and the lane's deliverable is
F3 plus whatever leg (a) shows.** This is written before the measurement so the outcome cannot be
presented as anticipated after the fact. Fitting and certifying an EQ rule for the new checkpoint
(the q-ridge machinery, roughly an hour of NNLS per rung) is the obvious extension and is **not**
in the four-job plan; it is named here as the first thing to buy with the reserve jobs if leg (a)
passes and leg (b) is merely weak rather than absent.

**A4 (2026-09-17, same time) — the generator patch and its parity proof.** Stage 2 needs training
data at the low-viscosity family, and the training chain draws its parameters inside
`burgers2d_film.sample_params`, not through `engines.params_draw`. This lane therefore carries its
own copy at `experiments/b-lowvisc/deps/burgers2d-coord-rom/burgers2d_film.py`, which differs from
the copy the incumbent's own training job staged in exactly one way: the viscosity bounds are read
from `BURGERS_NU_LO` / `BURGERS_NU_HI`, defaulting to the incumbent's `(0.01, 0.1)`, and the `z`
descriptor's log-$\nu$ centre and scale are derived from those bounds rather than written as
`log(sqrt(0.001))` and `0.5*log(10)`. `check_generator_parity.py` loads both modules and compares
`sample_params` over all seven draws the recipe makes (seeds 0/576, 1000/4032, 1/8, 0/128,
7090702/4, 911702/2, 17092026/6): under the defaults every column is **bitwise identical**,
including `z`; at the low-viscosity bounds the first four columns are bitwise identical and
$\nu_{\rm hi}/\nu_{\rm lo}=10$ to $1.07\times10^{-15}$. Recorded in
`checks/generator-parity.json`; the training job reruns the check on the cluster before its first
training step and fails if it does not pass. b-seeds' copy is left untouched, so its own
provenance hashes still hold.

**A5 (2026-09-17, after the gate job `lvg01` returned and before stage 2 started) — the gate
verdict, F4, and one audit check that fails without moving a number.** Recorded here so the
pre-registration and the outcome sit in one file.

* **G-a PASSES** at $12.092\times$ against a $2\times$ bar (POD-512 worst-all-times floor
  $0.6090\,\% \to 7.3643\,\%$; worst-evolved $83.79\times$). The degradation is monotone in rank:
  $1.09/1.16/1.77/2.29/4.05/12.09$ at $k=16/32/64/128/256/512$.
* **G-b PASSES** at $3.292\times$ against a $1.5\times$ bar, and *every* one of the eight tuned
  full-order settings costs more on the low-viscosity family ($1.140$–$3.292\times$), with Newton
  totals rising on every setting. §2 recorded leg (b) as the weaker leg before the measurement;
  it passed anyway, so the honest prior is recorded as having been too pessimistic.
* **Therefore stage 2 is submitted** (`lvt01`, slurm 3804337). The recipe is unchanged from §5:
  $K=16$, $R=512$, seed 0. The floors do **not** argue for raising $R$ — they argue that a linear
  basis of any affordable rank is bad here, which is the hypothesis, not a reason to change a
  second variable at once.
* **F4 IS TRIGGERED.** The low-viscosity $L=256$ discretisation error against the 4096-interval
  reference is $20.8206\,\%$ against the incumbent's $4.0265\,\%$, a ratio of $5.171\times$, above
  F4's $3\times$ bar; it remains $13.95\,\%$ at $L=512$ and $8.79\,\%$ at $L=1024$. Per §6 this
  does not stop the lane (the go/no-go is G-a) but the caveat is attached inline to every
  low-viscosity number and the cell is not offered as a paper headline without a finer-mesh
  confirmation. §3 predicted this consequence of first-order upwinding before the job ran.
* **One audit check fails and is not retracted.** `pod_orthonormal_incumbent` is $7.21\times10^{-5}$
  against a $10^{-8}$ bar. It is a consequence of leg (a) rather than a defect: the *incumbent*
  snapshot Gram is numerically rank-deficient at 512 (eigenvalue ratio $3.41\times10^{-12}$), so the
  Gram-eigenvector route loses orthogonality in the tail. Bounded in
  `reports/self-audit-gate.md` row 15 and generated into the report: the deviation is
  $4.06\times10^{-9}$ over the leading 256 modes, and recomputing the $k=512$ floor with the full
  oblique projector moves it $7.87\times10^{-7}$ relative ($\le5.20\times10^{-11}$ at every other
  rank). The low-viscosity basis, which carries the result, is at $1.05\times10^{-9}$.
* **The written self-audit substituting for Codex** (A1) is at `reports/self-audit-gate.md`.

**A6 (2026-09-17) — stage 3 is not yet pre-registerable beyond A3.** A3 fixed the panel's
columns. Nothing in `lvg01` changes them. Stage 3 remains conditional on `lvt01` producing a
checkpoint whose three-layer decomposition is worth panelling; if the low-viscosity bank floor,
best-found and solved all degrade by at least the POD-512 factor, **F2 fires and the lane stops
before the panel** with the negative as the deliverable.

**A7 (2026-09-17, after the training job `lvt01` returned, before any panel) — stage 2 collected;
F2 evaluated on its training-stage analogue and does NOT fire, provisionally.** `lvt01` = slurm
3804337 (A100, pax142, COMPLETED 0:0, 3h48m) ran stages A–C at the incumbent recipe with the
viscosity family as the only change; `audit_train.py` (16 checks, 0 failed) confirms the recipe
field by field, the inverted data fingerprint (same shapes, different sums), and every hash.

* F2 as written in §6 needs the **panel** bank floor / best-found / solved on the development
  cohort, which do not exist before stage 3. What stage 2 supplies is their like-for-like
  training-stage analogue on the 408 held-out test states (the incumbent's own test trajectories
  with $\nu/10$), against the incumbent's own training run (job 2837431) and b-seeds' reseeds.
* Bank floor (worst held-out) degrades $9.286\times$; best-found (worst held-out) degrades only
  $1.871\times$ ($2.7651\,\%\to5.1726\,\%$) against the POD-512 factor of $12.092\times$. F2 is
  conjunctive, so with best-found a factor 6.5 short of the bar it **cannot fire on the solved
  layer alone**; recorded as *provisionally not firing*, with the panel as the pre-registered
  evaluation. The stage-3 panel is therefore warranted under §5.
* The on-thesis observation, recorded with its cohort caveat: on the incumbent family the
  $K=16$ manifold's best-found ($2.77\,\%$, held-out) sits above the POD-512 floor ($0.61\,\%$,
  development); on the low-viscosity family it sits below it ($5.17\,\%$ vs $7.36\,\%$). The
  panel puts both on one cohort in one job; until then this is not a paper number.
* F4 still travels with every number (A5).
* Nothing submitted at this amendment; the next job is the coordinator's call. Recommendation
  recorded: `lvp01` (A3) first, the $L=512$ F4 confirmation second, because a finer-mesh
  confirmation of a cell that fails P is wasted compute.
* `collect.py` now excludes the 567 MB extraction npz from the Git archive for training attempts
  (its SHA256 is verified on both sides and recorded beside the chunks), as its own docstring
  always said it should; the gate attempt's archive is unaffected.

**A8 (2026-09-17, after A7, before the panel was submitted) — the stage-3 panel `lvp01` as
built, with the coordinator's two additions.** Approved by the coordinator against A3 verbatim:
fixed $M=1088$ ladder $q\in\{0,16,32,64,128,256\}$, the fixed $M=256$ control column $q\le128$,
the $(256,2176)$ cell, POD-LSPG at $k'\in\{16,\dots,512\}$ through the same weak objective, solver
and budget, the unrestricted $R=512$ bank at $M=1024$, and job `3780638`'s tuned full-order grid
(`fft_tight`, `dense_tight`, $\{10^{-2},10^{-3},10^{-4}\}\times\{\Delta t=0.005,0.01\}$), three timed
repetitions with burn-in, device sync and randomised order, one allocation on one 80 GB A100
(`--constraint=a100-80G`, `--mem 180G`, `--exclude pax007`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`
as `bpn301`). No EQ rule, no fast kernel, no FNO, no $q=512$.

*Mechanics, and the one thing a fresh checkpoint needs that the incumbent panel did not.* The
incumbent panel transferred `directions_qtd02.npz` — `directions.audited` (the `old` rule:
field-metric POD of the frozen head's static reconstruction residual) evaluated on the *incumbent*
checkpoint by job 3757505. Correction directions are a function of the checkpoint, so `lvp01`
builds its own in-job as stage 1 (`lv_directions.py`: `b-seeds/seeds_run.py`'s setup up to
`DIR.audited`, the qtd02 / b-seeds residual settings verbatim — 1024 residual snapshots, 4 starts,
budget 200, seed 20260915 — with the cohort and training draws at the low-viscosity bounds) and
writes the file and the `PROVENANCE.json` entry that stage 2 hash-asserts. Stage 2 is
`lv_panel.py`: b-panel's `panel.py` (job 3780638) with four `LV-PATCH` edits recorded in
`checks/lv_panel.diff`, none touching an arm's arithmetic: (1) the two cohort draws through
`lv_common.params_draw` at the config's bounds; (2) `extra_dense_cells` declaring explicit
$(q,M)$ cells; (3) those cells in the directions prefix-hash set; (4) every dense strict-tolerance
cell sharing the one compiled query per $q$ (queries are $M$-independent — $M$ enters through the
operator data at invocation — the b-qxm mechanics, which is also what keeps 12 cells resident).
`config-panel.json` is generated by `make_panel_config.py`: every shared key read from the
incumbent panel's config, the cohort hash and reference-field probes from `lvg01`, nothing typed.

*The coordinator's two additions.* (a) Every reduced rung and every full-order setting records
its error against the 4096-interval reference beside the same-grid error — `panel.py` already
writes `error_vs_reference` on every invocation and `worst_reference_percent` on every subject;
the smoke confirms every row carries it (`checks/smoke-panel64.json`), and the report will print
the two columns side by side so F4 travels with every row. (b) The non-dominated set on
(same-job median GPU ms, worst evolved %) over the admissible subjects against the tuned
full-order grid and POD-LSPG, by b-panel's rule verbatim (`audit_panel.py::nondominated`, applied
to `converged` subjects under b-panel DESIGN §5's convergence rule, which §6 adopts) — computed
in this lane's audit and report from `result.json`, never in the job.

*Smoke.* `checks/smoke-panel64.json`: both stages end to end at 64 intervals from the staged tree
— directions rank 64, field orthonormality $3.7\times10^{-14}$; the panel builds, warms and times
every family (four fixed-$M$ cells, two POD ranks, the free bank, the eight full-order settings),
`PANEL COMPLETE`, nothing dropped. It validates no number.

*Criterion P is evaluated on this job's admissible set*, per §6, with F4's caveat on every row.
Jobs: 3 of 8.

**A8.1 (2026-09-17, minutes after A8, job already queued) — correction to A8's smoke sentence.**
A8 says the smoke "confirms every row carries" the vs-reference error. The first draft of
`checks/smoke-panel64.json` tested a *guessed* key name (`error_vs_reference`) and recorded
`False`; the driver's actual per-invocation keys containing "ref" that are present on every row
are `[]`, and its per-subject keys are `[]`. The record is corrected to
the real keys; A8's claim does NOT stand and the report generator must derive the column from the saved fields against the reference artifacts. The job is
unaffected either way: the driver is `bpn301`'s, whose summary carries `worst_reference_percent`
for every subject family, and the reference fields are saved as artifacts so the column can be
recomputed by the audit from the fields regardless.

**A8.2 (2026-09-17, minutes after A8.1).** The `error` block keys are `['absolute_rms_max', 'current_relative_max', 'current_relative_per_time', 'fixed_initial_max', 'fixed_initial_per_time']`;
not every row carries a reference error there ([]). A8.1 stands: the report derives the column from
the saved fields against the reference artifacts.

**A8.3 (2026-09-17, closing the A8.1/A8.2 thread).** Which error the per-row `error` block holds is
settled by the `fft_tight` row itself: its `fixed_initial_max` is 0.040076 (the 64-vs-128-interval
discretisation error in the smoke; a same-grid error of the same-grid reference would be exactly 0),
so **the per-row `error` block is the error against the reference** — it is on every row, which is
what the coordinator's addition (a) asks for. The same-grid column is derived by the audit from each
row's saved field artifact against the same job's `fft_tight` fields (b-panel's audit route), which
is how `bpn301`'s summary obtained both `worst_all_times_percent` and `worst_reference_percent`.
A8 stands; A8.1 and A8.2 were the wrong-key detour, kept as written.

**A9 (2026-09-17, after the panel `lvp01` = slurm 3817807 returned) — the verdicts. P FAILS (F3);
F2 does NOT fire; the ladder is monotone but below the knob bar.** All 27 subjects converged
(zero budget exits), so P was evaluated on the whole panel. The non-dominated set on (same-job
median GPU ms, same-grid worst-evolved %) is four full-order settings — `nt1e-2_dt01`,
`nt1e-3_dt005`, `dense_tight`, `fft_tight` — and **no reduced subject of any kind**:
`nt1e-3_dt005` (54.9 ms, 0.197 %) beats every reduced subject on both axes, the cheapest of which
is `q0_M256` at 300.3 ms / 8.39 % and the most accurate `q256_M2176` at 2864 ms / 6.19 %. This is
F3 exactly as §6 and A3 anticipated: leg (b) raised the full-order cost 3.3×, but a dense reduced
query is 5.5× dearer than a full-order setting 40× more accurate. **The non-dominance loss is
structural in this discretisation, at low viscosity as at the incumbent's.**

The pre-registered F2 on the development cohort: bank floor 0.3918 → 7.4455 % (19.0×),
best-found 2.5447 → 11.3934 % (4.48×), solved $q=0$, $M=256$ 2.5629 → 8.3933 % (3.27×) against
the POD-512 factor 12.09×. F2 needs all three; only the linear bank does. **F2 does not fire** —
the nonlinear head degrades 3–4× where every linear object degrades 12–19×. A7's provisional
held-out reading (1.87× on best-found) is superseded and agreed in direction. Caveat recorded:
the low-viscosity `solved/best-found` is 0.737 — the 8-start reconstruction oracle found a worse
minimum than the solve did, so the best-found is an upper bound here and 4.48× over-states the
degradation; the solved layer (3.27×) is the cleaner statement.

Ladders: the fixed $M=1088$ column is monotone in the evolved error (9.05 → 6.67 %), all
converged, error span 1.357× over a cost span 3.25× — **below the 2× knob bar** (the incumbent's
$M=1088$ column spanned 2.437×). The $M=256$ control is **not** monotone (8.39 → 10.26 %):
test-starved at $q=128$, as b-qxm found. $(256, 2176)$ reaches 6.19 % (1.08× better than
$(256,1088)$ for 1.58× the cost) and is the most accurate reduced subject in the job. Every
neural rung beats every POD-LSPG rank on the same-grid error (best POD-LSPG 11.25 % evolved at
$k'=256$; $k'=512$ is worse, 17.78 %, against its own 10.28 % floor — the solver, not the
subspace) and the free $R=512$ bank's own solve (7.79 %); the reduced-only frontier is neural
from $q=64$ up. That is the on-thesis positive, and it is a reduced-vs-reduced statement only.

F4 on every row: all reduced rungs sit at 17.8–19.4 % against the 4096-interval reference while
the same-grid tight solve is 20.82 % from it; nothing on this mesh can be told apart against the
continuum below that. The audit (`audit_panel_source_bpanel.py`, verbatim, 0 failed checks)
recomputed every error from the saved fields; the vs-reference worst-evolved column is recomputed
in `reports/panel_tables.py` from the same fields.

Jobs used: 3 of 8. Recommendation on the $L=512$ F4 confirmation, not submitted: **do not run it
for this cell.** It would cost a gate + retrain + panel (≈ 1 + 6 + 2 h on 80 GB cards, 3 jobs)
to resolve a caveat on a cell that fails P for a structural reason no mesh refinement changes —
the full-order Newton step gets *relatively* cheaper on finer meshes with the exact Helmholtz
preconditioner. The one thing a finer mesh could change is the reduced-vs-POD comparison, which
already favours the head here. If the paper wants the F4 caveat closed for the reduced-vs-POD
statement, the cheapest honest step is the $L=512$ gate alone (leg (a) at 512, one job, ≈ 1 h),
not a retrain.

**A9.1 (2026-09-17, minutes after A9) — a hand-written sentence corrected by the generated
report.** A9 says `nt1e-3_dt005` (54.9 ms, 0.197 %) "beats every reduced subject on both axes".
It beats every *neural* subject (the cheapest neural rung is `q0_M256` at 300.3 ms); the cheapest
reduced subject overall is `pod16_M64` at 46.5 ms, which it does not undercut. The cheapest
full-order setting that beats *every* reduced subject on both axes is `nt1e-2_dt01` at 14.0 ms and
3.895 % (the most accurate reduced subject is 6.187 %), as `summary.json` rows
`P/cheapest_fom_beating_every_reduced_subject` record. P's verdict is unchanged. The lab-log entry
of the same hour carries the same slip and is corrected there.

**A10 (2026-09-20, after the independent report audit) — repair source attribution only.**
The report generator carried the gate job's commit into later training, panel and historical
comparator rows. That attribution is retracted. `reports/source_provenance.py` reconstructs the
three stages' actual revisions from retained job records and verifies staged source hashes against
the named Git objects. It separately checks the historical reseeds and incumbent panel, whose
comparison filename predates its attempt renaming. The older incumbent extraction/head run has
no retained run-commit record: its source revision is explicitly unknown, with its manifest and
result hashes retained. No revision is guessed from a later checkout. An excluded external
operator checkpoint in the incumbent panel archive is recorded as manifest-only evidence; it is
not used to certify a Git source revision or any low-viscosity numerical result.

The combined summary now has no single source revision. It exposes revisions by stage and by
row, plus a hashed provenance catalog. Derived comparisons keep their original primary job
labels and all contributing source records remain separately identified. The generated report
explains this correction. `reports/audit_provenance_repair.py` requires exact recursive equality
with the pre-repair summary after removing only source metadata, and rechecks archive evidence.
Every numerical value, failed check, physical-resolution caveat, frontier verdict and provisional
reading is preserved. No new experiment ran and no scientific acceptance gate was relaxed.
