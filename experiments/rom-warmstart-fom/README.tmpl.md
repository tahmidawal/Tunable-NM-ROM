# ROM-warm-started FOM — cost of an FOM-exact hybrid

**Question.** After the ROM reaches some tolerance, hand its solution to the FULL-ORDER
solver as an *initial guess* and finish to full accuracy. The answer is then FOM-exact by
construction, so the ROM's accuracy ceiling stops being a limitation and the only remaining
question is **cost**: what is the total, and how does it scale with `N`?

If the hybrid wins, every accuracy objection the reviewers raised — including *"a direct
solver beats you"* — is sidestepped, because the delivered field **is** the FOM's field.

Scope: **Poisson-2D** and **Burgers-2D**. Both arms reuse the frozen reference harnesses
(`poisson2d-rom-objective/`, `burgers2d-rom-latent-stepping/`); neither the ROM nor the PDE
operator is re-implemented here.

---

## TWO RISKS, STATED UP FRONT AND MEASURED

These are not caveats discovered afterwards; they are the two ways this experiment was
expected to fail, and both are measured rather than assumed.

**(a) Poisson: the ROM is not free, and at small `N` the FOM is very cheap.** The reference
cell measured the Poisson latent solve at 19.7–20.0 ms flat across a 289× DOF range while the
FOM CG went 5.6 → 96.0 ms. A hybrid that pays ~20 ms *before* it starts the FOM **cannot**
win below some resolution. Locating that crossover `N` — or establishing that there is none
inside the ladder — is a headline result here, not a footnote.

**(b) Burgers: the FOM already warm-starts.** `burgers2d_film`'s implicit solver starts each
Newton solve from `u_{n-1}`, which at `dt = 0.005` is an excellent guess. The ROM must beat
**that** bar. The win was expected to concentrate in the first steps and to be small or
negative overall. To keep the comparison honest a **third arm** is measured: linear
extrapolation `2u_{n-1} − u_{n-2}`, the classical trick a practitioner would reach for first.
The ROM is therefore compared against the best cheap alternative, not only the weakest one.

A third, related honesty measure was added on the Poisson side: the FD Poisson system on a
square with Dirichlet walls is **diagonalised exactly** by the same discrete sine basis the
ROM uses for its test modes, so an exact **direct** solve is available and is timed alongside
everything else. That is the strongest form of the reviewers' objection and it is reported,
not avoided.

---

## Design

### Poisson-2D (`wsf_poisson.py`)

```
total  =  t_pre  +  t_rom  +  t_decode  +  t_fom(from the ROM guess)
baseline                                =  the SAME CG from a ZERO start
```

| stage | what it is | charged to the hybrid? |
|---|---|---|
| `t_pre` | the per-query input projection `Λ^{-1} Φ_M^T f` (the mode table is a per-mesh constant, built offline and timed separately) | **yes** |
| `t_rom` | the jitted weak-form LM latent solve on the NNLS-EQ quadrature (`M=64` test modes, `m=256` meshfree points) | **yes** |
| `t_decode` | decoding the latent onto the FOM's interior grid — the array CG is handed | **yes** |
| `t_fom` | CG from that array to `tau_FOM` | **yes** |

- **ROM tolerance.** `rom_tau` is the **relative reduction of the weak-form objective from
  the initial guess**:

  > stop at the first *accepted* LM iterate with `V(z) ≤ rom_tau · V(z_0)`, where
  > `V(z) = ‖ Wl · (Φ_M^T (w_q ⊙ dec(z, pts))) − f_m ‖_2` and `z_0` is the initial latent
  > (the mean training latent). `rom_tau = 0` disables the test and the solver is then the
  > reference LM, stopping on its own relative-decrease / step-size / budget rules.

  This is the identical definition used by the sister `cost-to-tolerance` cell, so the two
  compose. A tolerance on `‖Au−f‖/‖f‖` is *unreachable*: at the weak-form solution that sits
  near 2e-1 while the field error is ~8e-3.
  Ladder: `0.5, 1e-1, 1e-2, 1e-3, 0`. The three middle values are the ones the sister
  `cost-to-tolerance` cell also runs; `0.5` probes the very loose end and `0` the
  reference solver's own stopping rules (reported as `ref. stops`, **not** as
  "converged" — the reference LM may stop on budget or lambda saturation).

- **`tau_FOM` ∈ {1e-6, 1e-8, 1e-10}**, on the relative discrete residual `‖Au−f‖_2/‖f‖_2`.

- **One CG kernel for both arms.** `wsf_util.make_cg` is a jitted, iteration-counting CG in
  which `x0` and `tau` are *runtime arguments*. The warm-started arm and the zero-start arm
  therefore execute the same compiled code with the same stopping test; the only difference
  is the value of `x0`. It is an **outer true-residual loop** around the textbook inner
  recursion: the returned iterate provably satisfies `‖b−Ax‖/‖b‖ ≤ tau` on the *recomputed*
  residual, not merely on the recursively updated one, because recursive-residual drift
  depends on the trajectory and would otherwise make "FOM-exact to tolerance" an
  initial-guess-dependent property. The testbed's own `jax.scipy.sparse.linalg.cg` cannot
  report an iteration count, so it is retained as the **correctness reference** (checked at
  every reported tolerance on several right-hand sides) and is additionally **timed with a
  runtime `x0` for both arms** as a baseline sensitivity check, rather than being the timed
  baseline itself.

- **What "FOM-exact" is checked against.** The correctness gate is *reference-free*: every
  row asserts that the delivered iterate's true relative residual is `≤ tau` in **both**
  arms. The reported `err_final` (against a reference solution computed by CG at 1e-13) is a
  secondary check, and at the tightest `tau` it is bounded below by the reference's own
  accuracy — the achievable relative residual of the reference grows with `N` (measured
  1.0e-13 at `N=128`, 5.7e-13 at `N=256`), an f64 floor of the FD operator, so the reference
  is only required to be 10x tighter than the tightest reported tolerance.

- **`N ∈ {32, 64, 128, 256, 512}`.** The coordinate decoder is meshfree, so the same `N=64`
  `K=8` hard-BC checkpoint is used at every `N` and the NNLS-EQ weights are **refit on each
  `N`'s grid** (as in the reference `followup/fu_timing.py`).

### Burgers-2D (`wsf_burgers.py`)

```
hybrid total = t_rom_ic + t_rom_rollout + t_decode + t_fom(from the ROM guesses)
baseline     = the SAME implicit chain warm-started from u_{n-1}
third arm    = the SAME implicit chain warm-started from 2u_{n-1} − u_{n-2}
```

- **One chain kernel for all three arms.** `make_chain` builds a single jitted 50-step
  backward-Euler chain whose only per-arm difference is a **traced** `guess_mode` integer.
  Newton stopping test, linear tolerance, operator, compilation and warm-up are therefore
  identical across arms by construction.
- **Newton stopping test:** `‖R(u, u_prev, ν)‖_2 ≤ tau · ‖u_prev‖_2` — the testbed's own
  convergence metric (`burgers2d_film.newton_step` reports exactly this ratio).
  `tau ∈ {1e-6, 1e-8, 1e-10}`.
- **Why a new Newton driver.** The testbed's Newton is a *fixed-length 8-iteration scan* and
  `jax.scipy.sparse.linalg.bicgstab` cannot report an iteration count, so neither can answer
  "how many iterations did the warm start save". The driver here imports the testbed's
  `residual` **verbatim** — the discrete operator is not re-implemented — and every `(N, tau)`
  asserts that the previous-step arm reproduces the testbed's own rollout.
- **BiCGStab NaN landmine.** Known from an earlier round: once the Newton residual reaches
  machine epsilon, BiCGStab's `rho`/`omega` inner products can underflow and return a NaN
  step. Here the Newton loop exits on its tolerance test before that can happen, and the
  counting BiCGStab still detects underflow of `rho`, `rhat^T v` or `t^T t` and any
  non-finite iterate, freezes on the last good state, and **reports** the occurrence
  (`bicgstab_breakdowns`, `newton_flags_nonzero`, `health_warning`). Breakdowns are counted,
  never dropped. The solver also includes the **alpha half-step convergence test**: without
  it, an exactly converged `s = r − alpha A p` gives `t·t = 0` and a naive implementation
  declares a breakdown and discards a converged iterate.

- **Solver-health gate.** A Burgers configuration is published only if every step of every
  arm met the Newton tolerance with finite arithmetic; otherwise the job aborts. A cheap
  *failed* warm solve must never be able to contribute a headline speedup.
- **`N ∈ {32, 64, 128, 256}`**, EQ weights refit per `N`, variant `lspg:eq256:weak64` with
  the hyper-reduced (EQ-node) cold start — the reference cell's headline configuration.

### Timing protocol (non-negotiable)

Every ladder point is measured **sequentially in one process on one GPU** — cross-`N` timings
from different GPUs are invalid and have burned this project before. Warm-up 2, median of 7,
`block_until_ready`, f64, `JAX_DEFAULT_MATMUL_PRECISION=highest`, `jax_backend=gpu` asserted
before any work.

Two further precautions were added **because a first round of results needed them**, and both
matter more than the effect being measured:

- **GPU burn-in.** In the first round the Poisson baseline at `N=512` measured **0.0468 ms per
  CG iteration** and the warm arm **0.0399 ms** for the same work — a **17% systematic bias**
  in favour of whichever arm was timed later, i.e. in favour of the hybrid. Both the
  instrumented and the library solver showed it, so it is the A100 ramping up after the long
  CPU-bound NNLS-EQ fit, not a code difference. Every mesh now burns the device in for 3 s
  before any measurement.
- **Paired arms.** The zero-start baseline is timed **back to back with the warm arm on the
  same right-hand side**, inside the ROM ladder, so any residual clock drift hits both arms
  equally. The old separated measurement is retained as `t_fom_baseline_unpaired_ms`. All
  first-round Poisson results were discarded and the jobs relaunched.

### What is deliberately *not* varied

- **`dt`.** The Burgers testbed's `dt = 0.005` is hard-coded in `burgers2d_film`. The
  previous-step guess gets worse as `dt` grows, so the ROM's advantage should grow with `dt` —
  but changing `dt` changes the FOM, the training snapshots and the ROM together, which is a
  different experiment. Everything here is at the testbed's `dt`; the per-step curves show
  where the win would have to come from.
- **Preconditioning.** The Poisson CG is unpreconditioned, as in the testbed. A
  preconditioner would shrink the iteration count in *both* arms and therefore shrink the
  absolute saving a warm start can buy.

---

## Files

```
wsf_util.py       timing protocol, provenance stamp, the counting CG + its reference check
wsf_poisson.py    the Poisson arm
wsf_burgers.py    the Burgers arm (counting BiCGStab + the 3-arm implicit chain)
wsf_summarize.py  runs/hybrid_points.json (the flat output schema) + SUMMARY_TABLES.md
wsf_facts.py      every number the README quotes, derived from the JSONs
wsf_render_readme.py  README.md = README.tmpl.md + those facts (unknown placeholder = error)
wsf_figs.py       the figures (reports-pipeline style, PNG + PDF)
wsf_style.py      a copy of the frozen reports figure style
cluster/          make_cell.sh / cells.sh / launch.sh / pull.sh (namespace wsfom/,
                  one job per dir) and CANCELLING.md (explicit job IDs only)
in/               the two checkpoints (git-ignored; sha256 recorded below)
runs/             pulled cluster output, logs, and the generated hybrid_points.json
```

## Jobs

Eleven Slurm jobs, each in its own directory under `/cluster/tufts/paralab/tawal01/wsfom/`,
all on `--gres=gpu:a100:1`, `-p gpu`, submitted simultaneously.

| job dir | role | what it produces |
|---|---|---|
| `wsp_n{32,64,128,256,512}` | panel | Poisson: EQ refit at that mesh, ROM accuracy, CG iteration counts from both starts, solver cross-checks, and the within-mesh cost breakdown |
| `wsb_n{32,64,128,256}` | panel | Burgers: EQ refit, ROM rollout accuracy, per-step Newton and BiCGStab counts for all three arms, NaN-guard checks |
| `wsp_cons` | consolidated | Poisson: the whole `(rom_tau x N)` grid and the pure-FOM baseline at every mesh, **sequentially in one job on one GPU** |
| `wsb_cons` | consolidated | Burgers: the whole 50-step rollout from all three starts at every mesh, **sequentially in one job on one GPU** |

**Which numbers come from where.** Iteration counts, accuracy and solver-health fields are
hardware-independent and are taken from the fanned-out panels; table P5 checks that the panel
and consolidated runs agree on them exactly. **Every cross-`N` wall-clock number — the
headline total-time figure, the crossover result and the Burgers cost curve — comes only from
the consolidated runs**, and `wsf_summarize.select_consolidated` enforces that by pooling
consolidated rows by (source file, Slurm job id, GPU, commit, harness source hash) and using
a single group. Within one panel, the cost breakdown is valid because one panel is one job on
one GPU.

## Results

> Every number in this section is **substituted from the JSONs** by
> `wsf_render_readme.py` (facts in `wsf_facts.py`); `README.md` is generated from
> `README.tmpl.md` and an unknown placeholder is a hard error. The tables live in
> `SUMMARY_TABLES.md`, generated by `wsf_summarize.py`. Nothing here is typed by hand.
> Sources: {{n_reports}} complete reports, {{n_rows}} rows, {{n_skipped}} incomplete report(s)
> dropped. Cross-`N` wall clock: Poisson from Slurm job `{{p_job}}` on `{{p_gpu}}`, Burgers
> from job `{{b_job}}` on `{{b_gpu}}`.

### The answer

**Poisson.** {{p_headline}} Per tolerance the crossover mesh is N = {{p_cross_1em06}} at
`tau_FOM = 1e-6`, N = {{p_cross_1em08}} at `1e-8` and N = {{p_cross_1em10}} at `1e-10`.

**Burgers.** {{b_headline}} Linear extrapolation `2u_{n-1} - u_{n-2}` — one line of code, no
ROM — beat the previous-step start in {{b_extrap_wins}} configurations, so the ROM does not
even clear the cheapest classical alternative.

### Poisson: the numbers

At `tau_FOM = 1e-10`, best over the ROM tolerance ladder:

| N | DOF | pure FOM (ms) | hybrid (ms) | speedup | best rom_tau | max CG iterations saved |
|---|---|---|---|---|---|---|
| 32 | {{p_dof_N32}} | {{p_fom_1em10_N32}} | {{p_total_1em10_N32}} | {{p_best_1em10_N32}}x | {{p_besttau_1em10_N32}} | {{p_maxsave_1em10_N32}}% |
| 64 | {{p_dof_N64}} | {{p_fom_1em10_N64}} | {{p_total_1em10_N64}} | {{p_best_1em10_N64}}x | {{p_besttau_1em10_N64}} | {{p_maxsave_1em10_N64}}% |
| 128 | {{p_dof_N128}} | {{p_fom_1em10_N128}} | {{p_total_1em10_N128}} | {{p_best_1em10_N128}}x | {{p_besttau_1em10_N128}} | {{p_maxsave_1em10_N128}}% |
| 256 | {{p_dof_N256}} | {{p_fom_1em10_N256}} | {{p_total_1em10_N256}} | {{p_best_1em10_N256}}x | {{p_besttau_1em10_N256}} | {{p_maxsave_1em10_N256}}% |
| 512 | {{p_dof_N512}} | {{p_fom_1em10_N512}} | {{p_total_1em10_N512}} | {{p_best_1em10_N512}}x | {{p_besttau_1em10_N512}} | {{p_maxsave_1em10_N512}}% |

### Why the saving is small — the mechanism

The converged ROM reaches a held-out field error of **{{p_rom_err}}**, but that field has a
relative *discrete residual* of **{{p_rom_resid}}** and, decisively, a relative **A-norm
error of {{p_rom_anorm}}**. CG's convergence is governed by the A-norm error, so the warm
start only removes the fraction of the error the ROM actually resolves in that norm. The
result is a **single-digit percentage** of the iterations, and it *shrinks as the tolerance
tightens* — at `N = {{p_nmax}}` the best saving is {{p_maxsave_1em06_Nmax}}% at `1e-6`,
{{p_maxsave_1em08_Nmax}}% at `1e-8` and {{p_maxsave_1em10_Nmax}}% at `1e-10` — because a
fixed head start buys a fixed number of iterations out of a total that grows with the number
of digits demanded.

The gap between what the ROM's answer *looks* worth and what it *is* worth is the striking
part. A post-hoc diagnostic — plain CG from a zero start, with its saved iterates graded
against the reference solution in the **relative L2 field error**; it never stops a solver on
the reference — shows that plain CG needs **{{p_worth_N512}} iterations** at `N = 512` before
it is as accurate as the ROM, out of {{p_baseiters_N512}} for the full solve. Yet
warm-starting from that same field saves only {{p_maxsave_1em10_N512}}% of them. Being *as
accurate as* CG's {{p_worth_N512}}th iterate in L2 is not the same as *being* CG's
{{p_worth_N512}}th iterate: what CG contracts monotonically is the **A-norm**, in which the
ROM's error is only {{p_rom_anorm}} of the initial one, and the part the ROM never resolved
still has to be removed from scratch.

**A bad guess is worse than no guess.** {{p_n_negative}} of the measured configurations
needed *more* CG iterations from the ROM start than from zero, the worst by
{{p_worst_negative}}% ({{p_worst_negative_where}}). Loosening the ROM tolerance to make the
ROM cheap does not help: it makes the guess actively harmful.

**The decode is the structural problem.** The latent solve is mesh-independent, but the
decode of the latent onto the FOM's grid is not — it grows from {{p_decode_N32}} ms at
`N = 32` to {{p_decode_N512}} ms at `N = 512`. Any hybrid that hands a *field* to a
full-order solver must pay that O(n) cost, which is exactly the cost the ROM exists to avoid.

### The direct solver, measured rather than avoided

The FD Poisson system on a square with Dirichlet walls is diagonalised **exactly** by the
same discrete sine basis the ROM uses for its test modes. The resulting direct solve is
exact to {{p_directerr_N512}} relative error and takes **{{p_direct_N512}} ms at `N = 512`**,
against {{p_fom_1em10_N512}} ms for CG at `1e-10` — **{{p_cgdirect_N512}}x faster than the
iterative FOM, and {{p_direct_N32}}–{{p_direct_N512}} ms across the whole ladder.** On this
problem the reviewers are right: a direct solver beats the iterative FOM, and therefore
beats the hybrid, by two orders of magnitude. Reporting this is the point — the hybrid
framing removes the *accuracy* objection, and this measurement shows it does not remove the
*cost* one on a separable model problem.

### Burgers: the ROM start loses to the previous time step

Variant `{{b_variant}}` (m = {{b_m}} EQ nodes), 50 backward-Euler steps, at
`tau_FOM = 1e-8`:

| N | Newton (prev) | Newton (extrap) | Newton (ROM) | BiCGStab (prev) | BiCGStab (extrap) | BiCGStab (ROM) |
|---|---|---|---|---|---|---|
| 32 | {{b_newt_prev_1em08_N32}} | {{b_newt_ext_1em08_N32}} | {{b_newt_rom_1em08_N32}} | {{b_lin_prev_1em08_N32}} | {{b_lin_ext_1em08_N32}} | {{b_lin_rom_1em08_N32}} |
| 64 | {{b_newt_prev_1em08_N64}} | {{b_newt_ext_1em08_N64}} | {{b_newt_rom_1em08_N64}} | {{b_lin_prev_1em08_N64}} | {{b_lin_ext_1em08_N64}} | {{b_lin_rom_1em08_N64}} |
| 128 | {{b_newt_prev_1em08_N128}} | {{b_newt_ext_1em08_N128}} | {{b_newt_rom_1em08_N128}} | {{b_lin_prev_1em08_N128}} | {{b_lin_ext_1em08_N128}} | {{b_lin_rom_1em08_N128}} |
| 256 | {{b_newt_prev_1em08_N256}} | {{b_newt_ext_1em08_N256}} | {{b_newt_rom_1em08_N256}} | {{b_lin_prev_1em08_N256}} | {{b_lin_ext_1em08_N256}} | {{b_lin_rom_1em08_N256}} |

Wall clock at `tau_FOM = 1e-8`: the pure FOM takes {{b_fom_1em08_N256}} ms at `N = 256`
against a hybrid total of {{b_total_1em08_N256}} ms ({{b_speed_1em08_N256}}x), of which
{{b_rom_1em08_N256}} ms is the ROM stage and {{b_dec_1em08_N256}} ms the decode. The ROM's
own trajectory error is {{b_err_1em08_N256}}, comfortably good enough to be a plausible
guess — and still the FOM's own `u_{n-1}` is a better one.

This is **risk (b) landing exactly where it was expected to**: at `dt = 0.005` the previous
state is already an excellent guess, so a guess that is 1–2% wrong in a *different* way makes
the first linear solve harder rather than easier. The extra BiCGStab iterations, not the
Newton count, are where the ROM loses.

### Verdict

The hybrid does what it was meant to do — the delivered field is FOM-exact, so every accuracy
objection is answered — and it is **not worth its cost** on either problem at the meshes
tested. {{p_verdict}}, against an iterative FOM that a direct solver already beats by
{{p_cgdirect_N512}}x. {{b_verdict}}.

Separately, and reaching beyond this cell: measuring a *tolerance-based* FOM exposed that
**both of this project's FOM baselines are over-converged** — the Burgers rollout by a fixed
iteration count, the Poisson CG by a fixed 1e-13 tolerance. Every previously reported speedup
in either family is inflated. The correction is quantified below.

The useful result is the **mechanism**, and it generalises beyond this cell: an approximate
solution is a good initial guess only in proportion to the error it removes *in the norm the
outer solver contracts*, and a nonlinear-manifold ROM trained to minimise a field error does
not concentrate its accuracy there. That is a statement about what warm-starting can ever
buy, not about this ROM.

### Caveats

- **Measured, not assumed, and both stated risks landed.** Risk (a) — the ROM costs more than
  a small-`N` FOM — is why the crossover sits at `N = {{p_crossover}}`. Risk (b) — the
  Burgers FOM already warm-starts — is why that arm loses.
- **`dt` is not varied.** The Burgers testbed hard-codes `dt = 0.005`, where the previous
  state is a very good guess. The ROM's advantage should grow with `dt`, but changing `dt`
  changes the FOM, the training snapshots and the ROM together; that is a different
  experiment. Everything here is at the testbed's `dt`.
- **The Poisson CG is unpreconditioned**, as in the testbed. A preconditioner would cut
  iterations in both arms and shrink the absolute saving a warm start can buy.
- **The direct-solve comparison is specific to a separable operator** on a square. It does not
  transfer to a general geometry or a variable coefficient — but the reviewers' objection was
  raised about exactly this kind of model problem.
- **`err_final` at the tightest tolerance is limited by the reference solution's own
  accuracy** (relative residual {{p_ref_res_N512}} at `N = 512`, an f64 floor of the FD
  operator that grows with `N`). The correctness gate is reference-free: every row asserts the
  delivered iterate's true relative residual is at most `tau`, and the largest ratio observed
  was {{p_max_final_resid_ratio}}x `tau` on Poisson and {{b_max_resid_ratio}}x on Burgers.
- **Solver health.** {{b_breakdowns}} BiCGStab breakdowns, {{b_flags}} non-zero Newton flags
  and {{b_warnings}} health warnings across every Burgers configuration; the known NaN
  landmine did not fire, and had it fired it would have been reported, not dropped.
- **The best-per-`N` hybrid is post-selected** — it minimises over the ROM tolerance ladder
  using the same timing samples it reports. The full ladder is the primary table (P1).
- **Panel/consolidated agreement**: {{consistency_checked}} hardware-independent quantities
  compared between the fanned-out per-mesh jobs and the single-GPU consolidation runs,
  {{consistency_bad}} disagreements.
- **`N = 1024` was not run.** The measured crossover at `N = {{p_crossover}}` is the answer
  to the question as posed (`N` up to 512); an extrapolated break-even is given in
  `SUMMARY_TABLES.md` P4b and is an extrapolation, not a measurement.

## CORRECTION TO THE RECORD: both FOM baselines in this project are over-converged

This section is not a caveat about this cell. It is a correction that applies to results
already written on other branches, and the numbers below are what is needed to fix them
without rerunning anything.

**Are the previously reported speedups wrong, or merely conservative?** They are **wrong, in
the flattering direction** — every one of them divides by a denominator that did more work
than the stated accuracy required. They are not conservative. The corrected values are
smaller, by the multipliers tabulated below.

### Burgers: a fixed 8 Newton iterations per step, with no tolerance test

`burgers2d_film.make_rollout` runs `NEWTON_ITERS = 8` Newton steps per time step inside a
`lax.scan`, with **no convergence test**. Over a 50-step rollout that is
**{{oc_b_newton_fixed}} Newton steps and {{oc_b_newton_fixed}} inner BiCGStab solves,
regardless of need.** The accuracy it lands on is far past anything asked for: its worst
per-step relative Newton residual is {{oc_b_resid_N256}} at `N = 256`.

A Newton solve that simply stops at a tolerance needs far fewer iterations for the same
accuracy class:

| N | fixed-8 rollout (ms) | Newton steps | achieved residual | at `tau=1e-10` (ms) | Newton steps | over-convergence factor |
|---|---|---|---|---|---|---|
| 32 | {{oc_b_t_fixed_N32}} | {{oc_b_newton_fixed}} | {{oc_b_resid_N32}} | {{oc_b_t_tol_1em10_N32}} | {{oc_b_newton_tol_1em10_N32}} | {{oc_b_factor_1em10_N32}}x |
| 64 | {{oc_b_t_fixed_N64}} | {{oc_b_newton_fixed}} | {{oc_b_resid_N64}} | {{oc_b_t_tol_1em10_N64}} | {{oc_b_newton_tol_1em10_N64}} | {{oc_b_factor_1em10_N64}}x |
| 128 | {{oc_b_t_fixed_N128}} | {{oc_b_newton_fixed}} | {{oc_b_resid_N128}} | {{oc_b_t_tol_1em10_N128}} | {{oc_b_newton_tol_1em10_N128}} | {{oc_b_factor_1em10_N128}}x |
| 256 | {{oc_b_t_fixed_N256}} | {{oc_b_newton_fixed}} | {{oc_b_resid_N256}} | {{oc_b_t_tol_1em10_N256}} | {{oc_b_newton_tol_1em10_N256}} | {{oc_b_factor_1em10_N256}}x |

`tau = 1e-10` is the closest match to what the fixed-8 scan actually reaches, so those columns
are a **like-for-like** comparison. At the looser and more realistic `tau = 1e-8` the
tolerance-based FOM costs {{oc_b_t_tol_1em08_N256}} ms at `N = 256`
({{oc_b_factor_1em08_N256}}x over-convergence).

**The instrument is anchored to the archived baseline.** This cell does not model the old
denominator, it re-times *the same function*: `burgers2d_film.make_rollout(n)` at batch 1.
Measured here against the archived value, the two agree to {{oc_b_archcheck_N32}}% at
`N = 32`, {{oc_b_archcheck_N64}}% at `N = 64`, {{oc_b_archcheck_N128}}% at `N = 128` and
{{oc_b_archcheck_N256}}% at `N = 256`. The correction is therefore not an extrapolation from a
different setup — it divides the *same* measured baseline by a properly converged one. (The
archived JSONs record only `cuda:0`, not the card model; agreement at this level is itself
evidence the archived runs used a comparable card.)

### Burgers: the correction to apply

The previously reported speedup is `S_old(N) = t_fixed8(N) / t_ROM(N)`. The corrected one at a
stated Newton tolerance is

```
S_corrected(N, tau) = S_old(N) x m(N, tau),    m(N, tau) = t_tolerance(N, tau) / t_fixed8(N)
```

so **multiply the old number by `m`**. Two versions of `m` are given. The **time** multiplier
is a ratio of two wall clocks measured together on one GPU in this cell; applying it to a
speedup measured on a *different* GPU assumes that ratio is machine-independent. The
**iteration** multiplier — Newton steps actually performed, {{oc_b_newton_fixed}} for the
fixed-8 scan against the measured tolerance-based count — is a pure work count and carries
across machines with no assumption.

On the Burgers side the time multiplier additionally compares the testbed's own rollout
against **this cell's** Newton driver, so it carries an implementation difference as well as a
tolerance difference (the two were verified to agree to 2.6e-16 on a representative linear
solve, but the outer loops are not the same code). **Where the two multipliers disagree,
quote the iteration one.** Old values read from `{{oc_b_old_json}}`:

| N | old FOM (ms) | old rollout speedup | old end-to-end | m (time) | m (iterations) | corrected rollout | corrected end-to-end |
|---|---|---|---|---|---|---|---|
| 32 | {{oc_b_old_fom_N32}} | {{oc_b_old_speed_N32}}x | {{oc_b_old_e2e_N32}}x | {{oc_b_mult_1em10_N32}} | {{oc_b_multit_1em10_N32}} | {{oc_b_new_speed_1em10_N32}}x | {{oc_b_new_e2e_1em10_N32}}x |
| 64 | {{oc_b_old_fom_N64}} | {{oc_b_old_speed_N64}}x | {{oc_b_old_e2e_N64}}x | {{oc_b_mult_1em10_N64}} | {{oc_b_multit_1em10_N64}} | {{oc_b_new_speed_1em10_N64}}x | {{oc_b_new_e2e_1em10_N64}}x |
| 128 | {{oc_b_old_fom_N128}} | {{oc_b_old_speed_N128}}x | {{oc_b_old_e2e_N128}}x | {{oc_b_mult_1em10_N128}} | {{oc_b_multit_1em10_N128}} | {{oc_b_new_speed_1em10_N128}}x | {{oc_b_new_e2e_1em10_N128}}x |
| 256 | {{oc_b_old_fom_N256}} | {{oc_b_old_speed_N256}}x | {{oc_b_old_e2e_N256}}x | {{oc_b_mult_1em10_N256}} | {{oc_b_multit_1em10_N256}} | {{oc_b_new_speed_1em10_N256}}x | {{oc_b_new_e2e_1em10_N256}}x |

**The headline numbers `exp/2026-08-16-burgers2d-rom-latent-stepping`, the consolidated report
and `HANDOFF.md` all quote — the 8.0x end-to-end at `N = 256` and the
0.72x -> 7.96x N-ladder — become {{oc_b_new_e2e_1em10_N32}}x -> {{oc_b_new_e2e_1em10_N256}}x
once the denominator is a converged solver rather than a fixed-8 one.**

The multiplier is **N-dependent**: it is {{oc_b_mult_1em10_N32}} at `N = 32` and
{{oc_b_mult_1em10_N256}} at `N = 256`. It cannot be applied as a single constant, and the
N-ladder's *shape* — the claim that the advantage grows with mesh — changes with it.

### Poisson: NOT a clean negative — the same problem, from a fixed tolerance

The question was whether the Poisson CG baseline shares the defect. **It does.**
`poisson2d-rom-objective/followup/fu_timing.fom_solve` times
`jax.scipy.sparse.linalg.cg(op, F, tol=mp.CG_TOL, maxiter=mp.CG_MAXITER)` with
`CG_TOL = {{oc_p_cg_tol}}` — the tolerance used to **manufacture the truth data**, not a
tolerance any consumer of the solution would request. A fixed, very tight tolerance has
exactly the same effect as a fixed iteration count.

The comparison below is **like-for-like in the solver**: both columns are
`jax.scipy.sparse.linalg.cg`, differing only in tolerance. (This cell's own instrumented CG is
about 15% cheaper per iteration than `jax.scipy`'s, so comparing the testbed's `jax.scipy`
baseline against the instrumented one would conflate "tighter tolerance" with "different
implementation". That mixed ratio is recorded as `oc_p_factor_mixed_*` and is **not** what is
quoted here.)

| N | jax.scipy CG at `CG_TOL` (ms) | iterations | achieved residual | jax.scipy CG at `tau=1e-10` (ms) | iterations | over-convergence factor |
|---|---|---|---|---|---|---|
| 32 | {{oc_p_t_fixed_N32}} | {{oc_p_iters_fixed_N32}} | {{oc_p_resid_N32}} | {{oc_p_t_tol_native_1em10_N32}} | {{oc_p_iters_tol_1em10_N32}} | {{oc_p_factor_1em10_N32}}x |
| 64 | {{oc_p_t_fixed_N64}} | {{oc_p_iters_fixed_N64}} | {{oc_p_resid_N64}} | {{oc_p_t_tol_native_1em10_N64}} | {{oc_p_iters_tol_1em10_N64}} | {{oc_p_factor_1em10_N64}}x |
| 128 | {{oc_p_t_fixed_N128}} | {{oc_p_iters_fixed_N128}} | {{oc_p_resid_N128}} | {{oc_p_t_tol_native_1em10_N128}} | {{oc_p_iters_tol_1em10_N128}} | {{oc_p_factor_1em10_N128}}x |
| 256 | {{oc_p_t_fixed_N256}} | {{oc_p_iters_fixed_N256}} | {{oc_p_resid_N256}} | {{oc_p_t_tol_native_1em10_N256}} | {{oc_p_iters_tol_1em10_N256}} | {{oc_p_factor_1em10_N256}}x |
| 512 | {{oc_p_t_fixed_N512}} | {{oc_p_iters_fixed_N512}} | {{oc_p_resid_N512}} | {{oc_p_t_tol_native_1em10_N512}} | {{oc_p_iters_tol_1em10_N512}} | {{oc_p_factor_1em10_N512}}x |

The same anchoring check on the Poisson side — this cell re-times
`jax.scipy.sparse.linalg.cg(op, F, tol=CG_TOL)`, the archived baseline function — gives
agreement of {{oc_p_archcheck_N32}}% at `N = 32`, {{oc_p_archcheck_N64}}% at `N = 64`,
{{oc_p_archcheck_N128}}% at `N = 128`, {{oc_p_archcheck_N256}}% at `N = 256` and
{{oc_p_archcheck_N512}}% at `N = 512`.

Applying the multiplier to the archived Poisson ladder in `{{oc_p_old_json}}`:

| N | old FOM (ms) | old solve-only speedup | old end-to-end | m (time) | m (iterations) | corrected solve-only | corrected end-to-end |
|---|---|---|---|---|---|---|---|
| 32 | {{oc_p_old_fom_N32}} | {{oc_p_old_speed_N32}}x | {{oc_p_old_e2e_N32}}x | {{oc_p_mult_1em10_N32}} | {{oc_p_multit_1em10_N32}} | {{oc_p_new_speed_1em10_N32}}x | {{oc_p_new_e2e_1em10_N32}}x |
| 64 | {{oc_p_old_fom_N64}} | {{oc_p_old_speed_N64}}x | {{oc_p_old_e2e_N64}}x | {{oc_p_mult_1em10_N64}} | {{oc_p_multit_1em10_N64}} | {{oc_p_new_speed_1em10_N64}}x | {{oc_p_new_e2e_1em10_N64}}x |
| 128 | {{oc_p_old_fom_N128}} | {{oc_p_old_speed_N128}}x | {{oc_p_old_e2e_N128}}x | {{oc_p_mult_1em10_N128}} | {{oc_p_multit_1em10_N128}} | {{oc_p_new_speed_1em10_N128}}x | {{oc_p_new_e2e_1em10_N128}}x |
| 256 | {{oc_p_old_fom_N256}} | {{oc_p_old_speed_N256}}x | {{oc_p_old_e2e_N256}}x | {{oc_p_mult_1em10_N256}} | {{oc_p_multit_1em10_N256}} | {{oc_p_new_speed_1em10_N256}}x | {{oc_p_new_e2e_1em10_N256}}x |
| 512 | {{oc_p_old_fom_N512}} | {{oc_p_old_speed_N512}}x | {{oc_p_old_e2e_N512}}x | {{oc_p_mult_1em10_N512}} | {{oc_p_multit_1em10_N512}} | {{oc_p_new_speed_1em10_N512}}x | {{oc_p_new_e2e_1em10_N512}}x |

### What to do with this

1. Restate every Burgers and Poisson speedup against a **tolerance-based** FOM, with the
   tolerance named. `tau = 1e-10` is the like-for-like choice against what the old baselines
   actually reached; `1e-8` is the more defensible engineering choice and makes the
   correction larger still.
2. Do **not** apply a single scalar — the multiplier depends on `N`, and it changes the slope
   of the N-ladders, not just their level.
2b. Prefer the **iteration** multiplier when correcting a number measured on other hardware;
   the time multiplier assumes the fixed-vs-tolerance work ratio transfers between GPUs.
3. The ROM-side numbers are unaffected. Nothing about the ROM's cost or accuracy changes;
   only the denominator does.
4. This does not, by itself, overturn the qualitative Burgers conclusion that the latent solve
   is mesh-independent while the FOM is not — that is a statement about *scaling*, and both
   baselines scale similarly. It does change every *absolute* speedup that has been quoted.

## Figures

| file | what it shows |
|---|---|
| `wsfom_poisson_total_vs_tau` | **the headline**: total hybrid time vs the ROM's own stopping tolerance, one line per `N`, with the flat pure-FOM baseline dashed in the same colour, one panel per `tau_FOM` |
| `wsfom_poisson_crossover` | where the hybrid starts to win: cost vs mesh for the pure FOM, the best hybrid, the ROM stage alone and the exact direct solve; plus the speedup vs `N` against the break-even line |
| `wsfom_poisson_iters` | CG iterations saved vs the ROM tolerance, and the same saving plotted against the ROM's A-norm error ratio — the mechanism |
| `wsfom_burgers_per_step` | Newton and inner BiCGStab iterations **per time step** for all three arms, beside the ROM's own per-step error |
| `wsfom_burgers_cost_vs_N` | Burgers cost vs mesh for all three arms, and total Newton iterations per trajectory |

PNG + PDF in `figs/`, copied to `/home/tahmid/Dev/pod-ae-nmrom/Plots/`. All cross-`N`
wall-clock series are drawn only from the consolidated runs.

<!-- RESULTS -->

---

## Provenance

### Inputs (git-ignored because of their size; sha256 recorded here)

| file | sha256 | origin |
|---|---|---|
| `in/autodec_K8_N64_hbc_stages.pkl` | `45b5ff291216981a…` | `poisson2d-rom-objective/runs/hbc_K8/` — the hard-BC `K=8`, `N=64` FiLM auto-decoder the reference cell's headline used |
| `in/blat_ad_N64_K8.pkl` | `aa07cd4a1471c59a…` | `exp/2026-08-16-burgers2d-rom-latent-stepping` `runs/ad_n64_k8/` — the `K=8`, `N=64` Burgers auto-decoder + POD basis |

### Reference harness sources imported unchanged

| file | sha256 |
|---|---|
| `poisson2d-rom-objective/pro_common.py` | `044d1c3aaf4727bc…` |
| `poisson2d-rom-objective/followup/fu_eq.py` | `92717e93c8c04c93…` |
| `burgers2d-rom-latent-stepping/blat_common.py` | `4fecfe2f87a25327…` |
| `burgers2d-rom-latent-stepping/followup/fu_common.py` | `35c5693d05cdad62…` |

plus `ms_parametric.py` / `ms_autodecoder.py` (from `exp/2026-08-14-multistage-precision`) and
`burgers2d_film.py` (from `exp/2026-08-14-burgers2d-coord-rom`), staged into the job's `deps/`
by `cluster/make_cell.sh` and checksummed into `MANIFEST.sha256` before every `scp`.

Every result row additionally carries `commit`, a **sha256 of every `wsf_*.py`** (so a dirty
tree cannot publish two different codes under one commit hash), `gpu`, `gpu_kind`,
`jax_backend`, `slurm_job_id`, `seed`, and `run_role`.

<!-- PROVENANCE -->
