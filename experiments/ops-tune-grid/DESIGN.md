# ops-tune-grid — data parity and per-family tuning for the FNO, U-Net and Transolver baselines on Burgers 2D at 256²

Pre-registration. Written before the first GPU job; every later change is an amendment in §A
with its date and reason. Branch `exp/2026-09-22-ops-tune-grid`, forked from
`exp/2026-09-22-ops-deeponet-b2d` at `306c939d`. Cluster namespace
`/cluster/tufts/paralab/tawal01/opstune_grid_20260922/`.

## 1. The question

The user's concern, verbatim: *"I think we should try and properly tune them so that our
comparison doesn't get questioned by reviewers."* Two things a reviewer will name, in the
order that matters:

- **Q1 — data parity.** The FNO, U-Net and Transolver arms in the paper trained on **128
  cases**. The NM-ROM head they are compared against trained on **4608 trajectories**. That
  is a 36× gap in trajectories and it has never been controlled. This lane establishes the
  true comparable accounting on both sides (§3), regenerates operator training data from the
  **same generator, same family, same seed sequence** at 128 / 512 / 2048 / 4608 cases, and
  reports operator error against training-set size.
- **Q2 — tuning.** No family was ever tuned. Learning rate, weight decay, batch size,
  scheduler and patience were inherited from the FNO lane for all four families; `refine`
  (one lower-learning-rate retrain of the validation-selected capacity) is the only
  family-level tuning any of them received, and no family's own published recipe was used.
  This lane runs a per-family hyperparameter grid (§5) and reports what each family reaches
  when it is actually tuned.

A third thing this lane must settle because it is entangled with both: **the equal-wall
budget rule gave the float32 U-Net and Transolver 2.4–3.4× more epochs than the float64 FNO
at the same 3000 s** (2.4–2.8× comparing the `*-refine` arms; 2.4–3.4× over all arms) (`no-second` report, "Equal wall in float32 buys more epochs than
float64"). §6 pre-registers how that is handled and what is reported.

**What this lane does not do.** It states no speed number (§8). It opens no final or held-out
cohort (§4.3). It retires no hypothesis: one PDE, one mesh, one seed.

## 2. Where it stands before this lane

On the same-allocation panel at 256² (job `4181372`, `ops-timing-panel`, 6 development
cases), worst / median **evolved** error:

| arm | worst evolved % | median evolved % |
|---|---:|---:|
| NM-ROM accurate `q256_M1088_eqtop_g0p001` | 0.5129 | 0.1789 |
| NM-ROM fast `q0_M64_eqcert_g1em06_fastL4` | 1.8891 | 1.0100 |
| Transolver `tsol-refine` (validation-selected) | 4.4593 | 2.3195 |
| U-Net `unet-refine` (validation-selected) | 4.5529 | 1.9662 |
| FNO `fno-large` (validation-selected) | 7.4164 | 2.5256 |

Against the **fine reference**, the 256² mesh's own discretisation error is **4.03 %**
(`reports/2026-09-19-paper-results.md`). Every operator arm above exceeds it. That is the
paper's key qualification. **If tuning or data parity pushes an operator below 4.03 %, that
is an important result and this lane reports it as such**, including if it costs the paper a
claim.

**One measurement caveat, stated now so it cannot be quietly dropped.** The panel's operator
percentages are scored against a **same-job converged 256-grid solve** (`fft_tight`), while
the 4.03 % is the **256-grid solution's own error against a finer reference**. They are on
different references and are not directly subtractable. This lane's own accuracy numbers are
on **validation-32 against the pinned 4096-interval reference restricted to 256** — which is
the same kind of quantity as the 4.03 %. **The cohorts still differ**: 4.03 % is measured on
the panel's six development cases and this lane reports on validation-32 and diagnosis-8. So
`gen01` measures the bar again, on validation-32, against the same pinned reference the
operators are scored against (§5.1 step 3), and the report compares against **that** number.
The published 4.03 % is quoted beside it for continuity, labelled with its own cohort.

## 3. The data-parity accounting — established before any job

Both sides draw from the **same 5-parameter Gaussian-bump family** through the **same file**
`experiments/mr-burgers2d/engines.py` (`params_draw`), verified byte-identical between the
two lanes' worktrees, and both use the output times $\{0, 0.05, 0.10, 0.15, 0.20, 0.25\}$.
What differs is the count, what a "state" means on each side, and **the fidelity of the
solver that produced the training targets**:

| | trajectories | states the model is fitted on | training-target solver | seconds per trajectory (A100) |
|---|---:|---:|---|---:|
| NM-ROM bank $g$ (job `2835788`) † | 576 | 16 384 states | **256 intervals, $\Delta t = 0.005$, direct** | 0.19 † |
| NM-ROM head $h_\theta$ (job `2837431`) † | **4608** | **131 072 states** (of 235 008 available at 51 states/traj) | **256 intervals, $\Delta t = 0.005$, direct** | 0.19 † |
| Operators, as published | **128** | 128 inputs → **640** supervised evolved states (5 per case) | **4096 intervals, $\Delta t = 1.5625\times10^{-4}$, restricted to 256** | **127.2** (median of 16 archived per-case times; mean 129.9, range 113.4–157.6) |
| Operators, this lane (target) | **4608** | 4608 inputs → **23 040** supervised evolved states | **1024 intervals, $\Delta t = 3.125\times10^{-4}$, restricted to 256** (§3.2) | **4.8** |

† **The two NM-ROM rows are quoted from those jobs' own records and are not independently
re-derivable from any artifact reachable in this repository** — the design audit looked and
could not find them. The report labels them as quoted rather than verified, and gives the job
ids so a reader can go to the source. Every other row here is checkable.

Three things follow, and the report states all three:

1. **The 128-vs-4608 gap is a data-*generation-cost* artefact, not a design choice.** The
   operators' training targets were held to a far stricter reference than the NM-ROM's own
   training trajectories: 127.2 s versus 0.19 s per trajectory, ≈670×. 128 cases is what
   about 4.5 GPU-hours buys at that fidelity.
2. **Trajectory parity is reachable; state parity is not.** At 4608 cases an operator sees
   4608 × 5 = 23 040 supervised evolved states against the head's 131 072 fitted states —
   still 5.7× fewer — because the operator's contract fixes **six** output times per
   trajectory while the head is fitted per state at **51** times. That is a property of the
   operator contract, not something this lane can or should change, and the report says so
   rather than claiming full parity.
3. **The NM-ROM's own training targets are coarser than anything this lane gives the
   operators — and `gen01` measures that rather than inferring it.** The head trained on
   256-interval, $\Delta t = 0.005$ solutions, a setting the calibration never ran, so the
   first version of this design inferred its error from the $\Delta t = 1.25\times10^{-3}$
   row. `gen01` now solves the 32 validation cases at exactly 256 / $\Delta t = 0.005$
   (§5.1 step 3) and reports the number. Like for like at present — deviation against
   deviation — it is 0.915 % for 256 / $1.25\times10^{-3}$ against 0.187 % for the
   operators' new 1024 / $3.125\times10^{-4}$ data, a factor of 4.9, and
   $\Delta t = 0.005$ is four times coarser again.

   **This is context, not a defence of §3.2**, and the report says so. The NM-ROM is
   *graded* against the fine reference and reaches 0.51 %; the coarseness of the bank it was
   fitted on does not license label noise on the operator side. The argument that does that
   work is the structural one in §3.2: the operators' evaluation targets are unchanged, so
   any fidelity bias is counted *against* the operator inside the reported number.

### 3.1 Seeds, disjointness and what is never regenerated

- New cases use `data.case_seed('train', i)` for `i` in `[0, 4608)` — the frozen protocol's
  own `SeedSequence([dataset_seed, pde_seed_code, split_code, index])`. Indices `0..127` are
  the **same physical cases** as the published training set; `128..4607` are new.
- **Validation-32 and the matched diagnosis-8 cohort are never regenerated.** Evaluation runs
  against the pinned cache
  `/cluster/tufts/paralab/tawal01/no_burgers_20260914/pilot-data01`, whose validation index
  SHA256 `468b9e70…` is asserted equal to the FNO job's, exactly as in the parent lanes.
- `dataset.load_pair` already refuses any train/validation overlap by case id, by generation
  seed **and** by physical input content; that check runs unchanged over the enlarged
  training set. `prepare_diagnosis_cohort.py`'s disjointness assertion likewise runs
  unchanged.
- Final / held-out splits have no split code and no generation command. `PROVENANCE.json`
  records `final_cohort_opened: false`.
- The `ops-timing-panel`'s six **development** cases are drawn by that lane's own JAX phase,
  not by `data.case_seed`, so they lie outside this seed space entirely and cannot collide.
  Nothing in this lane trains on, selects on, or opens them. The audit could not verify their
  seeds directly, and the report says so rather than implying a check that was not made.

### 3.2 Why the new data is generated at 1024 / $3.125\times10^{-4}$, and how that is controlled

Generating 4608 cases at the pinned 4096 / $1.5625\times10^{-4}$ reference costs
4608 × 127.2 s (the median of the 16 archived per-case generation times) = **163
A100-hours**. It does not fit this lane, this campaign, or the 09-25 deadline.

The **live** calibration — `refinement02`, index SHA256 `894daa5c…`, mirrored locally in
`checks/refinement02-reference-audit.json`, run under `protocol-refined.json` whose anchor is
the pinned 4096 / $1.5625\times10^{-4}$ — measured each candidate on 8 independent
development cases at output 256:

| solver setting | worst deviation from the pinned anchor | median | gate margin | work proxy | measured s/case |
|---|---:|---:|---:|---:|---:|
| 256, $1.25\times10^{-3}$ | 9.149e-3 | 6.686e-3 | 1.006e-2 | 1.31e7 | 1.20 |
| 512, $6.25\times10^{-4}$ | 4.344e-3 | 3.154e-3 | 5.251e-3 | 1.05e8 | 1.74 |
| **1024, $3.125\times10^{-4}$** | **1.874e-3** | **1.352e-3** | **2.781e-3** | 8.39e8 | **4.80** |
| 4096, $1.5625\times10^{-4}$ (the anchor) | 0 | 0 | 9.588e-4 | 2.68e10 | 68.7 |

The "gate margin" column is `worst_empirical_margin`: the maximum over the eight cases of
(that case's deviation **plus** that case's own anchor refinement margin). It is the maximum
of a per-case sum, not the sum of two column maxima, and the report quotes it as such.

**Both columns are observed maxima over eight development cases, not bounds over 4608
cases**, and neither is a certified error — the protocol itself calls space/time refinement
differences "empirical development evidence, not rigorous continuum error bounds". They size
the decision; they do not settle it.

At that size, operator errors in this comparison are **2–7 %**, so the label noise is roughly
10–25× below the quantity being measured.

**The strongest argument here is the one about where the bias lands, and it is structural
rather than empirical: the evaluation targets are unchanged.** Validation-32 and diagnosis-8
are scored against the pinned 4096 / $1.5625\times10^{-4}$ reference, exactly as every
published arm was. So if a model trained on 1024-fidelity targets learns the 1024 solver's
operator rather than the 4096 solver's, that discrepancy appears **inside** the reported
validation error, as a floor of order 0.19–0.28 %. It is not hidden outside the measurement;
it is counted against the model, at 10–25× below the signal. Nothing about the headline
comparison can be flattered by this choice.

Three pre-registered checks, and the eight-case calibration figure is explicitly **not** the
acceptance test:

- **G1, measured in `gen01` on the real training cases.** Every one of the 128 published
  training cases is re-solved at 1024 / $3.125\times10^{-4}$ and its fixed-initial error
  against its own pinned target recorded. The report gives the worst, median and mean of
  those 128 numbers. **Threshold: if the worst exceeds 0.5 %** — about 2.7× the worst already
  measured on development cases, and still a quarter of the smallest operator error here —
  the cheap-fidelity rungs are reported as confounded and labelled. The distribution is
  reported whatever it shows; the threshold only decides the label.
- **G2, a training control in `ladder01`.** `unet-pinned128` retrains the published U-Net
  configuration on the **same 128 physical cases** at pinned fidelity, under the same budget
  and the same step-matched schedule as `unet-n00128`. Their difference is the
  target-fidelity effect at fixed data size.
  **G2 bounds that effect; it cannot confirm its absence.** It is a single-seed A/B whose
  expected size (≲0.28 pp, much of it common-mode) is at or below this project's own measured
  one-seed noise on this exact configuration — `no-second`'s `ctrl-medium-seed2` moved
  `unet-medium`'s validation mean by +0.0435 pp, its median by −0.153 pp and its worst by
  −0.321 pp. That band is pre-registered here as G2's resolution limit, and a null G2 will be
  reported as "not resolvable at one seed", never as "fidelity does not matter". G2 also runs
  at the rung where the effect matters least; running it at the top rung would need
  pinned-fidelity data this lane cannot afford, and that limitation is reported too.
- **The reproduction gate**, before either. Case `burgers-train-00000` is regenerated at the
  **pinned** setting and its **arrays asserted bitwise identical** to the cached case. That
  proves this lane's generator reproduces the published data exactly, which is a stronger
  statement than any hash of a calibration JSON. A failure voids `gen01`. File-byte identity
  is **recorded but not asserted**: `np.savez` is deterministic, so the bytes normally match,
  but they also depend on the numpy version that wrote them, and a numpy upgrade must not
  void a numerical gate for a non-numerical reason.

## 4. Protocol

### 4.1 What is held fixed from the published lanes

The contract, the metric, the split, the reference, the boundary mask, the time convention
and the evaluation cohorts are **unchanged and uncopied** — the same `model.py`,
`dataset.py`, `train.py`, `evaluate_cohort.py`, `prepare_diagnosis_cohort.py` and
`spectral_conv_f64.py` as `no-second` and `ops-deeponet-b2d`, byte-identical except where
§5.4 lists a change. Seed 20260914 throughout, one seed; this lane claims no seed robustness.

$$E(\text{case}) = \max_{k=0,\dots,5} \frac{\lVert \hat u(t_k) - u^{\mathrm{ref}}(t_k)\rVert_{2,\mathrm{interior}}}{\lVert u^{\mathrm{ref}}(t_0)\rVert_{2,\mathrm{interior}}}.$$

### 4.2 Pre-registered selection metric — fixed before any job

**The selected arm of a family is `argmin`, over that family's arms, of the mean over the 32
validation cases of the per-case maximum over the six output times of the fixed-initial
relative error.** This is the published lanes' own rule, applied unchanged so tuned and
published arms are graded identically.

`no-second` recorded the pitfall it creates — the mean-selected arm had a *worse* worst case
than an unselected sibling in both families — so, also pre-registered:

- the report gives **every** arm's mean, median, p95, worst and threshold counts, never only
  the selected one;
- the report names, per family, **both** the selected arm **and** the arm with the lowest
  validation-32 worst case (the "best-worst-case arm"), side by side, always, whether or not
  they differ;
- the selection rule is not changed after seeing numbers and no alternative rule is
  introduced post hoc. If a different rule would have chosen differently, that is reported as
  an observation about the rule, not used as the verdict.

### 4.3 Cohorts

- **Validation-32** — selection and the headline accuracy table. This is the *tuning*
  cohort; it is used for selection by construction and the report says so wherever a tuned
  number appears.
- **Matched diagnosis-8** — rebuilt in-job from the same anchors the Burgers ROM lane used
  (job `3702709`), index SHA256 asserted equal to the FNO job's. **Never used for
  selection.** Reported as the matched-cohort column.
- **Final / held-out cohorts are never opened.** `dataset.py` refuses those splits by
  construction.

**The honesty rule this lane operates under: no tuning decision is ever taken on a final or
held-out cohort, and every arm reported here was selected on validation-32.** Because the
grid tunes on validation-32, validation-32 numbers for tuned arms are **optimistic** and the
report labels them so; the matched diagnosis-8 column, which nothing was selected on, is the
one a reviewer should weight.

## 5. The jobs

**Budget: ≤ 1 running GPU job, ≤ 6 total**, `squeue -u tawal01` before and after every submit,
one directory per job, `gpu` partition only, output under paralab only.

### 5.1 `gen01` — data (≈ 9 h wall, 13 h limit, A100)

In order, writing its report incrementally so a truncated run is still usable:

1. **Reproduction gate** — regenerate `burgers-train-00000` at the pinned 4096 /
   $1.5625\times10^{-4}$ setting; assert its arrays are identical to the cached case. (~137 s)
2. **Bulk, indices `0..4607`** at 1024 / $3.125\times10^{-4}$. Indices `0..127` are the same
   physical cases the published bank holds, so each one also yields **control G1**: its
   fixed-initial error against its own pinned target. Those 128 cases are then the
   cheap-fidelity 128-case training set control G2 uses. (~8.7 h)
3. **The discretisation bar on this lane's own cohort** — solve each of the 32 validation
   cases *on the 256 grid*, at the anchor's time step and at the panel's, and score by the
   identical metric against the same pinned reference. This is the 4.03 % qualification of §2
   measured on validation-32 instead of on the panel's six development cases. (~4 min)
4. Write prefix indices `index-00128/00512/02048/04608.json`. **Each is written the moment
   its case count is reached**, not at the end, so a truncated run still leaves usable
   indices; one is also always written at the count actually reached.

**`gen01` is not only a data job — it produces the cache both later jobs stage from.**
`worker_gen.py` writes one `DATA.sha256` over the pinned `train`/`validation`/`refinement`
directories and the new `trainbig`, then moves the whole verified tree to
`<namespace>/cache`, refusing to overwrite an existing one. `grid01` and `ladder01` set
`data_source` to that cache, so their sbatch preamble verifies the pinned data and the new
bank with the same one command. A truncated `gen01` therefore also determines what the later
jobs can read, which is why `make_ladder_spec.py` builds `ladder01`'s rungs from the indices
`gen01` actually reported rather than from the four this document names.

**The budget arithmetic, because it is not the obvious one.** The frozen `data.solve` runs a
**2 s GPU warm-up before every solve**, and the calibration's 4.80 s per case excludes it
(that timer starts after the warm-up). The real cost is therefore ~6.8 s per case and 4608
cases need ~8.7 h, not the ~6.1 h a naive 4608 × 4.8 s gives. The warm-up is inside the
generator this lane reuses byte-identically, so those 2.6 h are a price paid for
reproducibility, not an inefficiency to remove — removing it would mean forking the frozen
numerical path, which is the one thing §3.2's gate rests on not doing.

### 5.2 `grid01` — the tuning grid at the published 128 cases (≈ 16 h, A100)

Run on the **pinned** 128-case training set so the tuning effect is measured with data held
at exactly the published value. Screen budget **3000 s per arm — the published rule,
unchanged** — so every arm in this grid is directly comparable to the published arms. §6
handles the budget question separately rather than by moving this bar.

### 5.3 `ladder01` — error versus training-set size (≈ 15 h, 16 h limit, A100)

Each family's **published** configuration — `fno-large`, `unet-medium`, `tsol-small` —
unchanged, trained at 128 / 512 / 2048 / 4608 cases at the **published 3000 s** per arm.

**This job runs before `grid01`, and deliberately does not use tuned configurations.** A
ladder must vary one thing. Putting `grid01`'s winners here would confound tuning with data
and would also make the data result, which is the reviewer-facing one, wait on the grid.
Holding the published per-arm budget additionally makes the $n=128$ rung comparable to the
published arms, not only to its own ladder.

Every rung draws from the same cheap-fidelity bank, so target fidelity is constant along a
ladder; **control G2** (§3.2) is the same configuration and budget on the same 128 physical
cases at the pinned fidelity, and `unet-n4608-long` triples the wall at the top rung to test
whether the budget binds there.

**The schedule is held constant in gradient steps, not in epochs — without this the ladder
would not measure what it claims to.** Both of `train.py`'s patiences count *epochs*, and an
epoch is $\lceil N/8 \rceil$ gradient steps: 16 at $N=128$ but 576 at $N=4608$. Left at the
published 20 / 250, a 4608-case arm would reach roughly 55–70 epochs in 3000 s, fire the
plateau rule at most once, never be able to early-stop, and **finish at its initial learning
rate** — while the 128-case arm annealed all the way to the 1e-5 floor. A rung would then
differ from its neighbours in the learning-rate schedule as much as in the data, with a net
bias of unknown sign. `make_ladder_spec.py` therefore sets each rung's `patience` and
`plateau_patience` so that patience × steps-per-epoch equals the published 128-case value in
steps:

| rung | steps/epoch | `plateau_patience` | `patience` |
|---:|---:|---:|---:|
| 128 | 16 | 20 | 250 |
| 512 | 64 | 5 | 63 |
| 2048 | 256 | 2 | 16 |
| 4608 | 576 | 1 | 7 |

**Equal wall is only *approximately* equal steps, and the approximation favours the large-data
rungs.** Each epoch carries fixed costs that do not scale with $N$ — a 32-case validation
pass, a rewrite of `history.json`, and a `torch.save` of `last.pt` (plus `best.pt` on every
improvement). Those amortise over 16 steps at the bottom rung and 576 at the top, so the top
rung buys up to roughly a quarter more gradient steps at the same wall. `train.py` therefore
records `steps_per_epoch` and `optimisation_steps` per arm, and **the ladder is tabulated
against steps as well as against wall**, so a reader can see how much of any improvement is
data and how much is extra optimisation.

Jobs 4–6 are held in reserve for failures and for anything §7 says must be re-run. If the
budget runs short, `grid01` is the job that gets cut, not `ladder01`.

### 5.4 The grid, fixed before any job

Capacities and knobs per family. Published arms are named for reference and are **not**
re-run; their numbers are carried from the published audits.

**Each family gets the same four knob classes — capacity, normalisation, learning-rate
schedule, and learning rate — five arms each.** Published arms are named for reference and are
**not** re-run; their numbers are carried from the published audits.

| family | capacity | normalisation | schedule | learning rate | other |
|---|---|---|---|---|---|
| **FNO** (f64; published width 64 / modes 32 / layers 4, lr 1e-3) | `fno-modes48` (modes 32→48), `fno-width96` (width 64→96) | `fno-norm` (`norm='group_norm'`) | `fno-cosine` | `fno-lr3e-3` | `fno-epochmatch` — **budget control**, §6 |
| **U-Net** (f32; published base 32, GroupNorm(8), lr 1e-3) | `unet-base64` (base 32→64) | `unet-groups16` (GroupNorm 8→16) | `unet-cosine` | `unet-lr3e-3` | `unet-wd1e-3` (weight decay 1e-4→1e-3) |
| **Transolver** (f32; published dim 128, 8 layers, 8 heads, 64 slices, patch 4) | `tsol-layers6-d192` (dim 192, layers 8→6) | `tsol-slices32`, `tsol-slices128` (slices 64→32, →128) | `tsol-cosine` | `tsol-lr3e-3` | — |

Three choices in that table are worth defending, because the first version of this section
got them wrong:

- **Learning rate is in every family.** It was in one. It is the only knob the published lanes
  ever moved, and it moved results in both directions — the lower-lr `*-refine` retrain helped
  the U-Net (1.3523 vs 1.4341) and the Transolver (1.9493 vs 2.0064) and *hurt* the FNO
  (2.3501 vs 2.2811). Since 3e-4 is exactly what the published `*-refine` arms already did,
  the untried direction is **upward**, so the arms are at 3e-3.
- **`fno-layers6` and `tsol-patch2` were dropped.** At a fixed 3000 s they cost 1.41× and
  **2.12×** per step (measured, `checks/local-smoke-tune.json`), and every published arm ended
  on its wall still improving — so they are the two arms whose results would be least readable
  as capacity rather than as budget. Their configs stay on disk; they are simply not in the
  spec.
- **A cosine arm's horizon is its own epoch count, not the 4000-epoch cap.** Left at 4000 a
  cosine arm that only reaches ~700 epochs would decay by a few percent and be a
  constant-learning-rate run wearing a schedule's name. Each cosine config's `epochs` is the
  count that capacity actually reached in the published 3000 s (FNO 692, U-Net 1963,
  Transolver 1628), so the decay completes inside the budget. `tsol-cosine` keeps the
  published 10-epoch warm-up, so it varies one thing.

**Every code change this lane makes**, in full — the first version of this section said
"and nothing else" while listing four of them, which is the under-declaration the lab-log
rule exists to prevent:

| file | change | default behaviour |
|---|---|---|
| `model.py` | `norm=config.get('norm')` passed to `neuralop.FNO` | omitted → `None`, the parent value |
| `train.py` | `config['schedule'] in ('plateau', 'cosine')` | `'plateau'`, the parent scheduler |
| `train.py` | `config['plateau_patience']` | `20`, the parent value (§5.3 needs it step-matched) |
| `train.py` | records `training_cases`, `steps_per_epoch`, `optimisation_steps`, `schedule`, `plateau_patience` | additive |
| `worker_second.py` | per-arm `train_index`; `smoke_families`; a `smoke_tune.py` pre-flight over every config the job will train; `disjointness_indices` | omitted → the parent behaviour |
| `prepare_diagnosis_cohort.py` | `--also-disjoint-from`, checked but **not** recorded, so the cohort index and its SHA256 are unchanged | omitted → the parent behaviour |
| `audit.py` | admits `fno`; **records** `epochs`, `patience`, `weight_decay`, `learning_rate`, `schedule` instead of asserting them; admits this lane's bank indices as training indices | see below |
| `cluster/stage.py` | lane path, namespace, `ROOTCODE` staging of the frozen generator at its repository paths | — |
| new | `gen_bank.py`, `worker_gen.py`, `smoke_tune.py`, `make_ladder_spec.py`, `reports/sources.py`, the specs and configs | — |

**The `audit.py` demotion is the one that costs something, and it is bounded.** `epochs`,
`patience`, `weight_decay` and `learning_rate` are the knobs under test in `grid01`, so they
cannot also be asserted equal to the parent protocol there. But `patience` and `epochs` govern
early stopping and therefore *which checkpoint is selected*, so they are not free. What stays
asserted for every arm: `batch_size`, `seed`, the family whitelist, and — the assertion that
actually protects the comparison — **the validation index SHA256 equals the FNO job's**. The
`ladder01` arms vary none of these knobs by design, and the report tabulates the realised
value of every demoted field per arm.

## 6. The budget rule, and what this lane changes about it

The published rule is **equal wall, not equal epochs**. At 3000 s the float32 families ran
1628–2327 epochs and the float64 FNO ran 692 — a 2.4–3.4× epoch advantage to the families
this project introduced. **This lane does not change the screen rule** (§5.2): moving it
would confound tuning with budget. It adds one arm, `fno-epochmatch`, with an explicit
**epoch target of 1963** — the count `unet-medium` reached in its published 3000 s — under an
**8700 s wall cap**, and **reports both** the equal-wall number and the equal-epoch number
side by side. **`fno-epochmatch` is a budget control, not a tuning arm, and is excluded from
the §4.2 selection `argmin`** — it has 2.9× the wall of its five siblings and would otherwise
win the FNO's grid, letting a budget effect be reported as a tuning effect, which is the exact
confound §5.2 refuses to introduce. `specs/grid01.json` records it in `selection_excluded`.
The match is to a *published* U-Net's epoch count, and only the FNO gets one although the
Transolver also outran it; both asymmetries are stated in the report rather than repaired,
because repairing them costs arms the grid needs. 8700 s is sized from `fno-large`'s published throughput (692 epochs in 3003 s →
4.34 s/epoch → 8506 s for 1963 epochs) with ~2 % of slack; wall time does not *guarantee* a
matched epoch count, so the arm stops at whichever of the two bounds comes first and **the
report states the epoch count actually achieved** and which bound stopped it. If the wall cap
bound before 1963 epochs, the arm is reported as still budget-limited and the FNO's epoch
deficit as only partly closed.

Every arm's `result.json` records `stop_reason`, the epoch curve and the training-loss
history. The report carries, **generated and never asserted**, per arm: epochs run, best
epoch, stop reason, whether the wall budget bound, and `still_improving` (best epoch within
the last 5 % of completed epochs — `no-second`'s own definition, reused unchanged), together
with the train loss at the best epoch, the final train loss, and the number of epochs spent
at the 1e-5 learning-rate floor. Where the budget bound, the number is a **lower bound** and
the report says so beside it.

## 7. Pre-registered criteria

- **T1 (tuning moved it).** Per family, the `grid01`-selected arm's validation-32 mean,
  median and worst against the published selected arm's, as ratios. No threshold: this is
  reported, not passed or failed. The claim the paper may then make is "tuned", not "optimal".

  **T1 is biased toward "tuning helped", by construction, and the report says so beside the
  ratio.** Both arms are selected on the *same* 32 cases, but this lane's winner is the argmin
  of a 5–6-wide pool while the published winner was the argmin of a 3–4-wide pool, so the
  tuned side carries strictly more selection bias. The diagnosis-8 column, which nothing was
  selected on, is the one that does not; it has only 8 cases, and `no-second`'s own report
  calls a six-case cohort "a thin basis for a tail statement either way".

  **A second confound, also pre-registered:** every published arm ended on its wall budget
  while still improving, so a more expensive arm at a fixed 3000 s is scored with
  proportionally fewer epochs. **An arm that ends `stopped_by_wall_budget` with
  `still_improving` true is reported as a lower bound and is NOT eligible to be called "worse
  than" its baseline.** Measured step costs relative to each family's baseline —
  `fno-width96` 1.30×, `fno-norm` 1.13×, `fno-modes48` 1.02×, `unet-base64` 1.24× — are in
  `checks/local-smoke-tune.json` and in the report, so a reader can see which arms this
  applies to before reading their numbers.
- **T2 (data moved it).** Per family, validation-32 and diagnosis-8 error at 128 / 512 /
  2048 / 4608, **tabulated against optimisation steps as well as against wall**, because
  equal wall is not exactly equal steps (§5.3). **Declared in advance:** if error falls
  monotonically across the ladder and is still falling at 4608, the report states that the
  operator baselines are **data-limited at every size this lane could reach** and that the
  published 128-case numbers are an upper bound on what these families do at this budget, not
  a property of the architectures. Because `ladder01` trains the **published** configurations,
  T2's claim is about those configurations at this budget — not about the best these families
  can do, which is `grid01`'s separate question.
- **T3 (the discretisation bar).** The statistic is the **worst case-maximum fixed-initial
  error over the cohort**, and the cohorts are validation-32 and diagnosis-8. The bar is the
  256-grid's own discretisation error **measured by `gen01` on validation-32** (§5.1 step 3),
  with the paper's published 4.03 % quoted beside it and labelled with its own cohort.

  **Stated before any job, because otherwise this criterion reads as a discovery:** on
  validation-32 the published `unet-medium` is *already* below 4.03 % on mean (1.4341 %),
  median (1.3169 %) **and worst (3.9622 %)**, and on diagnosis-8 every published U-Net and
  Transolver arm is already below it (1.4712–2.3176 % worst). So "an operator below the
  discretisation error" is not new — **what is new is that the paper's qualification is stated
  against the panel's cohort and reference, where it holds, while on this lane's cohorts it
  does not.** The report says exactly that, names which arms, and states the consequence for
  the paper's wording. That is the finding this lane most wants to surface, not avoid.
- **T4 (what the baselines were given that our model was not).** The report tabulates, per
  arm: training cases, supervised states, training-target solver fidelity, wall budget,
  epochs, and whether a hyperparameter search selected it. Our own head's corresponding row
  (§3) sits beside them. **Wherever an operator received more, the report says so in the
  table and in prose**, because the paper should state that the baselines were given the
  stronger protocol.
- **T5 (a tuned baseline that beats us).** §2 says the panel's percentages and this lane's
  percentages are on different references and different cohorts and **are not subtractable**.
  T5 therefore does **not** compare this lane's numbers against the panel's NM-ROM 1.89 %; an
  earlier version of this criterion did, and it was wrong on its own document's terms. The
  measurable fact — the same `unet-refine` checkpoint scores 1.7110 % on diagnosis-8 and
  4.5529 % on the panel, 2.7× apart — shows how large that category error is.

  What T5 tests instead: **whether any arm in this lane beats the best published operator arm
  on this lane's own cohorts, under this lane's own reference** (diagnosis-8 worst 1.4712 %,
  `unet-small`; validation-32 worst 3.9622 %, `unet-medium`). If one does, it is reported and
  its checkpoint is handed to `ops-timing-panel` (§8), which is the only place a like-for-like
  comparison against the NM-ROM can be measured, in one allocation. **This lane states no
  operator-versus-NM-ROM verdict of any kind.**

**No hypothesis is retired by this lane.** One PDE, one mesh, one seed, one implementation
per family.

## 8. Timing — nothing from this lane is admissible

No arm in this lane is timed for any reported speed number, and no time from any job here is
divided by a time from any other job. Arms worth timing are handed to the
`ops-timing-panel` lane in its own harness format
(`experiments/ops-deeponet-b2d/reports/timing-handoff.json` is the pattern: checkpoint path
and SHA256, the `families.py` / `model.py` the harness needs, and the accuracy values that
must travel with the timing row). The handoff is written; this lane makes no speed claim.

## 9. Gates — a job whose log fails any of these is void

1. `jax_backend=gpu` and, for training jobs, `torch_backend=cuda`; the run ends with
   `ALL-DONE`.
2. `sha256sum -c MANIFEST.sha256` passes in the preamble, and every data directory verifies
   against its own checksum manifest before any training.
3. The pinned validation index SHA256 equals the FNO job's; the diagnosis cohort's index
   SHA256 equals the FNO job's.
4. `gen01` only: the reproduction gate (§3.2) is bit-identical, every generated case passes
   the frozen solver's own residual, Newton-budget and Dirichlet assertions, and the
   generated `input` equals `engines.initial` on the requested nodes exactly.
5. Every arm in a spec is present, complete and cohort-scored; a dropped arm fails the audit
   rather than passing quietly.
6. Every reported error is recomputed from the saved prediction fields by `audit.py`, which
   imports neither torch nor jax. Batch-8 selection score and batch-1 reported score may
   differ by at most 1e-5 for a float32 network (the inherited tolerance).
7. The supplied initial state is returned bitwise and the boundary is exactly zero in every
   saved prediction.
8. Exactly one job per submit directory, confirmed in `squeue` before and after submission.
9. Every number in the report is generated from the audit JSONs by `reports/generate_report.py`.
   No number is hand-typed.

## 10. Stop rules

- **Reproduction gate fails** → `gen01` is void, nothing downstream runs, and the lane
  reports the failure rather than working around it.
- **G1 worst exceeds 1.0 %** → the cheap-fidelity ladder rungs are reported as confounded and
  the headline reverts to the tuning result; the ladder is still reported, labelled.
- **Any gate in §9 fails** → that job is void and is not partially reported.
- **Budget exhausted before the ladder completes** → the completed rungs are reported with
  the missing rungs named, never interpolated. This is only implementable because `gen01`
  writes each prefix index as it is reached and `make_ladder_spec.py` builds the ladder from
  the indices that exist; naming `index-04608.json` in a spec before knowing it was written
  would have made a short generation fail the **whole** ladder audit rather than cost it one
  rung.
- **Disk.** The bank is ~16 GB, the cache ~20 GB, and each training job's preamble copies the
  cache into its own attempt directory, so the lane's live footprint peaks near **60 GB** on a
  share this project's `CLAUDE.md` describes as nearly full — and a full share fails a job
  with an *empty log* that reads as a code bug. `worker_gen.py` refuses to start below 80 GB
  free and prints `disk_free_bytes=`. The namespace is deleted when the lane closes.
- **A tuned arm beats the NM-ROM** → reported, not re-run until it stops beating it.

## A. Amendments

*(append-only; date, reason, action)*

- **A1 (2026-09-22, before `gen01`).** Independent audit of this design commissioned before
  the first GPU job; disposition recorded here before submission.

- **A2 (2026-09-22, before `gen01`) — Codex design audit, two findings, both accepted.**
  `reports/codex-design-audit-2026-09-22.md` (`gpt-6-astra`, headless, `-s read-only`).
  **Its sandbox could not read a single file** — every read failed with
  `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`, the same failure
  `no-second` recorded on 2026-09-17 and `ops-deeponet-b2d` on 2026-09-22 — so it audited
  only the numbers quoted in the prompt and said so rather than inventing file findings.
  That limits it to two findings, and both are real:

  - **MAJOR — the §3.2 fidelity arithmetic was wrong as written.** The text read as though
    3.06e-3 were 1.87e-3 + 1.386e-3; that sum is 3.26e-3. 3.056e-3 is the calibration gate's
    own per-case maximum of (deviation + that case's anchor margin), which is smaller because
    the two column maxima fall on different cases. §3.2 now states which quantity is which,
    and — the auditor's substantive point, accepted — that **both are observed maxima over
    eight development cases, not a bound over 4608**, and that neither is a certified error.
    The acceptance test is now stated explicitly to be control **G1**, measured on the real
    training cases, with its pre-registered 1.0 % failure threshold, not the calibration
    figure.
  - **MINOR — the epoch-match arm needed an epoch target, not just a wall.** 8700 s at
    `fno-large`'s published throughput implies ≈2005 epochs, and wall time does not guarantee
    an epoch count on a different node. §5.4 and §6 now pre-register an explicit **1963-epoch
    target under an 8700 s cap**, whichever binds first, and require the report to state the
    epochs actually achieved and which bound stopped the arm.

  Everything else in the auditor's reply is an explicit *unresolved* — data-parity
  accounting, generation-path equivalence, disjointness, selection and honesty, feasibility,
  omissions — because it could not read the files. **This audit therefore does not discharge
  §A1**, and the independent subagent audit commissioned in parallel with the same brief is
  what covers those questions; its disposition is §A3.

- **A3 (2026-09-22, before `gen01`) — independent design audit, 4 blockers and 17 majors,
  every finding accepted; none rejected.** `reports/design-audit-2026-09-22.md`. Commissioned
  with the brief in `reports/design-audit-prompt.txt` after Codex's sandbox failed (§A2), and
  delivered in two passes; it built and ran **every** proposed arm at batch 8 on 257², and
  re-derived the seed space, the calibration gate and the published comparison numbers from
  their own files. Nothing it checked in §2 or §4.3 was wrong. What it found:

  - **B (blocker) — §3.2's whole calibration table was quoted from the superseded calibration
    that FAILED.** `artifacts/calibration01/` ran under the *original* protocol, anchor
    4096 / $3.125\times10^{-4}$, and its gate at output 256 reads
    `passing: false, selected: null` — that failure is precisely why `protocol-refined.json`
    exists. The live gate is `refinement02` (`894daa5c…`), anchor 4096 / $1.5625\times10^{-4}$,
    which is the reference this lane's data actually uses. Every figure had been transcribed
    *correctly* — from the wrong file. §3.2 is regenerated from the live gate: the 1024
    setting's worst deviation is **1.874e-3** (was quoted 1.87e-3), its gate margin **2.781e-3
    = 0.28 %** (was 3.056e-3 = 0.31 %), and the anchor's own margin **9.588e-4** (was quoted
    1.386e-3, which belongs to a different anchor). **The decision strengthens slightly.** The
    row previously labelled "4096, $3.125\times10^{-4}$ (the anchor)" was not the anchor at all.
  - **B — the ladder's learning-rate schedule and early stopping were epoch-indexed, so the
    top rung would never have annealed.** At 4608 cases an arm reaches ~55–70 epochs against
    ~2000 at 128, so the plateau rule would fire at most once, early stopping (patience 250)
    would be mathematically unreachable, and the top rung would finish at its initial learning
    rate while the bottom rung sat at the 1e-5 floor. T2 is this lane's headline and the
    ladder could not have answered it: the rungs would have differed in schedule as much as in
    data, with a net bias of unknown sign. **This is the finding that would have silently
    destroyed the result, and it could not have been repaired after the jobs ran.** Fixed:
    `train.py` takes `plateau_patience`, and `make_ladder_spec.py` scales both patiences per
    rung so they are constant in gradient steps (§5.3).
  - **B — `gen01` would have stopped ~700 cases short, and that would have voided `ladder01`
    whole.** `data.solve` spins a 2 s GPU warm-up **before every call**, which the
    calibration's 4.80 s excludes, so 4608 cases need ~8.7 h against a declared 7.5 h. Worse,
    prefix indices were only written at the end, so `index-04608.json` would never have
    appeared, four ladder arms would have named a missing file, and `audit.py`'s
    `set(arms) == expected_arms` would have failed the **entire** ladder audit rather than
    costing it one rung — making §10's graceful-degradation rule unimplementable. (Found
    independently while re-budgeting, and confirmed by the audit.) Fixed three ways: the
    budget is 36 000 s under a 13 h limit; `gen_bank.py` writes each prefix index the moment
    it is reached; and `make_ladder_spec.py` builds the rungs from the indices that exist.
  - **B — §5.3 and `specs/ladder01.json` contradicted each other** on whether the ladder runs
    `grid01`-selected or published configurations. Resolved in favour of the spec (published,
    unchanged), which is also the only choice that leaves data size unconfounded with tuning.
    *The audit's corollary — that this makes `ladder01` independent of `grid01`, so the two
    could run concurrently and recover ~18 h against the deadline — is correct and is NOT
    taken: this lane is allotted one running job. It is recorded in the report as an
    opportunity the constraint cost.*
  - **M — `fno-epochmatch` sat inside the FNO's selection pool at 2.9× its siblings' wall**,
    so it would probably have won and a *budget* effect would have been reported as a *tuning*
    effect. Now declared a budget control, excluded from the §4.2 argmin
    (`selection_excluded`), reported in its own row.
  - **M — the knobs §1 promised were mostly absent**: learning rate appeared in one family of
    three, and weight decay, batch size and patience in none — while learning rate is the only
    knob the published lanes ever moved, and it moved results in both directions. The grid is
    rebuilt so every family gets capacity, normalisation, schedule **and** learning rate
    (§5.4); `fno-layers6` and `tsol-patch2`, the two most budget-confounded arms at 1.41× and
    2.12× step cost, make way.
  - **M — every grid arm is confounded with the wall budget**, because every published arm
    ended on its wall still improving, so a more expensive arm is scored with fewer epochs and
    "worse" would read as capacity. T1 now pre-registers that such an arm is a **lower bound**
    and may not be called worse than its baseline.
  - **M — T3 and T5 were already satisfied before any job ran, and T5 contradicted §2.** T5
    compared this lane's numbers against the panel's NM-ROM 1.89 %, which §2 states in terms
    are not subtractable — and the same `unet-refine` checkpoint scores 1.7110 % on
    diagnosis-8 against 4.5529 % on the panel, so the category error is 2.7×. Four *published*
    arms already clear 1.89 % on diagnosis-8. T3 named no statistic, and published
    `unet-medium` is already below 4.03 % on validation-32 mean, median **and worst**. Both
    criteria are rewritten: T3 names the statistic, the cohort and a bar `gen01` now measures
    on validation-32; T5 tests against the best published operator arm on this lane's own
    cohorts and hands any NM-ROM comparison to `ops-timing-panel`.
  - **M — T1 is biased toward "tuning helped"**: a 5–6-wide selection pool against a 3–4-wide
    one on the same 32 cases. Stated beside the ratio.
  - **M — G1's tripwire could not fire** (1.0 % against a worst already measured at 0.187 %)
    and **G2 has no resolving power** (a single-seed A/B whose expected effect is at or below
    this project's own measured one-seed noise on the same configuration). G1's threshold is
    now 0.5 % and it is labelled a measurement whose distribution is reported regardless; G2
    is pre-registered as **bounding** the effect, with the seed-control deltas quoted as its
    resolution limit, and a null result may not be read as "fidelity does not matter".
  - **M — §5.4's "and nothing else" was false**, and `audit.py` had quietly demoted five
    protocol assertions to recordings, two of which govern checkpoint selection. §5.4 now
    tabulates every code change and states exactly what remains asserted.
  - **M — the ladder's per-arm data-loading cost was unbudgeted.** Each 4608-case arm re-reads
    its training set several times inside `dataset.load_pair`, and `worker_second.train()`
    takes that out of later arms' wall. `make_ladder_spec.py` adds 14 000 s of slack.
  - **M — no disk budget and no disk stop rule**, on a share whose full-disk failure mode is an
    *empty log that reads as a code bug*. `worker_gen.py` now refuses to start below 80 GB free
    and prints `disk_free_bytes=`; §10 states the ~60 GB peak.
  - **M — the cache had no owner in the document.** §5.1 now says `gen01` produces the cache
    both later jobs stage from.
  - **Minors, all fixed:** "135 s per trajectory" was hand-typed against archived records whose
    median is 127.2 s (so 173 A100-h becomes 163); §3 item 3 compared a *margin* against a
    *deviation* and inferred "5× finer" from a setting never solved — `gen01` now measures
    256 / $\Delta t = 0.005$ directly; the reproduction gate asserts **arrays**, not bytes, and
    §3.2/§5.1/§9.4/§10 said bytes; `tsol-cosine` changed two variables; §1 quoted 2.4–2.8× and
    §6 2.4–3.4× for the same ratio over different arm sets; `gen_bank.py`'s docstring said 28×
    for 36×; the FNO had no training smoke; nothing checked the bank for internal duplicate
    inputs until training time; and the panel's six development cases were never declared
    collision-free.

  **Three things the audit could not verify, recorded as gaps rather than dropped:** the two
  NM-ROM rows of §3 (jobs `2835788` / `2837431` — no artifact is reachable from this
  repository, so §3 now marks them quoted-not-verified); the seeds of the panel's six
  development cases; and A100 epoch rates at 512 / 2048 / 4608 cases, which are projections
  from GB10 measurements scaled by published A100 epoch counts.
