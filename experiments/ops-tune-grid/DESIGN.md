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
budget rule gave the float32 U-Net and Transolver 2.4–2.8× more epochs than the float64 FNO
at the same 3000 s** (`no-second` report, "Equal wall in float32 buys more epochs than
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
the same kind of quantity as the 4.03 %, so the comparison this lane makes to that bar is
the apples-to-apples one, and the report says which comparison is which.

## 3. The data-parity accounting — established before any job

Both sides draw from the **same 5-parameter Gaussian-bump family** through the **same file**
`experiments/mr-burgers2d/engines.py` (`params_draw`), verified byte-identical between the
two lanes' worktrees, and both use the output times $\{0, 0.05, 0.10, 0.15, 0.20, 0.25\}$.
What differs is the count, what a "state" means on each side, and **the fidelity of the
solver that produced the training targets**:

| | trajectories | states the model is fitted on | training-target solver | seconds per trajectory (A100) |
|---|---:|---:|---|---:|
| NM-ROM bank $g$ (job `2835788`) | 576 | 16 384 states | **256 intervals, $\Delta t = 0.005$, direct** | 0.19 |
| NM-ROM head $h_\theta$ (job `2837431`) | **4608** | **131 072 states** (of 235 008 available at 51 states/traj) | **256 intervals, $\Delta t = 0.005$, direct** | 0.19 |
| Operators, as published | **128** | 128 inputs → **640** supervised evolved states (5 per case) | **4096 intervals, $\Delta t = 1.5625\times10^{-4}$, restricted to 256** | **135** |
| Operators, this lane (target) | **4608** | 4608 inputs → **23 040** supervised evolved states | **1024 intervals, $\Delta t = 3.125\times10^{-4}$, restricted to 256** (§3.2) | **4.8** |

Three things follow, and the report states all three:

1. **The 128-vs-4608 gap is a data-*generation-cost* artefact, not a design choice.** The
   operators' training targets were held to a far stricter reference than the NM-ROM's own
   training trajectories: 135 s versus 0.19 s per trajectory, ≈710×. 128 cases is what
   4.8 GPU-hours buys at that fidelity.
2. **Trajectory parity is reachable; state parity is not.** At 4608 cases an operator sees
   4608 × 5 = 23 040 supervised evolved states against the head's 131 072 fitted states —
   still 5.7× fewer — because the operator's contract fixes **six** output times per
   trajectory while the head is fitted per state at **51** times. That is a property of the
   operator contract, not something this lane can or should change, and the report says so
   rather than claiming full parity.
3. **The NM-ROM's own training targets are coarser than anything this lane gives the
   operators.** The head trained on 256-interval, $\Delta t = 0.005$ solutions. The
   calibration gate measured the deviation of a 256-interval, $\Delta t = 1.25\times10^{-3}$
   solve from the fine anchor at **1.03 %** worst; $\Delta t = 0.005$ is four times coarser
   still. The operators' new data at 1024 / $3.125\times10^{-4}$ deviates by at most
   **0.19 %** (§3.2). So the operators are trained on targets **at least 5× finer** than our
   own model's, at every rung of this lane's ladder.

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

### 3.2 Why the new data is generated at 1024 / $3.125\times10^{-4}$, and how that is controlled

Generating 4608 cases at the pinned 4096 / $1.5625\times10^{-4}$ reference costs 4608 × 135 s
= **173 A100-hours**. It does not fit this lane, this campaign, or the 09-25 deadline. The
frozen calibration (`artifacts/calibration01/calibration-index.json`, 8 independent
development cases, output 256) measured each candidate's deviation from the fine anchor:

| solver setting | worst deviation from anchor | median | work proxy | measured s/case |
|---|---:|---:|---:|---:|
| 256, $1.25\times10^{-3}$ | 9.13e-3 | 6.65e-3 | 1.31e7 | 1.20 |
| 512, $6.25\times10^{-4}$ | 4.33e-3 | 3.12e-3 | 1.05e8 | 1.74 |
| **1024, $3.125\times10^{-4}$** | **1.87e-3** | **1.33e-3** | 8.39e8 | **4.80** |
| 4096, $3.125\times10^{-4}$ (the anchor) | 0 | 0 | 1.34e10 | 68.7 |

The gate's own per-case margin for the 1024 setting — its deviation from the anchor **plus**
that case's anchor refinement margin, maximised over the eight cases — is **3.056e-3 =
0.31 %** (`candidates[2].worst_empirical_margin`). That is the maximum of a per-case sum, not
the sum of the two column maxima quoted above (1.87e-3 + 1.386e-3 = 3.26e-3, a looser figure
because the two maxima fall on different cases); the report quotes the gate's number and says
which it is. **Both are observed maxima over eight development cases, not a bound over 4608
cases**, and neither is a certified error: the protocol itself calls space/time refinement
differences "empirical development evidence, not rigorous continuum error bounds". They are
quoted here only to size the decision.

At that size, operator errors in this comparison are **2–7 %**, so the label noise is roughly
10–20× below the quantity being measured, and — see §3 item 3 — five or more times *finer*
than our own model's training targets.

That argument is not accepted on its own, and the eight-case calibration figure is explicitly
**not** the acceptance test. Two pre-registered controls measure it on the real data:

- **G1, measured in `gen01` on the real training cases, not on development cases.** Every one
  of the 128 published training cases is re-solved at 1024 / $3.125\times10^{-4}$ and the
  fixed-initial error against its own pinned 4096-reference target is recorded per case. The
  report gives the worst, median and mean of those 128 numbers. **If the worst exceeds
  1.0 %** — a quarter of the smallest operator error in the comparison — the ladder below
  256 cases is reported as confounded and the cheap-fidelity rungs are labelled accordingly.
- **G2, a training control in `ladder01`.** One arm retrains the tuned U-Net on the **same
  128 cases** at the cheap fidelity. Its validation-32 error is compared with the identical
  arm trained on the pinned-fidelity 128 cases. **Any difference between the two is the
  target-fidelity effect at fixed data size**, measured directly rather than argued. The
  ladder's 128 → 4608 movement is only attributed to data size to the extent it exceeds this
  control.

A **reproduction gate** runs before either: case `burgers-train-00000` is regenerated at the
**pinned** 4096 / $1.5625\times10^{-4}$ setting and asserted **bit-identical** to the cached
file. That proves this lane's generator reproduces the published data exactly, which is a
stronger statement than any hash of a calibration JSON. A failure voids `gen01`.

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

### 5.1 `gen01` — data (≈ 7 h, A100)

In order, writing its index incrementally so a truncated run is still usable:

1. **Reproduction gate** — regenerate `burgers-train-00000` at the pinned 4096 /
   $1.5625\times10^{-4}$ setting; assert bit-identical to the cached npz. (~135 s)
2. **G1** — re-solve training indices `0..127` at 1024 / $3.125\times10^{-4}$; record the
   per-case fixed-initial error against the pinned target. These 128 cases double as the
   cheap-fidelity 128-case training set used by control G2. (~10 min)
3. **Bulk** — solve training indices `128..4607` at 1024 / $3.125\times10^{-4}$. (~6.0 h)
4. Write prefix indices `index-00128/00512/02048/04608.json` over whatever completed.

### 5.2 `grid01` — the tuning grid at the published 128 cases (≈ 16 h, A100)

Run on the **pinned** 128-case training set so the tuning effect is measured with data held
at exactly the published value. Screen budget **3000 s per arm — the published rule,
unchanged** — so every arm in this grid is directly comparable to the published arms. §6
handles the budget question separately rather than by moving this bar.

### 5.3 `ladder01` — error versus training-set size (≈ 18 h, A100)

Each family's **`grid01`-selected** configuration, trained at 128 / 512 / 2048 / 4608 cases,
**4000 s of wall each**. At fixed batch size, equal wall is very nearly equal optimisation
steps, so a ladder rung differs from its neighbours in the data the same number of gradient
steps is drawn from — which is the question. Plus control **G2** (§3.2) and one 3× long run
at the top rung to test whether the budget binds there.

Jobs 4–6 are held in reserve for failures and for anything §7 says must be re-run.

### 5.4 The grid, fixed before any job

Capacities and knobs per family. Published arms are named for reference and are **not**
re-run; their numbers are carried from the published audits.

**FNO** (float64 by construction; published: width 64 / modes 32 / layers 4, lr 1e-3)

| arm | change from `fno-large` |
|---|---|
| `fno-modes48` | modes 32 → 48 (the 257² grid supports 128) |
| `fno-width96` | width 64 → 96 |
| `fno-layers6` | layers 4 → 6 |
| `fno-norm` | `norm='group_norm'` inside the FNO blocks |
| `fno-cosine` | `ReduceLROnPlateau` → cosine decay to 1e-5 |
| `fno-epochmatch` | `fno-large` unchanged, **epoch target 1963 with an 8700 s wall cap** — the epoch count `unet-medium` reached in 3000 s (§6) |

**U-Net** (float32; published: base 24/32/48, GroupNorm(8), lr 1e-3)

| arm | change from `unet-medium` |
|---|---|
| `unet-base64` | base 32 → 64 (above the published range) |
| `unet-groups16` | GroupNorm(8) → GroupNorm(16) |
| `unet-lr3e-3` | lr 1e-3 → 3e-3 |
| `unet-cosine` | cosine decay to 1e-5 |
| `unet-base48-cosine` | base 48 + cosine |

**Transolver** (float32; published: dim 128/192/256, 8 layers, 8 heads, 64 slices, patch 4)

| arm | change from `tsol-small` (dim 128) |
|---|---|
| `tsol-slices32` | slices 64 → 32 |
| `tsol-slices128` | slices 64 → 128 |
| `tsol-layers6-d192` | dim 192, layers 8 → 6 |
| `tsol-patch2` | patch 4 → 2 (4× the tokens) |
| `tsol-cosine` | cosine decay to 1e-5 |

**Code changes this requires**, and nothing else: `model.make_model` passes
`norm=config.get('norm')` to `neuralop.FNO`; `families.UNet2d` already takes `groups`;
`train.py` gains `config['schedule'] in ('plateau', 'cosine')` defaulting to `'plateau'`, the
existing behaviour; `worker_second.py` gains a per-arm `train_index` so a ladder rung can name
its own index file, defaulting to `data/train/index.json`. Every default reproduces the
current behaviour exactly, and the local contract smoke asserts it.

## 6. The budget rule, and what this lane changes about it

The published rule is **equal wall, not equal epochs**. At 3000 s the float32 families ran
1628–2327 epochs and the float64 FNO ran 692 — a 2.4–3.4× epoch advantage to the families
this project introduced. **This lane does not change the screen rule** (§5.2): moving it
would confound tuning with budget. It adds one arm, `fno-epochmatch`, with an explicit
**epoch target of 1963** — the count `unet-medium` reached in its published 3000 s — under an
**8700 s wall cap**, and **reports both** the equal-wall number and the equal-epoch number
side by side. 8700 s is sized from `fno-large`'s published throughput (692 epochs in 3003 s →
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
- **T2 (data moved it).** Per family, validation-32 and diagnosis-8 error at 128 / 512 /
  2048 / 4608. **Declared in advance:** if error falls monotonically across the ladder and is
  still falling at 4608, the report states that the operator baselines are **data-limited at
  every size this lane could reach** and that the published 128-case numbers are an upper
  bound on what these families do, not a property of the architectures.
- **T3 (the 4.03 % bar).** For every arm in this lane, whether its validation-32 and
  diagnosis-8 errors fall below **4.03 %**. **If any tuned or data-scaled arm does, the
  report says so in its first paragraph**, names the arm, and states the consequence for the
  paper's qualification. This is the finding this lane most wants to know about, not the one
  it wants to avoid.
- **T4 (what the baselines were given that our model was not).** The report tabulates, per
  arm: training cases, supervised states, training-target solver fidelity, wall budget,
  epochs, and whether a hyperparameter search selected it. Our own head's corresponding row
  (§3) sits beside them. **Wherever an operator received more, the report says so in the
  table and in prose**, because the paper should state that the baselines were given the
  stronger protocol.
- **T5 (a tuned baseline that beats us).** If any arm's diagnosis-8 or validation-32 worst
  falls below the NM-ROM fast arm's 1.89 %, that is reported in the first paragraph, with the
  cohort caveat of §2, and handed to the `ops-timing-panel` lane for timing.

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
  the missing rungs named, never interpolated.
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
