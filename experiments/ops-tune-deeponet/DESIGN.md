# ops-tune-deeponet — how strong can the DeepONet baseline on 2D Burgers at 256² honestly be made?

Pre-registration. Written before the first GPU job; every later change is an amendment in §A
with its date and reason. Branch `exp/2026-09-22-ops-tune-deeponet`, worktree
`worktrees/2026-09-22-ops-tune-deeponet`, forked from `exp/2026-09-22-ops-deeponet-b2d` at
`306c939d`. Cluster namespace `/cluster/tufts/paralab/tawal01/opstune_don_20260922/`.

Read `experiments/ops-deeponet-b2d/DESIGN.md` first: this lane inherits its contract, its
metric, its data, its gates and its DeepONet implementation unchanged, and changes only the
things named in §2.

## 1. The question

`ops-deeponet-b2d` trained four DeepONet arms on 2D viscous Burgers at 256² and got
14.79–18.22 % mean / 11.75–14.80 % median validation error against 1.35–2.28 % for the
selected U-Net / Transolver / FNO arms and 1.87 % for the NM-ROM. All four ended by **early
stopping**, and both of that lane's audits named the same two unresolved explanations:
**128 training cases**, and **an inherited schedule that was never tuned for this family**.

A reviewer will ask whether the paper's DeepONet was simply under-trained and under-fed. This
lane answers that, and only that:

- **Q1 (data parity).** What does each side of the comparison actually train on — how many
  trajectories, and how many solution states — stated precisely for the operators and for our
  NM-ROM bank and head?
- **Q2 (data starvation).** How does DeepONet's validation error move as the training set
  grows from 128 toward parity with our model's 4608 trajectories, at fixed compute?
- **Q3 (tuning).** With the data question separated, how much does DeepONet-specific tuning of
  the learning rate and schedule, the stopping rule, the trunk, the branch bottleneck and the
  output normalisation buy?
- **Q4.** What is the strongest DeepONet this lane can honestly build, and where does it land
  against the other three operator families and against our NM-ROM?

**What this lane cannot settle, and will not claim:** an architecture ceiling in either
direction, a speed number of any kind (§7), or anything about a second PDE, mesh or seed.

## 2. Q1 — the data-parity accounting, stated before any job

The two sides of the paper's comparison are trained on different amounts of data. This is the
accounting, derived from the pinned artefacts named beside each row; §6 of the report
regenerates it and the report never re-types it.

### 2.1 What an operator arm trains on

The pinned cache `/cluster/tufts/paralab/tawal01/no_burgers_20260914/pilot-data01`, generated
by job 3702709 from `experiments/neural-operator-burgers/data.py` at source commit
`5169c095`:

| quantity | value | where it comes from |
|---|---:|---|
| training trajectories | 128 | `train/index.json`, `count`; index sha `5333584b…` |
| supplied input fields | 128 | one per trajectory: $u(\cdot, 0)$ on the 257² nodal grid |
| supervised output fields | 128 × 5 = 640 | the five evolved times; $t = 0$ is returned bitwise and `train.py` drops it from the loss (`errors[:, 1:]`) |
| distinct solution states seen | 128 × 6 = 768 | including the supplied $t=0$ field |
| validation trajectories | 32 | `validation/index.json`, index sha `468b9e70…` |

A trajectory is one draw of $(c_x, c_y, w, a, \nu)$ from `engines.params_draw`, solved to
$t = 0.25$ and stored at $\{0, .05, .10, .15, .20, .25\}$.

### 2.2 What our NM-ROM trains on

The frozen Burgers checkpoint the paper reports, `sep_hfit_dense_mid_N256_dense.pkl`:

| component | trajectories | states | source |
|---|---:|---:|---|
| spatial bank $G$ | 576 | ≤ 16384 POD snapshots (`max_snaps`) | job 2835788; `sample_params(seed=0)`, `n_traj = 576` |
| head $h_\theta$ | **4608** | **131072 decoder codes** | job 2837431; those 576 plus `sample_params(seed=1000, m=4032)`; `hfit_n_traj = 4608`, `hfit_extra_traj = 4032` |

Each trajectory carries `num_steps = 50` steps, so 51 states; 4608 × 51 = 235008 states exist
and the incumbent checkpoint carries 131072 of them (LAB-LOG 2026-09-16, the head-training
lane's correction; the 4608/4032/576/50/16384 figures are read from
`reports/2026-09-10-historical-poisson-and-burgers-cost-audit.json`).

**The disparity, stated the way the paper must state it:** 4608 / 128 = **36× more
trajectories**, and 131072 / 640 = **205× more supervised states**. The two "states" numbers
are not the same kind of object — our head is fit to reconstruct individual states, an
operator is fit to map an initial state to five later ones — so the paper quotes the
trajectory ratio as the headline and the state counts beside it with that caveat.

`engines.params_draw` and `burgers2d_film.sample_params` are the same sequential draw over the
same ranges (LAB-LOG 2026-09-21), so the two sides are drawing from the **same family**; only
the count differs.

### 2.3 Why parity cannot be reached at the pinned reference protocol, and what is done instead

The pinned 128 cases were solved at the 4096-interval, $\Delta t = 1.5625\times10^{-4}$ anchor
and restricted to 256 — `wall_seconds_including_first_compile` ≈ **114 s per case** in the
generating job. 4608 cases at that setting is ≈ 146 GPU-hours, more than this lane's entire
budget. The protocol's own calibration (`pilot-data01/refinement/index.json`, gate status
`calibrated`, budget $10^{-3}$) measured the cheaper candidates against that anchor at output
256 over 8 independent calibration cases:

| reference setting | work proxy $L^2 n_t$ | worst empirical margin vs the anchor |
|---|---:|---:|
| 4096, $\Delta t = 1.5625\times10^{-4}$ (**pinned**) | 2.684e10 | 9.588e-4 (its own refinement differences) |
| 1024, $\Delta t = 3.125\times10^{-4}$ | 8.389e8 | 2.781e-3 |
| 512, $\Delta t = 6.25\times10^{-4}$ | 1.049e8 | 5.251e-3 |
| 256, $\Delta t = 1.25\times10^{-3}$ | 1.311e7 | 1.006e-2 |

**Pre-registered decision rule for the reference setting, fixed before the job.** `gen01`
profiles the 1024 and 512 settings on the first two training cases, then generates at the
**finest setting whose measured throughput completes 4608 cases inside the generation budget**;
if neither does, it generates at 512 and produces the largest nested prefix that fits. The
chosen setting, its measured rate and the realised count are recorded in the generated index
and reported. No setting coarser than 512 is used.

These new cases are **not claimed to be the pinned protocol**. Two things keep that honest:

1. **The ladder is a nested prefix of one draw.** `gen01` generates train cases
   $0, 1, 2, \dots$ with `data.py`'s own `case_seed('train', i)`, so cases 0–127 are *the same
   physical draws* as the pinned 128 — the protocol's declared `future_train_prefixes`
   extended. Rungs [128, 512, 2048, 4608] are prefixes of that one sequence, so training-set
   size is the only variable across rungs.
2. **The target-protocol difference is measured on the training distribution itself, not
   assumed.** Because cases 0–127 are the same draws, `gen01` compares each newly generated
   target with the pinned 4096-anchor target of the same case and records the per-case
   fixed-initial difference over all 128. That is a direct, on-distribution measurement of what
   the cheaper reference changes, and it is reported with its maximum and median.
3. **A trained control isolates it.** Arm `c-pinned128` trains the inherited configuration on
   the pinned 128 cases and arm `c-new128` trains it on the newly generated 128 — same cases,
   same schedule, different target protocol. If they differ by more than the pre-registered
   noise band (§5, T0) the ladder is reported with that caveat attached to every rung.

**Everything is graded against the pinned validation-32 and the pinned 8-case cohort**, which
are untouched 4096-anchor data. Only training targets change.

## 3. What is inherited and what changes

Byte-identical to `experiments/ops-deeponet-b2d` at `306c939d` (proved by
`check_inherited.py` → `checks/inherited-sources.json`): `dataset.py`, `model.py`,
`evaluate_cohort.py`, `prepare_diagnosis_cohort.py`, `spectral_conv_f64.py`,
`NEURALOPERATOR-LICENSE`, `check_fno_parity.py`, `check_burgers_error_definition.py`.

Byte-identical to the pinned generator at commit `5169c095` (same proof file):
`engines.py` (`experiments/mr-burgers2d/engines.py`, sha `820039e5…`), `sep_common.py`
(`experiments/separable-decoder/sep_common.py`, sha `a74f0279…`) and `data.py`
(`experiments/neural-operator-burgers/data.py`, sha `8d491d5e…`) with `protocol.json`
(`212bc898…`). These are exactly the modules that produced the pinned 128 cases.

Changed, and only in these ways:

| file | change |
|---|---|
| `families.py` | `DeepONet2d` gains `trunk_layers` (default 3, the inherited depth) and keeps `pool_bins` and `frequencies` as they already were; the U-Net, Transolver and the DeepONet defaults are otherwise untouched |
| `train.py` | validation cadence and patience measured in **optimisation steps** instead of epochs (§4.1); optional cosine schedule with warm-up (§4.2); optional per-output-time output scale (§4.3); `--pool` / `--pool-limit` for the prefix ladder (§4.4). Every option defaults to the inherited behaviour, so an inherited config trains exactly as before |
| `gen_more.py` | **new** — extends the training split past 128 cases at a declared reference setting, using `data.py`'s own solver, schema and case seeding |
| `build_pool.py` | **new** — packs a verified case index into three `.npy` arrays so 17 GB is read once per job instead of once per arm |
| `worker_tune.py` | spec-driven arm runner; the parent's `worker_second.py` with the refinement rule replaced by the composition rule of §5.3 and with per-arm data-source selection |
| `audit.py` | admits the new arm names and the new `stop_reason` values; recomputes every reported error from the saved prediction fields exactly as before |
| `cluster/stage.py` | lane path, namespace, two data mounts instead of one, symlinked read-only caches instead of copies, staged-file list |
| `cluster/collect.py` | lane path and namespace |
| `timing.py` | **removed from this lane.** No speed number is admissible here (§7); the file is not staged and no timing block runs |

### 3.1 What DeepONet is being given that the other three families were not

Stated here so the report and the paper can state it without hedging. The U-Net, Transolver
and FNO arms each got: 128 training cases, an inherited schedule, a 3000 s per-arm wall budget,
one seed, and one learning-rate refinement of the validation-selected capacity. This lane gives
DeepONet, and **only** DeepONet:

1. up to **4608** training cases instead of 128;
2. a **hyperparameter sweep** over ten one-factor arms plus a composed arm;
3. a **tuned schedule** (cosine with warm-up) and a stopping rule expressed in steps;
4. a **3× longer** final wall budget (9000 s) for the selected arm.

This is a deliberate asymmetry in the baseline's favour. The paper reports it as such. Every
number in the comparison table carries its arm's data size and wall budget so a reader can see
which rows are like-for-like: **the like-for-like row is the inherited configuration at 128
cases and a 3000 s budget**, which is `ops-deeponet-b2d`'s `don-small` and is reproduced here
as `c-pinned128`.

## 4. Protocol

### 4.1 Stopping and validation cadence — in steps, not epochs

At 128 cases an epoch is 16 optimisation steps; at 4608 it is 576. The inherited rule
(validate once per epoch, stop after 250 epochs without a new best) is therefore not one rule
but four different rules across the ladder, and it is unusable at the top rung. This lane
validates every `validate_every_steps` = 500 optimisation steps and stops after
`patience_evaluations` = 40 consecutive evaluations without a new best selection score — i.e.
20000 steps of no improvement, which at 128 cases is 1250 epochs, five times the inherited
patience, and is **the same rule at every rung**. The `ReduceLROnPlateau` scheduler, when used,
steps on the same cadence with patience 8 evaluations (4000 steps) — the inherited 20 epochs at
128 cases is 320 steps, which no longer makes sense across the ladder; 4000 steps is the
closest schedule-shaped equivalent at the top rung and is held constant across rungs.

Consequence, declared in advance: `c-pinned128` is therefore **not** bitwise `don-small`. It is
the inherited configuration and data under this lane's stopping rule, and the difference
between the two is the first thing the report shows, as the measurement of what the stopping
rule alone changed.

### 4.2 Schedule

`schedule: "plateau"` (default) is the inherited `ReduceLROnPlateau(0.5, min 1e-5)`.
`schedule: "cosine"` decays the learning rate from its initial value to `1e-6` over the arm's
**wall budget** (the fraction of the budget elapsed drives the cosine, because the step count
is not known in advance), after a linear warm-up over `warmup_steps`. A cosine arm cannot
early-stop below its budget by construction of the schedule; it is still subject to §4.1's rule
and, if that fires, the arm is recorded as `early_stopping` with the schedule unfinished.

### 4.3 Output normalisation

Inherited: one global scale, the RMS of all training targets, multiplying the network output.
`output_scale_mode: "per_time"` uses one scale per evolved output channel (five scales, the RMS
of that time's training targets). Both are computed from training data only.

### 4.4 The pool and the prefix ladder

`build_pool.py` reads a checksum-verified case index and writes `input.npy`, `target.npy`,
`parameters.npy` and `cases.json` (the ordered case ids, the index sha256, and each source
file's sha256). `train.py --pool <dir> --pool-limit N` asserts that the pool's recorded index
sha256 equals the `--train-index` it was given and then trains on the **first N cases in index
order**, which is `case_index` order, which is the nested prefix of §2.3. `--pool-limit`
absent means all cases.

### 4.5 Everything else is unchanged

Input/output contract, the boundary mask, the float64-outside-the-network precision, the
loss (mean squared fixed-initial relative error over the five evolved times), the error
metric, the seed (20260914), the batch size (8) unless an arm declares otherwise, AdamW with
weight decay 1e-4 unless an arm declares otherwise, and the 32-case validation set.

## 5. Pre-registered selection and criteria

### 5.1 The selection metric — unchanged from the parent lane

**The selected arm is `argmin`, over all arms of the stage being selected, of the mean over the
32 validation cases of the per-case maximum over the six output times of the fixed-initial
relative error.** This is the FNO lane's rule, applied unchanged.

As in the parent, and for the same reason:

- the report gives **every** arm's mean, median, worst and >5 % count, not only the selected one;
- if the selected arm's worst case is worse than any unselected sibling's, the report says so
  beside the verdict, naming both arms, and reports the **best-worst-case arm** as well;
- the rule is not changed after seeing the numbers and no alternative rule is introduced post
  hoc.

**Selection surface.** Validation-32 only. The matched eight-case ROM/FOM cohort is scored
**once, after every selection decision is fixed**, and is never used to choose anything. No
final or held-out cohort is opened; `dataset.py` refuses those splits by construction and
`PROVENANCE.json` records `final_cohort_opened: false`.

**Selection noise, declared in advance.** Choosing among ~12 arms on 32 cases will overfit the
selection surface somewhat. The direction of that bias is *in the baseline's favour*, which is
the conservative direction for this paper, and the report says so rather than pretending the
selected number is unbiased. The report also gives the full arm table so the spread is visible.

### 5.2 Criteria

- **T0 (target-protocol control).** `c-new128` and `c-pinned128` agree iff their validation
  means differ by less than **10 %** relative. If they do not, every ladder rung above 128 is
  reported with that discrepancy attached, and the data verdict is stated as a bound rather
  than a value.
- **T1 (data starvation).** DeepONet is **data-limited at 128 cases** iff the top rung's
  validation mean is at most **0.7×** the 128 rung's, at the same configuration and the same
  wall budget. It is **saturated at the top rung** iff the last rung-to-rung step improves the
  validation mean by less than **10 %** relative (the 10 % saturation threshold this project
  already used for the head's own density ladder).
- **T2 (tuning).** Tuning helped iff the composed arm improves the validation mean by at least
  **10 %** relative over `base` at the same data and budget.
- **T3 (competitive).** The best DeepONet is competitive with the FNO iff its validation-32
  worst **and** median are each within **1.5×** `fno-large`'s (2.2811 % mean, 1.8054 % median,
  6.3825 % worst) — the parent's D1 bar, unchanged, so the four families are graded identically.
- **T4 (cohort).** On the matched eight cases, whether the best DeepONet's worst fixed-initial
  error is below the ROM's 1.8671 %. Reported as a fourth draw on that question; one case
  decides this column and the report says so.
- **T5 (the discretisation watch-item).** The report states, for every arm, whether its worst
  validation error falls below **4.0265 %** — the converged same-grid full-order model's own
  worst error against the refined reference at 256 intervals, measured on the **6-case
  development cohort** of job bpn301, not on validation-32. Because the cohorts differ, this is
  reported as a context line and never as a like-for-like comparison. If an arm falls below it,
  that is flagged for the paper's appendix as the campaign brief requires.
- **T6 (honest negative).** If the tuned, data-fed DeepONet is still the weakest family, that
  is the result and it is reported as such, with the explicit statement that it is not an
  architecture ceiling: this lane tuned ten one-factor knobs around one implementation, at one
  mesh, one PDE and one seed.

### 5.3 The composition rule, fixed before the sweep

The composed arm `tuned` takes **every knob whose one-factor sweep arm improved the selection
metric by at least 2 % relative over `base`**, each at the value that arm used; where two arms
vary the same knob (the three learning rates, the two pool sizes) only the better one is taken.
If no arm clears 2 %, `tuned` is `base` and T2 fails by construction. If the composed arm is
worse than the best single-knob arm, **the best single-knob arm is the selected arm** under
§5.1 and the composition is reported as a failed combination.

## 6. The jobs

Budget: **≤ 1 running, ≤ 6 total**, shared account cap 6 running with three other lanes.
`squeue -u tawal01` is checked before and after every submission. One job per directory.

| job | what it does | arms | wall |
|---|---|---|---|
| `gen01` | profile, then generate train cases 0…4607 at the selected reference setting; compare cases 0–127 with the pinned targets | — | ≤ 8 h |
| `tun01` | `c-pinned128`, `c-new128`, `base-top` at 3000 s each; then the ten one-factor sweep arms at 1500 s each, at the top rung | 13 | ≤ 10 h |
| `lad01` | the ladder: `base` at the remaining rungs and `tuned` at every rung, 3000 s each | ≤ 6 | ≤ 7 h |
| `fin01` | the selected arm and, if different, the best-worst-case arm, at 9000 s at the top rung; then the matched-cohort scoring of every checkpoint this lane produced | ≤ 2 | ≤ 7 h |

Two jobs are held in reserve for failures. A job that dies in its preamble with zero training
GPU time does not count against the four (the `no-second` rule) but is recorded with its logs.

**The sweep arms**, fixed here before `gen01` runs. Baseline = the inherited `don-small`
(width 48, rank 256, trunk width 384, trunk layers 3, levels 3, pool bins 4, frequencies
(1, 2, 4), AdamW lr 1e-3 / wd 1e-4, plateau schedule, batch 8, float32 network, seed
20260914), at the top rung:

| arm | knob | value |
|---|---|---|
| `base` | — | the inherited configuration, at the top rung |
| `s-cos` | schedule | cosine to 1e-6 with 2000-step warm-up |
| `s-lr3e-3` | learning rate | 3e-3, cosine |
| `s-lr3e-4` | learning rate | 3e-4, cosine |
| `s-pool8` | branch bottleneck | `pool_bins` 8 (an 8 × 8 branch code instead of 4 × 4) |
| `s-pool16` | branch bottleneck | `pool_bins` 16 |
| `s-freq` | trunk features | frequencies (1, 2, 4, 8, 16) |
| `s-trunk` | trunk capacity | `trunk_width` 768, `trunk_layers` 4 |
| `s-rank` | rank | 512 |
| `s-pertime` | output normalisation | one output scale per evolved time |
| `s-batch32` | batch size | 32 (four times the gradient batch at the same wall budget) |

`s-lr3e-3`, `s-lr3e-4` and every arm after `s-cos` are run **with whichever of `plateau` or
`cosine` won in `s-cos` versus `base`**, decided inside the job from the recorded selection
scores, so the learning-rate and capacity knobs are not tested against a schedule already known
to be worse. That decision rule is fixed here and is recorded in the job's
`selection.json`.

## 7. Timing — nothing is admissible from this lane

No speed number is produced, quoted or implied here. `timing.py` is not staged and no timing
block runs. If a tuned arm is worth timing, its checkpoint path and SHA256 go into
`reports/timing-handoff.json` in the format `ops-timing-panel`'s harness consumes (the pattern
is `experiments/ops-deeponet-b2d/reports/timing-handoff.json`), together with the `families.py`
and `model.py` that harness needs, and the report says so. **Under no circumstance is a time
from any job of this lane divided by a time from any other job.**

## 8. Gates — a job whose log fails any of these is void

1. `jax_backend=gpu` (generation) and `torch_backend=cuda` (training) in the logs; the run ends
   with `ALL-DONE`.
2. `sha256sum -c MANIFEST.sha256` passes, and every mounted data cache verifies against its own
   `DATA.sha256` before any training or generation.
3. The validation index hash equals the FNO job's `468b9e70…`, and for arms trained on the
   pinned cache the train index hash equals `5333584b…`. For arms trained on the extended pool
   the train index hash equals the hash recorded in `gen01`'s manifest.
4. `gen01`'s generated cases 0–127 reproduce the pinned cases' **generation descriptors and
   input fields bitwise** (same `case_seed`, same `engines.initial`); only the evolved targets
   may differ, and their difference is measured, not assumed. A mismatch in the input field or
   descriptors voids the job.
5. Every arm in the spec is present and complete, or explicitly recorded as skipped for budget;
   a silently dropped arm fails the audit.
6. Every reported error is recomputed from the saved prediction fields by `audit.py`, which
   imports neither torch nor jax. Batch-8 selection score and batch-1 reported score may differ
   by at most 1e-5 for a float32 network (the parent's pre-registered tolerance).
7. The supplied initial state is returned bitwise and the boundary is exactly zero in every
   saved prediction.
8. Exactly one job per submit directory, confirmed in `squeue` before and after submission.
9. Nothing over 50 MB is committed; results and logs are pulled with checksums and the remote
   job directory is deleted afterwards.

## 9. Risks

- **The cheaper reference protocol is the main new risk.** It is measured on 128 real training
  cases and controlled by a trained arm (§2.3), and every evaluation cohort stays at the pinned
  anchor. If T0 fails, the ladder still answers Q2 in the direction it answers it, but the
  magnitude is reported as a bound.
- **Generation may not reach 4608.** The ladder is a nested prefix, so a short run yields a
  shorter but valid ladder. The realised top rung is whatever `gen01` produced and is reported
  as such.
- **One-factor-at-a-time misses interactions.** Eleven arms is what the budget affords. §5.3's
  composition rule tests exactly one interaction — all the winners together — and reports it
  honestly if it fails.
- **Selection noise on 32 cases** (§5.1), biased in the baseline's favour.
- **Single seed.** No seed robustness is claimed. If a job is left over, a two-seed repeat of
  the selected arm is the first thing to spend it on.
- **A larger training set changes host memory and I/O, not the science.** `build_pool.py` and
  the symlinked read-only caches exist for that; both are checked by gates 2 and 3.

## A. Amendments

*(append-only; date, reason, action)*

- **A1 (2026-09-22, before `gen01`).** Independent audit of this pre-registration commissioned
  before the first GPU job; disposition recorded here before submission.
