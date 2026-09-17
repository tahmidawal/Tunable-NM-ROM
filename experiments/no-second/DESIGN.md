# no-second — a second and third neural-operator baseline on the Burgers common dataset

Pre-registered 2026-09-17 before the first cluster job. Amendments are appended as
§A1, §A2, … and never rewrite what is above them.

## 1. Question

The paper's neural-operator comparison (table T14) currently has one operator family,
the FNO of lane `no-audit` (four capacities, equal wall budget, validation-selected,
f64, job `3710846`). The old reviews attacked the baselines as broken or untuned. Does
the same conclusion hold when at least one more strong, standard operator family is
trained on the **same data, same split, same reference, same metric and same
protocol**? Concretely: what worst and median fixed-initial error does a tuned
**U-Net** (and, if time allows, a **Transolver**) reach on the 32 validation cases and
on the matched 8-case ROM/FOM diagnosis cohort, and where does it sit relative to the
FNO, the ROM and the efficient FOM on those eight cases?

This is an accuracy-only lane. No speed ratio is formed here (cross-job); the
`b-panel` lane may time the saved checkpoints in its own same-job panel.

## 2. What is held fixed (identical to the FNO arm)

| Item | Value | Where it is enforced |
| --- | --- | --- |
| Dataset | Burgers common dataset, `/cluster/tufts/paralab/tawal01/no_burgers_20260914/pilot-data01` (read-only; copied into each attempt and re-verified against its `DATA.sha256`) | `cluster/stage.py`, `run.sbatch` |
| Split | 128 train / 32 validation, index SHA256 `5333584b…` / `468b9e70…` (recorded again in every `provenance.json` and in `summary.json`) | `dataset.load_pair` |
| Matched cohort | the 8 calibration cases with their 4096-interval, $\Delta t = 1.5625\times10^{-4}$ anchors restricted to the 256-interval grid, rebuilt in-job by the parent's `prepare_diagnosis_cohort.py` (unchanged), disjointness from training re-checked by seed and by input content | `prepare_diagnosis_cohort.py` |
| Inputs | normalised supplied initial field, normalised viscosity channel, x, y — `model.features`, unchanged | `model.py` |
| Outputs | five evolved fields as channels, masked to zero Dirichlet, supplied state prepended bitwise — `model.predict`, unchanged; direct multi-time output, not a rollout | `model.py` |
| Metric | $E=\max_{k}\lVert \hat u(t_k)-u^{\rm ref}(t_k)\rVert_{2,\rm int}/\lVert u^{\rm ref}(t_0)\rVert_{2,\rm int}$, `model.relative_errors` unchanged; independently recomputed with NumPy in the audit | `model.py`, `audit.py` |
| Loss | mean over the five evolved times of the squared relative error (identical) | `train.py` |
| Optimiser | AdamW, lr $10^{-3}$, weight decay $10^{-4}$, batch 8, gradient-norm clip 1.0, ReduceLROnPlateau(0.5, patience 20, min $10^{-5}$) on the validation mean case-max error | `train.py` |
| Selection | checkpoint with the best validation **mean case-maximum** error; early stopping after 250 epochs without improvement; epoch cap 4000 | `train.py` |
| Budget | **3000 s of wall per capacity on one A100**, exactly the FNO's; epoch counts differ by design (equal compute, not equal epochs) | `worker_second.py` |
| Refinement | the validation-selected capacity retrained from scratch at lr $3\times10^{-4}$ with the same budget (`*-refine`, as `fno-refine`) | `worker_second.py` |
| Seed | 20260914 (single seed, as the FNO) | configs |
| "Tuned" means | three capacities spanning the FNO's parameter range, each given the FNO's budget and selection rule, plus the lower-LR retrain of the winner. **Nothing is tuned on the 8-case cohort and nothing is tuned per case.** | this file |

Framework note. The brief described the parent as JAX/Flax; it is in fact PyTorch 2.11
(+ `neuraloperator` 2.0.0) with float64 throughout. The new families are therefore
implemented in PyTorch and reuse the parent's `train.py`, `dataset.py`,
`evaluate_cohort.py`, `timing.py` and `prepare_diagnosis_cohort.py` with the minimal
guarded changes listed in §6, so the protocol is shared by construction rather than
re-implemented. A local parity check (`check_fno_parity.py`) must reproduce the
parent's audited `fno-large` per-case errors through these files to $\le 10^{-9}$
before the first submission.

## 3. Arms

### 3.1 U-Net (job 1, `unet01`)

The PDEBench 2D baseline: four-level encoder–decoder, two 3×3 convolutions per level,
channels `base, 2·base, 4·base, 8·base`, bottleneck `16·base`, 2×2 max-pool down, 2×2
transposed convolution up with skip concatenation, GroupNorm(8 groups), GELU, 1×1
output head. The 257² grid is zero-padded to 272² (÷16) at the entry and cropped at
the exit; zero padding is the natural extension of a zero-Dirichlet field.

| Arm | `base` | Real parameters | Nearest FNO arm |
| --- | ---: | ---: | --- |
| `unet-small` | 24 | 4 368 389 | between fno-small (1.19 M) and fno-medium (5.78 M) |
| `unet-medium` | 32 (the PDEBench default) | 7 763 461 | fno-medium (5.78 M) |
| `unet-large` | 48 | 17 462 021 | fno-large (17.88 M) |
| `unet-refine` | winner at lr $3\times10^{-4}$ | as winner | fno-refine |

### 3.2 Transolver (job 2, `tsol01`)

Physics attention on a structured 2D mesh (Wu et al., ICML 2024), the
structured-mesh variant: per layer, 3×3 convolutional projections to per-head
features, softmax slice weights over 64 slices with a learned temperature, attention
among the slice tokens, broadcast back through the same weights, then a pre-norm
MLP (ratio 2). 8 layers, 8 heads, 64 slices, the paper's unified positional
encoding (distances to an 8×8 reference grid). Tokens are 4×4 patches of the grid
zero-padded to 260² (65² = 4 225 tokens, the token count regime of the paper's
structured benchmarks); the decoder is a per-token MLP to 5×4×4 values unpatched
onto the grid and cropped. A 10-epoch linear learning-rate warm-up to $10^{-3}$ is
the **one** protocol addition for this family (standard for transformers; the FNO
and U-Net use none). Otherwise identical protocol.

| Arm | `dim` | Real parameters |
| --- | ---: | ---: |
| `tsol-small` | 128 | 3 151 504 |
| `tsol-medium` | 192 | 7 041 680 |
| `tsol-large` | 256 | 12 475 024 |
| `tsol-refine` | winner at lr $3\times10^{-4}$ | as winner |

### 3.3 Precision

The FNO arm is float64/complex128. The brief allows float32 training for the new
families. Both new families therefore train in **IEEE float32 with TF32 disabled**
(`model.configure` sets `allow_tf32=False` for cuBLAS and cuDNN), with the
`Precision` wrapper keeping features, boundary mask, trajectory assembly, loss and
every reported error in float64, and predictions saved in float64. Whether this
choice matters is a pre-registered control (job 3, §3.4): the validation-selected
U-Net capacity retrained with the identical protocol in float64 at the identical
budget. Both checkpoints are saved so the `b-panel` lane can time either.

### 3.4 Controls and follow-ups (jobs 3–4, only if jobs 1–2 return)

- `unet02`: (a) the validation-selected U-Net capacity in float64, same budget
  (precision control); (b) the same capacity at a second seed (20260915), float32
  (seed-variance control). No refinement. (§A1: plus the validation-selected Transolver
  capacity in float64 if its pace allows, same budget.)
- Poisson U-Net on the Poisson operator-screen dataset of lane `no-poisson`, same
  protocol, if the Burgers arms finish early. Pre-registered as a separate §A
  amendment with its own gates before it is staged.

Job cap for this lane: 8. Retracted jobs count.

## 4. Pass/fail criteria and gates (pre-registered)

Gates that must pass for a run to be reported at all:

- G1 `jax_backend=gpu` and `torch_backend=cuda` in the log; GPU name recorded.
- G2 staged code byte-identical to the committed blob (`MANIFEST.sha256`,
  `PROVENANCE.json`, `COMMIT.txt`); data verified against the cache's `DATA.sha256`
  before training; `OUTPUTS.sha256` and `ALL-DONE` at the end.
- G3 training/validation index SHA256 identical to the FNO job's (`5333584b…`,
  `468b9e70…`), asserted by `audit.py`, and the 8-case cohort index identical to
  the FNO job's `cohort_index_sha256`.
- G4 the independent NumPy audit reproduces every reported per-case error from the
  saved prediction fields to $10^{-11}$ relative, confirms the supplied state is
  returned bitwise and the boundary is exactly zero, and confirms checkpoint hashes.
- G5 every arm is reported (all capacities, refine, controls), with parameters,
  epochs completed, best epoch, budget, and which of {early stopping, epoch cap,
  wall budget, signal} ended it. No arm is dropped for being bad.

Verdict criteria on the **32 validation cases** (the selection set) and, separately,
on the **8-case matched cohort** (never used for selection):

- V1 (baseline is competitive): the validation-selected arm of a family reaches a
  worst validation error no more than 1.5× the FNO's validation-selected worst
  (6.3825 %) **and** a median no more than 1.5× the FNO's (1.8054 %). Failing V1
  means the family is reported as *weaker than the FNO under this protocol* — it is
  still reported in full.
- V2 (family ordering on the matched cohort): the three methods' worst-case
  numbers on the 8 cases (FNO 2.4829 %, ROM 1.8671 %, FOM `same_nt1e-2_dt005`
  0.9978 %) are compared with the new family's validation-selected arm. The
  question is only *where it sits*; no threshold turns that into a win or loss,
  because the eight-case worst is a single case.
- V3 (precision control): if the float64 retrain of the selected U-Net capacity
  differs from its float32 twin by more than the seed-to-seed difference (job 3b)
  on validation worst or median, the float32 numbers are flagged as
  precision-sensitive in the report and the float64 numbers are reported alongside.

Falsification clause. If both new families fail V1 despite the budget being spent
(i.e. runs end by wall budget rather than early stopping, and the refine run does
not help), the honest conclusion is that under this exact budget and protocol the
FNO is the strongest of the screened families on this dataset — **not** that
U-Nets or Transolvers are weak in general; the report must then say the budget
bound is binding and give the epoch counts.

What would make me withdraw a number: any G-gate failure; a run whose validation
selection touched the 8-case cohort; a training/validation index hash that differs
from the FNO's; a nonfinite loss or a truncated log.

## 5. Reporting

`reports/generate_report.py` reads only `runs/*/audit.json` (this lane's NumPy
audits), the parent lane's `runs/fno_burgers02/field-audit.json` and the Burgers
lane's `checks/refinement02-diagnosis-audit.json` (for the ROM/FOM rows, labelled
"other job"), and writes `reports/2026-09-1x-no-second.md` and
`reports/summary.json` (rows: operator, capacity, params, budget_s, epochs, stop
reason, cohort, metric, value, job id, source SHA). No number is typed by hand.
Every table ends with a glossary. No speed ratio appears anywhere.

## 6. Code changes relative to the parent lane (all guarded)

- `model.py`: `family_of`, `smoke_bounded`, `make_model` dispatch on
  `config["family"]` (absent → FNO exactly as before); `check_dtypes` uses the
  model's declared `parameter_dtype` (default float64, so the FNO path is unchanged).
- `train.py`: smoke bounds via `smoke_bounded`; optional `warmup_epochs`
  (default 0 → identical behaviour); result JSON gains `family`, `parameter_dtype`,
  `stopped_by_early_stopping`, `stopped_by_epoch_cap`, `wall_budget_seconds`.
- `evaluate_cohort.py`, `timing.py`: `--pattern` (default `fno-*`, unchanged).
- New: `families.py`, `smoke_second.py`, `training_smoke_second.py`,
  `worker_second.py` (spec-driven copy of `worker_burgers_long.py`),
  `check_fno_parity.py`, `cluster/stage.py`, `cluster/collect.py`, `audit.py`,
  `reports/generate_report.py`.

## 7. Pre-submission gates (filled in before the first job)

- P1 local contract smoke (`checks/local-contract-smoke.json`): both families,
  both dtypes, both PDE contracts; f64 intermediates audited for the float64
  declaration; checkpoint bitwise parity. **Passed** (8/8 cases; the first attempt
  failed only because the smoke's own Poisson loss divided by a zero target norm —
  a smoke bug, fixed to the parent's smoke loss; no driver file changed).
- P2 local training smoke at 64 intervals through the real `train.py` for both
  families (`checks/local-training-smoke-{unet,transolver}.log`). **Passed.**
- P3 FNO parity: `checks/fno-parity.json`, `driver_max_abs_gap` $\le 10^{-9}$
  against the parent's audited `fno-large` per-case errors. **Passed**; the
  recorded gap is at f64 rounding level (see the JSON), and the NumPy-only
  recomputation of the audited numbers from the archived fields is exact.
- P4 Codex audit of this file: `reports/codex-design-audit.md`; accepted/rejected
  findings recorded in §A1.

## §A1 — Independent design audit (2026-09-17, before the first job)

Codex could not run (its sandbox failed to start, then the account hit its usage limit
until 2026-09-19; both traces are in the scratchpad and `reports/codex-design-audit.md`
records the failure). The audit was performed instead by an independent Claude
subagent given the same read-only, adversarial brief (findings reproduced verbatim in
`reports/design-audit-2026-09-17.md`). Findings and what was done with each:

| # | Severity | Finding | Action |
| --- | --- | --- | --- |
| 1 | MAJOR | Reported validation errors came from a batch-8 pass but the saved fields from a batch-1 pass; batched float32 kernels are not batch-invariant, so gate G4 (1e-11) would trip or be loosened post hoc. | **Accepted.** `train.py` now derives the reported errors and the saved fields from one batch-1 pass. The batch-8 selection score is retained as `batched_selection_score`; the audit asserts it equals the history entry and that the batch-1 vs batch-8 gap is ≤ 1e-5 relative (float32) / 1e-11 (float64) — pre-registered here. |
| 2 | MAJOR | `run.sbatch` no longer `exec`s the worker, so Slurm's USR1 never reached it. | **Accepted.** The worker runs in the background with a `trap` forwarding USR1, then `wait`. |
| 3 | MAJOR | `audit.py` could pass with an arm skipped or the cohort/timing missing. | **Accepted.** It now asserts the arm set equals the spec's (plus refine), the cohort and timing blocks are present for every arm, no signal stop, and the protocol constants (epochs 4000, patience 250, batch 8, wd 1e-4, lr ∈ {1e-3, 3e-4}, seed ∈ {20260914, 20260915}). |
| 4 | MAJOR | "Validation-selected arm" ambiguous about `refine`; the FNO reference was hard-coded. | **Accepted.** Rule: the validation-selected arm of a family is the argmin of the validation **mean case-maximum** error over every complete arm **including `refine`** (the worker's own selection score). The same rule is applied in code to the FNO audit to pick the FNO reference for V1; with the FNO's audited numbers that is `fno-large` (mean 2.2811 %), whose worst/median (6.3825 % / 1.8054 %) V1 quotes. |
| 5 | MINOR | Protocol constants recorded but not asserted. | Accepted (see 3). |
| 6 | MINOR | "Zero padding is the natural extension" is wrong as implemented: features are normalised, so the ring is 0 rather than the normalised boundary value. | **Accepted as wording**; behaviour unchanged (the ring is cropped and masked), documented in `families.py`. |
| 7 | MINOR | Transolver deviates from upstream: temperature clamp min 0.01 vs 0.1; two-layer MLP decoder vs LayerNorm + Linear; patch tokens vs per-node tokens; placeholder added unconditionally. | **Accepted**: clamp set to upstream 0.1; decoder set to upstream LayerNorm + one Linear (parameter counts become 3 108 240 / 6 952 208 / 12 322 960, superseding §3.2); described as "patchified" everywhere. Placeholder left (harmless). |
| 8 | NOTE | U-Net norm/activation differ from PDEBench (BatchNorm/tanh). | **Accepted as labelling**: "PDEBench topology with GroupNorm/GELU", which is what the brief asked for. |
| 12 | MINOR | Stop flags not mutually exclusive; 4000-epoch cap reachable by a fast float32 arm. | **Accepted.** `train.py` captures the loop-exit reason at the `break`. Reading rule: an arm ended by the epoch cap or by early stopping *below* budget is reported as such and is **not** read as budget-bound; the falsification clause's "budget is binding" applies only to wall-budget stops. |
| 13 | MINOR | No `a100-80G` constraint. | **Accepted**; added for A100 submissions, as the parent had. |
| 15 | MINOR | ROM/FOM rows read from another worktree by absolute path. | **Accepted**; hash-pinned copy at `checks/refinement02-diagnosis-audit.json` (SHA256 `ffa77d1b…`), asserted by the generator. |
| 17 | — | Fairness disclosures a reviewer will raise. | **Accepted, disclosed here**: (a) the Transolver capacities span 3.1–12.3 M, narrower than the FNO's 1.2–17.9 M; (b) lr/wd/patience/plateau were chosen for the FNO and are not re-tuned per family — `refine` is the only family-level tuning, and the Transolver paper's own recipe (OneCycle, wd 1e-5) is not used; (c) the precision and seed controls are pre-registered for the U-Net only; a Transolver float64 control is added to job 3 **if** its selected capacity's epoch pace allows (§3.4 amended accordingly); (d) an equal wall budget in float32 gives the new families more epochs than the float64 FNO had — favourable to them, and stated in the report; (e) an arm that early-stops long before the budget is read per finding 12. |
| 9–11, 14, 16, 18 | NOTE | Padding/patch arithmetic verified analytically; precision handling sound; warm-up/plateau interaction benign (the Transolver `refine` inherits the 10-epoch warm-up to 3e-4 — noted); directory/cache/CPU hazards handled; complete diff list confirms the shared protocol. | No action; `checks/fno-parity.json` re-run after these edits. |

Rejected: none. Every code change above is confined to `train.py` (final-pass errors,
stop reason), `families.py` (upstream clamp/decoder, docstrings), `audit.py`,
`cluster/stage.py` and `reports/generate_report.py`; the loss, optimiser, scheduler,
selection rule, budgets and metric are untouched, and the smokes and the FNO parity
check are re-run on the amended files before staging.

## §A2 — Poisson U-Net (job 4, `pois01`), pre-registered 2026-09-17 before submission

The brief's item 5, taken now because both Burgers jobs are running and the Poisson FNO
runs were short (all three early-stopped within 27 min).

- **Data.** The Poisson FNO job `3702464` dataset (128 train / 32 validation cases, 256
  intervals; source field → zero-Dirichlet solution; training target = declared discrete
  FD/DST solution; evaluation-only 2048-interval physical-reference sidecars). No cluster
  copy survives (both `no_poisson_20260914` and `no_audit_20260914` were cleaned), so the
  files were **re-uploaded from the Git-archived job** (archive SHA256 `b9d14a87…`, parts
  Git-tracked in the parent lane) to `no_second_20260917/poisson-data01/`. This is the one
  deviation from "data is never staged from this machine"; it is a byte-exact copy of the
  cluster-generated originals, `DATA-tv.sha256` (all 206 train/validation files) verified on
  the cluster after upload and again in the sbatch preamble, every case re-verified against
  its index hash by `dataset.load_index`, and the audit asserts the index SHA256 equal the
  FNO job's (`d20a5994…` / `65cc277b…`).
- **Protocol = the Poisson FNO's**, not the Burgers one: 500-epoch cap, patience 80,
  **7200 s wall per capacity**, AdamW lr 1e-3, wd 1e-4, batch 8, plateau scheduler, seed
  20260914, checkpoint selected on validation mean error against the discrete target.
  **No refinement run** (the Poisson FNO had none). U-Net capacities base 24/32/48 in
  float32 as for Burgers; loss = squared relative error against the discrete target.
- **Metrics** (the parent audit's two): discrete (vs training target) and physical candidate
  (vs the refinement sidecar), whole-field discrepancy over the field norm, recomputed with
  NumPy from the saved fields. Reported: worst, median, mean, p95, cases > 5 %.
- **Criterion V1-P**: the validation-selected U-Net (argmin validation mean discrete error)
  is compared with the FNO's validation-selected run under the same rule (`fno-large`, mean
  3.4504 %, median 2.0253 %, worst 29.3747 % physical candidate). Within 1.5× on worst and
  median = "competitive"; the worst is dominated by four >5 % cases for every FNO capacity,
  so the median and the >5 % count are reported alongside. No ROM/DST cohort here.
- Gates G1–G5 apply (G3 with the Poisson hashes; no cohort). One job, `03:30:00` on an
  A100-80G; global 12 000 s, reserve 600 s.

## §A3 — pois01 preamble failure (2026-09-17)

Job `3780224` (`pois01`) failed 51 s in, after both smokes passed, because the stager's
explicit file list omitted `configs/unet-poisson/*.json`. No training ran; nothing to
retract. `cluster/stage.py` now stages every file under `configs/`; the Poisson arm is
resubmitted unchanged as attempt `pois02`. Logs preserved in `runs/pois01/logs-failed/`.

## §A4 — Job-3 control decision (2026-09-17, after the three screens returned)

**Decision: submit it.** The screens did not merely show the new families to be
"competitive" (criterion V1); they put both **below** the FNO on validation mean and
median, and put the U-Net below the ROM on the matched eight-case worst. That turns this
lane's output from a defensive baseline into a positive, single-seed claim, and a positive
single-seed claim is exactly what the pre-registered seed control exists to defend. The two
objections a reviewer will raise first are now (a) "one seed", and (b) "your U-Net trained
in float32 while the FNO trained in float64, so the comparison is not like-for-like" — the
control job answers both directly, at one job of the five still available (cap 8, three
counted so far), eight days before the deadline, on a 3.3 h allocation.

Arms (generated by `make_control_spec.py` from the audited selections, no hand-typed
capacity): the validation-selected U-Net capacity in float64; the same capacity at seed
20260915 in float32; the validation-selected Transolver capacity in float64. Same 3000 s
budget, same protocol, no refinement. Reading rule, pre-registered here: if the float64
twin and the second seed both land within the screen's own capacity-to-capacity spread,
the float32 single-seed numbers stand as reported; if either moves the selected arm's
validation worst or median by more than that spread, the report flags the headline numbers
as precision- or seed-sensitive and quotes the control beside them.

## §A5 — The resolution knob: does a trained operator expose an accuracy–cost family? (job 5, `res01`)

Pre-registered 2026-09-17, before the job is staged, at the coordinator's request.

**Question.** The paper contrasts "a trained operator gives one accuracy–cost point" with "our
decoder gives a family". The obvious objection is that an operator has an inference-time knob
too: the grid it is evaluated on. An FNO is discretisation-invariant by construction and a
convolutional U-Net also runs at other grid sizes. So: **evaluated at coarser grids, does a
trained operator trace a usable accuracy–cost curve, or a degenerate one?**

**Arms.** The validation-selected checkpoint of each family, by the §A1 rule, frozen and
unmodified: `fno-large` (parent lane, job `3710846`), `unet-refine` (job `3780138`) and
`tsol-refine` (job `3780139`). The Transolver is beyond what was asked for and is included
because it costs nothing extra in the same job and three families answer the question more
firmly than two. Checkpoints are inputs here, uploaded to the namespace with a recorded
`CHECKPOINTS.sha256` and verified in-job against the SHA256 already recorded in the audits;
**no training happens in this job.**

**Resolutions.** 256, 128, 64 and 32 intervals (257², 129², 65², 33² nodes). Each divides 256,
so the input restriction is exact nested sampling and the output prolongation is exact at
nested nodes.

**Grading — identical to the Burgers lane's own coarse-mesh FOM arms** (`coarse_half_dt005`,
`coarse_quarter_dt01`), which is the precedent that makes this comparable:
`engines.make_fom(..., target=L)` restricts the supplied field by stride, solves on the coarse
grid, and returns `engines.output_field`, an **aligned bilinear prolongation** back to the
257² grid, where it is scored against the same reference. This lane does the same: restrict
the supplied 257² field by stride → run the operator at that grid → prolong every output time
back to 257² by the same aligned bilinear map → score with the same fixed-initial metric
against the same reference on 257². The prolongation is re-implemented here and checked
against `engines.output_field` to $\le 10^{-12}$ before use.

Per LANE-PROTOCOL rule 9, three error numbers are reported separately at every rung:
**worst-over-evolved-times** ($t>0$), **worst-over-all-times**, and the **$t=0$ term** on its
own. At a coarse rung $t=0$ is no longer exact — prolonging the restricted initial field is
lossy — and that loss is a genuine cost of this knob, so it must be visible rather than
folded into one number.

**Cost.** Same-job device-query timing per (model, resolution) with the parent lane's
`timing.py` protocol: 20-query burn-in per block, `torch.cuda.synchronize()` around every
repetition, 30 repetitions per case, every repetition retained. Ratios **within one model's
own curve, measured in this one job on this one GPU**, are same-job controls and are
admissible; no ratio is formed against any other job, model-family checkpoint timing from
another allocation, the ROM or the FOM.

**The control that decides what a degradation means.** `interp-floor@r`: take the *reference
itself*, restrict it to grid $r$ and prolong it back, then score it by the same metric. This
is the error of a **perfect** operator at that rung — pure discretisation and interpolation
loss. Reported beside every rung. If an operator's coarse error tracks the floor, the grid is
the limit and the knob is real; if it is far above the floor, the model itself is breaking
off-resolution.

**Pre-registered criterion, per operator.**

- **R-USABLE**: some rung below 256 reaches a same-job median device-query speedup $\ge 1.5\times$
  over that same model's own 256 evaluation **while** its worst-over-evolved-times error stays
  $\le 2\times$ that model's own worst-over-evolved-times error at 256.
- **R-DEGENERATE**: the first rung below 256 (128) already fails either half — more than 2× the
  error, or less than 1.5× the speedup.
- The full curve is reported for every operator and every rung regardless of the label, with
  the interpolation floor beside it.

**Falsification clause — this is the point of the experiment.** If any operator is R-USABLE,
then "a trained operator gives one accuracy–cost point" is **false as stated** and this lane
will say so plainly; the paper's contribution must then narrow honestly to the mechanism and
the structural condition rather than to the existence of a family. If every operator is
R-DEGENERATE, that is support for the framing, stated as what it is: evidence from three
families, one mesh ladder, one PDE and one seed, not a theorem about neural operators. A
result in either direction is reported with equal prominence, and neither outcome licenses a
speed claim against the ROM or the FOM.

**Gates.** G1–G4 as before (G3: the same validation-32 and matched-8 cases and hashes; no
training, so no arm-set or budget gate). Added: the prolongation check above; the assertion
that each loaded checkpoint's SHA256 equals the one recorded in the corresponding audit; and
the assertion that at rung 256 the reproduced errors equal the already-audited numbers for
that checkpoint to $\le 10^{-9}$ — i.e. the ladder's top rung must reproduce this lane's own
published result, or the job is void.

## §A6 — Report audit (2026-09-17, after the three screens) and corrections

Codex remained unavailable (quota until 2026-09-19), so the finished report was audited
against the raw JSONs by an independent Claude subagent with an adversarial read-only brief;
findings in `reports/report-audit-2026-09-17.md`. It recomputed every rendered cell, and
re-derived three arms end-to-end from the raw `.npz` fields without touching `audit.json`.

**Headline: no BLOCKER and no wrong number.** Every figure in every table was traceable and
correct, the selection rule and the V1 arithmetic were right, no forbidden cross-job ratio
appeared, and gate G5 held (every arm reported, including the bad ones). Accepted findings and
what changed — all in the generator, so the fixes are reproducible rather than edited prose:

| # | Severity | Finding | Action |
| --- | --- | --- | --- |
| 1 | MAJOR | The report referred to a "control below" that did not exist; `ctrl01` was submitted one minute before the report was generated. | **Accepted.** Every controls reference is now conditional on a control audit existing. |
| 2 | MAJOR | "The report says which, per arm, in the 'Ended by' column" was false — the column read `wall budget` for all eight arms and distinguished nothing, while 7 of 8 arms were in fact still improving. The "lower bound" caveat was given only to Poisson. | **Accepted, and it is the most substantive fix.** A generated **"Still improving?"** column now marks every arm by the same last-5%-of-epochs test, for this lane **and the FNO**, and a fairness bullet states that 7/8 lane arms and 4/4 FNO arms were still improving, so every Burgers number is a lower bound. |
| 3 | MAJOR | The pre-registered Poisson criterion V1-P (§A2) was never evaluated and no Poisson arm was named as selected — a favourable pre-registered verdict was simply missing. | **Accepted.** V1-P is now computed and stated: selected `unet-medium`, worst ratio 0.20×, median ratio 0.83× against the 1.50× bar, **pass**. |
| 4 | MAJOR | §A1 finding 17(d) required disclosing that equal wall in float32 buys more epochs than float64, and the report gave no FNO epoch or parameter counts at all. | **Accepted.** FNO rows (capacity, parameters, epochs, best epoch, still-improving, stop reason) are now in the capacities table, and a fairness bullet states the like-for-like selected-arm ratios (2.84× and 2.35× the FNO's epochs). |
| 5 | MAJOR | The timing table pooled two jobs under a "Same-job" heading, inviting exactly the cross-job read the caption forbade. | **Accepted.** One table per job, each captioned "these rows are mutually comparable", and the preamble says the screens' timings may not be compared with each other either. |
| 6 | MAJOR | "The Poisson FNO early-stopped inside the same cap" was a hard-coded string with no source, and the Poisson FNO's parameters/budget/stop reason were typed into the generator — contradicting "no number here is typed", on the very claim that makes the U-Net's Poisson result a lower bound relative to a converged FNO. | **Accepted.** `checks/fno-poisson-run-metadata.json` was extracted from that job's own Git archive (hash-pinned to it) and the contrast is now generated from it with epochs, best epochs, cap and patience quoted. |
| 7, 8 | MINOR | The verdict's parenthetical put the FNO's cohort *worst* after a worst and a median (on the 8-case median the FNO beats both new families), and the flattering clause led while `tsol-refine`'s tail is 1.46× the FNO's. | **Accepted.** The verdict now quotes FNO worst **and** median, and states both V1 ratios numerically. |
| 9, 10 | MINOR | The Transolver sweep is unbracketed (its best arm is the smallest screened), and §A1 17(a)/17(b) fairness caveats appeared nowhere in the report; winner margins are inside plausible seed noise. | **Accepted.** The fairness block states the capacity ranges, that nothing was tuned per family, that a best arm at the edge of its range may have its optimum outside it, and that family differences are not yet separated from seed variation. |
| 11 | MINOR | §A2 said "all 206 train/validation files"; the job verified **196**. | **Corrected here:** the number is 196 (`data_verified=196`; 128 train + 32 validation + 32 physical-reference + 4 index files). §A2's text stands as written, per the append-only rule; this row is the correction. |
| 12 | MINOR | The metric divides by the *initial* norm, so late-time percentages read smaller than per-time relative errors, and the max includes a $t_0$ term that is identically zero. | **Accepted.** Both are now stated under "How accuracy is defined", with the generated mean final/initial norm ratio. |
| 13 | MINOR | The reference was described more strongly than its own record. | **Accepted.** The report now quotes the lane's own wording and its worst empirical margin. |
| 14 | MINOR | The two Poisson metric columns are not independent evidence. | **Accepted.** Stated, with the generated target-vs-reference agreement, and noted as the check that rules out the known analytic-data inconsistency here. |
| 15 | MINOR | Three different commits were printed but never flagged as different builds. | **Accepted.** Stated in the preamble. |
| 17 | NOTE | `audit.py` asserts the split hashes against **literals typed into the file**; only the cohort hash is read programmatically. | **Accepted.** The audit now derives the expected train/validation hashes from the archived `DATA.sha256` manifest and cross-checks the literals against it, so a typo in either cannot pass. |

Rejected: none. Findings 16, 18, 19 and 20 were confirmations requiring no change; the auditor
also verified independently that the split hashes really are the FNO jobs' by streaming the
parent lane's 1.5 GB archives, and that the batch-1/batch-8 tolerance from §A1 was not
stretched after seeing data (measured gaps sit 4–1700× inside the pre-registered bar).

## §A7 — res01 preamble failure and the staging guard (2026-09-17)

Job `3783920` (`res01`) failed 59 s in on pax050, exit 1:0. Every preflight and smoke step
passed (`jax_backend=gpu`, 395 data files and 3/3 checkpoints hash-verified, resolution smoke
on all rungs, prolongation check 0.0, cohort rebuilt), and then `resolution.py` died on
`FileNotFoundError: [Errno 2] No such file or directory: 'code/checkpoints.json'`. Cause:
the stager's explicit file list omitted `checkpoints.json`, the frozen-checkpoint manifest
§A5 introduced — the same omission class as §A3. **Nothing was evaluated; nothing to
retract; zero science GPU time**, so it does not count toward the cap (rule 13).

Change: `cluster/stage.py` stages `checkpoints.json` and now runs `check_references()` at
staging time, which resolves every `code/<path>` literal in every staged `.py`/`.json`
against the staged tree and JSON-parses every staged JSON; run against the old `res01` tree
it fires on `code/checkpoints.json`. The science of §A5 is unchanged: `res02` is staged from
`specs/res02.json`, identical to `res01.json` except `job_name`. Record:
`runs/res01/FAILURE.json`, `runs/res01/FAILED.md`, `runs/res01/logs-failed/`.

Self-audit substitution (Codex quota until 2026-09-19 11:33): this amendment and the record
above were checked by the author against the pulled logs; the verbatim error line in
`FAILURE.json` is copied programmatically from `logs-failed/resolution.log`.

## §A8 — res02 result: the §A5 falsification clause fires for the FNO (2026-09-17)

Job `3787189` (`res02`, pax049, A100-80G, COMPLETED 0:0 in 2 m 14 s — evaluation only, no
training, so a short run is expected). `jax_backend=gpu`, `torch_backend=cuda`, 395 data files
and 3/3 checkpoints hash-verified in-job, resolution smoke on every rung for all three
families, prolongation check against `engines.output_field` = 0.0, cohort rebuilt 8/8,
`ALL-DONE`. Collected with checksums verified on both sides, audited by
`audit_resolution.py` (independent NumPy recomputation of every error from the saved
prolonged fields, every floor from the references, every timing median from the retained
30×8 repetition arrays), archived as 31 Git parts, remote directory deleted. The `ckpt01`
checkpoint upload was deleted with it: `res02` was its only consumer and all three
checkpoints are in Git archives (this lane's `unet01`/`tsol01`, the parent lane's
`fno_burgers02`, and the `res02` archive itself).

**Gates.** Top-rung gate: at 256 intervals every checkpoint reproduces its published
worst-over-evolved and worst-over-all validation numbers with gap **0.0** (`fno-large` 6.3825 %,
`unet-refine` 7.5176 %, `tsol-refine` 9.3183 %; tolerance 1e-9). Interpolation floor,
validation-32, worst over evolved times: 0.2255 % @128, 0.9233 % @64, 4.0157 % @32.

**Labels, applied literally as pre-registered (validation-32, same job, same GPU, within each
model's own curve).**

| checkpoint | rung | worst evolved | error ratio vs own 256 | median device query | speedup vs own 256 | error gate ≤2× | speed gate ≥1.5× |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `fno-large` | 128 | 7.2993 % | 1.14× | 3.463 ms | 2.12× | pass | pass |
| `fno-large` | 64 | 9.8214 % | 1.54× | 3.438 ms | 2.14× | pass | pass |
| `fno-large` | 32 | 13.8510 % | 2.17× | 3.326 ms | 2.21× | fail | pass |
| `unet-refine` | 128 | 38.1348 % | 5.07× | 2.347 ms | 2.51× | fail | pass |
| `tsol-refine` | 128 | 15.2905 % | 1.64× | 8.151 ms | 1.37× | pass | fail |

`fno-large` is **R-USABLE** (rungs 128 and 64); `unet-refine` and `tsol-refine` are
**R-DEGENERATE** at the first rung below 256. (The numbers above are copied from
`runs/res02/audit.json` for the record; the report and `summary.json` are generated from that
file and are authoritative.)

**Consequence — binding, as written in §A5.** The clause fires. The abstract's opening
sentence, "A learned PDE surrogate usually gives one accuracy–cost point; getting another
means retraining", is false as stated and is withdrawn, together with its restatements (the
introduction's existence framing of tunability and the glossary line that each neural operator
"is one trained model giving one accuracy–cost point"). The replacement is the narrower claim
the reviewer suggested and the report states: the correction rank $q$ is a run-time control
with a structural meaning — it moves the reachable set from the head's image to the bank's
span, nested trial manifolds of the same weak-residual solve, error monotone in $q$ with the
interpolation floor at zero at every rung — which resolution scaling of an operator is not.
What the operator's curve actually looks like is also on record, generated from the rows:
one step (256→128) buys the whole speedup (2.12×) at 1.14× error, and below 128 the query is
launch-bound (3.46 → 3.44 → 3.33 ms) while the error rises to 1.54× and 2.17×, tens of times
above the interpolation floor. Three families, one ladder, one PDE, one seed — evidence about
these checkpoints, not a theorem; and neither outcome licenses any speed claim against the
ROM or the FOM.

**Code changes in this session (all committed, none touching the science).**
- `audit_resolution.py`: timing entries now also carry p05/p95/mean re-derived from the
  retained samples and the timed-region definition. No gate changed.
- `reports/generate_report.py`: (i) `load_attempts` skipped nothing that lacked `spec` and so
  crashed on the resolution audit — a latent bug that could not show before `res02` landed;
  resolution audits (no `arms`) are now skipped there. (ii) The known `summary.json` defect:
  operator metadata keyed by `arm` alone collided across the Burgers FNO job `3710846` and the
  Poisson FNO job `3702464` (`fno-large` etc. in both). Every row now carries `pde`, an explicit
  `key = arm|job_id` and, where epochs are known, `still_improving`; the `(key, cohort, metric)`
  triple is asserted unique, and the Burgers-FNO still-improving count is asserted to equal the
  number of FNO arms (4 of 4, as the report states). Poisson-FNO rows now take epochs, best
  epoch and stop reason from the pinned `checks/fno-poisson-run-metadata.json` instead of
  `None`. (iii) The resolution section reports both cohorts (validation-32 and diagnosis-8),
  worst and median, the floor on each, the p05–p95 timing spread, per-rung gate flags, a
  generated curve-shape sentence per operator, and the withdrawal text above.

**Self-audit substitution (Codex quota until 2026-09-19 11:33).** `reports/self-audit-resolution.md`
lists each claim of the resolution section, the JSON field it rests on and the check run,
including a spot-check that recomputes the decisive `fno-large`/`unet-refine`/`tsol-refine`
128-vs-256 ratios directly from the archived `.npz` fields and timing samples without importing
`audit_resolution.py`. Codex should re-audit the report against the raw JSONs after
2026-09-19 11:33 if the lane is still open.

**Job count.** Counted: `unet01`, `tsol01`, `pois02`, `ctrl01`, `res02` = 5 of 8; `pois01` and
`res01` were preamble deaths with zero science GPU time and do not count (rule 13). `ctrl01`
(`3783831`) is still running and stays pending.
