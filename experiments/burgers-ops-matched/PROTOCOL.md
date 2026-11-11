# Protocol: neural operators trained on the ROM's own training information (2D viscous Burgers)

Written 2026-11-11 **before any training or test outcome of this lane exists**. Later changes are
appended under "Amendments" with date and reason; nothing above that heading is edited afterwards.

Lane `exp/2026-11-11-burgers-ops-matched`, forked from `822f27166` (`exp/2026-09-24-operators-all-pdes`,
the commit that holds the paper's Burgers 2D operator harness: `experiments/ops-all/g2d/`).
Cluster namespace `/cluster/tufts/paralab/tawal01/rebuttal-burgers-ops/`.

## 1. Question

The paper's operator baselines at $256^2$ were trained on 128 trajectories under a 3000 s wall cap,
while the ROM's bank saw 576 trajectories and its head 4,608. Does the ROM's accuracy advantage on the
64-case test cohort survive when the operators are trained on the same information, to convergence?

## 2. What "the same information" means here

**Parameter vectors (identical to the ROM's).** The ROM's training parameters are
`P = concat(params_draw(0, 576), params_draw(1000, 4032))` (bank: the first 576, job 2835788;
head: all 4,608, job 2837431; `params_draw` is the paper's sampler: centre $c_x,c_y\sim U(0.15,0.85)$,
width $\sim U(0.05,0.20)$, amplitude $\sim U(0.5,2)$, $\nu$ log-uniform on $[0.01,0.1]$).
The operators get exactly these vectors, in this order. Data ladder (nested prefixes of `P`):

| rung | trajectories | what it corresponds to |
|---|---|---|
| 128 | `P[0:128]` | the paper's operator training-set size (reproduction check) |
| 576 | `P[0:576]` | exactly the ROM bank's trajectories |
| 4608 | `P[0:4608]` | exactly the ROM head's trajectories |

**Generator and mesh (identical to the test reference).** Each trajectory is solved at 256 intervals
($257^2$ nodes), backward Euler, $\Delta t = 0.005$, first-order upwind advection, horizon $T=0.25$ —
the discretisation the ROM's training data used — with the solver the test reference uses
(`fft_tight`: Newton–BiCGStab, FFT Helmholtz preconditioner, Newton tolerance $10^{-6}$, linear $10^{-8}$).
The ROM's training set was produced by the same discretisation with a direct linear solve; the two agree to
solver tolerance (declared, not bitwise).

**Targets and horizon.** Input: the initial field $u_0$ on the $257^2$ grid and the viscosity $\nu$
(no Gaussian descriptors). Output: the five evolved fields at $t=0.05,0.10,0.15,0.20,0.25$, the times the
test metric scores. Same contract as the paper's operator rows.

**Declared differences that cannot be removed.**
1. *Saved states.* The ROM saw all 51 time levels of each trajectory (235,008 states; its head fit used
   131,072 of them, its bank 16,384 states of the 576). An operator that maps $u_0\mapsto$ five output times
   is supervised on 5 evolved fields per trajectory (23,040 fields at the 4,608 rung). The trajectories and
   their parameter vectors are the same; the intermediate time levels are not used by the operators.
2. *Validation is carved out of the pool.* The operators need a validation set for model selection and the
   brief requires it to come from the training pool, so at each rung the **last** cases of the rung are
   held out for validation and not trained on: 128 → 112 train + 16 validation; 576 → 512 + 64;
   4,608 → 4,352 + 256. The operators therefore train on slightly *fewer* trajectories than the ROM saw,
   never more, and never on anything outside `P`.
3. *Training targets changed relative to the paper.* In the paper's $256^2$ operator rows the training
   targets were 4,096-interval solutions restricted to the $257^2$ grid, whereas the test reference is the
   same-mesh 256-interval solve (the $256^2$ grid differs from the fine solution by about 4 % in this
   norm). Here the operators are trained on the same-mesh 256-interval solutions, i.e. the quantity the
   test scores and the quantity the ROM was trained on. So the 128 rung here is **not** a bitwise
   reproduction of the paper's row: it differs in targets (same-mesh), in validation (16 carved cases
   instead of 32 separate ones), in the stopping rule (convergence instead of 3000 s), and in FNO
   precision (item 4). The difference from the paper's numbers at this rung is reported as such.
4. *Precision.* All networks are float32 (the paper's U-Net and Transolver were float32; its FNO was
   float64, which costs about 5x per step and would not reach convergence at 4,608 cases in the time
   available). Features, loss and all reported errors of the final evaluation are computed in float64.
   TF32 is disabled.

## 3. Test cohort and metric (unchanged from the paper)

- test64 = `params_draw(20260916, 64)`; physical sha256 must equal `cd058fd2a297c202…` (asserted).
- Reference: `fft_tight` on the same mesh (256 intervals; 1,024 intervals for the $1024^2$ evaluation).
- Error of one case: $\max_{k=1..5}\ \lVert u(t_k)-u_{\rm ref}(t_k)\rVert_2/\lVert u_0\rVert_2$.
  Reported: worst and median over the 64 cases, in percent.
- The test cohort lives in its own directory, is never passed to a training job, and is evaluated
  **once, at the end, after all training and all selection is finished**, for every arm in one run.
  All selection uses validation only. Test results of non-selected arms are reported too (as a table of
  everything that was run), but the headline arm of each family is the validation-selected one.

## 4. Training (fixed before any run)

Common: AdamW, weight decay $10^{-4}$, batch 8, gradient clip 1.0, loss = mean squared relative error
over cases and the five evolved times (the paper's loss), best-validation checkpoint, selection score =
validation mean over cases of the per-case max-over-time relative error (the paper's score).

**No wall cap; schedule and stopping are counted in gradient steps, identical at every rung**
(an epoch is 14 steps at the 128 rung and 544 at the 4,608 rung, so epoch-counted patience would give the
rungs different schedules):
- validation every 500 steps;
- learning rate: `ReduceLROnPlateau`, factor 0.5, patience 8 validations (4,000 steps), relative
  improvement threshold $10^{-3}$, floor $10^{-6}$;
- early stopping: no relative improvement $>10^{-3}$ of the best validation score for 30 validations
  (15,000 steps);
- hard cap 400,000 steps. If a run is ended by the cap or by the job time limit instead of early
  stopping it is flagged "not converged" in every table, with its learning rate and the slope of its
  validation curve at the end.

**Tuning grid** (three configurations per family; informed only by an earlier validation-only screen on
128 cases, `opstune_grid_20260922/grid01`, never by any test number):

| arm | family | configuration | initial lr |
|---|---|---|---|
| F0 | FNO | width 64, 32 modes, 4 layers (the paper's architecture) | $10^{-3}$ |
| F1 | FNO | width 64, 48 modes, 4 layers | $10^{-3}$ |
| F2 | FNO | width 96, 32 modes, 4 layers | $10^{-3}$ |
| U0 | U-Net | base 32, 4 levels, GroupNorm 8 (the paper's architecture) | $3\times10^{-4}$ |
| U1 | U-Net | base 64 | $3\times10^{-4}$ |
| U2 | U-Net | base 32 | $10^{-3}$ |
| T0 | Transolver (only if a GPU is free) | dim 128, 8 layers, 8 heads, 64 slices, patch 4 (the paper's) | $3\times10^{-4}$, 5,000 warm-up steps |

Every grid arm is run at every rung with seed 0. **Selection:** per family and per rung, the arm with the
lowest validation score. **Seeds:** at the 4,608 rung the paper-architecture arms F0 and U0 are run with
three seeds (0, 1, 2) from the start, in parallel with the grid (waiting for the selection before starting
seeds would not fit the time). If the selected arm is not F0/U0, extra seeds of the selected arm are run
if time allows; otherwise it is reported single-seed and that is stated. Seed spread is reported as
min / median / max.

## 5. Evaluation at the end

1. $256^2$ test: every arm's best-validation checkpoint, float64 metric, batch 1.
2. Inference time: one A100 80GB (the ROM row's GPU type), batch 1; the timed region starts with $u_0$ and
   $\nu$ on the GPU and ends with the six fields on the GPU (feature construction, forward pass, masking and
   assembly inside), `torch.cuda.synchronize()` around each repetition, 20 untimed warm-up queries, then
   5 repetitions for each of the 64 cases; median, 5th/95th percentile, min and max over the 320 samples.
3. $1024^2$ (models trained at $256^2$, no retraining), against the 1,024-interval `fft_tight` reference
   of the same 64 cases, two routes, both reported for every family:
   (a) *zero-shot*: the network is applied directly to the $1025^2$ input;
   (b) *coarse-then-interpolate*: the input is injected to $257^2$ (every 4th node), the network is
   applied there, and the output is interpolated bilinearly to $1025^2$.
   Training at $1024^2$ on 4,608 cases is a stretch goal and is reported only if it converges.

## 6. Comparison and verdict rule (fixed now)

ROM rows are the paper's, same cohort, same metric, A100 80GB (lane `2026-09-25-burgers2d-test`,
`reports/summary.json`): $256^2$ accurate ($R'=384$) worst 0.87 %, median 0.026 %, 45.2 ms; fast
($R'=96$) worst 3.85 %, median 0.529 %, 14.2 ms. $1024^2$: accurate 1.25 % / 0.029 % / 49.1 ms;
fast 4.25 % / 0.558 % / 19.1 ms.

For each family, with its validation-selected arm at the 4,608 rung (median seed where seeds exist):
- the ROM's accuracy advantage **survives** against that family if the operator's worst-case test error
  is above the ROM's accurate setting (0.87 % at $256^2$);
- it is **lost** if the operator's worst-case error is at or below 0.87 %;
- independently, the operator **dominates the ROM's fast setting** if it is both more accurate than
  3.85 % worst-case and faster than 14.2 ms.
The median error is reported beside the worst-case one and the same three statements are made for it.
The answer is reported whichever way it goes, together with the data-ladder trend (128 → 576 → 4,608).

## Amendments

(none yet)

### 2026-11-11, A1–A4 — made after a 1,000-step speed smoke (job 7116986, 128 rung), before any training outcome and before the test cohort was touched

Reason for all four: the smoke measured the real cost per gradient step on an A100 80GB (FNO F0 93 ms, F2 145 ms,
U-Net base 32 53 ms, base 64 156 ms) and the parameter counts; the section-4 plan as written would need about
twice the GPU time available in this push.

- **A1 (correction).** The paper's U-Net has 17,462,021 parameters, which is `base = 48`, not 32 (base 32 has
  7,763,461). The U-Net arms are therefore U0 = base 48, lr $3\times10^{-4}$ (the paper's architecture);
  U1 = base 64, lr $3\times10^{-4}$; U2 = base 48, lr $10^{-3}$. (F0 has 17,877,317 parameters, the paper's count.)
- **A2 (schedule).** Plateau patience 6 validations (3,000 steps) instead of 8; learning-rate floor $10^{-5}$ (the
  paper harness's floor) instead of $10^{-6}$; early stopping after 20 validations (10,000 steps) without a relative
  improvement $>10^{-3}$ instead of 30. Still counted in steps and identical at every rung. Each run also carries a
  4.6 h wall safety budget; a run ended by it is flagged "not converged" exactly as section 4 says.
- **A3 (grid only where it matters).** The three-arm grid is run at the 4,608 rung only. The 128 and 576 rungs are
  run with the paper architectures F0 and U0 only (seed 0), so the data-ladder trend is reported at fixed
  architecture. Seeds 0, 1, 2 of F0 and U0 at 4,608 as before.
- **A4 (hardware).** The H100 node fails the mandatory JAX preflight (job 7116898), so it is excluded. Training runs
  share GPUs (two to four runs per GPU), so training wall time is not a cost measurement; steps and epochs are
  reported. Inference timing is unaffected (separate job, one process, A100 80GB).

### 2026-11-11 (about 11:15 EST), A5–A7 — made after the training outcomes (validation only) were collected, before any selection was frozen and before the test cohort was opened

What was seen when these were written: the validation histories and result files of the 14 training runs
(`inputs/runs/`, `reports/tables.md` Table 2 at commit `a782458e1`). Nothing of the test cohort: no job has
read `data/test` except its generator (job 7116901), and no evaluation of it exists.

- **A5 (one continuation round, because the first round did not converge).** 11 of the 14 runs were ended by the
  4.6 h safety budget of A2, not by the stopping rule (two to four runs shared each GPU, and five of the six jobs
  landed on L40S cards, slower than the A100 the budget was sized for). Several were still improving quickly at
  the cut (best validation score over the last 10 validations: F2 at 4,608 −15 %, F0 seed 1 −6 %, F0 at 576 −6 %),
  and for 7 of the 11 the best checkpoint is the last validation. Section 1 asks about operators trained *to
  convergence*; reporting these as they stand would under-train the comparators, which is the reviewers' complaint.
  Therefore every run that ended with `time_limit` is continued once:
  - start from the run's best-validation checkpoint (`best.pt`; the only state the trainer saved), at that
    checkpoint's step count, with the learning rate the run ended on; same data, same slices, same normalisation
    (asserted equal), same loss, same plateau and stopping rules (A2), counters restarted;
  - *declared deviation*: the optimiser's moment estimates and the scheduler's counters were not saved, so AdamW
    restarts with empty moments; the shuffling generator is reseeded with `seed + 1000`. The continuation can
    only replace the checkpoint if its validation score is lower than the first round's best;
  - wall budget 3.5 h per run, at most three jobs, GPUs restricted to A100 80GB / H200;
  - **this is the only continuation round.** A run that is still not stopped by the rule after it is reported
    with the flag "not converged" and its remaining rate of improvement, as section 4 says. No further training
    is started after this round, whatever the validation scores are.
  The three runs that stopped by the rule (F0 and U0 at 128, U0 seed 1 at 4,608) are not continued.
- **A6 (selection after the continuation).** Unchanged rule, applied to the final checkpoints: per family, the
  arm with the lowest validation score at the 4,608 rung, seed 0. If the selected FNO arm is not F0 it is
  reported single-seed (no GPU slots for extra seeds within this push: limit of three jobs), and the three seeds
  of F0 are reported beside it. Before the test cohort is opened, a selection manifest (every run's final
  checkpoint SHA-256, validation scores, selected arms) is committed.
- **A7 (evaluation mechanics; no change to what is measured).** The evaluator writes one JSON file per run as it
  goes (`<out>/<job>__<run>.json`) and never re-evaluates a run whose file exists, so that a job cut by its time
  limit can be resumed for the runs not yet evaluated without a second look at the ones already done. The
  $1024^2$ zero-shot route is evaluated after the $256^2$ results of all runs are on disk. Timing as in section 5
  (A100 80GB, batch 1, 20 warm-up queries, 5 repetitions × 64 cases). Any resumption is disclosed with what had
  been seen.
