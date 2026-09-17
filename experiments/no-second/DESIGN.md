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
  (seed-variance control). No refinement.
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
