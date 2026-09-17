# b-qxm — is the correction ladder's accuracy control the rank $q$, the test count $M$, or both?

Pre-registered before any job was submitted. Every amendment is appended to
[§9 Amendments](#9-amendments) with its reason and the date, never edited in place.

Worktree `worktrees/2026-09-17-b-qxm`, branch `exp/2026-09-17-b-qxm`, forked from
`exp/2026-09-16-q-ridge` at `7dc970fc`. Cluster namespace
`/cluster/tufts/paralab/tawal01/b_qxm_20260917/`. Nothing is merged; nothing is pushed.

---

## 1. The question

The passing Burgers $256^2$ dense correction ladder (`q-trajdirs`, job 3757505, `qtd02`)
uses $M = 4(K+q)$ weak test modes, so the test count grows with the correction rank:
$M = 64, 128, 192, 320, 576, 1088$ at $q = 0, 16, 32, 64, 128, 256$. Its worst evolved
error falls $1.889 \to 0.519$ %, a span of $3.64\times$. An external audit's strongest
objection: at $q = 0$, raising $M$ alone from 64 to 256 already moves the same metric
$1.889 \to 1.271$ % (`cclad01`, `qrg201`, `qtd02` agree to $10^{-9}$), which is
$\log(1.486)/\log(3.637) = 31$ % of the whole ladder's log-span. **So is the accuracy
control the rank $q$, the test count $M$, or both?** This lane separates them with a crossed
grid in which each factor moves while the other is held fixed.

What the archives already say, read before anything was submitted (all dense, budget 600
unless noted, worst over the six development cases, same-grid evolved metric):

| source | what moves | rungs | evolved % | span |
|---|---|---|---|---|
| `cclad01` (budget 180) / `qtd02` | $q$ at fixed $M = 256$ | $q = 0, 16, 32, 64, 128$ | $1.2710, 1.2305, 1.1869, 1.1255, 1.0418$ | $1.22\times$ |
| `qtd02` | $q$ with $M = 4(K+q)$ | $q = 0 \ldots 256$ | $1.8890 \ldots 0.5194$ | $3.64\times$ |
| `qrg201` | $M$ at fixed $q = 0$ | $M = 64, 128, 256$ | $1.8890, 1.4695, 1.2710$ | $1.49\times$ |
| `qrg201` | $M$ at fixed $q = 16$ | $M = 128, 256, 512$ | $1.3985, 1.2305, 1.2204$ | $1.15\times$ |
| `qrg201` | $M$ at fixed $q = 64$ | $M = 320, 640, 1280$ | $1.0843, 1.0619, 1.0588$ | $1.02\times$ |

Two things are visible and neither settles the question: the $M$ effect at fixed $q$ shrinks
quickly with $q$ ($1.49\times \to 1.02\times$), and the $q$ effect at fixed $M = 256$ is small
($1.22\times$ to $q = 128$) but that column stops at $q = 128$ because $M = 256 < K + 256$.
Nobody has run $q$ up to 256 at a fixed $M$ large enough to hold it, and nobody has run
the two factors crossed in one job with same-job controls. That is this lane.

### 1.1 Hypotheses, stated before the run

- **H(both).** Both factors matter and neither is negligible: at the largest fixed $M$
  (1088) the $q$-span $0 \to 256$ is at least $2\times$, and the $M$-span at $q = 0$ is at
  least $1.5\times$. Expected from the table above; the prior guess for the $M$ share of the
  scheduled ladder's log-span is between one third and one half.
- **H(rank).** The rank is the control: $q$-span at fixed $M \ge 2\times$ at every fixed $M$
  that holds the full ladder, and the $M$ effect saturates by $M \approx 256$ at every $q$.
- **H(tests).** The test count is the control: the $q$-span at fixed $M$ is $< 1.5\times$ at
  every fixed $M$, and the scheduled ladder's span is mostly $\Delta_M$.

The lane's answer is which of these the grid supports, with the shares quantified.

---

## 2. What is frozen

Everything except the two declared factors. Inherited verbatim from `q-ridge` (`qrg201`),
which inherited it from `b-ladder-top` / `cheap-corrections`; the constants are read from
`experiments/q-ridge/config-r2.json` by `make_configs.py`, never typed:

| item | value |
|---|---|
| checkpoint | `experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl`, SHA256 `18f0266ae6f0454200ec0b7bf94a18cde531feac9d3170d5099adc5d68d6b589` |
| latent dimension | $K = 16$; bank dimension $R = 512$ |
| mesh / time step | $L = 256$ intervals, $\Delta t = 0.005$, 50 steps to $t = 0.25$, output every 10 steps |
| reference | FOM at 4096 intervals, $\Delta t = 3.125\times10^{-4}$, restricted to the query grid |
| evaluation cohort | **bitwise `abl01`'s six cases**: `params_draw(7090702, 4)` then `params_draw(911702, 2)`; SHA256 `108f12dc…` gated |
| directions $C_q$ | `directions.audited` — the OLD rule, seed 20260915, 1024 residual snapshots, 4 starts, budget 200; nested in $q$; GPU-model dependent across jobs (a probe, not a gate) |
| head | $u(z,y) = G\,(h_\theta(z) + C_q y)$ |
| quadrature | **dense (exact)** in every cell. The EQ regression is a separate, closed question (`q-diag`, `q-ridge`); the grid must not be confounded with it |
| solver | `varpro.make_block_lm`, the budget-600 block-damped variable-projection solver, reached through `ridge.make_query` at $\lambda = 0$ (a Python branch to `topfix.make_query('base')`) |
| stopping rule | `gtol` $=10^{-6}$ on the normalized joint gradient; residual tolerance $10^{-9}\|u_{\text{gauss}}\|\sqrt{n_g}$; per-step budget **600**; IC budget 400 |
| initializer | fixed 48-point Gauss state fit + two-way extrapolation guard, verbatim |
| test modes | the first $M$ sine modes of the square in ascending discrete-Laplacian order (`engines.modes`), the same ordering every earlier job used; an explicit $M$ here, a multiplier there |
| output contract | supplied dense initial field on GPU $\to$ six dense GPU output fields |
| timing | 3 repetitions, randomised subject order (`order_seed` 911716), 0.25 s burn-in before each, all retained, medians reported |
| numerics | float64, `JAX_DEFAULT_MATMUL_PRECISION=highest`, `jax_backend=gpu` asserted |

Same-job controls, in the same allocation on the same GPU: `fft_loose`, `nt1e-2_dt01`,
`fft_tight`. `fft_tight` *is* the same-grid reference, so its same-grid error is zero by
construction.

Three metrics per arm, per time, per case, as every earlier lane reports them: **worst over
evolved times** ($t > 0$, the trajectory error proper), **worst over all output times**
($t = 0$ included), and the **$t = 0$ compression** separately. All against the same-job
`fft_tight` field; the 4096-interval reference error is retained beside them.

### 2.1 The three mechanics that differ from `qrg201`, and why none of them moves a number

1. **Explicit $M$ per cell** instead of a multiplier rule. `topfix.build_operators(bank, L, M,
   'dense')` takes an integer $M$; the parent computed it as `RULES[rule] * (K + q)` and
   passed it in. Same function, same argument.
2. **Dense operators are built once per distinct $M$ and shared across $q$.** For
   `quadrature='dense'` the builder ignores the head, so $(A, \Phi, \lambda)$ depend on $M$
   alone. The parent already cached by $(q, M, \ldots)$; caching by $M$ is the same
   arithmetic with fewer copies. The bank $G$ is a deterministic function of the checkpoint
   and is shared after an in-job bitwise check of a fresh build against the shared copy
   (gate `operators_shared_bitwise`). This is what lets a job hold 15 arms on a 40 GB A100.
   The sbatch also sets `XLA_PYTHON_CLIENT_MEM_FRACTION=0.55` (default 0.75): compiled
   executables are loaded *outside* XLA's preallocated pool, which is where `qtd01` died
   (`Failed to load in-memory CUBIN: CUDA_ERROR_OUT_OF_MEMORY` at its 21st subject); a
   smaller pool leaves ~18 GB for them on a 40 GB card, and the memory probe (§7) puts this
   lane's in-pool need (arrays plus peak workspace) under 10 GB. No arithmetic changes.
3. **The in-job invariance gate** is `t0_field_invariant_in_M`: the initial fit uses only the
   cold Gauss space and $C_q$, never the test modes, so at fixed $(q, \text{case})$ the $t = 0$
   field must agree across $M$ to $\le 10^{-12}$ (bitwise recorded as a probe; XLA may fuse
   two programs of different $M$ differently).

The parent's `bank_G.npz` (266 MB, needed only for its R3 audit) is not written; its SHA256
is recorded. Per-step bank coefficients are still written, untimed.

---

## 3. The grid

Rows $q \in \{0, 16, 32, 64, 128, 256\}$. Columns, with the constraint $M > K + q$ that
every earlier job also enforces (`overdetermined_weak_system`):

| column | $M$ | rows it exists for | why |
|---|---|---|---|
| `fixed256` | 256 | $q \le 128$ ($256 < K + 256$) | the fixed-$M$ ladder every archive already has; extended nowhere, reproduced in-job |
| `fixed1088` | 1088 $= 4(K+256)$ | all six | the only fixed $M$ in the scheduled ladder's range that holds every rung: the **pure-rank ladder** |
| `4x` | $4(K+q)$ | all six | the scheduled ladder (`qtd02`), reproduced in-job |
| `8x` | $8(K+q)$ | all six | a second schedule |
| `16x` | $16(K+q)$ | $q \le 64$ | a third schedule where it fits |
| `2x` | $2(K+q) = 544$ | $q = 256$ only | a third $M$ at the top rung, where `fixed256` cannot exist (`btq102` comparator) |
| `bridge` | $M_{k+1}$ at row $q_k$ | $q = 16, 32, 64$ (and $(0,128)$, $(128,1088)$, already columns) | the cells $(q_k, M_{k+1})$ that make the rung-by-rung decomposition of §4.2 exact |

That is 28 distinct $(q, M)$ cells (G1 runs 15, G2 runs 13 new ones plus two $q = 0$
anchors). The saturation sweep (§3.1) adds $M \in \{512, 2048, 4096\}$ at $q = 0$ and
$\{128, 512, 2048, 4096\}$ at $q = 64$, and re-runs $(0, 64)$, $(0, 128)$, $(0, 256)$,
$(0, 1088)$, $(64, 256)$, $(64, 320)$, $(64, 1088)$ so that every within-job cost ratio
of §4.3 has its own $4(K+q)$ denominator and every $q = 0$ fidelity gate is in-job.

**Solver control.** The inner linear solve is Gauss–Jordan for $K + q \le 64$ and LU above
(`gauss_jordan_max`, inherited), so it switches between the $q = 32$ and $q = 64$ rows — the
same step at which the fixed-$M$ ladders cross from G1 to G2. G1 therefore runs the
$(32, 1088)$ cell a second time with LU (`q32_M1088_dense_lu`, label `solver-control`), in
the same job as its Gauss–Jordan twin, so the switch is measured on its own and never enters
the grid as a hidden factor. It is excluded from every span and decomposition.

### 3.1 Jobs

Split by $q$ so that each job holds at most 18 jitted subjects (landmine: a job holds one
executable and one operator set per subject; 14 fit a 40 GB A100 in `qtd01`'s layout, 34 did
not — and `--gres=gpu:a100:1` may land on a 40 GB node). The local memory probe
`probe_memory.py` (§7, `checks/probe-memory.json`) measures the per-subject device footprint
at $L = 256$ before submission.

| job | attempt | config | ROM arms | subjects | contents |
|---|---|---|---|---|---|
| G1 | `bqx101` | `config-g1.json` | 16 | 19 | rows $q \in \{0, 16, 32\}$: all columns, plus the solver control |
| G2 | `bqx201` | `config-g2.json` | 15 | 18 | rows $q \in \{64, 128, 256\}$: all columns, plus two $q = 0$ **anchor** arms ($M = 256$, $M = 1088$) that are in-job fidelity gates and cross-job anchors |
| S1 | `bqx301` | `config-s1.json` | 14 | 17 | saturation: $M \in \{64, 128, 256, 512, 1088, 2048, 4096\}$ at $q = 0$, $\{128, 256, 320, 512, 1088, 2048, 4096\}$ at $q = 64$ |

Each attempt gets its own submit directory `experiments/b-qxm/runs/<attempt>/` and its own
remote directory under the namespace; `squeue -u tawal01` before and after every `sbatch`;
exactly one job per directory; `gpu` partition only; `pax007` excluded; A100 first, then
H100 / H200 / L40S after 3 h pending, science unchanged. `--mem 180G`, `--time 08:00:00`
(the parent's 19-arm job took 1.9 h, most of it the direction fit and the references).
Lane cap: 8 jobs; 3 planned, 5 in reserve for preamble deaths, OOMs and resubmissions.

**Cross-job rule.** Costs are compared **within a job only**: the fixed-$M$ ladders that cross
the G1/G2 split are reported as two within-job cost segments and never as one cost span.
Errors are compared across jobs, because the $q = 0$ cells of four earlier jobs on two GPU
models agree to $6\times10^{-13}$ and the $q > 0$ cells to $\le 3\times10^{-9}$ (the
direction matrix is GPU-model dependent), and because the anchors and the fidelity gates
(§5) measure exactly that agreement in this lane's own jobs. No conclusion below rests on
a cross-job difference smaller than $10^{-3}$ relative.

---

## 4. The decomposition, declared before any number exists

Let $e(q, M)$ be the worst-over-cases, worst-over-evolved-times same-grid error of the
converged cell $(q, M)$, in per cent; $\ell(q, M) = \log e(q, M)$. Every quantity below is
also computed for the all-times metric; the evolved metric is the headline because the
$t = 0$ compression is $M$-independent by construction (§2.1.3) and would only dilute the
$M$ effect.

### 4.1 Spans

- **Fixed-$M$ ($q$) span:** $S_q(M) = e(0, M)/e(q_{\max}(M), M)$, with $q_{\max}(256) = 128$,
  $q_{\max}(1088) = 256$; also the like-for-like $e(0, M)/e(128, M)$ at both $M$, and
  monotonicity of $e(\cdot, M)$ in $q$.
- **Fixed-$q$ ($M$) span:** $S_M(q) = \max_M e(q, M)/\min_M e(q, M)$ over the grid's $M$ at
  that $q$, and monotonicity of $e(q, \cdot)$ in $M$. The saturation sweep gives it over
  $M \in [64, 4096]$ at $q = 0$ and $q = 64$.
- **Scheduled span:** $S_{4x} = e(0, 64)/e(256, 1088)$, the ladder the paper has been
  quoting, reproduced in-job; likewise $S_{8x}$.

### 4.2 The log-additive decomposition of the scheduled ladder

The scheduled ladder moves both factors at once. Two exact telescoping identities separate
them — exact because they are identities on the grid, not model fits:

**Corner path** (primary). $M$ first at $q = 0$, then $q$ at $M = 1088$:

$$\log S_{4x} = \underbrace{\ell(0, 64) - \ell(0, 1088)}_{\Delta_M} + \underbrace{\ell(0, 1088) - \ell(256, 1088)}_{\Delta_q},$$

with shares $\Delta_M / \log S_{4x}$ and $\Delta_q / \log S_{4x}$. The other corner order ($q$
first at $M = 64$) does not exist: $64 < K + 256$. That asymmetry is itself a finding to
state: **the rank effect can only be measured at a large $M$**, and the fixed-$M = 1088$
column is the pure-rank ladder.

**Rung path** (secondary). Along consecutive scheduled rungs
$(q_k, M_k) \to (q_{k+1}, M_{k+1})$, through the bridge cell $(q_k, M_{k+1})$:

$$\ell(q_k, M_k) - \ell(q_{k+1}, M_{k+1}) = \underbrace{\ell(q_k, M_k) - \ell(q_k, M_{k+1})}_{\Delta_{M,k}} + \underbrace{\ell(q_k, M_{k+1}) - \ell(q_{k+1}, M_{k+1})}_{\Delta_{q,k}},$$

summed over the five rungs. This gives the $M$ share **rung by rung** (where in the ladder
the test count does its work) and a second total $\sum_k \Delta_{M,k} / \log S_{4x}$ to set
beside the corner path's. If the two paths disagree by more than 0.15 in share, the factors
interact strongly and the report says so rather than quoting one number.

**Variance shares** (tertiary, a check on the identities). On the balanced sub-grid
$q \in \{0, 16, 32, 64, 128\} \times M \in \{256, 1088\}$, a two-way decomposition of
$\ell$ into grand mean, $q$ main effect, $M$ main effect and interaction (sums of squares,
reported as fractions). This is what "which factor accounts for how much" means in the
usual statistical sense; the interaction fraction says whether "additive" is a fair word.

### 4.3 Saturation

At $q = 0$ and $q = 64$ from S1: the smallest $M^\star$ beyond which the next step of the
sweep ($\times 2$, or $512 \to 1088 \to 2048$) improves the evolved error by less than 5 %
(relative), the error at $M^\star$, and the **within-job** cost ratio
$c(q, M^\star)/c(q, 4(K+q))$ — S1 carries the $4(K+q)$ cell of both rows for that purpose. "$M$ is a cost-neutral setting up to
$X$" is claimed only if that ratio is $\le 1.1$ at $M = X$; otherwise the cost slope is
reported and no such sentence is written.

---

## 5. Gates

Blocking unless marked informational; every gate is evaluated by `audit_xm.py` in NumPy
from the retained fields and JSON, never by the driver's own bookkeeping.

1. `complete`, `backend_gpu`, `x64`, `precision_highest`, `bank_frozen`,
   `checkpoint_unchanged`, `final_cohort_unopened`, `artifacts_present`,
   `same_grid_baseline_present`.
2. `evaluation_cohort_bitwise_abl01` — the six physical cases hash to `108f12dc…`.
3. `reference_residuals` — every 4096-interval reference solve below $2\times10^{-11}$.
4. `recorded_errors_recomputed_from_saved_fields` — every reported error recomputed from the
   retained fields, agreeing to $10^{-9}$ relative.
5. `overdetermined_weak_system` — $M > K + q$ in every arm.
6. `every_rom_carries_exit_and_stationarity`, `every_subject_case_has_all_reps`,
   `repetition_output_identical`, `every_invocation_paired`.
7. `operators_shared_bitwise` — a fresh `build_operators` bank equals the shared bank
   bitwise, for every distinct $M$ built.
8. `t0_field_invariant_in_M` — at fixed $(q, \text{case})$ the $t = 0$ field agrees across
   $M$ to $\le 10^{-6}$ relative, the IC fit's own stationarity tolerance (each $M$ is a
   separate jitted program, and an ulp-level GEMM difference can move the IC
   Levenberg–Marquardt path within that tolerance, §9 A0); $\le 10^{-12}$ and bitwise are
   recorded as probes.
9. **Cross-job fidelity**, one gate per (arm, source) pair declared in the config:
   - unconditional at $10^{-9}$ on both metrics for every $q = 0$ pair: `q0_M64` against
     `btq101 q0_m4_dense_base`, `cclad01 q0_m4_dense_block`, `qrg201 q0_m4_dense_l0`,
     `qtd02 q0_M64_dense`; `q0_M128` against `qrg201 q0_m8_dense_l0`; `q0_M256` against
     `cclad01 q0_m256_dense_block`, `qrg201 q0_m16_dense_l0`, `qtd02 q0_M256_dense`. G1 and
     S1 each carry at least two of these; G2 carries `q0_M256` (three sources);
   - two-tier for $q > 0$ (the `b-ladder-top` convention: $10^{-9}$ if the directions hash
     is bitwise the source's, $10^{-3}$ otherwise, achieved difference always reported):
     `q16_M128`, `q16_M256`, `q16_M512`, `q32_M192`, `q32_M256`, `q64_M256`, `q64_M320`,
     `q64_M640`, `q64_M1280`, `q128_M256`, `q128_M576`, `q256_M544` (`btq102`
     `q256_m2_dense_base_b600`), `q256_M1088` (`qtd02`, `cclad01`) — 16 named cells in all,
     each against every earlier job that ran it (46 pairs over the three configs). A pair
     that compares fewer than two non-null metrics fails rather than passing vacuously.
     `cclad01`/`btq101` carry a directions hash (`802737d3…`) no later job has reproduced, so
     their $q > 0$ pairs will in practice be judged at $10^{-3}$; the achieved difference is
     what the report prints.
10. `anchors_agree_across_jobs` (report level) — `q0_M256` and `q0_M1088` in G2 reproduce G1
    to $10^{-9}$ on both metrics; `q0_M64`, `q0_M128`, `q0_M256`, `q0_M1088`, `q64_M256`,
    `q64_M1088` in S1 reproduce G1/G2 to $10^{-9}$ ($q = 0$) or two-tier ($q = 64$).
11. Informational probes: `directions_hash_matches_<source>`,
    `reference_fields_bitwise_match_<source>`, `t0_field_bitwise_in_M`.

A cell is **converged** iff every one of its 900 steps ($6 \times 3 \times 50$) exits on a
convergence criterion (no budget exit), the IC fit converges, and the worst normalised joint
gradient is $\le 10^{-6}$. A non-converged cell is reported with its flag and **excluded**
from every span and decomposition; if the exclusion breaks a path in §4.2, that path is
reported as unavailable, not patched.

---

## 6. Pre-registered verdict and falsification

**The headline the paper should carry**, decided by these rules and nothing else:

- **Headline the fixed-$M$ ladder** ($M = 1088$, pure rank) if it is monotone in $q$ on the
  evolved metric, every rung converged, and $S_q(1088) \ge 2\times$ — with the scheduled
  ladder reported beside it as "rank + test count" and the shares of §4.2 stated. This is the
  outcome under H(both) and H(rank).
- **Headline the scheduled ladder** only if no fixed-$M$ ladder reaches $2\times$; then the
  $M$ share must be printed next to the span wherever the span is quoted.

**The rank claim is FALSE** — and the lane reports it as such — if $S_q(M) < 1.5\times$ at
**every** fixed $M$ that holds the ladder (256 to $q = 128$; 1088 to $q = 256$). Then the test
count, not the rank, is the accuracy control; "correction rank" leaves the paper's claim
and $M$ enters it.

**H(rank) is false** if the $M$ effect does not saturate: $S_M(q) \ge 1.5\times$ over the
cells with $M \ge 2(K+q)$ (at least two tests per unknown, so the near-square $(64, 128)$
cell of the sweep cannot decide it) at some $q \ge 64$ within the grid, or the saturation
sweep shows no $M^\star \le 1088$ at $q = 64$.

**H(tests) is false** if $\Delta_q / \log S_{4x} \ge 0.5$ on the corner path.

A fixed-$M$ column with a non-converged rung reports its span as **unavailable**; it is not
patched by dropping the rung. The all-times metric is $t = 0$-dominated at $q \le 128$ and its
$\Delta_M$ is near zero by construction (the compression does not depend on $M$); the report
says so beside that table.

Whatever the outcome, the report prints: the full $6 \times$ (columns) table with the six
per-cell quantities the coordinator asked for (worst evolved %, worst all-times %, $t = 0$
compression %, median GPU ms, converged, median iterations), the spans, both decompositions,
the variance shares, and the saturation curves — all generated from the audit JSONs by
`reports/generate_xm.py`, with `summary.json` beside the Markdown.

---

## 7. Before the first job

- Local smoke `smoke_xm.py` (64 intervals, sub-minute): reproduces the parent lane's audited
  $q = 0$ fixture to $10^{-12}$ (gate 1); operators depend on $M$ alone and the bank is
  shareable bitwise (gate 2); $t = 0$ invariant in $M$ (gate 3); every arm stationary, and
  the explicit-$M$ cell at $M = 4(K+q)$ is **bitwise** the parent's rule-path arm (gate 4).
  Output `checks/smoke-xm.json`.
- Local device-memory probe `probe_memory.py` at $L = 256$ with random directions (values
  irrelevant, shapes exact): device bytes after each of six compiles at the largest cells
  each job holds. Output `checks/probe-memory.json`. Recorded as a local run over the
  sub-minute rule if it exceeds it.
- Codex audit of this file, headless and read-only, output `reports/codex-design.md`;
  accepted and rejected findings recorded in §9.
- Commit; then stage with `cluster/stage.py` (byte-checked against `HEAD`).

---

## 8. Reporting

`reports/2026-09-17-b-qxm.md` and `reports/summary.json` from `reports/generate_xm.py`,
reading only `checks/<attempt>-audit.json` and `checks/comparators.json`. `summary.json`:
one object per table row with `q`, `M`, `quadrature`, `metric`, `value`, `job_id`,
`attempt`, `arm`, `source_sha256`. Every table ends with a plain-language glossary.
Equations LaTeX, diagrams mermaid. Codex audit of the final report against the raw audit
JSONs: `reports/codex-report.md`.

---

## 9. Amendments

### A0 (2026-09-17, before the first job) — pre-submission audit and the smoke-gate finding

An independent read-only audit of this file and the code (Codex was over its usage limit
until 19 Sep; a fresh Claude Opus agent with no lane context was used instead, transcript
summary in `reports/design-audit.md`) returned 14 findings. Accepted and applied before
commit: the bank-share check now **asserts** rather than records (`q_xm.py`); S1 carries
$(64, 320)$ so the §4.3 cost ratio is within-job; the solver control at $(32, 1088)$ was added
(above); fixed-$M$ spans come only from the declared columns and report *unavailable* on a
non-converged rung; the H(rank) lever reads $M \ge 2(K+q)$; a fidelity pair comparing no
metric fails; the parent's `expected_directions_sha256` (unmeetable) is dropped; the cell
and named-cell counts and the $3\times10^{-9}$ sentence were corrected; the all-times caveat
was added. Rejected: computing $t = 0$ once per $q$ outside the per-$M$ program — that would
change the timed contract; the gate was re-tiered instead (below).

**The smoke gate-1 finding.** `smoke_xm.py`'s gate 1 — the retained $q = 0$ path at 64
intervals against the consolidated audited fixture — returned $2.14\times10^{-7}$ twice
(IC fit 111 iterations), while the identical construction in `smoke_gate1.py` returned
$1.56\times10^{-14}$ three times in a row (IC fit 86 iterations), both on the shared GB10
with three other agents' jobs on the GPU. The two scripts build the query identically; the
result is deterministic per process and differs between processes. The IC fit is a
Levenberg–Marquardt loop stopped at a $10^{-6}$ stationarity tolerance, so an ulp-level
difference in one GEMM (XLA's GPU autotuner chooses algorithms by timing them, which under
contention can pick differently per process) changes the iterate path and lands the fit
at a different point inside the tolerance — $10^{-7}$ relative in the field. **What this
means for the lane:** (i) the cluster jobs run one process per exclusively allocated GPU,
where four earlier jobs on two A100 models reproduced the $q = 0$ cells to
$6\times10^{-13}$, and the in-job fidelity gates decide whether that held again — the
protocol's $10^{-9}$ bar is enforced there, not assumed; (ii) the $t = 0$ invariance gate
is blocking at $10^{-6}$, the level this mechanism can reach, with $10^{-12}$ and bitwise as
probes; (iii) the local smoke's gate 1 is recorded with its IC iteration count and
`XLA_FLAGS`, asserted at the end so gates 2–5 still run; (iv) the protocol's requirement of
a local reproduction to $\le 10^{-9}$ is met by `checks/gate1-under-load.json`
($1.56\times10^{-14}$, three repetitions). If the cluster fidelity gates fail at $10^{-9}$,
the lane reports that as a retraction of the assumption, not as a pass at $10^{-3}$.

**Memory probe.** `probe_memory.py` (random directions, exact shapes, $L = 256$) measured
in-pool peaks of 1.1, 2.9, 2.9, 4.0, 6.6 GB after compiling $(0,64)$, $(0,1088)$, $(32,1088)$,
$(64,1280)$, $(256,2176)$ cumulatively, and `nvidia-smi` showed the process at 33.3 GB
against a 32 GB preallocated pool — about 1.3 GB outside the pool for five compiled
subjects. The $(64, 4096)$ cell is recorded in `checks/probe-memory.json` when it lands.
These are GB10 numbers under a 36 GB cgroup, not A100 numbers; they size the pool
fraction (0.55 leaves ~18 GB outside the pool on a 40 GB card) and nothing else. Two of
the five reserve attempts are earmarked for an `h100` resubmit if a CUBIN out-of-memory
recurs.

**Deviation recorded.** The probe held one of the three shared local GPU slots for 27
minutes (its last cell, $(64, 4096)$ with random directions, ran every step to the budget)
against the sub-minute rule for local runs; the coordinator flagged it while it ran. It was
not restarted, its result stands as written above, and nothing else in this lane runs
locally beyond the 64-interval smoke.

**Smoke result.** `checks/smoke-xm.json`: gate 1 $1.41\times10^{-14}$ (field) /
$6.1\times10^{-13}$ (latents), IC fit 86 iterations; operators bitwise per $M$ and the bank
shareable bitwise; $t = 0$ **bitwise** invariant across $M \in \{64, 96, 128, 256\}$ at
$q \in \{0, 8\}$ (8 of 8 cells); every cell stationary with zero budget exits; the explicit-$M$
cell at $M = 4(K+q)$ bitwise equal to the parent's rule-path arm; error monotone in $M$ at
both $q$. Wall time 135 s — over the sub-minute rule by the direction fit and eight
compiles; recorded as a deviation, not repeated.

### A1 (2026-09-17, before any result) — submission record

Source commit `ed431edb` (every staged file byte-checked against it by `cluster/stage.py`).
Three submissions from three attempt directories, `squeue -u tawal01` checked before and
after each, exactly one job per directory confirmed:

| job | attempt | config | Slurm job id | GPU request | time | submitted |
|---|---|---|---|---|---|---|
| G1 | `bqx101` | `config-g1.json` | 3780175 | `a100:1`, `--exclude pax007`, `--mem 180G` | 08:00:00 | 2026-09-17, pending (Priority) |
| G2 | `bqx201` | `config-g2.json` | 3780177 | same | 08:00:00 | same |
| S1 | `bqx301` | `config-s1.json` | 3780178 | same | 08:00:00 | same |

Remote: `/cluster/tufts/paralab/tawal01/b_qxm_20260917/<attempt>/`; paralab at 91 %
(439 GB free) at submission. Three of the eight-job cap used. If a job is still pending after
3 h it is cancelled and resubmitted as `h100`, then `h200`, then `l40s`, science unchanged;
each resubmission gets a new attempt directory and is recorded here.

### A2 (2026-09-17, after the results) — outcome against the pre-registered rules

All three jobs COMPLETED exit 0 on the `gpu` partition from commit `ed431edb`: 3780175
(`bqx101`, pax050, A100 80 GB, 2528 s), 3780177 (`bqx201`, pax051, **A100 40 GB**, 3715 s),
3780178 (`bqx301`, pax052, **A100 40 GB**, 3217 s). **Zero failed gates in all three.** The
$t = 0$ field was **bitwise** invariant in $M$ in all 54 $(q, \text{case})$ families
(worst relative difference exactly 0.0), so the re-tiering in A0 was not needed in the end —
the $10^{-12}$ probe passes too. Every one of the 19 unconditional $q = 0$ fidelity
reproductions landed at $\le 9.3\times10^{-13}$, i.e. the $10^{-9}$ bar held on the cluster,
which retires the A0 worry: the $2.1\times10^{-7}$ smoke deviation is a shared-GB10 artefact
and does not exist in a one-process, one-GPU allocation. The two-tier $q > 0$ pairs also came
in at $\le 9.2\times10^{-9}$, inside $10^{-3}$ by six orders.

**Memory.** Peak device use was 9.94 GB (G2) and 11.36 GB (S1) against the ~22 GB pool that
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.55` reserves on a **40 GB** card, with 18 and 17 jitted
subjects resident. Both landed on 40 GB nodes, so the binding case was exercised and the
split-by-$q$ sizing is now measured on the target hardware rather than extrapolated from the
GB10. No CUBIN out-of-memory occurred; the two earmarked reserve attempts were not needed.
Three of the eight-job cap used.

**Verdict.** H(both) holds, with the rank dominant: the pure-rank ladder at fixed $M = 1088$
is monotone, fully converged, and spans $2.437\times$, so by §6 the **fixed-$M$ ladder is the
headline** and the scheduled ladder is reported beside it. The rank claim is **not** false
($2.437\times \ge 1.5\times$). H(tests) is **false** (rank share $0.690 \ge 0.5$ on the corner
path). H(rank) is **false** as well: the $M$ effect does not saturate everywhere — at
$q = 256$ the span over $M \ge 2(K+q)$ is $2.025\times$ ($0.7566 \to 0.3736$ % for
$M = 544 \to 2176$). Corner and rung paths agree (test-count share 0.310 and 0.344, well
inside the 0.15 disagreement threshold), and the two-way variance shares on the balanced
sub-grid are rank 85.1 %, test count 6.1 %, interaction 8.8 %.

**The structure the decomposition exposes, which neither hypothesis predicted.** The split is
not uniform along the ladder: rung by rung, the test count does 84 %, 77 %, 40 %, 10 %, 5 % of
the work at $q = 0\to16, 16\to32, 32\to64, 64\to128, 128\to256$. The test count is what buys
the *bottom* of the ladder and the rank is what buys the *top*. Equivalently, $M$ saturates
in absolute terms at low $q$ ($M^\star = 256$ at $q = 0$, flat to $M = 4096$) but not at high
$q$; what is roughly invariant is the tests-per-unknown ratio, not $M$ itself.

**Solver control.** $(32, 1088)$ with LU reproduced its Gauss–Jordan twin **exactly**
(relative difference 0.0 on the evolved metric — the two solves agree to the last bit)
at $0.9999\times$ the cost in the same job, so the `gj`$\to$`lu` switch at $K + q > 64$ is
inert and does not contaminate the rank ladder. (This paragraph first carried a hand-typed
$4.7\times10^{-14}$, which was simply wrong; it is now read from `summary.json`. The
project's rule against hand-typed numbers earned its place again.)

**Saturation.** $q = 0$: $M^\star = 256$, $1.2710$ %, $1.210\times$ the $M = 64$ cost —
**not** cost-neutral by the $\le 1.1\times$ rule, so that sentence is not written. $q = 64$:
$M^\star = 256$, $1.1255$ %, $0.922\times$ the scheduled $M = 320$ cost — cheaper *and* within
3.7 % of it, so at $q = 64$ the scheduled rule overspends on tests. The $q = 64$ curve is
**non-monotone** at the tail ($1.0593 \to 1.0594 \to 1.0620$ % at $M = 1088, 2048, 4096$):
past saturation more tests very slightly hurt.

### A3 (2026-09-17, before the round-2 jobs) — two extensions, pre-registered

Round 1 answered the commissioned question. Two facts in its own output point past it, and the
coordinator asked for both. Declared here before either job was staged.

**First, a correction to my own round-1 framing, which had already reached the paper file.** I
wrote that "what is roughly invariant is the tests-per-unknown ratio, not $M$ itself". The data
does not support it: saturation sits at $M^\star/(K+q) \approx 16$ at $q = 0$, $\approx 4$ at
$q = 64$, and is **not reached by 8** at $q = 256$. I compared a $4\times \to 16\times$ range at
$q = 0$ against a $2\times \to 8\times$ range at $q = 256$ — different ranges — and generalised.
What the data does support is narrower: **the absolute $M$ needed to saturate grows with $q$, and
at $q = 256$ it is still growing at 8 tests per unknown.** E2 measures that curve properly.

#### E1 (`bqx401`, `config-e1.json`) — does the pure-rank ladder keep going past $q = 256$?

The fixed-$M = 1088$ ladder spans $2.437\times$ against a $2\times$ bar and the bank projection
floor is $0.39$ %, so most of the headroom is unclaimed. $q = R = 512$ is the whole bank: the
largest rank the decoder can carry.

| cell | role |
|---|---|
| $(0, 64)$ | the unconditional $10^{-9}$ fidelity anchor four earlier jobs also ran |
| $(q, 1088)$ for $q = 0, 16, 32, 64, 128, 256, 512$ | the whole fixed-$M$ column **in one job**, so the within-job cost ladder covers every rung — the exact place the round-1 generator went wrong |
| $(512, 2112) = 4(K+q)$, $(512, 3168) = 6(K+q)$ | the disambiguation: $M = 1088$ is only **2.06** tests per unknown at $q = 512$, and $q = 64$ at 1.6 tests per unknown was visibly starved ($1.8032$ % against $1.1255$ % at $3.2\times$). Without these, a bad $(512, 1088)$ cannot be told from a rank limit |

`cclad01` ran $(512, 2112)$ and $(512, 1056)$ at budget **180** and reached $0.2307$ % and
$0.4343$ % — better than anything this campaign has recorded — but **neither converged** (budget
exits), so they are cited as motivation and **not** used as gates. If $q = 512$ converges at
budget 600 near those values it is the best cell in the campaign; if it does not converge, that
is the finding and the rung is excluded from every span, as §5 requires.

#### E2 (`bqx501`, `config-e2.json`) — where does the best cell stop improving?

$(256, 2176)$ at $0.3736$ % is the campaign's best converged number. $q = 256$ against
$M \in \{1088, 2176, 3264, 4352, 6528\}$ — 4 to **24** tests per unknown — plus the $(0, 64)$ and
$(0, 1088)$ anchors. Requests an **80 GB** A100 by `--constraint=a100-80G`: the five distinct $M$
need 8.46 GB of test-mode matrices, and round 1 peaked at $2.6\times$ its own matrix storage, so
$\approx 22$ GB would sit exactly on the pool ceiling of a 40 GB card.

#### Gates and what would make the extensions uninformative

Same gates as §5. New: every round-2 cell that round 1 also ran is gated against **this lane's own
round-1 jobs** as well as the earlier ones. Round 1 established that the direction matrix is
**run-to-run nondeterministic even on one GPU model** — `bqx101`, `bqx201`, `bqx301` hash to three
different values — so every $q > 0$ pair is judged at the loose tier and the *achieved* difference
is the evidence; round 1 achieved $\le 9.2\times10^{-9}$ against four earlier jobs.

- **E1 is uninformative** if $(512, 1088)$, $(512, 2112)$ and $(512, 3168)$ all fail to converge:
  then nothing is learned about the rank past 256 and the lane says so.
- **E1 answers "the rank has run out"** only if $(512, 3168)$ converges and is no better than
  $(256, 1088)$; if instead $(512, 1088)$ is poor while $(512, 3168)$ is good, the cause is the
  test count, not the rank, and the fixed-$M = 1088$ column's $q = 512$ rung must be reported as
  test-starved rather than as a rank limit.
- **E2 answers "it flattens at $M^\star$"** only if the last step improves by less than 5 %; if the
  curve is still falling at $M = 6528$ the honest statement is that 24 tests per unknown does not
  saturate $q = 256$, and no $M^\star$ is quoted.

Reporting: `reports/2026-09-17-b-qxm.md` and `summary.json` are **updated in place** (one report per
question, not per session); the lab-log entry gets a short addendum, not a new entry.
