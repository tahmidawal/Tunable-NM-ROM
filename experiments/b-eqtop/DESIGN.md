# b-eqtop — can the top rungs of the Burgers EQ ladder be certified, and does a certified rule make the ladder monotone?

Pre-registered before any job was submitted. Amendments are appended to
[§10](#10-amendments) with their date and reason, never edited in place.

Worktree `worktrees/2026-09-17-b-eqtop`, branch `exp/2026-09-17-b-eqtop`, forked from
`exp/2026-09-16-q-ridge` at `7dc970fc`. Cluster namespace
`/cluster/tufts/paralab/tawal01/b_eqtop_20260917/`. Nothing is merged; nothing is pushed.

---

## 1. The question

`q-ridge` (job 3768168, attempt `qrg304`) showed that the evolved-times regression of the
Burgers $256^2$ correction ladder is the empirical-quadrature rule, and that refitting the
rule on states the ROM actually reaches and certifying it by its held-out
$$\rho(u) = \frac{\bigl\|\sum_{j=1}^{m} w_j \Phi(x_j)\,a(u)(x_j) - \Phi^\top a(u)\bigr\|_2}{\|\Phi^\top a(u)\|_2}$$
removes almost all of it. It left one violation: at $q = 128$ and $q = 256$ no rule with
$m \le 2048$ reaches the primary bar $\rho_{\max} \le 0.116$, the two top rungs run on
secondary-bar rules ($\rho_{95} \le 0.116$), and the ladder regresses $0.8925 \to 1.0361$ %
from $q = 128$ to $256$ while the dense twin reaches $0.5194$ %. Its log-log extrapolation
put the $q = 256$ crossing at $m \approx 2522$; the same two points at $q = 128$ give a much
shallower slope ($0.2786 \to 0.1908$ from $m = 1024 \to 2048$, i.e. $\alpha \approx 0.55$),
which would put that rung's crossing near $m \approx 5100$. The declared grid went to
$m = 8192$ but the job could construct only $m \le 2048$: the audited scipy Lawson–Hanson
fitter took about $1000$ s at $m = 2048$ on $\sim 8200$ design rows and its cost grows like
$m^2$ (measured $1024 \to 2048$: $4.1$–$4.4\times$), so $4096$ and $8192$ did not fit the
declared $1500$ s per rule.

Three facts about how those rules were built matter here and are the arms of this lane:

1. **Fit-state count.** The incumbent convention is $\mathrm{clip}(8192/M, 8, 64)$ states, so
   the $q = 128$ rule was fitted on **14** reachable states and the $q = 256$ rule on **8**,
   against a held-out set of 512. $\rho_{\max}$ is a tail statistic ($\rho_{95}$ is
   $3$–$4\times$ lower at those rungs); a rule fitted on 8 states has not seen the tail.
2. **Candidate pool.** The pool was 8192 interior points (the incumbent; `q-ridge` §A5
   reverted the 16384 it had declared), so $m = 6144$ would be three quarters of the pool.
3. **Row scaling.** Every design row (state $s$, test mode $j$) is normalised to unit norm,
   so the least-squares objective weights states by the pointwise size of their advection,
   while $\rho$ normalises each state by $\|\Phi^\top a(u_s)\|$. States with small
   $\|\Phi^\top a\|$ are under-weighted in the fit and count fully in $\rho_{\max}$.

**The question.** Is there, at $q = 128$ and $q = 256$, a reachable-state rule with
held-out $\rho_{\max} \le 0.116$ at some $m \le 6144$; at what $m$ does each construction
reach it; and, with the cheapest certified rule at every rung, is the rebuilt EQ ladder
monotone on the worst-over-evolved-times metric?

---

## 2. What is frozen

Everything `qrg304` froze, verbatim (checkpoint SHA256 `18f0266ae6f0…`, $K = 16$, $R = 512$,
$L = 256$, $\Delta t = 0.005$, 50 steps, the six opened development cases hashing to
`108f12dc…`, the OLD nested directions with seed 20260915, the budget-600 block-damped
variable-projection solver `varpro.make_block_lm` through `ridge.make_query` at
$\lambda = 0$, `gtol` $10^{-6}$, IC budget 400, the 48-point cold start, three timed
repetitions with 0.25 s burn-in and `order_seed` 911716, float64, highest matmul precision).
The reachable-state collection is `q_eqcert.py` phase 2 verbatim with the same seeds
(`collect_seed` 20260916, 24 fit and 8 certification trajectories, 12 unrolled LM iterates
per step), so the held-out certification states are the ones `qrg304` certified on; their
SHA256s are carried in the config and compared (informational: GPU-model dependent).

$M = 4(K+q)$ tests at every rung. The fitter is `varpro.bounded_nnls` — the cell's audited
block-greedy Lawson–Hanson with refit block $m/16$ — **run unchanged**, in a CPU worker
process whose copy of the function is gated bitwise against the original
(`smoke_eqtop.py` gate 3). No new fitter enters a reported number.

---

## 3. The manipulated variables — three declared fit arms

Every new rule is on the **reachable** population with a **16384-point candidate pool**
(seed `20260917 + q`, shared by the arms of a rung). The arms differ only in how the design
is formed:

| arm | fit states | row scaling | design rows at $q=128$ / $256$ | compression |
|---|---|---|---|---|
| `std` | incumbent $\mathrm{clip}(8192/M, 8, 64)$: 14 / 8 | per row, unit norm (incumbent) | 8064 / 8704 | none |
| `fs64` | 64 | per row, unit norm | 36864 / 69632 | exact QR to 16384 rows |
| `rhow64` | 64 | per **state**: every row of state $s$ divided by $\|\Phi^\top a(u_s)\|$ | 36864 / 69632 | exact QR to 16384 rows |

Under the `rhow64` scaling the least-squares objective is **exactly** $\sum_s \rho(u_s)^2$
over the fit states (smoke gate 5) — the certification metric's mean square.

**Compression.** With more rows than candidates the fitter's cost would grow with the
number of fit states. Writing $[D\,|\,b] = Q\,[R\,|\,c\,;\,0\,r]$ (thin QR, `mode='r'`),
$$\|Dw - b\|_2^2 = \|Rw - c\|_2^2 + r^2 \quad\text{and}\quad D^\top(b - Dw) = R^\top(c - Rw)$$
for every $w$, so the fitter on $(R, c)$ is the fitter on $(D, b)$: same gradients, same
block selections, same nonnegative least-squares subproblems, up to rounding. The reported
relative fit is on the original rows, $\sqrt{\|R_S w - c\|^2 + r^2}/\|b\|$. Gates: in-job,
$R^\top R$ reproduces $D^\top D$ on 32 random columns to $10^{-8}$ (else the job aborts);
in the smoke (gate 4), the same support and weights to $10^{-8}$, the same fit to
$10^{-10}$ and the same $\rho$ as the uncompressed fit on a real over-tall design.

---

## 4. The $m$ grid and the stopping rule

Per (rung, arm) the rules are fitted in an **ascending chain**:

- `std`: $m \in \{1024, 2048, 2560, 3072, 4096, 6144\}$ — 1024 and 2048 repeat `qrg304`'s
  sizes with the larger pool, so the pool effect at fixed $m$ is measured;
- `fs64`, `rhow64`: $m \in \{2048, 2560, 3072, 4096, 6144\}$ — 2048 measures the fit-state
  and scaling effects at `qrg304`'s size.

**Stopping rule (adaptive).** A chain stops after the first rule that is
**primary-certified** plus **one** further point (the confirmation, to pin the local slope
past the crossing); or when the grid is exhausted; or when a rule is truncated by its
walltime cap (a larger $m$ would be slower still). A truncated rule is disqualified.

**Walltime.** Each rule has a cap of 5400 s (`eq_seconds`); no new fit is launched after
11 h of job time (`fit_submission_deadline_seconds`); the timed phase runs only after
every worker has exited. The chains run **concurrently** in CPU worker processes (the
fitter is single-threaded in practice: the local probe in `checks/probe-fitter.json` shows
no speed-up from BLAS threads), each rule certified on the GPU by the driver as it lands.

---

## 5. Certification — the bars, declared

A rule is certified on a bar iff its held-out $\rho_{\max} \le$ the bar and it is not
truncated. Two bars, both declared here:

| bar | value | provenance |
|---|---|---|
| **primary** | $\rho_{\max} \le 0.116$ | `q-ridge` §A3.4: `q-diag`'s $\rho$ of the $q = 0$ incumbent rule at the state carrying the first-interval penalty |
| **tight** | $\rho_{\max} \le 0.06$ | the coordinator's tighter bar, declared before this job ran, to test whether the primary bar is *sufficient* at the top rung |

The secondary bar $\rho_{95} \le 0.116$ is reported for every rule and used only as the
fallback the primary ladder takes where no primary-certified rule exists (the row says so).
**A rule is never certified by its NNLS fit residual**; the fit is reported beside $\rho$ so
the anti-correlation can be seen again.

---

## 6. The rebuilt ladder — arms of the timed phase

Every rung $q \in \{0, 16, 32, 64, 128, 256\}$, $M = 4(K+q)$, one allocation, three timed
repetitions, same-job full-order controls `fft_tight` (the same-grid reference),
`fft_loose`, `nt1e-2_dt01`:

| arm | rule |
|---|---|
| `q{q}_eq_primary`, six rungs | the **cheapest** (smallest $m$; ties by arm order `std` < `fs64` < `rhow64`; `qrg304`'s archived rules rank before this job's at equal $m$) primary-certified rule; at $q \le 64$ these are `qrg304`'s archived $m = 1024$ rules, re-certified in-job |
| `q{q}_eq_tight`, where it differs from the primary arm | the cheapest tight-certified rule (at $q = 16$ this is the archived $m = 2048$ rule, $\rho_{\max} = 0.0452$) |
| `q128_eq_qrg304_m2048`, `q256_eq_qrg304_m2048` | `qrg304`'s chosen secondary-bar rules — the reproduction arms and the uncertified reference points of the $\rho$-vs-error table |
| `q{q}_dense`, $q \in \{0, 64, 128, 256\}$ | exact quadrature twins; 128 and 256 are the hybrid ladder's fallbacks |

At most 16 reduced arms (`max_rom_arms`), asserted in-job; the job requests an 80 GB A100
(`--constraint=a100-80G`) because a 40 GB card holds about 14 jitted queries.

Reported per arm: worst over evolved times, worst over all times, $t = 0$ compression,
median GPU ms with all repetitions retained, median iterations, converged flag, and per
case and per time. Errors are recomputed by the audit from the saved fields.

---

## 7. Pre-registered pass / fail

**P1 (certification).** At each of $q = 128$ and $q = 256$ some arm reaches the primary bar
at some $m \le 6144$. Reported per arm: the smallest certified $m$, and the log-log law
$\rho_{\max} \propto m^{-\alpha}$ fitted over the chain's untruncated points with the $m$ it
predicts for each bar.

**P2 (the ladder).** The primary ladder is non-increasing in $q$ on the worst-evolved-times
metric over all six rungs, every rung converged (zero budget exits, worst normalised joint
gradient $\le 10^{-6}$), and every EQ rung cheaper than its same-job dense twin where one
was run. Cost ratios are formed **within this job only**.

**P3 (the bar's sufficiency).** If P1 holds at $q = 256$ but the primary ladder still
regresses $128 \to 256$, the primary bar is declared insufficient at the top rung; the tight
ladder is reported with its own verdict, and the $\rho$-vs-evolved-error table at $q = 256$
(`qrg304`'s $\rho_{\max} = 0.168$ rule, the primary-certified rule, the tight-certified
rule) states what $\rho$ the top rung actually needs.

**Falsification.** If no arm reaches the primary bar at a rung by $m = 6144$ (or within the
walltime), the lane reports the $m$ at which each arm's law would reach it and **stops**,
and reports the **hybrid** ladder — primary-certified EQ where it exists, dense at the
uncertified rung — measured in the same job, with its cost. If the ladder is monotone only
because every EQ rung costs as much as dense, the lane says so.

---

## 8. Gates

Blocking unless marked informational.

1. `complete`, `fit_phase_complete`, `backend_gpu`, `x64`, `precision_highest`,
   `bank_frozen`, `checkpoint_unchanged`, `final_cohort_unopened`, `step_budget_600`,
   `evaluation_cohort_bitwise_abl01`, `fit_and_certification_trajectories_disjoint`,
   `reference_residuals` ($< 2\times10^{-11}$).
2. `archived_rules_recertify` — the 18 `qrg304` rules (12 reachable, 6 static),
   re-certified on this job's held-out states, reproduce their archived $\rho_{\max}$ to
   $10^{-6}$ relative. (Informational beside it: `collection_pools_bitwise_qrg304`,
   `directions_hash_matches_qrg304` — GPU-model dependent, the latter sets the fidelity tier.)
3. `certification_flags_recomputed` — the driver's certified flags equal the audit's
   recomputation from $\rho$ and the truncation flag; `every_rule_archived` — nodes and
   weights saved and hashed for every rule.
4. `recorded_errors_recomputed_from_saved_fields` ($10^{-9}$), `repetition_output_identical`,
   `every_subject_case_has_all_reps`, `every_invocation_paired`,
   `every_rom_carries_exit_and_stationarity`, `overdetermined_weak_system`,
   `same_grid_baseline_present`, `artifacts_present`, `bank_sha256_consistent`,
   `decoded_fields_match_saved_outputs` ($10^{-12}$).
5. **Cross-job fidelity** against `qrg304` (comparators generated from its committed audit by
   `make_comparators.py`), both metrics and the reference metric, two-tier tolerance as in
   `q-ridge`: $10^{-9}$ where the directions are bitwise (and unconditionally at $q = 0$),
   $10^{-3}$ otherwise: `q0_eq_primary` ↔ `q0_m4_eqcert`, `q16/32/64_eq_primary` ↔
   `q16/32/64_m4_eqcert`, `q128_eq_qrg304_m2048` ↔ `q128_m4_eqcert`,
   `q256_eq_qrg304_m2048` ↔ `q256_m4_eqcert`, `q0/64/256_dense` ↔ `q0/64/256_m4_dense`.
6. **Local smoke before submission** (`smoke_eqtop.py`, `checks/smoke-eqtop.json`):
   `eqtop.rho_of_field` reproduces `q-diag`'s archived $\rho$ for the `cclad01`
   `q0_m256_eq_block` rule on that job's own fields at 36 (case, time) points to
   $\le 10^{-9}$ (measured: bitwise); the coefficient-path certifier equals the field path to
   $10^{-10}$; the worker's fitter is `varpro.bounded_nnls` bitwise; compression
   equivalence; the state-scaling identity; the chain scheduler end to end through real
   subprocesses with the adaptive stop.

---

## 9. Jobs

| job | attempt | contents | GPU |
|---|---|---|---|
| 1 | `bet101` | §3–§6: the top-rung chains, the re-certified archive, the rebuilt ladder | A100 80 GB, `--exclude pax007`, 16 CPUs, 180 GB, 20 h |
| 2 | `bet201` | the $\rho$-vs-$m$ curve at $q \in \{0, 16, 32, 64\}$: `std` on $m \in \{256, 512, 1024, 2048, 3072, 4096\}$ (non-adaptive), `fs64` and `rhow64` at $m \in \{1024, 2048\}$; no timed phase | same request |

Both are submitted together (they are independent) into their own attempt directories;
`squeue` before and after every submission; one job per directory. Cap: 8 submissions,
retracted jobs counted, preamble deaths with zero GPU time not. Escalation to H100 / H200 /
L40S only after 3 h pending, with the science unchanged.

Fit-cost model used to size the grid (from `qrg304`'s own fit times at $\sim 8200$ rows,
which scale as $m^2$): $m = 2560, 3072, 4096, 6144 \approx 1000, 1450, 2600, 5900$ s for
`std`; the compressed arms have 16384 rows, about $2\times$. Six concurrent chains put the
fit phase at the longest chain, $\le 3.5$ h if the primary bar is reached by $m = 3072$,
$\le 6$ h if it is reached only at 6144. The local probe of the fitter's cost on the real
design is `checks/probe-fitter.json`.

Every job: `stage.py` byte-checks each staged file against `git show HEAD:`, writes
`COMMIT.txt`, `PROVENANCE.json`, `MANIFEST.sha256`; the sbatch activates the paralab venv,
exports `JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest`, asserts
`jax_backend=gpu` (exit 42 otherwise), writes `OUTPUTS.sha256`, ends with `ALL-DONE`;
`collect.py` verifies both manifests remotely, archives, verifies locally; the audit runs;
`preserve_archive.py` chunks the archive into Git; the remote attempt directory is deleted.

---

## 10. Amendments

(none yet)

### A1 (2026-09-17, before any submission) — independent audit unavailable; local probes; three corrections found by self-audit

**Codex is unavailable** (usage limit until 2026-09-19 11:33; `checks/codex-design.log`) and the
Claude-family auditor launched in its place died on an API session limit before reading a file.
Per the protocol notice, the pre-job audit is replaced by a written self-audit,
`reports/self-audit-design.md`, listing each claim, the code or JSON it rests on and the check
run. The Codex audit of the final report will be run after 2026-09-19 11:33 if the lane is
still open. This is a substitution, recorded as such.

**Local probes on the real design** (`checks/probe-fitter.json`; CPU-only, no GPU; the design
was built once on the GB10 in 156 s, over the sub-minute rule, recorded as a deviation together
with the 136 s smoke):

- `varpro.bounded_nnls` is **slower with BLAS threads** — 209 s at 1 thread against 585 s at 4
  for the same $m = 1024$ fit on 8064 rows — so every worker runs single-threaded and the chains
  run concurrently: `fit_workers` 8 (job 1) / 12 (job 2), `fit_threads` 1, 16 CPUs requested.
- Cost is about **linear in the rows** (413 s at 16128 rows vs 209 s at 8064, $m = 1024$) and
  **sub-quadratic in $m$** (299 / 209 / 682 s at $m = 512 / 1024 / 2048$ on 8064 rows; the
  pass count, not $m^2$, drives it). The §9 sizing stands as an upper bound.
- **Compression is exact in practice**: on a 17408 × 16384 real design the compressed fit
  returns the identical support (Jaccard 1.0), weights to $8\times10^{-14}$, the identical
  relative fit, gram check $8\times10^{-16}$; the CPU QR took 187 s single-threaded.

**Corrections made before submission**, each a defect of mine found while writing the
self-audit: (1) a reproduction arm whose rule is also the chosen primary rule was silently
de-duplicated away, which would have failed its fidelity gate as "arm missing" — it is now
recorded as an alias (`arm_aliases`) and the audit resolves the comparator through it; (2) the
`max_rom_arms` check was an assertion that would have killed the job after the fit phase — the
arms are now assembled in declared priority order (primary six, dense four, reproduction two,
tight extras) and cut at 16 with every dropped arm recorded; (3) the per-rule walltime cap is
raised from 5400 to 7200 s so that $m = 6144$ on a compressed design is constructible; the
11 h submission deadline and the 20 h limit are unchanged.
