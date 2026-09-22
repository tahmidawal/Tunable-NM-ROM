# ops-timing-panel — DESIGN

Pre-registered before the lane's first GPU job (2026-09-22). Lane of the 2026-09-20
speed-and-accuracy campaign (`reports/2026-09-20-speed-accuracy-campaign-protocol.md`).
Worktree `worktrees/2026-09-22-ops-timing-panel`, branch `exp/2026-09-22-ops-timing-panel`,
forked from `exp/2026-09-17-no-second` @ `ea812685`. Namespace
`/cluster/tufts/paralab/tawal01/opstime_20260922/`.

## 1. The question

The `no-second` lane trained a **U-Net** and a **Transolver** on the Burgers common dataset
and scored their accuracy, but **timed nothing against the ROM or the FOM** — its own report
says so in as many words: *"No speed ratio against the FNO, the ROM or the FOM is stated
anywhere: those were measured in other jobs."* The project's standing rule is that
**cross-job timing is inadmissible**; the only admissible speed statement is a panel measured
inside one allocation on one GPU.

So the question is narrow and mechanical:

> On 2D viscous Burgers at $256^2$, in **one allocation**, what are the query costs and the
> errors of the trained U-Net and Transolver checkpoints, beside the NM-ROM's fast and
> accurate settings, POD-LSPG, the FNO, and the named Newton–BiCGStab full-order solver?

Nothing is trained, tuned or selected here. This is an evaluation job.

## 2. What is copied, and from where

The harness is `b-panel`'s, copied verbatim; `COPIED-FROM.json` records every file with its
source worktree, source path, source commit and SHA256.

| what | from | commit |
|---|---|---|
| `panel.py`, `audit_panel.py`, `smoke_panel.py`, `make_comparators.py`, `cluster/*`, `inputs/*` | `exp/2026-09-17-b-panel` `experiments/b-panel/` | `25434a27` |
| `lib/` JAX modules (`engines`, `iterative_paths`, `accuracy_paths`, `arms`, `ladder`, `ablation`, `varpro`, `topfix`, `eqcert`, `fast`, `ladders`, `sep_common`), `lib/fno_panel.py` | same worktree, six sibling experiment directories | `25434a27` |
| `lib/` torch operator adapter (`model`, `families`, `dataset`, `spectral_conv_f64`) | `exp/2026-09-17-no-second` `experiments/no-second/` | `ea812685` |
| decoder checkpoint `sep_hfit_dense_mid_N256_dense.pkl` ($K=16$, $R=512$) | this worktree, byte-identical to b-panel's (`18f0266a…`) | — |
| 9 operator checkpoints | `operators.json` | see §3 |

Two deliberate changes, and only two:

1. **`cluster/stage.py`** — the lane's own paths and namespace, `lib/` flattened onto
   `PYTHONPATH` instead of six sibling directories, and the FNO phase generalised to an
   **operator phase** that runs the unmodified `lib/fno_panel.py` once per checkpoint.
2. **`audit_panel.py`** — `--fno-name` now takes a list, so several operator arms are scored
   by the same code that scored the FNO. Two new per-arm gates: the saved fields must match
   the SHA256 the timing process recorded, and $t_0$ must be returned bitwise.

`panel.py` is **not modified at all**: the arm set is trimmed purely in the config.

## 3. The arms

### Operator arms (the new content), 9

All nine are archived, frozen, hash-verified checkpoints. `stage.py` refuses to stage a file
whose SHA256 differs from the one the producing job recorded, and `smoke_operators.py` has
already verified all nine locally.

| arm | family | real params | role |
|---|---|---|---|
| `fno-large` | FNO (f64) | 17,877,317 | parent lane's selected arm; also a same-GPU cross-check against b-panel `bpn301` |
| `unet-small` / `unet-medium` / `unet-large` / `unet-refine` | U-Net (f32 net, f64 I/O) | 4.37 M / 7.76 M / 17.46 M / 7.76 M | `unet-refine` is the **validation-selected** arm; `unet-medium` is the arm with the **lowest worst validation case** (3.9622 % vs 7.5176 %), which the mean-optimising selection rule did not choose |
| `tsol-small` / `tsol-medium` / `tsol-large` / `tsol-refine` | Transolver (f32 net, f64 I/O) | 3.11 M / 6.95 M / 12.32 M / 3.11 M | `tsol-refine` is **validation-selected**; `tsol-large` has the **lowest worst validation case** (6.3953 % vs 9.3183 %) |

Both the selected and the best-worst-case arm of each family are reported, per the recorded
finding that validation selection once picked a worse worst case. Capacity arms are included
because they cost seconds each.

The operators are **direct multi-time outputs**, not autoregressive rollouts: one forward pass
returns $u(t_1)\dots u(t_5)$ and the supplied $u(t_0)$ is prepended bitwise. This is a recorded
fact, not something this lane re-derives.

### Same-allocation companions, 21 subjects (b-panel's, endpoints only)

| group | subjects |
|---|---|
| **NM-ROM fast** | `q0_M64_eqcert_g1em06_fastL4` (b-speed kernel), and `q0_M64_{eqcert,eqtop}_{g1em06,g0p001}` |
| **NM-ROM accurate** | `q256_M1088_{eqcert,eqtop}_{g1em06,g0p001}` |
| **dense-residual references** | `q0_M64_dense_g1em06`, `q256_M1088_dense_g1em06` |
| **POD-LSPG control** | `pod256_M1024_dense`, `pod512_M2048_dense` |
| **full-order (Newton–BiCGStab, FFT-Helmholtz preconditioner)** | `fft_tight`, `dense_tight`, `nt1e-2_dt01`, `nt1e-2_dt005`, `nt1e-3_dt01`, `nt1e-3_dt005`, `nt1e-4_dt01`, `nt1e-4_dt005` |

The middle rungs $q \in \{16,32,64,128\}$, the extra dense arm and the free bank are **dropped**:
`bpn301` already measured the whole $256^2$ ladder in its own allocation and this job is not a
re-measurement of it. The endpoints are kept because the operator arms need a fast and an
accurate NM-ROM setting to sit beside, in the same allocation.

## 4. Timing scope

Unchanged from b-panel / `no-audit`, which is the point:

- **GPU-query scope**: begins with the initial field and the parameters already resident on
  the GPU, ends when the complete six-time trajectory is resident on the GPU. For the
  operators this includes coordinate-channel construction, normalisation, the forward pass,
  boundary masking and trajectory assembly; for the ROM and FOM it is the complete query.
- **Complete-query scope**: GPU query **plus** the separately timed device-to-host copy of the
  finished trajectory. Reported as `complete ms`.
- Synchronised on both sides of every repetition; every timed block preceded by its own
  burn-in (**20** untimed identical queries for the operator arms, `burn_seconds` of untimed
  work for the JAX subjects); **5 retained repetitions** per case per arm (protocol bar is
  $\ge 5$); every individual repetition stored.
- **The reported median is the pooled median over (case $\times$ repetition)**, which is what
  `audit_panel.py` computes and what b-panel's published tables report. It is not the median
  of case medians; both exist in the JSON and the report says which column is which.
- Cohort: the panel's 6 development cases (4 from `eval_seed` 7090702, 2 fresh from 911702),
  identical for every arm. `audit_panel.py` asserts per operator arm that all six cases are
  present, that the device and host arrays have equal length, and that the length is 5.

**Declared deviations, inherited and restated.** Both are stated in the report, not glossed.

1. *Separate processes.* The source protocol times all models in one process. Here the JAX
   phase and each operator arm are separate processes **in one Slurm allocation on one GPU**,
   because JAX preallocates most of the device and would starve PyTorch. One allocation, one
   GPU, no ratio across jobs — the property that matters is preserved; the single-process
   wording is not. `audit_panel.py` gates that each operator arm's recorded GPU name equals the
   JAX phase's.
2. *The complete-query scopes are not byte-identical.* For the JAX subjects the complete query
   includes the host-to-device upload of the input as well as the device-to-host download of
   the trajectory; for the operator arms the input is already resident and only the download is
   added. The difference is a few hundred microseconds of upload that the operators are not
   charged, so it **favours the operators**, and it is inherited from b-panel unchanged so that
   the FNO row remains comparable with `bpn301`. Every operator speedup is therefore an upper
   bound on the complete-query scope, and the GPU-query column is unaffected.
3. *The scored fields come from an extra untimed query*, not from a retained timed repetition
   (`fno_panel.py` as written). The forward pass is deterministic in `eval` mode, and the audit
   checks the saved fields against the hash the timing process recorded and checks $t_0$
   bitwise against this panel's own `fft_tight` initial state, but repetition-to-repetition
   output equality is not proved for the operator arms as it is for the JAX subjects.

## 5. Accuracy

The same metric for every arm, recomputed independently in NumPy from the saved fields by
`audit_panel.py`:

$$E(\text{case}) = \max_{k} \frac{\lVert \hat u(t_k) - u^{\star}(t_k)\rVert_2}{\lVert u^{\star}(t_0)\rVert_2},$$

with $u^{\star}$ the **same-job converged `fft_tight` solve** at $256^2$ (`worst all %`,
`worst evolved %` over $k \ge 1$) and, separately, the 4096-interval refined reference
(`vs ref %`). Worst and median over the 6 cases are both reported.

**The stored accuracy numbers of the forked lane are not trusted and not reused.** Every error
in this lane's table is recomputed here, on this lane's cohort. The `no-second` numbers were
computed on a different cohort (32 validation cases; an 8-case ROM/FOM diagnosis set), so a
difference is expected and is **not** by itself a disagreement — §8 says what would be.

## 6. The FOM rule

The paper's rule, applied per row, with the metric fixed here so it cannot be chosen later:
the comparator for an arm is **the fastest tested full-order setting whose `worst evolved
same-grid` error over the 6 cases does not exceed that arm's `worst evolved same-grid` error**,
where "fastest" is measured on the same timing scope as the ratio being quoted. All eight
settings are timed in this allocation so the rule has candidates. Speedup
$S = T_{\text{FOM}}/T_{\text{arm}}$ is reported on both timing scopes, each against the FOM
selected on that scope. Where no tested FOM setting is at least as accurate as an arm, the row
says so instead of quoting a ratio. The selection is implemented in this lane's report
generator, not in `audit_panel.py`, and reads only the audit JSON.

## 7. Pass / fail

This is an **evaluation** lane, so the campaign's lane bar is a property of the NM-ROM arms
carried along, not of this lane's new content. Stated so it cannot be moved afterwards:

- **Primary deliverable (binary):** the U-Net and Transolver arms leave this lane with an
  *admissible* cost number — same allocation, same GPU, same cohort, same scope — or they do
  not. Three previously untimed cells become timed cells, or the job failed.
- **Carried-along NM-ROM bar** (campaign protocol), stated on a named arm and a named scope:
  the accurate arm is **`q256_M1088_eqtop_g0p001`**, and the bar is worst all-times same-grid
  error $\le 1\%$ **and** $\ge 5\times$ against the §6 FOM on the **GPU-query** scope. On
  `bpn301`'s audited numbers that arm is 0.9053 % and ~0.02–0.2× the FOM, so the bar is
  **expected to fail on the speed half at $256^2$**; the lane reports that failure rather than
  softening it.
- **No checkpoint, tolerance, rule set or mesh is selected or tuned here.** Nothing is chosen
  on the strength of anything this job prints. "Nothing is trained" means no network weights
  are updated: the POD bases, the EQ weight refits and the cold starts are still fitted
  in-job, from training-seed trajectories, exactly as `bpn301` fitted them.
- **A row is printed only if its arm's gates passed.** `audit_panel.py` marks FOM and operator
  rows `admissible=True` unconditionally (that flag is b-panel's ROM-convergence rule); this
  lane's report generator additionally refuses to print an operator row whose own gates
  (cohort/reps, $t_0$, field hash, GPU match) failed, and the collection step refuses the job
  if the audit's `failed` list is non-empty.

## 8. What would count as a disagreement with `no-second`

Reported if any of these holds, and reported as a *this-lane* result if none does:

1. an operator arm's recomputed error on a case that is **also** in `no-second`'s cohort
   differs from the value `no-second`'s audit JSON records for that same case by more than
   $10^{-6}$ relative (both are f64 recomputations of the same metric on the same field);
2. `fno-large`'s recomputed **error** row differs from `bpn301`'s, which is the same cohort,
   the same metric and the same code. Only the error is compared: `bpn301`'s *timings* are from
   another allocation and comparing them is exactly what the project forbids, so the cost
   columns of the two jobs are reported side by side as separate measurements and no ratio is
   formed between them;
3. the ordering of the capacity arms by worst error inverts relative to `no-second`'s — noted
   as an observation, **not** as a disagreement, because the cohorts differ (§5);
4. a checkpoint's SHA256 does not match the recorded one (this aborts staging).

Different cohorts giving different absolute percentages is **not** a disagreement and will be
labelled as such.

## 9. Stop rules

- **One job.** `opt101`, A100-80G, `--mem 180G`, 6 h wall (expected ≈ 45 min: `bpn301` spent
  ≈ 70 min on 47 subjects and 3 reps; this job has 21 subjects and 5 reps, and the ~27 min
  4096-interval reference dominates either way).
- Lane cap **3 jobs total, 1 running**. A second job only to recover a *mechanical* failure
  (staging, OOM, node fault) with no change of question; a third never without a recorded
  reason. A failed job that produced no science number does not count (b-panel rule 13).
- **Abort conditions:** preflight not printing `jax_backend=gpu` (resubmit on another GPU
  type, and that resubmission is the mechanical retry); a missing training index (exit 43);
  any checkpoint SHA256 mismatch; an empty log (check disk before suspecting code). Neither
  `panel.py` nor `audit_panel.py` exits non-zero on a failed gate, so the collection step
  **reads the audit's `failed` list and refuses the job if it is non-empty** — in particular if
  `fft_tight_converged_everywhere` failed, since every same-grid error is measured against it.
- If the operator phase fails for one arm, the job continues (`set +e` per arm) and the audit
  records that arm as absent rather than inventing a number.
- **No second question.** If the numbers suggest a follow-up (a different mesh, a retrained
  operator, an autoregressive variant), it is written down as open, not run.

## 10. Result integrity

Every item of the campaign checklist: recorded commit and config; `jax_backend=gpu`;
`JAX_DEFAULT_MATMUL_PRECISION=highest` and f64 throughout the JAX phase and throughout every
arm's data, features, boundary mask, trajectory assembly and error metric — the U-Net and
Transolver *networks* themselves run in IEEE float32 with TF32 disabled, by their own training
design, and `check_dtypes` enforces each network's declared parameter dtype; one job per directory with `squeue` checked before and
after; independent NumPy re-audit from the saved fields; checksum-verified pull; remote job
directory deleted; only code, configs, logs and small JSON summaries committed (nothing
> 50 MB, no field arrays). `HANDOFF.md` is kept current; the lab log is appended under
`flock`.

## 11. Independent audit

`codex exec -m gpt-6-astra`, headless, output in `checks/codex-design-audit.md`. The first
attempt under `-s read-only` could not start its sandbox in this environment
(`bwrap: loopback: Failed RTM_NEWADDR`); the audit was re-run with sandboxing bypassed and an
explicit read-only instruction, and `git status` afterwards confirmed it wrote only its own
output file. Six findings, disposition:

| # | finding | disposition |
|---|---|---|
| 1 | config trim safe (21 subjects, no dangling names) — but `checks/comparators.json` was missing, which would have failed all four cross-job fidelity gates | **accepted**: `comparators.json` copied in from b-panel |
| 2a | the reps gate covers JAX subjects only; `zip` would silently truncate unequal operator timing arrays | **accepted**: new per-arm gate `operator_cohort_and_reps_<name>` (all 6 cases, equal-length arrays, length 5) |
| 2b | the $t_0$ gate trusted a boolean the timing process recorded | **accepted**: now compares the operator's saved $t_0$ bitwise against this panel's `fft_tight` initial state |
| 2c | every operator arm was labelled `family='fno'` | **accepted**: `family` is now `unet` / `transolver` / `fno`; `kind` stays `fno` because the scoring branches key on it |
| 3 | staging complete, hashes really abort — but the training index is an unstaged external path whose absence would make the disjointness gate vanish | **accepted**: the batch script now fails with exit 43 if it is missing |
| 4 | §4 described medians of case medians (the audit reports pooled medians); the JAX and operator complete-query scopes differ; burn-in of 5 is thin; scored fields come from an extra untimed query | **accepted as text**: §4 now states the pooled median, declares the scope asymmetry and which side it favours, and declares the untimed scoring query. Burn-in raised 5 → 20 for the operator arms. The scope asymmetry is *not* code-fixed, deliberately, so the FNO row stays comparable with `bpn301` |
| 5 | operator metadata carried no GPU/cohort binding; operator and FOM rows are unconditionally `admissible`; failed gates do not make anything exit non-zero | **accepted**: GPU-match gate added; §7 and §9 now bind row printing and job acceptance to the gate results rather than to the `admissible` flag |
| 6 | §8.2 invited a forbidden cross-allocation timing comparison; §6/§7 left the metric, arm and scope unnamed; §8.1 had no tolerance; §10's blanket f64 claim conflicts with the f32 networks; "nothing is trained" vs the in-job POD fit; "three untimed cells" undefined against nine arms | **accepted**: §6, §7, §8 and §10 rewritten as above; §8.2 now compares errors only and says explicitly that the two jobs' timings are never divided |

No finding was rejected.

## A1 — amendment (2026-09-22, after `opt101` landed): the four DeepONet arms, job `opt201`

The sibling lane `ops-deeponet-b2d` (`exp/2026-09-22-ops-deeponet-b2d` @ `306c939d`, job
`4179556`) trained DeepONet on the same Burgers $256^2$ data, split, reference, metric, budget
and selection rule, and left `reports/timing-handoff.json` for this harness. It makes no speed
claim of its own, for the same reason `no-second` made none.

**`opt201` re-runs the whole `opt101` panel with four arms added**, so that the entire 2D
operator comparison comes out of one allocation rather than being stitched across two. The
config, cohort, repetitions, burn-in, timing scopes, FOM rule and gates are **unchanged**;
`opt101`'s numbers are not edited or merged into `opt201`'s — the two jobs stand separately and
the report quotes `opt201`.

### The added arms

| arm | role in the sibling lane | real params | stop reason |
|---|---|---|---|
| `don-small` | **validation-selected** | 2,569,349 | early stopping |
| `don-medium` | best worst case on that lane's 32 validation cases; not selected | 4,694,789 | early stopping |
| `don-refine` | capacity (lower-lr retrain of the selected capacity) | 2,569,349 | early stopping |
| `don-large` | capacity | 10,258,181 | early stopping |

### The module swap, and the parity gate that licenses it

`lib/families.py` and `lib/model.py` are replaced by the DeepONet lane's versions, which add a
`DeepONet2d` class, a `deeponet` dispatch branch, a `FAMILIES` entry and a `smoke_bounded`
branch. The diff reads as purely additive, but reading is not a gate:
`checks/parity_families.py` loads **each of `opt101`'s nine checkpoints twice** — once under the
modules `opt101` ran, once under the replacements — and requires the predicted trajectories to
be **bitwise equal**. It passed for all nine, `max_abs_diff = 0.000e+00`. Without that pass this
amendment does not proceed, because the nine existing rows would otherwise not be
re-measurements of the same models.

### Facts that travel with every DeepONet row

Pre-registered here so they cannot be dropped from the report later. All four come from the
sibling lane's own record:

1. **All four arms ended by EARLY STOPPING, not on the wall budget.** The 3000 s per-capacity
   budget did **not** bind. This is the opposite of the U-Net, Transolver and FNO arms, every
   one of which ended on its wall budget.
2. **`still_improving = no` is vacuous here.** It is defined against the patience window; an
   arm that early-stopped satisfies it by construction and it carries no information about
   whether the configuration had converged.
3. **Training loss was still falling in all four histories** when early stopping fired on the
   validation criterion.
4. **128 training cases.** A **data-limited** reading of the DeepONet result is live and the
   sibling lane did not separate it from a capacity reading. Therefore: **this lane does not
   present DeepONet's weakness as an architecture ceiling**, and any sentence that could be read
   that way is a defect in the report, not a finding.

### Accuracy must travel with the speed row

The handoff's selected-arm-vs-selected-arm ratios on the 32 validation cases (generated by that
lane, not typed here) are carried into the report beside the DeepONet timings, and the report
states the sibling lane's own caveat: these arms are far less accurate than the FNO, U-Net and
Transolver arms measured on the same data, split and metric, and the panel's existing note —
every operator arm already sits above the $256^2$ discretisation error — applies *a fortiori*.

### Disagreement check

The sibling lane's `diagnosis-8` cohort is the **calibration** split, the same split
`no-second` used and a different split from this panel's **development** cohort. Its worst
values (`don-small` 16.0229 %, `don-medium` 18.9841 %, `don-refine` 21.3719 %, `don-large`
26.6330 %) are therefore **not comparable** with the percentages this job computes, exactly as
§8 already says for `no-second`. What is compared is the **rank ordering** of the four arms,
and any SHA256 mismatch (which aborts staging). Every error is recomputed here from the saved
fields in NumPy; no value from the handoff is trusted as a measurement.

### Budget

`opt201` is job **2 of 3**. The stop rules of §9 are unchanged.
