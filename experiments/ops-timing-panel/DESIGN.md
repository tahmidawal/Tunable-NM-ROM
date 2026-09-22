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
  burn-in; **5 retained repetitions** per case per arm (protocol bar is $\ge 5$); medians of
  case medians reported; every individual repetition stored.
- Cohort: the panel's 6 development cases (4 from `eval_seed` 7090702, 2 fresh from 911702),
  identical for every arm.

**Declared deviation, inherited and restated.** The source protocol times all models in one
process. Here the JAX phase and each operator arm are separate processes **in one Slurm
allocation on one GPU**, because JAX preallocates most of the device and would starve PyTorch.
One allocation, one GPU, no ratio across jobs — the property that matters is preserved; the
single-process wording is not. This is stated in the report, not glossed.

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

The paper's rule, applied per row: the comparator for an arm is **the fastest tested
full-order setting whose error does not exceed that arm's error**. All eight settings are timed
in this allocation so the rule has candidates. Speedup $S = T_{\text{FOM}}/T_{\text{arm}}$ is
reported on both timing scopes. Where no tested FOM setting is at least as accurate as an arm,
the row says so instead of quoting a ratio.

## 7. Pass / fail

This is an **evaluation** lane, so the campaign's lane bar is a property of the NM-ROM arms
carried along, not of this lane's new content. Stated so it cannot be moved afterwards:

- **Primary deliverable (binary):** the U-Net and Transolver arms leave this lane with an
  *admissible* cost number — same allocation, same GPU, same cohort, same scope — or they do
  not. Three previously untimed cells become timed cells, or the job failed.
- **Carried-along NM-ROM bar** (campaign protocol): accurate setting $\le 1\%$ worst same-grid
  error **and** $\ge 5\times$ against the same-job FOM chosen by §6. On `bpn301`'s audited
  numbers the accurate arm is 0.905 % but only ~0.03–0.2× the FOM, so this bar is **expected to
  fail on the speed half at $256^2$** and the lane reports that failure rather than softening it.
- **Nothing is selected or tuned here.** No checkpoint, tolerance, rule set or mesh is chosen
  on the strength of anything this job prints.

## 8. What would count as a disagreement with `no-second`

Reported if any of these holds, and reported as a *this-lane* result if none does:

1. an operator arm's recomputed error on the **shared** subset of cases differs from the value
   `no-second`'s audit JSON records for the same case;
2. `fno-large`'s recomputed row differs from `bpn301`'s (same cohort, same code, different
   allocation) by more than the repetition spread;
3. the ordering of the capacity arms by worst error inverts relative to `no-second`'s;
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
  type, and that resubmission is the mechanical retry); any checkpoint SHA256 mismatch;
  `fft_tight` not converged everywhere; an empty log (check disk before suspecting code).
- If the operator phase fails for one arm, the job continues (`set +e` per arm) and the audit
  records that arm as absent rather than inventing a number.
- **No second question.** If the numbers suggest a follow-up (a different mesh, a retrained
  operator, an autoregressive variant), it is written down as open, not run.

## 10. Result integrity

Every item of the campaign checklist: recorded commit and config; `jax_backend=gpu`; f64 and
`JAX_DEFAULT_MATMUL_PRECISION=highest`; one job per directory with `squeue` checked before and
after; independent NumPy re-audit from the saved fields; checksum-verified pull; remote job
directory deleted; only code, configs, logs and small JSON summaries committed (nothing
> 50 MB, no field arrays). `HANDOFF.md` is kept current; the lab log is appended under
`flock`.
