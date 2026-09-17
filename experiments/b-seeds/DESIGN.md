# b-seeds — is the Burgers correction ladder a property of the method or of one lucky checkpoint?

Pre-registered before any job was submitted (2026-09-17). Every amendment is appended to
[§9 Amendments](#9-amendments) with its reason and the date, never edited in place.

Worktree `worktrees/2026-09-17-b-seeds`, branch `exp/2026-09-17-b-seeds`, forked from
`exp/2026-09-16-q-ridge` at `7dc970fc`. Cluster namespace
`/cluster/tufts/paralab/tawal01/b_seeds_20260917/`, one attempt directory per job. Nothing is
merged; nothing is pushed. Paper tables T12 (seed table) and T13 (sealed cohort).

---

## 1. The question

Reviewer 5mgh, M2. Every Burgers $256^2$ number the campaign has produced — the three-layer
decomposition 0.39 / 2.54 / 2.56 % (bank floor / best-found / solved), the dense correction
ladder $q\in\{0,16,32,64,128,256\}$ that passes the knob bar (q-trajdirs, job 3757505,
evolved worst 1.89 / 1.40 / 1.23 / 1.08 / 0.89 / 0.52 %, GPU 333 → 4355 ms) — was measured on
**one** frozen checkpoint, `sep_hfit_dense_mid_N256_dense.pkl` (SHA256 `18f0266ae6f0…`),
on **six opened development cases**. Two things are therefore unknown:

1. whether the ladder's shape (monotone, converged, $\ge2\times$ span in both error and
   cost) survives **retraining the decoder from a different random seed**, and
2. whether the numbers survive a **cohort that was never opened**.

This lane delivers mean ± std over three independently trained seeds on the development
cohort (T12), and the incumbent plus all three seeds on a sealed cohort opened only in the
last job (T13).

---

## 2. What is frozen

Everything about the *ladder measurement* is inherited verbatim from the audited qtd02 job
(q-trajdirs, `config-dense.json`, job 3757505), so numbers are comparable rung for rung:

| item | value |
|---|---|
| mesh / time step | $L=256$ intervals, $\Delta t=0.005$, 50 steps to $t=0.25$, output every 10 steps |
| reference | FOM at 4096 intervals, $\Delta t=3.125\times10^{-4}$, restricted to the query grid |
| development cohort | bitwise abl01's six cases: `params_draw(7090702, 4)` then `params_draw(911702, 2)` |
| directions $C_q$ | `directions.audited`, the OLD rule (static reconstruction residual, seed 20260915, 1024 snapshots, 4 starts, budget 200), nested in $q$, **recomputed per checkpoint** because it is a function of the head |
| head | $u(z,y)=G\,(h_\theta(z)+C_q y)$ |
| solver | `varpro.make_block_lm`, the budget-600 block-damped variable-projection solver; `gtol` $10^{-6}$; IC budget 400; `inner_damping` $10^{-10}$; `tau_y` 0.1 |
| ladders | `dense_m4`: $M=4(K+q)$, $q\in\{0,16,32,64,128,256\}$ (the primary); `dense_fixedM`: $M=256$, $q\le128$ (reported) |
| quadrature | dense (exact) everywhere in the ladder jobs; no EQ rule enters any ladder number |
| controls, same job | `fft_tight` (the same-grid reference), `fft_loose`, `nt1e-2_dt01` |
| timing | 3 repetitions, randomised subject order (`order_seed` 911723), 0.25 s burn-in before each, all retained |
| numerics | float64, `JAX_DEFAULT_MATMUL_PRECISION=highest`, `jax_backend=gpu` asserted |
| driver | `seeds_run.py` = `qtd_run.py` + four non-numerical edits (D2) |

Reported per arm, as qtd02 does: worst over all output times, worst over evolved times
($t>0$), the $t=0$ compression, worst against the 4096-interval reference, best-found, bank
projection floor, median GPU ms with all repetitions, iterations, budget exits, stationarity,
convergence.

**The incumbent runs in every job.** Each seed job also runs the incumbent checkpoint through
the identical development ladder, so (a) the seed-vs-incumbent cost ratio is a same-job
ratio, and (b) every job carries an in-job fidelity gate: the incumbent's eleven dense arms
must reproduce qtd02's worst-reference and same-grid errors to $10^{-9}$ relative.

---

## 3. The sealed cohort

**Definition, fixed here.** `params_draw(17092026, 6)` — six cases from the same family and
the same sequential RNG draw as every other cohort. `checks/sealed-cohort.json` (written by
`make_configs.py` before any job) records the values, their local SHA256, and the minimum
Euclidean distance in parameter space to every draw a Burgers checkpoint of this lineage was
trained, selected or diagnosed on: seeds 0/576 and 1000/4032 (bank and head training), 1/8
(r3 test), 0/128 (direction / EQ families), 7090702/4 and 911702/2 (development), 20260916/64
(b-head-train holdout). All distances exceed $10^{-6}$; all values lie inside the declared
ranges.

**Opening rule.** The sealed cohort is used in **exactly one job, the last one** (`final`),
after the three seed jobs have been collected, audited and archived, with the sealed
configurations (`config-sealed-*.json`) committed before the **first** job. It is never used
for any fit, selection, tuning or diagnosis. Every other job records `final_cohort_unopened:
true`; the final job records `false`, and the audit gates that the flag matches the cohort.
The opening is recorded in §9 and in the lab log.

**What is measured on it.** The incumbent and all three seeds, every rung of both ladders,
same-job FOM controls, three timed repetitions — the qtd02 configuration verbatim on six
different cases. Reported: development vs sealed worst error per rung and their ratio.

---

## 4. The training recipe — what the incumbent actually is, and how it is reproduced

### 4.1 The incumbent's real recipe (from its own job records)

The incumbent is the product of **three** scripts run in two jobs, not of the pipeline
`b-head-train` used:

| stage | script | job | what it does | recorded outcome |
|---|---|---|---|---|
| A bank + head, joint | `sep_burgers_r3.py` (`cluster/run_r3a.sbatch`) | 2835788, A100-40GB | joint auto-decoder training of $g$ (bank, $R=512$, `n_ff` 128, `g_hidden` 1024) and a first head (`h_hidden` 256) on **16384 states of the canonical 576 trajectories** (all $t\le5$ states plus a seeded random rest), 300000 Adam steps, `lr` $10^{-3}$, `p_sub` 4096 points per step, `wd` $10^{-5}$, EMA 0.999, last 10000 steps on all points, `lam_orth` $10^{-4}$, seed 0 | 300000 / 300000 steps in 10685 s, not time-capped, raw weights kept (EMA worse), train recon 0.717 % mean |
| B coefficient extraction | `sep_coeff_extract.py` (`run_dn256b.sbatch`) | 2837431 | on the frozen r3 bank: regenerate **4608** trajectories (576 canonical + 4032 from seed 1000), **pick 131072 of the 235008 states** (all 27648 states with $t\le5$ plus 103424 drawn at random with `default_rng(0)`), project them on the bank (identity $(\ast)$ gated at $1.4\times10^{-13}$) | `sep_coeff_N256_K16_R512.npz` |
| C head refit | `sep_hfit_run.py` arm `mid` (`run_dn256b.sbatch`) | 2837431 | new head $512\times2$ with linear skip, cold start, codes $0.1\,\mathcal N(0,1)$, 200000 Adam steps at batch 4096, `lr` $10^{-3}$, per-snapshot relative loss, `PRNGKey(SEED0+11)`; emitted with the r3 bank | 339 s, final loss $7.7\times10^{-5}$; **this is the incumbent** |

The bank the incumbent uses is therefore *itself trained*, and the head is trained on a
**131072-state pick with an early-time bias** (21 % of the picked states are $t\le5$, against
12 % in the raw data).

### 4.2 Why `b-head-train`'s like-for-like arm under-reproduced (the recorded suspect, confirmed as a real difference)

`b-head-train`'s `d4608k16rec` arm (3.96 % best-found vs 2.54 %) differs from stage C in
exactly these recorded ways: it expanded the 4608 trajectories into **all 235008 states at
stride 1** (1.79× more states, no early-time bias, the same 200000-step budget), and it used
its own projector (`solve_triangular` twice) and its own driver (`fit_ext`) rather than
`sep_hfit.fit`. Whether the state pick or the driver is the cause cannot be settled without a
job; **this lane does not test it**. It removes the question by running the incumbent's own
three scripts, byte-identical to the incumbent's staged manifests wherever the file exists
(`PROVENANCE-COPIES.json` and D4–D5 record the three files that are not).

### 4.3 The decision: full retrain, bank and head, per seed

Each seed $s\in\{1,2,3\}$ runs stages A → B → C with `SEED0 = s` and everything else at the
incumbent's values. `SEED0` seeds, and only seeds: the bank/head initialisation and the
16384-state pick and the 64 full-resolution check rows (stage A), the 131072-state pick and
the 64 identity rows (stage B), the head initialisation, the codes and the minibatch stream
(stage C), and the oracle encoder. **The data draws are fixed** (`data_seed` 0, `extra_seed`
1000, `test_seed` 1) — every seed sees the same 4608 trajectories. So a seed is an
independent draw of the *training randomness* of the whole decoder, bank included, which is
the strongest version of the reviewer's question. The bank floor, best-found and solved error
are reported per seed so each layer can be compared with the incumbent's 0.39 / 2.54 / 2.56 %.

### 4.4 Recorded deviations from the incumbent's job scripts

- **D1 — three environment values.** `TRAIN_ONLY=1` (stage A stops after the checkpoint and
  its full-interior reconstruction check; the round-3 EQ/speed/accuracy protocol that follows
  is not needed and its cost is not paid); `TIME_CAP=0` in stages A and C (the incumbent
  completed every step of both stages well inside its caps, 10685 s / 12600 s and 339 s /
  1500 s; a slower GPU must not silently truncate a seed, so the achieved step count is the
  recipe and the audit gates on `steps_done == 300000` and `== 200000`, `time_capped ==
  false`); `ARMS=mid` (the incumbent job also trained `wide` and `k32` arms it did not emit;
  each arm draws its own `PRNGKey(SEED0+11)`, so omitting them changes nothing in `mid`).
  The `TRAIN_ONLY` hook is the only source change to `sep_burgers_r3.py`; its diff is
  `checks/sep_burgers_r3.diff`.
- **D2 — `seeds_run.py`.** `qtd_run.py` with four non-numerical edits: `final_cohort_unopened`
  read from the config; `cohort_roles` / `cohort_note` from the config; the checkpoint label
  and lane recorded; the completion banner. Diff: `checks/seeds_run.diff`. The in-job
  incumbent fidelity gate (§2) is what proves the edits are numerically inert.
- **D3 — `audit_seeds.py`.** `audit_qtd.py` extended with a cohort mode (`dev`: bitwise
  abl01; `sealed`: values equal to the declared draw to $\le1$ ulp, disjointness recomputed on
  the values, `final_cohort_unopened == false`), the qtd02 comparator for the incumbent
  fidelity gate, the three-layer decomposition at $q=0$, and the training gates of §7. The
  audit imports neither driver nor JAX.
- **D4 — `sep_solvers.py`, `sep_common.py`.** The copies on this branch are the ones the
  incumbent's *head* job staged (hashes equal); the *bank* job staged earlier versions. The
  diffs are default-off additions (`w_extra`, `z_polish`, latent Fourier features) whose
  inactive paths are bit-identical **except** that the bank loss now averages
  `mean(w_ex * mean(err², axis=1))` with `w_ex ≡ 1` instead of `mean(err²)`: the same number,
  reduced in a different order, so a seed's optimisation path is not the incumbent's at the
  last bit. For a seed study that is immaterial; it is recorded so nobody expects `SEED0=0`
  to rebuild the incumbent bitwise.
- **D5 — `burgers2d_film.py`.** The generator the incumbent staged (SHA256 `c521b36a…`) carries
  an exact-Helmholtz preconditioner in the FOM's BiCGStab that the copy under
  `wave2d-rom-latent-stepping/deps` lacks. The incumbent's exact file is staged from this
  lane's `deps/`; its hash equals the incumbent manifests' entry (`PROVENANCE-COPIES.json`).
  `ms_autodecoder.py` / `ms_parametric.py` hash to the manifests' entries as well.
- **D6 — one job per seed.** Stages A–C, then the seed's development ladder (D), then the
  incumbent's development ladder (E), then an optional EQ certification (F, §5), run in ONE
  allocation per seed, so the three seeds run concurrently in three directories and the
  checkpoint never leaves the job that made it until it is collected. The ladder is the
  qtd02 job verbatim; only the checkpoint file differs.

---

## 5. EQ certification per seed (optional, budget permitting)

Stage F runs q-ridge's `q_eqcert.py` (job 3768168's driver) on the seed checkpoint with
`config-eqcert-seed{s}.json` = `config-r3.json` restricted to $q\in\{0,16,32,64\}$, reachable
population, $m=1024$ only, the primary bar $\rho_{\max}\le0.116$ declared in q-ridge, no
reproduction arms, no dense twins (the dense numbers come from stage D). It is `set +e`: a
failure there is recorded in `output/EQCERT-FAILED` and does not void stages A–E. If it does
not fit the budget it is simply not reported.

---

## 6. Pre-registered criteria

Let $e_q^{(s)}$ be the worst-over-evolved-times same-grid error of the `dense_m4` rung $q$ for
checkpoint $s$ on the development cohort, $\bar e_q$ the mean over the three seeds, $\hat
e_q^{(s)}$ the same on the sealed cohort.

**C1 — method, not checkpoint (primary).** The `dense_m4` ladder is non-increasing in $q$ on
the evolved metric with every rung converged on **at least 2 of 3 seeds**. Reported alongside:
the all-times metric, the `dense_fixedM` ladder, and the count of seeds on which each is
monotone.

**C2 — sealed confirmation.** $\hat{\bar e}_q/\bar e_q\le1.5$ at every rung of `dense_m4`
(seed means), and the incumbent's own ratio $\hat e_q^{(0)}/e_q^{(0)}\le1.5$ at every rung.
Reported per checkpoint and per rung.

**C3 — convergence.** Every rung of `dense_m4` converged (zero budget exits, stationary,
completed) on every checkpoint on both cohorts.

**C4 — the knob bar, secondary.** qtd02's criterion — converged non-dominated set of
`dense_m4` spanning $\ge2\times$ in evolved error **and** $\ge2\times$ in cost — holds on at
least 2 of 3 seeds.

**TR — training reproduction bar (a precondition, not a verdict).** A seed reproduces the
incumbent recipe if its development $q=0$ best-found is $\le1.5\times2.5447\,\%=3.82\,\%$ (the
factor at which b-head-train's like-for-like arm, 3.96 %, failed) and its bank floor is
$\le2\times0.3918\,\%$. Seeds failing TR are still run through every stage and reported in
every table; the verdict on C1–C4 is then labelled *conditional on an under-reproduced
recipe*.

**Falsification.**
- F1: C1 fails (monotone on $\le1$ seed) → the ladder's monotonicity is a property of the
  checkpoint, and the paper cannot claim it for the method.
- F2: C2 fails at any rung → the development numbers do not transfer; the sealed numbers are
  reported as the headline and the development ones demoted.
- F3: $\ge2$ seeds fail TR → the recipe is not reproduced in this lane; the seed table is
  provisional, the failure is reported with the training diagnostics (steps, loss, recon,
  oracle, fingerprints), and no training claim is made.

**Iteration rule.** If a seed job dies or a seed fails TR, diagnose from its training JSONs
(step counts, loss traces, reconstruction, oracle, data fingerprints, GPU model) and rerun
that seed **once**, unchanged unless an amendment names the change, within the 8-job cap.
Nothing is dropped: every seed that ran appears in every table.

---

## 7. Gates before any verdict

In-job (recorded in each `result.json`): `jax_backend=gpu`, float64, `highest`; the qtd02
gates verbatim (cohort bitwise abl01 on the development cohort, direction cohorts disjoint,
directions hashed and nested, rank covers the ladder, overdetermined weak system, every
invocation paired, every ROM carries exit and stationarity, step budget 600, checkpoint SHA256
unchanged before/after); stage A/B/C: FOM residual $\le10^{-8}$ on every regenerated
trajectory, identity $(\ast)$ mean deviation $<10^{-6}$, whitening round trip $<10^{-10}$.

Audit (`audit_seeds.py`, NumPy only): every reported error recomputed from the saved fields
($<10^{-9}$); every same-grid discrepancy against the same-job `fft_tight`; repetitions
identical; all reps present; `incumbent_reproduces_qtd02` (eleven arms, $10^{-9}$);
`training_complete` (300000 and 200000 steps, not capped); `pick_is_131072_with_27648_early`;
`training_data_fingerprint_matches_incumbent` (stage A and B sums and sums of squares against
`push_r3a` / `dn256b` JSONs, $10^{-9}$ relative — a value gate, because the FOM's last bits are
GPU-dependent); `seed_checkpoint_sha256_consistent` across the training record, the ladder
job and the sealed job; `sealed_cohort_values_match_declared` ($\le1$ ulp) and
`sealed_cohort_disjoint`; `final_cohort_flag_matches_cohort`.

Local smokes, before submission (`checks/`): `smoke_seeds.py` reproduces the consolidated
saved Burgers case through the retained solver to $\le10^{-12}$ (the parent lane's audited
baseline); the three training scripts run end to end at 64 intervals from a staged tree and
emit a checkpoint the ladder driver loads; `seeds_run.py` runs end to end at 64 intervals on
both a development and a sealed configuration and `audit_seeds.py` passes on its output.

---

## 8. Compute plan and the job cap

| job | attempt | contents | expected A100 time |
|---|---|---|---|
| 1–3 | `s1`, `s2`, `s3` | stages A–F for one seed, incumbent ladder included | ≈ 3.0 h bank + 0.3 h extract + 0.15 h head + 0.9 h seed ladder + 0.9 h incumbent ladder + ≈1.2 h EQ cert ≈ 6.5 h |
| 4 | `final` | sealed cohort: incumbent, seed 1, 2, 3, sequentially | ≈ 4 × 0.9 h |

Submitted as `--gres=gpu:a100:1`, `--mem 180G`, resubmitted as h100 → h200 → l40s only if
still pending after 3 h, never changing the science. Cap: 8 jobs; 4 planned, 4 reserved for
the iteration rule. After each job: checksum collection, local hash verification, the NumPy
audit, Git-chunked archive, then deletion of the exact remote attempt directory.

---

## 9. Amendments

*(none before the first submission)*
