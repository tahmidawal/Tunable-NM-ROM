# Independent design audit of the `b-seeds` lane (pre-submission)

**Provenance of this audit.** Produced on 2026-09-17 by a Claude agent (Fable 5.1) with **no access to the
lane's conversation**, as a substitute for the Codex audit that was unavailable (usage limit). Everything
below was derived by reading files under `worktrees/2026-09-17-b-seeds/` only: no GPU work, no JAX, no
job submission, no file modified except this one. Line numbers refer to the files as they stood at
commit `13155c88` plus the uncommitted working tree (see item 19).

Severity: **BLOCKER** = the pre-registration as written cannot deliver its own verdict; **MAJOR** = a
reviewer would reject or the job would be wasted; **MINOR** = fix cheaply before the first job;
**NOTE** = context or a verified non-issue.

---

## Findings

1. **BLOCKER — the incumbent fidelity gate is set at a tolerance the comparator itself did not meet.**
   `make_configs.py:89-93` writes `qtd02_expectations` with `tolerance: 1e-9` and *no second tier*;
   `DESIGN.md:57-60, 236-237` says the eleven incumbent arms "must reproduce qtd02 ... to $10^{-9}$
   relative"; `audit_seeds.py:606-637` gates `reproduces_qtd02_<arm>` on that single tier.
   But `comparators/qtd02-audit.json` records that qtd02 **failed its own 1e-9 first tier** against
   cclad01 on `old_q64_M256_dense`, `old_q64_M320_dense` (1.07e-9) and `old_q256_M1088_dense`
   (3.03e-9), passing only the `second_tier_tolerance: 1e-3`, and that its directions hash
   (`7d51a88c…`) did **not** match its expected comparator (`f270e5bf…`). Cross-job last-bit drift
   (GPU model, node) is therefore already documented for this exact pipeline; the seed jobs may land
   on A100-80GB / H100 while qtd02 ran on an A100-PCIE-40GB. The gate will very likely fail on some
   arms in every seed job, and the DESIGN reads a pass as the proof that the `seeds_run.py` edits are
   inert (D2, line 142-143) — so a spurious fail voids the lane's own inertness argument, and a post
   hoc waiver is exactly what pre-registration is meant to prevent.
   *Smallest fix:* add `second_tier_tolerance: 1e-3` to every `qtd02_expectations` entry
   (`make_configs.py:89`), make the audit gate on the second tier and report the 1e-9 tier as an
   informative probe (as `old_directions_hash_matches_comparator` already is, `audit_seeds.py:359`),
   and amend `DESIGN.md` §2/§7 accordingly before the first job.

2. **MAJOR — the inherited cclad01/btq201 `fidelity_expectations` will re-fail in every incumbent block.**
   `config-dev-incumbent.json:137-200` keeps qtd02's `fidelity_expectations` verbatim; the audit's
   `reproduces_<arm>` gates (`audit_seeds.py:~590-604`) use `passes_declared` (first tier). The three
   arms above will therefore appear in `failed` for every `s*-ladder_incumbent` audit and in the
   report's gate section (`generate_b_seeds.py` "Gates" block), exactly as they did for qtd02.
   *Fix:* set `fidelity_expectations: {}` for the incumbent dev config (qtd02 is now the comparator,
   item 1), or gate on `passes_second_tier`.

3. **MAJOR — C2/F2 are mis-calibrated: the sealed cohort is systematically harder than the development cohort.**
   From `checks/sealed-cohort.json`: viscosity min/median 0.0105/0.0193 (sealed) vs 0.0141/0.0317
   (dev); amplitude max 1.99 vs 1.88; the sharpness proxy $a/\nu$ is {27.7, 31.0, 32.1, 90.2,
   **125.6, 153.3**} sealed vs {19.8, 25.8, 34.1, 41.9, 72.4, 108.1} dev — two sealed cases are
   harder than anything the ladder has ever been measured on. qtd02's per-case evolved error at
   $q=0$ tracks $a/\nu$ almost monotonically ($108\to1.89\,\%$, $72\to1.47$, $42\to1.21$, $34\to0.70$,
   $26\to0.80$, $20\to0.42$; `comparators/qtd02-audit.json`, arm `q0_M64_dense`). C2 compares the
   **max** over cases, so a sealed/dev ratio above 1.5 is plausible from case difficulty alone, for
   the incumbent and every seed alike — and F2 (`DESIGN.md:213-214`) would then conclude "the
   development numbers do not transfer", which conflates difficulty with checkpoint over-fitting.
   *Fix (amendment before job 1, no redraw):* pre-register a difficulty-normalised co-criterion —
   the sealed/dev ratio of (solved evolved error / best-found) per rung, or a per-case comparison
   against each case's own best-found and bank floor, both of which the driver already writes
   (`seeds_run.py:402-416`) — and require per-case sealed errors in T13. Keep raw C2 as the headline
   number but state now that a C2 fail with a normalised pass is read as "harder cohort", not
   "non-transfer".

4. **MAJOR — the L40S fallback would kill the seed jobs.**
   `DESIGN.md:260-261` allows resubmission "h100 → h200 → l40s ... never changing the science";
   `stage.py:24` gives seed jobs 16 h; stage A is 300 000 f64 steps that took 10 685 s on an
   A100-40GB (`runs/push_r3a/.../sep_burgers_r3_N256_K16_R512.json`, `train.seconds`). L40S FP64
   throughput is roughly one tenth to one twentieth of an A100's, so stage A alone exceeds the wall
   clock; with `TIME_CAP=0` (D1) the scheduler kills the job with no checkpoint, burning one of the
   eight allowed jobs and ~16 GPU-hours. *Fix:* restrict the seed-job fallback to h100/h200, or set
   `HOURS_SEED` to ≥48 h when `gpu == 'l40s'` (`stage.py:214`). The final job (ladders only) is fine.

5. **MAJOR — the iteration rule cannot be executed by `stage.py`.**
   `stage.py:207-213` accepts only `s<digits>` (`int(attempt[1:])` becomes `SEED0`), and
   `stage.py:225-226` refuses an existing `runs/<attempt>` (`exist_ok=False`). A rerun of seed 1
   (`DESIGN.md:219-222`) has no valid name: `s1r` → `SystemExit`, `s11` → **SEED0 = 11** and a missing
   `config-dev-seed11.json`; reusing `s1` collides locally and with the remote `b_seeds_20260917/s1`.
   *Fix:* parse `^s(\d+)([a-z]*)$`, take the digits for `SEED0` and keep the suffix in the attempt /
   remote name; assert `config-dev-seed{seed}.json` exists.

6. **MAJOR — a rerun (or a fourth checkpoint) cannot reach the tables the DESIGN promises it will.**
   `DESIGN.md:221-222`: "every seed that ran appears in every table". But `FINAL_BODY`
   (`stage.py:185`) hard-codes `incumbent seed1 seed2 seed3`; `make_configs.py:20` writes sealed
   configs for seeds 1–3 only; `generate_b_seeds.py:21,131,136` requires `len(seeds) == 3` and keys
   `dev_seed` by `checkpoint_label`, so two runs labelled `seed1` overwrite each other. *Fix:* label
   reruns `seed{s}{suffix}`, derive the final job's checkpoint list from `checkpoints/*.pkl`, drop the
   `== 3` requirement, and say in §6 which of the two checkpoints of a rerun seed enters C1/C4 (both,
   as separate rows, is the honest reading of "nothing is dropped").

7. **MAJOR — T12 averages absolute GPU milliseconds across jobs.**
   `generate_b_seeds.py` T12 column "GPU ms, mean ± std (seeds)" (`ms(gm, 1)` in the per-rung row)
   pools `median_gpu_ms` from three different allocations and, under item 4's fallback, different GPU
   models. That is the cross-job cost comparison the protocol forbids (`DESIGN.md:58`,
   `audit_seeds.py:637`). The same-job seed/incumbent ratio column is the right quantity and is
   already there. *Fix:* delete the absolute column (or replace it by per-job "ms @ GPU model").

8. **MAJOR — `OUTPUTS.sha256` / `ALL-DONE` are written only after the non-fatal stage F.**
   `stage.py:166-178`: if `q_eqcert.py` (≈1.2 h estimated; the unrestricted qrg304 took 10 502 s)
   runs into the 16 h wall, SLURM kills the job after stages A–E completed, and `collect.py:33` then
   refuses the whole attempt (`sha256sum -c OUTPUTS.sha256` with `check=True`). *Fix:* write
   `OUTPUTS.sha256` and a `STAGES-AE-DONE` marker right after stage E and rewrite after F; or run F
   under `timeout` derived from `squeue -h -j "$SLURM_JOB_ID" -O TimeLeft`.

9. **MAJOR — the pre-submission smokes the DESIGN requires have no record on the branch.**
   `DESIGN.md:245-249` lists three local smokes "before submission (`checks/`)". `checks/` contains
   only `sealed-cohort.json` and three diffs; `smoke_chain.py:7,135` would write
   `checks/smoke-chain.json` and `smoke_seeds.py` a JSON, neither exists. Whether the staged layout,
   the three training scripts and the audit actually run end to end is therefore unverified *on the
   branch* (a scratch directory `b-seeds-smoke` exists outside the worktree, but "if it only exists
   in a scratchpad, it does not exist"). *Fix:* run both smokes, commit their JSONs, and cite them in
   §7 before staging.

10. **MINOR — question A, differences not in D1–D6.** Every numerical input is like-for-like (see
    item 25), but these mechanics differ from `run_r3a.sbatch` / `run_dn256b.sbatch` and are not
    recorded: global `OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8`, `JAX_ENABLE_X64=true`, `TMPDIR`,
    `XDG_CACHE_HOME`, `MPLCONFIGDIR` (`stage.py:95-98`); `--mem 180G` vs 120G/160G, `--time 16h` vs
    8h/4h, `--exclude pax007` (the node that trained the incumbent head), `--qos normal`
    (`stage.py:82-89`); `set -euo` vs dn256b's `set -uo` (no `-e`); the sibling layout
    (`experiments/<name>/`) instead of the incumbent's flat `code/` + `code/deps/` — module hashes
    are equal but `sys.path` order differs (`sep_common.py:43-50` picks `dirname(HERE)/<name>` now,
    `HERE/deps/<name>` then); output names `hfit_full.json` / `sep_hfit_seed{s}.pkl`; and the
    incumbent's hfit-full ran after two learning-curve `sep_hfit_run.py` processes in the same
    allocation (`run_dn256b.sbatch:51-57`, separate processes, no state carried). *Fix:* add a D7
    listing them as non-numerical.

11. **MINOR — "in-job fidelity gate" is audit-time.** `DESIGN.md:57-60` and D2 (`:142-143`) say every
    job "carries an in-job fidelity gate"; `seeds_run.py` never reads `qtd02_expectations` or
    `fidelity_expectations` (the diff `checks/seeds_run.diff` adds no such read) — the comparison
    happens only in `audit_seeds.py:606`. *Fix:* wording ("audit-time, from the job's `result.json`").

12. **MINOR — D4 cannot be audited from this branch.** HEAD's `sep_solvers.py` (`19df8382…`) and
    `sep_common.py` (`a74f0279…`) equal the dn256b manifest but not the push_r3a manifest
    (`5af7056b…`, `1f1c2796…`), and the r3a-era files are not in the repository, so the claim
    "inactive paths bit-identical except the `w_ex` reduction order" (`DESIGN.md:149-156`) is
    unverifiable here. *Fix:* commit `checks/sep_solvers_r3a.diff` and `checks/sep_common_r3a.diff`
    with both hashes.

13. **MINOR — the sealed cohort is exercised before the final job.** `smoke_chain.py:91-99` runs
    `config-sealed-seed1.json` (the real `eval_seed` 17092026) at 64 intervals on a toy checkpoint and
    writes its rung errors into `checks/smoke-chain.json`. Non-informative, but it contradicts the
    letter of §3 ("used in exactly one job"). *Fix:* smoke the sealed *mode* with a throwaway seed and a
    throwaway declared-cohort file, or record the exposure in §9.

14. **MINOR — C2 is vacuous on a missing incumbent block and asymmetric on missing seeds.**
    `generate_b_seeds.py` C2 block: `ratio_incumbent is None or ≤ 1.5` passes when the incumbent's
    sealed audit is absent; `mean_std` drops `None`, so dev and sealed means may be over different seed
    subsets. *Fix:* require all four sealed blocks and compute both means on the seed intersection.

15. **MINOR — hand-typed numbers in the audit and the generator.** `audit_seeds.py:754-755`
    (0.3918 / 2.5447 / 2.5629), `generate_b_seeds.py` TR block (`1.5 * 2.5447`, `2 * 0.3918`) and the
    incumbent training row (`'NVIDIA A100-PCIE-40GB'`, `300000`, `200000`, `0.211`). All check out
    against `comparators/qtd02-audit.json` (0.39184 / 2.54466 / 2.56287) and 27648/131072 = 0.2109,
    but they violate the lane's own rule. *Fix:* read them from the comparator and the incumbent JSONs.

16. **MINOR — the incumbent checkpoint hash is never gated.** `make_configs.py:24` defines
    `INCUMBENT_SHA` and never uses it; the audit's `--expect-checkpoint-sha256` is optional. *Fix:* put
    `expected_checkpoint_sha256` into both incumbent configs and gate it in the audit.

17. **MINOR — `checkpoints/` mechanics are undocumented.** `stage.py:249-254` expects
    `checkpoints/sep_hfit_seed{s}.sha256`, but neither `collect.py` nor `preserve_archive.py` writes
    it; three 36 MB pickles would be committed to Git on top of the chunked archive that already
    contains them. *Fix:* have `collect.py` write the `.sha256` from `output/train/TRAIN-SHA256.txt`
    and copy the pickle; say so in §8.

18. **MINOR — nothing forbids a seed (re)submission after the final job.** §3 and §6 leave open a
    rerun after the sealed cohort has been opened, which would be selection on sealed numbers.
    *Fix:* one sentence in §6: no seed job is submitted after the final job's submission; any later
    rerun voids T13.

19. **NOTE — the working tree is dirty**, so `stage.py:229-230` will refuse to stage:
    `experiments/b-seeds/cluster/collect.py` modified, `README.md` and `reports/` untracked. Commit first.

20. **NOTE — monotonicity is effectively strict.** `audit_seeds.py:55-56` uses an absolute 1e-12
    tolerance on percent values; a 1e-6 pp uptick fails C1. Same as qtd02; say so in §6.

21. **NOTE — C1 wording.** "non-increasing ... with every rung converged on at least 2 of 3 seeds" is
    implemented as (monotone **and** all-converged) on the same seed for ≥2 seeds
    (`generate_b_seeds.py` `c1_count`). State that explicitly.

22. **NOTE — `LOOSE=1` in stage B** is a flag the script marks "for SMOKE TESTS ONLY"
    (`sep_coeff_extract.py:73-78`); it is inherited verbatim from `run_dn256b.sbatch:44`, so it is
    like-for-like and the audit compares it (`recipe_values_match_incumbent`, key `loose`), but §4.1
    should mention it.

23. **NOTE — TR's 1.5× factor** is calibrated on b-head-train's 3.96 %, measured through a different
    driver and state pick (§4.2). Acceptable as a precondition; it is a heuristic, not a bar with a
    provenance in this pipeline.

24. **NOTE — the time budget holds on A100/H100.** qtd02 took 3 152 s (≈0.9 h; `qtd02-result.json`),
    the unrestricted qrg304 10 502 s, so the ≈6.5 h estimate is plausible against the 16 h limit.

25. **NOTE — what was checked and found correct** (questions A, B, D).
    - *Recipe:* stage B's `env` line equals `run_dn256b.sbatch:44-46` token for token; stage A equals
      `run_r3a.sbatch:23-34` except `SEED0`, `TIME_CAP=0`, `TRAIN_ONLY=1`, `OUT_PREFIX`; stage C equals
      `run_dn256b.sbatch:58-61` except `SEED0`, `ARMS=mid`, `TIME_CAP=0`, output names.
      `TIME_CAP=0` means *off* (`sep_solvers.py:493`, `sep_hfit.py:298`). Data draws are fixed:
      `data_seed = bc.SEED = 0` (`burgers2d_film.py:72`, `blat_common.py:94`), `TEST_SEED = 1`
      (`blat_common.py:167`), `EXTRA_SEED = 1000`; `SEED0` seeds exactly what D-4.3 claims
      (`sep_burgers_r3.py:326,339,365`; `sep_coeff_extract.py:112,220`; `sep_hfit_run.py:293,380`);
      `mid` is first in the incumbent's `ARMS` and each arm draws its own `PRNGKey(SEED0+11)`
      (`sep_hfit_run.py:352-382`), so `ARMS=mid` is inert. Every `TRAIN_FILES` entry at HEAD hashes to
      the dn256b manifest; the three relocated deps hash to both manifests (`PROVENANCE-COPIES.json`).
      `sep_burgers_r3.py` differs from both manifests only by the `TRAIN_ONLY` hook, placed after the
      checkpoint save and the full-interior check with `complete = True` (`checks/sep_burgers_r3.diff`).
    - *Layout:* `sep_common._bootstrap` (`:43-50`) finds the siblings, `blat_common` (`:51-58,63-69`)
      finds `deps/burgers2d-coord-rom`, `deps/multistage-precision` and `nda_arch`, `pro_common:48`
      finds its `deps/`; `ms_autodecoder` imports only `ms_parametric`. The final job needs no deps
      (`engines.py:16-17` imports only `sep_common`). Emitted pickles are NumPy-only
      (`sep_common.py:281-284`, `sep_hfit_run.py:387-411`) so the JAX-free audit loads them. All
      `PYTHONPATH` entries exist in the staged tree; no module-name shadowing. Every env name in the
      three `env` lines is read by its script; `SOURCE_COMMIT`, `JAX_DEFAULT_MATMUL_PRECISION` and
      `SLURM_JOB_ID` are read by the drivers. `set +e`/`rc=$?`/`set -e`, `test -s`, `tee` under
      `pipefail`, `sha256sum -c MANIFEST.sha256` in `$TASK_ROOT`, `find | xargs sha256sum` are correct.
    - *Sealed draw:* `bf.sample_params` (`burgers2d_film.py:173-182`) and `params_draw`
      (`engines.py:20-25`, `make_configs.py:27-32`) are the same column-wise draw, so the listed
      distances are computed on the right cohorts; every other cohort a checkpoint touched is an
      index set into `params_draw(0,128)` (`seeds_run.py:121-129`, `q_eqcert.py:103-109`) or a
      snapshot pick, hence covered; `17092026` occurs nowhere on the branch outside this lane; the
      audit recomputes the values (≤1 ulp) and the disjointness in-job (`audit_seeds.py:294-325`); the
      seed jobs stage no sealed config (`stage.py:210-213`), which their `PROVENANCE.json` will show.
    - *Criteria computability:* C1, C3, C4, TR and F1–F3 are computed from fields the driver writes
      (`converged`, `budget_exits`, `stationary`, `worst_bank_projection_percent`,
      `worst_best_found_percent`; `seeds_run.py:499-516, 402-416`).

---

## Verdict

**Fix first, then submit.** The recipe reproduction itself is sound — the seed job is the
incumbent's own three scripts with `SEED0` the only numerical variable, the staged layout resolves,
and the sealed draw is genuinely disjoint and auditable. But the pre-registration cannot deliver its
own verdict as written: the incumbent fidelity gate is set at a tolerance qtd02's own record shows
is not achievable across jobs (item 1, with item 2 as its echo), and C2/F2 read a max-statistic on a
visibly harder cohort as a transfer test (item 3). Both are one-line config changes plus a §9
amendment, and both must be made *before* the first job or they become post hoc. Items 4–9 are
operational: the L40S fallback, the un-stageable rerun, the cross-job cost column, the checksum
ordering, and the unrecorded smokes each cost either a wasted allocation or a reviewer's trust.
None requires a new draw, a new script or a change to the science; an hour of edits and the smoke
run, committed, and the lane is ready to stage `s1`, `s2`, `s3`.
