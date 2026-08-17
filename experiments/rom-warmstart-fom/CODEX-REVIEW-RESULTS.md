# Codex adversarial results audit — 2026-08-17 (post-run)

Second `codex exec` pass, read-only, auditing every table, figure and prose claim against the
raw JSONs. Scope it reports: **2,052 table body cells** (382 README, 1,670 SUMMARY_TABLES),
**304 fact substitutions**, **179 data-facing prose sentences**, **1,194 row-level arithmetic
identities**, plus a review of `wsf_verify.py`'s 130 checks.

## What it found clean

- `README.md` is an exact rendering of `README.tmpl.md` plus the facts; `SUMMARY_TABLES.md`
  and `runs/hybrid_points.json` are exact outputs of `wsf_summarize.py`; `hybrid_points.json`
  is an exact flattening of the 174 raw rows (87 panel, 87 consolidated).
- **No displayed table cell has a formatting or arithmetic mismatch** against the raw row its
  generator selected.
- The consolidated/panel wall-clock contract passes: no cross-`N` wall-clock statement in the
  README, the tables or the figures uses panel timing.
- Run integrity 11/11 on `complete`, `jax_backend=gpu`, GPU model, checksum verification, and
  row-vs-report provenance agreement. Every solver-health gate passes with margin.
- The correction-to-the-record **arithmetic** reproduces exactly, including the archived
  timing anchors and the engineering-vs-accuracy-matched claim.
- Both headline claims confirmed: no Poisson crossover (max 0.93306669x), and the Burgers
  hybrid never beats the previous-step baseline on total cost.

## MUST FIX — disposition

| # | Finding | Applied |
|---|---|---|
| 1 | **Invalid commit provenance.** The `commit` in all 11 reports (`72b44d…`) is not an object in this repo: `git -C <staged dir>` walked up into an unrelated ancestor repository on the cluster. | **Yes.** The authoritative commit is `63de48bd…`, stamped into every sbatch log by `make_cell.sh`; the job table and a new README note now report that and name the defect. `provenance()` now takes `WSF_COMMIT` from the environment, records `commit_source`, and refuses git discovery that cannot be shown to describe this file. `make_cell.sh` exports it. |
| 2 | **Timing repetitions discarded**, so the post-selection bias of "best rom_tau" cannot be reconstructed. | **Yes, by withdrawal + code fix.** The exact best-rung identity is now explicitly *unresolved*; the full ladder (P1) is the primary result. Codex's own proxy analysis (best/second gap ≤ baseline-variation proxy in **6 of 15** groups) is quoted in the caveat. `wsf_poisson.py`/`wsf_burgers.py` now persist every repetition (`t_fom_all_s`, `t_fom_baseline_all_s`, `t_rom_all_s`). These runs were **not** repeated, so the limitation stands for this data. |
| 3 | **Reference solve described wrong** (`1e-13`, achieved `1.0e-13`/`5.7e-13`; actually `REF_TAU=1e-12`, achieved `8.788e-13`/`9.832e-13`). Stale prose from the discarded first round. | **Yes.** The requested tolerance and every achieved residual are now generated facts. |
| 4a | Iteration multiplier called "more aggressive" and a lower bound — false; it is *larger* at 3 of 4 Burgers meshes. | **Yes.** Directionality removed; described as an independent work-count proxy, not a bound. |
| 4b | "Instrumented CG ~15% cheaper per iteration" — false; it is **1.3–7.3% slower**. | **Yes.** Reversed and quantified from the data. The claim came from the contaminated pre-burn-in measurement. |
| 4c | A residual "like-for-like" label on `tau=1e-10`. | **Yes.** Now "tightest available engineering rung", with the looseness ratios. |
| 5 | Burgers cross-check misstated what was averaged: **wall clock times trajectory 0 only**, counts average 4. Sister-cell claims not locally auditable. | **Yes.** The two populations are separated explicitly, and every sister-cell property is marked relayed-not-verified rather than "confirmed". |
| 6 | "Every number comes from the JSONs" false — 9 `xagent_*` keys, and this cell's own `14.86` was labelled relayed. | **Yes.** Archive and this-cell values are now derived locally and the spread computed; only the three sister timings remain hard-coded and are flagged as such at every use. |

## SHOULD FIX — applied

- Inconsistent rung selection between P3c (first rung) and `wsf_facts` (last rung), which
  disagreed by ~1.5%: both now take the **mean over the repeated rungs**, documented in code.
- Extrapolated break-even relabelled a **conditional toy calculation**, since it freezes the
  `N=512` overhead and saving although decode grows and the saving falls.
- "Worth 582 iterations" now states both sample populations (4 diagnostic cases vs 16).
- Direct-solver ratio: both operands now come from one aggregated row.
- "Agree exactly" → "all but one", with the disagreement's magnitude and a hedged cause.
- Poisson multiplier spread corrected from 2.75% (across rungs, mixing timing noise) to
  **0.93%** (per-mesh means); the raw rung spread is kept and labelled.
- The 0.13% anchor described as a **wall-time** agreement, not trajectory reproduction.
- "Fixed tolerance has exactly the same effect as a fixed iteration count" softened to name
  the two different mechanisms.
- Speculative `dt` and preconditioner caveats changed to "may"/"was not measured".
- Figures: one selector now drives both crossover panels (they previously chose different
  rungs at two meshes); the dashed baseline is the mean of repeated baselines rather than
  whichever row came first; post-selection is labelled **on** the figure; the right-panel
  title already stated the answer rather than asking "where does it win".

## Not applied, with reasons

- **Re-running to recover the timing arrays.** Codex offered rerun *or* withdrawal; withdrawal
  is taken. The headline is unaffected — breaking even needs a ~6.7% improvement on the best
  measured 0.933x, far outside the retained spread proxy — and a rerun would perturb every
  number in a finished, audited artifact to settle a secondary claim.
- **Alternating the warm/zero timing order** (NOTE). Real, but it needs a rerun; the pairing
  already removes the large drift, and the residual concern is recorded.

## Verdict, verbatim

> The core benchmark results are trustworthy: all runs completed on GPUs, solver-health gates
> pass, raw arithmetic is clean, the panel/consolidated wall-clock contract is now respected,
> Poisson has no measured crossover, and the Burgers hybrid never beats the previous-step
> baseline on total cost. The report is not yet trustworthy *as stated*, however, because its
> commit provenance is invalid, timing repetitions needed to justify post-selection were not
> retained, the reference-solve description is wrong, and several correction-to-record
> prescriptions and cross-cell claims are false or unauditable. The headline negative
> conclusions survive these defects; the provenance, selection language, and correction
> narrative require repair before publication.

Every MUST FIX has been repaired or explicitly withdrawn. The full unedited report follows.

---

# MUST FIX list

Audit snapshot: commit `db0ec3413f00eadf141005310c9b287fd1ad1039`, clean worktree. The branch changed externally during the audit; all findings below refer to that commit. I made no filesystem changes.

1. **MUST FIX — invalid run commit provenance.**  
   Location: all 11 raw reports, the generated job table at [README.md:762](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-rom-warmstart-fom/experiments/rom-warmstart-fom/README.md:762), and the source summaries at [SUMMARY_TABLES.md:285](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-rom-warmstart-fom/experiments/rom-warmstart-fom/SUMMARY_TABLES.md:285).  
   Stated: commit `72b44d40c86c…`.  
   Re-derived: every sbatch log identifies the staged cell as commit `63de48bdd5dd7c4a89494a05d0059bf48e27fe5e`; `72b44d…` is the program’s internally discovered commit and is not even an object in this repository. Its enormous unrelated “dirty” listing shows that Git discovery walked into the wrong ancestor repository.  
   Minimal correction: identify the run as wrapper commit `63de48d…`, retain `72b44d…` only as an explicitly invalid legacy field, and make future jobs pass the captured wrapper commit into the report. The recorded hashes of `wsf_poisson.py`, `wsf_burgers.py`, and `wsf_util.py` do match the present solver files, so the numerical code remains identifiable.

2. **MUST FIX — timing repetitions needed for the post-selection audit were discarded.**  
   Location: [wsf_util.py:49](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-rom-warmstart-fom/experiments/rom-warmstart-fom/wsf_util.py:49), [wsf_poisson.py:446](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-rom-warmstart-fom/experiments/rom-warmstart-fom/wsf_poisson.py:446), [wsf_burgers.py:528](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-rom-warmstart-fom/experiments/rom-warmstart-fom/wsf_burgers.py:528).  
   Stated: “best hybrid” is selected from median-of-seven timings.  
   Re-derived: `time_fn` produces each seven-sample array, but no current report contains a timing `t_all`/`*_all` field; only Poisson iteration arrays survive. The exact timing spread and best-of-noise bias therefore cannot be reconstructed.  
   Minimal correction: persist every timing repetition. Existing runs must be rerun to recover them, or the exact best-rung claims must be withdrawn and only the full ladder reported.

3. **MUST FIX — the stated Poisson reference solve is wrong.**  
   Location: [README.md:90](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-rom-warmstart-fom/experiments/rom-warmstart-fom/README.md:90) and the stale comment at [wsf_poisson.py:86](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-rom-warmstart-fom/experiments/rom-warmstart-fom/wsf_poisson.py:86).  
   Stated: `err_final` uses a reference computed at `1e-13`; achieved residuals are `1.0e-13` at `N=128` and `5.7e-13` at `N=256`.  
   Re-derived: `REF_TAU=1e-12`; consolidated maxima are `8.788e-13` and `9.832e-13`, respectively.  
   Minimal correction: say “reference CG requested at `1e-12`,” and substitute the recorded achieved residuals.

4. **MUST FIX — several prescriptions in “CORRECTION TO THE RECORD” are quantitatively false.**  
   Locations and corrections:

   - [README.md:545](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-rom-warmstart-fom/experiments/rom-warmstart-fom/README.md:545): the iteration multiplier is called “more aggressive” and a lower bound. Re-derived time/iteration multipliers are `0.267/0.262`, `0.230/0.281`, `0.209/0.297`, `0.230/0.301`. At three of four meshes the iteration correction is *larger*, hence not a lower bound on corrected speedup. Remove the directionality; describe it only as a hardware-independent work proxy.
   - [README.md:637](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-rom-warmstart-fom/experiments/rom-warmstart-fom/README.md:637): instrumented CG is said to be about 15% cheaper per iteration. Same-tolerance counting-CG/library-CG wall-time ratios are `1.013–1.073`: the instrumented solver is 1.3–7.3% slower, not 15% cheaper. Delete or reverse and quantify the statement.
   - [README.md:683](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-rom-warmstart-fom/experiments/rom-warmstart-fom/README.md:683): `tau=1e-10` is called “like-for-like” with the achieved archived accuracy. It is an engineering rung, not accuracy-matched: Poisson is 25.3–1315.6× looser and Burgers 94.3–107.2× looser. Use “tightest available engineering rung.”

5. **MUST FIX — the new Burgers cross-check text misstates what was averaged.**  
   Location: [README.md:476](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-rom-warmstart-fom/experiments/rom-warmstart-fom/README.md:476).  
   Stated: this cell “averages over 4 held-out trajectories,” matching the sister cell’s mean.  
   Re-derived: iteration counts and errors average four trajectories, but every Burgers wall clock times trajectory 0 only; the source states this explicitly at [wsf_burgers.py:535](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-rom-warmstart-fom/experiments/rom-warmstart-fom/wsf_burgers.py:535). The claims about the sister cell’s 16 trajectories, 1.5-second burn-in, symmetry, and `[-6%,0)` verdict band are not locally auditable.  
   Minimal correction: distinguish trajectory-0 timing from four-trajectory count/error averages and cite a pinned sister-cell source before calling the timing structure “confirmed.”

6. **MUST FIX — the “every number comes from the JSONs” provenance claim is false.**  
   Location: [README.md:214](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-rom-warmstart-fom/experiments/rom-warmstart-fom/README.md:214) and [wsf_facts.py:415](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-rom-warmstart-fom/experiments/rom-warmstart-fom/wsf_facts.py:415).  
   Stated: every number is derived from JSON; four `xagent_p_*` facts are external.  
   Re-derived: there are now nine `xagent_p_*` keys. Three sister timings are hard-coded (`3.66`, `7.144`, `14.795`), and four N=128 quantities are also hard-coded even though the archive and this-cell values could be derived locally. [README.md:662](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-rom-warmstart-fom/experiments/rom-warmstart-fom/README.md:662) incorrectly says this cell’s `14.86` is relayed.  
   Minimal correction: derive archive/current-cell values and spread from their JSONs; label only the sister timings external at every use, including the headline at README lines 342–344.

# Scope and clean results

I checked:

- **2,052 table body cells**: 382 in `README.md` and 1,670 in `SUMMARY_TABLES.md`; 1,945 contain numbers.
- **304 distinct README fact substitutions/aggregates**.
- **179 data-facing prose claim sentences**, counted as sentences containing a numeral or a comparative/measurement assertion.
- **1,194 independent row-level arithmetic identities** for totals, speedups, means, iteration savings, over-convergence factors, and DOFs.
- The newly added independent verifier’s **130 checks**; it reports 130/130 passing.

The good news:

- `README.md` is an exact rendering of `README.tmpl.md` plus the 304 facts.
- `SUMMARY_TABLES.md` and `runs/hybrid_points.json` are exact in-memory outputs of `wsf_summarize.py`.
- `hybrid_points.json` is an exact flattening of 174 raw rows: 87 panel and 87 consolidated.
- **No displayed table cell has a formatting or arithmetic mismatch against the raw row selected by its generator.**
- The final snapshot’s earlier P3b/P3c panel-timing leak has been fixed: all P1–P3 wall-clock tables now use `wsp_cons`.

There is still an arbitrary repeated-rung selection in the correction data: P3c takes the first consolidated ROM rung, while `wsf_facts.py` overwrites the same fact until the last rung. For example, at `N=128` SUMMARY reports native time/factor/multiplier `12.72 / 1.17 / 0.856`, while README reports `12.91 / 1.15 / 0.869`. Both are raw consolidated observations, but neither selection is documented. **SHOULD FIX:** use a declared rung or aggregate all five and publish the spread.

# Headline claims

## Poisson crossover

**Supported as measured.** The only legitimate source is [wsp_cons.json](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-rom-warmstart-fom/experiments/rom-warmstart-fom/runs/wsp_cons/out/wsp_cons.json): job `2511369`, one A100, all meshes sequentially.

All 75 configurations have speedup below one. The maximum is:

- `0.93306669×` at `N=512`, `rom_tau=0.01`, `tau_FOM=1e-6`.

Thus “no crossover in the measured ladder” is correct and not understated.

The extrapolated break-even table is algebraically reproduced exactly:

| `tau_FOM` | fitted exponent | assumed overhead | assumed saving | reported break-even N |
|---|---:|---:|---:|---:|
| 1e-6 | 1.140 | 7.434 ms | 0.0596 | 1099 |
| 1e-8 | 1.137 | 7.434 ms | 0.0378 | 1424 |
| 1e-10 | 1.136 | 12.83 ms | 0.0221 | 3320 |

**SHOULD FIX — overstated extrapolation.** [wsf_summarize.py:166](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-rom-warmstart-fom/experiments/rom-warmstart-fom/wsf_summarize.py:166) freezes the `N=512` overhead and saving fraction indefinitely. Its docstring claims the ROM stage is flat in `N`, but decode grows from `0.118` to `4.555` ms, while saving declines with tighter solves. The arithmetic supports only a conditional toy extrapolation, not [README.md:334](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-rom-warmstart-fom/experiments/rom-warmstart-fom/README.md:334)’s assertion that a crossover “plausibly exists.”

## Burgers versus the previous-step start

**Supported.** From the single consolidated job:

- Hybrid total beats previous-step wall time: `0/12`.
- ROM start wins Newton count: `4/12`.
- ROM start wins BiCGStab count: `0/12`.
- FOM stage alone wins: `1/12`, the reported `N=256`, `tau=1e-6` case (`434.7` versus `449.9` ms).
- Extrapolation wins Newton count `12/12` and BiCGStab count `10/12`.

The negative total-cost headline is correct.

# Prose findings

Additional prose corrections:

- **SHOULD FIX — mixed populations.** [README.md:268](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-rom-warmstart-fom/experiments/rom-warmstart-fom/README.md:268) compares “worth 582 iterations” with 1,772 full-solve iterations. `582.25` is computed over four diagnostic cases; `1772.44` and the 2.21% saving average 16 cases. For the same four cases, the baseline is `1776.25` and saving is `3.434%`. State both sample populations.

- **SHOULD FIX — inconsistent direct-solver ratio.** [README.md:291](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-rom-warmstart-fom/experiments/rom-warmstart-fom/README.md:291) says `81.58 / 0.166 = 494×`; the ratio is `490.712`, or **491×**. `494×` comes from another rung’s `82.134` ms baseline. Use one pair consistently.

- **SHOULD FIX — consistency wording.** [README.md:201](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-rom-warmstart-fom/experiments/rom-warmstart-fom/README.md:201) says panel and consolidated runs agree “exactly,” although P5 reports one disagreement. README’s 261 comparisons and P5’s 300 are different field sets. The difference is exactly `1860.4375` versus `1860.5`; calling its cause GPU reduction-order nondeterminism is plausible but not established. Say “all but one, plausibly due to reduction order.”

- **SHOULD FIX — Poisson multiplier spread is mislabeled as variation with N.** [README.md:687](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-rom-warmstart-fom/experiments/rom-warmstart-fom/README.md:687) quotes `0.856–0.880`, 2.75%. That range is over all 25 repeated ROM-rung timing blocks. Per-N means are `0.8646, 0.8652, 0.8688, 0.8608, 0.8631`, only **0.93%** max/min spread; using the displayed `rom_tau=0` choice gives **0.76%**.

- **SHOULD FIX — 0.13% is a timing anchor, not rollout reproduction.** [README.md:605](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-rom-warmstart-fom/experiments/rom-warmstart-fom/README.md:605) says the chain reproduces the archived rollout to 0.13%. That is the wall-time agreement between the retimed archived function and archived JSON at `N=128`; it is not trajectory/state agreement.

- **SHOULD FIX — “fixed tolerance has exactly the same effect as fixed iteration count.”** [README.md:630](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-rom-warmstart-fom/experiments/rom-warmstart-fom/README.md:630). Both can over-converge, but adaptive tolerance and fixed work are not exactly the same mechanism.

- **NOTE — speculative caveats should be conditional.** The claims that ROM advantage “should grow with `dt`” and that preconditioning “therefore” shrinks warm-start savings are not measured and are not guaranteed; use “may.”

- **NOTE — pairing reduces but does not equalize drift.** Poisson always times warm then zero at [wsf_poisson.py:443](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-rom-warmstart-fom/experiments/rom-warmstart-fom/wsf_poisson.py:443). Back-to-back timing reduces drift exposure, but fixed ordering does not guarantee drift “hits both equally.” Alternating order would be stronger.

# Figures

All ten PNG/PDF files exist, and every copy in `Plots/` is byte-identical. The cross-N source contract now passes:

- Poisson cross-N figures: only `runs/wsp_cons/out/wsp_cons.json`, job `2511369`, A100.
- Burgers cross-N figure: only `runs/wsb_cons/out/wsb_cons.json`, job `2511371`, A100.
- Panel rows are used only for hardware-independent iteration/per-step figures.

What the code must show:

- `wsfom_poisson_total_vs_tau`: three `tau_FOM` panels, five `N` curves versus categorical ROM tolerance, log-y, milliseconds. Every hybrid curve remains above its zero-start baseline. The dashed baseline is taken from the first dictionary row—effectively `rom_tau=0.5`—rather than an average of repeated baseline measurements. **SHOULD FIX** that arbitrary choice.

- `wsfom_poisson_crossover`: left panel log2-x/log-y in ms. At `tau=1e-10`, mean FOM costs are approximately `3.212, 6.545, 13.447, 26.370, 81.927` ms; best totals are `4.861, 8.453, 15.489, 30.172, 88.671` ms; the direct solve is `0.0633–0.1663` ms. Right panel is log2-x/linear-y and never crosses one.

- `wsfom_poisson_iters`: panel-preferred, `tau=1e-10`; left is linear percentage saving (`−5.04%` to `3.68%`), right has log A-norm-error x-axis (`0.0483–0.849`) and linear saving y-axis.

- `wsfom_burgers_per_step`: panel `N=256`, `tau=1e-10`; Newton and BiCGStab axes are linear and ROM error is log. Totals must be Newton `120.25/103/128.75` and BiCGStab `6527.75/6219.25/9286.75` for previous/extrapolated/ROM.

- `wsfom_burgers_cost_vs_N`: consolidated `tau=1e-10`; left x/y log in milliseconds, right linear Newton counts. Hybrid totals are `349.3, 439.3, 698.2, 1125.6` ms versus previous-step `52.63, 94.69, 253.78, 496.82` ms.

Axis units and scales are correct.

Figure findings:

- **SHOULD FIX — post-selection is not locally labeled.** SUMMARY P2 labels it, and README has a global caveat, but the README table, figure legends, axes, and caption simply say “best hybrid.” The user’s “everywhere” criterion is not met.

- **SHOULD FIX — inconsistent best selector.** The crossover left panel minimizes total time; the right maximizes the separately computed speedup. They select different rungs at `N=128, tau=1e-8` and `N=256, tau=1e-10`. Use the same chosen row.

- **SHOULD FIX — caption says “where the hybrid starts to win.”** [README.md:724](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-rom-warmstart-fom/experiments/rom-warmstart-fom/README.md:724). It never wins; say “whether the hybrid wins.”

# Consolidated/panel contract

This area now gets a clean bill of health.

- No cross-N wall-clock statement in the final `README.md`, final `SUMMARY_TABLES.md`, or `wsf_figs.py` uses panel timing.
- Poisson P5 checks 300 quantities and finds one disagreement.
- README’s combined Poisson/Burgers check covers 261 quantities and finds the same sole disagreement.
- All other iteration/error quantities agree.

# Run integrity

For the 11 raw report JSONs:

- `complete=true`: **11/11**.
- Logs contain `jax_backend=gpu`: **11/11**.
- GPU model: **11/11 NVIDIA A100 80GB PCIe**.
- Row `gpu`, `jax_backend`, `commit`, and `slurm_job_id` exactly match each report’s top-level provenance.
- `source_sha256` maps are populated in all 11 reports and identical. Raw report rows do not individually contain the map; flattened rows do. README’s “every result row” wording should distinguish those schemas.
- All 11 `LOCAL.sha256` files equal their `REMOTE.sha256` counterparts, and every listed payload verifies.
- The solver source hashes match the recorded hashes. Post-processing files have changed since execution, as expected.

Solver health:

- Poisson maximum delivered residual/tolerance: warm `0.999894`, zero `0.999801`.
- All 120 reported-tolerance CG subchecks have zero flags.
- The 40 reference-only subchecks contain four nonzero flags—two `N=512` cases duplicated in panel and consolidated—but their residuals remain below the special `1e-11` reference acceptance threshold. This does not invalidate any delivered row.
- Maximum counting-CG residual/threshold ratio: `0.999433`.
- Maximum relative difference versus `jax.scipy.cg`: `4.99e-12`.
- LM `rom_tau=0` is bitwise identical; preprocessing discrepancy is at most `8.67e-19`.

Burgers:

- `bicgstab_breakdowns=0`, `newton_flags_nonzero=0`, `health_warning=0`.
- All arms finite.
- Maximum Newton residual/tolerance ratio: `0.998381`.
- Maximum operator-agreement error is 0.2557 of its stated threshold.
- Maximum linear residual `9.81e-11` versus guard `1e-7`.
- Maximum BiCGStab difference versus JAX: `8.28e-10`.
- Every stated check passes.

# Correction-to-the-record arithmetic

The numerical correction tables themselves check out.

- Burgers engineering factors: `3.75, 4.35, 4.79, 4.35`.
- Burgers corrected rollout/end-to-end speedups:  
  `0.201/0.193`, `0.391/0.361`, `0.961/0.929`, `1.914/1.832`.
- Poisson displayed corrected solve/end-to-end speedups:  
  `0.244/0.241`, `0.341/0.336`, `0.662/0.644`, `1.345/1.207`, `4.199/3.072`.
- Every displayed corrected value equals archived speedup × selected time multiplier, modulo formatting.
- Archived timing anchors reproduce exactly: Burgers differences `0.212%, 6.823%, 0.127%, 0.526%`; Poisson `36.547%, 8.489%, 1.885%, 4.911%, 3.066%`.
- The claim that these are engineering rather than accuracy-matched factors is correct, with the looseness ratios given above.

The sister-agent timing values cannot be independently verified from this checkout. They are the only part of the correction arithmetic lacking a local primary source.

# Selection effects

The post-selection caveat is present globally and in SUMMARY P2, but not everywhere it appears.

Because the true seven-sample arrays were discarded, I used the only retained proxy: variation among the five repeated paired zero-start baselines at each `(N,tau_FOM)`.

- Proxy range/mean: minimum `0.469%`, median `1.319%`, maximum `2.898%`.
- Best-versus-second-best total-time gap: minimum `0.060%`, median `2.819%`, maximum `9.155%`.
- In **6 of 15** `(N,tau_FOM)` groups, the best/second gap is no larger than the same-group baseline variation proxy.

Thus several exact “best ROM tolerance” choices are plausibly timing-noise selections. The headline no-crossover conclusion is nevertheless robust: the best measured speedup is only `0.9331`, requiring a 6.69% improvement to break even—well beyond the retained same-group proxy variation.

# Independent verifier

`wsf_verify.py` passes 130/130 checks, but it is not sufficient evidence for every prose claim. In particular, its direct-ratio test chooses the first `N=512` row (`82.134/0.166254=494`) and merely checks that “494x” occurs somewhere in README; it does not verify that the adjacent stated operand is `81.58`. That is why it passes the inconsistent sentence identified above. **NOTE:** make verifier checks semantic and location-specific, and include the correction-to-record claims if it continues to describe itself as checking every headline.

# Verdict

The core benchmark results are trustworthy: all runs completed on GPUs, solver-health gates pass, raw arithmetic is clean, the panel/consolidated wall-clock contract is now respected, Poisson has no measured crossover, and the Burgers hybrid never beats the previous-step baseline on total cost. The report is not yet trustworthy *as stated*, however, because its commit provenance is invalid, timing repetitions needed to justify post-selection were not retained, the reference-solve description is wrong, and several correction-to-record prescriptions and cross-cell claims are false or unauditable. The headline negative conclusions survive these defects; the provenance, selection language, and correction narrative require repair before publication.