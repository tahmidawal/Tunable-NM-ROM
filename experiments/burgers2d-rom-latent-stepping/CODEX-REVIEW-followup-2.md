# Verification of the follow-up RESULTS (Codex, second pass, 2026-08-17)

After the cells finished, `codex exec` was run **read-only** a second time with a different
brief: not "review the code" but "re-derive every number in the prose, the markdown tables
and the figures from the raw JSONs, and flag any claim the cited measurement does not
support".  It was allowed to run read-only Python to recompute values.  The verbatim report
is appended below the disposition table; this file is archived identically on both branches
because the review covers both.

Its scale, in its own accounting: 1 311 populated table cells re-derived across 27 tables,
321 aggregate re-computations from per-sample arrays, 143 timing medians recomputed from the
seven-element repetition arrays, all 8 PNGs traced series-by-series back to JSON fields, and
a completion audit of all 36 cluster logs.  Everything it checked in the *generated* tables
matched; everything it found wrong was in hand-written README prose or in a figure title.

## Disposition — WRONG (all fixed)

| # | finding | fix |
|---|---|---|
| 1 | Poisson README M-ladder mixed nearest-init accuracy with the mean-init timing cell's errors (M=128 and M=256 columns), quoted 256 retained modes where 257 were retained, and printed one retained-mode count for all three arms although the on-grid (discrete eigenvalue) and meshfree (continuum eigenvalue) arms retain different counts at M=32. | Table rewritten from the `pM_M*` nearest-init rows (M=128: 7.17e-3 / 7.28e-3 / 7.42e-3; M=256: 7.13e-3 / 7.13e-3 / 7.25e-3), M′ listed **per arm**, and the accuracy/cost protocols stated under the table.  `fu_summarize.py` now emits the per-arm M′ too. |
| 2 | "POD at k=64 (2.54e-2)" was the *projection floor*, not the same-weak-objective result (4.892e-2, a square system). | Rewritten: the largest well-posed same-objective point is k=48 (3.54e-2, 4.6x above the coordinate ROM at k=8); the 2.54e-2 is named as the projection floor. |
| 3 | The `k^-1.2` extrapolation was unsupported and the series was not named. | Replaced with the measured fit of the **projection floor over k = 8…64**, ≈ k^−0.94 → k ≈ 240, explicitly labelled an extrapolation ~4x beyond the measured range. |
| 4 | "POD weak vs projection agree to <1%" — actually 1.93% at k=48. | Changed to "within 2% at every k ≤ 48 (worst case 1.93%, at k=48)". |
| 5 | "meshfree EQ within 17% of the full grid at every k ≤ 24" — actually 34% at k=16, 24% at k=24. | Restated with the real range and the reading that a bigger latent space needs more quadrature. |
| 6 | "FD-LSPG gets worse with k" is a tail-driven mean: at k=32 its median (2.71e-2) is better than its k=8 median (3.85e-2), and the causal amplification mechanism was not measured here. | Rewritten: the mean degrades but is tail-driven; what is robust is the 5–7x mean gap at every k and the budget-termination counts (12/16 vs 5/16 at k=32).  The mechanism is attributed to round 1, where it was measured. |
| 7 | "all three inits give the same number" fails at k=4 (1.72e-2 mean-init vs 1.57e-2 nearest, 9.9%) and at k=2. | Rewritten with the actual per-k behaviour, plus an explicit statement that every accuracy number in the section is nearest-init and every cost number is the timing cells' mean-init / source-0 protocol. |
| 8 | Poisson NNLS diagnostic: "five orders" (actually 3.9) and "below ~1e-4 at m=512" (actually 2.4e-4 / 2.7e-4). | Corrected, and the diagnostic is now described as correlated on this one ladder rather than validated. |
| 9 | Poisson M-ladder conclusions: "monotone cost" (full-grid times are 20.1, 45.3, 35.6, 38.1, 47.4 ms) and "within 5% once M ≥ 32" (41%/51% at M=32, 13%/17% at M=64; within a few % only from M=128). | Both corrected, with the non-monotone full-grid cost explained (at M=16 the solve is short enough that iteration-count differences dominate) and the EQ arm's genuine O(m) monotonicity kept. |
| 10 | Offline mode-table build times were quoted as 0.9 ms / 1.0 s instead of 2.5 ms / 0.51 s; "8.86e-3 from N=64 up" instead of 8.94e-3 at N=64. | Corrected from the JSON. |
| 11 | "nothing beyond k=8 pays for itself" contradicted the displayed k=16 numbers. | Removed and replaced by the actual trade, with the two protocols named. |
| 12 | Complexity-ladder overstatements: "sits on its inferred-latent floor at every point" (ratio 1.2–1.6 at k=32), "FD-LSPG 3–6x worse everywhere" (1.5x–35x), and "15x train/test gap" conflating the NB=1→NB=2 change in the held-out floor (15.4x) with the within-arm test/train ratio (7.9x).  The figure subtitle repeated the floor claim. | All three corrected with the measured numbers; the figure subtitle now says "stays within 1.0–1.6x of its own inferred-latent floor"; the causal reading is explicitly labelled a hypothesis with sample count, capacity and budget varied together. |
| 13 | Burgers: the k-ladder table displayed POD `full:fd` while the prose claimed "same residual and same test modes", and the crossover sentence was internally contradictory — POD-64 (1.4006e-2) is 15% *better* than coordinate K=8 (1.6543e-2), not worse. | The table now shows **both** POD rows (same weak objective *and* the frozen round's `full:fd` control) plus k=64.  The crossover is restated quantitatively: interpolating inside the measured k=32→64 range, POD needs k ≈ 60 to match coordinate K=8 (7.4x fewer latent dimensions); POD-64 beats coordinate K=8 and K=12 and is beaten by K=16. |
| 14 | "tracks its floor, never the solver" fails at K=24 (1.61) and K=32 (2.53); the POD/floor range is 1.006–1.15, not 1.05–1.13. | Both corrected. |
| 15 | Burgers multi-seed generated table: heading claimed the seed varies latent init (it does not — the latents are deterministically POD-initialised), and three POD variants measured only in the seed-0 cell were shown with a fabricated ±0.0e+00. | Heading rewritten; `fu_summarize.py` now aggregates a POD row only when it exists for **every** seed and otherwise prints "not run for every seed" with the cells it has. |
| 16 | The Burgers EQ-cold-start end-to-end speedups were cost **compositions**: the rollout was timed from the full-grid latent and the EQ fit time was added, while the EQ latent differs by 2–5% and neither the rollout cost nor its accuracy from that latent had been measured. | **Cell rerun** (`bt_n`, job 2491608) with a second rollout timed *and graded* from the hyper-reduced latent; the table now reports measured end-to-end times and the resulting trajectory error. |
| 17 | "the whole online path can be made mesh-free" is false for the primary `eq256:weak64` arm, which uses grid nodes and the FOM's upwind stencil. | Retitled "mesh-size independent" throughout, with an explicit note that only the `weakc`/`eqoff` arms are meshfree. |
| 18 | "cheapest arm in the study (2.33 ms/step)" ignored meshfree m=64/128 (1.50/1.82) and every POD arm; "`full:weak64` grows 67x exactly like the FOM" (the FOM grows 10.8x). | Both corrected; the 67x-vs-10.8x contrast is now the point being made rather than an error. |
| 19 | Both READMEs claimed medians appear in `FOLLOWUP_TABLES.md`; they did not. | Median columns/lines **added** to both generated files (Poisson k and M ladders, Burgers k ladder), and the caveats now match. |

## Disposition — MISLEADING (all addressed)

- **Protocol blending.** Accuracy is nearest-init, 16 sources; timing is mean-init, source 0.
  Every table that mixes them now says so in its header, and the Poisson `ms/iteration`
  column now states that it divides the timed source's solve by *its* iteration count (33 at
  N=32) rather than by the 16-source mean (27.8).
- **Budget terminations.** Counts are now printed per cell in both generated files, with the
  statement that they are included in the means and nothing is dropped — Poisson 0–2 of 16
  at k ≤ 16 and 5–12 of 16 at k=32; Burgers 233 of 800 time steps at K=32.  "Zero blow-ups"
  is explicitly distinguished from "converged at every step".
- **Causal language.** "Knee at the intrinsic dimension", "under-trained / poorly
  conditioned past K=16", and the sparse-coverage explanation of the NB=2/3 collapse are now
  presented as readings consistent with one ladder at one budget, with the missing controls
  named.
- **Fit-residual diagnostic.** Described as correlated on this single m ladder at one k and
  one M, not as a validated predictor.
- **Figure titles.** "online cost is independent of the mesh" → "the latent solve does not
  see the mesh" (Poisson) and "the hyper-reduced latent rollout does not see the mesh"
  (Burgers), since both figures themselves display the O(n) components.
- **Square-system exclusion.** The Burgers k figure dropped the k=64 weak point by testing
  `value < 1.0`; it now drops it by the *condition* k ≥ M and the README states why.
- **POD curve range.** The Poisson k figure stops at k=32 because the coordinate ladder does;
  stated in the table header, with the k=48 and k=64 POD points in the table.
- **"Reproducible to 3.6% / 6%"** now qualified as a sample coefficient of variation over
  three training seeds on one fixed test set.

## Not changed, with reasons

- The **degenerate-eigenshell truncation** in `blat_common.test_modes` (first-pass SHOULD
  item) remains as the frozen round left it, for comparability; it is in the caveats.
- Codex's observation that Poisson's follow-up EQ numbers are 6–8% above the frozen
  round-2 weak-collocation values is **expected and disclosed**: the follow-up sets
  `EQ_FIXED_SNAPS=1` so that every ladder point is fitted on the same decoder snapshots.
  Codex confirmed the write-up discloses it.

## Verbatim Codex report
## WRONG

- [Poisson README M-ladder](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/README.md:341) mixes nearest-init accuracy with mean-init timing-cell errors. From the nearest-init rows in `pM_M*/Mlad_M*.json`, the last two columns should be:

  - M=128: full/grid-EQ/meshfree-EQ = `7.17e-3 / 7.28e-3 / 7.42e-3`, not `7.24e-3 / 7.33e-3 / 7.40e-3`.
  - M=256: `7.13e-3 / 7.13e-3 / 7.25e-3`, not `7.06e-3 / 7.07e-3 / 7.18e-3`.
  - Retained modes at requested M=256 are 257, not 256.
  - At requested M=32, the full/grid rows retain 32 modes but the meshfree row retains 33; both the README and [generated table](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/followup/FOLLOWUP_TABLES.md:48) wrongly present one retained count for all three arms.

- Several Poisson k-ladder claims are numerically false in [README lines 279–301](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/README.md:279):

  - “POD at k=64 (2.54e-2)” is the projection floor, not the same-weak-objective POD result; that result is `4.892e-2`.
  - The stated `k^-1.2` fit is not supported. A log-log fit to the projection floor over k=8…64 gives about `k^-0.94` and k≈238; fitting the same weak objective through the valid k≤48 points gives about `k^-0.82` and k≈348. The series and fit window must be named.
  - POD weak versus projection differs by 1.93% at k=48, not `<1%`.
  - Meshfree EQ is not within 17% of full grid at every k≤24: it is 34.0% worse at k=16 and 23.8% worse at k=24.
  - “FD-LSPG gets worse with k” is driven by a heavy-tailed mean. At k=32 its mean is `1.890e-1`, median `2.713e-2`, and maximum `1.484`; the median is better than the k=8 median `3.847e-2`. The claimed causal amplification mechanism was not measured.
  - Mean and nearest initialization do not land on the same result at k=4: `1.720e-2` versus `1.565e-2`, a 9.9% difference. K=2 is also strongly initialization-dependent.

- The Poisson NNLS diagnostic paragraph is wrong. [README lines 334–339](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/README.md:334) says “five orders” and “below ~1e-4 at m=512.” The grid residual falls `1.4e-1 → 1.6e-5`, 3.94 orders; meshfree falls `1.2e-1 → 2.4e-5`, 3.70 orders. At m=512 the residuals are `2.4e-4` and `2.7e-4`, not below `1e-4`.

- The Poisson M-ladder conclusions are false. [README lines 351–355](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/README.md:351) says cost is monotone, but full-grid times are `20.1, 45.3, 35.6, 38.1, 47.4 ms` and meshfree-EQ times are `6.7, 14.8, 11.5, 16.6, 23.9 ms`. It also says m≈4M is within 5% once M≥32; at M=32 grid/meshfree EQ are 40.8%/50.7% worse than full, and at M=64 they are 13.1%/17.4% worse. Both arms are within 5% only from M=128 onward.

- The Poisson offline preprocessing times in [README lines 380–386](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/README.md:380) should be 2.54 ms at N=32 and 0.506 s at N=512, not 0.9 ms and 1.0 s. The [generated table’s values](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/followup/FOLLOWUP_TABLES.md:66) are correct. Likewise, “8.86e-3 from N=64 up” should say `8.94e-3 at N=64, then 8.86e-3 from N=128`.

- “Nothing beyond k=8 pays for itself” in the [Poisson cost discussion](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/README.md:409) contradicts the displayed data: nearest-init k=16 improves `7.65e-3 → 5.22e-3`, while the separate mean-init timing cell reports `20.2 → 15.9 ms`. At minimum, the claim must be removed and the differing initialization protocols disclosed.

- The Poisson complexity interpretation overstates the raw results. In [README lines 431–442](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/README.md:431):

  - The ROM does not sit on its inferred-latent floor at every point. At k=32, ROM/floor is 1.56 for NB=1, 1.36 for NB=2, and 1.21 for NB=3.
  - FD-LSPG is not “3–6x worse everywhere”; ratios range from about 1.5x to 35.4x.
  - At NB=2, k=8, the actual within-arm test/train ratio is `0.1092/0.01392 = 7.85x`, not a “15x train/test gap.” A factor of 15.4 is the change in held-out floor from NB=1 to NB=2.
  - The [figure subtitle](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/followup/fu_summarize.py:525) repeats the false “floor throughout” claim.

- The Burgers POD comparison is mislabeled and its crossover sentence is internally wrong. [README lines 263–278](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-burgers2d-rom-latent-stepping/experiments/burgers2d-rom-latent-stepping/README.md:263) compares coordinate `full:weak64` against the displayed POD `full:fd` series, so “same residual and same test modes” is false. Measured POD-64 is `1.4006e-2`, which is 15% better—not worse—than coordinate K=8 at `1.6543e-2`; coordinate K=16 at `9.6249e-3` does beat POD-64. Also, “tracks its floor, never the solver” fails at K=24 and 32, where ROM/floor is 1.61 and 2.53. POD/full-FD divided by its floor ranges approximately 1.006–1.148, not 1.05–1.13 with a 13% maximum.

- The Burgers multi-seed generated table is malformed. Its [heading](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-burgers2d-rom-latent-stepping/experiments/burgers2d-rom-latent-stepping/followup/FOLLOWUP_TABLES.md:30) wrongly says the seed changes latent initialization; the latents are deterministically POD-initialized. In [rows 41–44](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-burgers2d-rom-latent-stepping/experiments/burgers2d-rom-latent-stepping/followup/FOLLOWUP_TABLES.md:41), Galerkin, `eq512:weak64`, and `full:weak64` exist only in the seed-0 JSON, yet the rows omit the seed-1/2 cells and report zero standard deviation. Only `lspg:full:fd` was measured for all three seeds.

- The Burgers EQ-cold-start “end-to-end” speedups are not end-to-end measurements. [The timing code](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-burgers2d-rom-latent-stepping/experiments/burgers2d-rom-latent-stepping/followup/fu_timing.py:248) runs the rollout from the full-grid-fit latent `z_j`, then adds the separately measured EQ-fit time in the speedup formula. The EQ latent differs by 2–5%, and neither rollout cost nor rollout accuracy from that latent was measured. Therefore the [0.67x–7.55x columns](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-burgers2d-rom-latent-stepping/experiments/burgers2d-rom-latent-stepping/README.md:347) are only hypothetical cost compositions. Fix by rerunning/timing the rollout from `z_e`, or label them explicitly as cost-only estimates with downstream accuracy unmeasured.

- “The whole online path can be made mesh-free” is false for the primary Burgers `eq256:weak64` arm. It uses grid-node EQ plus FOM upwind stencils and its hyper-reduced cold start is explicitly limited to grid nodes. “Mesh-size-independent/hyper-reduced” is accurate; “mesh-free” is not.

- Two further Burgers numerical claims are wrong:

  - [“Cheapest arm in the study (2.33 ms/step)”](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-burgers2d-rom-latent-stepping/experiments/burgers2d-rom-latent-stepping/README.md:329) ignores meshfree m=64 and 128 at 1.50 and 1.82 ms/step, and the still-faster POD arms. It can be called the fastest coordinate m=256 arm.
  - [“Full:weak64 grows 67x exactly like the FOM”](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-burgers2d-rom-latent-stepping/experiments/burgers2d-rom-latent-stepping/README.md:359) is false: that arm grows 67.5x, while the FOM grows 10.8x.

- Both README caveats falsely say medians appear in `FOLLOWUP_TABLES.md`: [Poisson](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/README.md:455) and [Burgers](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-burgers2d-rom-latent-stepping/experiments/burgers2d-rom-latent-stepping/README.md:407). Neither generated file contains median columns.

## MISLEADING

- Poisson accuracy and cost are combined under different initialization/sample protocols. The m/M accuracy rows use nearest initialization, while `pt_m/timing_m.json` records `init: "mean"` and times test source 0. In `pt_n`/`pt_k`, “ROM iters” is the mean over 16 sources, but `ms/iteration` is timed source 0 divided by source 0’s iteration count. For example, N=32 displays 27.8 mean iterations, while `19.859/0.6018 = 33` is the timed source’s count. Add explicit “accuracy: nearest-init mean over 16; timing: mean-init source 0” labels.

- Budget terminations are included in means but are not disclosed. For Poisson k=32 nearest-init, budget termination occurs in 5/16 full-grid, 7/16 grid-EQ, 10/16 meshfree-EQ and 12/16 FD solves. In the complexity cell NB=3, k=32, the counts are 15/16, 14/16 and 14/16. Burgers has no blow-ups, but at K=32, `eq256:weak64` hits its per-step budget on 233/800 steps; `full:weak64` hits it on 171/800. No samples were dropped, but “zero blow-ups” is not equivalent to solver convergence. Add termination counts beside the means.

- The “knee equals intrinsic dimension” statements are descriptive correlations from one ladder, not causal measurements. Likewise, Burgers’ “poorly conditioned/under-trained” explanation has no condition-number or longer-training control, and Poisson’s claim that sparse parameter-space coverage causes the NB=2/3 result is not separated from fixed model capacity and fixed optimization budget. These should be presented as hypotheses consistent with the data.

- The fit-residual claims call the quantity a predictive a-priori diagnostic, but the same ladder used to formulate the threshold is the only validation. State that it is correlated on this one m-ladder; predictive validity on new decoders, sources, or M values was not tested.

- The cost-figure titles overgeneralize. Poisson’s [“online cost is independent of the mesh”](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/followup/fu_summarize.py:446) and Burgers’ [same title](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-burgers2d-rom-latent-stepping/experiments/burgers2d-rom-latent-stepping/followup/fu_summarize.py:410) sit above plots that themselves show O(n) input/decode/cold-start components. Rename them to “latent-solve/rollout cost is mesh-size independent.”

- Burgers’ k figure drops the k=64 same-weak-objective point (`1.658`) by testing whether its value is `<1.0`, rather than using an explicit square-system flag, and the README caption does not explain the omission. See [the filter](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-burgers2d-rom-latent-stepping/experiments/burgers2d-rom-latent-stepping/followup/fu_summarize.py:326). State that k=M=64 is a square weak Petrov–Galerkin system and exclude it via recorded system metadata.

- Poisson’s main k figure silently stops the POD curve at k=32 even though the raw POD ladder has a valid k=48 point; only k=64 has the stated square-system exclusion. Add that the main figure is restricted to same-k coordinate/POD points, or plot k=48.

- Burgers’ timing-based “sweet spot” combines 16-trajectory accuracy with timing from one trajectory, and the N=128/256 speedups apply only to K=8, not the whole claimed K=6–8 range. State both qualifications. Similarly, “Jacobian cost is linear in K” is theoretical; the measured change is 4.7x over a 16x K increase.

- The three-seed percentages—3.6% for Poisson and 6% for Burgers—are correctly computed sample coefficients of variation, but “reproducible” should be qualified as “observed across three training seeds on one fixed test set.”

## VERIFIED

- I checked all 27 numerical-result tables: 1,311 populated cells excluding the row-label column—219 in the Poisson README, 531 in its generated tables, 203 in the Burgers README, and 358 in its generated tables. Apart from the seven Poisson README M-ladder cells identified above, populated numeric cells match the selected JSON fields to their printed precision. Both generated Markdown files also exactly match an in-memory, read-only regeneration from their current scripts and JSONs.

- Row selection was verified, not merely values:

  - Poisson k/m/M accuracy uses `init:"nearest"`; full-grid rows use `scheme:"full"`; grid and meshfree EQ use `scheme:"nnls"` and `"nnlsoff"`; m comes from the row and retained M from `n_modes_retained`.
  - Burgers columns select the literal variant keys such as `lspg:full:weak64`, `lspg:eq256:weak64`, and `lspg:full:fd`.
  - Sample standard deviations in both multi-seed tables use `ddof=1`.
  - All requested m and M ladder points present in the JSONs appear in the generated tables.

- For Poisson, I independently recomputed mean, median, and maximum from the per-sample arrays for 107 result rows—321 aggregate checks—with no discrepancy. For Burgers, the JSONs do not retain per-trajectory vectors, so medians cannot be independently reconstructed; because every rollout completed, I recomputed 198 displayed means as the mean of their 51-entry `per_time_mean` arrays, with no discrepancy.

- I recomputed 143 timing medians from the seven-element repetition arrays across both studies; every stored timing median was correct. The displayed solve-only speedups are also the stated FOM median divided by the corresponding ROM median.

- All eight PNGs were inspected and their series traced through `fu_summarize.py` to raw fields. Apart from the omissions/titles noted above, axes, series, and legends correspond to the JSON quantities: nearest-init Poisson accuracy ladders, Burgers 16-trajectory means, timing medians, and the intended grid versus meshfree variants.

- Frozen-study consistency is good where configurations truly reproduce:

  - Poisson K=8 follow-up versus frozen hard-BC: full weak differs by only `2.47e-11` absolute, FD by `2.65e-14`, oracle by about `7e-16`, and training summaries by about `1e-15`.
  - The follow-up K=8 EQ values are 8.2% (grid) and 6.3% (meshfree) above the older weak-collocation values because the follow-up deliberately changed to fixed EQ snapshots; the write-up discloses that protocol change.
  - Burgers K=4/8/16 and seed 0 are the frozen JSON files reused verbatim, so their differences are exactly zero.

- Completion/failure audit: all 23 Poisson `.out` logs and all 13 Burgers `.out` logs, including both Burgers N-timing attempts, contain `jax_backend=gpu` and `ALL-DONE`. All 36 matching `.err` files are empty. I found no traceback, OOM, refusal, kill, or aborted mesh. The current timing JSONs are marked complete, and every tested FOM residual is below its asserted threshold.

- Burgers blow-ups total zero across all 186 coordinate/POD report variants inspected; all reported trajectory means therefore include all 16 trajectories. Poisson budget-terminated solves and Burgers budget-terminated time steps were included rather than silently dropped.

- The following headline arithmetic is correct: Poisson K=8 POD/coordinate = 23.1x; Poisson three-seed CV = 3.6%; Poisson N-ladder solve range = 19.71–19.98 ms over 289x DOF; Burgers K=8 POD/coordinate = 12.7x; Burgers three-seed CV = 6.2% (reported as 6%); Burgers m=256 full/EQ cost ratio = 11.2x; and Burgers rollout-only N=256 speedup = 8.09x.

- No files were modified.