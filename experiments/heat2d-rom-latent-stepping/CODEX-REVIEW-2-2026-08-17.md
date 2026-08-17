# Codex pass 2 — results audit of the Heat-2D write-up (2026-08-17)

A second read-only `codex exec` pass, run after the five cluster cells finished, whose only job
was to check **every number in `README.md` against the archived JSONs and logs** and to be
hostile to the prose. It recomputed the speedups, ratios and roundings itself. Prompt archived
at `smoke/codex-prompt2-2026-08-17.txt`.

## Triage — all FACTUAL ERRORS were real and are fixed

| finding | verdict | fix |
|---|---|---|
| POD `weak64` at k=64, N=64 quoted as `1.42e-1` (twice) — that is the **N=128** value; N=64 is `5.20e-2` | **real error, flattered the `M = k` collapse by 2.7x** | corrected in both places |
| six three-digit roundings wrong in the last digit (K=16 IC fit, K=16 `galerkin:full:fd`, K=16 `weak32`, K=16 `eq256`, N=128 `eq256`, POD-direct k=8) | real | all corrected |
| "`weakall` reproduces `fd` to all printed digits **of the iteration counts**" | **overstated** | corrected: trajectory errors match to every stored digit and reason histograms match exactly, but LSPG iteration counts differ by up to one accepted step in one of 16 cold starts (11.2500 vs 11.3125 at K=4); Galerkin matches exactly; `weakall` was never run at N=128 |
| "Every step terminates on the stall criterion" | **false for the Galerkin arms** (796–800 of 800 terminate on `tol`) | rewritten per solver |
| N-flatness ratios: step `1.09x` -> **1.08x**, rollout `1.008x` -> **1.010x** (three places), oracle `1.03x` -> **1.04x** | real; the rollout one flattered flatness | all corrected |
| end-to-end `287 ms` matched neither the directly timed pipeline (`293.9 ms`) nor the sum of components (`284.8 ms`) | real, flattered runtime by 2.4% | now `294 ms`, `0.168x`, labelled as the directly timed pipeline |
| timing protocol not uniformly "median of 7 after 2 warm-ups" (the two Python-loop paths use 3 after 1) | real | protocol paragraph rewritten |
| "3.2 ms/step" for the 119 ms scan rollout conflates two measurements (amortised 2.38 ms vs the per-step median 3.24 ms with a host sync) | real | both stated |
| oracle-gap multipliers **swapped**: K=8 is 1.62x (stated 1.5x), K=16 is 1.45x (stated 1.6x) | real | corrected, and K=4 (1.62x) added |
| "consistent in mean, median and max at every K" false against the best strong arm (K=16 `weak256` has the worse median) | real | stated |
| eq256 penalty sign not consistent (−0.9% at K=4, +5.0%, +6.3%, +4.8%); grid-vs-meshfree up to 2.9%; discrete-vs-continuum up to 0.33%; EQ diagnostic range is 7.4e-4–3.4e-3 | real | all four ranges corrected |
| "same-solver POD essentially attains its projection floor at every k" false at k=64 (8.2x above with `weak64`, 15% with `fd`) | real | restricted to k = 6/8/16/32 and the k=64 exception stated |
| the decay triple `5.7e-2 / 2.8e-1 / 7.7e-1` is not traceable from any archived output | fair | the exact sine-basis computation is now archived as `tools/heat_decay.py` and reproduces it |

## Triage — OVERREACH items applied

Weakened or requantified: the causal story for the weak-form win (three explanations left
entangled — truncation, the `alpha` metric, and a different solver path/stopping point: 6.30 vs
8.48 warm Jacobian evaluations and 795/5 vs 781/19 terminations — and **no** condition-number
data, so the earlier "far better conditioned" assertion was removed); `M ~ 16K` -> "the tested
minimum moves toward larger M as K grows, on broad unresolved plateaus"; `M >= 4K` -> "do not set
M = K, the oversampling factor was not measured, the M = 2K arms were fine"; "hyper-reduction is
essentially free" -> the 4.93x step saving and the −0.9%…+6.3% error range; "meshfree matches
grid" -> "within 3%, inconsistent sign, one pool seed"; "Galerkin and LSPG coincide" ->
"aggregates indistinguishable at the available precision, on different iteration counts";
"buys ~7x in latent dimension" -> "6.93x lower error at equal k=8; the matching POD rank was not
run and lies between 32 and 64"; the projected-IC ceiling corrected from "≈1x" to **0.75x**
(the 51-slice decode also scales with n); the published-ROM comparison relabelled "not a
head-to-head"; "a neural decoder cannot compete" -> "none of the 19 tested configurations beats
this FOM end to end"; "oracle floor" -> "budget-40 inferred-latent oracle **estimate**" (not a
proven minimum; the POD projection floor is exact).

Added to the caveats: timing uses test trajectory 0 only; one training run per K and one EQ-pool
seed, so training/pool variability is unmeasured; no confidence intervals; process start, model
load and host<->device transfer excluded for ROM *and* FOM; no condition numbers or residual
spectra; no out-of-distribution family; no horizon beyond 50 steps; no mesh finer than N=128;
and the N=64 vs N=128 accuracy comparison uses two separately trained decoders.

Not applied: Codex asked to soften "the 3.4x Stage-1 gap is explained by the decoder's PDE
inconsistency" — applied as "consistent with, not isolated". Everything else was applied.

## What Codex verified as correct

All identity-check entries in all four reports; every Stage-1 cell; the Stage-2 table apart from
the errors above; all 30 entries of the K=8 per-time table; **zero blow-ups across all 68
coordinate arms and all 80 POD arms** (`n_blowup = 0`, `n_completed = n_total = 16` in every
one); the K=8 `weak64` 795-stalled/5-budget split; every raw timing median and the POD-direct
speedup range 5.456x–37.657x; and full provenance (commit `90171caef6800d6decc71aab627b126c6b8db80d`,
`dirty=0`, `jax_backend=gpu`, job ids matching log filenames, and `ad_n64_k8` / `ad_n128_k8`
both on `host=pax007`, `NVIDIA A100 80GB PCIe` — which the N-flatness claim depends on).

---

## Full report as returned by Codex

## FACTUAL ERRORS

- The N=64 POD weak64 value at `k=64` is wrong. The README gives `1.42e-1`; all three N=64 reports give `5.1959e-2` at `pod_rom["k64:lspg:full:weak64"]["traj_rel_mean"]`. `1.42e-1` is the N=128 result (`1.4193e-1`). This makes the N=64 POD arm look 2.73× worse and materially flatters the claimed `M=k` collapse.

- Several three-significant-digit entries are incorrectly rounded:

  | Claim | Correct value and source | Direction |
  |---|---|---|
  | K=16 IC fit `2.32e-2` | `2.314858e-2` → `2.31e-2`, `ic_fit["rel_mean"]` | Worsens the ROM slightly. |
  | K=16 Galerkin FD `1.11e-2` | `1.104884e-2` → `1.10e-2`, `rom["galerkin:full:fd"]["traj_rel_mean"]` | Makes the strong comparator look worse, flattering the weak-arm margin. |
  | K=16 weak32 `2.07e-2` | `2.064893e-2` → `2.06e-2` | Worsens that arm. |
  | K=16 eq256 weak64 `1.14e-2` | `1.134804e-2` → `1.13e-2` | Exaggerates the hyper-reduction penalty. |
  | N=128 eq256 weak64 `1.83e-2` | `1.824899e-2` → `1.82e-2` | Worsens N=128 accuracy. |
  | POD-direct k=8 `1.31e-1` | `1.304876e-1` → `1.30e-1`, `pod_direct["k8"]["traj_rel_mean"]` | Worsens POD slightly. |

- The end-to-end weakall identity does not reproduce all printed iteration digits in every cell. At K=4:

  - `lspg:full:fd`: cold `11.2`, warm `5.51`
  - `lspg:full:weakall`: cold `11.3`, warm `5.50`

  These are the values printed in `2482870.out`, backed by `iters_cold_step0=11.25/11.3125` and `iters_warm_mean=5.51403/5.50128`. The trajectory errors do agree at their printed precision. N=128 has no weakall rollout variant, so the end-to-end claim was not tested in that cell.

- “Every step terminates on the stall criterion” is literally false. For K=8 weak64, `rom["lspg:full:weak64"]["reasons"]` is exactly `{"stalled": 795, "budget": 5}`. The advertised 795/800 and 5/800 split is correct, as is `res_over_res_init_mean=4.496e-2`; “the tolerance never binds” is correct only for this LSPG arm. Galerkin arms have hundreds of `tol` terminations.

- The N-flatness ratios must be recomputed from raw values:

  - EQ step: `3.494243/3.242271 = 1.077715×`, i.e. `1.08×`, not `1.09×`. The README is conservative here.
  - EQ rollout: `0.120359397/0.119183999 = 1.009862×`, i.e. `1.010×`, not `1.008×`. The smaller stated ratio flatters flatness and is repeated three times.
  - Oracle estimate: `0.012019619/0.011556218 = 1.04010×`, not `1.03×`; the stated value understates degradation.
  - FOM `2.461×`, full-step `3.918×`, error `0.9307×`, and rollout speedups `0.4134×/1.0075×` are correctly rounded.

- The timing table’s `287 ms` end-to-end number matches neither available interpretation:

  - The directly timed realizable pipeline is `timing["lspg:eq256:weak64"]["end_to_end_jit_ic_s"] = 293.900 ms`, giving `0.16763×`.
  - Adding the three independent raw medians gives `155.570 + 119.184 + 10.007 = 284.761 ms`, giving `0.17301×`.

  Because the row says “one realizable pipeline,” the direct value should be about `294 ms`; `287 ms` flatters runtime by 2.35%. The displayed `0.17×` remains correct either way.

- The timing protocol is not uniformly “median of 7 after 2 warm-ups.” `ic_solve_py` uses three repetitions after one warm-up, as does the Python-loop Galerkin timing. The table’s Python IC row is therefore mislabeled by its section heading.

- The `119 ms` scan rollout averages `2.384 ms/step`, not `3.2 ms/step`. The latter is the separately measured `rom[...]["step_time_ms_median"]=3.242 ms` from stepwise accuracy runs. Both are valid measurements, but the timing-table note conflates them.

- The oracle-gap multipliers in the verdict are reversed:

  - K=8 best/oracle: `0.018671587/0.011556218 = 1.6157×`, hence `1.6×`, not `1.5×`. The README flatters K=8.
  - K=16 best/oracle: `0.009728120/0.006706357 = 1.4506×`, hence `1.5×`, not `1.6×`. The README unfairly worsens K=16.

- “Consistent in mean, median and max at every K” is false when the best weak arm is compared with the best strong arm. At K=16, weak256 has the better mean and max, but a worse median: `1.04719e-2` versus Galerkin FD’s `9.97182e-3`.

- The statistical-summary arithmetic has three errors:

  - The eq256 penalty is not consistent in sign: relative to full weak64 it is `−0.873%` at K=4, `+5.013%` at K=8, `+6.330%` at K=16, and `+4.769%` at N=128.
  - Grid versus meshfree at `m=256` is not always below 1%: meshfree differs by `−2.934%` at K=16 and `−1.752%` at N=128, although these remain small aggregate differences.
  - Discrete versus continuum weak64 differs by `0.327%` at K=16, not `<0.2%`.

- The EQ diagnostic range is not consistently defined. Across all displayed EQ arms/cells, `eq_info["val_rel_proj_err"]` ranges from `7.373e-4` to `3.411e-3`, not `9.4e-4` to `3.4e-3`. If restricted to N=64 weak64 grid/meshfree arms, the range is `9.397e-4` to `2.819e-3`. All archived `w_min` values are positive and every `w_finite` is true.

- “Same-solver POD essentially attains its projection floor at every k” is false. At N=64, POD weak64 at `k=64` is `5.1959e-2` versus a projection floor of `6.3346e-3`, an 8.20× gap. Even full-FD same-solver POD is `7.3022e-3`, about 15% above the floor.

- The decay triple `5.7e-2/2.8e-1/7.7e-1`, and hence its derived `~17×`, is not stored in any supplied JSON or log. It may be reproducible from regenerated trajectories, but it is not traceable from the archived outputs as written.

## OVERREACH

- **Weak versus exact residual — descriptive result supported; mechanism not supported.** The K=8 mean ratio is genuinely `1.793×`. But weak64 is a truncated, weighted 64-component objective, not merely an orthogonal rotation of the 3844-component residual. The README correctly admits that the high-frequency-discard story is unproved, but then asserts that `PhiᵀJ` is “far better conditioned” without condition-number data. Effective termination also differs: weak64 has 795 stalls/5 budgets and 6.30 warm evaluations, while weakall has 781 stalls/19 budgets and 8.48 evaluations.

  Change to: “On these TEST trajectories, K=8 LSPG weak64 had 1.793× lower mean error than LSPG full FD/weakall. The experiment does not distinguish spectral filtering from weighting, Jacobian conditioning, or solver-path/stopping effects.”

- **Best M grows with K — weak exploratory trend only.** The apparent minima were chosen on TEST, K=4 weak16 versus weak32 differs by only about 11%, K=8 weak64 versus weak128 by 2%, and K=16 weak128 versus weak256 by 1.4%. The latter two are explicitly called ties elsewhere. The verdict’s `M ~ 16K` also contradicts the earlier warning that three K values are not a scaling law.

  Change to: “The tested mean-error minima move toward larger M as K increases, with broad, statistically unresolved plateaus; no scaling law was established.”

- **`M >= 4K` is not a hard requirement.** Only two `M=K` cases performed poorly; no rank or conditioning diagnostic establishes the proposed mechanism. Moreover, `M=2K` arms completed without blow-ups. The README’s caveat admits the factor four is unmeasured, but “Use M >= 4K” remains too categorical.

  Change to: “Avoiding `M=K` appears prudent in these two cases; the required oversampling factor was not measured.”

- **“Hyper-reduction is essentially free” is too strong.** The runtime result is solid: at N=64 K=8, the separately measured step median improves by `15.9906/3.2423 = 4.93×`. Accuracy changes range from a 0.9% improvement to a 6.3% degradation across cells, but there are no paired samples or confidence intervals.

  Change to: “Hyper-reduction reduced the K=8 step median by 4.93× while changing aggregate mean error by about 5%; across tested cells the change ranged from −0.9% to +6.3%.”

- **Meshfree versus grid is an aggregate tie, not a demonstrated equivalence.** Mean errors differ by at most 2.93%, with inconsistent sign, which is consistent with a tie for n=16. But there is one random pool seed, and the EQ validation latents are disjoint from the fit while still drawn from the training-latent population. The worst archived per-latent projection diagnostic reaches about `1.04e-2`. Also, meshfree EQ approximates the exact discrete-mode projections; it does not literally evaluate the grid-defined FOM operator exactly.

  Change to: “The single grid and meshfree rules produced aggregate errors within 3%; meshfree targets the same discrete weak residual but introduces fitted quadrature error.”

- **Galerkin and LSPG do not “coincide.”** Their weak64 mean errors agree within 0.31%, but their iteration counts and algorithms differ, and no state-by-state identity was archived. The filtering explanation is asserted, not isolated.

  Change to: “For weak64, Galerkin and LSPG trajectory-error aggregates are indistinguishable at the available precision.”

- **The nonlinear manifold does not buy 7× in latent dimension.** The supported fact is a 6.93× error reduction at equal dimension eight. The claimed POD rank near 40 was not run; the matching rank lies somewhere between tested ranks 32 and 64. That implies roughly a 4–8× dimension interval, not a measured 7× dimension advantage.

  Change to: “At dimension eight, the coordinate ROM’s mean error is 6.93× lower than POD weak64; matching POD rank was not measured and lies between the tested ranks 32 and 64.”

- **N-flatness is mechanistically plausible but empirically weak.** Fixed `m`, M, and K remove n from the latent-step graph, and the same-node result supports that mechanism. Nevertheless it is one timing trajectory, one variant, two separately trained models, two mesh sizes, and one GPU. “Online cost” is also too broad: the IC solve and full decode scale with n. The later rollout-only caveat is candid, but the earlier headline should say “hyper-reduced rollout cost.”

- **The projected-IC ceiling is optimistic.** Even if the N=128 IC cost became zero, the current rollout plus full decode would cost `120.359+42.381=162.740 ms`, yielding only `0.745×` versus the FOM, not `≈1×`. About `1.008×` is reachable only if the decode/output cost is excluded too.

- **The published-ROM comparison is not a controlled win.** The later caveat is unusually explicit and adequate, but “beaten at exactly the same speedup” is still too strong because hardware, test set, code, and timing protocol differ.

  Change to: “The cross-run published point reports a similar rounded speedup and higher error; this is not a head-to-head comparison.”

- **The negative timing verdict is strong enough; if anything, it is overgeneralized.** The data support: “None of the tested neural configurations beats this FOM end-to-end.” They do not support the universal “a neural decoder cannot compete,” nor is a single 256-point decoder evaluation separately timed. The direct-POD range itself is correct: `5.456×–37.657×`.

- **Statistical claims exceed the archive.** Stage-2 JSONs contain mean/median/max but no 16 per-trajectory rows, so sampling uncertainty and paired significance cannot be computed. The 20% rule is an unvalidated heuristic. The following should be called ties or unresolved:

  - K=8 weak64 versus weak128.
  - K=16 weak128 versus weak256.
  - K=4 weak16 versus weak32 for declaring a unique best M.
  - Weak256 versus Galerkin FD at K=16: 12% mean advantage but a worse median.
  - Grid versus meshfree, discrete versus continuum, alpha 1 versus 0, and the roughly 5% EQ penalty.
  - Strong-form Galerkin versus LSPG at K=4/K=8 does not satisfy the README’s own “20% in all three statistics” rule, although its direction is consistent.

  The large weak-versus-full-residual and coordinate-versus-POD gaps are much more credible for these fixed trained models, but model-training variability is unknown because there is one training seed/run per K.

- **TEST selection is acknowledged but still real selection.** “No variant was selected on TEST” is contradicted by bolding and quoting the minimum among 19 correlated TEST-evaluated arms. Full reporting does not remove winner’s bias.

  Change to: “All variants were evaluated and retrospectively ranked on TEST; best-per-K values are exploratory and selection-biased.”

- **Oracle access is clearly disclosed, but “floor” is too absolute.** `oracle_inferred_latent_test` is consistently labeled as held-out and unavailable to the ROM, so there is no hidden leakage. However, it is a budget-40, two-start solution of a nonconvex latent fit, not a proven representation minimum.

  Change “oracle floor” to “budget-40 inferred-latent oracle estimate.” The POD projection floor, by contrast, is a genuine projection floor.

- Material caveats still missing: timing uses only the first test trajectory; no training-seed or EQ-pool-seed replication; no uncertainty intervals; no compilation/model-loading or host-transfer cost in “end-to-end”; no out-of-distribution family or longer-horizon test; and no warning that the N=64/N=128 accuracy comparison uses separately trained decoders.

## SOUND

- Every identity-check table entry matches `report["checks"]` at the displayed precision in all four reports.

- Every Stage-1 table cell matches the two Stage-1 JSONs, including the oracle, both single-objective arms, both `IC_W` settings, and all means/medians/maxima/z-errors. The `3.4×` oracle ratio and `2.4e-2` true-z residual are numerically correct, although “explained by” should be weakened to “consistent with.”

- Apart from the explicitly listed rounding errors and the POD weak64 `k=64` copy error, the large Stage-2 table matches `train_rel_mean`, `ic_fit`, the inferred-latent oracle, all ROM means, POD projection floors, and POD controls.

- All 30 entries in the K=8 per-time table match the corresponding `per_time_mean` values. The terminal weak64/oracle ratio is `2.6023×`. The absolute gap is roughly flat after t-index 10, though the ratio is not.

- Zero blow-ups is fully verified: all 68 coordinate arms and all 80 POD arms have `n_blowup=0`, `n_completed=16`, and `n_total=16`.

- The K=8 weak64 `795 stalled / 5 budget` split is exact. No tolerance termination occurs in that specific LSPG arm.

- The main timing facts are correct: FOM `49.268 ms`, jitted IC `155.570 ms`, Python IC `235.729 ms`, EQ rollout `119.184 ms`, decode `10.007 ms`, POD-direct k=8 `3.453 ms` and `14.267×`, POD-direct k=64 `9.031 ms` and `5.456×`. N=128 POD speedups round correctly to `35×` and `13×`.

- Provenance checks pass. The five `.out` files carry full commit `90171caef6800d6decc71aab627b126c6b8db80d`, `dirty=0`, and GPU backend. K=8 N=64 and N=128 both report `host=pax007`, `NVIDIA A100 80GB PCIe`; K=4/K=16 report A100-PCIE-40GB. Job IDs match the log filenames, and timestamps are on 2026-08-17.

- The README honestly discloses heavy tails, absence of archived per-trajectory Stage-2 rows, TEST-side best-arm bias, the two-point nature of N-flatness, rollout-only versus end-to-end scaling, and the uncontrolled cross-code published comparison.

- Accuracy arms are not compared using a median for one arm and a mean for another. Accuracy tables consistently use mean-over-time then mean-over-trajectories; timing medians are a separate metric.