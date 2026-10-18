# Numerical experiments and results

This manuscript-style extract reports the completed campaign using the corrected paper snapshot. The measurements are collected; reporting remains provisional where independent audits or audit dispositions are outstanding, as noted beside the affected results.

The experiments assess three questions: whether a nonlinear head improves compression at matched dimension; whether nested corrections provide an accuracy–cost family without retraining; and whether any resulting operating point is competitive with tuned full-order and linear reduced models. We report negative results alongside the successful cases.

For the Burgers panels, $E_{\mathrm{evol}}$ is the maximum error over cases and evolved output times, $E_{\mathrm{all}}$ includes the initial state, and $E_0$ is initial-state compression error. These errors use the initial-field norm and the converged same-grid solution; $E_{\mathrm{ref}}$ instead uses the fine numerical reference. Other experiments retain their stated metrics, identified in the captions. Costs are medians of repeated measurements after GPU burn-in. Device time charges resident computation; complete-query time includes host input and output. All cost comparisons are within an allocation. For the Burgers panels, only solves meeting the fixed stationarity threshold $10^{-6}$ at every step, or the residual exit rule, and the applicable quadrature checks are admissible. A nondominated point has no admissible competitor with both lower cost and lower error.

On the smaller Burgers meshes, tuned full-order solvers dominate every admissible reduced model. At the largest mesh, the lowest-rank quadrature models reach the evolved-error frontier, while initial-state compression removes this advantage when all output times are scored. The largest-mesh panel uses different hardware, so the three rows do not establish a hardware-independent crossover.

**Table 1. Same-allocation Burgers comparisons.**

| Mesh | GPU | Admissible reduced | Nondominated, evolved | Nondominated, all times | $T_{\mathrm{ROM,min}}/T_{\mathrm{FOM,min}}$ |
|---|---|---|---|---|---|
| $256^2$ | NVIDIA A100 80GB PCIe | 27 | 0 | 0 | 4.48$\times$ |
| $512^2$ | NVIDIA A100 80GB PCIe | 23 | 0 | 0 | 2.92$\times$ |
| $1024^2$ | NVIDIA H200 | 16 | 3 | 0 | 2.27$\times$ |

A cost ratio above unity means the cheapest admissible reduced method costs more than the cheapest full-order setting; those settings need not have equal errors. Each row is a separate job. Eligibility follows the corrected fixed-tolerance rule. The intermediate-mesh protocol also has a disclosed post-data interpretation of a transfer gate; frontier membership is an empirical result under that disclosed protocol.

Generated source: [T05m_panel_summary](../worktrees/2026-09-16-paper-refresh/paper/tables-md/T05m_panel_summary.md).

Holding the test count $M$ fixed isolates correction rank $q$. The original fixed-$M=1088$ ladder reduces evolved error by $2.44\times$ for $5.16\times$ cost, with all displayed rungs converged. The smaller fixed-test ladder spans only $1.22\times$. This supports the correction-rank mechanism on the original registered range of one checkpoint. The later extension failed its acceptance rule and is excluded from this span. The separate registered hypothesis that both factors pass their individual thresholds is not satisfied; the qualitative observation that both affect error should not be presented as that formal pass.

**Table 2. Correction rank at fixed test count on Burgers.**

| $M$ | $q$ | $E_{\mathrm{evol}}$ (%) | Device time (ms) |
|---|---|---|---|
| 256 | 0 | 1.2710 | 406.2 |
| 256 | 64 | 1.1255 | 645.6 |
| 256 | 128 | 1.0418 | 921.8 |
| 1088 | 0 | 1.2657 | 848.0 |
| 1088 | 64 | 1.0593 | 1359.4 |
| 1088 | 128 | 0.8711 | 1886.5 |
| 1088 | 256 | 0.5194 | 4377.9 |

All displayed rows belong to the same timing job and development checkpoint. This is the fixed-test experiment; the seed replication below uses a scheduled test count. Provenance and amendment disclosures elsewhere in this lane remain under audit disposition.

Generated source: [T04_rank_vs_tests](../worktrees/2026-09-16-paper-refresh/paper/tables-md/T04_rank_vs_tests.md).

With $M=4(K+q)$, dense and sampled residual evaluation trace similar error ladders. Against the fine reference, the coarse-mesh discretisation error is $4.03\%$; the reduction error falls substantially while reference error changes much less. Quadrature reduces evaluation cost, but a passing stored rule and a reproducible construction are different claims.

**Table 3. Scheduled Burgers ladder and quadrature costs.**

| $q$ | $M$ | Dense $E_{\mathrm{evol}}$ (%) | $E_{\mathrm{all}}$ (%) | $E_0$ (%) | Dense $E_{\mathrm{ref}}$ (%) | Dense ms | EQ $E_{\mathrm{evol}}$ (%) | EQ ms | EQ construction |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 64 | 1.8890 | 2.5629 | 2.5629 | 4.56 | 283.9 | 1.8891 | 58.6 | confirmed (3 of 3 re-draws) |
| 16 | 128 | 1.3985 | 2.4806 | 2.4806 | 4.11 | 360.8 | 1.4270 | 81.1 | confirmed (3 of 3 re-draws) |
| 32 | 192 | 1.2336 | 2.3534 | 2.3534 | 4.08 | 434.7 | 1.2493 | 97.8 | confirmed (2 of 2 re-draws) |
| 64 | 320 | 1.0843 | 2.1489 | 2.1489 | 4.10 | 621.1 | 1.0840 | 148.9 | confirmed (2 of 2 re-draws) |
| 128 | 576 | 0.8930 | 1.8116 | 1.8116 | 4.08 | 1186.9 | 0.8931 | 273.5 | single-draw |
| 256 | 1088 | 0.5194 | 0.9053 | 0.9053 | 4.04 | 3939.8 | 0.5129 | 746.0 | single-draw |

Development cohort; one job. For these selected rules, dense and EQ all-times errors coincide at displayed precision and are set by initial compression. The EQ rules are the replication-selected set. Single-draw results are provisional with respect to construction reproducibility. The uncorrected development result is not a generalisation estimate; the sealed cohort below exposes its failure.

Generated source: [T03m_ladder_main](../worktrees/2026-09-16-paper-refresh/paper/tables-md/T03m_ladder_main.md), [T03_tunability](../worktrees/2026-09-16-paper-refresh/paper/tables-md/T03_tunability.md).

The sealed cohort was opened after choices were frozen. All 4 checkpoints yield monotone ladders, and 3 meet the full ladder criterion. Top-rung error ranges from 0.59% to 0.68%. The incumbent's uncorrected prediction converges to a wrong branch on one sealed case. The generalisation criterion therefore fails, and sealed results replace the development baseline as the headline. An unconverged rung in one retrained seed remains reported as run.

**Table 4. Sealed-cohort Burgers replication.**

| $q$ | $M$ | Incumbent worst evolved (%) | Retrained seeds: mean ± SD of worst evolved (%) | Checkpoints converged |
|---|---|---|---|---|
| 0 | 64 | 10.1120 | 2.8023 $\pm$ 0.6357 | 4 of 4 |
| 16 | 128 | 1.4617 | 1.7305 $\pm$ 0.1297 | 4 of 4 |
| 32 | 192 | 1.4458 | 1.5044 $\pm$ 0.1422 | 4 of 4 |
| 64 | 320 | 1.3252 | 1.3528 $\pm$ 0.1099 | 3 of 4 |
| 128 | 576 | 1.0964 | 1.0461 $\pm$ 0.0619 | 4 of 4 |
| 256 | 1088 | 0.6789 | 0.6188 $\pm$ 0.0345 | 4 of 4 |

The seed statistic averages each checkpoint's worst-case error; it is not the median error over individual cases. These are dense scheduled-test ladders, so they do not independently replicate the fixed-test experiment or quadrature construction. The independent audit supports the ladder verdicts but identifies outstanding check-count and disclosure defects.

Generated source: [T13_sealed](../worktrees/2026-09-16-paper-refresh/paper/tables-md/T13_sealed.md).

**Table 5. Sealed-cohort ladder acceptance by checkpoint.**

| Checkpoint | Monotone, evolved | Monotone, all times | Every rung converged | Error span | Cost span | Full criterion |
|---|---|---|---|---|---|---|
| `incumbent` | yes | yes | yes | 14.89$\times$ | 17.93$\times$ | yes |
| `seed1` | yes | yes | yes | 3.52$\times$ | 13.25$\times$ | yes |
| `seed2` | yes | yes | no | 4.81$\times$ | 13.17$\times$ | no |
| `seed3` | yes | yes | yes | 5.22$\times$ | 14.62$\times$ | yes |

Spans are computed within the sealed-cohort job. The incumbent's large error span includes its failed uncorrected cold start and should not be interpreted as typical improvement.

Generated source: [T13b_sealed_verdicts](../worktrees/2026-09-16-paper-refresh/paper/tables-md/T13b_sealed_verdicts.md).

At matched latent dimension, the nonlinear head is substantially more accurate than the tested linear and quadratic alternatives on Burgers. This comparison isolates compact representation; it does not establish a cost advantage over higher-dimensional POD or a tuned full-order solver.

**Table 6. Matched-dimension head ablation on Burgers.**

| Model | Dimension | Worst same-grid, all times (%) | Device time (ms) | Stationary |
|---|---|---|---|---|
| (a) neural head, EQ | 16 | 2.5629 | 47.6 | yes |
| (b) linear map (fit to head outputs) | 16 | 56.9296 | 19.9 | yes |
| (b) linear map (fit to truth) | 16 | 61.5226 | 19.7 | yes |
| (c) quadratic map | 16 | 31.8276 | 28.0 | yes |
| (e) POD-LSPG $k'{=}16$ | 16 | 61.6503 | 46.0 | yes |

A separate development-cohort job. Linear alternatives are the fitted maps actually tested, rather than a universal bound on every possible linear model. All-times error can be controlled by initial compression.

Generated source: [T06a_head_burgers](../worktrees/2026-09-16-paper-refresh/paper/tables-md/T06a_head_burgers.md).

Rules fitted on stored states are evaluated on held-out reachable states and then independently redrawn. A good fitting residual alone is insufficient. The original timed ladder is monotone for its passing draws, while the higher-rank constructions fail on some redraws. Construction labels must name the population and certification threshold to which their counts apply.

**Table 7. Held-out certification of the original timed quadrature constructions.**

| $q$ | $m$ | $\rho_{\max}$, timed draw | Primary construction status | EQ evolved error (%) | EQ ms | Dense ms |
|---|---|---|---|---|---|---|
| 0 | 1024 | 0.0153 | confirmed (3 of 3 re-draws) | 1.8891 | 59.1 | 292.9 |
| 16 | 1024 | 0.0935 | confirmed (3 of 3 re-draws) | 1.4270 | 80.6 | — |
| 32 | 1024 | 0.0533 | confirmed (2 of 2 re-draws) | 1.2493 | 97.7 | — |
| 64 | 1024 | 0.0531 | marginal (2 of 6 draws pass) | 1.2275 | 120.6 | 624.9 |
| 128 | 2048 | 0.0669 | marginal (4 of 5 draws pass) | 0.8936 | 246.9 | 1190.5 |
| 256 | 2048 | 0.1074 | marginal (1 of 5 draws pass) | 0.5389 | 722.2 | 3915.1 |

The primary threshold is $\rho_{\max}\le 0.116$. Timings compare methods within the original ladder job; construction counts include the separate replication experiment. This table uses the original timed constructions, whereas the scheduled panel above uses later selected rules; their high-rank statuses must not be pooled. Marginal constructions are provisional for reproducibility. Other status-label and retraction disclosures in the lane remain under audit disposition.

Generated source: [T09_eq_ladder](../worktrees/2026-09-16-paper-refresh/paper/tables-md/T09_eq_ladder.md).

On shared Burgers data, the validation-selected U-Net and Transolver are more accurate than the ROM configuration included in the operator comparison on its matched cohort. The FNO is less accurate there. This conclusion does not compare either operator with the highest correction rank, nor does it assert a uniform ordering of validation tails. The U-Net advantage also survives the available precision and second-seed controls.

**Table 8. Neural operators on shared Burgers data.**

| Method | Matched 8-case worst (%) | Validation worst (%) | Validation median (%) | Still improving |
|---|---|---|---|---|
| U-Net, validation-selected | 1.7110 | 7.5176 | 1.0806 | yes |
| Transolver, validation-selected | 1.5224 | 9.3183 | 1.3591 | no |
| ROM comparator | 1.8671 | — | — | — |
| FNO, validation-selected | 2.4829 | 6.3825 | 1.8054 | yes |
| Efficient FOM comparator | 0.9978 | — | — | — |

Error is relative to the fine numerical reference and normalized by the initial-field norm. All methods return the supplied initial state exactly in this comparison. The cohort, reference and ROM rollout convention differ from the correction-ladder panels. Operators were selected by validation mean error, not by the displayed matched-cohort tail. U-Net and Transolver have no same-allocation ROM timing comparison, so no speed ratios are given. Training-limit and pending-independent-audit qualifications apply.

Generated source: [T14_operators](../worktrees/2026-09-16-paper-refresh/paper/tables-md/T14_operators.md).

For the linear PDEs, unrestricted evolution or solution in the learned bank removes the nonlinear optimization cost. On Poisson the direct linear endpoint is more accurate and cheaper than the head, but the full-order transform remains superior. On heat it improves both error and cost relative to the head. On waves it attains nearly the best correction-rung accuracy at substantially lower solver cost, while matched-rank POD is more accurate. The small difference between the wave correction rung and the linear endpoint does not support claiming that the latter is strictly the most accurate rung.

**Table 9. Linear-PDE endpoints and competitive controls at the finest reported mesh.**

| PDE | Mesh | Method | Error metric | Worst error (%) | Device ms | Complete-query ms |
|---|---|---|---|---|---|---|
| Poisson | $1024^2$ | $q{=}0$ | Same-grid | 3.1495 | 3.941 | 6.394 |
| Poisson | $1024^2$ | linear top rung ($q{=}R$, QR) | Same-grid | 0.7421 | 1.830 | 4.424 |
| Poisson | $1024^2$ | POD $k'{=}512$ | Same-grid | 0.1838 | 4.572 | 6.972 |
| Poisson | $1024^2$ | direct DST | Same-grid | 0.0000 | 0.237 | 3.248 |
| Heat | $1024^2$ | linear bank, exact modal evolution ($q{=}R$, no head) | Physical, current-relative | 1.675830 | 0.561659 | 17.681166 |
| Heat | $1024^2$ | nonlinear head ($k{=}8$) | Physical, current-relative | 4.555479 | 12.318828 | 36.325427 |
| Heat | $1024^2$ | same-grid direct solve | Physical, current-relative | 0.000350 | 1.034563 | 25.068664 |
| Wave | $1024^2$ | `head_q0` | Energy-state | 11.3388 | 192.462 | 451.994 |
| Wave | $1024^2$ | `nested_q32` | Energy-state | 5.1164 | 2529.445 | 2789.801 |
| Wave | $1024^2$ | `linear_bank64` | Energy-state | 5.1335 | 4.562 | 265.785 |
| Wave | $1024^2$ | `pod_k64` | Energy-state | 1.5010 | 4.232 | 264.860 |
| Wave | $1024^2$ | `dst` | Energy-state | 0.0000 | 17.103 | 277.324 |

Compare methods within each PDE block only; the blocks use different error norms and jobs. The wave state includes displacement and velocity in its energy norm. An error displayed as zero is agreement at printed precision. Poisson and wave report audits remain pending; heat is an earlier independently audited cell. Device-only and complete-query costs can lead to different practical comparisons.

Generated source: [T11b_poisson](../worktrees/2026-09-16-paper-refresh/paper/tables-md/T11b_poisson.md), [T11d_heat](../worktrees/2026-09-16-paper-refresh/paper/tables-md/T11d_heat.md), [T11a_waves](../worktrees/2026-09-16-paper-refresh/paper/tables-md/T11a_waves.md).

On the L-shaped domain, the loss of a separable full-order transform changes the cost comparison. The reported head is $2.77\times$ and $6.07\times$ cheaper than the cheapest measured same-job full-order setting at the two larger meshes. POD is cheaper still, with approximately 15% more worst-case error at the finest mesh. This is a reduced-model cost benefit on a linear PDE, not evidence that nonlinearity is necessary.

**Table 10. L-shaped Poisson: complete-query cost and same-grid error.**

| Mesh | Cheapest FOM | FOM error (%) | FOM ms | POD-128 error (%) | POD ms | Head $q=64$ error (%) | Head ms | FOM/head cost |
|---|---|---|---|---|---|---|---|---|
| $64^2$ | sparse direct (SuperLU) | 0.000 | 1.28 | 2.591 | 2.711 | 2.300 | 2.846 | 0.45 |
| $128^2$ | sparse direct (SuperLU) | 0.000 | 2.48 | 2.483 | 2.764 | 2.164 | 2.806 | 0.88 |
| $256^2$ | sparse direct (SuperLU) | 0.000 | 8.39 | 2.457 | 2.850 | 2.131 | 3.028 | 2.77 |
| $512^2$ | CG $10^{-2}$ | 0.385 | 29.13 | 2.451 | 4.031 | 2.123 | 4.801 | 6.07 |

Every ratio uses its row's allocation. The head uses the signed-distance bank and the same reported correction rank across meshes. The numerical verifier was repaired and its comparisons match. Reporting remains provisional pending disclosure of post-data gates and fine-reference coverage. A separate free-bank experiment selected its test count using development data; it is not included here. No matched-rank POD-512 control was run on this domain.

Generated source: [T18m_lshape_main](../worktrees/2026-09-16-paper-refresh/paper/tables-md/T18m_lshape_main.md).

The tested decaying Navier–Stokes family does not establish the required nonlinear representation advantage. Across the retained settings, the ratio of POD error to best-found head error is 1.15–1.46, below the registered factor-of-two threshold. Increasing latent dimension, narrowing the family, and increasing training data do not produce a passing head. Some best-found fits terminate at their iteration budget, so these values are upper bounds on the achievable manifold error. Training-versus-held-out error ratios use different normalizations in the existing record and are not used here to claim a demonstrated capacity diagnosis.

**Table 11. Navier–Stokes representation gate on retained heads.**

| Setting | $K$ | Bank rank | Best-found median (%) | POD-K median (%) | POD/head error | Pass ≥2 |
|---|---|---|---|---|---|---|
| $256^2$ | 16 | 256 | 20.2417 | 24.0219 | 1.19 | no |
| $256^2$ ($K{=}32$) | 32 | 512 | 12.0948 | 13.9455 | 1.15 | no |
| $256^2$ (family dim. 8) | 16 | 256 | 5.2823 | 7.3651 | 1.39 | no |
| $256^2$ (2048 traj.) | 16 | 256 | 16.6598 | 24.2492 | 1.46 | no |

The unqualified mesh row is the original baseline; other rows identify changes to latent size, family or data. These are best-found representation errors, not deployed-trajectory errors. Original failed bank attempts are excluded and retained in the retraction history. The head advantage threshold was unchanged; FOM certification depends on separately disclosed amended gates.

Generated source: [T11e_ns](../worktrees/2026-09-16-paper-refresh/paper/tables-md/T11e_ns.md).

An exploratory ladder was run after the head gate failed. It reduces median trajectory error with correction rank, but the worst-case error increases at the first correction rung. POD-LSPG is more accurate at every matched dimension and cheaper at all but the largest one. No neural point is nondominated. The matched solve/manifold ratio is used below; the earlier comparison mixed different aggregations and was withdrawn.

**Table 12. Exploratory Navier–Stokes correction ladder after the failed representation gate.**

| $q$ | Matched dimension $K+q$ | Neural worst (%) | Neural median (%) | Neural ms | Solved/manifold | POD worst (%) | POD ms |
|---|---|---|---|---|---|---|---|
| 0 | 32 | 68.7654 | 41.3922 | 13387.5 | 2.51 | 54.6331 | 212.1 |
| 32 | 64 | 70.7701 | 36.8058 | 16979.1 | 2.31 | 35.5092 | 431.0 |
| 64 | 96 | 61.5127 | 32.7278 | 21289.9 | 2.20 | 27.7285 | 832.7 |
| 128 | 160 | 56.5456 | 25.6655 | 29011.5 | 1.91 | 17.1958 | 2398.2 |
| 256 | 288 | 33.0536 | 13.8903 | 46267.9 | 2.04 | 9.5434 | 9963.9 |
| 512 | 544 | 6.0140 | 2.4377 | 52592.3 | 1.40 | 4.1664 | 53902.5 |

Worst and median refer to the per-case maximum over evolved times. Solved/manifold uses the median-over-cases worst-evolved statistic on both sides. All times are from one job. This is exploratory, not the confirmatory phase on a passing head. Zero evolution budget exits in the source do not imply that every initial-state fit converged.

Generated source: [T11g_ns_ladder](../worktrees/2026-09-16-paper-refresh/paper/tables-md/T11g_ns_ladder.md).

Lower viscosity increases the representation difficulty of the linear models more than that of the nonlinear head. The fixed-test ladder spans 1.36$\times$ in evolved error for 3.25$\times$ cost, below the registered accuracy-span bar. Every neural rung is more accurate than the tested POD-LSPG ranks, yet a tuned full-order setting dominates all reduced subjects. The converged discrete reference itself has 20.8% error against the fine reference. Consequently these are under-resolved, same-discretisation comparisons; they do not establish a corresponding continuum-accuracy advantage.

**Table 13. Under-resolved low-viscosity Burgers: fixed-test ladder and controls.**

| Method | Worst evolved (%) | Worst all times (%) | Fine-reference error (%) | Device ms |
|---|---|---|---|---|
| $q=0$, $M=1088$ | 9.0500 | 9.0500 | 17.98 | 557.4 |
| $q=16$, $M=1088$ | 8.7562 | 8.7562 | 18.03 | 681.4 |
| $q=32$, $M=1088$ | 8.6360 | 8.6360 | 18.08 | 772.2 |
| $q=64$, $M=1088$ | 8.3165 | 8.3165 | 18.36 | 913.2 |
| $q=128$, $M=1088$ | 7.6276 | 7.6276 | 18.70 | 1220.7 |
| $q=256$, $M=1088$ | 6.6718 | 6.6718 | 19.44 | 1810.3 |
| POD-LSPG $k'=256$, $M=1024$ | 11.2540 | 15.2899 | 21.96 | 881.2 |
| `nt1e-2_dt01` | 3.8950 | 3.8950 | 19.33 | 14.0 |

The POD row is the most accurate tested POD-LSPG subject on evolved error; the FOM row is the cheapest full-order setting dominating all reduced subjects. No resolved confirmation panel was run. These conclusions remain provisional pending the independent report audit and resolution confirmation.

Generated source: [T20_lowvisc_ladder](../worktrees/2026-09-16-paper-refresh/paper/tables-md/T20_lowvisc_ladder.md), [T20b_lowvisc_panel](../worktrees/2026-09-16-paper-refresh/paper/tables-md/T20b_lowvisc_panel.md).

At a frozen checkpoint, the Burgers cached solve changes from 44.16 to 43.58 ms while the state size increases 264-fold. Complete-query time increases by $1.52\times$. This separates mesh-independent reduced work from dense input/output. The inherited configuration does not beat its cheapest full-order comparator; the later corrected panel above is a different experiment.

**Table 14. Frozen-checkpoint mesh transfer.**

| PDE | Intervals | Unknowns | Cached ms | Device ms | Complete-query ms | Worst reference error (%) |
|---|---|---|---|---|---|---|
| Burgers | 64 | 3,969 | 44.2 | 48.5 | 50.4 | 10.8570 |
| Burgers | 128 | 16,129 | 44.7 | 46.5 | 48.1 | 6.5431 |
| Burgers | 256 | 65,025 | 44.7 | 48.2 | 50.6 | 4.5546 |
| Burgers | 512 | 261,121 | 45.1 | 47.4 | 53.7 | 3.7168 |
| Burgers | 1024 | 1,046,529 | 43.6 | 50.1 | 76.8 | 3.8847 |
| Poisson | 64 | 3,969 | 2.1 | 2.3 | 3.2 | 6.1119 |
| Poisson | 128 | 16,129 | 2.1 | 2.2 | 3.2 | 6.1107 |
| Poisson | 256 | 65,025 | 2.1 | 2.2 | 3.3 | 6.1106 |
| Poisson | 512 | 261,121 | 1.9 | 2.3 | 3.8 | 6.1106 |
| Poisson | 1024 | 1,046,529 | 2.0 | 3.0 | 6.6 | 6.1106 |

One job per PDE with no retraining across meshes. Cached work excludes dense query input/output; all three cost definitions are preserved.

Generated source: [T10_mesh_ladder](../worktrees/2026-09-16-paper-refresh/paper/tables-md/T10_mesh_ladder.md).

Fusing residual and Jacobian work and folding the head output layer into the bank reduces implementation cost while preserving the solved fields. The measured improvement is below the intended factor-of-two target; kernel optimization alone does not change the representation floor.

**Table 15. Implementation speed at numerical parity.**

| attempt | intervals | arm | incumbent ms | arm ms | GPU speedup | incl. host | parity | 2$\times$ target |
|---|---|---|---|---|---|---|---|---|
| spd01 | 256 | `L4` | 46.856 | 30.824 | 1.520x | 1.478x | 2.565e-13 | FAIL |
| fine01 | 512 | `L4` | 46.928 | 31.106 | 1.509x | 1.454x | 5.053e-13 | FAIL |
| fine01 | 1024 | `L4` | 49.283 | 33.544 | 1.469x | 1.310x | 6.951e-13 | FAIL |
| comp01 | 256 | `C1` | 46.627 | 30.723 | 1.518x | 1.482x | 2.562e-13 | FAIL |
| comp01 | 1024 | `C1` | 49.804 | 33.282 | 1.496x | 1.408x | 7.129e-13 | FAIL |

Speedups use paired timings from each attempt. Parity is a field-disagreement diagnostic, not a comparison with the physical reference. The arm identifiers are source implementation labels.

Generated source: [T15_speed](../worktrees/2026-09-16-paper-refresh/paper/tables-md/T15_speed.md).

Stopping tolerances primarily reduce work after the representation error has saturated. Aggressive iteration caps instead cause early termination and large errors. These observations motivate correction rank as the accuracy control and solver settings as secondary cost controls.

**Table 16. Solver controls on the inherited held-out Burgers cohort.**

| setting | worst % (32 held-out) | device ms | early-stopped |
|---|---|---|---|
| EQ $m{=}256$, tol $10^{-6}$, cap 180 | 7.245 | 51.0 | 0/96 |
| EQ $m{=}512$, tol $10^{-8}$ | 6.712 | 64.7 | 0/96 |
| EQ $m{=}512$, tol $10^{-3}$ | 6.701 | 41.6 | 0/96 |
| EQ $m{=}512$, cap 2 (early-stopped) | 90.324 | 28.4 | 96/96 |
| FOM Newton $10^{-2}$, $\Delta t{=}0.01$ | 6.807 | 9.9 | 0/96 |
| FOM Newton $10^{-2}$, $\Delta t{=}0.005$ | 35.357 | 17.4 | 0/96 |
| FOM Newton $10^{-4}$, $\Delta t{=}0.005$ | 6.171 | 22.4 | 0/96 |
| FOM Newton $10^{-6}$, $\Delta t{=}0.005$ | 6.172 | 88.3 | 0/96 |
| FOM $128^2$, $\Delta t{=}0.005$ | 9.849 | 21.2 | 0/96 |
| FOM $64^2$, $\Delta t{=}0.01$ | 16.095 | 14.0 | 0/96 |

This earlier cohort and reference differ from the main panel. Rows are reported as measured; the loose-tolerance reduced rows do not automatically satisfy the corrected main-panel eligibility rule.

Generated source: [T08_solver_knobs](../worktrees/2026-09-16-paper-refresh/paper/tables-md/T08_solver_knobs.md).

The training sweep varies data volume, latent dimension, reconstruction objectives and smoothness terms. None of these retrains improves on the incumbent's best-found representation error. The nominal recipe reproduction is itself worse than the incumbent, so this experiment does not establish a general absence of benefit from additional data or model capacity.

**Table 17. Burgers head-training study.**

| arm (EQ query) | $K$ | bank floor % | best-found % | solved all % | solved evolved % | device ms | conv. |
|---|---|---|---|---|---|---|---|
| incumbent (4608 traj., $K{=}16$) | 16 | 0.3918 | 2.5447 | 2.5629 | 1.9002 | 48.7 | yes |
| 128 traj., $K{=}16$ | 16 | 0.3918 | 12.6496 | 12.6496 | 7.2529 | 45.4 | yes |
| 512 traj. | 16 | 0.3918 | 6.8202 | 6.8204 | 3.7153 | 39.3 | yes |
| 2048 traj. | 16 | 0.3918 | 3.5570 | 3.5574 | 1.8752 | 46.6 | yes |
| 4608 traj. (like-for-like retrain) | 16 | 0.3918 | 3.9616 | 3.9710 | 1.9522 | 59.1 | yes |
| 128 traj., $K{=}32$ | 32 | 0.3918 | 8.8489 | 8.8496 | 6.2351 | 80.7 | yes |
| 2048 traj., $K{=}32$ | 32 | 0.3918 | 3.1132 | 3.1137 | 2.4882 | 54.5 | yes |
| 128 traj., weak-residual term | 16 | 0.3918 | 14.2164 | 14.2165 | 7.5935 | 43.9 | yes |
| 128 traj., trajectory term | 16 | 0.3918 | 12.8237 | 12.8238 | 7.8543 | 42.8 | yes |
| 128 traj., code-smoothness term | 16 | 0.3918 | 13.6497 | 13.6505 | 13.4135 | 36.8 | yes |
| selected: 2048 traj., $K{=}32$, weak term | 32 | 0.3918 | 2.8289 | 3.1275 | 3.1275 | 55.1 | yes |
| joint bank$+$head, $R{=}512$ | 32 | 2.8517 | 5.0383 | 5.0471 | 2.4399 | 46.3 | yes |

All query costs are from the evaluation allocation. Bank floor and best-found values describe representation; solved columns describe deployed queries. Training schedules do not reproduce the incumbent exactly, limiting causal interpretation. The withdrawn wider joint-bank arm is not included.

Generated source: [T16_training](../worktrees/2026-09-16-paper-refresh/paper/tables-md/T16_training.md).

The experiments establish a correction-rank family and a strong compact-representation advantage on the tested Burgers family. They support a narrower computational conclusion: some reduced operating points are useful at the largest Burgers mesh and on the nonseparable Poisson domain, while linear reduced models or tuned full-order solvers are often preferable. No operator-wide superiority, general nonlinear-manifold speedup, or successful confirmatory Navier–Stokes ladder follows from these measurements.

**Provenance and reporting status.** All empirical numbers and table cells above are selected from the committed paper's generated tables and number JSON, at `10498e2e0e264661d702b7606f49133ef7d5be17`. The paper generator reads lane summary, analysis and audit JSONs and generated reports derived from run artifacts. The [extract manifest](2026-09-19-paper-results.manifest.json) preserves every input hash, table mapping and the paper's upstream provenance. This extract changes presentation only; it is not a new raw-field audit. Independent report audits remain pending for no-second, p-linear, w-ladder and b-lowvisc; open dispositions in other lanes are stated beside the affected tables. The corrected Burgers admissibility rule and NS matched-statistic comparison are included. No proposed follow-up experiment is presented as measured evidence.

**Glossary.**

- **FOM / ROM / NM-ROM:** full-order numerical model; reduced-order model; reduced model whose trial states are constrained by a nonlinear decoder.

- **Bank / head / bank floor / best-found:** the learned spatial basis; the small nonlinear map into its coefficients; best linear projection error in that basis; lowest reconstruction error found by the recorded multistart fitting procedure. Best-found is an upper bound, not a certified global optimum.

- **$K$, dimension, $R$, $q$, $M$, $m$:** head latent dimension; number of solved coordinates; spatial-bank rank; number of added correction coefficients; weak-test count; quadrature-node count. The coupled solve has dimension $K+q$. At $q=R$, the reachable states span the full bank.

- **Rung / ladder / fixed-test / scheduled-test:** one correction-rank setting; a collection of settings; constant weak-test count; weak-test count increased with the number of unknowns.

- **Worst / median / all times / evolved / initial / vs ref / fine-reference error:** maximum or median under the caption's stated aggregation; including the initial state; excluding it; the initial-state error alone; error against a finer numerical reference. $E_{\mathrm{evol}}$, $E_{\mathrm{all}}$, $E_0$ and $E_{\mathrm{ref}}$ denote those Burgers errors.

- **Same-grid / physical current-relative / energy-state:** comparison with the converged solution of the same discrete equations; heat error normalized by the physical reference at the evaluated time; the wave error norm combining displacement and velocity. These norms are not interchangeable.

- **Development / validation / sealed / checkpoint / seed / SD:** cases used during design; cases used to select trained models; cases opened only after choices were frozen; saved learned parameters; random initialization or sampling identifier; sample standard deviation across retrained checkpoints.

- **Admissible / stationary / converged / valid / conv. / early-stopped / budget exit:** accepted under the specified accuracy and solver gates; meeting the gradient criterion; meeting an accepted stopping rule; count of accepted invocations; abbreviation for convergence; stopping without convergence; reaching an iteration limit. Evolution and initial-state exits are distinct.

- **Nondominated / frontier / full criterion / pass:** no accepted competitor is both cheaper and more accurate; the set of such methods; the stated conjunction of ladder requirements; whether that conjunction or a named gate is met. Frontier membership is defined relative to the measured candidate set and metric.

- **Device ms / cached ms / complete-query ms / incl. host / GPU speedup / cost span:** resident computation time; reduced computation with dense input/output excluded; host-to-host query time; speed ratio including host overhead; old/new device time within an attempt; largest/smallest cost in one ladder. A millisecond is one thousandth of a second.

- **$T_{\mathrm{ROM,min}}/T_{\mathrm{FOM,min}}$, FOM/head cost, error span, POD/head, solved/manifold, parity:** cheapest reduced/full-order time ratio; full-order/head time ratio; largest/smallest error in the stated ladder; POD representation error divided by best-found head error; matched solved/best-found error ratio; relative disagreement between implementation outputs.

- **EQ / NNLS / $\rho_{\max}$ / confirmed / marginal / single-draw:** empirical quadrature; nonnegative least-squares fitting of its weights; maximum held-out relative error of its projected nonlinear term; every recorded redraw passed the named bar; some recorded draws failed; only one draw measured. A draw is one sampled rule-construction dataset. A missing table entry means not supplied or not measured, not zero.

- **POD / POD-LSPG / linear map / quadratic map / QR / DST / CG / LM:** proper orthogonal decomposition, a linear basis fitted to training states; least-squares Petrov–Galerkin evolution in that basis; the fitted linear or quadratic coefficient map; orthogonal matrix factorization for a direct reduced solve; discrete sine transform; conjugate gradient; Levenberg–Marquardt nonlinear least squares.

- **U-Net / FNO / Transolver / still improving:** convolutional encoder-decoder operator; Fourier neural operator; attention-based neural operator; validation improvement was continuing at the training limit. Operator architectures and capacities were selected under the original protocol, not reselected for this extract.

- **Signed-distance bank / sparse direct / SuperLU / under-resolved:** basis with a boundary-vanishing distance factor; solution by sparse matrix factorization; the sparse-direct implementation; mesh error large enough to prevent a continuum-accuracy conclusion.

- **Attempt / arm / L4 / C1 / incumbent / trajectory term / weak term / code smoothness / joint bank+head:** one run directory; one tested configuration; recorded implementation variants; original checkpoint; a temporal training objective; projected-residual training term; penalty on variation of latent codes; simultaneous training of both model parts. Their numerical outcomes retain the source configuration labels.

- **Job / allocation / GPU / A100 / H200 / f64 / commit / SHA256:** scheduler execution identifier; assigned compute resources; graphics processor; processor models; double precision; immutable source revision; content hash used to identify an input.
