# Shared-decoder complete-trajectory replay: cost-screen result

This is a final negative result for the frozen development cost screen. Stronger accuracy qualification was not run after the predeclared cost stop.

The complete standalone K16 solve followed by sequential K64 replay took 1.865604336 times the direct solver's time by the registered paired statistic. It was slower in all 42/42 paired repetitions. The final problem, physical mesh, fine initial fit and original temporal prediction remained fixed.

| Case | Direct median ms | Replay median ms | Paired median ratio | Paired range | Accepted guesses |
|---|---:|---:|---:|---|---:|
| 0 | 75.629147 | 137.586797 | 1.820579060 | 1.777607–1.828656 | 5 |
| 1 | 60.087036 | 113.484465 | 1.881514144 | 1.777138–1.890505 | 4 |
| 2 | 61.054719 | 111.282109 | 1.813988002 | 1.791552–1.856827 | 2 |
| 3 | 57.536749 | 105.864069 | 1.849694529 | 1.839938–1.860957 | 3 |
| 4 | 48.414887 | 91.193208 | 1.883831842 | 1.871109–1.938745 | 2 |
| 5 | 53.799018 | 102.082855 | 1.901578944 | 1.882578–1.925379 | 6 |

Each timing includes all initial fits, the coarse trajectory and preview fields where applicable, guard evaluations, fine evolution and requested fine output fields. Ratios pair methods within the same repetition; a ratio of the displayed median times need not equal the paired median ratio.

| Work over all development trajectories | Direct | Replay |
|---|---:|---:|
| Fine corrections | 472 | 531 |
| Fine physics Jacobian evaluations | 772 | 831 |
| Coarse corrections | — | 436 |
| Coarse physics Jacobian evaluations | — | 736 |
| Fine initial-fit iterations | 163 | 163 |
| Coarse initial-fit iterations | — | 122 |

The guard accepted 22 and rejected 278 of 300 coarse guesses. Accepted guesses lowered the original fine objective, but the subsequent fine solves used more corrections in every case. Coarse and fine Jacobians have different widths; their counts are diagnostics, not a substitute for measured total runtime. Initial-fit Jacobian counts are not exposed by the unchanged initializer, and their full runtime is included.

All 900 regular coarse/fine trajectory certificates passed the production stopping rule; maximum fresh stationarity indicator was 0.000989255253. Independent AD/analytic checks, preserved-field promotion, original/disabled fine parity, initial-fit parity, timing witnesses, and both injected coarse-failure controls pass. A failed coarse solve stops its history and disables later guesses. Unavailable field interiors are NaN while boundary walls remain zero.

The registered limits were median paired ratio ≤0.85 and worst case ≤1.10. The observed worst case was 1.901578944, and all 6 case medians exceeded the latter limit. Numerical and timing controls passed; the cost gate failed.

Run settings: L=1024, R=512, M=256, gauss128, BE, dt=0.005, 50 steps, 7 paired repetitions per case. Job 6619863 completed successfully using 179 allocated GPU-seconds on NVIDIA H200, GPU-fd4f0a46-c358-45d3-3aa1-bffac2aeb85e, 143771 MiB; f64/highest and mandatory GPU checks passed. The log is complete and stderr is empty. Setup took 22.461196 seconds and is reported separately from query timing while remaining inside allocated cost.

No tight/tighter baseline agreement, successor quadrature qualification, physical PDE accuracy, sealed-cohort evaluation, three-dimensional extension or repeated correction cycle was run. This does not establish an accuracy gain or reject every possible shared-decoder strategy. The canonical coordinator log owns the combined preparation/test resource total and merge decision.

Source commit: `3472549aaaf00f319b1cc2585e385ebdcd2d98a0`. Result SHA256: `7bd0e996199866c071a051c8f87afa2a4310647aa35fddc3db858ef855393a7a`. Model manifest SHA256: `e6be23e896289e1eee0ae8a14ec6104886114c60dbbdf2cdcea1780dbac2bacc`. Raw data, exact staged inputs, accounting and logs are under `../runs/screen01`; generated audits and retention proofs are beside this file.

## Plain-language glossary

Case: a fixed development initial condition and viscosity. Direct: the original fine decoder solve with predictions from its earlier accepted states. Replay: first solve the complete small-decoder trajectory, then solve every fine step again using only accepted fine history. K: number of latent coordinates. L: physical mesh interval count. R: frozen spatial-bank width. M: weak test-mode count. Gauss rule: spatial integration points and weights. BE: backward Euler time stepping. dt: time-step size. Guard: the objective comparison that accepts or rejects an offered starting point. Correction: one nonlinear solver update. Jacobian: derivatives used by that solver. Stationarity: the stated derivative-based stopping condition; it is not physical-error certification. AD: automatic differentiation. Paired ratio: replay time divided by direct time for the same case and timing block. Paired range: smallest and largest retained paired ratios, without discarding repetitions. Accepted guesses: offers that passed the fixed fine-objective guard. GPU-seconds: allocation duration including setup and checks. SHA256: a content hash binding the report to retained evidence.
