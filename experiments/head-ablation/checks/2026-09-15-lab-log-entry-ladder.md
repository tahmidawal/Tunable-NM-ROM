

## 2026-09-15
### Burgers fixed-weight correction ladder — complete and audited; error monotone in q but the upper rungs stop converging, and no rung beats the efficient FOM on both axes

The coordinator asked how much accuracy a frozen checkpoint can buy at inference time, and at what cost, by solving q extra fixed linear bank directions on top of the neural head: u(z,y) = G(h_theta(z) + C_q y). Every network weight, the bank, the weak objective, the test-mode family, the initializer policy, the trust radius, the iteration budgets and the stopping tolerance are the head-ablation arm (a) contract; q and the time step are the only knobs. Unlike Poisson, the Burgers weak residual is quadratic in the coefficients through the upwind advection term, so y cannot be eliminated analytically and the whole augmented vector is solved by the same Levenberg-Marquardt iteration.

Worktree `worktrees/2026-09-14-head-ablation`, branch `exp/2026-09-14-head-ablation` at `805ba2cd5f8ac2bfa190438f0eaa660019fda00a`. Namespace `/cluster/tufts/paralab/tawal01/headabl_20260914/qlad01`, job `3713867` on `NVIDIA A100-PCIE-40GB`, source `313374fab24ee74e69d0355ebf60f53d49cefc1a`, 256 intervals, 6 opened development cases (the same six arm (a) used; the eight refinement calibration cases belong to a different lane), 3 timed repetitions with GPU burn-in before every block, all repetition arrays retained. Elapsed 3716.4 s.

**Fidelity gates.** Through the corrected-head wrapper at q=0 the local smoke reproduces the consolidated saved Burgers case to 1.403e-14 relative and is bit-identical (0.0e+00) to the incumbent `accuracy_paths.make_rom`.
In the job, `q0_eq` reproduces the head-ablation job's `a_neural_eq` on all 6 cases to a worst relative difference of 7.174e-13, with 0 of 6 output fields bitwise identical across the two jobs. Every recorded error was independently recomputed from the retained output fields by NumPy.

**Direction rule (offline, nested, recorded).** field-metric POD of the decoder-output residual eta - h_theta(z*), with z* the best-found head code under the shared LM rule; nested in q. 1024 seeded snapshots, 4 multistart fits at budget 200, seed 20260915; the head's own best-found relative fit over them is 0.5584% median, 6.9161% worst; available rank 512. The fit is offline and one-time and enters no query timing, but at 1757.6 s it dominated the job's setup cost, and the retained stderr shows a single XLA slow-operation alarm covering nearly the whole stage: the cost is compilation of a doubly vectorised Levenberg-Marquardt while_loop with a forward-mode Jacobian inside, not arithmetic. Flattening that loop would remove most of the setup cost without changing any reported number; worth doing before this rule is reused. Residual energy captured: q=0 0.0000%, q=16 45.2788%, q=64 80.2613%, q=128 94.7010%, q=256 99.8307%, q=512 100.0000%.

The primary metric is the same-grid discrepancy against the converged same-mesh full-order solve, because the refined-reference metric also contains this mesh's discretization error.

| q | solved dim | M | quad | best-found % | worst same-grid % | worst reference % | median iters/step | budget exits | completed | median GPU ms | median host ms |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 16 | 64 | dense | 2.5447 | 2.5629 | 4.5575 | 3.0 | 0 | yes | 336.793 | 339.257 |
| 0 (M control) | 16 | 2112 | dense | 2.5447 | 2.5629 | 4.0687 | 3.0 | 0 | yes | 1361.309 | 1364.314 |
| 0 (dt 0.01) | 16 | 64 | dense | 2.5447 | 2.5629 | 5.5576 | 4.0 | 0 | yes | 262.223 | 264.619 |
| 0 | 16 | 64 | eq | 2.5447 | 2.5629 | 4.5546 | 3.0 | 0 | yes | 48.758 | 51.278 |
| 0 (dt 0.01) | 16 | 64 | eq | 2.5447 | 2.5629 | 5.5638 | 4.0 | 0 | yes | 45.469 | 48.158 |
| 16 | 32 | 128 | dense | 2.4615 | 2.4806 | 4.1065 | 3.0 | 0 | yes | 498.271 | 500.443 |
| 16 | 32 | 128 | eq | 2.4615 | 2.4806 | 4.1197 | 3.0 | 0 | yes | 83.233 | 85.642 |
| 64 | 80 | 320 | dense | 2.1386 | 2.1489 | 4.1013 | 3.0 | 6 | no | 1266.072 | 1268.568 |
| 128 | 144 | 576 | dense | 1.8105 | 1.8116 | 4.0797 | 3.0 | 15 | no | 2538.979 | 2541.763 |
| 256 | 272 | 1088 | dense | 0.9016 | 0.9053 | 4.0361 | 6.0 | 60 | no | 9306.086 | 9308.814 |
| 512 | 528 | 2112 | dense | 0.3918 | 0.6027 | 4.0392 | 2.0 | 9 | no | 9502.081 | 9504.550 |
| FOM fft_tight | - | - | - | - | 0.0000 | 4.0265 | 2.0 | - | - | 89.090 | 91.780 |
| FOM nt1e-2 | - | - | - | - | 3.7127 | 2.4737 | 1.0 | - | - | 15.523 | 17.944 |

**Monotonicity.** Error along q=[0, 16, 64, 128, 256, 512] is monotone decreasing ([2.5629, 2.4806, 2.1489, 1.8116, 0.9053, 0.6027] percent); cost is monotone increasing ([336.793, 498.271, 1266.072, 2538.979, 9306.086, 9502.081] median GPU ms).
The upper rungs are NOT converged solves: q=64 (6 iteration-budget exits, worst normalized gradient 2.60e-02), q=128 (15 iteration-budget exits, worst normalized gradient 3.26e-02), q=256 (60 iteration-budget exits, worst normalized gradient 5.55e-02), q=512 (9 iteration-budget exits, worst normalized gradient 2.59e-01) failed to complete under the shared stopping rule. They are legitimate approximate points on an error/cost curve but keep their true stopping status and must not be read as a converged accuracy curve. The trust radius and per-step budget were deliberately held at the q=0 values so q is the only knob, and that is what binds as the solved dimension grows.
Restricted to the rungs that do converge (q=[0, 16]), the error falls only from 2.5629% to 2.4806% for a 1.479-fold cost increase.
Hyper-reduction, not correction capacity, is the lever that moves cost: at q=0 the empirical-quadrature arm costs 48.758 ms against 336.793 ms dense, a factor 6.908, with worst same-grid errors differing by 0.0000 percentage points. It is not available above q=16 because the nonnegative-least-squares rule at m=4M stops being constructible there.
Hyper-reduction, not correction capacity, is the lever that moves cost: at q=16 the empirical-quadrature arm costs 83.233 ms against 498.271 ms dense, a factor 5.986, with worst same-grid errors differing by 0.0000 percentage points. It is not available above q=16 because the nonnegative-least-squares rule at m=4M stops being constructible there.
Time-step knob at q=0 (dense): doubling the step to 0.01 costs 262.223 ms against 336.793 ms, a factor 0.779 only, because the initial fit and the decode do not scale with step count; worst same-grid is unchanged at 2.5629% but the median rises from 1.6957% to 1.9413%. It buys little and costs accuracy on the typical case.
Time-step knob at q=0 (eq): doubling the step to 0.01 costs 45.469 ms against 48.758 ms, a factor 0.933 only, because the initial fit and the decode do not scale with step count; worst same-grid is unchanged at 2.5629% but the median rises from 1.7013% to 1.9508%. It buys little and costs accuracy on the typical case.

Two effects grow together because the weak objective needs M > K+q, so M=4(K+q) grows with q. The q=0 control at the ladder's largest test count M=2112 costs 1361.309 ms against 336.793 ms at M=64, a factor 4.042, at 2.5629% against 2.5629%, so most of the ladder's cost growth is the growing test count rather than the extra unknowns.

Best rung q=512 removes 1.9602 percentage points for 9165.288 extra median GPU ms, 0.00021 points per millisecond, a cost factor 28.213 over q=0.

Against the same-job full-order `nt1e-2` (3.7127% same-grid, 15.523 ms median GPU, 17.944 ms complete host query): no rung dominates it on both axes.
Against the same-job full-order `fft_tight` (0.0000% same-grid, 89.090 ms median GPU, 91.780 ms complete host query): no rung dominates it on both axes.

**Recorded deviations.** Empirical quadrature is fitted only at q in [0, 16]; above that m=4M grows with q and the bounded nonnegative-least-squares fit is not constructible inside the job budget, so those rungs use the exact dense grid sum, stated per row, with paired eq/dense rows at the same q isolating the quadrature effect. At q=R the reachable set coincides with the head-ablation free-bank arm (d), but the parameterization is redundant by K dimensions and the test count differs, so it is the same reachable set and not the same solver; the two are not expected to agree numerically. The trust radius, budgets and tolerance stay at arm (a) values for every rung, so a fixed trust radius is a tighter restriction on a larger step, which is part of what the ladder measures. Arms above 64 unknowns use a pivoted dense step solve rather than the incumbent unrolled Gauss-Jordan, which is more accurate, not weaker. The stationarity column is not a quality ranking; the completion column is the honest status.

Source-generated report and figure: `experiments/head-ablation/reports/2026-09-14-burgers-correction-ladder.md` with `-cost.png` / `-cost.pdf` beside it, both produced by `reports/generate_correction_ladder.py`. Raw archive Git-tracked as bounded chunks under `experiments/head-ablation/artifacts/qlad01/`: qlad01 sha256 9f0de364a4b0f25b9d702e3ef0d570229a427de208dff9941d2fdeacc54bb097 (6 chunks). The exact remote attempt directory was removed after checksum collection and the namespace is empty. Not pushed, per the coordinator's standing instruction; commits are local only.

**Open.** Other meshes, other checkpoints, more than one training seed, the sealed final cohort, and whether a direction rule fitted to trajectory error rather than reconstruction residual would move the curve. No earlier numerical result is retracted and no worktree was merged.
