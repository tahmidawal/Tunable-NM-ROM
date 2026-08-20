# Hybrid warm starts at N=2048

This final report extends the audited Poisson-2D and Burgers-2D hybrid studies to N=2048. It separates genuine learned NM-ROM results from classical solver improvements; every number below is generated from checksummed, independently audited run JSONs.

## Technical summary

The genuine Poisson K=8 NM-ROM warm start has one supported crossover: at tolerance 1e-06, it takes 1472.986 ms versus 1547.144 ms for counting CG, 1.050x with ratio interval [1.018, 1.067]. The two tighter tolerances are unsupported.

The genuine Burgers FiLM NM-ROM wins none of its three same-job comparisons against either optimized cubic history or the classical correction. The practical Burgers result is instead nonlearned: at tolerance 1e-06, the charged residual-plus-Helmholtz correction reduces 532.817 ms to 343.673 ms, 1.550x, with paired-saving interval [54.945, 195.984] ms. Its tighter rows are unsupported.

Operationally, Poisson's learned crossover is not competitive with the structure-aware solver: the eligible q=1024 spectral warm start is hundreds of times faster than zero-start CG at all three tolerances.

## Poisson: a learned crossover only at loose tolerance

The learned arm is the fixed-N=64 K=8 trust-region NM-ROM decode followed by the same true-residual counting CG as the zero-start baseline. Each row uses eight fresh cases, 12 repetitions per method and case, exact AB/BA position balance, a fresh burn before both orders, and whole-case bootstrap inference.

| FOM tolerance | K8 NM-ROM+CG ms | zero-start CG ms | speedup [95% CI] | paired saving ms [95% CI] | case signs K8/zero | verdict |
|---:|---:|---:|---|---|---:|---|
| 1e-06 | 1472.986 | 1547.144 | 1.050x [1.018, 1.067] | 53.330 [33.637, 84.796] | 8/0 | supported faster |
| 1e-08 | 1757.347 | 1802.813 | 1.026x [0.993, 1.052] | 35.801 [-6.507, 90.458] | 6/2 | unsupported / inconclusive |
| 1e-10 | 2015.330 | 2008.121 | 0.996x [0.985, 1.003] | -7.876 [-33.079, 8.625] | 4/4 | unsupported / inconclusive |

All 576 learned/zero timing records meet their true-residual and boundary gates. The diagnostic outlier count is 0; no sample was removed.

## Poisson: the production solver remains classical

The separate balanced five-method block compares each method at every clock position and uses the same cases and GPU within the block. The fastest eligible method at every tolerance is the partial q=1024 sine-mode solve followed, when needed, by counting CG.

| FOM tolerance | spectral q=1024 ms | same-block zero CG ms | speedup vs zero | mean finishing CG iterations | eligible |
|---:|---:|---:|---:|---:|:---:|
| 1e-06 | 1.921 | 1533.607 | 798.3x | 0.750 | yes |
| 1e-08 | 2.470 | 1787.715 | 723.8x | 2.125 | yes |
| 1e-10 | 3.734 | 1991.052 | 533.3x | 5.000 | yes |

Dense and FFT DST direct solves also pass the 1e-6 and 1e-8 gates, but their measured true residuals miss 1e-10. The q=1024 arm remains eligible there. This rectangular, separable Poisson family is therefore a strong case for exploiting known operator structure rather than adding a learned warm start.

## Burgers: the genuine FiLM NM-ROM remains slower

The learned sensitivity uses a genuine K=8 weak FiLM NM-ROM with M=64, m=256, at most two latent Jacobians per step, fixed-N=64 decoding and prolongation, and a charged exact-residual guard. The table contains only comparisons made inside its own H200 job and seed; absolute times are not mixed with the separate classical final job. A positive control-minus-FiLM saving would favor FiLM.

| FOM tolerance | control | control ms | FiLM ms | control/FiLM | paired control-minus-FiLM ms [95% CI] | FiLM supported faster |
|---:|---|---:|---:|---:|---|:---:|
| 1e-06 | cubic | 756.309 | 862.278 | 0.877x | -100.433 [-111.816, -91.028] | no |
| 1e-06 | dynamic | 713.903 | 862.420 | 0.828x | -190.550 [-267.018, -70.379] | no |
| 1e-08 | cubic | 1418.694 | 1497.729 | 0.947x | -104.314 [-117.931, -49.726] | no |
| 1e-08 | dynamic | 1331.101 | 1497.767 | 0.889x | -211.999 [-552.563, 30.375] | no |
| 1e-10 | cubic | 2213.739 | 2333.497 | 0.949x | -119.626 [-135.276, -104.982] | no |
| 1e-10 | dynamic | 2012.698 | 2333.343 | 0.863x | -225.143 [-433.529, -157.890] | no |

The residual guard accepts a median of 4 of 50 steps while the arm performs 100 reduced Jacobians per trajectory. That construction does not repay itself at N=2048.

## Burgers: a classical loose-tolerance speedup

The practical arm is not learned and is not an NM-ROM. Before every fine FOM step it evaluates one target-grid exact-upwind residual and applies one exact Helmholtz inverse to the live cubic prediction. Its measured wall time includes all 50 residual evaluations and 50 inverses; the reported finishing Newton/BiCGStab counters exclude those extra operations.

| FOM tolerance | cubic ms | corrected ms | speedup | paired saving ms [95% trajectory CI] | finish Newton cubic/corrected | finish BiCG cubic/corrected | verdict |
|---:|---:|---:|---:|---|---:|---:|---|
| 1e-06 | 532.817 | 343.673 | 1.550x | 178.175 [54.945, 195.984] | 57.5/19.0 | 63.0/24.5 | supported faster |
| 1e-08 | 923.747 | 927.397 | 0.996x | 5.293 [-27.191, 67.302] | 63.0/54.5 | 131.0/113.0 | unsupported / inconclusive |
| 1e-10 | 1191.344 | 1209.532 | 0.985x | 45.404 [-20.894, 266.999] | 70.0/58.0 | 177.5/163.0 | unsupported / inconclusive |

All 288 timed records and 144 burns pass the timing-grid, order, solver-health, accuracy, and same-invocation checks. The one supported row is 1e-6; the point estimates at 1e-8 and 1e-10 do not establish a benefit.

## Reference and failure audit

The Burgers final timing was allowed only after two separately compiled exact-Helmholtz reference routes passed on all four fresh trajectories. Their worst actual outer residual is 9.538e-13; maximum step and trajectory field disagreements are 4.683e-14 and 1.793e-14. Both routes have zero flags and breakdowns.

Two earlier Burgers N=2048 timing attempts are excluded pre-science because the inherited public JAX reference froze on one development trajectory. A batch-shape diagnostic then reproduced nonfinite public updates and did not license a replacement. The final reference design was frozen prospectively and evaluated on a new seed before any final timing. An earlier diagnostic also exposed a shell `producer && checker` false-success pattern; that diagnostic is excluded, and the authoritative final uses separate fail-closed producer and checker commands.

## Scope and limitations

Poisson used seed 20260826 on an NVIDIA A100 80GB PCIe (job 2670159); the learned Burgers sensitivity used seed 20260828 on an NVIDIA H200 (job 2672536); the classical Burgers final used seed 20260830 on an NVIDIA H200 (job 2680178). No wall-clock number is compared across jobs or GPU types.

All timings are warmed, compiled steady-state online costs. Checkpoint loading, compilation, data/reference generation, and first-query latency are excluded. Poisson inference uses eight timed cases; Burgers uses four trajectories, so its trajectory-cluster intervals—especially at tighter tolerances—remain sample-limited. These results establish N=2048 behavior for the tested families, not a universal ranking on irregular domains or different operators.

## Recommended operating policy

1. On this separable Poisson rectangle, use the q=1024 spectral warm start rather than the learned hybrid. Retain the K8 result as evidence that learned construction can cross counting CG at large N, not as the production winner.
2. For this Burgers family at tolerance 1e-6, use the charged residual-plus-Helmholtz correction. Keep cubic history at 1e-8 and 1e-10 because the N=2048 correction rows are unsupported.
3. Stop tuning the current Burgers FiLM warm start. A future learned route needs a materially better transported trajectory manifold and much lower online construction, and should first be tested on nonseparable settings where exact structured solvers are unavailable.

## Generated inputs

- Poisson final JSON: `worktrees/2026-08-20-poisson-hybrid-2048/experiments/poisson-hybrid-1024/runs/n2048final1/out/n2048final1.json`
- Poisson independent audit: `worktrees/2026-08-20-poisson-hybrid-2048/experiments/poisson-hybrid-1024/runs/n2048final1/INDEPENDENT-AUDIT.json`
- Burgers classical final JSON: `worktrees/2026-08-20-burgers-hybrid-2048/experiments/burgers-hybrid-2048/runs/final3/out/final.json`
- Burgers classical independent audit: `worktrees/2026-08-20-burgers-hybrid-2048/experiments/burgers-hybrid-2048/runs/final3/INDEPENDENT-AUDIT.json`
- Burgers learned JSON: `worktrees/2026-08-20-burgers-hybrid-2048/experiments/burgers-hybrid-2048/runs/learned1/out/learned.json`
- Burgers learned audit: `worktrees/2026-08-20-burgers-hybrid-2048/experiments/burgers-hybrid-2048/runs/learned1/LEARNED-AUDIT.json`
