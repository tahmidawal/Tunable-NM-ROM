# Poisson hybrid speed push (generated)

This is the generated closing audit for the bounded NM-ROM warm-start speed search. Its numeric claims come directly from the checksummed run JSONs listed below.

## Outcome

The final solver-aligned one-update candidate fails every attribution gate at both development meshes. It therefore does not advance to the frozen N=1024 confirmation, and the Poisson nonlinear-warm-start search is exhausted under the pre-registered construction and production-control budgets.

The strongest defensible genuine NM-ROM result remains the earlier K8 conditional nonlinear decoder followed by counting CG: one small balanced win against zero-start counting CG at N=1024 and tolerance 1e-6. It is not a production speed win: the eligible dense or spectral/direct controls remain hundreds of times faster.

## One-update mechanism gates

`param1_c64_q0` is a direct source-parameter surrogate, not an NM-ROM. The update arms are conditional nonlinear NM-ROMs because their latent variable is optimized online; they additionally require the known source parameters for initialization.

A candidate passes a mesh only when construction is below 0.6 ms and A-error, mean counting-CG iterations, and same-job total are all strictly lower than the matched direct surrogate. Total is the stored mean of case medians from the common rotated timing block.
The alpha=1 seed-0 cohort is the canonical held-out slice 512:528, after the 512 checkpoint-training cases; the alpha=.5 round uses the wholly new seed shown below.

| round | seed | N | candidate | construct ms | A-error | CG iters | total ms | direct construct ms | direct A-error | direct CG iters | direct total ms | gate | accepted | objective drop |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| alpha=1 field | 0 | 64 | `paramlm1_m16_c64_q0` | 0.418 | 3.421e-02 | 145.12 | 5.349 | 0.220 | 2.659e-02 | 139.19 | 4.927 | fail | 16/16 | 15.9% |
| alpha=1 field | 0 | 64 | `paramlm1_m24_c64_q0` | 0.413 | 2.894e-02 | 142.25 | 5.314 | 0.220 | 2.659e-02 | 139.19 | 4.927 | fail | 16/16 | 16.4% |
| alpha=1 field | 0 | 64 | `paramlm1_m32_c64_q0` | 0.438 | 2.921e-02 | 142.06 | 5.321 | 0.220 | 2.659e-02 | 139.19 | 4.927 | fail | 16/16 | 20.2% |
| alpha=1 field | 0 | 256 | `paramlm1_m16_c64_q0` | 0.425 | 5.079e-02 | 612.69 | 20.762 | 0.209 | 4.560e-02 | 593.50 | 19.756 | fail | 16/16 | 15.9% |
| alpha=1 field | 0 | 256 | `paramlm1_m24_c64_q0` | 0.418 | 4.687e-02 | 599.19 | 20.373 | 0.209 | 4.560e-02 | 593.50 | 19.756 | fail | 16/16 | 16.4% |
| alpha=1 field | 0 | 256 | `paramlm1_m32_c64_q0` | 0.438 | 4.701e-02 | 604.88 | 20.335 | 0.209 | 4.560e-02 | 593.50 | 19.756 | fail | 16/16 | 20.2% |
| alpha=.5 energy | 20260821 | 64 | `paramritz1_m24_c64_q0` | 0.418 | 3.916e-02 | 145.19 | 5.391 | 0.210 | 3.084e-02 | 142.94 | 5.125 | fail | 16/16 | 21.4% |
| alpha=.5 energy | 20260821 | 256 | `paramritz1_m24_c64_q0` | 0.421 | 5.346e-02 | 610.81 | 20.801 | 0.215 | 4.770e-02 | 604.00 | 20.312 | fail | 16/16 | 21.4% |

Both objectives successfully optimize their truncated weak targets, but neither improves the global FOM-relevant energy error or CG work. The final alpha=.5 result therefore isolates a truncated-objective/global-A mismatch rather than a rejected or failed Gauss--Newton step.

## Strongest genuine NM-ROM versus matched counting CG

These rows are the fresh-seed, fully balanced AB/BA audit. A crossover is supported only when both the case-clustered speed interval is above one and the paired learned-minus-zero interval is below zero.

| N | tolerance | construct ms | NM-ROM+FOM ms | zero CG ms | speedup | speedup 95% CI | paired delta 95% CI ms | verdict |
|---:|---:|---:|---:|---:|---:|---|---|---|
| 512 | 1e-06 | 4.236 | 62.006 | 62.269 | 1.004 | [0.984, 1.058] | [-2.691, 1.243] | inconclusive/tie versus counting CG |
| 512 | 1e-08 | 4.236 | 73.868 | 73.751 | 0.998 | [0.973, 1.018] | [-1.324, 2.046] | inconclusive/tie versus counting CG |
| 512 | 1e-10 | 4.236 | 85.432 | 82.404 | 0.965 | [0.941, 0.981] | [1.946, 4.989] | supported slower than counting CG |
| 1024 | 1e-06 | 4.255 | 211.996 | 217.578 | 1.026 | [1.014, 1.077] | [-14.642, -2.418] | supported faster than counting CG |
| 1024 | 1e-08 | 4.255 | 253.531 | 257.030 | 1.014 | [0.993, 1.031] | [-8.976, -0.573] | inconclusive/tie versus counting CG |
| 1024 | 1e-10 | 4.255 | 290.348 | 288.587 | 0.994 | [0.980, 1.010] | [-1.534, 6.457] | inconclusive/tie versus counting CG |

## N=1024 production controls

The controls below were measured in the same fresh-seed rotated job. Eligibility uses the maximum true residual recomputed from each timed invocation. The fastest eligible row at each tolerance is the production comparator.

| tolerance | method | total ms | max true residual | eligible | fastest eligible |
|---:|---|---:|---:|---|---|
| 1e-06 | `dense_dst_direct` | 0.665 | 2.480e-10 | yes | yes |
| 1e-06 | `fft_dst_direct` | 1.095 | 3.960e-11 | yes | no |
| 1e-06 | `spectral_q1024` | 0.803 | 2.480e-10 | yes | no |
| 1e-08 | `dense_dst_direct` | 0.673 | 2.480e-10 | yes | yes |
| 1e-08 | `fft_dst_direct` | 1.096 | 3.960e-11 | yes | no |
| 1e-08 | `spectral_q1024` | 0.814 | 2.480e-10 | yes | no |
| 1e-10 | `dense_dst_direct` | 0.723 | 2.480e-10 | no | no |
| 1e-10 | `fft_dst_direct` | 1.095 | 3.960e-11 | yes | no |
| 1e-10 | `spectral_q1024` | 0.931 | 8.619e-11 | yes | yes |

## Audit trail

| run | seed | job | GPU | commit | source hash |
|---|---:|---:|---|---|---|
| `paramlmgate1.json` | 0 | 2667580 | NVIDIA A100-PCIE-40GB | 9a7b07cd9e06 | 5394425ffe171601 |
| `paramritzg1.json` | 20260821 | 2667673 | NVIDIA A100-PCIE-40GB | 46127cd77320 | bca433e257c8c49a |
| `final1.json` | 20260819 | 2662802 | NVIDIA A100 80GB PCIe | f0c9dfa10a24 | 639c813e2a913768 |
| `pairfinal1.json` | 20260820 | 2664551 | NVIDIA A100 80GB PCIe | 92447ead81a9 | 34fc25856c5e4991 |

All four inputs are complete GPU/f64/highest-precision runs. The two one-update runs persist six raw timing repetitions for each of eight timed cases, retain direct and spectral controls, and pass their same-invocation true-residual gates. Pull checksums are stored beside each run; the cluster job directories were deleted after verified pulls.

## Input files

- `experiments/poisson-hybrid-1024/runs/paramlmgate1/out/paramlmgate1.json`
- `experiments/poisson-hybrid-1024/runs/paramritzg1/out/paramritzg1.json`
- `experiments/poisson-hybrid-1024/runs/final1/out/final1.json`
- `experiments/poisson-hybrid-1024/runs/pairfinal1/out/pairfinal1.json`
