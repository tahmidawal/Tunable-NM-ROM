### T-1 Provenance (one job per N; the GPU differs across N)

| N | job | node | GPU | backend | commit | jax | checkpoint | trained in-job | complete | secs |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 3038943 | pax051 | NVIDIA A100-PCIE-40GB | gpu | 922ea45312 | 0.10.2 | sep_burgers_N64_K16_R64.pkl | False | True | 108 |
| 256 | 3038921 | pax007 | NVIDIA A100 80GB PCIe | gpu | 922ea45312 | 0.10.2 | sep_burgers_N256_K16_R64.pkl | False | True | 223 |
| 1024 | 3038923 | pax106 | NVIDIA A100 80GB PCIe | gpu | 922ea45312 | 0.10.2 | sep_burgers_N1024_K16_R64.pkl | False | True | 966 |

### T-2 Positivity audit: truth (training + test states, interior points) and decoded states

| N | train traj / states | truth min u (train) | truth frac<0 (train) | truth min u (test) | truth frac<0 (test) | assert ok | decoded train states: min u / frac points u<=0 / frac all-positive | full-arm rollout: min u / frac states touching u<=0 / frac points u<=0 | tensor-arm rollout: min u / frac states touching u<=0 |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 576 / 29376 | -6.83e-31 | 2.5e-05 | 2.83e-33 | 0.0e+00 | True | -1.02e-01 / 16.682% / 2.0% | -1.10e-01 / 92.6% / 17.21% | -1.10e-01 / 92.6% |
| 256 | 576 / 29376 | -4.76e-31 | 3.0e-05 | 2.11e-34 | 0.0e+00 | True | -1.28e-01 / 16.982% / 0.6% | -8.26e-02 / 98.8% / 18.83% | -8.26e-02 / 98.8% |
| 1024 | 104 / 5304 | -1.13e-35 | 3.5e-08 | 1.11e-34 | 0.0e+00 | True | -6.14e-02 / 15.296% / 3.2% | -1.92e-01 / 100.0% / 21.88% | -1.92e-01 / 100.0% |

### T-3 Gates (asserted unless marked recorded)

| N | bank==meshfree | gate 0 | L | A | FOMR | TB | TA (states) | T0 (all-positive states) | TQ r rel med / max (recorded) | TQ J rel max (recorded) | TQ states with u<=0 | STEP | ROLL | IC gram vs full dz | R-lite recon mean | test FOM res | train FOM res |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 7.3e-16 | 4.0e-16 | 0.0e+00 | 3.0e-13 | 8.2e-16 | 0.0e+00 | 1.7e-14 (8192) | 1.7e-15 (161) | 3.3e-06 / 3.5e-04 | 8.6e-04 | 32/32 | 0e+00 | 0e+00 | 6.7e-08 | 3.98e-02 | 9.9e-13 | 1.0e-12 |
| 256 | 2.9e-16 | 7.2e-16 | 2.1e-16 | 1.1e-13 | 4.2e-16 | 1.6e-15 | 1.9e-14 (8192) | 1.2e-15 (52) | 7.5e-07 / 6.7e-05 | 1.7e-04 | 32/32 | 0e+00 | 0e+00 | 2.1e-13 | 4.10e-02 | 9.9e-13 | 1.0e-12 |
| 1024 | 6.9e-16 | 3.5e-15 | 0.0e+00 | 1.9e-13 | 6.0e-16 | 1.4e-15 | 8.8e-15 (2048) | 9.0e-16 (66) | 3.7e-07 / 5.2e-05 | 1.2e-04 | 32/32 | 0e+00 | 0e+00 | 1.6e-12 | 1.53e-02 | 1.0e-12 | 9.9e-13 |

### T-4 Accuracy per arm per N (mean rel-L2 over 8 test trajectories x 51 states; fused e2e output of the last timed rep) and tensor-vs-full per-trajectory max |diff|

| N | full err | ex err | tensor err | ex_learned err | tensor-full max abs diff | tensor/full err ratio | tensor/ex err ratio | tensor-full latent dev max | stop hist identical tensor/full (per traj) | attempts identical |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 2.308651e-02 | 2.319010e-02 | 2.309414e-02 | 2.730832e-02 | 7.95e-05 | 1.00033 | 0.9959 | 1.3e-04 | True (True) | True |
| 256 | 2.451134e-02 | 2.449564e-02 | 2.451214e-02 | 2.792914e-02 | 7.30e-06 | 1.00003 | 1.0007 | 4.3e-05 | True (True) | True |
| 1024 | 6.381029e-02 | 6.390634e-02 | 6.381390e-02 | 6.403727e-02 | 1.99e-05 | 1.00006 | 0.9986 | 3.4e-04 | True (True) | True |

### T-5 Stop-reason histograms and LM counts (8 traj x 50 steps)

| N | arm | stop reasons | LM attempts / traj | accepted Jacobians / traj | IC rel err |
|---|---|---|---|---|---|
| 64 | full | {'stalled': 400} | 112.6 | 156.9 | 7.545e-02 |
| 64 | ex | {'stalled': 400} | 112.6 | 156.9 | 7.545e-02 |
| 64 | tensor | {'stalled': 400} | 112.6 | 156.9 | 7.545e-02 |
| 64 | ex_learned | {'stalled': 400} | 114.4 | 157.6 | 7.545e-02 |
| 256 | full | {'stalled': 400} | 113.5 | 157.4 | 7.875e-02 |
| 256 | ex | {'stalled': 400} | 113.5 | 157.4 | 7.875e-02 |
| 256 | tensor | {'stalled': 400} | 113.5 | 157.4 | 7.875e-02 |
| 256 | ex_learned | {'stalled': 400} | 113.1 | 157.1 | 7.875e-02 |
| 1024 | full | {'stalled': 400} | 108.9 | 154.6 | 1.484e-01 |
| 1024 | ex | {'stalled': 400} | 108.9 | 154.6 | 1.484e-01 |
| 1024 | tensor | {'stalled': 400} | 108.9 | 154.6 | 1.484e-01 |
| 1024 | ex_learned | {'stalled': 400} | 109.6 | 155.1 | 1.484e-01 |

### T-6 Cost split per arm per N (ms per trajectory, median over 8 traj x 5 timed reps; arms interleaved AB/BA; ic / solve / dec are separately blocked phases, e2e is one fused jit)

| N | GPU | arm | ic ms | latent solve ms | decode ms | split sum ms | fused e2e ms | solve ratio vs ex | solve ratio vs full | e2e ratio vs ex | e2e ratio vs full |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 | NVIDIA A100-PCIE-40GB | full | 1.98 | 32.59 | 0.40 | 35.23 | 34.81 | 1.131 | n/a | 1.132 | n/a |
| 64 | NVIDIA A100-PCIE-40GB | ex | 1.98 | 28.83 | 0.40 | 31.54 | 30.75 | n/a | 0.884 | n/a | 0.883 |
| 64 | NVIDIA A100-PCIE-40GB | tensor | 1.99 | 27.54 | 0.40 | 30.30 | 29.60 | 0.955 | 0.845 | 0.963 | 0.850 |
| 64 | NVIDIA A100-PCIE-40GB | ex_learned | 2.00 | 27.55 | 0.40 | 30.45 | 29.83 | 0.956 | 0.845 | 0.970 | 0.857 |
| 256 | NVIDIA A100 80GB PCIe | full | 2.16 | 74.08 | 0.55 | 76.82 | 76.16 | 2.547 | n/a | 2.441 | n/a |
| 256 | NVIDIA A100 80GB PCIe | ex | 2.18 | 29.08 | 0.55 | 31.76 | 31.20 | n/a | 0.393 | n/a | 0.410 |
| 256 | NVIDIA A100 80GB PCIe | tensor | 2.16 | 27.70 | 0.54 | 30.34 | 29.84 | 0.952 | 0.374 | 0.956 | 0.392 |
| 256 | NVIDIA A100 80GB PCIe | ex_learned | 2.19 | 27.60 | 0.53 | 30.36 | 30.20 | 0.949 | 0.373 | 0.968 | 0.397 |
| 1024 | NVIDIA A100 80GB PCIe | full | 4.77 | 670.03 | 1.29 | 679.63 | 681.89 | 22.814 | n/a | 19.747 | n/a |
| 1024 | NVIDIA A100 80GB PCIe | ex | 5.07 | 29.37 | 1.18 | 36.64 | 34.53 | n/a | 0.044 | n/a | 0.051 |
| 1024 | NVIDIA A100 80GB PCIe | tensor | 4.79 | 27.13 | 1.15 | 33.23 | 32.56 | 0.924 | 0.040 | 0.943 | 0.048 |
| 1024 | NVIDIA A100 80GB PCIe | ex_learned | 4.86 | 27.46 | 1.15 | 35.20 | 33.37 | 0.935 | 0.041 | 0.966 | 0.049 |

### T-7 FOM cost per N (standardised tol-Newton ladder, same GPU as the ROM arms; matched = cheapest rung at least as accurate as the tensor arm; closest = rung with error closest to the tensor arm's in log; tightest = most accurate rung)

| N | GPU | tensor err / e2e ms | matched rung (nt, lt) err / ms | paired tensor vs matched: ROM ms / FOM ms / speedup | closest rung err / ms | tightest rung (nt, lt) err / ms | paired ex vs matched speedup | paired full vs matched speedup |
|---|---|---|---|---|---|---|---|---|
| 64 | NVIDIA A100-PCIE-40GB | 2.309e-02 / 29.60 | (3e-03, 2e-03) 3.75e-04 / 12.36 | 29.57 / 12.37 / 0.42x | 3.46e-02 / 13.34 | (1e-04, 5e-05) 1.35e-04 / 16.29 | 0.41x | 0.36x |
| 256 | NVIDIA A100 80GB PCIe | 2.451e-02 / 29.84 | (3e-03, 2e-03) 3.54e-04 / 17.26 | 29.53 / 17.24 / 0.58x | 3.05e-02 / 18.91 | (1e-04, 5e-05) 1.33e-04 / 23.61 | 0.56x | 0.23x |
| 1024 | NVIDIA A100 80GB PCIe | 6.381e-02 / 32.56 | (1e-02, 5e-03) 2.97e-02 / 119.99 | 34.11 / 117.69 / 3.45x | 2.97e-02 / 119.99 | (1e-04, 5e-05) 1.39e-04 / 230.72 | 3.31x | 0.18x |

### T-8 Full FOM ladder per N (err = mean rel-L2 vs the 8-Newton truth; ms median over 8 traj x 5 reps)

| N | newton_tol | lin_tol | err | ms | Newton iters / traj |
|---|---|---|---|---|---|
| 64 | 3e-01 | 1e-02 | 8.885e-01 | 2.41 | 0 |
| 64 | 3e-01 | 1e-01 | 8.885e-01 | 2.41 | 0 |
| 64 | 1e-01 | 5e-03 | 8.586e-01 | 2.38 | 0 |
| 64 | 1e-01 | 5e-02 | 8.586e-01 | 2.37 | 0 |
| 64 | 3e-02 | 2e-03 | 2.799e-01 | 5.80 | 12 |
| 64 | 3e-02 | 1e-02 | 2.825e-01 | 4.93 | 12 |
| 64 | 1e-02 | 5e-04 | 3.456e-02 | 13.34 | 40 |
| 64 | 1e-02 | 5e-03 | 3.460e-02 | 10.96 | 40 |
| 64 | 3e-03 | 2e-04 | 3.538e-04 | 14.80 | 50 |
| 64 | 3e-03 | 2e-03 | 3.746e-04 | 12.36 | 50 |
| 64 | 1e-03 | 5e-05 | 3.140e-04 | 15.34 | 50 |
| 64 | 1e-03 | 5e-04 | 3.172e-04 | 14.41 | 50 |
| 64 | 1e-04 | 5e-06 | 1.350e-04 | 18.56 | 54 |
| 64 | 1e-04 | 5e-05 | 1.349e-04 | 16.29 | 54 |
| 256 | 3e-01 | 1e-02 | 8.241e-01 | 2.31 | 0 |
| 256 | 3e-01 | 1e-01 | 8.248e-01 | 2.27 | 0 |
| 256 | 1e-01 | 5e-03 | 8.204e-01 | 2.32 | 0 |
| 256 | 1e-01 | 5e-02 | 8.099e-01 | 2.32 | 0 |
| 256 | 3e-02 | 2e-03 | 2.727e-01 | 7.51 | 12 |
| 256 | 3e-02 | 1e-02 | 2.727e-01 | 6.21 | 12 |
| 256 | 1e-02 | 5e-04 | 3.050e-02 | 18.91 | 41 |
| 256 | 1e-02 | 5e-03 | 3.058e-02 | 14.80 | 41 |
| 256 | 3e-03 | 2e-04 | 3.257e-04 | 21.09 | 50 |
| 256 | 3e-03 | 2e-03 | 3.538e-04 | 17.26 | 50 |
| 256 | 1e-03 | 5e-05 | 3.013e-04 | 21.89 | 50 |
| 256 | 1e-03 | 5e-04 | 3.291e-04 | 19.93 | 50 |
| 256 | 1e-04 | 5e-06 | 1.327e-04 | 27.23 | 54 |
| 256 | 1e-04 | 5e-05 | 1.325e-04 | 23.61 | 54 |
| 1024 | 3e-01 | 1e-02 | 8.105e-01 | 6.54 | 1 |
| 1024 | 3e-01 | 1e-01 | 8.105e-01 | 6.50 | 1 |
| 1024 | 1e-01 | 5e-03 | 7.899e-01 | 6.63 | 1 |
| 1024 | 1e-01 | 5e-02 | 7.911e-01 | 6.48 | 1 |
| 1024 | 3e-02 | 2e-03 | 2.751e-01 | 54.16 | 12 |
| 1024 | 3e-02 | 1e-02 | 2.752e-01 | 40.63 | 12 |
| 1024 | 1e-02 | 5e-04 | 2.946e-02 | 172.14 | 41 |
| 1024 | 1e-02 | 5e-03 | 2.975e-02 | 119.99 | 41 |
| 1024 | 3e-03 | 2e-04 | 3.363e-04 | 199.22 | 50 |
| 1024 | 3e-03 | 2e-03 | 3.486e-04 | 171.32 | 50 |
| 1024 | 1e-03 | 5e-05 | 3.332e-04 | 209.08 | 50 |
| 1024 | 1e-03 | 5e-04 | 3.399e-04 | 186.33 | 50 |
| 1024 | 1e-04 | 5e-06 | 1.389e-04 | 279.12 | 54 |
| 1024 | 1e-04 | 5e-05 | 1.386e-04 | 230.72 | 54 |

### T-9 Per-trajectory tensor vs full vs ex

| N | traj | nu | full err | tensor err | abs diff | latent dev | reasons equal | ex err | ex_learned err |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 0 | 0.0420 | 3.865360e-02 | 3.873314e-02 | 7.95e-05 | 1.30e-04 | True | 3.863032e-02 | 4.304280e-02 |
| 64 | 1 | 0.0598 | 2.512413e-02 | 2.511381e-02 | 1.03e-05 | 4.15e-05 | True | 2.600508e-02 | 3.544234e-02 |
| 64 | 2 | 0.0410 | 3.173297e-02 | 3.170937e-02 | 2.36e-05 | 4.93e-05 | True | 3.174210e-02 | 4.115690e-02 |
| 64 | 3 | 0.0827 | 2.935092e-02 | 2.936066e-02 | 9.74e-06 | 1.30e-05 | True | 2.931515e-02 | 2.932627e-02 |
| 64 | 4 | 0.0110 | 1.126162e-02 | 1.126231e-02 | 6.89e-07 | 9.58e-06 | True | 1.125836e-02 | 1.211858e-02 |
| 64 | 5 | 0.0338 | 2.245688e-02 | 2.246114e-02 | 4.27e-06 | 5.97e-05 | True | 2.234367e-02 | 2.992400e-02 |
| 64 | 6 | 0.0288 | 1.619106e-02 | 1.619180e-02 | 7.43e-07 | 8.83e-06 | True | 1.631722e-02 | 1.725647e-02 |
| 64 | 7 | 0.0115 | 9.920909e-03 | 9.920909e-03 | 5.65e-10 | 2.26e-08 | True | 9.908921e-03 | 1.019920e-02 |
| 256 | 0 | 0.0420 | 3.566773e-02 | 3.567503e-02 | 7.30e-06 | 4.28e-05 | True | 3.562597e-02 | 3.752441e-02 |
| 256 | 1 | 0.0598 | 2.931112e-02 | 2.930621e-02 | 4.91e-06 | 1.17e-05 | True | 2.920344e-02 | 4.536309e-02 |
| 256 | 2 | 0.0410 | 3.645035e-02 | 3.644848e-02 | 1.87e-06 | 2.19e-05 | True | 3.629407e-02 | 4.204298e-02 |
| 256 | 3 | 0.0827 | 3.047653e-02 | 3.048020e-02 | 3.67e-06 | 3.09e-06 | True | 3.046671e-02 | 3.015629e-02 |
| 256 | 4 | 0.0110 | 1.370183e-02 | 1.370194e-02 | 1.16e-07 | 3.97e-06 | True | 1.369295e-02 | 1.404276e-02 |
| 256 | 5 | 0.0338 | 2.140601e-02 | 2.140730e-02 | 1.30e-06 | 1.02e-05 | True | 2.150823e-02 | 2.311774e-02 |
| 256 | 6 | 0.0288 | 1.798530e-02 | 1.798617e-02 | 8.66e-07 | 3.64e-06 | True | 1.806181e-02 | 1.967567e-02 |
| 256 | 7 | 0.0115 | 1.109182e-02 | 1.109182e-02 | 7.70e-12 | 1.62e-09 | True | 1.111197e-02 | 1.151021e-02 |
| 1024 | 0 | 0.0420 | 1.725098e-01 | 1.725297e-01 | 1.99e-05 | 3.37e-04 | True | 1.730410e-01 | 1.730034e-01 |
| 1024 | 1 | 0.0598 | 3.519681e-02 | 3.519724e-02 | 4.34e-07 | 3.15e-06 | True | 3.533795e-02 | 3.536948e-02 |
| 1024 | 2 | 0.0410 | 1.290261e-01 | 1.290321e-01 | 5.96e-06 | 3.54e-05 | True | 1.287539e-01 | 1.293086e-01 |
| 1024 | 3 | 0.0827 | 3.961182e-02 | 3.961151e-02 | 3.09e-07 | 5.22e-06 | True | 3.964823e-02 | 4.003803e-02 |
| 1024 | 4 | 0.0110 | 2.310262e-02 | 2.310277e-02 | 1.47e-07 | 6.07e-06 | True | 2.313458e-02 | 2.336492e-02 |
| 1024 | 5 | 0.0338 | 6.017277e-02 | 6.017476e-02 | 1.99e-06 | 3.20e-05 | True | 6.044264e-02 | 5.915221e-02 |
| 1024 | 6 | 0.0288 | 3.473080e-02 | 3.473110e-02 | 2.97e-07 | 6.05e-06 | True | 3.470563e-02 | 3.575964e-02 |
| 1024 | 7 | 0.0115 | 1.613159e-02 | 1.613209e-02 | 5.02e-07 | 1.27e-05 | True | 1.618680e-02 | 1.630189e-02 |

### T-10 Cross-N cost ratios (DIFFERENT jobs and GPUs per N -- ratios, not exponents; the GPU is named per row)

| N | GPU | tensor solve ms | ex solve ms | full solve ms | tensor/ex | tensor/full | tensor ic ms | tensor dec ms | tensor e2e ms |
|---|---|---|---|---|---|---|---|---|---|
| 64 | NVIDIA A100-PCIE-40GB | 27.54 | 28.83 | 32.59 | 0.955 | 0.845 | 1.99 | 0.40 | 29.60 |
| 256 | NVIDIA A100 80GB PCIe | 27.70 | 29.08 | 74.08 | 0.952 | 0.374 | 2.16 | 0.54 | 29.84 |
| 1024 | NVIDIA A100 80GB PCIe | 27.13 | 29.37 | 670.03 | 0.924 | 0.040 | 4.79 | 1.15 | 32.56 |

### T-11 Tensor build and NNLS fit (offline costs)

| N | Q shape | Q MiB | build s (one chunking) | T asymmetry rel | TB | NNLS m | NNLS rel fit | NNLS s |
|---|---|---|---|---|---|---|---|---|
| 64 | [64, 64, 64] | 2.0 | 1.17 | 1.35 | 0.0e+00 | 256 | 4.98e-03 | 11 |
| 256 | [64, 64, 64] | 2.0 | 1.77 | 1.76 | 1.6e-15 | 256 | 4.95e-03 | 23 |
| 1024 | [64, 64, 64] | 2.0 | 1.86 | 1.96 | 1.4e-15 | 256 | 5.99e-03 | 26 |
