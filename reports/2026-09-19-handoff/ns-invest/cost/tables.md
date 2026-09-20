## A. Fitted constants (A100 80GB PCIe `pax105`, f64, from the six neural rungs)

- tensor (M,R,R) = (2176,512,512) f64 = **4.56 GB**
- `a` = one streaming pass over the tensor (`Q @ cm`, `ns2d_rom.py:214`): **2.59 ms** -> effective bandwidth 1.76 TB/s (A100-80GB-PCIe spec 1.94 TB/s)
- `c` = one tangent column of `jax.jacfwd(fun)` (`ns2d_rom.py:226`, a batched (R,R)@(R,T) matmul over M): **0.0814 ms/column** -> 14.0 TFLOP/s f64 (A100 f64 tensor-core peak 19.5)
- crossover: jacfwd is compute-bound once T > a/c = 32 columns; every rung (T >= 32) is on the compute side

## B. Check against the measured neural rungs (model uses the MEASURED iteration count of each rung)

| rung | T=K+q | LM it/step | residual passes/step | jacfwd passes/step | model ms/step | measured ms/step | model/measured | measured s/query |
|---|---|---|---|---|---|---|---|---|
| neural_q0 | 32 | 2.08 | 7.2 | 3.1 | 26.6 | 26.8 | 0.99 | 13.4 |
| neural_q32 | 64 | 2.05 | 7.1 | 3.0 | 34.3 | 34.0 | 1.01 | 17.0 |
| neural_q64 | 96 | 2.08 | 7.2 | 3.1 | 42.7 | 42.6 | 1.00 | 21.3 |
| neural_q128 | 160 | 2.04 | 7.1 | 3.0 | 58.1 | 58.0 | 1.00 | 29.0 |
| neural_q256 | 288 | 2.12 | 7.2 | 3.1 | 92.1 | 92.5 | 1.00 | 46.3 |
| neural_q512 | 544 | 1.08 | 5.2 | 2.1 | 105.5 | 105.2 | 1.00 | 52.6 |

Share of the per-step time in the jacfwd term (model): q=0: 30 %, q=32: 46 %, q=64: 56 %, q=128: 68 %, q=256: 80 %, q=512: 87 %

## C. Out-of-sample check: POD-LSPG subjects (tensor (M,k,k); jacfwd = batched (k,k)@(k,k) -> 2Mk^3 FLOP)

| subject | k | LM it/step | tensor GB | pass ms | jacfwd ms | model ms/step | measured ms/step | model/measured |
|---|---|---|---|---|---|---|---|---|
| pod_k32 | 32 | 1.00 | 0.018 | 0.010 | 0.01 | 0.07 | 0.42 | 0.17 |
| pod_k64 | 64 | 1.00 | 0.071 | 0.041 | 0.08 | 0.37 | 0.86 | 0.42 |
| pod_k96 | 96 | 1.00 | 0.160 | 0.091 | 0.27 | 1.01 | 1.67 | 0.60 |
| pod_k160 | 160 | 1.00 | 0.446 | 0.253 | 1.27 | 3.81 | 4.80 | 0.80 |
| pod_k288 | 288 | 1.00 | 1.444 | 0.820 | 7.42 | 18.96 | 19.93 | 0.95 |
| pod_k512 | 512 | 1.00 | 4.563 | 2.593 | 41.69 | 96.44 | 86.63 | 1.11 |
| pod_k544 | 544 | 1.00 | 5.152 | 2.927 | 50.01 | 114.76 | 107.81 | 1.06 |

The k=32 row is pure launch/loop latency: measured minus model = **0.35 ms/step** (~10 us per kernel over the ~35 kernels a step launches). Used below as the floor `lat`.

## D. The full-order solve in the same job

- FOM ntol 1e-3 (1 Newton/step, BiCGStab with exact FFT Helmholtz preconditioner): **421 ms / query = 0.84 ms/step**, worst evolved error 4.1e-05
- converged FOM ntol 1e-11 (2 Newton/step): 1178 ms / query
- one FOM residual = 3 FFT2 of 256^2 + 9-point stencils: ~16 MFLOP and ~3 MB of traffic; one ROM residual = 1.14 GFLOP and 4.56 GB -> the ROM residual is ~73x the FLOPs and ~1451x the bytes of the FOM residual.

## E. Per-fix estimates (A100 constants above; per query = 500 steps; `lat` floor added to every design)

| design (q=0, K=32, R=512) | per residual/pass ms | per Jacobian ms | ms/step | s/query | vs FOM 0.42 s |
|---|---|---|---|---|---|
| as measured (q=0, M=2176, jacfwd, 2.08 it/step) | 2.593 | 2.606 | 27.07 | 13.54 | 32.13x |
| F1 analytic Jacobian only (reuse Q@cm; 1 pass gives r and J) | 2.593 | 2.598 | 27.05 | 13.53 | 32.11x |
| F1 + cap 1 LM it/step, keep the 2-residual extrapolation test | 2.593 | 2.598 | 18.61 | 9.31 | 22.09x |
| F1 + cap 1 it + always take the extrapolated start (no test) | 2.593 | 2.598 | 8.24 | 4.12 | 9.78x |
| F2 M=4(K+q)=128 test modes, everything else as written | 0.153 | 0.153 | 2.07 | 1.03 | 2.46x |
| F2 + F1 (M=128, analytic J) | 0.153 | 0.153 | 2.07 | 1.03 | 2.45x |
| F2 + F1 + cap 1 it, no extrapolation test | 0.153 | 0.153 | 0.91 | 0.46 | 1.08x |
| F6 f32 tensor only (M=2176, jacfwd, 2.08 it) -- stopping rule must change | 1.296 | 1.303 | 13.79 | 6.90 | 16.37x |
| F4 EQ m=2048 nodes at M=2176, analytic J, 2.08 it/step | 0.106 | 0.135 | 1.68 | 0.84 | 1.99x |
| F4 EQ m=2048 at M=2176, analytic J, cap 1 it, no test | 0.106 | 0.135 | 0.83 | 0.41 | 0.98x |
| F4 EQ m=4096 at M=2176, analytic J, cap 1 it, no test | 0.212 | 0.270 | 1.20 | 0.60 | 1.43x |
| F4+F2 EQ m=512 at M=128, analytic J, cap 1 it, no test  (the Burgers q=0 recipe) | 0.022 | 0.024 | 0.52 | 0.26 | 0.62x |
| F4+F2 EQ m=1024 at M=128, analytic J, 2 it/step with test | 0.043 | 0.048 | 0.95 | 0.48 | 1.13x |

Same designs at q=128 (T=160, M=4(K+q)=640 where F2 applies):

| design (q=128) | pass ms | Jacobian ms | ms/step | s/query | vs FOM |
|---|---|---|---|---|---|
| as measured (M=2176, jacfwd, 2.04 it) | 2.593 | 13.028 | 58.56 | 29.28 | 69.51x |
| F1 analytic J | 2.593 | 2.618 | 26.86 | 13.43 | 31.88x |
| F2 M=640 + F1 | 0.763 | 0.770 | 8.26 | 4.13 | 9.80x |
| F4 EQ m=2048 at M=2176, F1, 2 it | 0.106 | 0.235 | 1.97 | 0.99 | 2.34x |
| F4+F2 EQ m=2048 at M=640, F1, cap 1 it, no test | 0.092 | 0.149 | 0.84 | 0.42 | 1.00x |

Latency floor used: 0.35 ms/step = 0.18 s/query at 500 steps -- that alone is 0.42x the FOM. Nothing on this GPU with this loop structure beats the FOM by more than ~2.4x at 500 steps unless the per-step kernel count also drops.

## F. Burgers panel (bpn301, job 3789570, A100 80GB PCIe, 50 steps of dt=0.005, L=256, K=16, R=512) per-step costs

| arm | M | m | T | LM it/step (median) | ms/query | ms/step |
|---|---|---|---|---|---|---|
| q0_M64_dense_g1em06 | 64 | None | 16 | 3.0 | 285 | 5.69 |
| q0_M256_dense_g1em06 | 256 | None | 16 | 3.0 | 341 | 6.83 |
| q128_M576_dense_g1em06 | 576 | None | 144 | 3.0 | 1191 | 23.82 |
| q256_M1088_dense_g1em06 | 1088 | None | 272 | 4.5 | 3945 | 78.91 |
| q0_M64_eqcert_g1em06 | 64 | 1024 | 16 | 3.0 | 60 | 1.20 |
| q0_M64_eqcert_g0p001 | 64 | 1024 | 16 | 2.0 | 44 | 0.87 |
| q0_M64_eqcert_g1em06_fastL4 | 64 | 1024 | 16 | 3.0 | 40 | 0.81 |
| q128_M576_eqcert_g1em06 | 576 | 2048 | 144 | 3.0 | 248 | 4.95 |
| q256_M1088_eqcert_g1em06 | 1088 | 2048 | 272 | 4.5 | 697 | 13.94 |
| pod32_M128_dense | 128 | None | 32 | 2.0 | 81 | 1.62 |
| pod512_M2048_dense | 2048 | None | 512 | 2.0 | 2792 | 55.84 |
| nt1e-3_dt005 | None | None |  | -1.0 | 32 | 0.64 |
| fft_tight | None | None |  | -1.0 | 91 | 1.81 |

