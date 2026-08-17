## Stage 2 — latent-stepping ROM (held-out TEST_SEED trajectories, 16 each)

### N=128, K=8  (NVIDIA A100 80GB PCIe; job 2470764)

auto-decoder TRAIN recon 3.77e-03 · ORACLE inferred-latent floor (held-out) 1.28e-02 · IC-fit misfit (u0, cold start) 2.41e-02 · max FOM rel residual 1.0e-12 · FOM rollout 1160 ms

| variant (solver:colloc:objective) | m | traj rel-L2 mean | median | max | blow-ups | iters cold / warm | step ms | rollout ms | speedup (rollout) | end-to-end* |
|---|---|---|---|---|---|---|---|---|---|---|
| `lspg:full:fd` | 15876 | 2.24e-02 | 2.40e-02 | 4.46e-02 | 0/16 | 18.7 / 7.5 | 296.7 | 17016 | 0.07x | 0.06x |
| `galerkin:full:fd` | 15876 | 1.84e-02 | 1.93e-02 | 3.81e-02 | 0/16 | 26.4 / 7.3 | 271.8 | 14275 | 0.08x | 0.07x |
| `lspg:rand512:fd` | 512 | 2.76e-02 | 2.48e-02 | 4.86e-02 | 0/16 | 17.6 / 7.8 | 12.3 | 591 | 1.96x | 0.59x |
| `lspg:offgrid512:fd` | 512 | 2.52e-01 | 1.35e-01 | 1.00e+00 | 0/16 | 16.2 / 13.0 | 66.4 | 4054 | 0.29x | 0.21x |
| `lspg:full:weak64` | 15876 | 1.90e-02 | 1.61e-02 | 5.88e-02 | 0/16 | 7.9 / 5.8 | 296.5 | 13428 | 0.09x | 0.08x |
| `lspg:eq256:weak64` | 256 | 1.89e-02 | 1.67e-02 | 5.12e-02 | 0/16 | 8.1 / 5.8 | 6.5 | 269 | 4.31x | 0.70x |
| `lspg:eq512:weak64` | 512 | 1.94e-02 | 1.62e-02 | 6.33e-02 | 0/16 | 9.2 / 5.8 | 11.3 | 477 | 2.43x | 0.64x |
| `lspg:full:weak256` | 15876 | 1.66e-02 | 1.58e-02 | 2.90e-02 | 0/16 | 7.4 / 5.8 | 295.8 | 13113 | 0.09x | 0.08x |
| `lspg:eq512:weak256` | 512 | 1.72e-02 | 1.66e-02 | 3.13e-02 | 0/16 | 8.9 / 5.8 | 11.1 | 469 | 2.47x | 0.63x |
| `lspg:eq1024:weak256` | 1024 | 1.67e-02 | 1.59e-02 | 2.95e-02 | 0/16 | 9.8 / 5.9 | 21.2 | 914 | 1.27x | 0.50x |
| `galerkin:full:weak64` | 15876 | 1.91e-02 | 1.61e-02 | 5.84e-02 | 0/16 | 9.2 / 6.1 | 274.7 | 12364 | 0.09x | 0.08x |
| `lspg:full:weakc64` | 15876 | 3.03e-02 | 2.66e-02 | 5.59e-02 | 0/16 | 7.5 / 5.9 | 61.5 | 2671 | 0.43x | 0.29x |
| `lspg:eq512:weakc64` | 512 | 2.97e-02 | 2.65e-02 | 5.48e-02 | 0/16 | 7.9 / 5.9 | 4.2 | 167 | 6.93x | 0.76x |
| `lspg:eqoff512:weakc64` | 512 | 2.98e-02 | 2.67e-02 | 5.51e-02 | 0/16 | 7.9 / 5.9 | 4.2 | 161 | 7.21x | 0.77x |

POD control (same solver), projection floors k8=2.01e-01, k16=9.27e-02, k32=4.04e-02, k64=1.40e-02:

| k | variant | traj rel-L2 mean | median | iters warm | step ms | rollout ms | speedup |
|---|---|---|---|---|---|---|---|
| 8 | `lspg:full:fd` | 2.16e-01 | 1.71e-01 | 3.9 | 1.3 | 50 | 23.10x |
| 8 | `galerkin:full:fd` | 2.16e-01 | 1.68e-01 | 3.0 | 3.8 | nan | nanx |
| 8 | `lspg:full:weak64` | 2.15e-01 | 1.67e-01 | 3.9 | 1.6 | nan | nanx |
| 8 | `lspg:eq512:weak64` | 2.15e-01 | 1.67e-01 | 3.9 | 1.3 | nan | nanx |
| 16 | `lspg:full:fd` | 1.05e-01 | 7.52e-02 | 4.0 | 1.5 | 53 | 21.75x |
| 16 | `galerkin:full:fd` | 1.02e-01 | 7.10e-02 | 3.0 | 3.8 | nan | nanx |
| 16 | `lspg:full:weak64` | 9.93e-02 | 6.77e-02 | 4.0 | 1.8 | nan | nanx |
| 16 | `lspg:eq512:weak64` | 9.93e-02 | 6.77e-02 | 4.0 | 1.4 | nan | nanx |
| 32 | `lspg:full:fd` | 4.98e-02 | 3.57e-02 | 4.0 | 1.7 | 68 | 17.13x |
| 32 | `galerkin:full:fd` | 4.58e-02 | 3.17e-02 | 3.1 | 3.8 | nan | nanx |
| 32 | `lspg:full:weak64` | 4.34e-02 | 2.82e-02 | 4.0 | 2.1 | nan | nanx |
| 32 | `lspg:eq512:weak64` | 4.34e-02 | 2.82e-02 | 4.0 | 1.5 | nan | nanx |
| 64 | `lspg:full:fd` | 1.78e-02 | 1.19e-02 | 4.1 | 2.3 | 101 | 11.45x |
| 64 | `galerkin:full:fd` | 1.60e-02 | 9.53e-03 | 3.2 | 4.6 | nan | nanx |
| 64 | `lspg:full:weak64` | 8.26e+01 | 9.68e-01 | 8.5 | 3.3 | nan | nanx |
| 64 | `lspg:eq512:weak64` | 1.13e+02 | 2.80e+00 | 10.3 | 2.1 | nan | nanx |

per-time (t-index 0/10/20/30/40/50): oracle 2.32e-02 / 1.49e-02 / 1.28e-02 / 1.11e-02 / 1.02e-02 / 1.01e-02
; `lspg:full:fd` 2.41e-02 / 2.22e-02 / 2.22e-02 / 2.22e-02 / 2.24e-02 / 2.28e-02
; `lspg:full:weak64` 2.41e-02 / 2.26e-02 / 1.90e-02 / 1.73e-02 / 1.66e-02 / 1.66e-02
; `lspg:eq256:weak64` 2.41e-02 / 2.27e-02 / 1.87e-02 / 1.71e-02 / 1.62e-02 / 1.63e-02
; `lspg:full:weakc64` 2.41e-02 / 2.70e-02 / 2.89e-02 / 3.19e-02 / 3.46e-02 / 3.66e-02

### N=64, K=16  (NVIDIA A100-PCIE-40GB; job 2468412)

auto-decoder TRAIN recon 3.95e-03 · ORACLE inferred-latent floor (held-out) 7.40e-03 · IC-fit misfit (u0, cold start) 1.75e-02 · max FOM rel residual 1.0e-12 · FOM rollout 424 ms

| variant (solver:colloc:objective) | m | traj rel-L2 mean | median | max | blow-ups | iters cold / warm | step ms | rollout ms | speedup (rollout) | end-to-end* |
|---|---|---|---|---|---|---|---|---|---|---|
| `lspg:full:fd` | 3844 | 1.11e-02 | 8.57e-03 | 2.19e-02 | 0/16 | 20.1 / 7.7 | 160.4 | 7718 | 0.05x | 0.05x |
| `galerkin:full:fd` | 3844 | 9.51e-03 | 6.17e-03 | 2.11e-02 | 0/16 | 20.2 / 7.7 | 162.4 | 7703 | 0.06x | 0.05x |
| `lspg:rand512:fd` | 512 | 1.59e-02 | 1.47e-02 | 3.32e-02 | 0/16 | 21.2 / 8.3 | 22.1 | 1043 | 0.41x | 0.19x |
| `lspg:offgrid512:fd` | 512 | 9.07e-02 | 5.95e-02 | 2.84e-01 | 0/16 | 18.6 / 13.1 | 79.4 | 4188 | 0.10x | 0.08x |
| `lspg:full:weak64` | 3844 | 9.62e-03 | 6.99e-03 | 2.23e-02 | 0/16 | 10.0 / 6.6 | 141.1 | 6343 | 0.07x | 0.06x |
| `lspg:eq256:weak64` | 256 | 1.10e-02 | 7.94e-03 | 2.41e-02 | 0/16 | 12.2 / 6.7 | 11.5 | 466 | 0.91x | 0.25x |
| `lspg:eq512:weak64` | 512 | 9.91e-03 | 7.27e-03 | 2.24e-02 | 0/16 | 11.8 / 6.6 | 19.7 | 823 | 0.52x | 0.21x |
| `lspg:full:weak256` | 3844 | 8.97e-03 | 6.01e-03 | 2.17e-02 | 0/16 | 10.6 / 6.5 | 139.8 | 6374 | 0.07x | 0.06x |
| `lspg:eq512:weak256` | 512 | 9.85e-03 | 6.65e-03 | 2.22e-02 | 0/16 | 10.5 / 6.5 | 18.5 | 828 | 0.51x | 0.21x |
| `lspg:eq1024:weak256` | 1024 | 9.15e-03 | 6.21e-03 | 2.17e-02 | 0/16 | 11.4 / 6.5 | 38.4 | 1718 | 0.25x | 0.14x |
| `galerkin:full:weak64` | 3844 | 9.63e-03 | 7.00e-03 | 2.23e-02 | 0/16 | 11.1 / 6.4 | 141.3 | 6199 | 0.07x | 0.06x |
| `lspg:full:weakc64` | 3844 | 4.31e-02 | 3.90e-02 | 9.62e-02 | 0/16 | 11.2 / 6.7 | 30.3 | 1288 | 0.33x | 0.17x |
| `lspg:eq512:weakc64` | 512 | 4.26e-02 | 3.87e-02 | 9.60e-02 | 0/16 | 10.2 / 6.7 | 6.7 | 236 | 1.80x | 0.29x |
| `lspg:eqoff512:weakc64` | 512 | 4.26e-02 | 3.87e-02 | 9.65e-02 | 0/16 | 11.6 / 6.7 | 6.6 | 235 | 1.81x | 0.29x |

POD control (same solver), projection floors k8=1.96e-01, k16=8.90e-02, k32=3.79e-02, k64=1.22e-02:

| k | variant | traj rel-L2 mean | median | iters warm | step ms | rollout ms | speedup |
|---|---|---|---|---|---|---|---|
| 8 | `lspg:full:fd` | 2.09e-01 | 1.65e-01 | 3.9 | 1.4 | 35 | 12.10x |
| 8 | `galerkin:full:fd` | 2.10e-01 | 1.64e-01 | 3.0 | 4.2 | nan | nanx |
| 8 | `lspg:full:weak64` | 2.09e-01 | 1.60e-01 | 3.9 | 1.6 | nan | nanx |
| 8 | `lspg:eq512:weak64` | 2.09e-01 | 1.60e-01 | 3.9 | 1.6 | nan | nanx |
| 16 | `lspg:full:fd` | 9.73e-02 | 7.04e-02 | 4.0 | 1.5 | 42 | 10.02x |
| 16 | `galerkin:full:fd` | 9.66e-02 | 6.81e-02 | 3.0 | 4.2 | nan | nanx |
| 16 | `lspg:full:weak64` | 9.50e-02 | 6.58e-02 | 4.0 | 1.6 | nan | nanx |
| 16 | `lspg:eq512:weak64` | 9.50e-02 | 6.58e-02 | 4.0 | 1.6 | nan | nanx |
| 32 | `lspg:full:fd` | 4.32e-02 | 3.07e-02 | 4.0 | 1.6 | 44 | 9.68x |
| 32 | `galerkin:full:fd` | 4.20e-02 | 2.83e-02 | 3.1 | 4.2 | nan | nanx |
| 32 | `lspg:full:weak64` | 4.06e-02 | 2.62e-02 | 4.0 | 1.7 | nan | nanx |
| 32 | `lspg:eq512:weak64` | 4.06e-02 | 2.62e-02 | 4.0 | 1.6 | nan | nanx |
| 64 | `lspg:full:fd` | 1.40e-02 | 8.92e-03 | 4.1 | 2.0 | 63 | 6.72x |
| 64 | `galerkin:full:fd` | 1.34e-02 | 8.01e-03 | 3.2 | 4.6 | nan | nanx |
| 64 | `lspg:full:weak64` | 1.66e+00 | 2.17e-01 | 8.6 | 3.0 | nan | nanx |
| 64 | `lspg:eq512:weak64` | 6.18e-01 | 2.17e-01 | 7.2 | 2.3 | nan | nanx |

per-time (t-index 0/10/20/30/40/50): oracle 1.75e-02 / 8.89e-03 / 7.24e-03 / 5.81e-03 / 5.18e-03 / 5.32e-03
; `lspg:full:fd` 1.75e-02 / 1.22e-02 / 1.10e-02 / 1.02e-02 / 9.63e-03 / 9.50e-03
; `lspg:full:weak64` 1.75e-02 / 1.21e-02 / 9.63e-03 / 7.57e-03 / 6.88e-03 / 7.04e-03
; `lspg:eq256:weak64` 1.75e-02 / 1.36e-02 / 1.09e-02 / 9.00e-03 / 8.28e-03 / 8.36e-03
; `lspg:full:weakc64` 1.75e-02 / 2.83e-02 / 4.07e-02 / 5.01e-02 / 5.72e-02 / 6.22e-02

### N=64, K=4  (NVIDIA A100 80GB PCIe; job 2468407)

auto-decoder TRAIN recon 6.37e-03 · ORACLE inferred-latent floor (held-out) 5.24e-02 · IC-fit misfit (u0, cold start) 8.09e-02 · max FOM rel residual 1.0e-12 · FOM rollout 420 ms

| variant (solver:colloc:objective) | m | traj rel-L2 mean | median | max | blow-ups | iters cold / warm | step ms | rollout ms | speedup (rollout) | end-to-end* |
|---|---|---|---|---|---|---|---|---|---|---|
| `lspg:full:fd` | 3844 | 8.90e-02 | 8.78e-02 | 2.18e-01 | 0/16 | 9.8 / 6.7 | 43.8 | 1833 | 0.23x | 0.15x |
| `galerkin:full:fd` | 3844 | 8.51e-02 | 8.56e-02 | 2.07e-01 | 0/16 | 12.1 / 7.7 | 52.1 | 2082 | 0.20x | 0.14x |
| `lspg:rand512:fd` | 512 | 9.39e-02 | 8.66e-02 | 1.89e-01 | 0/16 | 11.7 / 7.1 | 8.2 | 312 | 1.34x | 0.33x |
| `lspg:offgrid512:fd` | 512 | 2.67e-01 | 1.30e-01 | 1.06e+00 | 0/16 | 14.6 / 7.0 | 16.2 | 608 | 0.69x | 0.28x |
| `lspg:full:weak64` | 3844 | 7.74e-02 | 7.68e-02 | 1.79e-01 | 0/16 | 6.8 / 5.8 | 45.3 | 1941 | 0.22x | 0.15x |
| `lspg:eq256:weak64` | 256 | 7.74e-02 | 7.80e-02 | 1.80e-01 | 0/16 | 7.2 / 5.8 | 5.1 | 185 | 2.26x | 0.38x |
| `lspg:eq512:weak64` | 512 | 7.75e-02 | 7.63e-02 | 1.80e-01 | 0/16 | 6.9 / 5.8 | 7.8 | 300 | 1.40x | 0.34x |
| `lspg:full:weak256` | 3844 | 8.08e-02 | 8.07e-02 | 1.90e-01 | 0/16 | 7.4 / 5.9 | 45.4 | 1922 | 0.22x | 0.14x |
| `lspg:eq512:weak256` | 512 | 8.09e-02 | 7.98e-02 | 1.90e-01 | 0/16 | 7.4 / 6.0 | 7.8 | 307 | 1.37x | 0.35x |
| `lspg:eq1024:weak256` | 1024 | 8.05e-02 | 7.75e-02 | 1.90e-01 | 0/16 | 7.4 / 5.9 | 15.1 | 576 | 0.73x | 0.27x |
| `galerkin:full:weak64` | 3844 | 7.76e-02 | 7.77e-02 | 1.80e-01 | 0/16 | 8.6 / 6.7 | 51.5 | 2040 | 0.21x | 0.14x |
| `lspg:full:weakc64` | 3844 | 8.93e-02 | 8.86e-02 | 1.79e-01 | 0/16 | 6.6 / 5.8 | 11.7 | 436 | 0.96x | 0.31x |
| `lspg:eq512:weakc64` | 512 | 8.94e-02 | 8.88e-02 | 1.79e-01 | 0/16 | 6.8 / 5.8 | 3.7 | 126 | 3.32x | 0.39x |
| `lspg:eqoff512:weakc64` | 512 | 8.93e-02 | 8.90e-02 | 1.79e-01 | 0/16 | 6.9 / 5.8 | 3.7 | 133 | 3.17x | 0.40x |

POD control (same solver), projection floors k8=1.96e-01, k16=8.90e-02, k32=3.79e-02, k64=1.22e-02:

| k | variant | traj rel-L2 mean | median | iters warm | step ms | rollout ms | speedup |
|---|---|---|---|---|---|---|---|
| 8 | `lspg:full:fd` | 2.09e-01 | 1.65e-01 | 3.9 | 1.3 | 38 | 11.10x |
| 8 | `galerkin:full:fd` | 2.10e-01 | 1.64e-01 | 3.0 | 3.9 | nan | nanx |
| 8 | `lspg:full:weak64` | 2.09e-01 | 1.60e-01 | 3.9 | 1.4 | nan | nanx |
| 8 | `lspg:eq512:weak64` | 2.09e-01 | 1.60e-01 | 3.9 | 1.4 | nan | nanx |
| 16 | `lspg:full:fd` | 9.73e-02 | 7.04e-02 | 4.0 | 1.3 | 39 | 10.89x |
| 16 | `galerkin:full:fd` | 9.66e-02 | 6.81e-02 | 3.0 | 3.8 | nan | nanx |
| 16 | `lspg:full:weak64` | 9.50e-02 | 6.58e-02 | 4.0 | 1.4 | nan | nanx |
| 16 | `lspg:eq512:weak64` | 9.50e-02 | 6.58e-02 | 4.0 | 1.4 | nan | nanx |
| 32 | `lspg:full:fd` | 4.32e-02 | 3.07e-02 | 4.0 | 1.4 | 45 | 9.31x |
| 32 | `galerkin:full:fd` | 4.20e-02 | 2.83e-02 | 3.1 | 3.8 | nan | nanx |
| 32 | `lspg:full:weak64` | 4.06e-02 | 2.62e-02 | 4.0 | 1.6 | nan | nanx |
| 32 | `lspg:eq512:weak64` | 4.06e-02 | 2.62e-02 | 4.0 | 1.5 | nan | nanx |
| 64 | `lspg:full:fd` | 1.40e-02 | 8.92e-03 | 4.1 | 1.8 | 70 | 6.00x |
| 64 | `galerkin:full:fd` | 1.34e-02 | 8.01e-03 | 3.2 | 3.9 | nan | nanx |
| 64 | `lspg:full:weak64` | 1.66e+00 | 2.17e-01 | 8.6 | 2.9 | nan | nanx |
| 64 | `lspg:eq512:weak64` | 6.18e-01 | 2.17e-01 | 7.2 | 2.3 | nan | nanx |

per-time (t-index 0/10/20/30/40/50): oracle 8.09e-02 / 7.00e-02 / 5.22e-02 / 4.29e-02 / 4.00e-02 / 4.10e-02
; `lspg:full:fd` 8.09e-02 / 8.89e-02 / 8.85e-02 / 8.92e-02 / 9.34e-02 / 9.53e-02
; `lspg:full:weak64` 8.09e-02 / 7.97e-02 / 7.69e-02 / 7.51e-02 / 7.82e-02 / 8.05e-02
; `lspg:eq256:weak64` 8.09e-02 / 7.98e-02 / 7.68e-02 / 7.50e-02 / 7.80e-02 / 8.02e-02
; `lspg:full:weakc64` 8.09e-02 / 8.44e-02 / 8.60e-02 / 8.97e-02 / 9.76e-02 / 1.04e-01

### N=64, K=8  (NVIDIA H100 PCIe; job 2468411)

auto-decoder TRAIN recon 3.52e-03 · ORACLE inferred-latent floor (held-out) 1.15e-02 · IC-fit misfit (u0, cold start) 2.31e-02 · max FOM rel residual 1.0e-12 · FOM rollout 321 ms

| variant (solver:colloc:objective) | m | traj rel-L2 mean | median | max | blow-ups | iters cold / warm | step ms | rollout ms | speedup (rollout) | end-to-end* |
|---|---|---|---|---|---|---|---|---|---|---|
| `lspg:full:fd` | 3844 | 2.01e-02 | 2.05e-02 | 3.16e-02 | 0/16 | 13.4 / 6.7 | 46.1 | 2435 | 0.13x | 0.11x |
| `galerkin:full:fd` | 3844 | 1.80e-02 | 1.81e-02 | 3.06e-02 | 0/16 | 16.3 / 7.2 | 44.8 | 2312 | 0.14x | 0.11x |
| `lspg:rand512:fd` | 512 | 2.30e-02 | 2.30e-02 | 5.06e-02 | 0/16 | 14.3 / 6.9 | 8.2 | 416 | 0.77x | 0.31x |
| `lspg:offgrid512:fd` | 512 | 1.75e-01 | 6.07e-02 | 1.07e+00 | 0/16 | 17.9 / 10.2 | 19.9 | 1301 | 0.25x | 0.17x |
| `lspg:full:weak64` | 3844 | 1.65e-02 | 1.53e-02 | 3.75e-02 | 0/16 | 8.2 / 5.7 | 46.5 | 2012 | 0.16x | 0.12x |
| `lspg:eq256:weak64` | 256 | 1.74e-02 | 1.56e-02 | 3.95e-02 | 0/16 | 8.9 / 5.7 | 4.7 | 196 | 1.64x | 0.40x |
| `lspg:eq512:weak64` | 512 | 1.68e-02 | 1.54e-02 | 3.83e-02 | 0/16 | 8.8 / 5.7 | 7.9 | 341 | 0.94x | 0.33x |
| `lspg:full:weak256` | 3844 | 1.67e-02 | 1.60e-02 | 3.16e-02 | 0/16 | 9.6 / 5.9 | 46.1 | 2043 | 0.16x | 0.12x |
| `lspg:eq512:weak256` | 512 | 1.70e-02 | 1.65e-02 | 3.08e-02 | 0/16 | 9.9 / 5.9 | 7.8 | 335 | 0.96x | 0.32x |
| `lspg:eq1024:weak256` | 1024 | 1.68e-02 | 1.60e-02 | 3.03e-02 | 0/16 | 10.4 / 6.0 | 15.3 | 652 | 0.49x | 0.24x |
| `galerkin:full:weak64` | 3844 | 1.65e-02 | 1.53e-02 | 3.75e-02 | 0/16 | 17.2 / 5.9 | 44.3 | 1992 | 0.16x | 0.12x |
| `lspg:full:weakc64` | 3844 | 4.56e-02 | 4.04e-02 | 9.55e-02 | 0/16 | 8.6 / 5.9 | 9.8 | 422 | 0.76x | 0.30x |
| `lspg:eq512:weakc64` | 512 | 4.51e-02 | 4.00e-02 | 9.53e-02 | 0/16 | 8.6 / 5.9 | 3.1 | 113 | 2.83x | 0.44x |
| `lspg:eqoff512:weakc64` | 512 | 4.53e-02 | 3.99e-02 | 9.56e-02 | 0/16 | 7.8 / 5.8 | 3.1 | 114 | 2.82x | 0.44x |

POD control (same solver), projection floors k8=1.96e-01, k16=8.90e-02, k32=3.79e-02, k64=1.22e-02:

| k | variant | traj rel-L2 mean | median | iters warm | step ms | rollout ms | speedup |
|---|---|---|---|---|---|---|---|
| 8 | `lspg:full:fd` | 2.09e-01 | 1.65e-01 | 3.9 | 0.9 | 33 | 9.62x |
| 8 | `galerkin:full:fd` | 2.10e-01 | 1.64e-01 | 3.0 | 3.4 | nan | nanx |
| 8 | `lspg:full:weak64` | 2.09e-01 | 1.60e-01 | 3.9 | 1.0 | nan | nanx |
| 8 | `lspg:eq512:weak64` | 2.09e-01 | 1.60e-01 | 3.9 | 1.0 | nan | nanx |
| 16 | `lspg:full:fd` | 9.73e-02 | 7.04e-02 | 4.0 | 1.0 | 36 | 9.02x |
| 16 | `galerkin:full:fd` | 9.66e-02 | 6.81e-02 | 3.0 | 3.4 | nan | nanx |
| 16 | `lspg:full:weak64` | 9.50e-02 | 6.58e-02 | 4.0 | 1.1 | nan | nanx |
| 16 | `lspg:eq512:weak64` | 9.50e-02 | 6.58e-02 | 4.0 | 1.1 | nan | nanx |
| 32 | `lspg:full:fd` | 4.32e-02 | 3.07e-02 | 4.0 | 1.1 | 36 | 8.83x |
| 32 | `galerkin:full:fd` | 4.20e-02 | 2.83e-02 | 3.1 | 3.4 | nan | nanx |
| 32 | `lspg:full:weak64` | 4.06e-02 | 2.62e-02 | 4.0 | 1.2 | nan | nanx |
| 32 | `lspg:eq512:weak64` | 4.06e-02 | 2.62e-02 | 4.0 | 1.2 | nan | nanx |
| 64 | `lspg:full:fd` | 1.40e-02 | 8.92e-03 | 4.1 | 1.5 | 56 | 5.68x |
| 64 | `galerkin:full:fd` | 1.34e-02 | 8.01e-03 | 3.2 | 3.4 | nan | nanx |
| 64 | `lspg:full:weak64` | 1.66e+00 | 2.17e-01 | 8.6 | 2.5 | nan | nanx |
| 64 | `lspg:eq512:weak64` | 6.18e-01 | 2.17e-01 | 7.2 | 1.9 | nan | nanx |

per-time (t-index 0/10/20/30/40/50): oracle 2.32e-02 / 1.44e-02 / 1.10e-02 / 9.17e-03 / 8.37e-03 / 8.07e-03
; `lspg:full:fd` 2.31e-02 / 2.21e-02 / 2.00e-02 / 1.89e-02 / 1.86e-02 / 1.83e-02
; `lspg:full:weak64` 2.31e-02 / 1.91e-02 / 1.65e-02 / 1.47e-02 / 1.40e-02 / 1.37e-02
; `lspg:eq256:weak64` 2.31e-02 / 2.05e-02 / 1.75e-02 / 1.55e-02 / 1.47e-02 / 1.44e-02
; `lspg:full:weakc64` 2.31e-02 / 3.25e-02 / 4.35e-02 / 5.18e-02 / 5.78e-02 / 6.20e-02


## Stage 1 — space-time LSPG with the (z,t) sweep decoder (N=64, 16 test, budget 100)

| IC_W | arm | traj rel-L2 mean | median | max | |z-z*| |
|---|---|---|---|---|---|
| sqrt(50) | `ic` | 2.25e-01 | 1.71e-01 | 8.66e-01 | 6.71e-01 |
| sqrt(50) | `resid` | 9.52e-01 | 9.43e-01 | 1.16e+00 | 1.71e+00 |
| sqrt(50) | `both` | 2.22e-02 | 1.14e-02 | 1.80e-01 | 5.82e-02 |
| 1 | `ic` | 2.25e-01 | 1.71e-01 | 8.66e-01 | 6.71e-01 |
| 1 | `resid` | 9.52e-01 | 9.43e-01 | 1.16e+00 | 1.71e+00 |
| 1 | `both` | 7.29e-03 | 3.47e-03 | 5.19e-02 | 1.21e-02 |

oracle (true z) 3.83e-03 mean / 1.65e-02 max; FOM trajectory residual through the FOM residual 2.8e-13; ours-vs-FOM residual max|diff| 0.0e+00; decoder residual at true z (rel) 2.94e-03
