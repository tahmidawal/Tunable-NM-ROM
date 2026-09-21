# diag01 development diagnosis

Generated from `runs/diag01/output/summary.json`. Job 4139559, commit `010cf0666aad50f8ddbcfb89b129d5c8ceff645a`, NVIDIA A100-PCIE-40GB, GPU-a840f593-3e39-4ee0-681d-f3898b49b09b, 40960 MiB. Development seed only. The final cohort was not opened. Evolved worst is the worst case of the worst evolved time. Evolved median is the median across cases of that per-case worst.

| rank | floor worst | floor over 5% | Galerkin dt=0.001 worst | over 5% | one-interval worst |
|---:|---:|---:|---:|---:|---:|
| 64 | 53.860% | 16/16 | 54.448% | 16/16 | 53.958% |
| 128 | 36.883% | 16/16 | 39.715% | 16/16 | 37.051% |
| 256 | 27.433% | 16/16 | 29.639% | 16/16 | 27.671% |
| 512 | 17.808% | 16/16 | 19.792% | 16/16 | 18.061% |
| 1024 | 10.269% | 10/16 | 11.555% | 11/16 | 10.437% |
| 1536 | 7.426% | 7/16 | 8.498% | 8/16 | 7.591% |
| 2048 | 5.510% | 2/16 | 6.643% | 2/16 | 5.733% |
| 3072 | 3.841% | 0/16 | 4.597% | 0/16 | 4.007% |

Largest rank also at dt=0.004: Galerkin evolved worst 4.610%, 0/16 over 5%.

| rank | oracle-shift evolved worst | cases over 5% | centroid travel worst |
|---:|---:|---:|---:|
| 64 | 0.128% | 0/16 | 0.049473 |
| 128 | 0.020% | 0/16 | 0.049473 |
| 237 | 0.003% | 0/16 | 0.049473 |

| arm | median ms |
|---|---:|
| FOM dt0.001 | 25.562 |
| FOM dt0.004 | 6.881 |
| FOM dt0.01 | 3.129 |
| Galerkin rank 1024, dt=0.001 | 308.305 |
| Galerkin rank 256, dt=0.001 | 108.982 |
| Galerkin rank 3072, dt=0.001 | 853.237 |
| Galerkin rank 64, dt=0.001 | 59.089 |
| weak POD rank 64, dt=0.004, 2048 tests | 597.826 |

