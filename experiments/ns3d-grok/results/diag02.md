# diag02 frozen initial center

Generated from `runs/diag02/output/summary.json`. Job 4140791, commit `e7b9dba53b69cc10d2b82a72056efe69c8220842`. Development only. The per-time oracle recenters every saved truth. The solved arms shift by the centroid of the initial field and do not look at later truth.

| rank | oracle worst | frozen-projection worst | Galerkin dt=0.001 | dt=0.004 | dt=0.01 |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.737% | 28.301% | 30.616% | 30.610% | 30.576% |
| 64 | 0.128% | 23.274% | 26.155% | 26.148% | 26.107% |
| 128 | 0.020% | 17.197% | 19.598% | 19.595% | 19.583% |

Frozen-projection worst at each saved time, starting at $t=0$:

- rank 32: 0.311%, 10.685%, 17.320%, 22.222%, 25.512%, 28.301%
- rank 64: 0.052%, 9.450%, 15.001%, 18.811%, 21.311%, 23.274%
- rank 128: 0.004%, 6.747%, 10.389%, 13.093%, 15.240%, 17.197%

| arm | median ms |
|---|---:|
| FOM dt=0.001 | 24.429 |
| FOM dt=0.004 | 6.337 |
| FOM dt=0.01 | 2.816 |
| shifted Galerkin r128_dt0p0010 | 94.921 |
| shifted Galerkin r128_dt0p0040 | 44.509 |
| shifted Galerkin r128_dt0p0100 | 34.404 |
| shifted Galerkin r32_dt0p0010 | 73.390 |
| shifted Galerkin r32_dt0p0040 | 38.710 |
| shifted Galerkin r32_dt0p0100 | 31.281 |
| shifted Galerkin r64_dt0p0010 | 81.555 |
| shifted Galerkin r64_dt0p0040 | 40.181 |
| shifted Galerkin r64_dt0p0100 | 31.870 |

