# S0 resource estimate — audit before submission

Status: planning-only, generated before any Phase-2 scientific submission.

The exact mixed-resolution workload is 5,712 free-oracle fits per arm and 17,136
total: 9,792 fits at N=64, 4,896 at N=128, and 2,448 at N=256.  A local excluded
single-snapshot bracket measured these conservative serial fit times:

| N | R=24 | R=32 | R=48 | LSMR iterations |
|---:|---:|---:|---:|---:|
| 64 | 0.0494 s | 0.0435 s | 0.0421 s | 500 |
| 128 | 0.3182 s | 0.2556 s | 0.3157 s | 500 |
| 256 | 1.1821 s | 1.2198 s | 1.2585 s | 500 |

Using the largest measured time at each N gives a serial projection estimate of
about 5,122 seconds (1.42 hours).  Scientific S0 uses a deterministic ordered
eight-worker thread map and fixes OMP/OpenBLAS/MKL to one thread.  The allocation
retains more than 4x the serial estimate for reference generation, compilation,
timing, I/O, and imperfect memory-bandwidth scaling; no fit is truncated.

Planned isolated cell and request:

```
cell: s0_spline_r1
job name: ctol_b10_s0_spline_r1
partition: gpu
GPU: h200:1
CPU: 8
host memory: 64G
walltime: 06:00:00
environment: SMOKE=0 TIME_REPS=10 TIME_WARM=1 BURN_SECONDS=3
             ORACLE_WORKERS=8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
             MKL_NUM_THREADS=1
command: $PY b10_s0_spline.py ../out/s0.json ../out/s0.npz
```

The N=1024 decoded output is 0.428 GB.  The preregistered compiled-executable
eligibility gate is 20 GB, measured separately for direct, mandatory-weak, and
maximum-one-update kernels.  The 64 GB host request covers eight simultaneous
sparse matrices, all accumulated coefficient/error arrays, reference trajectories,
compression, and a wide safety margin.  The H200 request is for the N=1024 paired
FOM/ROM panel; no cross-job wall-clock number is used for eligibility.
