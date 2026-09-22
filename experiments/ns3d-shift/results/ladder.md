# Resolution ladder: does the shift ROM stay accurate and get fast?

Generated from each job's `summary.json` by `write_ladder_table.py`. Accuracy is the full development cohort; timing is one case in a single interleaved block per job, complete queries. **Comparator rule:** the fastest tested *stable* CNAB2 setting whose evolved worst is no larger than the ROM's. A CNAB2 setting counts as unstable when its evolved worst exceeds 100 %; unstable settings are never comparators.

## Headline per mesh

| mesh | best setting meeting 5 % | evolved worst | over 5 % | ROM ms | comparator | stable? | comparator error | comparator ms | paired speedup | stability-limited FOM | that speedup |
|---:|---|---:|---:|---:|---|---|---:|---:|---:|---|---:|
| 32^3 | r64 dt0.02 it3 | 0.602% | 0/16 | 5.302 | CNAB2 dt=0.005 | stable | 0.152% | 5.342 | **1.01x** | CNAB2 dt=0.01 | 0.59x |
| 64^3 | r64 dt0.04 it3 | 1.946% | 0/16 | 5.370 | CNAB2 dt=0.005 | stable | 0.382% | 19.086 | **3.55x** | CNAB2 dt=0.005 | 3.55x |

The **stability-limited FOM** is the cheapest stable CNAB2 that itself meets the 5 % target, i.e. the cheapest the FOM can honestly be run at that mesh. Where it equals the matched-accuracy comparator, the speedup is pure throughput; where it is coarser, the extra margin comes from the reduced model being more accurate than the FOM at that step, not from taking a step the FOM cannot.

## 32^3

### Representation floor (is accuracy limited by the bank?)

| rank | oracle-shift floor, evolved worst | over 5 % |
|---:|---:|---:|
| 64 | 0.125% | 0/16 |
| 128 | 0.019% | 0/16 |

### Accuracy-cost frontier

| rank | dt | steps | sweeps | evolved worst | over 5 % | query ms | comparator | comparator ms | paired speedup |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 64 | 0.02 | 10 | 2 | 0.603% | 0/16 | 3.918 | CNAB2 dt=0.005 | 5.342 | 1.363x |
| 64 | 0.04 | 5 | 2 | 2.007% | 0/16 | 2.498 | CNAB2 dt=0.01 | 3.114 | 1.246x |
| 64 | 0.02 | 10 | 3 | 0.602% | 0/16 | 5.302 | CNAB2 dt=0.005 | 5.342 | 1.008x |
| 64 | 0.04 | 5 | 3 | 1.913% | 0/16 | 3.193 | CNAB2 dt=0.01 | 3.114 | 0.975x |
| 128 | 0.04 | 5 | 2 | 5.020% | 1/16 | 3.652 | CNAB2 dt=0.01 | 3.114 | 0.853x |
| 64 | 0.04 | 5 | 4 | 1.912% | 0/16 | 3.893 | CNAB2 dt=0.01 | 3.114 | 0.800x |
| 64 | 0.02 | 10 | 4 | 0.602% | 0/16 | 6.726 | CNAB2 dt=0.005 | 5.342 | 0.794x |
| 64 | 0.01 | 20 | 2 | 0.504% | 0/16 | 6.857 | CNAB2 dt=0.005 | 5.342 | 0.779x |
| 128 | 0.04 | 5 | 3 | 3.898% | 0/16 | 4.845 | CNAB2 dt=0.01 | 3.114 | 0.643x |
| 64 | 0.01 | 20 | 3 | 0.504% | 0/16 | 9.494 | CNAB2 dt=0.005 | 5.342 | 0.563x |
| 128 | 0.04 | 5 | 4 | 3.869% | 0/16 | 5.911 | CNAB2 dt=0.01 | 3.114 | 0.527x |
| 128 | 0.02 | 10 | 2 | 1.630% | 0/16 | 6.004 | CNAB2 dt=0.01 | 3.114 | 0.519x |
| 128 | 0.01 | 20 | 2 | 0.606% | 0/16 | 11.208 | CNAB2 dt=0.005 | 5.342 | 0.477x |
| 64 | 0.01 | 20 | 4 | 0.504% | 0/16 | 12.891 | CNAB2 dt=0.005 | 5.342 | 0.414x |
| 128 | 0.02 | 10 | 3 | 1.536% | 0/16 | 8.496 | CNAB2 dt=0.01 | 3.114 | 0.367x |
| 128 | 0.01 | 20 | 3 | 0.579% | 0/16 | 15.998 | CNAB2 dt=0.005 | 5.342 | 0.334x |
| 128 | 0.02 | 10 | 4 | 1.532% | 0/16 | 11.031 | CNAB2 dt=0.01 | 3.114 | 0.282x |
| 128 | 0.01 | 20 | 4 | 0.577% | 0/16 | 20.745 | CNAB2 dt=0.005 | 5.342 | 0.258x |

**Stability and step size.** For CNAB2 at this mesh, no tested step blew up; the coarsest step that both stays finite and meets the 5 % target is 0.01 (20 steps) -- that is the stability-limited comparator. The reduced model runs at 0.04 (5 steps), because its step is solved implicitly at the midpoint instead of advanced explicitly. Where the two comparator columns differ, the gap between them is the part of the margin that comes from a step the FOM cannot take rather than from throughput.

### Baselines and the FOM ladder

| arm | evolved worst | over 5 % | median ms | note |
|---|---:|---:|---:|---|
| reference LM arm (rank 64, dt 0.01) | 0.504% | 0/16 | 23.954 | the pre-fix solver, same job |
| centroid tracker (rank 64, dt 0.01) | 2.514% | 0/16 | 14.108 | the arm to beat |
| CNAB2 dt=0.004 (50 steps) | 0.094% | 0/16 | 6.694 |  |
| CNAB2 dt=0.005 (40 steps) | 0.152% | 0/16 | 5.342 |  |
| CNAB2 dt=0.01 (20 steps) | 0.788% | 0/16 | 3.114 |  |
| CNAB2 dt=0.02 (10 steps) | 76.694% | 10/16 | 1.741 |  |

### Cost breakdown

| piece | median ms |
|---|---:|
| initial r128 | 0.288 |
| initial r64 | 0.381 |
| output r128 | 0.196 |
| output r64 | 0.197 |
| four extra output reconstructions (r64_dt0.01_it2) | 0.756 |

The six-output contract plus the initial projection is 1.366 ms at rank 64. A solve of zero cost would therefore cap the speedup at **2.28x** against the stability-limited FOM (CNAB2 dt=0.01, 3.114 ms) at this mesh.

Independent NumPy recomputation: worst disagreement 8.674e-19; driver-fix parity against the LM arm 7.870e-09.

GPU: NVIDIA A100 80GB PCIe, GPU-d881b2b0-b08f-83e6-0f3d-e646626625cc, 81920 MiB. Job 4178112, commit `6ec038ec27defe8d72392851eaed43c13dafd92f`. Operator checks and the GPU-centering cross-check are in the summary.

## 64^3

### Representation floor (is accuracy limited by the bank?)

| rank | oracle-shift floor, evolved worst | over 5 % |
|---:|---:|---:|
| 64 | 0.129% | 0/16 |
| 128 | 0.020% | 0/16 |

### Accuracy-cost frontier

| rank | dt | steps | sweeps | evolved worst | over 5 % | query ms | comparator | comparator ms | paired speedup |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 64 | 0.04 | 5 | 2 | 1.990% | 0/16 | 4.748 | CNAB2 dt=0.005 | 19.086 | 4.019x |
| 64 | 0.04 | 5 | 3 | 1.946% | 0/16 | 5.370 | CNAB2 dt=0.005 | 19.086 | 3.554x |
| 64 | 0.02 | 10 | 2 | 0.617% | 0/16 | 6.099 | CNAB2 dt=0.005 | 19.086 | 3.129x |
| 128 | 0.04 | 5 | 2 | 4.033% | 0/16 | 7.273 | CNAB2 dt=0.005 | 19.086 | 2.624x |
| 64 | 0.02 | 10 | 3 | 0.616% | 0/16 | 7.565 | CNAB2 dt=0.005 | 19.086 | 2.523x |
| 128 | 0.04 | 5 | 3 | 3.757% | 0/16 | 8.377 | CNAB2 dt=0.005 | 19.086 | 2.278x |
| 64 | 0.01 | 20 | 2 | 0.526% | 0/16 | 8.986 | CNAB2 dt=0.005 | 19.086 | 2.124x |
| 128 | 0.02 | 10 | 2 | 1.356% | 0/16 | 9.661 | CNAB2 dt=0.005 | 19.086 | 1.976x |
| 64 | 0.01 | 20 | 3 | 0.526% | 0/16 | 11.996 | CNAB2 dt=0.005 | 19.086 | 1.591x |
| 128 | 0.02 | 10 | 3 | 1.371% | 0/16 | 12.335 | CNAB2 dt=0.005 | 19.086 | 1.547x |
| 128 | 0.01 | 20 | 2 | 0.518% | 0/16 | 14.839 | CNAB2 dt=0.005 | 19.086 | 1.286x |
| 128 | 0.01 | 20 | 3 | 0.508% | 0/16 | 19.560 | CNAB2 dt=0.005 | 19.086 | 0.976x |

**Stability and step size.** For CNAB2 at this mesh, steps at or above 0.01 blew up (evolved worst over 100 %); the coarsest step that both stays finite and meets the 5 % target is 0.005 (40 steps) -- that is the stability-limited comparator. The reduced model runs at 0.04 (5 steps), because its step is solved implicitly at the midpoint instead of advanced explicitly. Where the two comparator columns differ, the gap between them is the part of the margin that comes from a step the FOM cannot take rather than from throughput.

### Baselines and the FOM ladder

| arm | evolved worst | over 5 % | median ms | note |
|---|---:|---:|---:|---|
| reference LM arm (rank 64, dt 0.01) | 0.526% | 0/16 | 26.095 | the pre-fix solver, same job |
| centroid tracker (rank 64, dt 0.01) | 2.529% | 0/16 | 55.781 | the arm to beat |
| CNAB2 dt=0.001 (200 steps) | 0.000% | 0/16 | 89.244 | this is the reference itself |
| CNAB2 dt=0.002 (100 steps) | 0.019% | 0/16 | 45.476 |  |
| CNAB2 dt=0.004 (50 steps) | 0.093% | 0/16 | 23.484 |  |
| CNAB2 dt=0.005 (40 steps) | 0.382% | 0/16 | 19.086 |  |
| CNAB2 dt=0.01 (20 steps) | 686.143% | 6/16 | 10.413 | **unstable** |
| CNAB2 dt=0.02 (10 steps) | 272.038% | 12/16 | 5.716 | **unstable** |

### Cost breakdown

| piece | median ms |
|---|---:|
| initial r128 | 0.785 |
| initial r64 | 0.577 |
| output r128 | 0.696 |
| output r64 | 0.479 |
| four extra output reconstructions (r128_dt0.01_it2) | 3.115 |
| four extra output reconstructions (r128_dt0.02_it2) | 2.886 |
| four extra output reconstructions (r128_dt0.04_it2) | 3.050 |
| four extra output reconstructions (r64_dt0.01_it2) | 2.114 |
| four extra output reconstructions (r64_dt0.02_it2) | 2.062 |
| four extra output reconstructions (r64_dt0.04_it2) | 2.114 |

The six-output contract plus the initial projection is 2.972 ms at rank 64. A solve of zero cost would therefore cap the speedup at **6.42x** against the stability-limited FOM (CNAB2 dt=0.005, 19.086 ms) at this mesh.

Independent NumPy recomputation: worst disagreement 0.000e+00; driver-fix parity against the LM arm 1.429e-08.

GPU: NVIDIA A100 80GB PCIe, GPU-d881b2b0-b08f-83e6-0f3d-e646626625cc, 81920 MiB. Job 4178149, commit `eceb1b779d92897cc7b9f6da7abdf60c8ef1757e`. Operator checks and the GPU-centering cross-check are in the summary.

