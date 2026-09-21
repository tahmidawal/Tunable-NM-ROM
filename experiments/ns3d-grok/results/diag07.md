# diag07 sealed coefficient tracker

Generated from `runs/diag07/output/summary.json`. Job 4148215, commit `f12b8fc23e04fa2ec371eab4509a0694e2519ffd`. Device: [CudaDevice(id=0)]. Seed 202609211, 32 cases, opened once. Setting frozen from diag06: centered POD rank 64, startup step, dt=0.01, Fourier tail 1e-06. Error and time for each method come from the same timed calls. The comparator is the fastest tested CNAB2 step whose evolved worst is no larger than the coefficient ROM's.

| method | evolved worst | evolved median | cases over 5% | median ms | parity vs grid |
|---|---:|---:|---:|---:|---:|
| coefficient ROM | 4.799% | 3.289% | 0/32 | 6.478 | 9.866e-07 |
| grid tracker | 4.799% | 3.289% | 0/32 | 14.766 | |
| CNAB2 dt=0.004 | 0.129% | 0.057% | 0/32 | 6.608 | |
| CNAB2 dt=0.005 | 0.207% | 0.092% | 0/32 | 5.414 | |
| CNAB2 dt=0.01 chosen | 2.412% | 0.400% | 0/32 | 3.030 | |
| CNAB2 dt=0.02 | 257.380% | 15.076% | 23/32 | 1.839 | |

Paired speedup (chosen CNAB2 ms / coefficient ms): 0.467736.

Coefficient ROM per-time worst, including t=0:

| time index | worst |
|---:|---:|
| 0 | 0.091% |
| 1 | 4.799% |
| 2 | 4.199% |
| 3 | 3.448% |
| 4 | 2.953% |
| 5 | 2.696% |

