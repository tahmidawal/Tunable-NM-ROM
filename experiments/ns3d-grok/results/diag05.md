# diag05 sealed center tracker

Generated from `runs/diag05/output/summary.json`. Job 4142139, commit `9c40c70c166fd29906164da86c9c0087eefa9319`. Device: [CudaDevice(id=0)]. Seed 202609203, 32 cases, opened once. Setting frozen before this job: centered POD rank 64, ROM-centroid shift every startup step, dt=0.01. Error and time for each method come from the same timed calls. The comparator is the fastest tested CNAB2 step whose evolved worst is no larger than the tracker's.

| method | evolved worst | evolved median | cases over 5% | median ms |
|---|---:|---:|---:|---:|
| tracked ROM | 4.606% | 3.335% | 0/32 | 15.704 |
| CNAB2 dt=0.004 | 0.118% | 0.055% | 0/32 | 6.825 |
| CNAB2 dt=0.005 | 0.190% | 0.089% | 0/32 | 5.653 |
| CNAB2 dt=0.01 chosen | 2.471% | 0.387% | 0/32 | 3.173 |
| CNAB2 dt=0.02 | 271.503% | 7.854% | 21/32 | 1.945 |

Paired speedup (chosen CNAB2 ms / tracked ms): 0.202034.

Tracked per-time worst, including t=0:

| time index | worst |
|---:|---:|
| 0 | 0.048% |
| 1 | 4.606% |
| 2 | 4.029% |
| 3 | 3.381% |
| 4 | 2.956% |
| 5 | 2.676% |

