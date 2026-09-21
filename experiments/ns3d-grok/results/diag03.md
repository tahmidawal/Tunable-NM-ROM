# diag03 every-step center tracking

Generated from `runs/diag03/output/summary.json`. Job 4141126, commit `70f1573ac86a327204fdbd47a960040cdc0b9d3a`. Development only. `rom` recenters from the decoded field. `truth` recenters from the true field and is not a model. `interval` recenters only at the saved output times. Each substep is the Galerkin startup step, not multistep CNAB2. The timing column is that Python loop, not a fused query.

| rank | mode | dt | evolved worst | cases over 5% |
|---:|---|---:|---:|---:|
| 64 | interval | 0.004 | 11.626% | 16/16 |
| 64 | interval | 0.01 | 11.608% | 16/16 |
| 64 | rom | 0.004 | 1.975% | 0/16 |
| 64 | rom | 0.01 | 4.563% | 0/16 |
| 64 | truth | 0.004 | 0.888% | 0/16 |
| 64 | truth | 0.01 | 2.243% | 0/16 |
| 128 | interval | 0.004 | 6.898% | 11/16 |
| 128 | interval | 0.01 | 6.893% | 11/16 |
| 128 | rom | 0.004 | 0.809% | 0/16 |
| 128 | rom | 0.01 | 2.040% | 0/16 |
| 128 | truth | 0.004 | 0.634% | 0/16 |
| 128 | truth | 0.01 | 1.610% | 0/16 |

| arm | median ms |
|---|---:|
| Python ROM tracker rank 64 dt=0.004 | 395.549 |
| Python ROM tracker rank 64 dt=0.01 | 180.909 |

