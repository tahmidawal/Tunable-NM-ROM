# mesh03: where the co-moving query's time goes

Generated from `/tmp/claude-1002/-home-tahmid-Dev-pod-ae-nmrom-Tunable-NM-ROM-Claude/6777170d-0e90-4670-b7cb-4ee006705422/scratchpad/pulls/mesh03/summary.json` by `write_cost_table.py`. Job 4176669, commit `cd44b0e0e2d1e5a45fa471939b6fc285af001f51`, NVIDIA A100 80GB PCIe, GPU-eb7290b8-220f-9aa0-5ebb-01d7b49f2bb5, 81920 MiB, device [CudaDevice(id=0)]. Rank 64, gauge 0.0, development seed 202609202 only. Accuracy is the full 16-case cohort; timing is case 0, 7 interleaved repetitions after burn-in.

## Grid-sized pieces the ROM cannot avoid

| piece | median ms |
|---|---:|
| initial centering projection | 0.542 |
| one output reconstruction | 0.449 |

## CNAB2 comparators, same cohort and allocation

| dt | evolved worst | evolved median | over 5% | median ms |
|---:|---:|---:|---:|---:|
| 0.004 | 0.093% | 0.055% | 0/16 | 23.658 |
| 0.005 | 0.382% | 0.088% | 0/16 | 19.251 |
| 0.01 | 686.143% | 2.746% | 6/16 | 10.410 |
| 0.02 | 272.038% | 19.068% | 12/16 | 5.741 |

## Co-moving ROM: accuracy and complete-query cost

| M | dt | steps | evolved worst | over 5% | median LM iters | query ms | one-output ms | outputs ms | comparator | paired speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 292 | 0.04 | 5 | 2.073% | 0/16 | 4.0 | 10.831 | 8.561 | 2.270 | CNAB2 dt=0.005 | 1.777x |
| 292 | 0.02 | 10 | 0.644% | 0/16 | 3.0 | 15.116 | 13.101 | 2.015 | CNAB2 dt=0.005 | 1.274x |
| 292 | 0.01 | 20 | 0.494% | 0/16 | 3.0 | 25.910 | 23.845 | 2.064 | CNAB2 dt=0.005 | 0.743x |

The *outputs* column is the whole query minus the same trajectory asked for one output instead of five, so it isolates the four extra laboratory-frame reconstructions. The comparator is the fastest tested CNAB2 step whose evolved worst is no larger than that row's.

## Independent verification

`verify_cost.py` recomputed every saved setting from its fields in NumPy with a different reduction. Worst disagreement 3.469e-18.

