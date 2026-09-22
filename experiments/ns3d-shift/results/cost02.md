# cost02: where the co-moving query's time goes

Generated from `/tmp/claude-1002/-home-tahmid-Dev-pod-ae-nmrom-Tunable-NM-ROM-Claude/6777170d-0e90-4670-b7cb-4ee006705422/scratchpad/pulls/cost02/summary.json` by `write_cost_table.py`. Job 4176596, commit `dc8eb9f7b274736f0134742345ae32243fe44051`, NVIDIA A100-PCIE-40GB, GPU-a840f593-3e39-4ee0-681d-f3898b49b09b, 40960 MiB, device [CudaDevice(id=0)]. Rank 64, gauge 0.0, development seed 202609202 only. Accuracy is the full 16-case cohort; timing is case 0, 7 interleaved repetitions after burn-in.

## Grid-sized pieces the ROM cannot avoid

| piece | median ms |
|---|---:|
| initial centering projection | 0.264 |
| one output reconstruction | 0.171 |

## CNAB2 comparators, same cohort and allocation

| dt | evolved worst | evolved median | over 5% | median ms |
|---:|---:|---:|---:|---:|
| 0.004 | 0.094% | 0.055% | 0/16 | 6.738 |
| 0.005 | 0.152% | 0.089% | 0/16 | 5.531 |
| 0.01 | 0.788% | 0.382% | 0/16 | 3.128 |
| 0.02 | 76.694% | 9.942% | 10/16 | 1.932 |

## Co-moving ROM: accuracy and complete-query cost

| M | dt | steps | evolved worst | over 5% | median LM iters | query ms | one-output ms | outputs ms | comparator | paired speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 96 | 0.04 | 5 | 4.751% | 0/16 | 4.0 | 13.446 | 12.644 | 0.802 | CNAB2 dt=0.01 | 0.233x |
| 96 | 0.02 | 10 | 3.427% | 0/16 | 4.0 | 14.267 | 13.613 | 0.654 | CNAB2 dt=0.01 | 0.219x |
| 96 | 0.01 | 20 | 3.891% | 0/16 | 3.0 | 21.566 | 20.856 | 0.710 | CNAB2 dt=0.01 | 0.145x |
| 292 | 0.04 | 5 | 2.164% | 0/16 | 4.0 | 8.282 | 7.694 | 0.587 | CNAB2 dt=0.01 | 0.378x |
| 292 | 0.02 | 10 | 0.639% | 0/16 | 3.0 | 13.073 | 12.414 | 0.659 | CNAB2 dt=0.005 | 0.423x |
| 292 | 0.01 | 20 | 0.449% | 0/16 | 3.0 | 23.918 | 23.103 | 0.815 | CNAB2 dt=0.005 | 0.231x |
| 1024 | 0.04 | 5 | 1.181% | 0/16 | 4.0 | 9.736 | 9.283 | 0.454 | CNAB2 dt=0.01 | 0.321x |
| 1024 | 0.02 | 10 | 0.337% | 0/16 | 3.0 | 16.241 | 15.621 | 0.621 | CNAB2 dt=0.005 | 0.341x |
| 1024 | 0.01 | 20 | 0.217% | 0/16 | 3.0 | 30.256 | 29.542 | 0.714 | CNAB2 dt=0.005 | 0.183x |

The *outputs* column is the whole query minus the same trajectory asked for one output instead of five, so it isolates the four extra laboratory-frame reconstructions. The comparator is the fastest tested CNAB2 step whose evolved worst is no larger than that row's.

## Independent verification

`verify_cost.py` recomputed every saved setting from its fields in NumPy with a different reduction. Worst disagreement 1.735e-17.

