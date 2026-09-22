# pilot01 development pilot: the frame as an online unknown

Generated from `/tmp/claude-1002/-home-tahmid-Dev-pod-ae-nmrom-Tunable-NM-ROM-Claude/6777170d-0e90-4670-b7cb-4ee006705422/scratchpad/pulls/pilot01/summary.json` by `write_pilot_table.py`. Job 4176514, commit `76ecd68722b007065a192a05b87d51d9c73224f1`, NVIDIA A100-PCIE-40GB, GPU-a840f593-3e39-4ee0-681d-f3898b49b09b, 40960 MiB, device [CudaDevice(id=0)]. Development seed 202609202 only; the final cohort was not opened. Evolved worst is the worst case of the worst evolved time.

## Floors

| rank | A0 fixed uncentred bank | over 5% | A1 centered bank, oracle shift | over 5% |
|---:|---:|---:|---:|---:|
| 32 | 85.413% | 16/16 | 0.737% | 0/16 |
| 64 | 70.027% | 16/16 | 0.128% | 0/16 |
| 128 | 47.119% | 16/16 | 0.020% | 0/16 |

Centered POD available rank: 128. Worst centroid travel over the horizon: 0.049473 of the box.

## Harness checks

| check | value | meaning |
|---|---:|---|
| `delta == 0` vs `ns3d_rom.make_run` | 5.669e-15 | the co-moving residual reduces exactly to the existing weak ROM |
| complete-query equivariance | 1.120e-15 | the solved query on a shifted input is the shift of the solved query |
| advection tensor vs full grid | 1.452e-14 | |
| derivative projection vs full grid | 9.487e-15 | |
| diffusion identity | 4.586e-15 | `Phi^T Lap G == -diag(lam) A` |
| finite-difference shift sign | 3.709e-11 | independent sign check on `-D_d a` |
| `S_d` skew symmetry | 1.568e-14 | the phase row needs it |
| bank orthonormality | 1.332e-15 | |
| test orthogonality | 1.719e-15 | |

## Identifiability of the frame

| quantity | no gauge | with gauge |
|---|---:|---:|
| smallest / largest singular value of the full Jacobian | 3.526e-05 | 3.859e-05 |
| condition number of the full Jacobian | 2.836e+04 | 2.591e+04 |
| largest singular value of the coefficient block | 1.001e+00 | 1.421e+00 |
| **deflated** smallest sv of `(I - Ja Ja+) Jdelta`, over largest sv of `Ja` | 2.995e+01 | 5.073e+01 |
| analytic vs AD delta column | 5.612e-17 | 5.612e-17 |

Translation tangents `d_d(G a)` captured inside `span(G)`: 0.8177, 0.8447, 0.8889 (rank 3 of 3). The pre-registered identifiability threshold was a deflated ratio above 1e-6.

## Solved arms, development cohort

| arm | rank | M | dt | gauge | budget | evolved worst | evolved median | over 5% | median LM iters | on budget | centroid gap worst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 64 | 292 | 0.01 | 0.0 | 60 | 234.855% | 192.284% | 16/16 | 10.0 | 2.5% | 0.04647 |
| B1 | 64 | 292 | 0.01 | 0.0 | 60 | 0.449% | 0.286% | 0/16 | 3.0 | 0.0% | 0.00006 |
| B2 | 64 | 292 | 0.01 | 1.0 | 60 | 20.500% | 16.222% | 16/16 | 3.0 | 0.0% | 0.00606 |
| B2_M1024 | 64 | 1024 | 0.01 | 1.0 | 60 | 10.783% | 7.879% | 16/16 | 3.0 | 0.0% | 0.00201 |
| B2_M96 | 64 | 96 | 0.01 | 1.0 | 60 | 73.996% | 32.042% | 16/16 | 3.0 | 0.0% | 0.02970 |
| B2_b20 | 64 | 292 | 0.01 | 1.0 | 20 | 20.500% | 16.222% | 16/16 | 3.0 | 0.0% | 0.00606 |
| B2_dt0.005 | 64 | 292 | 0.005 | 1.0 | 60 | 25.341% | 16.077% | 16/16 | 3.0 | 0.0% | 0.00854 |
| B2_r128 | 128 | 292 | 0.01 | 1.0 | 60 | 24.764% | 17.121% | 16/16 | 3.0 | 0.0% | 0.00508 |
| B2_r32 | 32 | 292 | 0.01 | 1.0 | 60 | 22.411% | 12.843% | 16/16 | 3.0 | 0.0% | 0.00372 |
| B2_w0.1 | 64 | 292 | 0.01 | 0.1 | 60 | 10.320% | 6.703% | 15/16 | 3.0 | 0.0% | 0.00174 |
| B2_w10.0 | 64 | 292 | 0.01 | 10.0 | 60 | 20.802% | 16.428% | 16/16 | 3.0 | 0.0% | 0.00616 |
| C tracker | 64 | - | 0.01 | - | - | 2.538% | 1.661% | 0/16 | - | - | - |
| CNAB2 dt=0.002 | - | - | 0.002 | - | - | 0.019% | 0.011% | 0/16 | - | - | - |
| CNAB2 dt=0.004 | - | - | 0.004 | - | - | 0.094% | 0.055% | 0/16 | - | - | - |
| CNAB2 dt=0.005 | - | - | 0.005 | - | - | 0.152% | 0.089% | 0/16 | - | - | - |
| CNAB2 dt=0.01 | - | - | 0.01 | - | - | 0.788% | 0.382% | 0/16 | - | - | - |
| CNAB2 dt=0.02 | - | - | 0.02 | - | - | 76.694% | 9.942% | 10/16 | - | - | - |

## Complete-query cost, paired

complete queries; initial centering, projection and every output reconstruction are inside the timed call; interleaved repetitions after burn-in. Case 0, 7 retained repetitions.

| arm | median ms | min ms | max ms | worst evolved error of the timed output |
|---|---:|---:|---:|---:|
| B0 | 93.346 | 93.248 | 163.801 | 173.265% |
| B1 | 23.875 | 23.831 | 39.855 | 0.317% |
| B2 | 24.700 | 24.617 | 41.604 | 12.344% |
| CNAB2_dt0.002 | 12.818 | 12.805 | 20.818 | 0.010% |
| CNAB2_dt0.004 | 6.741 | 6.719 | 8.007 | 0.047% |
| CNAB2_dt0.005 | 5.537 | 5.520 | 5.773 | 0.076% |
| CNAB2_dt0.01 | 3.131 | 3.088 | 3.358 | 0.323% |
| CNAB2_dt0.02 | 1.919 | 1.883 | 1.941 | 2.570% |
| C_tracker | 14.865 | 14.834 | 21.639 | 2.111% |

- **B0**: comparator CNAB2_dt0.02 at 1.919 ms, paired speedup 0.021x (worst evolved 173.265% vs 2.570%).
- **B1**: comparator CNAB2_dt0.005 at 5.537 ms, paired speedup 0.232x (worst evolved 0.317% vs 0.076%).
- **B2**: comparator CNAB2_dt0.02 at 1.919 ms, paired speedup 0.078x (worst evolved 12.344% vs 2.570%).

## Independent verification

`verify_pilot.py` recomputed every saved arm from the fields in NumPy with a different reduction. Worst disagreement 8.882e-16.

