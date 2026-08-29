### JOB A — ladder on one GPU

Job 3033260, node pax050, GPU **NVIDIA A100 80GB PCIe**, commit 0d96222e49, jax 0.10.2, 7 timed reps + 2 burn, order: reps outermost; N ascending on even reps, descending on odd; arms inner; trajectories innermost; complete=True, 87 s.

| N | arm | ic ms | solve ms | dec ms | e2e ms | LM attempts / traj | us per LM attempt (median over traj) | err | vs committed err | committed solve ms (own job) |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | oracle | 6.46 | 21.42 | 0.47 | 28.75 | 217.2 | 105.1 | 5.009008e-03 | 4.4e-09 | 38.64 |
| 128 | base_tight | 6.23 | 16.47 | 0.47 | 23.74 | 215.0 | 82.7 | 5.480428e-03 | 3.8e-09 | 37.51 |
| 128 | tensor | 6.23 | 17.09 | 0.46 | 24.06 | 217.4 | 82.9 | 5.009175e-03 | nan | nan |
| 256 | oracle | 9.33 | 26.22 | 0.40 | 33.53 | 249.2 | 97.2 | 6.164550e-03 | 7.4e-09 | 47.55 |
| 256 | base_tight | 9.28 | 19.94 | 0.39 | 27.12 | 240.5 | 75.0 | 6.236229e-03 | 5.6e-09 | 45.87 |
| 256 | tensor | 9.29 | 20.21 | 0.38 | 28.14 | 249.2 | 76.2 | 6.164636e-03 | nan | nan |
| 512 | oracle | 7.66 | 21.69 | 0.39 | 31.18 | 220.8 | 112.7 | 4.876680e-03 | 2.7e-09 | 38.06 |
| 512 | base_tight | 7.65 | 14.77 | 0.39 | 25.72 | 212.1 | 79.8 | 5.194389e-03 | 3.1e-09 | 34.30 |
| 512 | tensor | 7.65 | 15.68 | 0.39 | 26.33 | 220.8 | 81.9 | 4.876705e-03 | nan | nan |
| 1024 | oracle | 6.85 | 21.26 | 0.47 | 27.56 | 202.5 | 112.9 | 5.537167e-03 | 5.7e-09 | 35.54 |
| 1024 | base_tight | 6.87 | 15.49 | 0.48 | 21.70 | 211.5 | 82.0 | 5.614597e-03 | 6.2e-09 | 34.25 |
| 1024 | tensor | 6.87 | 16.00 | 0.47 | 21.55 | 202.5 | 84.5 | 5.537195e-03 | nan | nan |
| 2048 | oracle | 8.29 | 20.43 | 0.48 | 28.79 | 203.8 | 116.4 | 5.078899e-03 | 6.2e-09 | 35.73 |
| 2048 | base_tight | 8.23 | 15.91 | 0.47 | 25.50 | 209.1 | 80.5 | 5.533699e-03 | 4.7e-09 | 36.54 |
| 2048 | tensor | 8.27 | 14.44 | 0.47 | 23.67 | 203.8 | 81.4 | 5.078901e-03 | nan | nan |
| 4096 | oracle | 6.91 | 20.84 | 0.41 | 27.61 | 201.2 | 119.5 | 4.485117e-03 | 4.5e-10 | 38.23 |
| 4096 | base_tight | 6.78 | 13.62 | 0.40 | 20.43 | 197.9 | 82.4 | 4.888627e-03 | 1.4e-10 | 32.71 |
| 4096 | tensor | 6.79 | 14.31 | 0.39 | 21.87 | 201.2 | 83.0 | 4.485115e-03 | nan | nan |

#### Tensor vs oracle inside JOB A

| N | max per-traj abs err diff | stop hist identical | LM attempt counts identical | solve ratio tensor/oracle | solve ratio tensor/NNLS-32 | e2e ratio tensor/oracle | e2e ratio tensor/NNLS-32 |
|---|---|---|---|---|---|---|---|
| 128 | 1.20e-06 | True | False | 0.797 | 1.037 | 0.837 | 1.013 |
| 256 | 6.40e-07 | True | True | 0.771 | 1.014 | 0.839 | 1.038 |
| 512 | 2.45e-07 | True | True | 0.723 | 1.061 | 0.845 | 1.024 |
| 1024 | 2.26e-07 | True | True | 0.753 | 1.033 | 0.782 | 0.993 |
| 2048 | 3.42e-08 | True | True | 0.707 | 0.908 | 0.822 | 0.928 |
| 4096 | 5.36e-09 | True | True | 0.687 | 1.051 | 0.792 | 1.070 |

#### Slopes from JOB A alone: exponent p in ms ~ N^p (least-squares fit of log ms vs log N)

| arm | N range | solve ms | us per LM attempt | ic ms | e2e ms |
|---|---|---|---|---|---|
| oracle | 128..4096 | -0.0374 | +0.0487 | -0.0055 | -0.0322 |
| base_tight | 128..4096 | -0.0652 | +0.0092 | -0.0020 | -0.0455 |
| tensor | 128..4096 | -0.0772 | +0.0095 | -0.0011 | -0.0493 |

### JOBs B/C — large N, each its own job and GPU (DIFFERENT JOB/GPU from JOB A: compare with care)

| N | job | node | GPU | arm | ic ms | solve ms | dec ms | e2e ms | err | stop hist | trained in-job (recon rel-L2 / train s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 16384 | 3033636 | pax007 | NVIDIA A100 80GB PCIe | oracle | 6.51 | 21.44 | 0.63 | 30.08 | 4.474627e-03 | {'0': 4, '2': 396} | 3.760e-03 / 283 |
| 16384 | 3033636 | pax007 | NVIDIA A100 80GB PCIe | base_tight | 6.49 | 13.90 | 0.63 | 23.14 | 4.799584e-03 | {'0': 2, '2': 398} | 3.760e-03 / 283 |
| 16384 | 3033636 | pax007 | NVIDIA A100 80GB PCIe | tensor | 6.48 | 14.18 | 0.63 | 23.49 | 4.474627e-03 | {'0': 4, '2': 396} | 3.760e-03 / 283 |
| 16384 | 3033636 | pax007 | NVIDIA A100 80GB PCIe | nodes_tight | 6.56 | 14.09 | 0.63 | 23.33 | 4.476662e-03 | {'0': 3, '2': 397} | 3.760e-03 / 283 |
| 16384 | 3033636 | pax007 | NVIDIA A100 80GB PCIe | tensor_nolean | 6.51 | 16.47 | 0.65 | 25.27 | 4.474627e-03 | {'0': 4, '2': 396} | 3.760e-03 / 283 |
| 65536 | 3033264 | pax105 | NVIDIA A100 80GB PCIe | oracle | 6.83 | 32.95 | 0.65 | 39.09 | 4.532280e-03 | {'2': 398, '0': 2} | 3.759e-03 / 1012 |
| 65536 | 3033264 | pax105 | NVIDIA A100 80GB PCIe | base_tight | 6.79 | 13.52 | 0.59 | 21.04 | 4.894651e-03 | {'2': 398, '0': 2} | 3.759e-03 / 1012 |
| 65536 | 3033264 | pax105 | NVIDIA A100 80GB PCIe | tensor | 6.81 | 13.82 | 0.61 | 21.15 | 4.532280e-03 | {'2': 398, '0': 2} | 3.759e-03 / 1012 |
| 65536 | 3033264 | pax105 | NVIDIA A100 80GB PCIe | nodes_tight | 6.87 | 15.76 | 0.60 | 24.00 | 4.612792e-03 | {'2': 398, '0': 2} | 3.759e-03 / 1012 |
| 65536 | 3033264 | pax105 | NVIDIA A100 80GB PCIe | tensor_nolean | 6.80 | 15.88 | 0.54 | 23.22 | 4.532280e-03 | {'2': 398, '0': 2} | 3.759e-03 / 1012 |

#### Large-N tensor vs oracle (within each job)

| N | max per-traj abs err diff | stop hist identical | latent dev max | solve ratio tensor/oracle | solve ratio tensor/NNLS-32 | e2e ratio tensor/oracle | e2e ratio tensor/NNLS-32 | FOM 1e-3 ms/traj (err) | FOM 1e-8 ms/traj (err) |
|---|---|---|---|---|---|---|---|---|---|
| 16384 | 5.76e-10 | True | 7.1e-07 | 0.661 | 1.020 | 0.781 | 1.015 | 9.28 (1.50e-04) | 15.87 (5.06e-10) |
| 65536 | 1.72e-10 | True | 3.9e-08 | 0.419 | 1.022 | 0.541 | 1.005 | 9.78 (1.51e-04) | 16.82 (5.08e-10) |

#### Large-N gates

| N | J (at N) | T2 (at N, traj) | E | F | C | G rel | V rel | TB | TA | T0 (n states) | TQ r rel med / max | oracle parity vs in-job scale run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16384 | 1.2e-16 (16384) | 3.8e-14 (16384, 1) | 2.7e-13 | 4.3e-15 | 1.2e-12 | 1.8e-14 | 2.3e-14 | 1.0e-15 | 7.7e-15 | 6.1e-15 (2897) | 3.4e-11 / 3.1e-06 | 3.6e-10 |
| 65536 | 1.2e-16 (16384) | 3.7e-14 (16384, 1) | 1.5e-12 | 2.5e-14 | 0.0e+00 | 5.5e-14 | 7.9e-14 | 1.7e-15 | 1.2e-14 | 8.4e-15 (2917) | 9.3e-12 / 7.9e-07 | 2.7e-09 |

### Slopes across the FULL range (JOB A points + large-N points from DIFFERENT jobs/GPUs — cross-job, indicative only)

| arm | N range | solve ms exponent | e2e ms exponent | solve ms at each N |
|---|---|---|---|---|
| oracle | 128..65536 | +0.0350 | +0.0248 | 128:21.4, 256:26.2, 512:21.7, 1024:21.3, 2048:20.4, 4096:20.8, 16384:21.4, 65536:32.9 |
| base_tight | 128..65536 | -0.0456 | -0.0292 | 128:16.5, 256:19.9, 512:14.8, 1024:15.5, 2048:15.9, 4096:13.6, 16384:13.9, 65536:13.5 |
| tensor | 128..65536 | -0.0479 | -0.0304 | 128:17.1, 256:20.2, 512:15.7, 1024:16.0, 2048:14.4, 4096:14.3, 16384:14.2, 65536:13.8 |

#### Same, large-N jobs only (N=16384..65536, two separate jobs)

| arm | solve ms exponent | e2e ms exponent |
|---|---|---|
| oracle | +0.3101 | +0.1891 |
| base_tight | -0.0197 | -0.0683 |
| tensor | -0.0186 | -0.0756 |
