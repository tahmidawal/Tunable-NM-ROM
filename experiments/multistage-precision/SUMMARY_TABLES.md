### (A0) single-function fit — N=128, tanh 64x3, Adam 20000 + L-BFGS 2000

| stage | n_freq | eps_in | rel residual after | secs |
|---|---|---|---|---|
| 0 | 8 | 2.28e-03 | 3.07e-04 | 306 |
| 1 | 30 | 7.00e-07 | 2.63e-06 | 309 |
| 2 | 63 | 6.00e-09 | 1.36e-08 | 403 |
| 3 | 63 | 3.10e-11 | 3.33e-11 | 394 |

### (1) true-z parametric decoder — ./runs/control/c1
N=64, n_train 1, n_val 16, 20000 Adam steps/stage, batch 1, P_SUB 0, hidden 128x4, const_lr 0, lbfgs 0, z_ff 0, seed 0

| stage | n_freq | f_d (cyc) | eps_in | TRAIN global / mean-rel | VAL global / mean-rel | VAL by amp quartile (low..high) | final batch loss |
|---|---|---|---|---|---|---|---|
| 0 | 6 | 1.0 | 1.16e-03 | 1.84e-04 / 1.84e-04 | 9.80e-01 / 1.18e+00 | 1.9e+00 / 9.0e-01 / 1.0e+00 / 8.8e-01 | 3.4e-08 |
| 1 | 28 | 12.0 | 2.13e-07 | 7.58e-11 / 7.58e-11 | 9.80e-01 / 1.18e+00 | 1.9e+00 / 9.0e-01 / 1.0e+00 / 8.8e-01 | 1.7e-13 |

### (1) true-z parametric decoder — ./runs/control/c16
N=64, n_train 16, n_val 16, 20000 Adam steps/stage, batch 16, P_SUB 1024, hidden 128x4, const_lr 0, lbfgs 300, z_ff 0, seed 0

| stage | n_freq | f_d (cyc) | eps_in | TRAIN global / mean-rel | VAL global / mean-rel | VAL by amp quartile (low..high) | final batch loss |
|---|---|---|---|---|---|---|---|
| 0 | 6 | 1.0 | 3.39e-03 | 8.87e-04 / 9.97e-04 | 6.28e-01 / 5.14e-01 | 4.2e-01 / 4.1e-01 / 6.6e-01 / 5.7e-01 | 3.4e-08 |
| 1 | 22 | 9.0 | 3.01e-06 | 1.65e-05 / 1.62e-05 | 6.27e-01 / 5.14e-01 | 4.2e-01 / 4.1e-01 / 6.6e-01 / 5.7e-01 | 1.3e-05 |

### (1) true-z parametric decoder — ./runs/control/c16_2x
N=64, n_train 16, n_val 16, 40000 Adam steps/stage, batch 16, P_SUB 0, hidden 128x4, const_lr 1, lbfgs 0, z_ff 0, seed 0

| stage | n_freq | f_d (cyc) | eps_in | TRAIN global / mean-rel | VAL global / mean-rel | VAL by amp quartile (low..high) | final batch loss |
|---|---|---|---|---|---|---|---|
| 0 | 6 | 1.0 | 3.39e-03 | 2.26e-03 / 2.49e-03 | 5.68e-01 / 5.09e-01 | 5.9e-01 / 3.3e-01 / 6.0e-01 / 5.2e-01 | 2.0e-07 |
| 1 | 18 | 7.0 | 7.67e-06 | 7.42e-05 / 5.79e-05 | 5.67e-01 / 5.09e-01 | 5.9e-01 / 3.3e-01 / 6.0e-01 / 5.2e-01 | 2.9e-05 |

### (1) true-z parametric decoder — ./runs/control/c16_lbfgs
N=64, n_train 16, n_val 16, 20000 Adam steps/stage, batch 16, P_SUB 0, hidden 128x4, const_lr 0, lbfgs 300, z_ff 0, seed 0

| stage | n_freq | f_d (cyc) | eps_in | TRAIN global / mean-rel | VAL global / mean-rel | VAL by amp quartile (low..high) | final batch loss |
|---|---|---|---|---|---|---|---|
| 0 | 6 | 1.0 | 3.39e-03 | 8.29e-04 / 9.37e-04 | 6.25e-01 / 5.24e-01 | 4.5e-01 / 4.2e-01 / 6.7e-01 / 5.6e-01 | 3.1e-08 |
| 1 | 22 | 9.0 | 2.81e-06 | 1.60e-05 / 1.54e-05 | 6.25e-01 / 5.24e-01 | 4.5e-01 / 4.2e-01 / 6.7e-01 / 5.6e-01 | 1.3e-05 |

### (1) true-z parametric decoder — ./runs/control/c256
N=64, n_train 256, n_val 16, 20000 Adam steps/stage, batch 256, P_SUB 1024, hidden 128x4, const_lr 0, lbfgs 0, z_ff 0, seed 0

| stage | n_freq | f_d (cyc) | eps_in | TRAIN global / mean-rel | VAL global / mean-rel | VAL by amp quartile (low..high) | final batch loss |
|---|---|---|---|---|---|---|---|
| 0 | 6 | 1.0 | 2.39e-03 | 2.65e-03 / 2.74e-03 | 5.64e-03 / 9.55e-03 | 2.3e-02 / 7.2e-03 / 3.8e-03 / 4.5e-03 | 3.7e-07 |
| 1 | 6 | 1.0 | 6.34e-06 | 6.20e-04 / 6.42e-04 | 5.47e-03 / 9.96e-03 | 2.4e-02 / 7.3e-03 / 3.7e-03 / 4.3e-03 | 2.9e-03 |

### (1) true-z parametric decoder — ./runs/control/c256_bw
N=64, n_train 256, n_val 16, 20000 Adam steps/stage, batch 256, P_SUB 1024, hidden 128x4, const_lr 0, lbfgs 0, z_ff 0, seed 0

| stage | n_freq | f_d (cyc) | eps_in | TRAIN global / mean-rel | VAL global / mean-rel | VAL by amp quartile (low..high) | final batch loss |
|---|---|---|---|---|---|---|---|
| 0 | 32 | 1.0 | 2.39e-03 | 2.14e-03 / 2.13e-03 | 3.84e-03 / 5.79e-03 | 1.3e-02 / 4.2e-03 / 2.7e-03 / 3.1e-03 | 2.2e-07 |
| 1 | 32 | 1.0 | 5.11e-06 | 7.01e-04 / 7.19e-04 | 3.71e-03 / 5.70e-03 | 1.3e-02 / 4.1e-03 / 2.4e-03 / 3.1e-03 | 5.5e-03 |

### (1) true-z parametric decoder — ./runs/control/c4
N=64, n_train 4, n_val 16, 20000 Adam steps/stage, batch 4, P_SUB 0, hidden 128x4, const_lr 0, lbfgs 0, z_ff 0, seed 0

| stage | n_freq | f_d (cyc) | eps_in | TRAIN global / mean-rel | VAL global / mean-rel | VAL by amp quartile (low..high) | final batch loss |
|---|---|---|---|---|---|---|---|
| 0 | 6 | 1.0 | 7.74e-04 | 3.74e-04 / 3.83e-04 | 8.01e-01 / 8.93e-01 | 1.2e+00 / 8.0e-01 / 8.8e-01 / 7.4e-01 | 9.6e-08 |
| 1 | 26 | 11.0 | 2.90e-07 | 2.58e-07 / 2.29e-07 | 8.01e-01 / 8.93e-01 | 1.2e+00 / 8.0e-01 / 8.8e-01 / 7.4e-01 | 2.6e-07 |

### (1) true-z parametric decoder — ./runs/control/c64
N=64, n_train 64, n_val 16, 20000 Adam steps/stage, batch 64, P_SUB 0, hidden 128x4, const_lr 0, lbfgs 0, z_ff 0, seed 0

| stage | n_freq | f_d (cyc) | eps_in | TRAIN global / mean-rel | VAL global / mean-rel | VAL by amp quartile (low..high) | final batch loss |
|---|---|---|---|---|---|---|---|
| 0 | 6 | 1.0 | 2.63e-03 | 1.90e-03 / 1.95e-03 | 1.12e-01 / 1.10e-01 | 1.9e-01 / 8.4e-02 / 1.1e-01 / 6.0e-02 | 2.0e-07 |
| 1 | 16 | 6.0 | 4.99e-06 | 1.45e-04 / 1.34e-04 | 1.13e-01 / 1.10e-01 | 1.9e-01 / 8.4e-02 / 1.1e-01 / 6.0e-02 | 2.6e-04 |

### (1) true-z parametric decoder — ./runs/control/c64_bw
N=64, n_train 64, n_val 16, 20000 Adam steps/stage, batch 64, P_SUB 0, hidden 128x4, const_lr 0, lbfgs 0, z_ff 0, seed 0

| stage | n_freq | f_d (cyc) | eps_in | TRAIN global / mean-rel | VAL global / mean-rel | VAL by amp quartile (low..high) | final batch loss |
|---|---|---|---|---|---|---|---|
| 0 | 32 | 1.0 | 2.63e-03 | 1.66e-03 / 1.62e-03 | 7.46e-02 / 7.55e-02 | 1.0e-01 / 7.6e-02 / 8.5e-02 / 3.8e-02 | 1.4e-07 |
| 1 | 38 | 4.0 | 4.36e-06 | 2.93e-04 / 2.78e-04 | 7.45e-02 / 7.56e-02 | 1.0e-01 / 7.6e-02 / 8.6e-02 / 3.8e-02 | 1.4e-03 |

### (1) true-z parametric decoder — ./runs/control/c64_lbfgs
N=64, n_train 64, n_val 16, 20000 Adam steps/stage, batch 64, P_SUB 0, hidden 128x4, const_lr 0, lbfgs 300, z_ff 0, seed 0

| stage | n_freq | f_d (cyc) | eps_in | TRAIN global / mean-rel | VAL global / mean-rel | VAL by amp quartile (low..high) | final batch loss |
|---|---|---|---|---|---|---|---|
| 0 | 6 | 1.0 | 2.63e-03 | 1.80e-03 / 1.86e-03 | 1.11e-01 / 1.11e-01 | 1.9e-01 / 8.4e-02 / 1.1e-01 / 5.9e-02 | 2.0e-07 |
| 1 | 16 | 6.0 | 4.72e-06 | 1.28e-04 / 1.19e-04 | 1.12e-01 / 1.11e-01 | 1.9e-01 / 8.4e-02 / 1.1e-01 / 6.0e-02 | 2.6e-04 |

### (1) true-z parametric decoder — ./runs/par512
N=64, n_train 512, n_val 64, 20000 Adam steps/stage, batch 32, P_SUB 1024, hidden 128x4, const_lr 0, lbfgs 0, z_ff 0, seed 0

| stage | n_freq | f_d (cyc) | eps_in | TRAIN global / mean-rel | VAL global / mean-rel | VAL by amp quartile (low..high) | final batch loss |
|---|---|---|---|---|---|---|---|
| 0 | 6 | 1.0 | 2.36e-03 | 3.96e-03 / 3.69e-03 | 6.08e-03 / 4.94e-03 | 6.1e-03 / 4.3e-03 / 4.7e-03 / 4.7e-03 | 4.6e-07 |
| 1 | 6 | 1.0 | 9.35e-06 | 1.48e-03 / 1.54e-03 | 5.86e-03 / 4.51e-03 | 5.7e-03 / 4.0e-03 / 4.4e-03 / 3.9e-03 | 5.4e-03 |
| 2 | 6 | 1.0 | 3.48e-06 | 1.15e-03 / 1.16e-03 | 5.89e-03 / 4.52e-03 | 5.8e-03 / 4.1e-03 / 4.4e-03 / 3.8e-03 | 2.2e-02 |

### (1) true-z parametric decoder — ./runs/par512_bw
N=64, n_train 512, n_val 64, 20000 Adam steps/stage, batch 32, P_SUB 1024, hidden 128x4, const_lr 0, lbfgs 0, z_ff 0, seed 0

| stage | n_freq | f_d (cyc) | eps_in | TRAIN global / mean-rel | VAL global / mean-rel | VAL by amp quartile (low..high) | final batch loss |
|---|---|---|---|---|---|---|---|
| 0 | 32 | 1.0 | 2.36e-03 | 3.93e-03 / 3.51e-03 | 5.43e-03 / 4.69e-03 | 5.6e-03 / 4.3e-03 / 4.4e-03 / 4.5e-03 | 4.2e-07 |
| 1 | 32 | 1.0 | 9.26e-06 | 1.74e-03 / 1.69e-03 | 4.90e-03 / 4.18e-03 | 5.2e-03 / 3.9e-03 / 3.9e-03 / 3.7e-03 | 6.3e-03 |
| 2 | 32 | 1.0 | 4.12e-06 | 1.46e-03 / 1.41e-03 | 4.84e-03 / 4.12e-03 | 5.2e-03 / 3.8e-03 / 3.9e-03 / 3.6e-03 | 2.3e-02 |

### (1) true-z parametric decoder — ./runs/par512_full
N=64, n_train 512, n_val 64, 20000 Adam steps/stage, batch 512, P_SUB 256, hidden 128x4, const_lr 0, lbfgs 0, z_ff 0, seed 0

| stage | n_freq | f_d (cyc) | eps_in | TRAIN global / mean-rel | VAL global / mean-rel | VAL by amp quartile (low..high) | final batch loss |
|---|---|---|---|---|---|---|---|
| 0 | 6 | 1.0 | 2.36e-03 | 2.83e-03 / 2.83e-03 | 5.22e-03 / 4.39e-03 | 5.6e-03 / 4.2e-03 / 4.1e-03 / 3.7e-03 | 3.0e-07 |
| 1 | 6 | 1.0 | 6.68e-06 | 8.98e-04 / 9.40e-04 | 5.13e-03 / 4.10e-03 | 5.6e-03 / 4.0e-03 / 3.7e-03 / 3.1e-03 | 3.7e-03 |

### (1) true-z parametric decoder — ./runs/par512_zff
N=64, n_train 512, n_val 64, 20000 Adam steps/stage, batch 32, P_SUB 1024, hidden 128x4, const_lr 0, lbfgs 0, z_ff 4, seed 0

| stage | n_freq | f_d (cyc) | eps_in | TRAIN global / mean-rel | VAL global / mean-rel | VAL by amp quartile (low..high) | final batch loss |
|---|---|---|---|---|---|---|---|
| 0 | 6 | 1.0 | 2.36e-03 | 5.17e-03 / 4.78e-03 | 3.08e-02 / 2.98e-02 | 3.8e-02 / 3.0e-02 / 2.8e-02 / 2.3e-02 | 8.1e-07 |
| 1 | 6 | 1.0 | 1.22e-05 | 1.75e-03 / 1.72e-03 | 3.16e-02 / 3.08e-02 | 3.9e-02 / 3.0e-02 / 3.0e-02 / 2.4e-02 | 3.8e-03 |

### (2) auto-decoder K_LAT=4 — ./runs/autodec/ad_K4
N=64, n_train 512, n_val 64, 20000 Adam steps/stage, batch 32, P_SUB 1024, hidden 128x4, const_lr 0, lbfgs 0, z_ff 0, seed 0; LM budget 60 attempts, m_eq 512, n_test 16

| stage | n_freq | eps_in | TRAIN global / mean-rel (learned latents) | VAL stage-0 latents fixed | VAL LM-inferred best / mean-start / nearest-start | VAL by amp quartile |
|---|---|---|---|---|---|---|
| 0 | 6 | 2.36e-03 | 1.59e-02 / 1.48e-02 | 2.03e-02 | 2.03e-02 / 7.84e-02 / 2.81e-02 | 2.5e-02 / 1.7e-02 / 2.1e-02 / 1.9e-02 |
| 1 | 8 | 3.76e-05 | 6.34e-03 / 6.32e-03 | 2.36e-02 | 2.15e-02 / 7.95e-02 / 3.04e-02 | 2.5e-02 / 1.9e-02 / 2.2e-02 / 1.9e-02 |
| 2 | 8 | 1.50e-05 | 4.88e-03 / 5.02e-03 | 2.35e-02 | 2.13e-02 / 7.99e-02 / 3.02e-02 | 2.5e-02 / 1.9e-02 / 2.2e-02 / 1.9e-02 |

(stage-0 Adam-inferred val latents, secondary: 6.25e-02)

ROM (LM Gauss-Newton on the ghost-zero FD residual, held-out sources):

| colloc | stages | init | ROM rel-L2 mean / med / max | oracle (same start) | oracle best-of | ‖r‖ LM / oracle / ‖f‖ | bnd block | acc/rej | reasons | z-norm LM / oracle | NN-lat dist |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 1 | mean | 4.84e-01 / 3.92e-01 / 9.60e-01 | 7.71e-02 | 1.86e-02 | 3.4e+00 / 3.4e+00 / 5.1e+00 | 8.5e-05 | 36/7 | {'budget': 7, 'converged': 9} | 0.31 / 0.33 | 0.08 |
| full | 1 | nearest | 2.13e-01 / 1.70e-01 / 8.39e-01 | 3.58e-02 | 1.86e-02 | 1.9e+00 / 2.4e+00 / 5.1e+00 | 8.3e-05 | 23/0 | {'budget': 2, 'converged': 14} | 0.30 / 0.32 | 0.06 |
| full | 2 | mean | 6.18e-01 / 6.87e-01 / 9.57e-01 | 5.08e-02 | 2.11e-02 | 4.5e+00 / 3.1e+00 / 5.1e+00 | 6.2e-05 | 55/0 | {'budget': 10, 'converged': 6} | 0.25 / 0.30 | 0.08 |
| full | 2 | nearest | 2.16e-01 / 1.31e-01 / 7.70e-01 | 4.00e-02 | 2.11e-02 | 2.7e+00 / 3.0e+00 / 5.1e+00 | 7.8e-05 | 45/0 | {'budget': 4, 'converged': 12} | 0.26 / 0.30 | 0.05 |
| full | 2 | staged | 2.66e-01 / 2.35e-01 / 8.30e-01 | 4.00e-02 | 2.11e-02 | 2.3e+00 / 3.0e+00 / 5.1e+00 | 8.2e-05 | 30/0 | {'budget': 10, 'converged': 6} | 0.27 / 0.30 | 0.05 |
| full | 3 | mean | 6.15e-01 / 6.77e-01 / 9.62e-01 | 5.08e-02 | 2.10e-02 | 4.5e+00 / 3.1e+00 / 5.1e+00 | 6.6e-05 | 60/0 | {'budget': 11, 'converged': 5} | 0.26 / 0.30 | 0.07 |
| full | 3 | nearest | 2.18e-01 / 1.45e-01 / 7.74e-01 | 3.98e-02 | 2.10e-02 | 2.7e+00 / 3.0e+00 / 5.1e+00 | 6.1e-05 | 46/0 | {'budget': 6, 'converged': 10} | 0.26 / 0.30 | 0.05 |
| full | 3 | staged | 2.67e-01 / 2.32e-01 / 8.30e-01 | 3.98e-02 | 2.10e-02 | 2.3e+00 / 3.0e+00 / 5.1e+00 | 6.4e-05 | 30/0 | {'budget': 12, 'converged': 4} | 0.27 / 0.30 | 0.05 |
| m512 | 1 | mean | 7.04e-01 / 7.94e-01 / 9.68e-01 | 7.71e-02 | 1.86e-02 | 1.6e+00 / 1.2e+00 / 1.7e+00 | 6.4e-05 | 36/7 | {'budget': 6, 'converged': 10} | 0.28 / 0.33 | 0.12 |
| m512 | 1 | nearest | 3.06e-01 / 2.98e-01 / 7.82e-01 | 3.58e-02 | 1.86e-02 | 6.3e-01 / 8.9e-01 / 1.7e+00 | 8.3e-05 | 24/0 | {'budget': 2, 'converged': 14} | 0.28 / 0.32 | 0.07 |
| m512 | 2 | mean | 7.71e-01 / 8.22e-01 / 9.69e-01 | 5.08e-02 | 2.11e-02 | 1.7e+00 / 1.2e+00 / 1.7e+00 | 5.6e-05 | 37/20 | {'budget': 11, 'converged': 5} | 0.18 / 0.30 | 0.08 |
| m512 | 2 | nearest | 4.02e-01 / 2.39e-01 / 2.22e+00 | 4.00e-02 | 2.11e-02 | 1.0e+00 / 1.2e+00 / 1.7e+00 | 9.1e-05 | 36/0 | {'budget': 3, 'converged': 13} | 0.25 / 0.30 | 0.05 |
| m512 | 2 | staged | 3.64e-01 / 3.71e-01 / 8.38e-01 | 4.00e-02 | 2.11e-02 | 7.6e-01 / 1.2e+00 / 1.7e+00 | 8.3e-05 | 26/0 | {'budget': 9, 'converged': 7} | 0.26 / 0.30 | 0.06 |
| m512 | 3 | mean | 7.70e-01 / 8.25e-01 / 9.41e-01 | 5.08e-02 | 2.10e-02 | 1.7e+00 / 1.2e+00 / 1.7e+00 | 5.4e-05 | 38/19 | {'budget': 10, 'converged': 6} | 0.17 / 0.30 | 0.08 |
| m512 | 3 | nearest | 3.21e-01 / 2.17e-01 / 9.59e-01 | 3.98e-02 | 2.10e-02 | 1.0e+00 / 1.2e+00 / 1.7e+00 | 6.9e-05 | 36/0 | {'budget': 8, 'converged': 8} | 0.26 / 0.30 | 0.05 |
| m512 | 3 | staged | 3.64e-01 / 3.67e-01 / 8.45e-01 | 3.98e-02 | 2.10e-02 | 7.7e-01 / 1.2e+00 / 1.7e+00 | 7.2e-05 | 30/0 | {'budget': 10, 'converged': 6} | 0.27 / 0.30 | 0.06 |

### (2) auto-decoder K_LAT=4 — ./runs/autodec/ad_K4_zff
N=64, n_train 512, n_val 64, 20000 Adam steps/stage, batch 32, P_SUB 1024, hidden 128x4, const_lr 0, lbfgs 0, z_ff 4, seed 0; LM budget 60 attempts, m_eq 512, n_test 16

| stage | n_freq | eps_in | TRAIN global / mean-rel (learned latents) | VAL stage-0 latents fixed | VAL LM-inferred best / mean-start / nearest-start | VAL by amp quartile |
|---|---|---|---|---|---|---|
| 0 | 6 | 2.36e-03 | 8.69e-03 / 9.10e-03 | 3.24e-02 | 3.24e-02 / 1.18e-01 / 4.22e-02 | 4.4e-02 / 3.8e-02 / 2.5e-02 / 2.2e-02 |
| 1 | 8 | 2.05e-05 | 3.16e-03 / 3.07e-03 | 3.32e-02 | 3.10e-02 / 1.28e-01 / 4.19e-02 | 4.1e-02 / 3.8e-02 / 2.4e-02 / 2.1e-02 |
| 2 | 16 | 7.45e-06 | 2.14e-03 / 2.09e-03 | 3.33e-02 | 3.09e-02 / 1.19e-01 / 4.19e-02 | 3.9e-02 / 3.9e-02 / 2.4e-02 / 2.1e-02 |

(stage-0 Adam-inferred val latents, secondary: 4.02e-02)

ROM (LM Gauss-Newton on the ghost-zero FD residual, held-out sources):

| colloc | stages | init | ROM rel-L2 mean / med / max | oracle (same start) | oracle best-of | ‖r‖ LM / oracle / ‖f‖ | bnd block | acc/rej | reasons | z-norm LM / oracle | NN-lat dist |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 1 | mean | 4.27e-01 / 2.18e-01 / 9.10e-01 | 1.18e-01 | 3.13e-02 | 3.1e+00 / 3.3e+00 / 5.1e+00 | 1.2e-04 | 36/11 | {'budget': 7, 'converged': 9} | 0.24 / 0.30 | 0.07 |
| full | 1 | nearest | 1.69e-01 / 1.18e-01 / 5.95e-01 | 4.11e-02 | 3.13e-02 | 1.6e+00 / 2.1e+00 / 5.1e+00 | 1.3e-04 | 20/0 | {'converged': 16} | 0.29 / 0.30 | 0.04 |
| full | 2 | mean | 4.72e-01 / 3.18e-01 / 9.61e-01 | 1.57e-01 | 2.85e-02 | 3.2e+00 / 3.4e+00 / 5.1e+00 | 8.5e-05 | 44/2 | {'budget': 6, 'converged': 10} | 0.25 / 0.36 | 0.06 |
| full | 2 | nearest | 1.62e-01 / 1.40e-01 / 5.28e-01 | 3.86e-02 | 2.85e-02 | 1.8e+00 / 2.1e+00 / 5.1e+00 | 1.0e-04 | 36/0 | {'budget': 5, 'converged': 11} | 0.30 / 0.30 | 0.04 |
| full | 2 | staged | 1.73e-01 / 1.40e-01 / 6.01e-01 | 3.86e-02 | 2.85e-02 | 1.8e+00 / 2.1e+00 / 5.1e+00 | 1.0e-04 | 28/0 | {'budget': 8, 'converged': 8} | 0.30 / 0.30 | 0.04 |
| full | 3 | mean | 5.02e-01 / 4.02e-01 / 9.61e-01 | 1.35e-01 | 2.71e-02 | 3.6e+00 / 3.4e+00 / 5.1e+00 | 6.0e-05 | 42/0 | {'budget': 6, 'converged': 10} | 0.25 / 0.36 | 0.06 |
| full | 3 | nearest | 1.59e-01 / 1.28e-01 / 6.06e-01 | 3.87e-02 | 2.71e-02 | 1.9e+00 / 2.1e+00 / 5.1e+00 | 1.0e-04 | 31/0 | {'budget': 3, 'converged': 13} | 0.30 / 0.30 | 0.04 |
| full | 3 | staged | 1.68e-01 / 1.38e-01 / 6.06e-01 | 3.87e-02 | 2.71e-02 | 1.9e+00 / 2.1e+00 / 5.1e+00 | 1.0e-04 | 28/0 | {'budget': 8, 'converged': 8} | 0.30 / 0.30 | 0.04 |
| m512 | 1 | mean | 7.29e-01 / 7.93e-01 / 9.64e-01 | 1.18e-01 | 3.13e-02 | 1.7e+00 / 1.1e+00 / 1.7e+00 | 3.3e-05 | 28/16 | {'budget': 5, 'converged': 11} | 0.11 / 0.30 | 0.07 |
| m512 | 1 | nearest | 2.09e-01 / 1.69e-01 / 5.79e-01 | 4.11e-02 | 3.13e-02 | 5.5e-01 / 6.6e-01 / 1.7e+00 | 1.2e-04 | 26/0 | {'budget': 3, 'converged': 13} | 0.29 / 0.30 | 0.05 |
| m512 | 2 | mean | 7.89e-01 / 8.10e-01 / 9.77e-01 | 1.57e-01 | 2.85e-02 | 1.7e+00 / 1.2e+00 / 1.7e+00 | 4.5e-05 | 30/17 | {'budget': 2, 'converged': 14} | 0.10 / 0.36 | 0.06 |
| m512 | 2 | nearest | 2.01e-01 / 1.67e-01 / 5.88e-01 | 3.86e-02 | 2.85e-02 | 5.9e-01 / 7.4e-01 / 1.7e+00 | 1.0e-04 | 30/0 | {'budget': 4, 'converged': 12} | 0.30 / 0.30 | 0.04 |
| m512 | 2 | staged | 2.09e-01 / 1.56e-01 / 5.88e-01 | 3.86e-02 | 2.85e-02 | 6.4e-01 / 7.4e-01 / 1.7e+00 | 1.1e-04 | 26/0 | {'budget': 9, 'converged': 7} | 0.29 / 0.30 | 0.06 |
| m512 | 3 | mean | 7.41e-01 / 8.02e-01 / 9.56e-01 | 1.35e-01 | 2.71e-02 | 1.7e+00 / 1.2e+00 / 1.7e+00 | 4.7e-05 | 30/15 | {'budget': 3, 'converged': 13} | 0.11 / 0.36 | 0.06 |
| m512 | 3 | nearest | 2.04e-01 / 1.74e-01 / 5.88e-01 | 3.87e-02 | 2.71e-02 | 6.0e-01 / 7.3e-01 / 1.7e+00 | 9.5e-05 | 31/0 | {'budget': 4, 'converged': 12} | 0.30 / 0.30 | 0.04 |
| m512 | 3 | staged | 2.04e-01 / 1.69e-01 / 5.88e-01 | 3.87e-02 | 2.71e-02 | 6.5e-01 / 7.3e-01 / 1.7e+00 | 1.0e-04 | 24/0 | {'budget': 8, 'converged': 8} | 0.30 / 0.30 | 0.05 |

### (2) auto-decoder K_LAT=8 — ./runs/autodec/ad_K8
N=64, n_train 512, n_val 64, 20000 Adam steps/stage, batch 32, P_SUB 1024, hidden 128x4, const_lr 0, lbfgs 0, z_ff 0, seed 0; LM budget 60 attempts, m_eq 512, n_test 16

| stage | n_freq | eps_in | TRAIN global / mean-rel (learned latents) | VAL stage-0 latents fixed | VAL LM-inferred best / mean-start / nearest-start | VAL by amp quartile |
|---|---|---|---|---|---|---|
| 0 | 6 | 2.36e-03 | 1.17e-02 / 9.40e-03 | 8.26e-03 | 8.26e-03 / 1.28e-02 / 8.26e-03 | 9.9e-03 / 8.3e-03 / 7.4e-03 / 7.5e-03 |
| 1 | 8 | 2.75e-05 | 3.39e-03 / 3.16e-03 | 1.21e-02 | 8.59e-03 / 9.56e-03 / 8.91e-03 | 1.0e-02 / 8.8e-03 / 8.0e-03 / 7.4e-03 |
| 2 | 12 | 8.00e-06 | 2.10e-03 / 2.16e-03 | 1.20e-02 | 8.38e-03 / 9.38e-03 / 8.78e-03 | 1.0e-02 / 8.7e-03 / 7.8e-03 / 7.0e-03 |

(stage-0 Adam-inferred val latents, secondary: 1.08e-02)

ROM (LM Gauss-Newton on the ghost-zero FD residual, held-out sources):

| colloc | stages | init | ROM rel-L2 mean / med / max | oracle (same start) | oracle best-of | ‖r‖ LM / oracle / ‖f‖ | bnd block | acc/rej | reasons | z-norm LM / oracle | NN-lat dist |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 1 | mean | 2.20e-01 / 6.44e-02 / 9.35e-01 | 2.58e-02 | 7.78e-03 | 9.9e-01 / 1.0e+00 / 5.1e+00 | 8.8e-05 | 34/9 | {'budget': 5, 'converged': 11} | 0.53 / 0.42 | 0.24 |
| full | 1 | nearest | 6.25e-02 / 5.06e-02 / 1.63e-01 | 7.78e-03 | 7.78e-03 | 8.2e-01 / 1.0e+00 / 5.1e+00 | 7.9e-05 | 24/0 | {'budget': 1, 'converged': 15} | 0.47 / 0.42 | 0.21 |
| full | 2 | mean | 3.01e-01 / 1.17e-01 / 9.50e-01 | 8.07e-03 | 7.89e-03 | 1.8e+00 / 1.2e+00 / 5.1e+00 | 7.1e-05 | 60/0 | {'budget': 16} | 0.45 / 0.39 | 0.23 |
| full | 2 | nearest | 7.66e-02 / 6.31e-02 / 1.99e-01 | 8.22e-03 | 7.89e-03 | 1.1e+00 / 1.3e+00 / 5.1e+00 | 6.8e-05 | 58/0 | {'budget': 10, 'converged': 6} | 0.42 / 0.41 | 0.15 |
| full | 2 | staged | 8.85e-02 / 6.64e-02 / 2.69e-01 | 8.22e-03 | 7.89e-03 | 1.1e+00 / 1.3e+00 / 5.1e+00 | 6.5e-05 | 30/0 | {'budget': 16} | 0.40 / 0.41 | 0.18 |
| full | 3 | mean | 3.94e-01 / 2.02e-01 / 9.50e-01 | 7.94e-03 | 7.72e-03 | 2.5e+00 / 1.2e+00 / 5.1e+00 | 5.5e-05 | 60/0 | {'budget': 16} | 0.48 / 0.40 | 0.21 |
| full | 3 | nearest | 8.55e-02 / 5.91e-02 / 2.35e-01 | 8.03e-03 | 7.72e-03 | 1.0e+00 / 1.3e+00 / 5.1e+00 | 5.4e-05 | 60/0 | {'budget': 14, 'converged': 2} | 0.42 / 0.41 | 0.17 |
| full | 3 | staged | 1.00e-01 / 6.70e-02 / 2.74e-01 | 8.03e-03 | 7.72e-03 | 1.1e+00 / 1.3e+00 / 5.1e+00 | 5.6e-05 | 30/0 | {'budget': 16} | 0.40 / 0.41 | 0.17 |
| m512 | 1 | mean | 3.84e-01 / 1.77e-01 / 9.75e-01 | 2.58e-02 | 7.78e-03 | 8.4e-01 / 3.5e-01 / 1.7e+00 | 9.5e-05 | 35/10 | {'budget': 8, 'converged': 8} | 0.62 / 0.42 | 0.27 |
| m512 | 1 | nearest | 9.65e-02 / 9.14e-02 / 2.00e-01 | 7.78e-03 | 7.78e-03 | 2.5e-01 / 3.5e-01 / 1.7e+00 | 8.5e-05 | 24/0 | {'budget': 2, 'converged': 14} | 0.50 / 0.42 | 0.21 |
| m512 | 2 | mean | 4.66e-01 / 4.05e-01 / 9.85e-01 | 8.07e-03 | 7.89e-03 | 1.0e+00 / 4.8e-01 / 1.7e+00 | 8.2e-05 | 38/14 | {'budget': 14, 'converged': 2} | 0.47 / 0.39 | 0.27 |
| m512 | 2 | nearest | 1.17e-01 / 9.67e-02 / 3.58e-01 | 8.22e-03 | 7.89e-03 | 3.6e-01 / 4.6e-01 / 1.7e+00 | 7.1e-05 | 50/0 | {'budget': 11, 'converged': 5} | 0.40 / 0.41 | 0.16 |
| m512 | 2 | staged | 1.57e-01 / 1.24e-01 / 3.43e-01 | 8.22e-03 | 7.89e-03 | 3.3e-01 / 4.6e-01 / 1.7e+00 | 7.0e-05 | 30/0 | {'budget': 15, 'converged': 1} | 0.38 / 0.41 | 0.17 |
| m512 | 3 | mean | 4.31e-01 / 3.44e-01 / 9.21e-01 | 7.94e-03 | 7.72e-03 | 1.1e+00 / 4.8e-01 / 1.7e+00 | 7.7e-05 | 38/20 | {'budget': 14, 'converged': 2} | 0.49 / 0.40 | 0.25 |
| m512 | 3 | nearest | 1.28e-01 / 1.12e-01 / 3.47e-01 | 8.03e-03 | 7.72e-03 | 3.5e-01 / 5.0e-01 / 1.7e+00 | 5.2e-05 | 40/0 | {'budget': 6, 'converged': 10} | 0.40 / 0.41 | 0.17 |
| m512 | 3 | staged | 1.61e-01 / 1.40e-01 / 3.35e-01 | 8.03e-03 | 7.72e-03 | 3.4e-01 / 5.0e-01 / 1.7e+00 | 6.7e-05 | 30/0 | {'budget': 15, 'converged': 1} | 0.38 / 0.41 | 0.17 |

### (2) auto-decoder K_LAT=8 — ./runs/autodec/ad_K8_bw
N=64, n_train 512, n_val 64, 20000 Adam steps/stage, batch 32, P_SUB 1024, hidden 128x4, const_lr 0, lbfgs 0, z_ff 0, seed 0; LM budget 100 attempts, m_eq 512, n_test 16

| stage | n_freq | eps_in | TRAIN global / mean-rel (learned latents) | VAL stage-0 latents fixed | VAL LM-inferred best / mean-start / nearest-start | VAL by amp quartile |
|---|---|---|---|---|---|---|
| 0 | 32 | 2.36e-03 | 1.08e-02 / 8.67e-03 | 8.41e-03 | 8.41e-03 / 1.64e-02 / 8.41e-03 | 1.0e-02 / 8.2e-03 / 7.8e-03 / 7.2e-03 |
| 1 | 34 | 2.55e-05 | 3.42e-03 / 3.21e-03 | 1.64e-02 | 9.10e-03 / 9.81e-03 / 9.43e-03 | 1.1e-02 / 9.4e-03 / 8.6e-03 / 7.7e-03 |
| 2 | 36 | 8.07e-06 | 2.41e-03 / 2.33e-03 | 1.63e-02 | 8.81e-03 / 9.69e-03 / 9.19e-03 | 1.1e-02 / 9.2e-03 / 8.2e-03 / 7.3e-03 |

(stage-0 Adam-inferred val latents, secondary: 1.38e-02)

ROM (LM Gauss-Newton on the ghost-zero FD residual, held-out sources):

| colloc | stages | init | ROM rel-L2 mean / med / max | oracle (same start) | oracle best-of | ‖r‖ LM / oracle / ‖f‖ | bnd block | acc/rej | reasons | z-norm LM / oracle | NN-lat dist |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 1 | mean | 3.69e-01 / 2.14e-01 / 9.86e-01 | 7.59e-03 | 7.59e-03 | 2.4e+00 / 1.3e+00 / 5.1e+00 | 4.7e-05 | 39/7 | {'converged': 14, 'budget': 2} | 0.56 / 0.42 | 0.22 |
| full | 1 | nearest | 1.03e-01 / 7.85e-02 / 2.60e-01 | 7.59e-03 | 7.59e-03 | 1.2e+00 / 1.3e+00 / 5.1e+00 | 4.3e-05 | 20/0 | {'converged': 16} | 0.42 / 0.42 | 0.19 |
| full | 2 | mean | 5.38e-01 / 5.59e-01 / 9.72e-01 | 8.65e-03 | 8.18e-03 | 3.4e+00 / 1.8e+00 / 5.1e+00 | 3.9e-05 | 82/0 | {'converged': 9, 'budget': 7} | 0.45 / 0.41 | 0.23 |
| full | 2 | nearest | 1.18e-01 / 8.76e-02 / 3.00e-01 | 8.25e-03 | 8.18e-03 | 1.3e+00 / 1.6e+00 / 5.1e+00 | 3.6e-05 | 58/0 | {'converged': 9, 'budget': 7} | 0.38 / 0.41 | 0.13 |
| full | 2 | staged | 1.20e-01 / 9.24e-02 / 3.22e-01 | 8.25e-03 | 8.18e-03 | 1.4e+00 / 1.6e+00 / 5.1e+00 | 3.6e-05 | 50/0 | {'converged': 4, 'budget': 12} | 0.39 / 0.41 | 0.17 |
| full | 3 | mean | 5.44e-01 / 5.69e-01 / 9.72e-01 | 8.75e-03 | 7.90e-03 | 3.4e+00 / 1.9e+00 / 5.1e+00 | 3.5e-05 | 90/0 | {'converged': 7, 'budget': 9} | 0.45 / 0.41 | 0.25 |
| full | 3 | nearest | 1.11e-01 / 7.61e-02 / 3.01e-01 | 7.92e-03 | 7.90e-03 | 1.3e+00 / 1.5e+00 / 5.1e+00 | 2.8e-05 | 74/0 | {'converged': 11, 'budget': 5} | 0.39 / 0.40 | 0.13 |
| full | 3 | staged | 1.17e-01 / 8.54e-02 / 3.45e-01 | 7.92e-03 | 7.90e-03 | 1.4e+00 / 1.5e+00 / 5.1e+00 | 2.9e-05 | 50/0 | {'converged': 6, 'budget': 10} | 0.39 / 0.40 | 0.17 |
| m512 | 1 | mean | 3.76e-01 / 2.39e-01 / 9.74e-01 | 7.59e-03 | 7.59e-03 | 9.7e-01 / 5.0e-01 / 1.7e+00 | 5.4e-05 | 46/7 | {'converged': 14, 'budget': 2} | 0.51 / 0.42 | 0.22 |
| m512 | 1 | nearest | 1.10e-01 / 1.07e-01 / 2.40e-01 | 7.59e-03 | 7.59e-03 | 3.9e-01 / 5.0e-01 / 1.7e+00 | 3.8e-05 | 26/0 | {'converged': 16} | 0.42 / 0.42 | 0.20 |
| m512 | 2 | mean | 6.50e-01 / 8.69e-01 / 9.88e-01 | 8.65e-03 | 8.18e-03 | 1.5e+00 / 7.1e-01 / 1.7e+00 | 3.9e-05 | 64/32 | {'converged': 7, 'budget': 9} | 0.44 / 0.41 | 0.28 |
| m512 | 2 | nearest | 1.89e-01 / 1.59e-01 / 7.87e-01 | 8.25e-03 | 8.18e-03 | 5.2e-01 / 6.3e-01 / 1.7e+00 | 3.8e-05 | 64/0 | {'converged': 10, 'budget': 6} | 0.40 / 0.41 | 0.16 |
| m512 | 2 | staged | 1.42e-01 / 1.13e-01 / 3.95e-01 | 8.25e-03 | 8.18e-03 | 4.8e-01 / 6.3e-01 / 1.7e+00 | 3.7e-05 | 38/0 | {'converged': 6, 'budget': 10} | 0.41 / 0.41 | 0.17 |
| m512 | 3 | mean | 7.29e-01 / 8.69e-01 / 9.67e-01 | 8.75e-03 | 7.90e-03 | 1.6e+00 / 7.1e-01 / 1.7e+00 | 3.9e-05 | 64/30 | {'converged': 8, 'budget': 8} | 0.49 / 0.41 | 0.29 |
| m512 | 3 | nearest | 1.89e-01 / 1.47e-01 / 7.96e-01 | 7.92e-03 | 7.90e-03 | 5.3e-01 / 6.3e-01 / 1.7e+00 | 3.0e-05 | 62/0 | {'converged': 11, 'budget': 5} | 0.41 / 0.40 | 0.15 |
| m512 | 3 | staged | 1.39e-01 / 1.04e-01 / 4.25e-01 | 7.92e-03 | 7.90e-03 | 4.6e-01 / 6.3e-01 / 1.7e+00 | 3.0e-05 | 46/0 | {'converged': 6, 'budget': 10} | 0.41 / 0.40 | 0.17 |

### diag — residual smoothness in the conditioning variable (whitened NN corr) vs fields

| object | NN1 corr (true z) | NN5 corr (true z) | NN1 corr (latent) | spectral centroid (cyc/unit) | global rel |
|---|---|---|---|---|---|
| fields_in_true_z | 0.950 | 0.911 | nan | 2.66 | nan |
| truez_resid_after_1 | 0.375 | 0.230 | nan | 10.91 | 3.96e-03 |
| truez_resid_after_2 | 0.082 | 0.050 | nan | 15.05 | 1.48e-03 |
| truez_resid_after_3 | -0.012 | -0.004 | nan | 16.68 | 1.15e-03 |
| autodec_K4_fields_in_latent | 0.893 | 0.846 | nan | nan | nan |
| autodec_K4_resid_after_1 | 0.207 | 0.120 | 0.073 | 9.35 | 1.59e-02 |
| autodec_K4_resid_after_2 | 0.118 | 0.081 | 0.080 | 14.60 | 6.34e-03 |
| autodec_K4_resid_after_3 | 0.005 | 0.004 | -0.073 | 16.00 | 4.88e-03 |
| autodec_K8_fields_in_latent | 0.855 | 0.808 | nan | nan | nan |
| autodec_K8_resid_after_1 | 0.134 | 0.086 | 0.052 | 8.95 | 1.17e-02 |
| autodec_K8_resid_after_2 | 0.182 | 0.139 | 0.121 | 14.48 | 3.39e-03 |
| autodec_K8_resid_after_3 | -0.001 | -0.004 | -0.034 | 16.57 | 2.10e-03 |
