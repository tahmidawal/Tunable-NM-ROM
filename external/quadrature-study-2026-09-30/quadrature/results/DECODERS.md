# Decoder architectures with off-mesh quadrature (burgers 2D)

Every decoder is a bank G(x) (R = 128) with the same latent head (k = 32) and nested corrections; only the spatial representation changes. `partial decoding` = evaluating G and its gradient at the quadrature points.

## Representation quality on the training mesh (256²)

| decoder | bank floor median % | bank floor worst % | head fit median % | head fit worst % |
|---|---|---|---|---|
| RFF coordinate MLP (paper) | 0.235 | 6.741 | 1.088 | 12.421 |
| SIREN coordinate MLP | 0.212 | 8.410 | 0.686 | 6.403 |
| Gaussian RBF bank | 0.278 | 8.824 | 0.670 | 5.891 |
| fixed sine bank (no training) | 0.235 | 17.159 | 0.582 | 6.364 |
| POD modes + cubic interp. | 0.208 | 4.272 | 0.745 | 7.753 |
| POD modes + bilinear interp. | 0.208 | 4.272 | 0.745 | 7.753 |

## Quadrature error of the tested advection term (worst / median rho over 96 states, continuum target, accurate M)

Reference check = agreement of 200² and 300² Gauss references (a large value means the integrand is not smooth enough for either to be converged).

| decoder | gauss_tensor 32 | gauss_tensor 48 | gauss_tensor 64 | fibonacci 17 | fibonacci 19 | fibonacci 20 | sobol 4096 | smolyak_cc 10 |
|---|---|---|---|---|---|---|---|---|
| RFF coordinate MLP (paper) | 7.8e-02 / 1.1e-03 | 4.5e-04 / 4.6e-06 | 3.9e-06 / 3.6e-08 | 4.1e-02 / 3.0e-05 | 3.2e-03 / 2.1e-06 | 6.3e-04 / 4.6e-07 | 4.6e-02 / 2.0e-02 | 7.6e-02 / 6.7e-04 |
| SIREN coordinate MLP | 9.5e-02 / 7.1e-04 | 2.4e-04 / 1.4e-07 | 1.5e-07 / 3.1e-11 | 2.5e-02 / 4.1e-05 | 1.3e-03 / 2.3e-06 | 2.9e-04 / 5.4e-07 | 5.8e-02 / 2.3e-02 | 8.8e-02 / 3.0e-04 |
| Gaussian RBF bank | 6.7e-02 / 8.4e-04 | 1.3e-05 / 2.3e-08 | 6.4e-13 / 1.5e-14 | 1.1e-02 / 4.3e-05 | 5.6e-04 / 2.3e-06 | 1.3e-04 / 5.6e-07 | 5.1e-02 / 2.3e-02 | 3.5e-02 / 1.2e-04 |
| fixed sine bank (no training) | 5.5e-02 / 3.1e-04 | 2.3e-08 / 1.8e-11 | 2.0e-14 / 1.4e-14 | 6.2e-04 / 4.0e-05 | 3.0e-05 / 2.1e-06 | 8.3e-06 / 5.3e-07 | 3.3e-02 / 2.3e-02 | 4.9e-03 / 2.2e-05 |
| POD modes + cubic interp. | 1.4e-01 / 1.4e-03 | 6.3e-02 / 1.7e-04 | 1.7e-02 / 1.2e-04 | 1.2e-01 / 1.4e-04 | 2.5e-02 / 7.2e-05 | 5.6e-03 / 6.2e-05 | 5.5e-02 / 2.3e-02 | 1.4e-01 / 1.3e-03 |
| POD modes + bilinear interp. | 1.3e-01 / 1.4e-02 | 4.9e-02 / 1.1e-02 | 4.7e-02 / 1.2e-02 | 9.0e-02 / 5.7e-03 | 4.3e-02 / 3.9e-03 | 2.8e-02 / 3.6e-03 | 6.6e-02 / 2.3e-02 | 1.7e-01 / 2.3e-02 |

Reference checks (GL200 vs GL300 worst rho): RFF coordinate MLP (paper): 1.14e-14; SIREN coordinate MLP: 1.27e-14; Gaussian RBF bank: 1.16e-14; fixed sine bank (no training): 8.63e-15; POD modes + cubic interp.: 1.33e-03; POD modes + bilinear interp.: 2.51e-02.


## End-to-end rollouts (worst error vs refined reference %, and distance from the dense rollout of the same decoder %)

| decoder | N | q | dense | Gauss 32² | Fibonacci 4181 | Sobol 4096 | Smolyak CC 8 |
|---|---|---|---|---|---|---|---|
| RFF coordinate MLP (paper) | 128 | 0 | 22.37 (0.00) | 22.24 (4.16) | 22.24 (4.15) | 22.28 (4.01) | 22.48 (9.67) |
| RFF coordinate MLP (paper) | 128 | 64 | 14.05 (0.00) | 13.81 (4.67) | 13.80 (4.68) | 13.85 (4.52) | 21.59 (18.32) |
| RFF coordinate MLP (paper) | 256 | 0 | 22.31 (0.00) | 22.28 (2.18) | 22.28 (2.18) | 22.32 (2.04) | 22.53 (8.46) |
| RFF coordinate MLP (paper) | 256 | 64 | 13.90 (0.00) | 13.88 (2.47) | 13.87 (2.48) | 13.92 (2.33) | 21.61 (18.36) |
| RFF coordinate MLP (paper) | 512 | 0 | 22.29 (0.00) | 22.28 (1.12) | 22.28 (1.12) | 22.33 (0.99) | 22.53 (7.87) |
| RFF coordinate MLP (paper) | 512 | 64 | 13.86 (0.00) | 13.88 (1.27) | 13.88 (1.28) | 13.92 (1.15) | 21.61 (18.32) |
| RFF coordinate MLP (paper) | 1024 | 0 | 19.16 (0.00) | 22.28 (0.57) | 22.28 (0.57) | 22.33 (0.45) | 22.53 (7.66) |
| RFF coordinate MLP (paper) | 1024 | 64 | - | 13.88 (nan) | 13.88 (nan) | 13.92 (nan) | 21.61 (nan) |
| SIREN coordinate MLP | 128 | 0 | 22.40 (0.00) | 22.27 (3.92) | 22.27 (3.92) | 22.32 (3.81) | 22.08 (5.92) |
| SIREN coordinate MLP | 128 | 64 | 14.12 (0.00) | 13.73 (4.65) | 13.72 (4.65) | 13.79 (4.49) | 18.50 (14.56) |
| SIREN coordinate MLP | 256 | 0 | 22.32 (0.00) | 22.28 (2.05) | 22.28 (2.05) | 22.32 (1.95) | 22.09 (5.15) |
| SIREN coordinate MLP | 256 | 64 | 13.86 (0.00) | 13.74 (2.46) | 13.73 (2.46) | 13.80 (2.31) | 18.50 (14.64) |
| SIREN coordinate MLP | 512 | 0 | 22.29 (0.00) | 22.28 (1.05) | 22.28 (1.05) | 22.32 (0.95) | 22.09 (4.83) |
| SIREN coordinate MLP | 512 | 64 | 13.77 (0.00) | 13.74 (1.27) | 13.73 (1.27) | 13.80 (1.13) | 18.49 (14.81) |
| Gaussian RBF bank | 128 | 0 | 24.53 (0.00) | 24.49 (4.19) | 24.49 (4.19) | 24.53 (4.05) | 24.11 (6.42) |
| Gaussian RBF bank | 128 | 64 | 13.65 (0.00) | 13.32 (4.79) | 13.31 (4.79) | 13.39 (4.62) | 15.20 (12.29) |
| Gaussian RBF bank | 256 | 0 | 24.49 (0.00) | 24.49 (2.21) | 24.49 (2.21) | 24.53 (2.06) | 24.11 (5.71) |
| Gaussian RBF bank | 256 | 64 | 13.40 (0.00) | 13.32 (2.54) | 13.32 (2.54) | 13.39 (2.37) | 15.17 (12.25) |
| Gaussian RBF bank | 512 | 0 | 24.48 (0.00) | 24.49 (1.13) | 24.49 (1.13) | 24.53 (1.00) | 24.11 (5.46) |
| Gaussian RBF bank | 512 | 64 | 13.34 (0.00) | 13.32 (1.31) | 13.32 (1.31) | 13.39 (1.16) | 15.17 (12.45) |
| fixed sine bank (no training) | 128 | 0 | 22.98 (0.00) | 22.98 (4.13) | 22.98 (4.13) | 23.02 (3.99) | 22.63 (7.17) |
| fixed sine bank (no training) | 128 | 64 | 9.46 (0.00) | 8.95 (4.94) | 8.95 (4.94) | 9.00 (4.75) | 11.13 (6.66) |
| fixed sine bank (no training) | 256 | 0 | 22.95 (0.00) | 22.98 (2.16) | 22.98 (2.16) | 23.02 (2.03) | 22.63 (6.12) |
| fixed sine bank (no training) | 256 | 64 | 9.05 (0.00) | 8.95 (2.62) | 8.95 (2.62) | 8.99 (2.43) | 11.11 (7.10) |
| fixed sine bank (no training) | 512 | 0 | 22.96 (0.00) | 22.98 (1.11) | 22.98 (1.11) | 23.02 (0.99) | 22.63 (5.72) |
| fixed sine bank (no training) | 512 | 64 | 8.96 (0.00) | 8.95 (1.35) | 8.94 (1.36) | 8.99 (1.18) | 11.11 (7.56) |
| POD modes + cubic interp. | 128 | 0 | 24.18 (0.00) | 24.16 (4.09) | 24.16 (4.09) | 24.18 (3.97) | 24.16 (11.09) |
| POD modes + cubic interp. | 128 | 64 | 14.38 (0.00) | 14.11 (4.50) | 14.14 (4.49) | 14.17 (4.34) | 24.24 (19.04) |
| POD modes + cubic interp. | 256 | 0 | 24.19 (0.00) | 24.20 (2.15) | 24.20 (2.14) | 24.22 (2.03) | 24.21 (10.08) |
| POD modes + cubic interp. | 256 | 64 | 14.61 (0.00) | 14.56 (2.39) | 14.60 (2.37) | 14.63 (2.23) | 24.54 (20.25) |
| POD modes + cubic interp. | 512 | 0 | 24.19 (0.00) | 24.20 (1.10) | 24.20 (1.10) | 24.22 (0.99) | 24.20 (9.66) |
| POD modes + cubic interp. | 512 | 64 | 14.52 (0.00) | 14.50 (1.24) | 14.54 (1.22) | 14.57 (1.08) | 24.49 (20.69) |
| POD modes + bilinear interp. | 128 | 0 | 24.18 (0.00) | 24.17 (4.23) | 24.17 (4.06) | 24.18 (4.03) | 24.19 (8.64) |
| POD modes + bilinear interp. | 128 | 64 | 14.38 (0.00) | 14.12 (4.69) | 14.14 (4.46) | 14.18 (4.43) | 20.64 (15.79) |
| POD modes + bilinear interp. | 256 | 0 | 24.19 (0.00) | 24.21 (2.29) | 24.21 (2.11) | 24.22 (2.10) | 24.23 (7.56) |
| POD modes + bilinear interp. | 256 | 64 | 14.61 (0.00) | 14.57 (2.58) | 14.61 (2.34) | 14.64 (2.31) | 21.18 (16.83) |
| POD modes + bilinear interp. | 512 | 0 | 24.19 (0.00) | 24.20 (1.25) | 24.20 (1.07) | 24.21 (1.06) | 24.22 (7.13) |
| POD modes + bilinear interp. | 512 | 64 | 14.47 (0.00) | 14.46 (1.44) | 14.49 (1.20) | 14.52 (1.17) | 21.04 (17.10) |