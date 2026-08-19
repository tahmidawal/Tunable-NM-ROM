# Poisson hybrid results (generated)

This file is generated from the listed run JSONs; do not edit its numeric tables.

## Provenance

| run | job | GPU | commit | source hash |
|---|---:|---|---|---|
| dev1.json | 2653621 | NVIDIA A100 80GB PCIe | 6346c25a82c0 | d68afdfbd6ab5887 |
| gcal1.json | 2654296 | NVIDIA A100-PCIE-40GB | 8a2f159e393d | 070968b312bd49af |
| specdev1.json | 2659541 | NVIDIA A100 80GB PCIe | 0e58b4d5ced3 | 6b8fd188c0493fb4 |
| specdev2.json | 2661884 | NVIDIA A100-PCIE-40GB | 2e4ac0d56b2e | 66e416261c0fc69f |
| specdev3.json | 2662182 | NVIDIA A100 80GB PCIe | 717062ff5f3f | a3d45b08415677b5 |
| specdev4.json | 2662361 | NVIDIA A100 80GB PCIe | b7a5263fba81 | a3d45b08415677b5 |
| specdev5.json | 2662493 | NVIDIA A100 80GB PCIe | d01c94cd0b58 | 639c813e2a913768 |
| final1.json | 2662802 | NVIDIA A100 80GB PCIe | f0c9dfa10a24 | 639c813e2a913768 |
| pairfinal1.json | 2664551 | NVIDIA A100 80GB PCIe | 92447ead81a9 | 34fc25856c5e4991 |

## Audited learned-hybrid conclusion

The conservative gate calls a learned crossover only when both the paired case-clustered speedup interval lies above one and the paired learned-minus-zero delta interval lies below zero. The balanced audit yields **1 supported faster row(s)**; all other learned rows are ties, inconclusive, or slower. The earlier overlapping multi-arm K8 rows are omitted below because these balanced rows supersede them.

- N=1024, tolerance=1e-06: 1.026x, clustered 95% CI [1.014, 1.077].

## End-to-end rows

Multi-arm rows use the stored mean of case medians and are mechanism evidence only for small learned-versus-zero differences because that rotation was not position-balanced. Balanced AB/BA rows use the median across case medians and are authoritative for the optimized K8-versus-zero comparison. Confidence intervals resample whole cases.

| run | design | N | tolerance | arm | construct ms | total ms | total 95% CI ms | speedup/zero | speedup 95% CI | CG iters | guess A-error | final residual | meets tau | outliers | direct ms |
|---|---|---:|---:|---|---:|---:|---|---:|---|---:|---:|---:|---|---:|---:|
| dev1 | multi-arm rotation | 256 | 1e-06 | `lmmean_cfull_q0` | 7.235 | 26.193 | — | 0.775 | — | 624.6 | 7.62e-02 | 9.97e-07 | yes | — | 0.083 |
| dev1 | multi-arm rotation | 256 | 1e-06 | `lmmean_c64_q0` | 6.236 | 25.203 | — | 0.804 | — | 625.4 | 8.40e-02 | 9.98e-07 | yes | — | 0.083 |
| dev1 | multi-arm rotation | 256 | 1e-06 | `lmtrmean_c64_q0` | 6.157 | 24.858 | — | 0.816 | — | 618.7 | 6.34e-02 | 9.97e-07 | yes | — | 0.083 |
| dev1 | multi-arm rotation | 256 | 1e-06 | `lmnearest_c64_q0` | 6.425 | 25.339 | — | 0.802 | — | 630.4 | 8.90e-02 | 1.00e-06 | yes | — | 0.083 |
| dev1 | multi-arm rotation | 256 | 1e-06 | `param1_c64_q0` | 0.202 | 18.756 | — | 1.084 | — | 593.5 | 4.56e-02 | 9.95e-07 | yes | — | 0.083 |
| dev1 | multi-arm rotation | 256 | 1e-06 | `paramall_c64_q0` | 0.473 | 18.854 | — | 1.079 | — | 582.4 | 4.45e-02 | 1.00e-06 | yes | — | 0.083 |
| dev1 | multi-arm rotation | 256 | 1e-06 | `param1_c64_q8` | 0.224 | 5.122 | — | 3.972 | — | 157.1 | 4.49e-02 | 1.00e-06 | yes | — | 0.083 |
| dev1 | multi-arm rotation | 256 | 1e-06 | `spectral_q8` | 0.067 | 4.572 | — | 4.440 | — | 141.1 | 1.58e-01 | 9.94e-07 | yes | — | 0.083 |
| dev1 | multi-arm rotation | 256 | 1e-06 | `spectral_q16` | 0.066 | 2.023 | — | 10.043 | — | 50.7 | 2.58e-02 | 9.87e-07 | yes | — | 0.083 |
| dev1 | multi-arm rotation | 512 | 1e-06 | `lmmean_cfull_q0` | 10.583 | 69.820 | — | 0.903 | — | 1273.8 | 7.63e-02 | 1.00e-06 | yes | — | 0.158 |
| dev1 | multi-arm rotation | 512 | 1e-06 | `lmmean_c64_q0` | 6.283 | 65.701 | — | 0.960 | — | 1278.9 | 8.77e-02 | 1.00e-06 | yes | — | 0.158 |
| dev1 | multi-arm rotation | 512 | 1e-06 | `lmnearest_c64_q0` | 6.448 | 65.881 | — | 0.959 | — | 1282.1 | 9.21e-02 | 9.99e-07 | yes | — | 0.158 |
| dev1 | multi-arm rotation | 512 | 1e-06 | `param1_c64_q0` | 0.207 | 58.427 | — | 1.084 | — | 1212.6 | 5.18e-02 | 9.98e-07 | yes | — | 0.158 |
| dev1 | multi-arm rotation | 512 | 1e-06 | `paramall_c64_q0` | 0.459 | 58.207 | — | 1.089 | — | 1196.1 | 5.02e-02 | 9.99e-07 | yes | — | 0.158 |
| dev1 | multi-arm rotation | 512 | 1e-06 | `param1_c64_q8` | 0.248 | 16.029 | — | 3.944 | — | 324.8 | 5.12e-02 | 9.98e-07 | yes | — | 0.158 |
| dev1 | multi-arm rotation | 512 | 1e-06 | `spectral_q8` | 0.073 | 14.630 | — | 4.317 | — | 292.6 | 1.58e-01 | 9.99e-07 | yes | — | 0.158 |
| dev1 | multi-arm rotation | 512 | 1e-06 | `spectral_q16` | 0.074 | 6.124 | — | 10.311 | — | 107.1 | 2.57e-02 | 9.99e-07 | yes | — | 0.158 |
| dev1 | multi-arm rotation | 1024 | 1e-06 | `lmmean_cfull_q0` | 24.054 | 230.328 | — | 0.946 | — | 2591.0 | 7.63e-02 | 9.98e-07 | yes | — | 0.626 |
| dev1 | multi-arm rotation | 1024 | 1e-06 | `lmmean_c64_q0` | 6.579 | 213.212 | — | 1.026 | — | 2600.5 | 8.97e-02 | 1.00e-06 | yes | — | 0.626 |
| dev1 | multi-arm rotation | 1024 | 1e-06 | `lmnearest_c64_q0` | 6.764 | 215.264 | — | 1.022 | — | 2604.9 | 9.39e-02 | 1.00e-06 | yes | — | 0.626 |
| dev1 | multi-arm rotation | 1024 | 1e-06 | `param1_c64_q0` | 0.215 | 204.345 | — | 1.088 | — | 2454.9 | 5.61e-02 | 9.99e-07 | yes | — | 0.626 |
| dev1 | multi-arm rotation | 1024 | 1e-06 | `paramall_c64_q0` | 0.495 | 204.992 | — | 1.071 | — | 2444.8 | 5.36e-02 | 1.00e-06 | yes | — | 0.626 |
| dev1 | multi-arm rotation | 1024 | 1e-06 | `param1_c64_q8` | 0.300 | 56.477 | — | 3.995 | — | 669.5 | 5.56e-02 | 9.99e-07 | yes | — | 0.626 |
| dev1 | multi-arm rotation | 1024 | 1e-06 | `spectral_q8` | 0.082 | 50.104 | — | 4.318 | — | 604.4 | 1.58e-01 | 1.00e-06 | yes | — | 0.626 |
| dev1 | multi-arm rotation | 1024 | 1e-06 | `spectral_q16` | 0.081 | 21.855 | — | 10.006 | — | 224.8 | 2.57e-02 | 9.98e-07 | yes | — | 0.626 |
| gcal1 | multi-arm rotation | 256 | 1e-06 | `groupn_rt30_c64_q0` | 2.288 | 24.476 | — | 0.898 | — | 674.8 | 5.41e-01 | 9.91e-07 | yes | — | 0.163 |
| gcal1 | multi-arm rotation | 256 | 1e-06 | `groupn_rt10_c64_q0` | 2.767 | 23.888 | — | 0.920 | — | 638.0 | 2.61e-01 | 9.86e-07 | yes | — | 0.163 |
| gcal1 | multi-arm rotation | 256 | 1e-06 | `groupn_rt03_c64_q0` | 2.937 | 23.651 | — | 0.929 | — | 631.0 | 1.31e-01 | 9.86e-07 | yes | — | 0.163 |
| gcal1 | multi-arm rotation | 256 | 1e-06 | `groupn_rt01_c64_q0` | 3.967 | 24.343 | — | 0.903 | — | 627.0 | 7.24e-02 | 9.99e-07 | yes | — | 0.163 |
| gcal1 | multi-arm rotation | 256 | 1e-06 | `groupn_rt30_c64_q8` | 2.345 | 8.360 | — | 2.629 | — | 176.5 | 1.42e-01 | 9.74e-07 | yes | — | 0.163 |
| gcal1 | multi-arm rotation | 256 | 1e-06 | `groupn_rt10_c64_q8` | 2.843 | 8.500 | — | 2.586 | — | 168.0 | 1.08e-01 | 9.76e-07 | yes | — | 0.163 |
| gcal1 | multi-arm rotation | 256 | 1e-06 | `groupn_rt03_c64_q8` | 3.023 | 8.702 | — | 2.526 | — | 166.5 | 8.40e-02 | 9.89e-07 | yes | — | 0.163 |
| gcal1 | multi-arm rotation | 256 | 1e-06 | `spectral_q8` | 0.084 | 5.139 | — | 4.277 | — | 146.2 | 1.58e-01 | 9.74e-07 | yes | — | 0.163 |
| gcal1 | multi-arm rotation | 256 | 1e-06 | `spectral_q16` | 0.084 | 2.361 | — | 9.311 | — | 60.5 | 2.58e-02 | 9.67e-07 | yes | — | 0.163 |
| specdev1 | multi-arm rotation | 256 | 1e-06 | `spectral_q8` | 0.067 | 4.390 | [4.097, 4.631] | 4.555 | [4.388, 4.813] | 139.3 | 1.58e-01 | 9.91e-07 | yes | 0 | 0.135 |
| specdev1 | multi-arm rotation | 256 | 1e-06 | `spectral_q16` | 0.066 | 1.794 | [1.414, 2.166] | 11.145 | [9.351, 13.892] | 51.5 | 2.58e-02 | 9.69e-07 | yes | 0 | 0.135 |
| specdev1 | multi-arm rotation | 256 | 1e-06 | `spectral_q24` | 0.069 | 0.987 | [0.580, 1.402] | 20.252 | [14.372, 34.009] | 24.5 | 3.96e-03 | 9.54e-07 | yes | 0 | 0.135 |
| specdev1 | multi-arm rotation | 256 | 1e-06 | `spectral_q32` | 0.068 | 0.699 | [0.365, 1.058] | 28.589 | [19.359, 54.085] | 15.3 | 7.31e-04 | 9.44e-07 | yes | 0 | 0.135 |
| specdev1 | multi-arm rotation | 256 | 1e-06 | `spectral_q48` | 0.069 | 0.493 | [0.265, 0.748] | 40.586 | [27.335, 74.454] | 9.0 | 7.30e-05 | 9.89e-07 | yes | 0 | 0.135 |
| specdev1 | multi-arm rotation | 256 | 1e-06 | `spectral_q64` | 0.071 | 0.400 | [0.240, 0.574] | 49.953 | [35.109, 82.302] | 6.0 | 3.89e-05 | 9.79e-07 | yes | 1 | 0.135 |
| specdev1 | multi-arm rotation | 512 | 1e-06 | `spectral_q8` | 0.072 | 14.132 | [13.043, 15.140] | 4.492 | [4.308, 4.758] | 290.2 | 1.58e-01 | 9.95e-07 | yes | 0 | 0.351 |
| specdev1 | multi-arm rotation | 512 | 1e-06 | `spectral_q16` | 0.074 | 5.458 | [4.072, 6.818] | 11.633 | [9.586, 15.000] | 108.5 | 2.57e-02 | 9.44e-07 | yes | 0 | 0.351 |
| specdev1 | multi-arm rotation | 512 | 1e-06 | `spectral_q24` | 0.076 | 2.815 | [1.439, 4.247] | 22.557 | [15.316, 44.164] | 52.8 | 3.95e-03 | 9.61e-07 | yes | 0 | 0.351 |
| specdev1 | multi-arm rotation | 512 | 1e-06 | `spectral_q32` | 0.079 | 1.902 | [0.741, 3.158] | 33.375 | [20.647, 83.932] | 33.2 | 7.29e-04 | 9.56e-07 | yes | 0 | 0.351 |
| specdev1 | multi-arm rotation | 512 | 1e-06 | `spectral_q48` | 0.080 | 1.221 | [0.479, 2.060] | 51.978 | [31.668, 129.963] | 20.0 | 7.51e-05 | 9.64e-07 | yes | 0 | 0.351 |
| specdev1 | multi-arm rotation | 512 | 1e-06 | `spectral_q64` | 0.082 | 0.950 | [0.404, 1.537] | 66.820 | [42.196, 153.983] | 14.3 | 4.14e-05 | 8.57e-07 | yes | 2 | 0.351 |
| specdev1 | multi-arm rotation | 1024 | 1e-06 | `spectral_q8` | 0.082 | 49.106 | [45.650, 52.047] | 4.476 | [4.285, 4.746] | 599.8 | 1.58e-01 | 1.00e-06 | yes | 0 | 1.105 |
| specdev1 | multi-arm rotation | 1024 | 1e-06 | `spectral_q16` | 0.080 | 19.333 | [14.738, 23.850] | 11.370 | [9.388, 14.697] | 227.8 | 2.57e-02 | 9.98e-07 | yes | 0 | 1.105 |
| specdev1 | multi-arm rotation | 1024 | 1e-06 | `spectral_q24` | 0.085 | 9.743 | [4.902, 14.786] | 22.562 | [15.114, 43.882] | 112.7 | 3.95e-03 | 9.99e-07 | yes | 0 | 1.105 |
| specdev1 | multi-arm rotation | 1024 | 1e-06 | `spectral_q32` | 0.085 | 6.219 | [2.199, 10.625] | 35.346 | [20.833, 101.711] | 70.2 | 7.29e-04 | 9.99e-07 | yes | 0 | 1.105 |
| specdev1 | multi-arm rotation | 1024 | 1e-06 | `spectral_q48` | 0.113 | 3.945 | [1.281, 6.835] | 55.723 | [32.796, 170.026] | 43.5 | 7.56e-05 | 9.72e-07 | yes | 1 | 1.105 |
| specdev1 | multi-arm rotation | 1024 | 1e-06 | `spectral_q64` | 0.117 | 2.934 | [0.985, 4.977] | 74.928 | [43.449, 221.260] | 31.5 | 4.20e-05 | 9.60e-07 | yes | 1 | 1.105 |
| specdev2 | multi-arm rotation | 256 | 1e-06 | `spectral_q32` | 0.077 | 0.730 | [0.369, 1.114] | 29.475 | [19.464, 57.501] | 15.3 | 7.31e-04 | 9.44e-07 | yes | 2 | 0.138 |
| specdev2 | multi-arm rotation | 256 | 1e-06 | `spectral_q64` | 0.080 | 0.427 | [0.252, 0.617] | 50.364 | [35.111, 84.501] | 6.0 | 3.89e-05 | 9.79e-07 | yes | 3 | 0.138 |
| specdev2 | multi-arm rotation | 256 | 1e-06 | `spectral_q96` | 0.084 | 0.360 | [0.244, 0.486] | 59.710 | [44.532, 87.056] | 3.7 | 1.91e-05 | 7.23e-07 | yes | 2 | 0.138 |
| specdev2 | multi-arm rotation | 256 | 1e-06 | `spectral_q128` | 0.085 | 0.325 | [0.239, 0.421] | 66.179 | [52.135, 88.724] | 2.5 | 1.04e-05 | 4.70e-07 | yes | 4 | 0.138 |
| specdev2 | multi-arm rotation | 512 | 1e-06 | `spectral_q32` | 0.083 | 1.641 | [0.633, 2.765] | 32.981 | [20.174, 84.545] | 33.2 | 7.29e-04 | 9.56e-07 | yes | 1 | 0.332 |
| specdev2 | multi-arm rotation | 512 | 1e-06 | `spectral_q64` | 0.086 | 0.852 | [0.373, 1.391] | 63.516 | [39.687, 143.788] | 14.3 | 4.14e-05 | 8.57e-07 | yes | 1 | 0.332 |
| specdev2 | multi-arm rotation | 512 | 1e-06 | `spectral_q96` | 0.097 | 0.622 | [0.314, 0.966] | 87.051 | [56.975, 170.443] | 8.7 | 2.21e-05 | 7.94e-07 | yes | 1 | 0.332 |
| specdev2 | multi-arm rotation | 512 | 1e-06 | `spectral_q128` | 0.101 | 0.512 | [0.289, 0.755] | 105.773 | [72.973, 184.863] | 6.0 | 1.39e-05 | 7.55e-07 | yes | 1 | 0.332 |
| specdev2 | multi-arm rotation | 1024 | 1e-06 | `spectral_q32` | 0.100 | 6.293 | [2.096, 10.786] | 36.173 | [21.621, 107.382] | 70.2 | 7.29e-04 | 9.99e-07 | yes | 1 | 1.276 |
| specdev2 | multi-arm rotation | 1024 | 1e-06 | `spectral_q64` | 0.135 | 2.984 | [1.003, 5.155] | 76.290 | [44.473, 224.986] | 31.5 | 4.20e-05 | 9.60e-07 | yes | 0 | 1.276 |
| specdev2 | multi-arm rotation | 1024 | 1e-06 | `spectral_q96` | 0.140 | 2.029 | [0.753, 3.448] | 112.177 | [67.167, 299.020] | 19.7 | 2.29e-05 | 9.20e-07 | yes | 0 | 1.276 |
| specdev2 | multi-arm rotation | 1024 | 1e-06 | `spectral_q128` | 0.152 | 1.572 | [0.657, 2.581] | 144.849 | [89.359, 345.332] | 14.0 | 1.48e-05 | 8.83e-07 | yes | 0 | 1.276 |
| specdev3 | multi-arm rotation | 256 | 1e-06 | `spectral_q64` | 0.072 | 0.393 | [0.227, 0.574] | 50.931 | [35.046, 87.100] | 6.0 | 3.89e-05 | 9.79e-07 | yes | 3 | 0.141 |
| specdev3 | multi-arm rotation | 256 | 1e-06 | `spectral_q128` | 0.079 | 0.310 | [0.234, 0.392] | 64.592 | [51.722, 84.468] | 2.5 | 1.04e-05 | 4.70e-07 | yes | 4 | 0.141 |
| specdev3 | multi-arm rotation | 256 | 1e-06 | `spectral_q192` | 0.080 | 0.292 | [0.230, 0.361] | 68.481 | [55.927, 85.975] | 1.3 | 2.74e-06 | 3.06e-07 | yes | 3 | 0.141 |
| specdev3 | multi-arm rotation | 256 | 1e-06 | `spectral_q256` | 0.080 | 0.215 | [0.206, 0.226] | 93.310 | [89.619, 96.831] | 0.0 | 5.04e-14 | 4.87e-12 | yes | 4 | 0.141 |
| specdev3 | multi-arm rotation | 512 | 1e-06 | `spectral_q64` | 0.082 | 0.900 | [0.365, 1.513] | 69.292 | [42.966, 169.386] | 14.3 | 4.14e-05 | 8.57e-07 | yes | 3 | 0.339 |
| specdev3 | multi-arm rotation | 512 | 1e-06 | `spectral_q128` | 0.095 | 0.553 | [0.303, 0.831] | 112.730 | [76.395, 203.557] | 6.0 | 1.39e-05 | 7.55e-07 | yes | 2 | 0.339 |
| specdev3 | multi-arm rotation | 512 | 1e-06 | `spectral_q192` | 0.097 | 0.440 | [0.282, 0.614] | 141.862 | [101.807, 218.687] | 3.5 | 6.81e-06 | 8.77e-07 | yes | 3 | 0.339 |
| specdev3 | multi-arm rotation | 512 | 1e-06 | `spectral_q256` | 0.109 | 0.393 | [0.274, 0.521] | 158.684 | [121.792, 224.597] | 2.3 | 3.72e-06 | 8.90e-07 | yes | 1 | 0.339 |
| specdev3 | multi-arm rotation | 1024 | 1e-06 | `spectral_q64` | 0.116 | 2.790 | [0.902, 4.843] | 79.162 | [46.564, 242.841] | 31.5 | 4.20e-05 | 9.60e-07 | yes | 0 | 1.085 |
| specdev3 | multi-arm rotation | 1024 | 1e-06 | `spectral_q128` | 0.141 | 1.434 | [0.579, 2.367] | 154.043 | [93.064, 378.083] | 14.0 | 1.48e-05 | 8.83e-07 | yes | 0 | 1.085 |
| specdev3 | multi-arm rotation | 1024 | 1e-06 | `spectral_q192` | 0.150 | 1.023 | [0.488, 1.615] | 215.972 | [138.268, 447.659] | 8.3 | 7.88e-06 | 9.25e-07 | yes | 0 | 1.085 |
| specdev3 | multi-arm rotation | 1024 | 1e-06 | `spectral_q256` | 0.191 | 0.853 | [0.481, 1.262] | 258.831 | [177.928, 457.241] | 5.7 | 4.95e-06 | 8.03e-07 | yes | 0 | 1.085 |
| specdev4 | multi-arm rotation | 1024 | 1e-06 | `spectral_q192` | 0.149 | 1.033 | [0.463, 1.648] | 209.552 | [133.712, 461.477] | 8.3 | 7.88e-06 | 9.25e-07 | yes | 3 | 1.087 |
| specdev4 | multi-arm rotation | 1024 | 1e-06 | `spectral_q256` | 0.192 | 0.885 | [0.479, 1.327] | 244.724 | [165.462, 443.368] | 5.7 | 4.95e-06 | 8.03e-07 | yes | 0 | 1.087 |
| specdev4 | multi-arm rotation | 1024 | 1e-06 | `spectral_q384` | 0.223 | 0.723 | [0.468, 0.999] | 299.545 | [219.684, 456.257] | 3.3 | 2.42e-06 | 7.03e-07 | yes | 0 | 1.087 |
| specdev4 | multi-arm rotation | 1024 | 1e-06 | `spectral_q512` | 0.296 | 0.696 | [0.512, 0.893] | 310.908 | [246.789, 417.828] | 2.2 | 1.32e-06 | 6.76e-07 | yes | 0 | 1.087 |
| specdev4 | multi-arm rotation | 1024 | 1e-06 | `spectral_q768` | 0.466 | 0.778 | [0.667, 0.902] | 278.157 | [244.156, 319.214] | 1.2 | 3.51e-07 | 5.79e-07 | yes | 0 | 1.087 |
| specdev4 | multi-arm rotation | 1024 | 1e-06 | `spectral_q1024` | 0.635 | 0.801 | [0.779, 0.829] | 270.291 | [263.355, 278.055] | 0.0 | 2.91e-13 | 1.59e-10 | yes | 0 | 1.087 |
| specdev5 | multi-arm rotation | 512 | 1e-06 | `spectral_q256` | 0.107 | 0.360 | [0.246, 0.488] | 173.143 | [129.602, 251.176] | 2.3 | 3.72e-06 | 8.90e-07 | yes | 2 | 0.327 |
| specdev5 | multi-arm rotation | 512 | 1e-08 | `spectral_q256` | 0.107 | 0.429 | [0.265, 0.610] | 170.798 | [121.521, 274.013] | 4.0 | 3.72e-06 | 9.36e-09 | yes | 1 | 0.329 |
| specdev5 | multi-arm rotation | 512 | 1e-10 | `spectral_q256` | 0.107 | 0.522 | [0.292, 0.771] | 158.247 | [106.702, 281.485] | 6.0 | 3.72e-06 | 8.25e-11 | yes | 3 | 0.333 |
| specdev5 | multi-arm rotation | 512 | 1e-06 | `spectral_q384` | 0.130 | 0.364 | [0.284, 0.452] | 171.250 | [140.187, 218.664] | 1.2 | 9.84e-07 | 7.95e-07 | yes | 1 | 0.327 |
| specdev5 | multi-arm rotation | 512 | 1e-08 | `spectral_q384` | 0.130 | 0.400 | [0.289, 0.520] | 183.147 | [142.254, 251.807] | 2.0 | 9.84e-07 | 7.84e-09 | yes | 1 | 0.329 |
| specdev5 | multi-arm rotation | 512 | 1e-10 | `spectral_q384` | 0.130 | 0.472 | [0.319, 0.641] | 175.221 | [130.754, 258.391] | 3.3 | 9.84e-07 | 5.80e-11 | yes | 3 | 0.333 |
| specdev5 | multi-arm rotation | 512 | 1e-06 | `spectral_q512` | 0.159 | 0.284 | [0.275, 0.294] | 219.480 | [212.280, 224.461] | 0.0 | 1.34e-13 | 2.80e-11 | yes | 2 | 0.327 |
| specdev5 | multi-arm rotation | 512 | 1e-08 | `spectral_q512` | 0.159 | 0.299 | [0.279, 0.321] | 245.200 | [230.616, 261.386] | 0.0 | 1.34e-13 | 2.80e-11 | yes | 2 | 0.329 |
| specdev5 | multi-arm rotation | 512 | 1e-10 | `spectral_q512` | 0.159 | 0.291 | [0.282, 0.301] | 284.000 | [274.846, 293.704] | 0.0 | 1.34e-13 | 2.80e-11 | yes | 2 | 0.333 |
| specdev5 | multi-arm rotation | 512 | 1e-06 | `spectral_q1024` | 0.158 | 0.286 | [0.277, 0.296] | 217.673 | [209.316, 225.421] | 0.0 | 1.34e-13 | 2.80e-11 | yes | 1 | 0.327 |
| specdev5 | multi-arm rotation | 512 | 1e-08 | `spectral_q1024` | 0.158 | 0.291 | [0.279, 0.305] | 252.069 | [241.068, 261.921] | 0.0 | 1.34e-13 | 2.80e-11 | yes | 1 | 0.329 |
| specdev5 | multi-arm rotation | 512 | 1e-10 | `spectral_q1024` | 0.158 | 0.291 | [0.284, 0.298] | 284.343 | [276.726, 292.227] | 0.0 | 1.34e-13 | 2.80e-11 | yes | 3 | 0.333 |
| specdev5 | multi-arm rotation | 1024 | 1e-06 | `spectral_q256` | 0.192 | 0.806 | [0.422, 1.228] | 271.755 | [180.213, 514.548] | 5.7 | 4.95e-06 | 8.03e-07 | yes | 0 | 1.078 |
| specdev5 | multi-arm rotation | 1024 | 1e-08 | `spectral_q256` | 0.192 | 1.140 | [0.543, 1.788] | 225.653 | [144.993, 470.128] | 9.7 | 4.95e-06 | 9.21e-09 | yes | 0 | 1.084 |
| specdev5 | multi-arm rotation | 1024 | 1e-10 | `spectral_q256` | 0.192 | 1.463 | [0.627, 2.378] | 197.614 | [122.326, 458.252] | 13.8 | 4.95e-06 | 9.83e-11 | yes | 1 | 1.074 |
| specdev5 | multi-arm rotation | 1024 | 1e-06 | `spectral_q384` | 0.224 | 0.702 | [0.452, 0.975] | 311.766 | [226.422, 479.606] | 3.3 | 2.42e-06 | 7.03e-07 | yes | 0 | 1.078 |
| specdev5 | multi-arm rotation | 1024 | 1e-08 | `spectral_q384` | 0.224 | 0.902 | [0.507, 1.324] | 285.285 | [193.257, 505.015] | 5.8 | 2.42e-06 | 7.55e-09 | yes | 0 | 1.084 |
| specdev5 | multi-arm rotation | 1024 | 1e-10 | `spectral_q384` | 0.224 | 1.096 | [0.564, 1.675] | 263.807 | [173.999, 509.834] | 8.3 | 2.42e-06 | 9.96e-11 | yes | 0 | 1.074 |
| specdev5 | multi-arm rotation | 1024 | 1e-06 | `spectral_q512` | 0.297 | 0.660 | [0.486, 0.849] | 331.838 | [260.008, 446.619] | 2.2 | 1.32e-06 | 6.76e-07 | yes | 0 | 1.078 |
| specdev5 | multi-arm rotation | 1024 | 1e-08 | `spectral_q512` | 0.297 | 0.804 | [0.536, 1.093] | 319.779 | [237.399, 475.720] | 3.8 | 1.32e-06 | 7.51e-09 | yes | 0 | 1.084 |
| specdev5 | multi-arm rotation | 1024 | 1e-10 | `spectral_q512` | 0.297 | 0.936 | [0.563, 1.345] | 308.909 | [216.284, 510.221] | 5.7 | 1.32e-06 | 9.36e-11 | yes | 0 | 1.074 |
| specdev5 | multi-arm rotation | 1024 | 1e-06 | `spectral_q1024` | 0.629 | 0.764 | [0.757, 0.771] | 286.635 | [284.370, 289.109] | 0.0 | 2.91e-13 | 1.59e-10 | yes | 1 | 1.078 |
| specdev5 | multi-arm rotation | 1024 | 1e-08 | `spectral_q1024` | 0.629 | 0.769 | [0.763, 0.776] | 334.601 | [330.901, 338.005] | 0.0 | 2.91e-13 | 1.59e-10 | yes | 0 | 1.084 |
| specdev5 | multi-arm rotation | 1024 | 1e-10 | `spectral_q1024` | 0.629 | 0.824 | [0.760, 0.891] | 350.927 | [325.318, 380.194] | 0.3 | 2.91e-13 | 7.16e-11 | yes | 0 | 1.074 |
| final1 | multi-arm rotation | 32 | 1e-06 | `lmmean_cfull_q0` | 3.558 | 6.292 | [5.497, 7.207] | 0.420 | [0.371, 0.478] | 73.8 | 5.68e-02 | 9.95e-07 | yes | 1 | 0.175 |
| final1 | multi-arm rotation | 32 | 1e-08 | `lmmean_cfull_q0` | 3.558 | 6.455 | [5.723, 7.271] | 0.462 | [0.405, 0.522] | 87.8 | 5.68e-02 | 9.44e-09 | yes | 1 | 0.173 |
| final1 | multi-arm rotation | 32 | 1e-10 | `lmmean_cfull_q0` | 3.558 | 6.904 | [6.191, 7.707] | 0.489 | [0.433, 0.549] | 102.8 | 5.68e-02 | 9.84e-11 | yes | 1 | 0.173 |
| final1 | multi-arm rotation | 32 | 1e-06 | `lmtrmean_c64_q0` | 3.071 | 5.719 | [4.946, 6.669] | 0.462 | [0.387, 0.547] | 74.0 | 5.73e-02 | 9.77e-07 | yes | 0 | 0.175 |
| final1 | multi-arm rotation | 32 | 1e-08 | `lmtrmean_c64_q0` | 3.071 | 5.993 | [5.243, 6.943] | 0.497 | [0.424, 0.578] | 88.0 | 5.73e-02 | 9.86e-09 | yes | 0 | 0.173 |
| final1 | multi-arm rotation | 32 | 1e-10 | `lmtrmean_c64_q0` | 3.071 | 6.438 | [5.711, 7.388] | 0.524 | [0.452, 0.605] | 102.0 | 5.73e-02 | 9.35e-11 | yes | 0 | 0.173 |
| final1 | multi-arm rotation | 32 | 1e-06 | `groupn_rt30_c64_q0` | 1.646 | 4.465 | [4.012, 4.929] | 0.592 | [0.530, 0.670] | 79.3 | 4.35e-01 | 9.71e-07 | yes | 0 | 0.175 |
| final1 | multi-arm rotation | 32 | 1e-08 | `groupn_rt30_c64_q0` | 1.646 | 4.769 | [4.268, 5.202] | 0.625 | [0.565, 0.697] | 93.8 | 4.35e-01 | 9.75e-09 | yes | 0 | 0.173 |
| final1 | multi-arm rotation | 32 | 1e-10 | `groupn_rt30_c64_q0` | 1.646 | 5.203 | [4.707, 5.638] | 0.649 | [0.593, 0.714] | 107.2 | 4.35e-01 | 9.68e-11 | yes | 0 | 0.173 |
| final1 | multi-arm rotation | 32 | 1e-06 | `groupn_rt30_c64_q8` | 1.658 | 2.666 | [2.158, 3.170] | 0.992 | [0.822, 1.242] | 19.2 | 1.37e-01 | 9.94e-07 | yes | 1 | 0.175 |
| final1 | multi-arm rotation | 32 | 1e-08 | `groupn_rt30_c64_q8` | 1.658 | 2.768 | [2.312, 3.201] | 1.076 | [0.918, 1.293] | 26.2 | 1.37e-01 | 9.76e-09 | yes | 1 | 0.173 |
| final1 | multi-arm rotation | 32 | 1e-10 | `groupn_rt30_c64_q8` | 1.658 | 3.008 | [2.576, 3.427] | 1.122 | [0.977, 1.325] | 33.3 | 1.37e-01 | 8.68e-11 | yes | 0 | 0.173 |
| final1 | multi-arm rotation | 32 | 1e-06 | `spectral_q512` | 0.061 | 0.204 | [0.200, 0.210] | 12.961 | [12.388, 13.440] | 0.0 | 4.46e-15 | 4.07e-14 | yes | 2 | 0.175 |
| final1 | multi-arm rotation | 32 | 1e-08 | `spectral_q512` | 0.061 | 0.204 | [0.198, 0.210] | 14.609 | [14.108, 15.117] | 0.0 | 4.46e-15 | 4.07e-14 | yes | 1 | 0.173 |
| final1 | multi-arm rotation | 32 | 1e-10 | `spectral_q512` | 0.061 | 0.209 | [0.198, 0.221] | 16.141 | [15.279, 16.856] | 0.0 | 4.46e-15 | 4.07e-14 | yes | 1 | 0.173 |
| final1 | multi-arm rotation | 32 | 1e-06 | `spectral_q1024` | 0.061 | 0.206 | [0.196, 0.216] | 12.852 | [12.266, 13.483] | 0.0 | 4.46e-15 | 4.07e-14 | yes | 4 | 0.175 |
| final1 | multi-arm rotation | 32 | 1e-08 | `spectral_q1024` | 0.061 | 0.203 | [0.196, 0.210] | 14.698 | [13.977, 15.465] | 0.0 | 4.46e-15 | 4.07e-14 | yes | 2 | 0.173 |
| final1 | multi-arm rotation | 32 | 1e-10 | `spectral_q1024` | 0.061 | 0.202 | [0.198, 0.207] | 16.725 | [16.045, 17.474] | 0.0 | 4.46e-15 | 4.07e-14 | yes | 1 | 0.173 |
| final1 | multi-arm rotation | 64 | 1e-06 | `lmmean_cfull_q0` | 3.756 | 8.692 | [7.848, 9.549] | 0.576 | [0.520, 0.640] | 150.5 | 5.84e-02 | 9.76e-07 | yes | 1 | 0.155 |
| final1 | multi-arm rotation | 64 | 1e-08 | `lmmean_cfull_q0` | 3.756 | 9.464 | [8.594, 10.340] | 0.619 | [0.559, 0.685] | 180.7 | 5.84e-02 | 9.71e-09 | yes | 1 | 0.147 |
| final1 | multi-arm rotation | 64 | 1e-10 | `lmmean_cfull_q0` | 3.756 | 10.496 | [9.620, 11.401] | 0.623 | [0.570, 0.682] | 210.8 | 5.84e-02 | 9.87e-11 | yes | 1 | 0.154 |
| final1 | multi-arm rotation | 64 | 1e-06 | `lmtrmean_c64_q0` | 3.072 | 8.052 | [7.198, 9.108] | 0.622 | [0.549, 0.704] | 151.2 | 5.59e-02 | 9.67e-07 | yes | 0 | 0.155 |
| final1 | multi-arm rotation | 64 | 1e-08 | `lmtrmean_c64_q0` | 3.072 | 8.842 | [8.061, 9.819] | 0.662 | [0.590, 0.734] | 179.8 | 5.59e-02 | 9.87e-09 | yes | 0 | 0.147 |
| final1 | multi-arm rotation | 64 | 1e-10 | `lmtrmean_c64_q0` | 3.072 | 9.828 | [8.971, 10.840] | 0.665 | [0.598, 0.734] | 209.8 | 5.59e-02 | 9.99e-11 | yes | 0 | 0.154 |
| final1 | multi-arm rotation | 64 | 1e-06 | `groupn_rt30_c64_q0` | 1.640 | 6.995 | [6.517, 7.442] | 0.716 | [0.668, 0.771] | 162.8 | 4.45e-01 | 9.84e-07 | yes | 0 | 0.155 |
| final1 | multi-arm rotation | 64 | 1e-08 | `groupn_rt30_c64_q0` | 1.640 | 7.773 | [7.231, 8.267] | 0.753 | [0.709, 0.806] | 192.7 | 4.45e-01 | 9.69e-09 | yes | 0 | 0.147 |
| final1 | multi-arm rotation | 64 | 1e-10 | `groupn_rt30_c64_q0` | 1.640 | 8.529 | [7.966, 8.988] | 0.766 | [0.725, 0.816] | 219.0 | 4.45e-01 | 9.91e-11 | yes | 0 | 0.154 |
| final1 | multi-arm rotation | 64 | 1e-06 | `groupn_rt30_c64_q8` | 1.664 | 3.229 | [2.774, 3.649] | 1.551 | [1.361, 1.817] | 39.7 | 1.48e-01 | 9.14e-07 | yes | 0 | 0.155 |
| final1 | multi-arm rotation | 64 | 1e-08 | `groupn_rt30_c64_q8` | 1.664 | 3.622 | [3.135, 4.071] | 1.616 | [1.437, 1.859] | 53.8 | 1.48e-01 | 9.92e-09 | yes | 0 | 0.147 |
| final1 | multi-arm rotation | 64 | 1e-10 | `groupn_rt30_c64_q8` | 1.664 | 4.034 | [3.609, 4.450] | 1.620 | [1.464, 1.824] | 67.8 | 1.48e-01 | 9.19e-11 | yes | 0 | 0.154 |
| final1 | multi-arm rotation | 64 | 1e-06 | `spectral_q512` | 0.065 | 0.232 | [0.226, 0.241] | 21.557 | [20.809, 22.271] | 0.0 | 1.00e-14 | 2.37e-13 | yes | 1 | 0.155 |
| final1 | multi-arm rotation | 64 | 1e-08 | `spectral_q512` | 0.065 | 0.218 | [0.212, 0.226] | 26.905 | [26.065, 27.714] | 0.0 | 1.00e-14 | 2.37e-13 | yes | 4 | 0.147 |
| final1 | multi-arm rotation | 64 | 1e-10 | `spectral_q512` | 0.065 | 0.220 | [0.215, 0.226] | 29.674 | [28.984, 30.520] | 0.0 | 1.00e-14 | 2.37e-13 | yes | 1 | 0.154 |
| final1 | multi-arm rotation | 64 | 1e-06 | `spectral_q1024` | 0.065 | 0.216 | [0.209, 0.223] | 23.155 | [22.110, 24.154] | 0.0 | 1.00e-14 | 2.37e-13 | yes | 3 | 0.155 |
| final1 | multi-arm rotation | 64 | 1e-08 | `spectral_q1024` | 0.065 | 0.216 | [0.207, 0.225] | 27.088 | [26.059, 28.237] | 0.0 | 1.00e-14 | 2.37e-13 | yes | 3 | 0.147 |
| final1 | multi-arm rotation | 64 | 1e-10 | `spectral_q1024` | 0.065 | 0.217 | [0.207, 0.231] | 30.071 | [28.409, 31.605] | 0.0 | 1.00e-14 | 2.37e-13 | yes | 2 | 0.154 |
| final1 | multi-arm rotation | 128 | 1e-06 | `lmmean_cfull_q0` | 3.937 | 13.446 | [12.582, 14.388] | 0.730 | [0.681, 0.779] | 306.0 | 5.96e-02 | 9.78e-07 | yes | 0 | 0.180 |
| final1 | multi-arm rotation | 128 | 1e-08 | `lmmean_cfull_q0` | 3.937 | 15.619 | [14.824, 16.549] | 0.765 | [0.719, 0.811] | 368.2 | 5.96e-02 | 9.86e-09 | yes | 0 | 0.193 |
| final1 | multi-arm rotation | 128 | 1e-10 | `lmmean_cfull_q0` | 3.937 | 17.306 | [16.547, 18.110] | 0.770 | [0.730, 0.811] | 427.7 | 5.96e-02 | 9.95e-11 | yes | 1 | 0.188 |
| final1 | multi-arm rotation | 128 | 1e-06 | `lmtrmean_c64_q0` | 3.032 | 12.788 | [11.955, 13.794] | 0.767 | [0.712, 0.827] | 310.3 | 5.72e-02 | 9.63e-07 | yes | 0 | 0.180 |
| final1 | multi-arm rotation | 128 | 1e-08 | `lmtrmean_c64_q0` | 3.032 | 14.898 | [14.095, 15.865] | 0.802 | [0.751, 0.854] | 368.8 | 5.72e-02 | 9.94e-09 | yes | 0 | 0.193 |
| final1 | multi-arm rotation | 128 | 1e-10 | `lmtrmean_c64_q0` | 3.032 | 16.582 | [15.547, 17.703] | 0.804 | [0.755, 0.854] | 428.8 | 5.72e-02 | 9.92e-11 | yes | 0 | 0.188 |
| final1 | multi-arm rotation | 128 | 1e-06 | `groupn_rt30_c64_q0` | 1.649 | 12.002 | [11.543, 12.387] | 0.818 | [0.790, 0.850] | 334.2 | 4.44e-01 | 9.84e-07 | yes | 0 | 0.180 |
| final1 | multi-arm rotation | 128 | 1e-08 | `groupn_rt30_c64_q0` | 1.649 | 14.109 | [13.567, 14.575] | 0.847 | [0.820, 0.877] | 394.0 | 4.44e-01 | 9.98e-09 | yes | 0 | 0.193 |
| final1 | multi-arm rotation | 128 | 1e-10 | `groupn_rt30_c64_q0` | 1.649 | 15.507 | [14.682, 16.224] | 0.860 | [0.832, 0.896] | 444.8 | 4.44e-01 | 9.99e-11 | yes | 0 | 0.188 |
| final1 | multi-arm rotation | 128 | 1e-06 | `groupn_rt30_c64_q8` | 1.663 | 4.506 | [4.099, 4.866] | 2.177 | [2.008, 2.390] | 83.2 | 1.47e-01 | 9.76e-07 | yes | 0 | 0.180 |
| final1 | multi-arm rotation | 128 | 1e-08 | `groupn_rt30_c64_q8` | 1.663 | 5.566 | [5.109, 6.010] | 2.146 | [2.000, 2.340] | 112.0 | 1.47e-01 | 9.71e-09 | yes | 0 | 0.193 |
| final1 | multi-arm rotation | 128 | 1e-10 | `groupn_rt30_c64_q8` | 1.663 | 6.312 | [5.802, 6.788] | 2.112 | [1.986, 2.268] | 140.2 | 1.47e-01 | 9.96e-11 | yes | 0 | 0.188 |
| final1 | multi-arm rotation | 128 | 1e-06 | `spectral_q512` | 0.073 | 0.242 | [0.228, 0.256] | 40.604 | [38.440, 42.564] | 0.0 | 3.05e-14 | 1.30e-12 | yes | 1 | 0.180 |
| final1 | multi-arm rotation | 128 | 1e-08 | `spectral_q512` | 0.073 | 0.250 | [0.237, 0.265] | 47.688 | [45.328, 50.255] | 0.0 | 3.05e-14 | 1.30e-12 | yes | 1 | 0.193 |
| final1 | multi-arm rotation | 128 | 1e-10 | `spectral_q512` | 0.073 | 0.245 | [0.233, 0.257] | 54.470 | [51.978, 56.504] | 0.0 | 3.05e-14 | 1.30e-12 | yes | 1 | 0.188 |
| final1 | multi-arm rotation | 128 | 1e-06 | `spectral_q1024` | 0.073 | 0.227 | [0.216, 0.238] | 43.308 | [41.282, 45.536] | 0.0 | 3.05e-14 | 1.30e-12 | yes | 1 | 0.180 |
| final1 | multi-arm rotation | 128 | 1e-08 | `spectral_q1024` | 0.073 | 0.245 | [0.234, 0.259] | 48.686 | [46.439, 51.140] | 0.0 | 3.05e-14 | 1.30e-12 | yes | 1 | 0.193 |
| final1 | multi-arm rotation | 128 | 1e-10 | `spectral_q1024` | 0.073 | 0.242 | [0.234, 0.248] | 55.185 | [54.101, 56.683] | 0.0 | 3.05e-14 | 1.30e-12 | yes | 1 | 0.188 |
| final1 | multi-arm rotation | 256 | 1e-06 | `lmmean_cfull_q0` | 4.776 | 23.797 | [22.757, 24.840] | 0.845 | [0.812, 0.880] | 622.8 | 6.00e-02 | 9.97e-07 | yes | 0 | 0.159 |
| final1 | multi-arm rotation | 256 | 1e-08 | `lmmean_cfull_q0` | 4.776 | 27.290 | [26.317, 28.326] | 0.862 | [0.828, 0.896] | 752.0 | 6.00e-02 | 9.96e-09 | yes | 0 | 0.150 |
| final1 | multi-arm rotation | 256 | 1e-10 | `lmmean_cfull_q0` | 4.776 | 31.042 | [30.197, 31.943] | 0.853 | [0.827, 0.879] | 867.7 | 6.00e-02 | 9.82e-11 | yes | 0 | 0.145 |
| final1 | multi-arm rotation | 256 | 1e-06 | `lmtrmean_c64_q0` | 3.111 | 22.386 | [21.472, 23.418] | 0.899 | [0.861, 0.940] | 631.8 | 6.55e-02 | 9.92e-07 | yes | 0 | 0.159 |
| final1 | multi-arm rotation | 256 | 1e-08 | `lmtrmean_c64_q0` | 3.111 | 25.612 | [24.689, 26.646] | 0.919 | [0.884, 0.955] | 752.5 | 6.55e-02 | 9.89e-09 | yes | 0 | 0.150 |
| final1 | multi-arm rotation | 256 | 1e-10 | `lmtrmean_c64_q0` | 3.111 | 29.490 | [28.510, 30.629] | 0.898 | [0.865, 0.932] | 870.7 | 6.55e-02 | 9.96e-11 | yes | 0 | 0.145 |
| final1 | multi-arm rotation | 256 | 1e-06 | `groupn_rt30_c64_q0` | 1.651 | 22.704 | [22.078, 23.182] | 0.886 | [0.864, 0.914] | 681.5 | 4.45e-01 | 9.99e-07 | yes | 0 | 0.159 |
| final1 | multi-arm rotation | 256 | 1e-08 | `groupn_rt30_c64_q0` | 1.651 | 25.999 | [25.236, 26.600] | 0.905 | [0.887, 0.930] | 800.2 | 4.45e-01 | 1.00e-08 | yes | 0 | 0.150 |
| final1 | multi-arm rotation | 256 | 1e-10 | `groupn_rt30_c64_q0` | 1.651 | 29.051 | [28.185, 29.741] | 0.912 | [0.894, 0.936] | 902.3 | 4.45e-01 | 9.99e-11 | yes | 0 | 0.145 |
| final1 | multi-arm rotation | 256 | 1e-06 | `groupn_rt30_c64_q8` | 1.672 | 7.175 | [6.664, 7.598] | 2.804 | [2.639, 3.024] | 173.7 | 1.50e-01 | 9.77e-07 | yes | 0 | 0.159 |
| final1 | multi-arm rotation | 256 | 1e-08 | `groupn_rt30_c64_q8` | 1.672 | 8.855 | [8.379, 9.253] | 2.658 | [2.550, 2.803] | 230.3 | 1.50e-01 | 9.92e-09 | yes | 0 | 0.150 |
| final1 | multi-arm rotation | 256 | 1e-10 | `groupn_rt30_c64_q8` | 1.672 | 10.588 | [10.112, 11.001] | 2.502 | [2.418, 2.613] | 287.0 | 1.50e-01 | 9.82e-11 | yes | 0 | 0.145 |
| final1 | multi-arm rotation | 256 | 1e-06 | `spectral_q512` | 0.080 | 0.246 | [0.244, 0.248] | 81.655 | [79.997, 83.092] | 0.0 | 6.17e-14 | 7.65e-12 | yes | 3 | 0.159 |
| final1 | multi-arm rotation | 256 | 1e-08 | `spectral_q512` | 0.080 | 0.249 | [0.241, 0.256] | 94.617 | [91.687, 97.669] | 0.0 | 6.17e-14 | 7.65e-12 | yes | 1 | 0.150 |
| final1 | multi-arm rotation | 256 | 1e-10 | `spectral_q512` | 0.080 | 0.247 | [0.237, 0.257] | 107.296 | [102.844, 111.687] | 0.0 | 6.17e-14 | 7.65e-12 | yes | 2 | 0.145 |
| final1 | multi-arm rotation | 256 | 1e-06 | `spectral_q1024` | 0.081 | 0.239 | [0.237, 0.241] | 84.234 | [82.327, 85.783] | 0.0 | 6.17e-14 | 7.65e-12 | yes | 2 | 0.159 |
| final1 | multi-arm rotation | 256 | 1e-08 | `spectral_q1024` | 0.081 | 0.234 | [0.228, 0.244] | 100.457 | [96.674, 103.395] | 0.0 | 6.17e-14 | 7.65e-12 | yes | 1 | 0.150 |
| final1 | multi-arm rotation | 256 | 1e-10 | `spectral_q1024` | 0.081 | 0.231 | [0.226, 0.236] | 114.778 | [112.218, 117.694] | 0.0 | 6.17e-14 | 7.65e-12 | yes | 1 | 0.145 |
| final1 | multi-arm rotation | 512 | 1e-06 | `lmmean_cfull_q0` | 8.186 | 66.785 | [64.484, 68.793] | 0.932 | [0.905, 0.963] | 1267.5 | 6.01e-02 | 9.98e-07 | yes | 0 | 0.343 |
| final1 | multi-arm rotation | 512 | 1e-08 | `lmmean_cfull_q0` | 8.186 | 78.017 | [76.479, 79.641] | 0.938 | [0.917, 0.959] | 1524.5 | 6.01e-02 | 9.97e-09 | yes | 0 | 0.338 |
| final1 | multi-arm rotation | 512 | 1e-10 | `lmmean_cfull_q0` | 8.186 | 89.454 | [88.672, 90.314] | 0.915 | [0.903, 0.926] | 1756.3 | 6.01e-02 | 7.04e-11 | yes | 0 | 0.339 |
| final1 | multi-arm rotation | 512 | 1e-06 | `groupn_rt30_c64_q0` | 1.668 | 65.889 | [64.629, 66.930] | 0.945 | [0.930, 0.966] | 1388.7 | 4.46e-01 | 1.00e-06 | yes | 0 | 0.343 |
| final1 | multi-arm rotation | 512 | 1e-08 | `groupn_rt30_c64_q0` | 1.668 | 76.020 | [74.287, 77.287] | 0.963 | [0.947, 0.984] | 1620.2 | 4.46e-01 | 9.93e-09 | yes | 0 | 0.338 |
| final1 | multi-arm rotation | 512 | 1e-10 | `groupn_rt30_c64_q0` | 1.668 | 86.608 | [85.180, 87.712] | 0.945 | [0.937, 0.956] | 1830.3 | 4.46e-01 | 7.05e-11 | yes | 0 | 0.339 |
| final1 | multi-arm rotation | 512 | 1e-06 | `groupn_rt30_c64_q8` | 1.700 | 18.639 | [18.042, 19.177] | 3.340 | [3.242, 3.460] | 355.3 | 1.53e-01 | 9.99e-07 | yes | 0 | 0.343 |
| final1 | multi-arm rotation | 512 | 1e-08 | `groupn_rt30_c64_q8` | 1.700 | 24.021 | [23.466, 24.531] | 3.047 | [2.998, 3.113] | 471.5 | 1.53e-01 | 9.96e-09 | yes | 0 | 0.338 |
| final1 | multi-arm rotation | 512 | 1e-10 | `groupn_rt30_c64_q8` | 1.700 | 29.332 | [28.762, 29.799] | 2.791 | [2.756, 2.837] | 585.5 | 1.53e-01 | 1.00e-10 | yes | 0 | 0.339 |
| final1 | multi-arm rotation | 512 | 1e-06 | `spectral_q512` | 0.159 | 0.350 | [0.335, 0.364] | 177.830 | [171.219, 186.174] | 0.0 | 1.70e-13 | 4.21e-11 | yes | 2 | 0.343 |
| final1 | multi-arm rotation | 512 | 1e-08 | `spectral_q512` | 0.159 | 0.334 | [0.326, 0.342] | 219.094 | [213.983, 225.181] | 0.0 | 1.70e-13 | 4.21e-11 | yes | 1 | 0.338 |
| final1 | multi-arm rotation | 512 | 1e-10 | `spectral_q512` | 0.159 | 0.327 | [0.316, 0.338] | 250.473 | [241.591, 260.095] | 0.0 | 1.70e-13 | 4.21e-11 | yes | 1 | 0.339 |
| final1 | multi-arm rotation | 512 | 1e-06 | `spectral_q1024` | 0.160 | 0.322 | [0.312, 0.334] | 193.227 | [186.668, 199.285] | 0.0 | 1.70e-13 | 4.21e-11 | yes | 1 | 0.343 |
| final1 | multi-arm rotation | 512 | 1e-08 | `spectral_q1024` | 0.160 | 0.313 | [0.308, 0.321] | 233.885 | [229.434, 237.377] | 0.0 | 1.70e-13 | 4.21e-11 | yes | 1 | 0.338 |
| final1 | multi-arm rotation | 512 | 1e-10 | `spectral_q1024` | 0.160 | 0.311 | [0.307, 0.316] | 262.935 | [258.525, 267.061] | 0.0 | 1.70e-13 | 4.21e-11 | yes | 1 | 0.339 |
| final1 | multi-arm rotation | 1024 | 1e-06 | `lmmean_cfull_q0` | 21.125 | 229.748 | [222.311, 235.040] | 0.993 | [0.965, 1.027] | 2575.7 | 6.01e-02 | 9.97e-07 | yes | 0 | 1.095 |
| final1 | multi-arm rotation | 1024 | 1e-08 | `lmmean_cfull_q0` | 21.125 | 272.502 | [267.761, 277.332] | 0.990 | [0.966, 1.012] | 3093.2 | 6.01e-02 | 9.99e-09 | yes | 0 | 1.096 |
| final1 | multi-arm rotation | 1024 | 1e-10 | `lmmean_cfull_q0` | 21.125 | 309.347 | [306.949, 311.598] | 0.978 | [0.969, 0.987] | 3550.0 | 6.01e-02 | 9.98e-11 | yes | 0 | 1.095 |
| final1 | multi-arm rotation | 1024 | 1e-06 | `groupn_rt30_c64_q0` | 1.682 | 230.713 | [224.669, 236.498] | 0.989 | [0.965, 1.019] | 2815.7 | 4.47e-01 | 9.99e-07 | yes | 0 | 1.095 |
| final1 | multi-arm rotation | 1024 | 1e-08 | `groupn_rt30_c64_q0` | 1.682 | 269.228 | [261.589, 275.765] | 1.002 | [0.977, 1.032] | 3278.0 | 4.47e-01 | 9.94e-09 | yes | 0 | 1.096 |
| final1 | multi-arm rotation | 1024 | 1e-10 | `groupn_rt30_c64_q0` | 1.682 | 304.216 | [298.764, 309.844] | 0.995 | [0.978, 1.012] | 3696.7 | 4.47e-01 | 9.49e-11 | yes | 0 | 1.095 |
| final1 | multi-arm rotation | 1024 | 1e-06 | `groupn_rt30_c64_q8` | 1.754 | 61.921 | [60.193, 63.525] | 3.686 | [3.571, 3.799] | 730.3 | 1.54e-01 | 9.98e-07 | yes | 0 | 1.095 |
| final1 | multi-arm rotation | 1024 | 1e-08 | `groupn_rt30_c64_q8` | 1.754 | 81.103 | [79.305, 83.020] | 3.326 | [3.254, 3.391] | 962.3 | 1.54e-01 | 1.00e-08 | yes | 0 | 1.096 |
| final1 | multi-arm rotation | 1024 | 1e-10 | `groupn_rt30_c64_q8` | 1.754 | 99.834 | [98.189, 101.499] | 3.031 | [2.984, 3.078] | 1191.3 | 1.54e-01 | 9.04e-11 | yes | 0 | 1.095 |
| final1 | multi-arm rotation | 1024 | 1e-06 | `spectral_q512` | 0.300 | 0.759 | [0.590, 0.924] | 300.855 | [250.657, 382.019] | 2.3 | 1.78e-06 | 3.37e-07 | yes | 1 | 1.095 |
| final1 | multi-arm rotation | 1024 | 1e-08 | `spectral_q512` | 0.300 | 0.908 | [0.663, 1.152] | 297.272 | [235.410, 412.657] | 4.2 | 1.78e-06 | 6.72e-09 | yes | 1 | 1.096 |
| final1 | multi-arm rotation | 1024 | 1e-10 | `spectral_q512` | 0.300 | 1.130 | [0.771, 1.487] | 267.742 | [204.160, 391.296] | 6.8 | 1.78e-06 | 7.31e-11 | yes | 1 | 1.095 |
| final1 | multi-arm rotation | 1024 | 1e-06 | `spectral_q1024` | 0.635 | 0.803 | [0.792, 0.819] | 284.043 | [279.179, 288.461] | 0.0 | 4.15e-13 | 2.48e-10 | yes | 0 | 1.095 |
| final1 | multi-arm rotation | 1024 | 1e-08 | `spectral_q1024` | 0.635 | 0.814 | [0.807, 0.821] | 331.315 | [328.388, 334.629] | 0.0 | 4.15e-13 | 2.48e-10 | yes | 0 | 1.096 |
| final1 | multi-arm rotation | 1024 | 1e-10 | `spectral_q1024` | 0.635 | 0.931 | [0.865, 0.995] | 325.154 | [303.840, 350.296] | 0.7 | 4.15e-13 | 8.62e-11 | yes | 0 | 1.095 |
| pairfinal1 | balanced AB/BA | 512 | 1e-06 | `lmtrmean_c64_q0` | 4.236 | 62.006 | [59.055, 63.281] | 1.004 | [0.984, 1.058] | 1262.2 | 7.87e-02 | 9.96e-07 | yes | 0 | — |
| pairfinal1 | balanced AB/BA | 512 | 1e-08 | `lmtrmean_c64_q0` | 4.236 | 73.868 | [71.799, 74.830] | 0.998 | [0.973, 1.018] | 1521.8 | 7.87e-02 | 9.99e-09 | yes | 0 | — |
| pairfinal1 | balanced AB/BA | 512 | 1e-10 | `lmtrmean_c64_q0` | 4.236 | 85.432 | [83.195, 86.409] | 0.965 | [0.941, 0.981] | 1764.0 | 7.87e-02 | 1.00e-10 | yes | 0 | — |
| pairfinal1 | balanced AB/BA | 1024 | 1e-06 | `lmtrmean_c64_q0` | 4.255 | 211.996 | [203.201, 214.291] | 1.026 | [1.014, 1.077] | 2571.1 | 8.06e-02 | 9.99e-07 | yes | 0 | — |
| pairfinal1 | balanced AB/BA | 1024 | 1e-08 | `lmtrmean_c64_q0` | 4.255 | 253.531 | [247.292, 255.137] | 1.014 | [0.993, 1.031] | 3087.1 | 8.06e-02 | 9.98e-09 | yes | 0 | — |
| pairfinal1 | balanced AB/BA | 1024 | 1e-10 | `lmtrmean_c64_q0` | 4.255 | 290.348 | [283.919, 291.603] | 0.994 | [0.980, 1.010] | 3563.8 | 8.06e-02 | 8.15e-11 | yes | 0 | — |

## Classical and native baselines

These are from the same authoritative rotated blocks as the end-to-end rows.

| run | N | tolerance | baseline | total ms | total 95% CI ms | iterations | outliers | max residual | meets tau | mean relative L2 |
|---|---:|---:|---|---:|---|---:|---:|---:|---|---:|
| gcal1 | 256 | 1e-06 | `zero_cg` | 21.980 | — | 667.0 | — | 9.99e-07 | yes | 3.47e-08 |
| gcal1 | 256 | 1e-06 | `fft_dst_direct` | 0.163 | — | — | — | 1.19e-12 | yes | 0.00e+00 |
| gcal1 | 256 | 1e-06 | `native_zero` | 21.633 | — | — | — | 9.99e-07 | yes | 3.47e-08 |
| gcal1 | 256 | 1e-06 | `native_spectral_q8` | 4.993 | — | — | — | 9.74e-07 | yes | 1.11e-08 |
| gcal1 | 256 | 1e-06 | `native_groupn_rt10_c64_q8` | 8.381 | — | — | — | 9.76e-07 | yes | 1.13e-08 |
| specdev1 | 256 | 1e-06 | `zero_cg` | 19.992 | [19.650, 20.377] | 659.2 | 0 | 9.99e-07 | yes | 3.64e-08 |
| specdev1 | 256 | 1e-06 | `fft_dst_direct` | 0.135 | [0.132, 0.140] | — | 1 | 1.19e-12 | yes | 0.00e+00 |
| specdev1 | 256 | 1e-06 | `native_zero` | 19.318 | [18.972, 19.689] | — | 0 | 9.99e-07 | yes | 3.64e-08 |
| specdev1 | 256 | 1e-06 | `native_jacobi_zero` | 23.383 | [22.926, 23.870] | — | 0 | 9.99e-07 | yes | 3.64e-08 |
| specdev1 | 256 | 1e-06 | `native_spectral_q16` | 1.605 | [1.230, 1.964] | — | 0 | 9.69e-07 | yes | 3.67e-09 |
| specdev1 | 256 | 1e-06 | `native_spectral_q32` | 0.562 | [0.246, 0.898] | — | 0 | 9.44e-07 | yes | 1.63e-09 |
| specdev1 | 256 | 1e-06 | `native_spectral_q64` | 0.294 | [0.153, 0.456] | — | 0 | 9.79e-07 | yes | 3.04e-10 |
| specdev1 | 512 | 1e-06 | `zero_cg` | 63.488 | [61.963, 65.405] | 1342.2 | 0 | 1.00e-06 | yes | 2.57e-08 |
| specdev1 | 512 | 1e-06 | `fft_dst_direct` | 0.351 | [0.346, 0.356] | — | 0 | 9.24e-12 | yes | 0.00e+00 |
| specdev1 | 512 | 1e-06 | `native_zero` | 62.733 | [61.169, 64.701] | — | 0 | 1.00e-06 | yes | 2.57e-08 |
| specdev1 | 512 | 1e-06 | `native_jacobi_zero` | 68.892 | [67.173, 71.104] | — | 0 | 1.00e-06 | yes | 2.57e-08 |
| specdev1 | 512 | 1e-06 | `native_spectral_q16` | 5.251 | [3.917, 6.584] | — | 0 | 9.44e-07 | yes | 2.93e-09 |
| specdev1 | 512 | 1e-06 | `native_spectral_q32` | 1.730 | [0.607, 2.970] | — | 0 | 9.56e-07 | yes | 1.43e-09 |
| specdev1 | 512 | 1e-06 | `native_spectral_q64` | 0.841 | [0.313, 1.414] | — | 1 | 8.57e-07 | yes | 1.67e-10 |
| specdev1 | 1024 | 1e-06 | `zero_cg` | 219.815 | [216.443, 223.370] | 2725.2 | 0 | 9.99e-07 | yes | 1.85e-08 |
| specdev1 | 1024 | 1e-06 | `fft_dst_direct` | 1.105 | [1.096, 1.114] | — | 0 | 2.68e-11 | yes | 0.00e+00 |
| specdev1 | 1024 | 1e-06 | `native_zero` | 221.991 | [217.824, 226.238] | — | 0 | 9.99e-07 | yes | 1.85e-08 |
| specdev1 | 1024 | 1e-06 | `native_jacobi_zero` | 246.511 | [242.911, 250.408] | — | 0 | 1.00e-06 | yes | 1.86e-08 |
| specdev1 | 1024 | 1e-06 | `native_spectral_q16` | 18.862 | [14.224, 23.330] | — | 0 | 9.98e-07 | yes | 2.16e-09 |
| specdev1 | 1024 | 1e-06 | `native_spectral_q32` | 5.983 | [1.916, 10.454] | — | 1 | 9.99e-07 | yes | 1.30e-09 |
| specdev1 | 1024 | 1e-06 | `native_spectral_q64` | 2.779 | [0.880, 4.816] | — | 2 | 9.60e-07 | yes | 1.26e-10 |
| specdev2 | 256 | 1e-06 | `zero_cg` | 21.525 | [21.045, 22.048] | 659.2 | 0 | 9.99e-07 | yes | 3.64e-08 |
| specdev2 | 256 | 1e-06 | `fft_dst_direct` | 0.138 | [0.137, 0.140] | — | 2 | 1.19e-12 | yes | 0.00e+00 |
| specdev2 | 256 | 1e-06 | `native_zero` | 21.172 | [20.769, 21.598] | — | 0 | 9.99e-07 | yes | 3.64e-08 |
| specdev2 | 256 | 1e-06 | `native_jacobi_zero` | 24.305 | [23.760, 24.861] | — | 0 | 9.99e-07 | yes | 3.64e-08 |
| specdev2 | 256 | 1e-06 | `native_spectral_q64` | 0.322 | [0.169, 0.487] | — | 4 | 9.79e-07 | yes | 3.04e-10 |
| specdev2 | 256 | 1e-06 | `native_spectral_q128` | 0.213 | [0.153, 0.279] | — | 3 | 4.70e-07 | yes | 5.25e-11 |
| specdev2 | 512 | 1e-06 | `zero_cg` | 54.122 | [53.232, 55.055] | 1342.2 | 0 | 1.00e-06 | yes | 2.57e-08 |
| specdev2 | 512 | 1e-06 | `fft_dst_direct` | 0.332 | [0.331, 0.335] | — | 0 | 9.24e-12 | yes | 0.00e+00 |
| specdev2 | 512 | 1e-06 | `native_zero` | 58.166 | [57.163, 59.214] | — | 0 | 1.00e-06 | yes | 2.57e-08 |
| specdev2 | 512 | 1e-06 | `native_jacobi_zero` | 70.409 | [69.259, 71.597] | — | 0 | 1.00e-06 | yes | 2.57e-08 |
| specdev2 | 512 | 1e-06 | `native_spectral_q64` | 0.739 | [0.288, 1.244] | — | 2 | 8.57e-07 | yes | 1.67e-10 |
| specdev2 | 512 | 1e-06 | `native_spectral_q128` | 0.409 | [0.208, 0.630] | — | 1 | 7.55e-07 | yes | 5.70e-11 |
| specdev2 | 1024 | 1e-06 | `zero_cg` | 227.641 | [224.153, 231.370] | 2725.2 | 0 | 9.99e-07 | yes | 1.85e-08 |
| specdev2 | 1024 | 1e-06 | `fft_dst_direct` | 1.276 | [1.274, 1.278] | — | 0 | 2.68e-11 | yes | 0.00e+00 |
| specdev2 | 1024 | 1e-06 | `native_zero` | 223.683 | [221.276, 226.231] | — | 0 | 9.99e-07 | yes | 1.85e-08 |
| specdev2 | 1024 | 1e-06 | `native_jacobi_zero` | 246.402 | [242.919, 250.405] | — | 0 | 1.00e-06 | yes | 1.86e-08 |
| specdev2 | 1024 | 1e-06 | `native_spectral_q64` | 2.887 | [0.914, 5.023] | — | 0 | 9.60e-07 | yes | 1.26e-10 |
| specdev2 | 1024 | 1e-06 | `native_spectral_q128` | 1.454 | [0.540, 2.447] | — | 0 | 8.83e-07 | yes | 4.03e-11 |
| specdev3 | 256 | 1e-06 | `zero_cg` | 20.020 | [19.690, 20.390] | 659.2 | 0 | 9.99e-07 | yes | 3.64e-08 |
| specdev3 | 256 | 1e-06 | `fft_dst_direct` | 0.141 | [0.137, 0.148] | — | 2 | 1.19e-12 | yes | 0.00e+00 |
| specdev3 | 256 | 1e-06 | `native_zero` | 19.665 | [19.329, 20.033] | — | 0 | 9.99e-07 | yes | 3.64e-08 |
| specdev3 | 256 | 1e-06 | `native_jacobi_zero` | 24.084 | [23.674, 24.567] | — | 0 | 9.99e-07 | yes | 3.64e-08 |
| specdev3 | 256 | 1e-06 | `native_spectral_q128` | 0.204 | [0.148, 0.265] | — | 3 | 4.70e-07 | yes | 5.25e-11 |
| specdev3 | 256 | 1e-06 | `native_spectral_q256` | 0.128 | [0.126, 0.131] | — | 6 | 4.87e-12 | yes | 1.02e-15 |
| specdev3 | 512 | 1e-06 | `zero_cg` | 62.372 | [61.369, 63.415] | 1342.2 | 0 | 1.00e-06 | yes | 2.57e-08 |
| specdev3 | 512 | 1e-06 | `fft_dst_direct` | 0.339 | [0.332, 0.346] | — | 0 | 9.24e-12 | yes | 0.00e+00 |
| specdev3 | 512 | 1e-06 | `native_zero` | 61.588 | [60.647, 62.558] | — | 0 | 1.00e-06 | yes | 2.57e-08 |
| specdev3 | 512 | 1e-06 | `native_jacobi_zero` | 67.886 | [66.824, 68.977] | — | 0 | 1.00e-06 | yes | 2.57e-08 |
| specdev3 | 512 | 1e-06 | `native_spectral_q128` | 0.437 | [0.220, 0.678] | — | 3 | 7.55e-07 | yes | 5.70e-11 |
| specdev3 | 512 | 1e-06 | `native_spectral_q256` | 0.278 | [0.193, 0.372] | — | 2 | 8.90e-07 | yes | 2.13e-11 |
| specdev3 | 1024 | 1e-06 | `zero_cg` | 220.838 | [217.412, 224.376] | 2725.2 | 0 | 9.99e-07 | yes | 1.85e-08 |
| specdev3 | 1024 | 1e-06 | `fft_dst_direct` | 1.085 | [1.082, 1.090] | — | 0 | 2.68e-11 | yes | 0.00e+00 |
| specdev3 | 1024 | 1e-06 | `native_zero` | 210.586 | [207.689, 214.135] | — | 0 | 9.99e-07 | yes | 1.85e-08 |
| specdev3 | 1024 | 1e-06 | `native_jacobi_zero` | 240.038 | [236.209, 244.028] | — | 0 | 1.00e-06 | yes | 1.86e-08 |
| specdev3 | 1024 | 1e-06 | `native_spectral_q128` | 1.362 | [0.508, 2.290] | — | 0 | 8.83e-07 | yes | 4.03e-11 |
| specdev3 | 1024 | 1e-06 | `native_spectral_q256` | 0.737 | [0.379, 1.123] | — | 0 | 8.03e-07 | yes | 1.42e-11 |
| specdev4 | 1024 | 1e-06 | `zero_cg` | 216.476 | [212.736, 220.306] | 2725.2 | 0 | 9.99e-07 | yes | 1.85e-08 |
| specdev4 | 1024 | 1e-06 | `fft_dst_direct` | 1.087 | [1.079, 1.095] | — | 0 | 2.68e-11 | yes | 0.00e+00 |
| specdev4 | 1024 | 1e-06 | `native_zero` | 220.101 | [216.167, 224.121] | — | 0 | 9.99e-07 | yes | 1.85e-08 |
| specdev4 | 1024 | 1e-06 | `native_jacobi_zero` | 248.723 | [244.689, 253.072] | — | 0 | 1.00e-06 | yes | 1.86e-08 |
| specdev4 | 1024 | 1e-06 | `native_spectral_q256` | 0.743 | [0.377, 1.140] | — | 1 | 8.03e-07 | yes | 1.42e-11 |
| specdev4 | 1024 | 1e-06 | `native_spectral_q512` | 0.553 | [0.409, 0.711] | — | 0 | 6.76e-07 | yes | 4.79e-12 |
| specdev5 | 512 | 1e-06 | `zero_cg` | 62.297 | [61.266, 63.409] | 1342.2 | 0 | 1.00e-06 | yes | 2.57e-08 |
| specdev5 | 512 | 1e-06 | `fft_dst_direct` | 0.327 | [0.322, 0.333] | — | 0 | 9.24e-12 | yes | 0.00e+00 |
| specdev5 | 512 | 1e-06 | `dense_dst_direct` | 0.176 | [0.174, 0.177] | — | 3 | 2.80e-11 | yes | 1.52e-15 |
| specdev5 | 512 | 1e-06 | `native_zero` | 61.156 | [60.067, 62.247] | — | 0 | 1.00e-06 | yes | 2.57e-08 |
| specdev5 | 512 | 1e-06 | `native_jacobi_zero` | 67.581 | [66.462, 68.777] | — | 0 | 1.00e-06 | yes | 2.57e-08 |
| specdev5 | 512 | 1e-06 | `native_spectral_q512` | 0.212 | [0.210, 0.214] | — | 2 | 2.80e-11 | yes | 1.52e-15 |
| specdev5 | 512 | 1e-06 | `native_spectral_q1024` | 0.212 | [0.208, 0.219] | — | 1 | 2.80e-11 | yes | 1.52e-15 |
| specdev5 | 512 | 1e-08 | `zero_cg` | 73.343 | [72.603, 74.070] | 1580.8 | 0 | 9.98e-09 | yes | 1.90e-10 |
| specdev5 | 512 | 1e-08 | `fft_dst_direct` | 0.329 | [0.324, 0.335] | — | 0 | 9.24e-12 | yes | 0.00e+00 |
| specdev5 | 512 | 1e-08 | `dense_dst_direct` | 0.183 | [0.175, 0.195] | — | 2 | 2.80e-11 | yes | 1.52e-15 |
| specdev5 | 512 | 1e-08 | `native_zero` | 71.955 | [71.252, 72.639] | — | 0 | 9.98e-09 | yes | 1.90e-10 |
| specdev5 | 512 | 1e-08 | `native_jacobi_zero` | 79.568 | [78.790, 80.296] | — | 0 | 9.91e-09 | yes | 1.90e-10 |
| specdev5 | 512 | 1e-08 | `native_spectral_q512` | 0.223 | [0.209, 0.236] | — | 2 | 2.80e-11 | yes | 1.52e-15 |
| specdev5 | 512 | 1e-08 | `native_spectral_q1024` | 0.215 | [0.208, 0.223] | — | 1 | 2.80e-11 | yes | 1.52e-15 |
| specdev5 | 512 | 1e-10 | `zero_cg` | 82.656 | [81.912, 83.248] | 1780.8 | 0 | 7.18e-11 | yes | 1.80e-12 |
| specdev5 | 512 | 1e-10 | `fft_dst_direct` | 0.333 | [0.329, 0.338] | — | 0 | 9.24e-12 | yes | 0.00e+00 |
| specdev5 | 512 | 1e-10 | `dense_dst_direct` | 0.181 | [0.178, 0.185] | — | 3 | 2.80e-11 | yes | 1.52e-15 |
| specdev5 | 512 | 1e-10 | `native_zero` | 81.152 | [80.412, 81.766] | — | 0 | 1.09e-10 | no | 1.80e-12 |
| specdev5 | 512 | 1e-10 | `native_jacobi_zero` | 89.683 | [88.835, 90.334] | — | 0 | 1.10e-10 | no | 1.80e-12 |
| specdev5 | 512 | 1e-10 | `native_spectral_q512` | 0.222 | [0.214, 0.236] | — | 1 | 2.80e-11 | yes | 1.52e-15 |
| specdev5 | 512 | 1e-10 | `native_spectral_q1024` | 0.219 | [0.210, 0.234] | — | 2 | 2.80e-11 | yes | 1.52e-15 |
| specdev5 | 1024 | 1e-06 | `zero_cg` | 218.954 | [215.638, 222.445] | 2725.2 | 0 | 9.99e-07 | yes | 1.85e-08 |
| specdev5 | 1024 | 1e-06 | `fft_dst_direct` | 1.078 | [1.075, 1.082] | — | 0 | 2.68e-11 | yes | 0.00e+00 |
| specdev5 | 1024 | 1e-06 | `dense_dst_direct` | 0.646 | [0.642, 0.649] | — | 1 | 1.59e-10 | yes | 1.95e-15 |
| specdev5 | 1024 | 1e-06 | `native_zero` | 215.654 | [211.868, 219.875] | — | 0 | 9.99e-07 | yes | 1.85e-08 |
| specdev5 | 1024 | 1e-06 | `native_jacobi_zero` | 239.610 | [234.442, 245.230] | — | 0 | 1.00e-06 | yes | 1.86e-08 |
| specdev5 | 1024 | 1e-06 | `native_spectral_q512` | 0.553 | [0.415, 0.694] | — | 1 | 6.76e-07 | yes | 4.79e-12 |
| specdev5 | 1024 | 1e-06 | `native_spectral_q1024` | 0.702 | [0.697, 0.707] | — | 0 | 1.59e-10 | yes | 1.95e-15 |
| specdev5 | 1024 | 1e-08 | `zero_cg` | 257.185 | [254.849, 259.447] | 3197.8 | 0 | 9.90e-09 | yes | 1.31e-10 |
| specdev5 | 1024 | 1e-08 | `fft_dst_direct` | 1.084 | [1.079, 1.089] | — | 0 | 2.68e-11 | yes | 0.00e+00 |
| specdev5 | 1024 | 1e-08 | `dense_dst_direct` | 0.656 | [0.647, 0.672] | — | 0 | 1.59e-10 | yes | 1.95e-15 |
| specdev5 | 1024 | 1e-08 | `native_zero` | 253.546 | [250.206, 257.156] | — | 0 | 1.00e-08 | no | 1.31e-10 |
| specdev5 | 1024 | 1e-08 | `native_jacobi_zero` | 281.643 | [277.245, 286.770] | — | 0 | 9.90e-09 | yes | 1.30e-10 |
| specdev5 | 1024 | 1e-08 | `native_spectral_q512` | 0.693 | [0.464, 0.945] | — | 0 | 7.51e-09 | yes | 4.38e-14 |
| specdev5 | 1024 | 1e-08 | `native_spectral_q1024` | 0.708 | [0.699, 0.718] | — | 0 | 1.59e-10 | yes | 1.95e-15 |
| specdev5 | 1024 | 1e-10 | `zero_cg` | 289.176 | [286.524, 291.203] | 3597.0 | 0 | 9.44e-11 | yes | 1.23e-12 |
| specdev5 | 1024 | 1e-10 | `fft_dst_direct` | 1.074 | [1.071, 1.079] | — | 0 | 2.68e-11 | yes | 0.00e+00 |
| specdev5 | 1024 | 1e-10 | `dense_dst_direct` | 0.658 | [0.646, 0.671] | — | 0 | 1.59e-10 | no | 1.95e-15 |
| specdev5 | 1024 | 1e-10 | `native_zero` | 284.633 | [280.824, 288.188] | — | 0 | 2.72e-10 | no | 1.23e-12 |
| specdev5 | 1024 | 1e-10 | `native_jacobi_zero` | 316.273 | [311.448, 321.213] | — | 0 | 2.72e-10 | no | 1.23e-12 |
| specdev5 | 1024 | 1e-10 | `native_spectral_q512` | 0.831 | [0.490, 1.199] | — | 0 | 9.36e-11 | yes | 1.60e-15 |
| specdev5 | 1024 | 1e-10 | `native_spectral_q1024` | 0.730 | [0.699, 0.764] | — | 0 | 7.16e-11 | yes | 1.68e-15 |
| final1 | 32 | 1e-06 | `zero_cg` | 2.644 | [2.529, 2.725] | 76.3 | 0 | 9.90e-07 | yes | 9.27e-08 |
| final1 | 32 | 1e-06 | `fft_dst_direct` | 0.175 | [0.172, 0.180] | — | 2 | 3.15e-14 | yes | 0.00e+00 |
| final1 | 32 | 1e-06 | `dense_dst_direct` | 0.077 | [0.070, 0.086] | — | 5 | 4.07e-14 | yes | 6.15e-16 |
| final1 | 32 | 1e-06 | `native_zero` | 2.434 | [2.332, 2.510] | — | 0 | 9.90e-07 | yes | 9.27e-08 |
| final1 | 32 | 1e-06 | `native_jacobi_zero` | 2.599 | [2.482, 2.673] | — | 0 | 9.90e-07 | yes | 9.27e-08 |
| final1 | 32 | 1e-06 | `native_lmtrmean_c64_q0` | 5.485 | [4.687, 6.510] | — | 0 | 9.77e-07 | yes | 8.46e-08 |
| final1 | 32 | 1e-06 | `native_groupn_rt30_c64_q0` | 4.197 | [3.716, 4.671] | — | 0 | 9.71e-07 | yes | 8.85e-08 |
| final1 | 32 | 1e-06 | `native_spectral_q512` | 0.130 | [0.119, 0.143] | — | 5 | 4.07e-14 | yes | 6.15e-16 |
| final1 | 32 | 1e-06 | `native_spectral_q1024` | 0.121 | [0.115, 0.129] | — | 4 | 4.07e-14 | yes | 6.15e-16 |
| final1 | 32 | 1e-08 | `zero_cg` | 2.979 | [2.901, 3.067] | 91.3 | 0 | 9.95e-09 | yes | 7.68e-10 |
| final1 | 32 | 1e-08 | `fft_dst_direct` | 0.173 | [0.171, 0.174] | — | 0 | 3.15e-14 | yes | 0.00e+00 |
| final1 | 32 | 1e-08 | `dense_dst_direct` | 0.074 | [0.071, 0.080] | — | 3 | 4.07e-14 | yes | 6.15e-16 |
| final1 | 32 | 1e-08 | `native_zero` | 2.788 | [2.717, 2.861] | — | 0 | 9.95e-09 | yes | 7.68e-10 |
| final1 | 32 | 1e-08 | `native_jacobi_zero` | 2.981 | [2.903, 3.064] | — | 0 | 9.95e-09 | yes | 7.68e-10 |
| final1 | 32 | 1e-08 | `native_lmtrmean_c64_q0` | 5.761 | [4.978, 6.777] | — | 0 | 9.86e-09 | yes | 8.27e-10 |
| final1 | 32 | 1e-08 | `native_groupn_rt30_c64_q0` | 4.476 | [3.945, 4.945] | — | 0 | 9.75e-09 | yes | 6.94e-10 |
| final1 | 32 | 1e-08 | `native_spectral_q512` | 0.120 | [0.117, 0.123] | — | 1 | 4.07e-14 | yes | 6.15e-16 |
| final1 | 32 | 1e-08 | `native_spectral_q1024` | 0.115 | [0.113, 0.118] | — | 3 | 4.07e-14 | yes | 6.15e-16 |
| final1 | 32 | 1e-10 | `zero_cg` | 3.375 | [3.297, 3.464] | 104.2 | 0 | 8.56e-11 | yes | 4.73e-12 |
| final1 | 32 | 1e-10 | `fft_dst_direct` | 0.173 | [0.172, 0.175] | — | 1 | 3.15e-14 | yes | 0.00e+00 |
| final1 | 32 | 1e-10 | `dense_dst_direct` | 0.075 | [0.071, 0.081] | — | 3 | 4.07e-14 | yes | 6.15e-16 |
| final1 | 32 | 1e-10 | `native_zero` | 3.154 | [3.085, 3.227] | — | 0 | 8.56e-11 | yes | 4.73e-12 |
| final1 | 32 | 1e-10 | `native_jacobi_zero` | 3.389 | [3.312, 3.472] | — | 0 | 8.56e-11 | yes | 4.73e-12 |
| final1 | 32 | 1e-10 | `native_lmtrmean_c64_q0` | 6.276 | [5.532, 7.253] | — | 0 | 9.35e-11 | yes | 7.69e-12 |
| final1 | 32 | 1e-10 | `native_groupn_rt30_c64_q0` | 4.966 | [4.442, 5.446] | — | 0 | 9.68e-11 | yes | 5.63e-12 |
| final1 | 32 | 1e-10 | `native_spectral_q512` | 0.125 | [0.119, 0.136] | — | 1 | 4.07e-14 | yes | 6.15e-16 |
| final1 | 32 | 1e-10 | `native_spectral_q1024` | 0.119 | [0.115, 0.125] | — | 1 | 4.07e-14 | yes | 6.15e-16 |
| final1 | 64 | 1e-06 | `zero_cg` | 5.009 | [4.911, 5.091] | 156.2 | 0 | 9.97e-07 | yes | 7.07e-08 |
| final1 | 64 | 1e-06 | `fft_dst_direct` | 0.155 | [0.154, 0.156] | — | 0 | 9.73e-14 | yes | 0.00e+00 |
| final1 | 64 | 1e-06 | `dense_dst_direct` | 0.092 | [0.090, 0.095] | — | 2 | 2.37e-13 | yes | 6.82e-16 |
| final1 | 64 | 1e-06 | `native_zero` | 4.739 | [4.635, 4.827] | — | 0 | 9.97e-07 | yes | 7.07e-08 |
| final1 | 64 | 1e-06 | `native_jacobi_zero` | 5.141 | [5.035, 5.239] | — | 0 | 9.97e-07 | yes | 7.07e-08 |
| final1 | 64 | 1e-06 | `native_lmtrmean_c64_q0` | 7.791 | [6.946, 8.820] | — | 0 | 9.67e-07 | yes | 6.08e-08 |
| final1 | 64 | 1e-06 | `native_groupn_rt30_c64_q0` | 6.646 | [6.113, 7.114] | — | 0 | 9.84e-07 | yes | 6.23e-08 |
| final1 | 64 | 1e-06 | `native_spectral_q512` | 0.142 | [0.137, 0.146] | — | 1 | 2.37e-13 | yes | 6.82e-16 |
| final1 | 64 | 1e-06 | `native_spectral_q1024` | 0.135 | [0.131, 0.139] | — | 1 | 2.37e-13 | yes | 6.82e-16 |
| final1 | 64 | 1e-08 | `zero_cg` | 5.854 | [5.758, 5.942] | 187.2 | 0 | 1.00e-08 | yes | 5.63e-10 |
| final1 | 64 | 1e-08 | `fft_dst_direct` | 0.147 | [0.145, 0.148] | — | 1 | 9.73e-14 | yes | 0.00e+00 |
| final1 | 64 | 1e-08 | `dense_dst_direct` | 0.092 | [0.088, 0.098] | — | 3 | 2.37e-13 | yes | 6.82e-16 |
| final1 | 64 | 1e-08 | `native_zero` | 5.599 | [5.502, 5.689] | — | 0 | 1.00e-08 | yes | 5.63e-10 |
| final1 | 64 | 1e-08 | `native_jacobi_zero` | 6.059 | [5.966, 6.145] | — | 0 | 9.97e-09 | yes | 5.64e-10 |
| final1 | 64 | 1e-08 | `native_lmtrmean_c64_q0` | 8.496 | [7.664, 9.571] | — | 0 | 9.87e-09 | yes | 6.62e-10 |
| final1 | 64 | 1e-08 | `native_groupn_rt30_c64_q0` | 7.436 | [6.880, 7.920] | — | 0 | 9.69e-09 | yes | 4.89e-10 |
| final1 | 64 | 1e-08 | `native_spectral_q512` | 0.134 | [0.129, 0.140] | — | 2 | 2.37e-13 | yes | 6.82e-16 |
| final1 | 64 | 1e-08 | `native_spectral_q1024` | 0.129 | [0.125, 0.134] | — | 0 | 2.37e-13 | yes | 6.82e-16 |
| final1 | 64 | 1e-10 | `zero_cg` | 6.535 | [6.453, 6.609] | 211.7 | 0 | 9.94e-11 | yes | 4.37e-12 |
| final1 | 64 | 1e-10 | `fft_dst_direct` | 0.154 | [0.152, 0.156] | — | 0 | 9.73e-14 | yes | 0.00e+00 |
| final1 | 64 | 1e-10 | `dense_dst_direct` | 0.091 | [0.089, 0.093] | — | 1 | 2.37e-13 | yes | 6.82e-16 |
| final1 | 64 | 1e-10 | `native_zero` | 6.331 | [6.232, 6.433] | — | 0 | 9.94e-11 | yes | 4.37e-12 |
| final1 | 64 | 1e-10 | `native_jacobi_zero` | 6.856 | [6.751, 6.961] | — | 0 | 9.91e-11 | yes | 4.37e-12 |
| final1 | 64 | 1e-10 | `native_lmtrmean_c64_q0` | 9.456 | [8.624, 10.618] | — | 0 | 9.99e-11 | yes | 5.10e-12 |
| final1 | 64 | 1e-10 | `native_groupn_rt30_c64_q0` | 8.265 | [7.738, 8.736] | — | 0 | 9.91e-11 | yes | 4.48e-12 |
| final1 | 64 | 1e-10 | `native_spectral_q512` | 0.142 | [0.136, 0.149] | — | 2 | 2.37e-13 | yes | 6.82e-16 |
| final1 | 64 | 1e-10 | `native_spectral_q1024` | 0.136 | [0.131, 0.141] | — | 0 | 2.37e-13 | yes | 6.82e-16 |
| final1 | 128 | 1e-06 | `zero_cg` | 9.812 | [9.617, 9.955] | 320.2 | 0 | 9.96e-07 | yes | 5.19e-08 |
| final1 | 128 | 1e-06 | `fft_dst_direct` | 0.180 | [0.175, 0.185] | — | 1 | 9.32e-13 | yes | 0.00e+00 |
| final1 | 128 | 1e-06 | `dense_dst_direct` | 0.100 | [0.095, 0.104] | — | 2 | 1.30e-12 | yes | 9.86e-16 |
| final1 | 128 | 1e-06 | `native_zero` | 9.521 | [9.329, 9.671] | — | 0 | 9.96e-07 | yes | 5.19e-08 |
| final1 | 128 | 1e-06 | `native_jacobi_zero` | 10.368 | [10.152, 10.541] | — | 0 | 9.96e-07 | yes | 5.20e-08 |
| final1 | 128 | 1e-06 | `native_lmtrmean_c64_q0` | 12.512 | [11.669, 13.525] | — | 0 | 9.63e-07 | yes | 4.07e-08 |
| final1 | 128 | 1e-06 | `native_groupn_rt30_c64_q0` | 11.873 | [11.330, 12.316] | — | 0 | 9.84e-07 | yes | 4.61e-08 |
| final1 | 128 | 1e-06 | `native_spectral_q512` | 0.148 | [0.140, 0.155] | — | 1 | 1.30e-12 | yes | 9.86e-16 |
| final1 | 128 | 1e-06 | `native_spectral_q1024` | 0.139 | [0.135, 0.143] | — | 1 | 1.30e-12 | yes | 9.86e-16 |
| final1 | 128 | 1e-08 | `zero_cg` | 11.945 | [11.739, 12.117] | 382.0 | 0 | 1.00e-08 | yes | 3.87e-10 |
| final1 | 128 | 1e-08 | `fft_dst_direct` | 0.193 | [0.186, 0.199] | — | 1 | 9.32e-13 | yes | 0.00e+00 |
| final1 | 128 | 1e-08 | `dense_dst_direct` | 0.114 | [0.104, 0.125] | — | 0 | 1.30e-12 | yes | 9.86e-16 |
| final1 | 128 | 1e-08 | `native_zero` | 11.515 | [11.334, 11.677] | — | 0 | 1.00e-08 | yes | 3.87e-10 |
| final1 | 128 | 1e-08 | `native_jacobi_zero` | 12.657 | [12.435, 12.839] | — | 0 | 9.80e-09 | yes | 3.83e-10 |
| final1 | 128 | 1e-08 | `native_lmtrmean_c64_q0` | 14.578 | [13.844, 15.516] | — | 0 | 9.94e-09 | yes | 4.64e-10 |
| final1 | 128 | 1e-08 | `native_groupn_rt30_c64_q0` | 13.913 | [13.354, 14.399] | — | 0 | 9.98e-09 | yes | 3.52e-10 |
| final1 | 128 | 1e-08 | `native_spectral_q512` | 0.160 | [0.148, 0.173] | — | 2 | 1.30e-12 | yes | 9.86e-16 |
| final1 | 128 | 1e-08 | `native_spectral_q1024` | 0.157 | [0.148, 0.167] | — | 0 | 1.30e-12 | yes | 9.86e-16 |
| final1 | 128 | 1e-10 | `zero_cg` | 13.331 | [13.134, 13.550] | 430.8 | 0 | 9.73e-11 | yes | 3.12e-12 |
| final1 | 128 | 1e-10 | `fft_dst_direct` | 0.188 | [0.183, 0.193] | — | 0 | 9.32e-13 | yes | 0.00e+00 |
| final1 | 128 | 1e-10 | `dense_dst_direct` | 0.107 | [0.103, 0.111] | — | 1 | 1.30e-12 | yes | 9.86e-16 |
| final1 | 128 | 1e-10 | `native_zero` | 12.962 | [12.791, 13.134] | — | 0 | 9.73e-11 | yes | 3.12e-12 |
| final1 | 128 | 1e-10 | `native_jacobi_zero` | 14.164 | [14.029, 14.310] | — | 0 | 9.73e-11 | yes | 3.13e-12 |
| final1 | 128 | 1e-10 | `native_lmtrmean_c64_q0` | 16.430 | [15.398, 17.559] | — | 0 | 9.92e-11 | yes | 3.77e-12 |
| final1 | 128 | 1e-10 | `native_groupn_rt30_c64_q0` | 15.493 | [14.674, 16.235] | — | 0 | 9.99e-11 | yes | 3.13e-12 |
| final1 | 128 | 1e-10 | `native_spectral_q512` | 0.157 | [0.150, 0.164] | — | 2 | 1.30e-12 | yes | 9.86e-16 |
| final1 | 128 | 1e-10 | `native_spectral_q1024` | 0.150 | [0.144, 0.157] | — | 0 | 1.30e-12 | yes | 9.86e-16 |
| final1 | 256 | 1e-06 | `zero_cg` | 20.119 | [19.788, 20.386] | 656.2 | 0 | 1.00e-06 | yes | 3.44e-08 |
| final1 | 256 | 1e-06 | `fft_dst_direct` | 0.159 | [0.152, 0.167] | — | 1 | 1.80e-12 | yes | 0.00e+00 |
| final1 | 256 | 1e-06 | `dense_dst_direct` | 0.110 | [0.105, 0.118] | — | 7 | 7.65e-12 | yes | 9.97e-16 |
| final1 | 256 | 1e-06 | `native_zero` | 19.684 | [19.344, 19.991] | — | 0 | 1.00e-06 | yes | 3.44e-08 |
| final1 | 256 | 1e-06 | `native_jacobi_zero` | 22.991 | [22.547, 23.351] | — | 0 | 9.93e-07 | yes | 3.41e-08 |
| final1 | 256 | 1e-06 | `native_lmtrmean_c64_q0` | 22.110 | [21.214, 23.154] | — | 0 | 9.92e-07 | yes | 2.90e-08 |
| final1 | 256 | 1e-06 | `native_groupn_rt30_c64_q0` | 22.221 | [21.510, 22.802] | — | 0 | 9.99e-07 | yes | 3.43e-08 |
| final1 | 256 | 1e-06 | `native_spectral_q512` | 0.158 | [0.155, 0.162] | — | 6 | 7.65e-12 | yes | 9.97e-16 |
| final1 | 256 | 1e-06 | `native_spectral_q1024` | 0.151 | [0.148, 0.156] | — | 5 | 7.65e-12 | yes | 9.97e-16 |
| final1 | 256 | 1e-08 | `zero_cg` | 23.536 | [23.320, 23.738] | 778.0 | 0 | 9.74e-09 | yes | 2.63e-10 |
| final1 | 256 | 1e-08 | `fft_dst_direct` | 0.150 | [0.146, 0.158] | — | 1 | 1.80e-12 | yes | 0.00e+00 |
| final1 | 256 | 1e-08 | `dense_dst_direct` | 0.103 | [0.101, 0.105] | — | 5 | 7.65e-12 | yes | 9.97e-16 |
| final1 | 256 | 1e-08 | `native_zero` | 23.011 | [22.805, 23.198] | — | 0 | 9.74e-09 | yes | 2.63e-10 |
| final1 | 256 | 1e-08 | `native_jacobi_zero` | 27.593 | [27.320, 27.818] | — | 0 | 9.74e-09 | yes | 2.63e-10 |
| final1 | 256 | 1e-08 | `native_lmtrmean_c64_q0` | 25.328 | [24.371, 26.406] | — | 0 | 9.89e-09 | yes | 3.12e-10 |
| final1 | 256 | 1e-08 | `native_groupn_rt30_c64_q0` | 25.396 | [24.613, 25.993] | — | 0 | 1.00e-08 | yes | 2.51e-10 |
| final1 | 256 | 1e-08 | `native_spectral_q512` | 0.151 | [0.147, 0.156] | — | 6 | 7.65e-12 | yes | 9.97e-16 |
| final1 | 256 | 1e-08 | `native_spectral_q1024` | 0.145 | [0.143, 0.148] | — | 4 | 7.65e-12 | yes | 9.97e-16 |
| final1 | 256 | 1e-10 | `zero_cg` | 26.490 | [26.262, 26.685] | 874.5 | 0 | 9.94e-11 | yes | 2.21e-12 |
| final1 | 256 | 1e-10 | `fft_dst_direct` | 0.145 | [0.142, 0.148] | — | 0 | 1.80e-12 | yes | 0.00e+00 |
| final1 | 256 | 1e-10 | `dense_dst_direct` | 0.101 | [0.099, 0.103] | — | 5 | 7.65e-12 | yes | 9.97e-16 |
| final1 | 256 | 1e-10 | `native_zero` | 25.801 | [25.557, 26.020] | — | 0 | 9.94e-11 | yes | 2.21e-12 |
| final1 | 256 | 1e-10 | `native_jacobi_zero` | 30.517 | [30.275, 30.724] | — | 0 | 9.94e-11 | yes | 2.21e-12 |
| final1 | 256 | 1e-10 | `native_lmtrmean_c64_q0` | 28.763 | [27.796, 29.882] | — | 0 | 9.96e-11 | yes | 2.55e-12 |
| final1 | 256 | 1e-10 | `native_groupn_rt30_c64_q0` | 28.373 | [27.509, 29.075] | — | 0 | 9.99e-11 | yes | 2.62e-12 |
| final1 | 256 | 1e-10 | `native_spectral_q512` | 0.148 | [0.146, 0.151] | — | 6 | 7.65e-12 | yes | 9.97e-16 |
| final1 | 256 | 1e-10 | `native_spectral_q1024` | 0.142 | [0.138, 0.147] | — | 5 | 7.65e-12 | yes | 9.97e-16 |
| final1 | 512 | 1e-06 | `zero_cg` | 62.248 | [61.225, 63.045] | 1336.3 | 0 | 9.98e-07 | yes | 2.40e-08 |
| final1 | 512 | 1e-06 | `fft_dst_direct` | 0.343 | [0.340, 0.348] | — | 1 | 1.34e-11 | yes | 0.00e+00 |
| final1 | 512 | 1e-06 | `dense_dst_direct` | 0.192 | [0.187, 0.199] | — | 3 | 4.21e-11 | yes | 1.58e-15 |
| final1 | 512 | 1e-06 | `native_zero` | 61.384 | [60.340, 62.203] | — | 0 | 9.98e-07 | yes | 2.40e-08 |
| final1 | 512 | 1e-06 | `native_jacobi_zero` | 67.723 | [66.576, 68.583] | — | 0 | 9.97e-07 | yes | 2.40e-08 |
| final1 | 512 | 1e-06 | `native_lmtrmean_c64_q0` | 62.257 | [61.052, 63.432] | — | 0 | 9.98e-07 | yes | 1.97e-08 |
| final1 | 512 | 1e-06 | `native_groupn_rt30_c64_q0` | 65.693 | [64.398, 66.777] | — | 0 | 1.00e-06 | yes | 2.22e-08 |
| final1 | 512 | 1e-06 | `native_spectral_q512` | 0.245 | [0.238, 0.254] | — | 1 | 4.21e-11 | yes | 1.58e-15 |
| final1 | 512 | 1e-06 | `native_spectral_q1024` | 0.232 | [0.229, 0.238] | — | 0 | 4.21e-11 | yes | 1.58e-15 |
| final1 | 512 | 1e-08 | `zero_cg` | 73.189 | [72.534, 73.866] | 1578.3 | 0 | 9.96e-09 | yes | 1.71e-10 |
| final1 | 512 | 1e-08 | `fft_dst_direct` | 0.338 | [0.336, 0.341] | — | 0 | 1.34e-11 | yes | 0.00e+00 |
| final1 | 512 | 1e-08 | `dense_dst_direct` | 0.186 | [0.184, 0.188] | — | 3 | 4.21e-11 | yes | 1.58e-15 |
| final1 | 512 | 1e-08 | `native_zero` | 72.328 | [71.734, 72.986] | — | 0 | 9.96e-09 | yes | 1.71e-10 |
| final1 | 512 | 1e-08 | `native_jacobi_zero` | 79.812 | [79.238, 80.443] | — | 0 | 9.93e-09 | yes | 1.71e-10 |
| final1 | 512 | 1e-08 | `native_lmtrmean_c64_q0` | 72.956 | [71.057, 74.757] | — | 0 | 9.91e-09 | yes | 2.48e-10 |
| final1 | 512 | 1e-08 | `native_groupn_rt30_c64_q0` | 75.913 | [74.183, 77.272] | — | 0 | 9.93e-09 | yes | 1.83e-10 |
| final1 | 512 | 1e-08 | `native_spectral_q512` | 0.240 | [0.233, 0.252] | — | 1 | 4.21e-11 | yes | 1.58e-15 |
| final1 | 512 | 1e-08 | `native_spectral_q1024` | 0.232 | [0.228, 0.238] | — | 0 | 4.21e-11 | yes | 1.58e-15 |
| final1 | 512 | 1e-10 | `zero_cg` | 81.852 | [81.242, 82.394] | 1769.8 | 0 | 7.22e-11 | yes | 1.56e-12 |
| final1 | 512 | 1e-10 | `fft_dst_direct` | 0.339 | [0.336, 0.344] | — | 0 | 1.34e-11 | yes | 0.00e+00 |
| final1 | 512 | 1e-10 | `dense_dst_direct` | 0.186 | [0.184, 0.189] | — | 3 | 4.21e-11 | yes | 1.58e-15 |
| final1 | 512 | 1e-10 | `native_zero` | 80.591 | [80.036, 81.087] | — | 0 | 1.18e-10 | no | 1.57e-12 |
| final1 | 512 | 1e-10 | `native_jacobi_zero` | 89.199 | [88.474, 89.826] | — | 0 | 1.18e-10 | no | 1.57e-12 |
| final1 | 512 | 1e-10 | `native_lmtrmean_c64_q0` | 83.690 | [82.380, 84.948] | — | 0 | 1.21e-10 | no | 1.67e-12 |
| final1 | 512 | 1e-10 | `native_groupn_rt30_c64_q0` | 85.224 | [83.752, 86.361] | — | 0 | 1.19e-10 | no | 1.68e-12 |
| final1 | 512 | 1e-10 | `native_spectral_q512` | 0.233 | [0.232, 0.235] | — | 2 | 4.21e-11 | yes | 1.58e-15 |
| final1 | 512 | 1e-10 | `native_spectral_q1024` | 0.227 | [0.226, 0.228] | — | 1 | 4.21e-11 | yes | 1.58e-15 |
| final1 | 1024 | 1e-06 | `zero_cg` | 228.212 | [224.043, 232.448] | 2718.2 | 0 | 9.98e-07 | yes | 1.62e-08 |
| final1 | 1024 | 1e-06 | `fft_dst_direct` | 1.095 | [1.089, 1.101] | — | 0 | 3.96e-11 | yes | 0.00e+00 |
| final1 | 1024 | 1e-06 | `dense_dst_direct` | 0.665 | [0.658, 0.672] | — | 0 | 2.48e-10 | yes | 1.82e-15 |
| final1 | 1024 | 1e-06 | `native_zero` | 215.611 | [212.029, 218.718] | — | 0 | 9.98e-07 | yes | 1.62e-08 |
| final1 | 1024 | 1e-06 | `native_jacobi_zero` | 245.832 | [241.921, 249.327] | — | 0 | 9.98e-07 | yes | 1.62e-08 |
| final1 | 1024 | 1e-06 | `native_lmtrmean_c64_q0` | 221.183 | [215.386, 225.595] | — | 0 | 1.00e-06 | yes | 1.37e-08 |
| final1 | 1024 | 1e-06 | `native_groupn_rt30_c64_q0` | 231.076 | [225.957, 236.422] | — | 0 | 9.99e-07 | yes | 1.60e-08 |
| final1 | 1024 | 1e-06 | `native_spectral_q512` | 0.611 | [0.473, 0.745] | — | 1 | 3.37e-07 | yes | 2.73e-12 |
| final1 | 1024 | 1e-06 | `native_spectral_q1024` | 0.727 | [0.719, 0.737] | — | 0 | 2.48e-10 | yes | 1.82e-15 |
| final1 | 1024 | 1e-08 | `zero_cg` | 269.776 | [266.851, 272.050] | 3193.3 | 0 | 9.97e-09 | yes | 1.12e-10 |
| final1 | 1024 | 1e-08 | `fft_dst_direct` | 1.096 | [1.090, 1.101] | — | 0 | 3.96e-11 | yes | 0.00e+00 |
| final1 | 1024 | 1e-08 | `dense_dst_direct` | 0.673 | [0.660, 0.688] | — | 0 | 2.48e-10 | yes | 1.82e-15 |
| final1 | 1024 | 1e-08 | `native_zero` | 253.128 | [250.299, 255.188] | — | 0 | 9.97e-09 | yes | 1.12e-10 |
| final1 | 1024 | 1e-08 | `native_jacobi_zero` | 288.718 | [285.718, 291.359] | — | 0 | 9.96e-09 | yes | 1.12e-10 |
| final1 | 1024 | 1e-08 | `native_lmtrmean_c64_q0` | 263.049 | [258.900, 266.765] | — | 0 | 9.98e-09 | yes | 1.70e-10 |
| final1 | 1024 | 1e-08 | `native_groupn_rt30_c64_q0` | 266.284 | [261.433, 270.113] | — | 0 | 1.00e-08 | no | 1.28e-10 |
| final1 | 1024 | 1e-08 | `native_spectral_q512` | 0.775 | [0.562, 0.989] | — | 1 | 6.72e-09 | yes | 5.28e-14 |
| final1 | 1024 | 1e-08 | `native_spectral_q1024` | 0.731 | [0.720, 0.744] | — | 0 | 2.48e-10 | yes | 1.82e-15 |
| final1 | 1024 | 1e-10 | `zero_cg` | 302.598 | [300.507, 304.302] | 3573.5 | 0 | 9.80e-11 | yes | 1.09e-12 |
| final1 | 1024 | 1e-10 | `fft_dst_direct` | 1.095 | [1.090, 1.103] | — | 0 | 3.96e-11 | yes | 0.00e+00 |
| final1 | 1024 | 1e-10 | `dense_dst_direct` | 0.723 | [0.684, 0.759] | — | 0 | 2.48e-10 | no | 1.82e-15 |
| final1 | 1024 | 1e-10 | `native_zero` | 283.453 | [281.519, 285.085] | — | 0 | 3.96e-10 | no | 1.09e-12 |
| final1 | 1024 | 1e-10 | `native_jacobi_zero` | 323.293 | [321.337, 325.181] | — | 0 | 3.99e-10 | no | 1.09e-12 |
| final1 | 1024 | 1e-10 | `native_lmtrmean_c64_q0` | 303.023 | [299.040, 306.484] | — | 0 | 4.14e-10 | no | 1.15e-12 |
| final1 | 1024 | 1e-10 | `native_groupn_rt30_c64_q0` | 301.683 | [298.088, 306.016] | — | 0 | 4.23e-10 | no | 1.13e-12 |
| final1 | 1024 | 1e-10 | `native_spectral_q512` | 0.994 | [0.662, 1.318] | — | 1 | 7.31e-11 | yes | 1.14e-15 |
| final1 | 1024 | 1e-10 | `native_spectral_q1024` | 0.796 | [0.766, 0.824] | — | 0 | 8.62e-11 | yes | 1.34e-15 |

## Authoritative balanced learned-versus-zero confirmation

Each adjacent pair is burn, learned-first/zero-second, reburn, zero-first/learned-second. Positive paired delta means the learned hybrid is slower. No timing outlier is removed.

| run | N | tolerance | learned ms | zero ms | learned-zero ms | delta 95% CI ms | speedup | speedup 95% CI | case signs L/Z/T | repetition signs L/Z/T | learned/zero outliers | learned/zero iterations | verdict |
|---|---:|---:|---:|---:|---:|---|---:|---|---|---|---|---|---|
| pairfinal1 | 512 | 1e-06 | 62.006 | 62.269 | -0.799 | [-2.691, 1.243] | 1.004 | [0.984, 1.058] | 5/3/0 | 56/40/0 | 0/0 | 1262.2/1351.5 | inconclusive/tie |
| pairfinal1 | 512 | 1e-08 | 73.868 | 73.751 | 0.460 | [-1.324, 2.046] | 0.998 | [0.973, 1.018] | 3/5/0 | 37/59/0 | 0/0 | 1521.8/1584.4 | inconclusive/tie |
| pairfinal1 | 512 | 1e-10 | 85.432 | 82.404 | 3.300 | [1.946, 4.989] | 0.965 | [0.941, 0.981] | 0/8/0 | 0/96/0 | 0/0 | 1764.0/1786.5 | supported slower |
| pairfinal1 | 1024 | 1e-06 | 211.996 | 217.578 | -5.874 | [-14.642, -2.418] | 1.026 | [1.014, 1.077] | 7/1/0 | 88/8/0 | 0/0 | 2571.1/2744.1 | supported faster |
| pairfinal1 | 1024 | 1e-08 | 253.531 | 257.030 | -4.040 | [-8.976, -0.573] | 1.014 | [0.993, 1.031] | 7/1/0 | 83/13/0 | 0/0 | 3087.1/3209.0 | inconclusive/tie |
| pairfinal1 | 1024 | 1e-10 | 290.348 | 288.587 | 2.469 | [-1.534, 6.457] | 0.994 | [0.980, 1.010] | 3/5/0 | 31/65/0 | 0/0 | 3563.8/3608.5 | inconclusive/tie |

### Balanced per-case medians

`L1/L2` and `Z1/Z2` are the learned and zero medians when each method ran first/second.

| run | N | tolerance | case | learned ms | zero ms | learned-zero ms | L1/L2 ms | Z1/Z2 ms | repetition signs L/Z/T | outliers L/Z |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| pairfinal1 | 512 | 1e-06 | 0 | 62.454 | 64.867 | -2.446 | 62.408/62.454 | 64.900/64.851 | 12/0/0 | 0/0 |
| pairfinal1 | 512 | 1e-06 | 1 | 72.495 | 65.225 | 7.297 | 72.630/72.394 | 65.258/65.218 | 0/12/0 | 0/0 |
| pairfinal1 | 512 | 1e-06 | 2 | 58.565 | 61.231 | -2.691 | 58.650/58.349 | 61.345/61.199 | 12/0/0 | 0/0 |
| pairfinal1 | 512 | 1e-06 | 3 | 63.281 | 62.234 | 1.009 | 63.500/63.165 | 62.336/62.156 | 0/12/0 | 0/0 |
| pairfinal1 | 512 | 1e-06 | 4 | 59.055 | 63.988 | -4.924 | 59.195/58.879 | 64.016/63.942 | 12/0/0 | 0/0 |
| pairfinal1 | 512 | 1e-06 | 5 | 60.470 | 61.884 | -1.404 | 60.518/60.348 | 61.935/61.823 | 12/0/0 | 0/0 |
| pairfinal1 | 512 | 1e-06 | 6 | 61.923 | 60.692 | 1.243 | 61.992/61.897 | 60.774/60.606 | 0/12/0 | 0/0 |
| pairfinal1 | 512 | 1e-06 | 7 | 62.089 | 62.305 | -0.195 | 62.305/62.038 | 62.361/62.277 | 8/4/0 | 0/0 |
| pairfinal1 | 512 | 1e-08 | 0 | 74.084 | 75.323 | -1.324 | 74.138/73.910 | 75.376/75.323 | 12/0/0 | 0/0 |
| pairfinal1 | 512 | 1e-08 | 1 | 82.287 | 75.278 | 7.016 | 82.155/82.442 | 75.287/75.270 | 0/12/0 | 0/0 |
| pairfinal1 | 512 | 1e-08 | 2 | 71.232 | 72.751 | -1.516 | 71.195/71.310 | 72.731/72.768 | 12/0/0 | 0/0 |
| pairfinal1 | 512 | 1e-08 | 3 | 74.830 | 74.304 | 0.488 | 74.858/74.753 | 74.349/74.253 | 0/12/0 | 0/0 |
| pairfinal1 | 512 | 1e-08 | 4 | 74.164 | 73.493 | 0.696 | 74.134/74.164 | 73.522/73.432 | 0/12/0 | 0/0 |
| pairfinal1 | 512 | 1e-08 | 5 | 71.799 | 71.376 | 0.432 | 71.821/71.751 | 71.408/71.361 | 1/11/0 | 0/0 |
| pairfinal1 | 512 | 1e-08 | 6 | 73.653 | 71.638 | 2.046 | 73.728/73.595 | 71.663/71.638 | 0/12/0 | 0/0 |
| pairfinal1 | 512 | 1e-08 | 7 | 72.688 | 74.009 | -1.236 | 72.670/72.987 | 74.043/73.958 | 12/0/0 | 0/0 |
| pairfinal1 | 512 | 1e-10 | 0 | 85.214 | 83.644 | 1.595 | 85.237/85.208 | 83.721/83.506 | 0/12/0 | 0/0 |
| pairfinal1 | 512 | 1e-10 | 1 | 94.935 | 83.613 | 11.301 | 94.926/94.935 | 83.740/83.457 | 0/12/0 | 0/0 |
| pairfinal1 | 512 | 1e-10 | 2 | 83.137 | 81.208 | 1.946 | 83.181/83.132 | 81.351/81.171 | 0/12/0 | 0/0 |
| pairfinal1 | 512 | 1e-10 | 3 | 85.894 | 82.918 | 3.059 | 85.977/85.886 | 83.251/82.720 | 0/12/0 | 0/0 |
| pairfinal1 | 512 | 1e-10 | 4 | 85.650 | 81.890 | 3.797 | 85.656/85.634 | 82.004/81.788 | 0/12/0 | 0/0 |
| pairfinal1 | 512 | 1e-10 | 5 | 83.195 | 79.626 | 3.541 | 83.204/83.129 | 79.717/79.490 | 0/12/0 | 0/0 |
| pairfinal1 | 512 | 1e-10 | 6 | 84.805 | 79.843 | 4.989 | 84.846/84.752 | 79.992/79.843 | 0/12/0 | 0/0 |
| pairfinal1 | 512 | 1e-10 | 7 | 86.409 | 83.731 | 2.975 | 86.440/86.409 | 83.731/83.732 | 0/12/0 | 0/0 |
| pairfinal1 | 1024 | 1e-06 | 0 | 211.578 | 225.964 | -14.642 | 211.563/211.592 | 226.127/225.891 | 12/0/0 | 0/0 |
| pairfinal1 | 1024 | 1e-06 | 1 | 224.095 | 226.555 | -2.418 | 224.207/223.987 | 226.561/226.555 | 12/0/0 | 0/0 |
| pairfinal1 | 1024 | 1e-06 | 2 | 203.201 | 215.187 | -11.997 | 203.256/203.107 | 215.282/215.057 | 12/0/0 | 0/0 |
| pairfinal1 | 1024 | 1e-06 | 3 | 214.291 | 217.976 | -3.626 | 214.259/214.321 | 218.156/217.817 | 12/0/0 | 0/0 |
| pairfinal1 | 1024 | 1e-06 | 4 | 199.095 | 223.257 | -24.144 | 199.023/199.106 | 223.311/223.057 | 12/0/0 | 0/0 |
| pairfinal1 | 1024 | 1e-06 | 5 | 207.701 | 215.426 | -7.697 | 207.936/207.592 | 215.397/215.445 | 12/0/0 | 0/0 |
| pairfinal1 | 1024 | 1e-06 | 6 | 212.415 | 212.285 | 0.124 | 212.461/212.375 | 212.371/212.249 | 4/8/0 | 0/0 |
| pairfinal1 | 1024 | 1e-06 | 7 | 213.149 | 217.181 | -4.050 | 213.152/213.113 | 217.276/217.083 | 12/0/0 | 0/0 |
| pairfinal1 | 1024 | 1e-08 | 0 | 253.839 | 262.887 | -8.976 | 253.807/253.989 | 262.901/262.785 | 12/0/0 | 0/0 |
| pairfinal1 | 1024 | 1e-08 | 1 | 258.019 | 261.862 | -4.107 | 258.019/257.867 | 262.227/261.764 | 12/0/0 | 0/0 |
| pairfinal1 | 1024 | 1e-08 | 2 | 244.301 | 253.308 | -9.052 | 244.318/244.301 | 253.421/253.170 | 12/0/0 | 0/0 |
| pairfinal1 | 1024 | 1e-08 | 3 | 255.137 | 259.120 | -3.972 | 255.275/254.959 | 259.205/258.847 | 12/0/0 | 0/0 |
| pairfinal1 | 1024 | 1e-08 | 4 | 254.046 | 255.935 | -1.937 | 254.173/254.039 | 256.142/255.907 | 12/0/0 | 0/0 |
| pairfinal1 | 1024 | 1e-08 | 5 | 247.292 | 248.073 | -0.573 | 247.292/247.366 | 248.214/247.978 | 11/1/0 | 0/0 |
| pairfinal1 | 1024 | 1e-08 | 6 | 253.224 | 249.842 | 3.488 | 253.224/253.195 | 250.167/249.556 | 0/12/0 | 0/0 |
| pairfinal1 | 1024 | 1e-08 | 7 | 250.432 | 258.126 | -7.739 | 250.513/250.323 | 258.135/258.028 | 12/0/0 | 0/0 |
| pairfinal1 | 1024 | 1e-10 | 0 | 290.076 | 294.539 | -4.412 | 291.508/289.948 | 294.522/295.617 | 12/0/0 | 0/0 |
| pairfinal1 | 1024 | 1e-10 | 1 | 300.965 | 294.311 | 6.527 | 301.041/300.722 | 294.293/294.442 | 0/12/0 | 0/0 |
| pairfinal1 | 1024 | 1e-10 | 2 | 283.906 | 285.374 | -1.534 | 284.149/283.706 | 285.614/285.316 | 12/0/0 | 0/0 |
| pairfinal1 | 1024 | 1e-10 | 3 | 291.603 | 291.619 | -0.084 | 291.643/291.481 | 291.705/291.585 | 7/5/0 | 0/0 |
| pairfinal1 | 1024 | 1e-10 | 4 | 290.620 | 287.331 | 3.317 | 290.633/290.569 | 287.340/287.261 | 0/12/0 | 0/0 |
| pairfinal1 | 1024 | 1e-10 | 5 | 283.919 | 279.167 | 4.864 | 284.187/283.838 | 279.147/279.312 | 0/12/0 | 0/0 |
| pairfinal1 | 1024 | 1e-10 | 6 | 288.778 | 280.838 | 8.051 | 289.044/288.707 | 280.810/280.889 | 0/12/0 | 0/0 |
| pairfinal1 | 1024 | 1e-10 | 7 | 291.493 | 289.843 | 1.621 | 291.716/291.344 | 289.889/289.831 | 0/12/0 | 0/0 |

## Spectral-control paired comparisons

Positive delta means the candidate is slower than the named spectral control.

| run | N | tolerance | candidate | control | delta ms | 95% CI ms | control/candidate |
|---|---:|---:|---|---|---:|---|---:|
| gcal1 | 256 | 1e-06 | `groupn_rt30_c64_q0` | `spectral_q8` | 19.325 | — | 0.210 |
| gcal1 | 256 | 1e-06 | `groupn_rt10_c64_q0` | `spectral_q8` | 18.728 | — | 0.215 |
| gcal1 | 256 | 1e-06 | `groupn_rt03_c64_q0` | `spectral_q8` | 18.504 | — | 0.217 |
| gcal1 | 256 | 1e-06 | `groupn_rt01_c64_q0` | `spectral_q8` | 19.218 | — | 0.211 |
| gcal1 | 256 | 1e-06 | `groupn_rt30_c64_q8` | `spectral_q8` | 3.228 | — | 0.615 |
| gcal1 | 256 | 1e-06 | `groupn_rt10_c64_q8` | `spectral_q8` | 3.375 | — | 0.605 |
| gcal1 | 256 | 1e-06 | `groupn_rt03_c64_q8` | `spectral_q8` | 3.574 | — | 0.591 |
| gcal1 | 256 | 1e-06 | `spectral_q16` | `spectral_q8` | -2.771 | — | 2.177 |
| specdev1 | 256 | 1e-06 | `spectral_q8` | `spectral_q16` | 2.602 | [2.401, 2.775] | 0.409 |
| specdev1 | 256 | 1e-06 | `spectral_q16` | `spectral_q8` | -2.602 | [-2.775, -2.401] | 2.447 |
| specdev1 | 256 | 1e-06 | `spectral_q24` | `spectral_q16` | -0.809 | [-0.884, -0.739] | 1.817 |
| specdev1 | 256 | 1e-06 | `spectral_q24` | `spectral_q8` | -3.410 | [-3.649, -3.156] | 4.447 |
| specdev1 | 256 | 1e-06 | `spectral_q32` | `spectral_q16` | -1.091 | [-1.190, -0.984] | 2.565 |
| specdev1 | 256 | 1e-06 | `spectral_q32` | `spectral_q8` | -3.693 | [-3.913, -3.479] | 6.277 |
| specdev1 | 256 | 1e-06 | `spectral_q48` | `spectral_q16` | -1.299 | [-1.449, -1.101] | 3.642 |
| specdev1 | 256 | 1e-06 | `spectral_q48` | `spectral_q8` | -3.902 | [-4.089, -3.706] | 8.911 |
| specdev1 | 256 | 1e-06 | `spectral_q64` | `spectral_q16` | -1.393 | [-1.602, -1.159] | 4.482 |
| specdev1 | 256 | 1e-06 | `spectral_q64` | `spectral_q8` | -3.991 | [-4.170, -3.794] | 10.968 |
| specdev1 | 512 | 1e-06 | `spectral_q8` | `spectral_q16` | 8.649 | [8.106, 9.144] | 0.386 |
| specdev1 | 512 | 1e-06 | `spectral_q16` | `spectral_q8` | -8.649 | [-9.143, -8.105] | 2.589 |
| specdev1 | 512 | 1e-06 | `spectral_q24` | `spectral_q16` | -2.648 | [-2.836, -2.481] | 1.939 |
| specdev1 | 512 | 1e-06 | `spectral_q24` | `spectral_q8` | -11.334 | [-11.917, -10.689] | 5.021 |
| specdev1 | 512 | 1e-06 | `spectral_q32` | `spectral_q16` | -3.575 | [-3.921, -3.160] | 2.869 |
| specdev1 | 512 | 1e-06 | `spectral_q32` | `spectral_q8` | -12.242 | [-12.837, -11.625] | 7.429 |
| specdev1 | 512 | 1e-06 | `spectral_q48` | `spectral_q16` | -4.245 | [-4.860, -3.537] | 4.468 |
| specdev1 | 512 | 1e-06 | `spectral_q48` | `spectral_q8` | -12.898 | [-13.416, -12.300] | 11.570 |
| specdev1 | 512 | 1e-06 | `spectral_q64` | `spectral_q16` | -4.472 | [-5.203, -3.636] | 5.744 |
| specdev1 | 512 | 1e-06 | `spectral_q64` | `spectral_q8` | -13.164 | [-13.721, -12.464] | 14.874 |
| specdev1 | 1024 | 1e-06 | `spectral_q8` | `spectral_q16` | 29.793 | [27.451, 31.821] | 0.394 |
| specdev1 | 1024 | 1e-06 | `spectral_q16` | `spectral_q8` | -29.793 | [-31.798, -27.451] | 2.540 |
| specdev1 | 1024 | 1e-06 | `spectral_q24` | `spectral_q16` | -9.572 | [-10.563, -8.624] | 1.984 |
| specdev1 | 1024 | 1e-06 | `spectral_q24` | `spectral_q8` | -39.387 | [-42.127, -36.521] | 5.040 |
| specdev1 | 1024 | 1e-06 | `spectral_q32` | `spectral_q16` | -13.084 | [-14.539, -11.476] | 3.109 |
| specdev1 | 1024 | 1e-06 | `spectral_q32` | `spectral_q8` | -42.906 | [-45.721, -40.280] | 7.896 |
| specdev1 | 1024 | 1e-06 | `spectral_q48` | `spectral_q16` | -15.363 | [-17.090, -12.847] | 4.901 |
| specdev1 | 1024 | 1e-06 | `spectral_q48` | `spectral_q8` | -45.187 | [-47.210, -43.072] | 12.449 |
| specdev1 | 1024 | 1e-06 | `spectral_q64` | `spectral_q16` | -16.357 | [-18.711, -13.346] | 6.590 |
| specdev1 | 1024 | 1e-06 | `spectral_q64` | `spectral_q8` | -46.188 | [-48.140, -43.906] | 16.739 |
| specdev3 | 256 | 1e-06 | `spectral_q64` | `spectral_q128` | 0.086 | [-0.004, 0.186] | 0.788 |
| specdev3 | 256 | 1e-06 | `spectral_q64` | `spectral_q192` | 0.107 | [-0.002, 0.226] | 0.744 |
| specdev3 | 256 | 1e-06 | `spectral_q64` | `spectral_q256` | 0.184 | [0.024, 0.356] | 0.546 |
| specdev3 | 256 | 1e-06 | `spectral_q128` | `spectral_q192` | 0.024 | [-0.001, 0.048] | 0.943 |
| specdev3 | 256 | 1e-06 | `spectral_q128` | `spectral_q256` | 0.097 | [0.023, 0.175] | 0.692 |
| specdev3 | 256 | 1e-06 | `spectral_q128` | `spectral_q64` | -0.086 | [-0.186, 0.004] | 1.268 |
| specdev3 | 256 | 1e-06 | `spectral_q192` | `spectral_q128` | -0.024 | [-0.048, -0.000] | 1.060 |
| specdev3 | 256 | 1e-06 | `spectral_q192` | `spectral_q256` | 0.077 | [0.023, 0.135] | 0.734 |
| specdev3 | 256 | 1e-06 | `spectral_q192` | `spectral_q64` | -0.107 | [-0.226, 0.002] | 1.345 |
| specdev3 | 256 | 1e-06 | `spectral_q256` | `spectral_q128` | -0.097 | [-0.175, -0.023] | 1.445 |
| specdev3 | 256 | 1e-06 | `spectral_q256` | `spectral_q192` | -0.077 | [-0.135, -0.023] | 1.363 |
| specdev3 | 256 | 1e-06 | `spectral_q256` | `spectral_q64` | -0.184 | [-0.363, -0.024] | 1.832 |
| specdev3 | 512 | 1e-06 | `spectral_q64` | `spectral_q128` | 0.351 | [0.063, 0.667] | 0.615 |
| specdev3 | 512 | 1e-06 | `spectral_q64` | `spectral_q192` | 0.468 | [0.087, 0.896] | 0.488 |
| specdev3 | 512 | 1e-06 | `spectral_q64` | `spectral_q256` | 0.514 | [0.096, 0.981] | 0.437 |
| specdev3 | 512 | 1e-06 | `spectral_q128` | `spectral_q192` | 0.119 | [0.025, 0.214] | 0.795 |
| specdev3 | 512 | 1e-06 | `spectral_q128` | `spectral_q256` | 0.159 | [0.030, 0.302] | 0.710 |
| specdev3 | 512 | 1e-06 | `spectral_q128` | `spectral_q64` | -0.351 | [-0.676, -0.063] | 1.627 |
| specdev3 | 512 | 1e-06 | `spectral_q192` | `spectral_q128` | -0.119 | [-0.214, -0.025] | 1.258 |
| specdev3 | 512 | 1e-06 | `spectral_q192` | `spectral_q256` | 0.043 | [0.006, 0.085] | 0.894 |
| specdev3 | 512 | 1e-06 | `spectral_q192` | `spectral_q64` | -0.468 | [-0.903, -0.087] | 2.047 |
| specdev3 | 512 | 1e-06 | `spectral_q256` | `spectral_q128` | -0.159 | [-0.296, -0.030] | 1.408 |
| specdev3 | 512 | 1e-06 | `spectral_q256` | `spectral_q192` | -0.043 | [-0.084, -0.006] | 1.119 |
| specdev3 | 512 | 1e-06 | `spectral_q256` | `spectral_q64` | -0.514 | [-0.956, -0.096] | 2.290 |
| specdev3 | 1024 | 1e-06 | `spectral_q64` | `spectral_q128` | 1.356 | [0.322, 2.474] | 0.514 |
| specdev3 | 1024 | 1e-06 | `spectral_q64` | `spectral_q192` | 1.780 | [0.424, 3.253] | 0.367 |
| specdev3 | 1024 | 1e-06 | `spectral_q64` | `spectral_q256` | 1.940 | [0.424, 3.589] | 0.306 |
| specdev3 | 1024 | 1e-06 | `spectral_q128` | `spectral_q192` | 0.424 | [0.098, 0.788] | 0.713 |
| specdev3 | 1024 | 1e-06 | `spectral_q128` | `spectral_q256` | 0.586 | [0.093, 1.126] | 0.595 |
| specdev3 | 1024 | 1e-06 | `spectral_q128` | `spectral_q64` | -1.356 | [-2.512, -0.322] | 1.946 |
| specdev3 | 1024 | 1e-06 | `spectral_q192` | `spectral_q128` | -0.424 | [-0.783, -0.098] | 1.402 |
| specdev3 | 1024 | 1e-06 | `spectral_q192` | `spectral_q256` | 0.175 | [0.028, 0.332] | 0.834 |
| specdev3 | 1024 | 1e-06 | `spectral_q192` | `spectral_q64` | -1.780 | [-3.281, -0.424] | 2.728 |
| specdev3 | 1024 | 1e-06 | `spectral_q256` | `spectral_q128` | -0.586 | [-1.117, -0.097] | 1.680 |
| specdev3 | 1024 | 1e-06 | `spectral_q256` | `spectral_q192` | -0.175 | [-0.332, -0.028] | 1.198 |
| specdev3 | 1024 | 1e-06 | `spectral_q256` | `spectral_q64` | -1.940 | [-3.589, -0.424] | 3.270 |
| specdev4 | 1024 | 1e-06 | `spectral_q192` | `spectral_q1024` | 0.240 | [-0.303, 0.834] | 0.775 |
| specdev4 | 1024 | 1e-06 | `spectral_q192` | `spectral_q256` | 0.167 | [0.003, 0.343] | 0.856 |
| specdev4 | 1024 | 1e-06 | `spectral_q192` | `spectral_q384` | 0.313 | [0.005, 0.650] | 0.700 |
| specdev4 | 1024 | 1e-06 | `spectral_q192` | `spectral_q512` | 0.344 | [-0.035, 0.754] | 0.674 |
| specdev4 | 1024 | 1e-06 | `spectral_q192` | `spectral_q768` | 0.261 | [-0.189, 0.747] | 0.753 |
| specdev4 | 1024 | 1e-06 | `spectral_q256` | `spectral_q1024` | 0.083 | [-0.303, 0.516] | 0.905 |
| specdev4 | 1024 | 1e-06 | `spectral_q256` | `spectral_q192` | -0.167 | [-0.348, -0.002] | 1.168 |
| specdev4 | 1024 | 1e-06 | `spectral_q256` | `spectral_q384` | 0.161 | [0.008, 0.335] | 0.817 |
| specdev4 | 1024 | 1e-06 | `spectral_q256` | `spectral_q512` | 0.182 | [-0.034, 0.424] | 0.787 |
| specdev4 | 1024 | 1e-06 | `spectral_q256` | `spectral_q768` | 0.107 | [-0.180, 0.427] | 0.880 |
| specdev4 | 1024 | 1e-06 | `spectral_q384` | `spectral_q1024` | -0.071 | [-0.306, 0.187] | 1.108 |
| specdev4 | 1024 | 1e-06 | `spectral_q384` | `spectral_q192` | -0.313 | [-0.650, -0.005] | 1.429 |
| specdev4 | 1024 | 1e-06 | `spectral_q384` | `spectral_q256` | -0.161 | [-0.327, -0.008] | 1.224 |
| specdev4 | 1024 | 1e-06 | `spectral_q384` | `spectral_q512` | 0.020 | [-0.048, 0.095] | 0.963 |
| specdev4 | 1024 | 1e-06 | `spectral_q384` | `spectral_q768` | -0.042 | [-0.186, 0.113] | 1.077 |
| specdev4 | 1024 | 1e-06 | `spectral_q512` | `spectral_q1024` | -0.096 | [-0.269, 0.095] | 1.150 |
| specdev4 | 1024 | 1e-06 | `spectral_q512` | `spectral_q192` | -0.344 | [-0.762, 0.035] | 1.484 |
| specdev4 | 1024 | 1e-06 | `spectral_q512` | `spectral_q256` | -0.182 | [-0.424, 0.034] | 1.270 |
| specdev4 | 1024 | 1e-06 | `spectral_q512` | `spectral_q384` | -0.020 | [-0.096, 0.048] | 1.038 |
| specdev4 | 1024 | 1e-06 | `spectral_q512` | `spectral_q768` | -0.074 | [-0.155, 0.015] | 1.118 |
| specdev4 | 1024 | 1e-06 | `spectral_q768` | `spectral_q1024` | -0.026 | [-0.130, 0.087] | 1.029 |
| specdev4 | 1024 | 1e-06 | `spectral_q768` | `spectral_q192` | -0.261 | [-0.763, 0.189] | 1.327 |
| specdev4 | 1024 | 1e-06 | `spectral_q768` | `spectral_q256` | -0.107 | [-0.423, 0.180] | 1.137 |
| specdev4 | 1024 | 1e-06 | `spectral_q768` | `spectral_q384` | 0.042 | [-0.115, 0.185] | 0.929 |
| specdev4 | 1024 | 1e-06 | `spectral_q768` | `spectral_q512` | 0.074 | [-0.015, 0.155] | 0.895 |
| specdev4 | 1024 | 1e-06 | `spectral_q1024` | `spectral_q192` | -0.240 | [-0.834, 0.303] | 1.290 |
| specdev4 | 1024 | 1e-06 | `spectral_q1024` | `spectral_q256` | -0.083 | [-0.513, 0.303] | 1.104 |
| specdev4 | 1024 | 1e-06 | `spectral_q1024` | `spectral_q384` | 0.071 | [-0.187, 0.306] | 0.902 |
| specdev4 | 1024 | 1e-06 | `spectral_q1024` | `spectral_q512` | 0.096 | [-0.095, 0.269] | 0.869 |
| specdev4 | 1024 | 1e-06 | `spectral_q1024` | `spectral_q768` | 0.026 | [-0.087, 0.130] | 0.972 |
| specdev5 | 512 | 1e-06 | `spectral_q256` | `spectral_q1024` | 0.077 | [-0.036, 0.205] | 0.795 |
| specdev5 | 512 | 1e-06 | `spectral_q256` | `spectral_q384` | 0.012 | [-0.040, 0.067] | 1.011 |
| specdev5 | 512 | 1e-06 | `spectral_q256` | `spectral_q512` | 0.079 | [-0.033, 0.205] | 0.789 |
| specdev5 | 512 | 1e-08 | `spectral_q256` | `spectral_q1024` | 0.155 | [-0.011, 0.336] | 0.678 |
| specdev5 | 512 | 1e-08 | `spectral_q256` | `spectral_q384` | 0.048 | [-0.031, 0.135] | 0.933 |
| specdev5 | 512 | 1e-08 | `spectral_q256` | `spectral_q512` | 0.150 | [-0.014, 0.329] | 0.697 |
| specdev5 | 512 | 1e-10 | `spectral_q256` | `spectral_q1024` | 0.240 | [0.007, 0.505] | 0.557 |
| specdev5 | 512 | 1e-10 | `spectral_q256` | `spectral_q384` | 0.073 | [-0.020, 0.180] | 0.903 |
| specdev5 | 512 | 1e-10 | `spectral_q256` | `spectral_q512` | 0.237 | [0.006, 0.500] | 0.557 |
| specdev5 | 512 | 1e-06 | `spectral_q384` | `spectral_q1024` | 0.067 | [-0.001, 0.138] | 0.787 |
| specdev5 | 512 | 1e-06 | `spectral_q384` | `spectral_q256` | -0.012 | [-0.067, 0.040] | 0.989 |
| specdev5 | 512 | 1e-06 | `spectral_q384` | `spectral_q512` | 0.073 | [0.006, 0.141] | 0.780 |
| specdev5 | 512 | 1e-08 | `spectral_q384` | `spectral_q1024` | 0.105 | [0.012, 0.208] | 0.727 |
| specdev5 | 512 | 1e-08 | `spectral_q384` | `spectral_q256` | -0.048 | [-0.136, 0.031] | 1.072 |
| specdev5 | 512 | 1e-08 | `spectral_q384` | `spectral_q512` | 0.109 | [0.012, 0.215] | 0.747 |
| specdev5 | 512 | 1e-10 | `spectral_q384` | `spectral_q1024` | 0.168 | [0.023, 0.326] | 0.616 |
| specdev5 | 512 | 1e-10 | `spectral_q384` | `spectral_q256` | -0.073 | [-0.180, 0.020] | 1.107 |
| specdev5 | 512 | 1e-10 | `spectral_q384` | `spectral_q512` | 0.172 | [0.029, 0.326] | 0.617 |
| specdev5 | 512 | 1e-06 | `spectral_q512` | `spectral_q1024` | 0.002 | [-0.001, 0.006] | 1.008 |
| specdev5 | 512 | 1e-06 | `spectral_q512` | `spectral_q256` | -0.079 | [-0.201, 0.033] | 1.268 |
| specdev5 | 512 | 1e-06 | `spectral_q512` | `spectral_q384` | -0.073 | [-0.140, -0.006] | 1.282 |
| specdev5 | 512 | 1e-08 | `spectral_q512` | `spectral_q1024` | 0.007 | [0.003, 0.012] | 0.973 |
| specdev5 | 512 | 1e-08 | `spectral_q512` | `spectral_q256` | -0.150 | [-0.330, 0.015] | 1.436 |
| specdev5 | 512 | 1e-08 | `spectral_q512` | `spectral_q384` | -0.109 | [-0.210, -0.012] | 1.339 |
| specdev5 | 512 | 1e-10 | `spectral_q512` | `spectral_q1024` | 0.002 | [0.001, 0.004] | 0.999 |
| specdev5 | 512 | 1e-10 | `spectral_q512` | `spectral_q256` | -0.237 | [-0.483, -0.006] | 1.795 |
| specdev5 | 512 | 1e-10 | `spectral_q512` | `spectral_q384` | -0.172 | [-0.325, -0.029] | 1.621 |
| specdev5 | 512 | 1e-06 | `spectral_q1024` | `spectral_q256` | -0.077 | [-0.199, 0.036] | 1.257 |
| specdev5 | 512 | 1e-06 | `spectral_q1024` | `spectral_q384` | -0.067 | [-0.139, 0.000] | 1.271 |
| specdev5 | 512 | 1e-06 | `spectral_q1024` | `spectral_q512` | -0.002 | [-0.006, 0.001] | 0.992 |
| specdev5 | 512 | 1e-08 | `spectral_q1024` | `spectral_q256` | -0.155 | [-0.336, 0.011] | 1.476 |
| specdev5 | 512 | 1e-08 | `spectral_q1024` | `spectral_q384` | -0.105 | [-0.208, -0.012] | 1.376 |
| specdev5 | 512 | 1e-08 | `spectral_q1024` | `spectral_q512` | -0.007 | [-0.012, -0.003] | 1.028 |
| specdev5 | 512 | 1e-10 | `spectral_q1024` | `spectral_q256` | -0.240 | [-0.504, -0.007] | 1.797 |
| specdev5 | 512 | 1e-10 | `spectral_q1024` | `spectral_q384` | -0.168 | [-0.330, -0.024] | 1.623 |
| specdev5 | 512 | 1e-10 | `spectral_q1024` | `spectral_q512` | -0.002 | [-0.004, -0.001] | 1.001 |
| specdev5 | 1024 | 1e-06 | `spectral_q256` | `spectral_q1024` | 0.048 | [-0.335, 0.482] | 0.948 |
| specdev5 | 1024 | 1e-06 | `spectral_q256` | `spectral_q384` | 0.126 | [-0.021, 0.293] | 0.872 |
| specdev5 | 1024 | 1e-06 | `spectral_q256` | `spectral_q512` | 0.153 | [-0.062, 0.396] | 0.819 |
| specdev5 | 1024 | 1e-08 | `spectral_q256` | `spectral_q1024` | 0.372 | [-0.228, 1.025] | 0.674 |
| specdev5 | 1024 | 1e-08 | `spectral_q256` | `spectral_q384` | 0.251 | [0.031, 0.485] | 0.791 |
| specdev5 | 1024 | 1e-08 | `spectral_q256` | `spectral_q512` | 0.344 | [0.007, 0.701] | 0.706 |
| specdev5 | 1024 | 1e-10 | `spectral_q256` | `spectral_q1024` | 0.643 | [-0.136, 1.474] | 0.563 |
| specdev5 | 1024 | 1e-10 | `spectral_q256` | `spectral_q384` | 0.383 | [0.061, 0.738] | 0.749 |
| specdev5 | 1024 | 1e-10 | `spectral_q256` | `spectral_q512` | 0.533 | [0.063, 1.045] | 0.640 |
| specdev5 | 1024 | 1e-06 | `spectral_q384` | `spectral_q1024` | -0.075 | [-0.311, 0.187] | 1.088 |
| specdev5 | 1024 | 1e-06 | `spectral_q384` | `spectral_q256` | -0.126 | [-0.291, 0.021] | 1.147 |
| specdev5 | 1024 | 1e-06 | `spectral_q384` | `spectral_q512` | 0.028 | [-0.043, 0.111] | 0.940 |
| specdev5 | 1024 | 1e-08 | `spectral_q384` | `spectral_q1024` | 0.123 | [-0.260, 0.547] | 0.853 |
| specdev5 | 1024 | 1e-08 | `spectral_q384` | `spectral_q256` | -0.251 | [-0.485, -0.031] | 1.264 |
| specdev5 | 1024 | 1e-08 | `spectral_q384` | `spectral_q512` | 0.100 | [-0.025, 0.241] | 0.892 |
| specdev5 | 1024 | 1e-10 | `spectral_q384` | `spectral_q1024` | 0.262 | [-0.197, 0.749] | 0.752 |
| specdev5 | 1024 | 1e-10 | `spectral_q384` | `spectral_q256` | -0.383 | [-0.716, -0.061] | 1.335 |
| specdev5 | 1024 | 1e-10 | `spectral_q384` | `spectral_q512` | 0.148 | [-0.002, 0.316] | 0.854 |
| specdev5 | 1024 | 1e-06 | `spectral_q512` | `spectral_q1024` | -0.102 | [-0.272, 0.093] | 1.158 |
| specdev5 | 1024 | 1e-06 | `spectral_q512` | `spectral_q256` | -0.153 | [-0.395, 0.062] | 1.221 |
| specdev5 | 1024 | 1e-06 | `spectral_q512` | `spectral_q384` | -0.028 | [-0.111, 0.044] | 1.064 |
| specdev5 | 1024 | 1e-08 | `spectral_q512` | `spectral_q1024` | 0.030 | [-0.233, 0.317] | 0.956 |
| specdev5 | 1024 | 1e-08 | `spectral_q512` | `spectral_q256` | -0.344 | [-0.701, -0.007] | 1.417 |
| specdev5 | 1024 | 1e-08 | `spectral_q512` | `spectral_q384` | -0.100 | [-0.240, 0.025] | 1.121 |
| specdev5 | 1024 | 1e-10 | `spectral_q512` | `spectral_q1024` | 0.116 | [-0.192, 0.438] | 0.880 |
| specdev5 | 1024 | 1e-10 | `spectral_q512` | `spectral_q256` | -0.533 | [-1.045, -0.063] | 1.563 |
| specdev5 | 1024 | 1e-10 | `spectral_q512` | `spectral_q384` | -0.148 | [-0.316, 0.002] | 1.171 |
| specdev5 | 1024 | 1e-06 | `spectral_q1024` | `spectral_q256` | -0.048 | [-0.482, 0.335] | 1.055 |
| specdev5 | 1024 | 1e-06 | `spectral_q1024` | `spectral_q384` | 0.075 | [-0.187, 0.311] | 0.919 |
| specdev5 | 1024 | 1e-06 | `spectral_q1024` | `spectral_q512` | 0.102 | [-0.094, 0.272] | 0.864 |
| specdev5 | 1024 | 1e-08 | `spectral_q1024` | `spectral_q256` | -0.372 | [-1.003, 0.228] | 1.483 |
| specdev5 | 1024 | 1e-08 | `spectral_q1024` | `spectral_q384` | -0.123 | [-0.536, 0.260] | 1.173 |
| specdev5 | 1024 | 1e-08 | `spectral_q1024` | `spectral_q512` | -0.030 | [-0.307, 0.233] | 1.046 |
| specdev5 | 1024 | 1e-10 | `spectral_q1024` | `spectral_q256` | -0.643 | [-1.475, 0.136] | 1.776 |
| specdev5 | 1024 | 1e-10 | `spectral_q1024` | `spectral_q384` | -0.262 | [-0.751, 0.197] | 1.330 |
| specdev5 | 1024 | 1e-10 | `spectral_q1024` | `spectral_q512` | -0.116 | [-0.438, 0.192] | 1.136 |
| final1 | 32 | 1e-06 | `lmmean_cfull_q0` | `spectral_q1024` | 6.067 | [5.257, 6.996] | 0.033 |
| final1 | 32 | 1e-06 | `lmmean_cfull_q0` | `spectral_q512` | 6.050 | [5.245, 6.961] | 0.032 |
| final1 | 32 | 1e-08 | `lmmean_cfull_q0` | `spectral_q1024` | 6.248 | [5.520, 7.086] | 0.031 |
| final1 | 32 | 1e-08 | `lmmean_cfull_q0` | `spectral_q512` | 6.249 | [5.519, 7.089] | 0.032 |
| final1 | 32 | 1e-10 | `lmmean_cfull_q0` | `spectral_q1024` | 6.700 | [5.989, 7.499] | 0.029 |
| final1 | 32 | 1e-10 | `lmmean_cfull_q0` | `spectral_q512` | 6.692 | [5.974, 7.497] | 0.030 |
| final1 | 32 | 1e-06 | `lmtrmean_c64_q0` | `spectral_q1024` | 5.519 | [4.742, 6.483] | 0.036 |
| final1 | 32 | 1e-06 | `lmtrmean_c64_q0` | `spectral_q512` | 5.492 | [4.705, 6.475] | 0.036 |
| final1 | 32 | 1e-08 | `lmtrmean_c64_q0` | `spectral_q1024` | 5.783 | [5.024, 6.786] | 0.034 |
| final1 | 32 | 1e-08 | `lmtrmean_c64_q0` | `spectral_q512` | 5.793 | [5.037, 6.793] | 0.034 |
| final1 | 32 | 1e-10 | `lmtrmean_c64_q0` | `spectral_q1024` | 6.228 | [5.495, 7.204] | 0.031 |
| final1 | 32 | 1e-10 | `lmtrmean_c64_q0` | `spectral_q512` | 6.238 | [5.498, 7.212] | 0.032 |
| final1 | 32 | 1e-06 | `groupn_rt30_c64_q0` | `spectral_q1024` | 4.251 | [3.811, 4.697] | 0.046 |
| final1 | 32 | 1e-06 | `groupn_rt30_c64_q0` | `spectral_q512` | 4.251 | [3.807, 4.707] | 0.046 |
| final1 | 32 | 1e-08 | `groupn_rt30_c64_q0` | `spectral_q1024` | 4.552 | [4.024, 4.987] | 0.043 |
| final1 | 32 | 1e-08 | `groupn_rt30_c64_q0` | `spectral_q512` | 4.558 | [4.041, 4.986] | 0.043 |
| final1 | 32 | 1e-10 | `groupn_rt30_c64_q0` | `spectral_q1024` | 5.002 | [4.491, 5.441] | 0.039 |
| final1 | 32 | 1e-10 | `groupn_rt30_c64_q0` | `spectral_q512` | 5.000 | [4.494, 5.433] | 0.040 |
| final1 | 32 | 1e-06 | `groupn_rt30_c64_q8` | `spectral_q1024` | 2.466 | [1.982, 2.972] | 0.077 |
| final1 | 32 | 1e-06 | `groupn_rt30_c64_q8` | `spectral_q512` | 2.460 | [1.970, 2.967] | 0.077 |
| final1 | 32 | 1e-08 | `groupn_rt30_c64_q8` | `spectral_q1024` | 2.564 | [2.105, 3.003] | 0.073 |
| final1 | 32 | 1e-08 | `groupn_rt30_c64_q8` | `spectral_q512` | 2.561 | [2.100, 3.004] | 0.074 |
| final1 | 32 | 1e-10 | `groupn_rt30_c64_q8` | `spectral_q1024` | 2.804 | [2.362, 3.217] | 0.067 |
| final1 | 32 | 1e-10 | `groupn_rt30_c64_q8` | `spectral_q512` | 2.792 | [2.362, 3.199] | 0.070 |
| final1 | 32 | 1e-06 | `spectral_q512` | `spectral_q1024` | 0.003 | [-0.000, 0.006] | 1.008 |
| final1 | 32 | 1e-08 | `spectral_q512` | `spectral_q1024` | 0.002 | [-0.002, 0.005] | 0.994 |
| final1 | 32 | 1e-10 | `spectral_q512` | `spectral_q1024` | 0.003 | [-0.000, 0.006] | 0.965 |
| final1 | 32 | 1e-06 | `spectral_q1024` | `spectral_q512` | -0.003 | [-0.006, 0.000] | 0.992 |
| final1 | 32 | 1e-08 | `spectral_q1024` | `spectral_q512` | -0.002 | [-0.005, 0.002] | 1.006 |
| final1 | 32 | 1e-10 | `spectral_q1024` | `spectral_q512` | -0.003 | [-0.006, 0.000] | 1.036 |
| final1 | 64 | 1e-06 | `lmmean_cfull_q0` | `spectral_q1024` | 8.462 | [7.622, 9.312] | 0.025 |
| final1 | 64 | 1e-06 | `lmmean_cfull_q0` | `spectral_q512` | 8.462 | [7.627, 9.314] | 0.027 |
| final1 | 64 | 1e-08 | `lmmean_cfull_q0` | `spectral_q1024` | 9.242 | [8.372, 10.112] | 0.023 |
| final1 | 64 | 1e-08 | `lmmean_cfull_q0` | `spectral_q512` | 9.243 | [8.359, 10.127] | 0.023 |
| final1 | 64 | 1e-10 | `lmmean_cfull_q0` | `spectral_q1024` | 10.272 | [9.373, 11.195] | 0.021 |
| final1 | 64 | 1e-10 | `lmmean_cfull_q0` | `spectral_q512` | 10.247 | [9.352, 11.176] | 0.021 |
| final1 | 64 | 1e-06 | `lmtrmean_c64_q0` | `spectral_q1024` | 7.805 | [6.950, 8.859] | 0.027 |
| final1 | 64 | 1e-06 | `lmtrmean_c64_q0` | `spectral_q512` | 7.815 | [6.970, 8.847] | 0.029 |
| final1 | 64 | 1e-08 | `lmtrmean_c64_q0` | `spectral_q1024` | 8.600 | [7.789, 9.595] | 0.024 |
| final1 | 64 | 1e-08 | `lmtrmean_c64_q0` | `spectral_q512` | 8.613 | [7.819, 9.586] | 0.025 |
| final1 | 64 | 1e-10 | `lmtrmean_c64_q0` | `spectral_q1024` | 9.597 | [8.716, 10.679] | 0.022 |
| final1 | 64 | 1e-10 | `lmtrmean_c64_q0` | `spectral_q512` | 9.586 | [8.716, 10.669] | 0.022 |
| final1 | 64 | 1e-06 | `groupn_rt30_c64_q0` | `spectral_q1024` | 6.758 | [6.274, 7.215] | 0.031 |
| final1 | 64 | 1e-06 | `groupn_rt30_c64_q0` | `spectral_q512` | 6.757 | [6.269, 7.203] | 0.033 |
| final1 | 64 | 1e-08 | `groupn_rt30_c64_q0` | `spectral_q1024` | 7.556 | [7.012, 8.045] | 0.028 |
| final1 | 64 | 1e-08 | `groupn_rt30_c64_q0` | `spectral_q512` | 7.544 | [6.996, 8.026] | 0.028 |
| final1 | 64 | 1e-10 | `groupn_rt30_c64_q0` | `spectral_q1024` | 8.295 | [7.755, 8.770] | 0.025 |
| final1 | 64 | 1e-10 | `groupn_rt30_c64_q0` | `spectral_q512` | 8.301 | [7.758, 8.769] | 0.026 |
| final1 | 64 | 1e-06 | `groupn_rt30_c64_q8` | `spectral_q1024` | 2.987 | [2.481, 3.422] | 0.067 |
| final1 | 64 | 1e-06 | `groupn_rt30_c64_q8` | `spectral_q512` | 2.985 | [2.525, 3.403] | 0.072 |
| final1 | 64 | 1e-08 | `groupn_rt30_c64_q8` | `spectral_q1024` | 3.398 | [2.915, 3.850] | 0.060 |
| final1 | 64 | 1e-08 | `groupn_rt30_c64_q8` | `spectral_q512` | 3.394 | [2.931, 3.830] | 0.060 |
| final1 | 64 | 1e-10 | `groupn_rt30_c64_q8` | `spectral_q1024` | 3.815 | [3.352, 4.245] | 0.054 |
| final1 | 64 | 1e-10 | `groupn_rt30_c64_q8` | `spectral_q512` | 3.807 | [3.333, 4.234] | 0.055 |
| final1 | 64 | 1e-06 | `spectral_q512` | `spectral_q1024` | 0.012 | [0.008, 0.016] | 0.931 |
| final1 | 64 | 1e-08 | `spectral_q512` | `spectral_q1024` | 0.005 | [0.003, 0.007] | 0.993 |
| final1 | 64 | 1e-10 | `spectral_q512` | `spectral_q1024` | 0.010 | [0.005, 0.015] | 0.987 |
| final1 | 64 | 1e-06 | `spectral_q1024` | `spectral_q512` | -0.012 | [-0.016, -0.008] | 1.074 |
| final1 | 64 | 1e-08 | `spectral_q1024` | `spectral_q512` | -0.005 | [-0.007, -0.003] | 1.007 |
| final1 | 64 | 1e-10 | `spectral_q1024` | `spectral_q512` | -0.010 | [-0.015, -0.005] | 1.013 |
| final1 | 128 | 1e-06 | `lmmean_cfull_q0` | `spectral_q1024` | 13.212 | [12.340, 14.169] | 0.017 |
| final1 | 128 | 1e-06 | `lmmean_cfull_q0` | `spectral_q512` | 13.203 | [12.333, 14.157] | 0.018 |
| final1 | 128 | 1e-08 | `lmmean_cfull_q0` | `spectral_q1024` | 15.361 | [14.545, 16.254] | 0.016 |
| final1 | 128 | 1e-08 | `lmmean_cfull_q0` | `spectral_q512` | 15.337 | [14.544, 16.194] | 0.016 |
| final1 | 128 | 1e-10 | `lmmean_cfull_q0` | `spectral_q1024` | 17.061 | [16.302, 17.857] | 0.014 |
| final1 | 128 | 1e-10 | `lmmean_cfull_q0` | `spectral_q512` | 17.049 | [16.285, 17.847] | 0.014 |
| final1 | 128 | 1e-06 | `lmtrmean_c64_q0` | `spectral_q1024` | 12.545 | [11.691, 13.563] | 0.018 |
| final1 | 128 | 1e-06 | `lmtrmean_c64_q0` | `spectral_q512` | 12.541 | [11.676, 13.570] | 0.019 |
| final1 | 128 | 1e-08 | `lmtrmean_c64_q0` | `spectral_q1024` | 14.648 | [13.825, 15.637] | 0.016 |
| final1 | 128 | 1e-08 | `lmtrmean_c64_q0` | `spectral_q512` | 14.637 | [13.819, 15.617] | 0.017 |
| final1 | 128 | 1e-10 | `lmtrmean_c64_q0` | `spectral_q1024` | 16.335 | [15.316, 17.464] | 0.015 |
| final1 | 128 | 1e-10 | `lmtrmean_c64_q0` | `spectral_q512` | 16.324 | [15.313, 17.434] | 0.015 |
| final1 | 128 | 1e-06 | `groupn_rt30_c64_q0` | `spectral_q1024` | 11.764 | [11.303, 12.150] | 0.019 |
| final1 | 128 | 1e-06 | `groupn_rt30_c64_q0` | `spectral_q512` | 11.765 | [11.294, 12.155] | 0.020 |
| final1 | 128 | 1e-08 | `groupn_rt30_c64_q0` | `spectral_q1024` | 13.859 | [13.304, 14.324] | 0.017 |
| final1 | 128 | 1e-08 | `groupn_rt30_c64_q0` | `spectral_q512` | 13.837 | [13.259, 14.317] | 0.018 |
| final1 | 128 | 1e-10 | `groupn_rt30_c64_q0` | `spectral_q1024` | 15.263 | [14.437, 15.969] | 0.016 |
| final1 | 128 | 1e-10 | `groupn_rt30_c64_q0` | `spectral_q512` | 15.254 | [14.423, 15.957] | 0.016 |
| final1 | 128 | 1e-06 | `groupn_rt30_c64_q8` | `spectral_q1024` | 4.274 | [3.872, 4.635] | 0.050 |
| final1 | 128 | 1e-06 | `groupn_rt30_c64_q8` | `spectral_q512` | 4.256 | [3.833, 4.631] | 0.054 |
| final1 | 128 | 1e-08 | `groupn_rt30_c64_q8` | `spectral_q1024` | 5.324 | [4.839, 5.743] | 0.044 |
| final1 | 128 | 1e-08 | `groupn_rt30_c64_q8` | `spectral_q512` | 5.321 | [4.848, 5.752] | 0.045 |
| final1 | 128 | 1e-10 | `groupn_rt30_c64_q8` | `spectral_q1024` | 6.079 | [5.568, 6.553] | 0.038 |
| final1 | 128 | 1e-10 | `groupn_rt30_c64_q8` | `spectral_q512` | 6.059 | [5.532, 6.543] | 0.039 |
| final1 | 128 | 1e-06 | `spectral_q512` | `spectral_q1024` | 0.010 | [0.005, 0.015] | 0.938 |
| final1 | 128 | 1e-08 | `spectral_q512` | `spectral_q1024` | 0.013 | [0.002, 0.026] | 0.980 |
| final1 | 128 | 1e-10 | `spectral_q512` | `spectral_q1024` | 0.009 | [0.002, 0.017] | 0.987 |
| final1 | 128 | 1e-06 | `spectral_q1024` | `spectral_q512` | -0.010 | [-0.015, -0.005] | 1.067 |
| final1 | 128 | 1e-08 | `spectral_q1024` | `spectral_q512` | -0.013 | [-0.026, -0.002] | 1.021 |
| final1 | 128 | 1e-10 | `spectral_q1024` | `spectral_q512` | -0.009 | [-0.017, -0.002] | 1.013 |
| final1 | 256 | 1e-06 | `lmmean_cfull_q0` | `spectral_q1024` | 23.540 | [22.494, 24.654] | 0.010 |
| final1 | 256 | 1e-06 | `lmmean_cfull_q0` | `spectral_q512` | 23.529 | [22.485, 24.642] | 0.010 |
| final1 | 256 | 1e-08 | `lmmean_cfull_q0` | `spectral_q1024` | 27.049 | [26.092, 28.096] | 0.009 |
| final1 | 256 | 1e-08 | `lmmean_cfull_q0` | `spectral_q512` | 27.019 | [26.051, 28.068] | 0.009 |
| final1 | 256 | 1e-10 | `lmmean_cfull_q0` | `spectral_q1024` | 30.810 | [30.004, 31.663] | 0.007 |
| final1 | 256 | 1e-10 | `lmmean_cfull_q0` | `spectral_q512` | 30.778 | [29.978, 31.617] | 0.008 |
| final1 | 256 | 1e-06 | `lmtrmean_c64_q0` | `spectral_q1024` | 22.132 | [21.207, 23.123] | 0.011 |
| final1 | 256 | 1e-06 | `lmtrmean_c64_q0` | `spectral_q512` | 22.090 | [21.153, 23.089] | 0.011 |
| final1 | 256 | 1e-08 | `lmtrmean_c64_q0` | `spectral_q1024` | 25.375 | [24.432, 26.411] | 0.009 |
| final1 | 256 | 1e-08 | `lmtrmean_c64_q0` | `spectral_q512` | 25.337 | [24.396, 26.378] | 0.010 |
| final1 | 256 | 1e-10 | `lmtrmean_c64_q0` | `spectral_q1024` | 29.253 | [28.259, 30.412] | 0.008 |
| final1 | 256 | 1e-10 | `lmtrmean_c64_q0` | `spectral_q512` | 29.224 | [28.218, 30.401] | 0.008 |
| final1 | 256 | 1e-06 | `groupn_rt30_c64_q0` | `spectral_q1024` | 22.442 | [21.808, 22.924] | 0.011 |
| final1 | 256 | 1e-06 | `groupn_rt30_c64_q0` | `spectral_q512` | 22.441 | [21.813, 22.929] | 0.011 |
| final1 | 256 | 1e-08 | `groupn_rt30_c64_q0` | `spectral_q1024` | 25.762 | [24.984, 26.361] | 0.009 |
| final1 | 256 | 1e-08 | `groupn_rt30_c64_q0` | `spectral_q512` | 25.731 | [24.952, 26.331] | 0.010 |
| final1 | 256 | 1e-10 | `groupn_rt30_c64_q0` | `spectral_q1024` | 28.813 | [27.942, 29.500] | 0.008 |
| final1 | 256 | 1e-10 | `groupn_rt30_c64_q0` | `spectral_q512` | 28.778 | [27.900, 29.475] | 0.008 |
| final1 | 256 | 1e-06 | `groupn_rt30_c64_q8` | `spectral_q1024` | 6.925 | [6.414, 7.349] | 0.033 |
| final1 | 256 | 1e-06 | `groupn_rt30_c64_q8` | `spectral_q512` | 6.906 | [6.385, 7.345] | 0.034 |
| final1 | 256 | 1e-08 | `groupn_rt30_c64_q8` | `spectral_q1024` | 8.620 | [8.137, 9.028] | 0.026 |
| final1 | 256 | 1e-08 | `groupn_rt30_c64_q8` | `spectral_q512` | 8.604 | [8.109, 9.022] | 0.028 |
| final1 | 256 | 1e-10 | `groupn_rt30_c64_q8` | `spectral_q1024` | 10.348 | [9.871, 10.761] | 0.022 |
| final1 | 256 | 1e-10 | `groupn_rt30_c64_q8` | `spectral_q512` | 10.330 | [9.858, 10.740] | 0.023 |
| final1 | 256 | 1e-06 | `spectral_q512` | `spectral_q1024` | 0.012 | [0.011, 0.013] | 0.969 |
| final1 | 256 | 1e-08 | `spectral_q512` | `spectral_q1024` | 0.016 | [0.007, 0.025] | 0.942 |
| final1 | 256 | 1e-10 | `spectral_q512` | `spectral_q1024` | 0.013 | [0.006, 0.020] | 0.935 |
| final1 | 256 | 1e-06 | `spectral_q1024` | `spectral_q512` | -0.012 | [-0.013, -0.011] | 1.032 |
| final1 | 256 | 1e-08 | `spectral_q1024` | `spectral_q512` | -0.016 | [-0.025, -0.007] | 1.062 |
| final1 | 256 | 1e-10 | `spectral_q1024` | `spectral_q512` | -0.013 | [-0.020, -0.006] | 1.070 |
| final1 | 512 | 1e-06 | `lmmean_cfull_q0` | `spectral_q1024` | 66.438 | [64.158, 68.430] | 0.005 |
| final1 | 512 | 1e-06 | `lmmean_cfull_q0` | `spectral_q512` | 66.396 | [64.102, 68.404] | 0.005 |
| final1 | 512 | 1e-08 | `lmmean_cfull_q0` | `spectral_q1024` | 77.706 | [76.167, 79.319] | 0.004 |
| final1 | 512 | 1e-08 | `lmmean_cfull_q0` | `spectral_q512` | 77.674 | [76.131, 79.265] | 0.004 |
| final1 | 512 | 1e-10 | `lmmean_cfull_q0` | `spectral_q1024` | 89.148 | [88.360, 89.999] | 0.003 |
| final1 | 512 | 1e-10 | `lmmean_cfull_q0` | `spectral_q512` | 89.115 | [88.326, 89.970] | 0.004 |
| final1 | 512 | 1e-06 | `lmtrmean_c64_q0` | `spectral_q1024` | 62.203 | [60.944, 63.405] | 0.005 |
| final1 | 512 | 1e-06 | `lmtrmean_c64_q0` | `spectral_q512` | 62.144 | [60.852, 63.350] | 0.006 |
| final1 | 512 | 1e-08 | `lmtrmean_c64_q0` | `spectral_q1024` | 72.883 | [70.920, 74.697] | 0.004 |
| final1 | 512 | 1e-08 | `lmtrmean_c64_q0` | `spectral_q512` | 72.850 | [70.866, 74.682] | 0.005 |
| final1 | 512 | 1e-10 | `lmtrmean_c64_q0` | `spectral_q1024` | 84.712 | [83.374, 86.015] | 0.004 |
| final1 | 512 | 1e-10 | `lmtrmean_c64_q0` | `spectral_q512` | 84.683 | [83.325, 86.001] | 0.004 |
| final1 | 512 | 1e-06 | `groupn_rt30_c64_q0` | `spectral_q1024` | 65.555 | [64.222, 66.577] | 0.005 |
| final1 | 512 | 1e-06 | `groupn_rt30_c64_q0` | `spectral_q512` | 65.528 | [64.195, 66.563] | 0.005 |
| final1 | 512 | 1e-08 | `groupn_rt30_c64_q0` | `spectral_q1024` | 75.687 | [73.953, 76.949] | 0.004 |
| final1 | 512 | 1e-08 | `groupn_rt30_c64_q0` | `spectral_q512` | 75.666 | [73.940, 76.912] | 0.004 |
| final1 | 512 | 1e-10 | `groupn_rt30_c64_q0` | `spectral_q1024` | 86.284 | [84.852, 87.420] | 0.004 |
| final1 | 512 | 1e-10 | `groupn_rt30_c64_q0` | `spectral_q512` | 86.247 | [84.792, 87.380] | 0.004 |
| final1 | 512 | 1e-06 | `groupn_rt30_c64_q8` | `spectral_q1024` | 18.307 | [17.692, 18.852] | 0.017 |
| final1 | 512 | 1e-06 | `groupn_rt30_c64_q8` | `spectral_q512` | 18.272 | [17.654, 18.821] | 0.019 |
| final1 | 512 | 1e-08 | `groupn_rt30_c64_q8` | `spectral_q1024` | 23.671 | [23.096, 24.165] | 0.013 |
| final1 | 512 | 1e-08 | `groupn_rt30_c64_q8` | `spectral_q512` | 23.684 | [23.088, 24.201] | 0.014 |
| final1 | 512 | 1e-10 | `groupn_rt30_c64_q8` | `spectral_q1024` | 28.998 | [28.422, 29.461] | 0.011 |
| final1 | 512 | 1e-10 | `groupn_rt30_c64_q8` | `spectral_q512` | 28.982 | [28.404, 29.445] | 0.011 |
| final1 | 512 | 1e-06 | `spectral_q512` | `spectral_q1024` | 0.026 | [0.018, 0.034] | 0.920 |
| final1 | 512 | 1e-08 | `spectral_q512` | `spectral_q1024` | 0.020 | [0.013, 0.029] | 0.937 |
| final1 | 512 | 1e-10 | `spectral_q512` | `spectral_q1024` | 0.011 | [0.009, 0.014] | 0.953 |
| final1 | 512 | 1e-06 | `spectral_q1024` | `spectral_q512` | -0.026 | [-0.034, -0.018] | 1.087 |
| final1 | 512 | 1e-08 | `spectral_q1024` | `spectral_q512` | -0.020 | [-0.029, -0.013] | 1.068 |
| final1 | 512 | 1e-10 | `spectral_q1024` | `spectral_q512` | -0.011 | [-0.014, -0.009] | 1.050 |
| final1 | 1024 | 1e-06 | `lmmean_cfull_q0` | `spectral_q1024` | 228.939 | [221.508, 234.226] | 0.003 |
| final1 | 1024 | 1e-06 | `lmmean_cfull_q0` | `spectral_q512` | 228.999 | [221.697, 234.154] | 0.003 |
| final1 | 1024 | 1e-08 | `lmmean_cfull_q0` | `spectral_q1024` | 271.666 | [266.923, 276.556] | 0.003 |
| final1 | 1024 | 1e-08 | `lmmean_cfull_q0` | `spectral_q512` | 271.532 | [266.954, 276.277] | 0.003 |
| final1 | 1024 | 1e-10 | `lmmean_cfull_q0` | `spectral_q1024` | 308.417 | [306.145, 310.644] | 0.003 |
| final1 | 1024 | 1e-10 | `lmmean_cfull_q0` | `spectral_q512` | 308.239 | [306.210, 310.248] | 0.004 |
| final1 | 1024 | 1e-06 | `lmtrmean_c64_q0` | `spectral_q1024` | 211.498 | [208.672, 214.176] | 0.004 |
| final1 | 1024 | 1e-06 | `lmtrmean_c64_q0` | `spectral_q512` | 211.532 | [208.858, 214.082] | 0.004 |
| final1 | 1024 | 1e-08 | `lmtrmean_c64_q0` | `spectral_q1024` | 249.435 | [245.890, 252.678] | 0.003 |
| final1 | 1024 | 1e-08 | `lmtrmean_c64_q0` | `spectral_q512` | 249.338 | [245.968, 252.486] | 0.004 |
| final1 | 1024 | 1e-10 | `lmtrmean_c64_q0` | `spectral_q1024` | 289.930 | [286.398, 293.074] | 0.003 |
| final1 | 1024 | 1e-10 | `lmtrmean_c64_q0` | `spectral_q512` | 289.716 | [286.413, 292.643] | 0.004 |
| final1 | 1024 | 1e-06 | `groupn_rt30_c64_q0` | `spectral_q1024` | 229.875 | [223.985, 235.700] | 0.003 |
| final1 | 1024 | 1e-06 | `groupn_rt30_c64_q0` | `spectral_q512` | 229.880 | [224.078, 235.635] | 0.003 |
| final1 | 1024 | 1e-08 | `groupn_rt30_c64_q0` | `spectral_q1024` | 268.364 | [260.782, 275.036] | 0.003 |
| final1 | 1024 | 1e-08 | `groupn_rt30_c64_q0` | `spectral_q512` | 268.274 | [260.749, 274.770] | 0.003 |
| final1 | 1024 | 1e-10 | `groupn_rt30_c64_q0` | `spectral_q1024` | 303.295 | [297.877, 308.795] | 0.003 |
| final1 | 1024 | 1e-10 | `groupn_rt30_c64_q0` | `spectral_q512` | 303.043 | [297.802, 308.285] | 0.004 |
| final1 | 1024 | 1e-06 | `groupn_rt30_c64_q8` | `spectral_q1024` | 61.126 | [59.403, 62.729] | 0.013 |
| final1 | 1024 | 1e-06 | `groupn_rt30_c64_q8` | `spectral_q512` | 61.180 | [59.436, 62.926] | 0.012 |
| final1 | 1024 | 1e-08 | `groupn_rt30_c64_q8` | `spectral_q1024` | 80.282 | [78.501, 82.134] | 0.010 |
| final1 | 1024 | 1e-08 | `groupn_rt30_c64_q8` | `spectral_q512` | 80.161 | [78.213, 82.203] | 0.011 |
| final1 | 1024 | 1e-10 | `groupn_rt30_c64_q8` | `spectral_q1024` | 98.903 | [97.187, 100.634] | 0.009 |
| final1 | 1024 | 1e-10 | `groupn_rt30_c64_q8` | `spectral_q512` | 98.722 | [96.731, 100.754] | 0.011 |
| final1 | 1024 | 1e-06 | `spectral_q512` | `spectral_q1024` | -0.045 | [-0.202, 0.107] | 1.059 |
| final1 | 1024 | 1e-08 | `spectral_q512` | `spectral_q1024` | 0.091 | [-0.155, 0.322] | 0.897 |
| final1 | 1024 | 1e-10 | `spectral_q512` | `spectral_q1024` | 0.198 | [-0.116, 0.485] | 0.823 |
| final1 | 1024 | 1e-06 | `spectral_q1024` | `spectral_q512` | 0.045 | [-0.103, 0.201] | 0.944 |
| final1 | 1024 | 1e-08 | `spectral_q1024` | `spectral_q512` | -0.091 | [-0.321, 0.155] | 1.115 |
| final1 | 1024 | 1e-10 | `spectral_q1024` | `spectral_q512` | -0.198 | [-0.495, 0.100] | 1.214 |

## Run files

- `experiments/poisson-hybrid-1024/runs/dev1/out/dev1.json`
- `experiments/poisson-hybrid-1024/runs/gcal1/out/gcal1.json`
- `experiments/poisson-hybrid-1024/runs/specdev1/out/specdev1.json`
- `experiments/poisson-hybrid-1024/runs/specdev2/out/specdev2.json`
- `experiments/poisson-hybrid-1024/runs/specdev3/out/specdev3.json`
- `experiments/poisson-hybrid-1024/runs/specdev4/out/specdev4.json`
- `experiments/poisson-hybrid-1024/runs/specdev5/out/specdev5.json`
- `experiments/poisson-hybrid-1024/runs/final1/out/final1.json`
- `experiments/poisson-hybrid-1024/runs/pairfinal1/out/pairfinal1.json`
