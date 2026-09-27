| arm | family | solved unknowns | trial-basis columns | quadratic terms P | M | worst evolved % | median evolved % | worst all-times % | GPU-query ms | complete-query ms | FOM by the rule (GPU) | speedup (GPU) | speedup (complete) | converged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `qman4_lin_M16` | quadratic manifold | 4 | 5 | 0 | 16 | 71.3433 | 40.8715 | 102.2009 | 16.005 | 17.801 | `nt1e-2_dt01` | 0.563× | 0.601× | yes |
| `qman4_quad_M16` | quadratic manifold | 4 | 15 | 10 | 16 | 70.5690 | 40.4710 | 101.6566 | 17.043 | 18.691 | `nt1e-2_dt01` | 0.528× | 0.573× | yes |
| `qman8_lin_M32` | quadratic manifold | 8 | 9 | 0 | 32 | 66.3756 | 37.2161 | 98.8038 | 14.802 | 16.015 | `nt1e-2_dt01` | 0.608× | 0.668× | yes |
| `qman8_quad_M32` | quadratic manifold | 8 | 45 | 36 | 32 | 66.3818 | 37.2415 | 98.8002 | 17.646 | 18.913 | `nt1e-2_dt01` | 0.510× | 0.566× | yes |
| `qman8_quad_M64` | quadratic manifold | 8 | 45 | 36 | 64 | 66.4239 | 37.1685 | 98.8002 | 20.418 | 22.568 | `nt1e-2_dt01` | 0.441× | 0.474× | yes |
| `qman8_quadg0p0001_M32` | quadratic manifold | 8 | 45 | 36 | 32 | 64.9936 | 36.5788 | 97.9015 | 20.752 | 22.157 | `nt1e-2_dt01` | 0.434× | 0.483× | yes |
| `q0_M64_dense_g1em06` | NM-ROM | 16 | — | — | 64 | 0.7644 | 0.7333 | 2.3473 | 188.015 | 191.146 | `dense_tight` | 0.228× | 0.229× | yes |
| `q0_M64_eqcert_g1em06` | NM-ROM | 16 | — | — | 64 | 1.1380 | 0.9597 | 2.3473 | 87.002 | 89.009 | `dense_tight` | 0.492× | 0.492× | yes |
| `q0_M64_eqcert_g1em06_fastL4` | NM-ROM (fast kernel) | 16 | — | — | 64 | 1.1380 | 0.9597 | 2.3473 | 61.906 | 64.354 | `dense_tight` | 0.691× | 0.680× | yes |
| `pod8_M32_dense` | POD-LSPG | 8 | — | — | 32 | 66.6050 | 37.3897 | 99.0236 | 15.361 | 16.997 | `nt1e-2_dt01` | 0.586× | 0.630× | yes |
| `pod16_M64_dense` | POD-LSPG | 16 | — | — | 64 | 65.3293 | 36.3026 | 97.3310 | 21.170 | 22.396 | `nt1e-2_dt01` | 0.425× | 0.478× | yes |
| `dense_tight` | FOM | — | — | — | — | 0.0000 | 0.0000 | 0.0000 | 42.788 | 43.790 | — (is a FOM) | — | — | — |
| `fft_tight` | FOM | — | — | — | — | 0.0000 | 0.0000 | 0.0000 | 81.191 | 83.667 | — (is a FOM) | — | — | — |
| `nt1e-2_dt01` | FOM | — | — | — | — | 1.4285 | 1.2725 | 1.4285 | 9.005 | 10.706 | — (is a FOM) | — | — | — |
