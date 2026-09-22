| arm | family | role | worst evolved % | median evolved % | worst all-times % | GPU-query ms | complete-query ms | FOM by the rule (GPU) | speedup (GPU) | speedup (complete) |
|---|---|---|---|---|---|---|---|---|---|---|
| `q0_M64_dense_g1em06` | NM-ROM | — | 1.8890 | 1.0059 | 2.5629 | 290.001 | 292.287 | `nt1e-3_dt01` | 0.068× | 0.075× |
| `q0_M64_eqcert_g0p001` | NM-ROM | — | 1.8898 | 1.0103 | 2.5628 | 41.854 | 44.061 | `nt1e-3_dt01` | 0.472× | 0.498× |
| `q0_M64_eqcert_g1em06` | NM-ROM | — | 1.8891 | 1.0100 | 2.5629 | 56.551 | 58.705 | `nt1e-3_dt01` | 0.349× | 0.374× |
| `q0_M64_eqtop_g0p001` | NM-ROM | — | 1.8898 | 1.0103 | 2.5628 | 42.184 | 44.288 | `nt1e-3_dt01` | 0.468× | 0.496× |
| `q0_M64_eqtop_g1em06` | NM-ROM | — | 1.8891 | 1.0100 | 2.5629 | 56.762 | 58.841 | `nt1e-3_dt01` | 0.348× | 0.373× |
| `q256_M1088_dense_g1em06` | NM-ROM | — | 0.5194 | 0.1790 | 0.9053 | 3925.346 | 3927.609 | `nt1e-3_dt005` | 0.008× | 0.008× |
| `q256_M1088_eqcert_g0p001` | NM-ROM | — | 1.0324 | 0.3607 | 1.0324 | 428.724 | 430.800 | `nt1e-3_dt005` | 0.072× | 0.077× |
| `q256_M1088_eqcert_g1em06` | NM-ROM | — | 1.0361 | 0.3607 | 1.0361 | 696.343 | 698.364 | `nt1e-3_dt005` | 0.045× | 0.048× |
| `q256_M1088_eqtop_g0p001` | NM-ROM | — | 0.5129 | 0.1789 | 0.9053 | 465.736 | 467.925 | `nt1e-3_dt005` | 0.067× | 0.071× |
| `q256_M1088_eqtop_g1em06` | NM-ROM | — | 0.5129 | 0.1789 | 0.9053 | 745.842 | 748.153 | `nt1e-3_dt005` | 0.042× | 0.044× |
| `q0_M64_eqcert_g1em06_fastL4` | NM-ROM (fast kernel) | — | 1.8891 | 1.0100 | 2.5629 | 38.175 | 40.203 | `nt1e-3_dt01` | 0.517× | 0.546× |
| `pod256_M1024_dense` | POD-LSPG | — | 0.7109 | 0.2834 | 3.7698 | 900.359 | 902.367 | `nt1e-3_dt005` | 0.035× | 0.037× |
| `pod512_M2048_dense` | POD-LSPG | — | 0.2184 | 0.0320 | 0.6125 | 2773.714 | 2776.812 | `nt1e-3_dt005` | 0.011× | 0.012× |
| `fno-large` | FNO | validation-selected | 7.4164 | 2.5256 | 7.4164 | 7.376 | 7.619 | `nt1e-2_dt01` | 1.203× | 1.425× |
| `unet-large` | U-Net | — | 4.4890 | 1.9391 | 4.4890 | 10.078 | 10.316 | `nt1e-2_dt01` | 0.881× | 1.052× |
| `unet-medium` | U-Net | best worst case on no-second's validation set; not selected | 4.7595 | 2.1116 | 4.7595 | 5.854 | 6.094 | `nt1e-2_dt01` | 1.516× | 1.782× |
| `unet-refine` | U-Net | validation-selected | 4.5529 | 1.9662 | 4.5529 | 5.860 | 6.101 | `nt1e-2_dt01` | 1.514× | 1.779× |
| `unet-small` | U-Net | — | 5.1391 | 1.6909 | 5.1391 | 9.431 | 9.674 | `nt1e-2_dt01` | 0.941× | 1.122× |
| `tsol-large` | Transolver | best worst case on no-second's validation set; not selected | 5.8304 | 3.9494 | 5.8304 | 15.249 | 15.542 | `nt1e-2_dt01` | 0.582× | 0.699× |
| `tsol-medium` | Transolver | — | 5.9581 | 3.5878 | 5.9581 | 12.813 | 13.055 | `nt1e-2_dt01` | 0.693× | 0.832× |
| `tsol-refine` | Transolver | validation-selected | 4.4593 | 2.3195 | 4.4593 | 11.071 | 11.327 | `nt1e-2_dt01` | 0.802× | 0.959× |
| `tsol-small` | Transolver | — | 4.8978 | 2.1771 | 4.8978 | 11.056 | 11.304 | `nt1e-2_dt01` | 0.803× | 0.960× |
| `dense_tight` | FOM | — | 0.0000 | 0.0000 | 0.0000 | 60.950 | 63.148 | — (is a FOM) | — | — |
| `fft_tight` | FOM | — | 0.0000 | 0.0000 | 0.0000 | 88.278 | 90.376 | — (is a FOM) | — | — |
| `nt1e-2_dt005` | FOM | — | 3.7127 | 1.4783 | 3.7127 | 15.311 | 17.502 | — (is a FOM) | — | — |
| `nt1e-2_dt01` | FOM | — | 3.1999 | 1.4192 | 3.1999 | 8.875 | 10.857 | — (is a FOM) | — | — |
| `nt1e-3_dt005` | FOM | — | 0.0489 | 0.0335 | 0.0489 | 31.081 | 33.285 | — (is a FOM) | — | — |
| `nt1e-3_dt01` | FOM | — | 1.5179 | 1.1980 | 1.5179 | 19.735 | 21.952 | — (is a FOM) | — | — |
| `nt1e-4_dt005` | FOM | — | 0.0338 | 0.0153 | 0.0338 | 35.771 | 38.057 | — (is a FOM) | — | — |
| `nt1e-4_dt01` | FOM | — | 1.5109 | 1.1722 | 1.5109 | 27.925 | 30.203 | — (is a FOM) | — | — |
| `don-large` | DeepONet | **newly timed** | 34.8360 | 18.1868 | 34.8360 | 10.574 | 10.818 | `nt1e-2_dt01` | 0.839× | 1.004× |
| `don-medium` | DeepONet | **newly timed** — best worst case on the DeepONet lane's validation set; not selected | 33.6863 | 16.0695 | 33.6863 | 5.638 | 5.897 | `nt1e-2_dt01` | 1.574× | 1.841× |
| `don-refine` | DeepONet | **newly timed** | 32.5309 | 14.2475 | 32.5309 | 3.712 | 3.957 | `nt1e-2_dt01` | 2.391× | 2.743× |
| `don-small` | DeepONet | **newly timed** — validation-selected | 36.2959 | 14.7016 | 36.2959 | 3.721 | 3.963 | `nt1e-2_dt01` | 2.385× | 2.740× |
