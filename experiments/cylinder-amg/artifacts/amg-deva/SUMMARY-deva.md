
### Development Re 95 (one run each)

| solver | status | targets | e_u RMS | e_u max | e_p RMS | St rel | CD_max rel | CL_max abs | wall s (median) | min-max | setup s | solve s | Krylov it/solve (max) | inner it/solve |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lu | completed | 6/6 | 0.00831 | 0.0106 | 0.00442 | 0.00183 | 0.0022 | 0.014 | 49.81 | 49.81-49.81 | 5.92 | 18.3 | - (-) | - |
| amg1_sa_cc_r1e-10 | completed | 6/6 | 0.00831 | 0.0106 | 0.00442 | 0.00183 | 0.0022 | 0.014 | 162.5 | 162.5-162.5 | 4.15 | 121 | 40.5 (80) | 0 |
| amg1_sa_cc_r1e-8 | completed | 6/6 | 0.00831 | 0.0106 | 0.00442 | 0.00183 | 0.0022 | 0.0141 | 128.8 | 128.8-128.8 | 3.96 | 86.4 | 27.5 (50) | 0 |
| amg1_sa_cc_r1e-6 | completed | 6/6 | 0.00832 | 0.0107 | 0.00447 | 0.0018 | 0.00221 | 0.0143 | 89.97 | 89.97-89.97 | 3.85 | 50.1 | 15.9 (34) | 0 |
| amg1_sa_cc_r1e-4 | completed | 2/6 | 0.0116 | 0.0152 | 0.333 | 26.9 | 0.0113 | 0.848 | 91.77 | 91.77-91.77 | 3.79 | 47.8 | 14 (20) | 0 |
| amg1_rs_cc_r1e-6 | completed | 3/6 | 0.00832 | 0.0106 | 0.179 | 8.8 | 0.0148 | 0.456 | 103 | 103-103 | 1.43 | 65.3 | 19.3 (44) | 0 |
| amg1_sa_mass_r1e-6 | failed | -/6 | - | - | - | - | - | - | - | --- | - | - | - (-) | - |

LU runs identical across jobs: True 
selected: {'amg1_sa_cc': 'amg1_sa_cc_r1e-8', 'amg2_sa_cc': None}
