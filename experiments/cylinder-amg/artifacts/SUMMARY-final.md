
### Development Re 95 (one run each)

| solver | status | targets | e_u RMS | e_u max | e_p RMS | St rel | CD_max rel | CL_max abs | wall s (median) | min-max | setup s | solve s | Krylov it/solve (max) | inner it/solve |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lu | completed | 6/6 | 0.00831 | 0.0106 | 0.00442 | 0.00183 | 0.0022 | 0.014 | 49.81 | 49.81-49.81 | 5.92 | 18.3 | - (-) | - |
| lu (2) | completed | 6/6 | 0.00831 | 0.0106 | 0.00442 | 0.00183 | 0.0022 | 0.014 | 63.76 | 63.76-63.76 | 21.3 | 17.4 | - (-) | - |
| lu (3) | completed | 6/6 | 0.00831 | 0.0106 | 0.00442 | 0.00183 | 0.0022 | 0.014 | 49.73 | 49.73-49.73 | 8.37 | 18 | - (-) | - |
| amg1_sa_cc_r1e-10 | completed | 6/6 | 0.00831 | 0.0106 | 0.00442 | 0.00183 | 0.0022 | 0.014 | 162.5 | 162.5-162.5 | 4.15 | 121 | 40.5 (80) | 0 |
| amg1_sa_cc_r1e-8 | completed | 6/6 | 0.00831 | 0.0106 | 0.00442 | 0.00183 | 0.0022 | 0.0141 | 128.8 | 128.8-128.8 | 3.96 | 86.4 | 27.5 (50) | 0 |
| amg1_sa_cc_r1e-6 | completed | 6/6 | 0.00832 | 0.0107 | 0.00447 | 0.0018 | 0.00221 | 0.0143 | 89.97 | 89.97-89.97 | 3.85 | 50.1 | 15.9 (34) | 0 |
| amg1_sa_cc_r1e-4 | completed | 2/6 | 0.0116 | 0.0152 | 0.333 | 26.9 | 0.0113 | 0.848 | 91.77 | 91.77-91.77 | 3.79 | 47.8 | 14 (20) | 0 |
| amg1_rs_cc_r1e-6 | completed | 3/6 | 0.00832 | 0.0106 | 0.179 | 8.8 | 0.0148 | 0.456 | 103 | 103-103 | 1.43 | 65.3 | 19.3 (44) | 0 |
| amg1_sa_mass_r1e-6 | failed | -/6 | - | - | - | - | - | - | - | --- | - | - | - (-) | - |
| amg2_sa_cc_r1e-10 | completed | 6/6 | 0.00831 | 0.0106 | 0.00442 | 0.00183 | 0.0022 | 0.014 | 1345 | 1345-1345 | 3.72 | 1.32e+03 | 22.8 (52) | 1.87e+03 |
| amg2_sa_cc_r1e-8 | completed | 6/6 | 0.00831 | 0.0106 | 0.00442 | 0.00183 | 0.0022 | 0.014 | 1006 | 1006-1006 | 3.21 | 987 | 17 (34) | 1.39e+03 |
| amg2_sa_cc_r1e-6 | completed | 6/6 | 0.00831 | 0.0106 | 0.00445 | 0.00179 | 0.00221 | 0.014 | 823.7 | 823.7-823.7 | 3.44 | 800 | 13.8 (19) | 1.11e+03 |
| amg2_sa_cc_r1e-4 | completed | 2/6 | 0.0123 | 0.016 | 0.569 | 18 | 0.00668 | 0.655 | 754.6 | 754.6-754.6 | 3.65 | 731 | 12.7 (18) | 1.01e+03 |

LU runs identical across jobs: True 
selected: {'amg1_sa_cc': 'amg1_sa_cc_r1e-8', 'amg2_sa_cc': 'amg2_sa_cc_r1e-8'}

### Held-out Re 105, allocation amg-hoa (5 timed repetitions each)

| solver | status | targets | e_u RMS | e_u max | e_p RMS | St rel | CD_max rel | CL_max abs | wall s (median) | min-max | setup s | solve s | Krylov it/solve (max) | inner it/solve |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lu [amg-hoa] | completed | 6/6 | 0.00939 | 0.0125 | 0.00521 | 0.0021 | 0.00271 | 0.0149 | 52.95 | 51.22-53.92 | 6.93 | 20.7 | - (-) | - |
| amg1_sa_cc_r1e-8 [amg-hoa] | completed | 6/6 | 0.00939 | 0.0125 | 0.00521 | 0.0021 | 0.00271 | 0.0149 | 126.2 | 125.2-134.6 | 4.06 | 88 | 28.7 (51) | 0 |
| amg1_sa_cc_r1e-6 [amg-hoa] | completed | 6/6 | 0.0094 | 0.0126 | 0.00524 | 0.00213 | 0.00273 | 0.0151 | 99.19 | 96.39-102.3 | 4.05 | 57.8 | 16.7 (35) | 0 |
| lu_b [amg-hoa] | completed | 6/6 | 0.00939 | 0.0125 | 0.00521 | 0.0021 | 0.00271 | 0.0149 | 57.42 | 49.25-61.31 | 9.71 | 18.4 | - (-) | - |

wall / paired LU of amg-hoa (faster LU block): {'lu [amg-hoa]': 1.0, 'amg1_sa_cc_r1e-8 [amg-hoa]': 2.383, 'amg1_sa_cc_r1e-6 [amg-hoa]': 1.873, 'lu_b [amg-hoa]': 1.084}

### Held-out Re 105, allocation amg-hob (5 timed repetitions each)

| solver | status | targets | e_u RMS | e_u max | e_p RMS | St rel | CD_max rel | CL_max abs | wall s (median) | min-max | setup s | solve s | Krylov it/solve (max) | inner it/solve |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lu [amg-hob] | completed | 6/6 | 0.00939 | 0.0125 | 0.00521 | 0.0021 | 0.00271 | 0.0149 | 81.93 | 60.66-151 | 35.9 | 20.1 | - (-) | - |
| amg2_sa_cc_r1e-8 [amg-hob] | completed | 6/6 | 0.00939 | 0.0125 | 0.00521 | 0.0021 | 0.00271 | 0.0149 | 1158 | 1154-1179 | 3.16 | 1.14e+03 | 17.9 (35) | 1.49e+03 |
| lu_b [amg-hob] | completed | 6/6 | 0.00939 | 0.0125 | 0.00521 | 0.0021 | 0.00271 | 0.0149 | 54.13 | 53.93-62.87 | 10.1 | 17.8 | - (-) | - |

wall / paired LU of amg-hob (faster LU block): {'lu [amg-hob]': 1.514, 'amg2_sa_cc_r1e-8 [amg-hob]': 21.397, 'lu_b [amg-hob]': 1.0}
wall minus (factor/setup + solve) timers, i.e. the part no linear solver can remove: {'lu [amg-hoa]': 24.59, 'amg1_sa_cc_r1e-8 [amg-hoa]': 36.0, 'amg1_sa_cc_r1e-6 [amg-hoa]': 35.61, 'lu_b [amg-hoa]': 27.52, 'lu [amg-hob]': 24.97, 'amg2_sa_cc_r1e-8 [amg-hob]': 14.99, 'lu_b [amg-hob]': 27.75}
wall / record 42.25 s: {'lu [amg-hoa]': 1.253, 'amg1_sa_cc_r1e-8 [amg-hoa]': 2.986, 'amg1_sa_cc_r1e-6 [amg-hoa]': 2.348, 'lu_b [amg-hoa]': 1.359, 'lu [amg-hob]': 1.939, 'amg2_sa_cc_r1e-8 [amg-hob]': 27.414, 'lu_b [amg-hob]': 1.281}
