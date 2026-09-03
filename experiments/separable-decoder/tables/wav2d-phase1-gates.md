| N | BC | gate | value | pass rule | negative control | verdict |
|---|---|---|---|---|---|---|
| 128 | ref | F0-stencil | 1.82e-16 | ≤ 1e-13 | 2.54e-08 | PASS |
| 128 | ref | F0a | 1.41e-16 | ≤ 1e-13 | 4.70e-03 | PASS |
| 128 | ref | F0d-sym | 0 | ≤ 1e-15 | 1.26e-03 | PASS |
| 128 | ref | F1a | 6.94e-15 | ≤ 1e-10 | 0.598 | PASS |
| 128 | ref | F1a-form | 1.54e-16 | ≤ 1e-14 | 7.89e-03 | PASS |
| 128 | ref | F4 | 0 | ≤ 1e-10 | 0.598 | PASS |
| 128 | ref | V1alg | 1.80e-14 over 10 steps; full horizon 1.04e-11 | ≤ 1e-13 | 1.03e-03 | PASS |
| 128 | ref | V1cg | CG ladder 2.70e-05, 2.40e-08, 3.41e-11 (monotone True); achieved CG resid 6.11e-14 | ≤ 1e-08 | — | PASS |
| 128 | ref | F2-spatial | bump order 2.1892 (errors 0.052, 0.013, 2.52e-03); two-mode order 2.1943; N [33, 65, 129] vs 257 | see design | wrong-reference order 0.3335 | PASS |
| 128 | ref | F2-temporal | order 2.0104 (errors 8.14e-04, 2.03e-04, 5.02e-05) | see design | BE order 0.921, separation 575.00x | PASS |
| 128 | ref | V0 | 3.36e-15 | ≤ 1e-13 | 9.46e-06 | PASS |
| 128 | abs | F0-stencil | 2.02e-16 | ≤ 1e-13 | 2.81e-08 | PASS |
| 128 | abs | F0b-zero-mode | 3.27e-18 | ≤ 1e-13 | 7.68e-08 | PASS |
| 128 | abs | F0b | 8.97e-17 | ≤ 1e-13 | 5.25e-03 | PASS |
| 128 | abs | F0d-sym | 0 | ≤ 1e-15 | 0.056 | PASS |
| 128 | abs | F0d-spd | min eig/max(M) = 0.271 | see design | — | PASS |
| 128 | abs | F0c | 2.18e-14 | ≤ 1e-12 | 0.058 | PASS |
| 128 | abs | F1b | 7.98e-14 | ≤ 1e-10 | 3.13e-05 | PASS |
| 128 | abs | F4 | 0 | ≤ 1e-10 | 2.24e+45 | PASS |
| 128 | abs | V1alg | 1.74e-14 over 10 steps; full horizon 3.73e-12 | ≤ 1e-13 | 9.68e-03 | PASS |
| 128 | abs | V1cg | CG ladder 1.55e-05, 5.44e-08, 1.36e-09 (monotone True); achieved CG resid 2.07e-15 | ≤ 1e-08 | — | PASS |
| 128 | abs | F2-spatial | bump order 2.2151 (errors 0.026, 6.02e-03, 1.20e-03); two-mode order 1.8042; N [33, 65, 129] vs 257 | see design | wrong-reference order 0.3419 | PASS |
| 128 | abs | F2-temporal | order 2.0107 (errors 3.97e-04, 9.91e-05, 2.45e-05) | see design | BE order 0.957, separation 576.25x | PASS |
| 128 | abs | F5 | -4.51e-15 | ≤ 1e-12 | 2.828 | PASS |
| 128 | abs | F3 | slope 4.0096; fraction/prediction 1.0215, 1.0052, 1.0013, 1.0003 at N [64, 128, 256, 512]; plateau/fraction 1.0000, 1.0000, 1.0000, 1.0000; y-var 5.2e-15, 4.2e-15, 4.9e-15, 5.4e-15 | see design | reflective retains 1.0000 | PASS |
| 64 | ref | F0-stencil | 2.43e-16 | ≤ 1e-13 | 3.33e-08 | PASS |
| 64 | ref | F0a | 1.20e-16 | ≤ 1e-13 | 4.70e-03 | PASS |
| 64 | ref | F0d-sym | 0 | ≤ 1e-15 | 2.55e-03 | PASS |
| 64 | ref | F1a | 2.49e-15 | ≤ 1e-10 | 0.387 | PASS |
| 64 | ref | F1a-form | 1.19e-16 | ≤ 1e-14 | 0.015 | PASS |
| 64 | ref | F4 | 4.44e-16 | ≤ 1e-10 | 0.387 | PASS |
| 64 | ref | V1alg | 1.75e-14 over 10 steps; full horizon 2.96e-11 | ≤ 1e-13 | 2.80e-04 | PASS |
| 64 | ref | V1cg | CG ladder 7.71e-07, 7.49e-07, 1.62e-10 (monotone True); achieved CG resid 3.61e-16 | ≤ 1e-08 | — | PASS |
| 64 | ref | F2-spatial | bump order 2.1892 (errors 0.052, 0.013, 2.52e-03); two-mode order 2.1943; N [33, 65, 129] vs 257 | see design | wrong-reference order 0.3335 | PASS |
| 64 | ref | F2-temporal | order 2.0104 (errors 8.09e-04, 2.02e-04, 4.98e-05) | see design | BE order 0.921, separation 576.87x | PASS |
| 64 | ref | V0 | 2.35e-15 | ≤ 1e-13 | 8.09e-06 | PASS |
| 64 | abs | F0-stencil | 2.25e-16 | ≤ 1e-13 | 3.08e-08 | PASS |
| 64 | abs | F0b-zero-mode | 3.24e-18 | ≤ 1e-13 | 3.13e-07 | PASS |
| 64 | abs | F0b | 8.42e-17 | ≤ 1e-13 | 5.25e-03 | PASS |
| 64 | abs | F0d-sym | 0 | ≤ 1e-15 | 0.079 | PASS |
| 64 | abs | F0d-spd | min eig/max(M) = 0.260 | see design | — | PASS |
| 64 | abs | F0c | 8.41e-15 | ≤ 1e-12 | 0.058 | PASS |
| 64 | abs | F1b | 7.01e-14 | ≤ 1e-10 | 1.51e-05 | PASS |
| 64 | abs | F4 | 0 | ≤ 1e-10 | 1.48e+20 | PASS |
| 64 | abs | V1alg | 1.61e-14 over 10 steps; full horizon 1.64e-11 | ≤ 1e-13 | 2.10e-03 | PASS |
| 64 | abs | V1cg | CG ladder 1.96e-06, 6.07e-08, 3.00e-11 (monotone True); achieved CG resid 1.22e-15 | ≤ 1e-08 | — | PASS |
| 64 | abs | F2-spatial | bump order 2.2151 (errors 0.026, 6.02e-03, 1.20e-03); two-mode order 1.8042; N [33, 65, 129] vs 257 | see design | wrong-reference order 0.3419 | PASS |
| 64 | abs | F2-temporal | order 2.0107 (errors 3.99e-04, 9.95e-05, 2.46e-05) | see design | BE order 0.957, separation 576.34x | PASS |
| 64 | abs | F5 | -4.51e-15 | ≤ 1e-12 | 2.828 | PASS |
| 64 | abs | F3 | slope 4.0096; fraction/prediction 1.0215, 1.0052, 1.0013, 1.0003 at N [64, 128, 256, 512]; plateau/fraction 1.0000, 1.0000, 1.0000, 1.0000; y-var 5.5e-15, 4.2e-15, 6.0e-15, 6.2e-15 | see design | reflective retains 1.0000 | PASS |

| N | BC | reported quantity | value |
|---|---|---|---|
| 128 | abs | family-blob energy ratio E(T)/E0, E(4T)/E0 (F1/F4 trajectory) | 7.29e-04, 1.58e-05 |
| 128 | abs | family widest blob (w=0.190) spatial order, NOT gated | 1.1920 (errors 8.77e-03, 3.85e-03, 1.68e-03) |
| 128 | ref | family widest blob (w=0.190) spatial order, NOT gated | 0.3479 (errors 0.044, 0.036, 0.027) |
| 128 | ref | V0 energy agreement with the frozen 08-14 FOM | 1.56e-15 |
| 128 | both | provenance | commit b13e7c60, backend gpu, jax 0.10.1, matmul highest, wall 98.3 s |
| 64 | abs | family-blob energy ratio E(T)/E0, E(4T)/E0 (F1/F4 trajectory) | 1.00e-03, 3.12e-05 |
| 64 | abs | family widest blob (w=0.190) spatial order, NOT gated | 1.1920 (errors 8.77e-03, 3.85e-03, 1.68e-03) |
| 64 | ref | family widest blob (w=0.190) spatial order, NOT gated | 0.3479 (errors 0.044, 0.036, 0.027) |
| 64 | ref | V0 energy agreement with the frozen 08-14 FOM | 4.79e-15 |
| 64 | both | provenance | commit b13e7c60, backend gpu, jax 0.10.1, matmul highest, wall 62.1 s |
