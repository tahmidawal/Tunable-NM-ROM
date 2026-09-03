| N | BC | head | K | params | final loss | D0 | D1 held-out/POD-K (ctrl shuffled) | D2 min cond (ctrl dup.) | G0a ratio, gap (ctrl) | G0b tangent/POD-K (ctrl random) | G0 | predicted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 | abs | sup | 6 | 42560 | 9.99e-03 | PASS (orth 1.05e-14, floor 0.0256, sigma_R/sigma_1 0.017) | 0.2655 (7.640) PASS | 2.49e-03 (0) PASS | 0.8209, -0.0273 (10.9204) PASS | 0.3172 (1.2178) PASS | PASS | PASS |
| 64 | abs | auto | 8 | 42944 | 6.78e-03 | PASS (orth 1.05e-14, floor 0.0256, sigma_R/sigma_1 0.017) | 0.2183 (12.623) PASS | 0.082 (0) PASS | 0.9872, -9.682e-04 (16.5866) PASS | 0.4368 (1.3750) PASS | PASS | FAIL |
| 64 | abs | auto+vc | 8 | 42944 | 0.023 | PASS (orth 1.05e-14, floor 0.0256, sigma_R/sigma_1 0.017) | 0.2222 (12.422) PASS | 0.095 (0) PASS | 0.9460, -4.332e-03 (12.7449) PASS | 0.3565 (1.3750) PASS | PASS | PASS |
| 64 | ref | sup | 6 | 42560 | 0.036 | PASS (orth 1.07e-14, floor 0.0693, sigma_R/sigma_1 0.021) | 0.6492 (3.912) FAIL | 3.66e-03 (0) PASS | 0.8134, -0.0564 (6.1720) PASS | 0.8594 (1.2799) PASS | PASS | PASS |
| 64 | ref | auto | 8 | 42944 | 0.032 | PASS (orth 1.07e-14, floor 0.0693, sigma_R/sigma_1 0.021) | 0.6156 (5.080) FAIL | 0.077 (0) PASS | 1.1192, 0.0198 (10.5424) PASS | 0.9531 (1.3883) PASS | PASS | FAIL |
| 64 | ref | auto+vc | 8 | 42944 | 0.159 | PASS (orth 1.07e-14, floor 0.0693, sigma_R/sigma_1 0.021) | 0.6173 (5.065) FAIL | 0.095 (0) PASS | 1.1044, 0.0177 (10.5143) PASS | 0.9150 (1.3883) PASS | PASS | PASS |

| N | BC | head | held-out oracle median | train oracle median | POD-K median | POD-R ceiling median | G0b tangent median / POD-K median (n states) |
|---|---|---|---|---|---|---|---|
| 64 | abs | sup | 0.12524 | 0.15256 | 0.47162 | 0.01771 | 0.25035 / 0.78936 (788) |
| 64 | abs | auto | 0.07464 | 0.07561 | 0.34186 | 0.01771 | 0.30094 / 0.68898 (788) |
| 64 | abs | auto+vc | 0.07596 | 0.08029 | 0.34186 | 0.01771 | 0.24565 / 0.68898 (788) |
| 64 | ref | sup | 0.24580 | 0.30217 | 0.37862 | 0.06385 | 0.64902 / 0.75518 (796) |
| 64 | ref | auto | 0.18615 | 0.16633 | 0.30239 | 0.06385 | 0.65304 / 0.68520 (796) |
| 64 | ref | auto+vc | 0.18667 | 0.16902 | 0.30239 | 0.06385 | 0.62695 / 0.68520 (796) |
