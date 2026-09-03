| N | BC | head | K | params | final loss | D0 | D1 held-out/POD-K (ctrl shuffled) | D2 min cond (ctrl dup.) | G0a ratio, gap (ctrl) | G0b tangent/POD-K (ctrl random) | G0 | predicted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | abs | sup | 6 | 42560 | 9.69e-03 | PASS (orth 1.11e-14, floor 0.0250, sigma_R/sigma_1 0.016) | 0.3107 (6.493) PASS | 2.95e-03 (0) PASS | 0.7326, -0.0535 (7.8544) PASS | 0.3247 (1.2232) PASS | PASS | PASS |
| 128 | abs | auto | 8 | 42944 | 6.16e-03 | PASS (orth 1.11e-14, floor 0.0250, sigma_R/sigma_1 0.016) | 0.2118 (13.076) PASS | 0.064 (0) PASS | 0.9984, -1.179e-04 (16.6938) PASS | 0.4161 (1.3953) PASS | PASS | FAIL |
| 128 | abs | auto+vc | 8 | 42944 | 0.021 | PASS (orth 1.11e-14, floor 0.0250, sigma_R/sigma_1 0.016) | 0.2152 (12.868) PASS | 0.056 (0) PASS | 0.9692, -2.334e-03 (17.2197) PASS | 0.3418 (1.3953) PASS | PASS | PASS |
| 128 | ref | sup | 6 | 42560 | 0.037 | PASS (orth 1.04e-14, floor 0.0767, sigma_R/sigma_1 0.021) | 0.6846 (3.677) FAIL | 3.71e-03 (0) PASS | 0.9572, -0.0117 (4.6852) PASS | 0.8981 (1.1787) FAIL | FAIL | PASS |
| 128 | ref | auto | 8 | 42944 | 0.032 | PASS (orth 1.04e-14, floor 0.0767, sigma_R/sigma_1 0.021) | 0.6345 (4.852) FAIL | 0.082 (0) PASS | 1.0789, 0.0142 (10.3949) PASS | 0.9553 (1.2379) PASS | PASS | FAIL |
| 128 | ref | auto+vc | 8 | 42944 | 0.161 | PASS (orth 1.04e-14, floor 0.0767, sigma_R/sigma_1 0.021) | 0.6258 (4.921) FAIL | 0.104 (0) PASS | 1.0718, 0.0128 (10.3677) PASS | 0.9358 (1.2379) PASS | PASS | PASS |
| 64 | abs | sup | 6 | 42560 | 9.99e-03 | PASS (orth 1.05e-14, floor 0.0256, sigma_R/sigma_1 0.017) | 0.2655 (7.640) PASS | 2.49e-03 (0) PASS | 0.8209, -0.0273 (10.9204) PASS | 0.3172 (1.2178) PASS | PASS | PASS |
| 64 | abs | auto | 8 | 42944 | 6.78e-03 | PASS (orth 1.05e-14, floor 0.0256, sigma_R/sigma_1 0.017) | 0.2183 (12.623) PASS | 0.082 (0) PASS | 0.9872, -9.682e-04 (16.5866) PASS | 0.4368 (1.3750) PASS | PASS | FAIL |
| 64 | abs | auto+vc | 8 | 42944 | 0.023 | PASS (orth 1.05e-14, floor 0.0256, sigma_R/sigma_1 0.017) | 0.2222 (12.422) PASS | 0.095 (0) PASS | 0.9460, -4.332e-03 (12.7449) PASS | 0.3565 (1.3750) PASS | PASS | PASS |
| 64 | ref | sup | 6 | 42560 | 0.036 | PASS (orth 1.07e-14, floor 0.0693, sigma_R/sigma_1 0.021) | 0.6492 (3.912) FAIL | 3.66e-03 (0) PASS | 0.8134, -0.0564 (6.1720) PASS | 0.8594 (1.2799) PASS | PASS | PASS |
| 64 | ref | auto | 8 | 42944 | 0.032 | PASS (orth 1.07e-14, floor 0.0693, sigma_R/sigma_1 0.021) | 0.6156 (5.080) FAIL | 0.077 (0) PASS | 1.1192, 0.0198 (10.5424) PASS | 0.9531 (1.3883) PASS | PASS | FAIL |
| 64 | ref | auto+vc | 8 | 42944 | 0.159 | PASS (orth 1.07e-14, floor 0.0693, sigma_R/sigma_1 0.021) | 0.6173 (5.065) FAIL | 0.095 (0) PASS | 1.1044, 0.0177 (10.5143) PASS | 0.9150 (1.3883) PASS | PASS | PASS |

| N | BC | head | held-out oracle median | train oracle median | POD-K median | POD-R ceiling median | G0b tangent median / POD-K median (n states) |
|---|---|---|---|---|---|---|---|
| 128 | abs | sup | 0.14647 | 0.19993 | 0.47135 | 0.01743 | 0.25511 / 0.78562 (788) |
| 128 | abs | auto | 0.07232 | 0.07243 | 0.34141 | 0.01743 | 0.28232 / 0.67847 (788) |
| 128 | abs | auto+vc | 0.07347 | 0.07580 | 0.34141 | 0.01743 | 0.23188 / 0.67847 (788) |
| 128 | ref | sup | 0.26083 | 0.27249 | 0.38102 | 0.07293 | 0.74116 / 0.82529 (796) |
| 128 | ref | auto | 0.19377 | 0.17960 | 0.30540 | 0.07293 | 0.74147 / 0.77618 (796) |
| 128 | ref | auto+vc | 0.19113 | 0.17833 | 0.30540 | 0.07293 | 0.72639 / 0.77618 (796) |
| 64 | abs | sup | 0.12524 | 0.15256 | 0.47162 | 0.01771 | 0.25035 / 0.78936 (788) |
| 64 | abs | auto | 0.07464 | 0.07561 | 0.34186 | 0.01771 | 0.30094 / 0.68898 (788) |
| 64 | abs | auto+vc | 0.07596 | 0.08029 | 0.34186 | 0.01771 | 0.24565 / 0.68898 (788) |
| 64 | ref | sup | 0.24580 | 0.30217 | 0.37862 | 0.06385 | 0.64902 / 0.75518 (796) |
| 64 | ref | auto | 0.18615 | 0.16633 | 0.30239 | 0.06385 | 0.65304 / 0.68520 (796) |
| 64 | ref | auto+vc | 0.18667 | 0.16902 | 0.30239 | 0.06385 | 0.62695 / 0.68520 (796) |
