# Final Poisson correction family

Audited frozen correction-family results on already-opened development cases. These are a combined change in linear capacity and analytic elimination; no final-cohort claim.

## all

| Intervals | Method | GPU ms | Host ms | Worst error % | Adjusted error % | Invalid calls | Pass 5% | Bank pass 3% |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 64 | cg_1e-01 | 4.033062 | 4.919995 | 9.627059 | 9.627920 | 0 | False | n/a |
| 64 | cg_1e-06 | 8.385411 | 9.331011 | 0.273175 | 0.273959 | 0 | True | n/a |
| 64 | cg_3e-02 | 4.725828 | 5.616743 | 1.376769 | 1.377565 | 0 | True | n/a |
| 64 | dst | 0.256050 | 1.892254 | 0.273175 | 0.273959 | 0 | True | n/a |
| 64 | original_relative | 4.830095 | 6.358523 | 7.281747 | 7.282586 | 0 | False | False |
| 64 | r128_q0 | 4.766349 | 6.275453 | 7.555330 | 7.556171 | 0 | False | False |
| 64 | r128_q16 | 4.893498 | 6.425868 | 7.034354 | 7.035191 | 0 | False | False |
| 64 | r128_q32 | 4.910128 | 6.465681 | 6.111946 | 6.112346 | 0 | False | False |
| 64 | r128_q8 | 4.852020 | 6.369687 | 7.209430 | 7.210268 | 0 | False | False |
| 256 | cg_1e-01 | 15.021542 | 16.278673 | 3.036038 | 3.036847 | 0 | True | n/a |
| 256 | cg_1e-06 | 34.804448 | 36.210974 | 0.016506 | 0.017291 | 0 | True | n/a |
| 256 | cg_3e-02 | 17.501463 | 18.643897 | 0.721700 | 0.722488 | 0 | True | n/a |
| 256 | dst | 0.209141 | 2.184999 | 0.016506 | 0.017291 | 0 | True | n/a |
| 256 | original_relative | 4.622388 | 6.362127 | 7.280264 | 7.281103 | 0 | False | False |
| 256 | r128_q0 | 4.633263 | 6.336176 | 7.553790 | 7.554631 | 0 | False | False |
| 256 | r128_q16 | 4.813237 | 6.573889 | 7.032166 | 7.033003 | 0 | False | False |
| 256 | r128_q32 | 4.744899 | 6.652622 | 6.110598 | 6.110998 | 0 | False | False |
| 256 | r128_q8 | 4.764289 | 6.587218 | 7.207542 | 7.208380 | 0 | False | False |
| 1024 | cg_1e-01 | 89.205305 | 96.314497 | 1.182643 | 1.183437 | 0 | True | n/a |
| 1024 | cg_1e-06 | 251.533784 | 258.338489 | 0.000785 | 0.001570 | 0 | True | n/a |
| 1024 | cg_3e-02 | 103.618530 | 110.952986 | 0.269520 | 0.270304 | 0 | True | n/a |
| 1024 | dst | 0.395368 | 8.098022 | 0.000785 | 0.001570 | 0 | True | n/a |
| 1024 | original_relative | 3.927122 | 11.218030 | 7.280248 | 7.281086 | 0 | False | False |
| 1024 | r128_q0 | 4.412962 | 11.692039 | 7.553765 | 7.554606 | 0 | False | False |
| 1024 | r128_q16 | 5.232792 | 12.490237 | 7.032119 | 7.032956 | 0 | False | False |
| 1024 | r128_q32 | 6.043163 | 13.092479 | 6.110576 | 6.110976 | 0 | False | False |
| 1024 | r128_q8 | 5.031713 | 12.251328 | 7.207505 | 7.208343 | 0 | False | False |

## existing_development

| Intervals | Method | GPU ms | Host ms | Worst error % | Adjusted error % | Invalid calls | Pass 5% | Bank pass 3% |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 64 | cg_1e-01 | 4.037548 | 4.918754 | 8.816888 | 8.817684 | 0 | False | n/a |
| 64 | cg_1e-06 | 8.350572 | 9.269099 | 0.257217 | 0.257954 | 0 | True | n/a |
| 64 | cg_3e-02 | 4.646911 | 5.583930 | 1.134792 | 1.135430 | 0 | True | n/a |
| 64 | dst | 0.255500 | 1.893389 | 0.257217 | 0.257954 | 0 | True | n/a |
| 64 | original_relative | 4.837546 | 6.375568 | 6.802558 | 6.802961 | 0 | False | False |
| 64 | r128_q0 | 4.827752 | 6.334804 | 7.241974 | 7.242379 | 0 | False | False |
| 64 | r128_q16 | 4.977213 | 6.508232 | 6.728617 | 6.729020 | 0 | False | False |
| 64 | r128_q32 | 4.985445 | 6.527685 | 6.111946 | 6.112346 | 0 | False | False |
| 64 | r128_q8 | 4.912461 | 6.430540 | 7.016663 | 7.017067 | 0 | False | False |
| 256 | cg_1e-01 | 14.907802 | 16.048990 | 2.272480 | 2.273038 | 0 | True | n/a |
| 256 | cg_1e-06 | 34.540677 | 35.740303 | 0.015460 | 0.016195 | 0 | True | n/a |
| 256 | cg_3e-02 | 17.390469 | 18.545393 | 0.548574 | 0.549313 | 0 | True | n/a |
| 256 | dst | 0.201958 | 2.057395 | 0.015460 | 0.016195 | 0 | True | n/a |
| 256 | original_relative | 4.547189 | 6.280640 | 6.801564 | 6.801967 | 0 | False | False |
| 256 | r128_q0 | 4.581484 | 6.287084 | 7.241040 | 7.241445 | 0 | False | False |
| 256 | r128_q16 | 4.739108 | 6.473097 | 6.727422 | 6.727825 | 0 | False | False |
| 256 | r128_q32 | 4.713714 | 6.456587 | 6.110598 | 6.110998 | 0 | False | False |
| 256 | r128_q8 | 4.738160 | 6.493816 | 7.015667 | 7.016070 | 0 | False | False |
| 1024 | cg_1e-01 | 89.356721 | 96.569828 | 1.019173 | 1.019810 | 0 | True | n/a |
| 1024 | cg_1e-06 | 252.406877 | 259.173320 | 0.000735 | 0.001470 | 0 | True | n/a |
| 1024 | cg_3e-02 | 103.952786 | 111.513531 | 0.255270 | 0.256004 | 0 | True | n/a |
| 1024 | dst | 0.406698 | 8.137751 | 0.000735 | 0.001470 | 0 | True | n/a |
| 1024 | original_relative | 3.985213 | 11.415966 | 6.801556 | 6.801959 | 0 | False | False |
| 1024 | r128_q0 | 4.500077 | 11.808928 | 7.241031 | 7.241435 | 0 | False | False |
| 1024 | r128_q16 | 5.327551 | 12.593797 | 6.727403 | 6.727805 | 0 | False | False |
| 1024 | r128_q32 | 6.080753 | 13.301051 | 6.110576 | 6.110976 | 0 | False | False |
| 1024 | r128_q8 | 5.169768 | 12.315184 | 7.015655 | 7.016059 | 0 | False | False |

## new_development

| Intervals | Method | GPU ms | Host ms | Worst error % | Adjusted error % | Invalid calls | Pass 5% | Bank pass 3% |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 64 | cg_1e-01 | 4.023542 | 4.932503 | 9.627059 | 9.627920 | 0 | False | n/a |
| 64 | cg_1e-06 | 8.503380 | 9.436510 | 0.273175 | 0.273959 | 0 | True | n/a |
| 64 | cg_3e-02 | 4.864014 | 5.706542 | 1.376769 | 1.377565 | 0 | True | n/a |
| 64 | dst | 0.257035 | 1.892254 | 0.273175 | 0.273959 | 0 | True | n/a |
| 64 | original_relative | 4.830095 | 6.266830 | 7.281747 | 7.282586 | 0 | False | False |
| 64 | r128_q0 | 4.660647 | 6.093872 | 7.555330 | 7.556171 | 0 | False | False |
| 64 | r128_q16 | 4.787164 | 6.277555 | 7.034354 | 7.035191 | 0 | False | False |
| 64 | r128_q32 | 4.800403 | 6.338857 | 5.923512 | 5.924340 | 0 | False | False |
| 64 | r128_q8 | 4.665021 | 6.105634 | 7.209430 | 7.210268 | 0 | False | False |
| 256 | cg_1e-01 | 15.305478 | 16.659749 | 3.036038 | 3.036847 | 0 | True | n/a |
| 256 | cg_1e-06 | 35.243944 | 36.709281 | 0.016506 | 0.017291 | 0 | True | n/a |
| 256 | cg_3e-02 | 17.610967 | 18.977283 | 0.721700 | 0.722488 | 0 | True | n/a |
| 256 | dst | 0.238504 | 2.600783 | 0.016506 | 0.017291 | 0 | True | n/a |
| 256 | original_relative | 4.986715 | 7.147474 | 7.280264 | 7.281103 | 0 | False | False |
| 256 | r128_q0 | 4.772726 | 7.005186 | 7.553790 | 7.554631 | 0 | False | False |
| 256 | r128_q16 | 4.875962 | 7.384203 | 7.032166 | 7.033003 | 0 | False | False |
| 256 | r128_q32 | 5.037488 | 7.360116 | 5.918432 | 5.919260 | 0 | False | False |
| 256 | r128_q8 | 4.971688 | 7.154437 | 7.207542 | 7.208380 | 0 | False | False |
| 1024 | cg_1e-01 | 88.036112 | 95.129410 | 1.182643 | 1.183437 | 0 | True | n/a |
| 1024 | cg_1e-06 | 251.043694 | 257.866076 | 0.000785 | 0.001570 | 0 | True | n/a |
| 1024 | cg_3e-02 | 102.515089 | 109.570386 | 0.269520 | 0.270304 | 0 | True | n/a |
| 1024 | dst | 0.359407 | 8.036413 | 0.000785 | 0.001570 | 0 | True | n/a |
| 1024 | original_relative | 3.893853 | 11.111320 | 7.280248 | 7.281086 | 0 | False | False |
| 1024 | r128_q0 | 4.233030 | 11.392655 | 7.553765 | 7.554606 | 0 | False | False |
| 1024 | r128_q16 | 5.107983 | 12.221615 | 7.032119 | 7.032956 | 0 | False | False |
| 1024 | r128_q32 | 5.602443 | 12.495991 | 5.918245 | 5.919073 | 0 | False | False |
| 1024 | r128_q8 | 4.716730 | 11.759694 | 7.207505 | 7.208343 | 0 | False | False |

## Glossary

Intervals: cells per axis, so nodes per axis are one larger. GPU ms: pooled median timed device-query milliseconds. Host ms: input, device and output time. Worst error: largest current-relative full-field discrepancy against the fine numerical reference. Adjusted error: conservative error including measured reference refinement. Invalid calls: invocations failing the declared numerical gates. Pass 5%: every case/repetition passes adjusted physical error, refinement and numerical gates. Bank pass 3%: full-bank projection diagnostic meets the separate proposed target. Correction counts 8/16/32 add linear coordinates to the same 16-variable nonlinear head; q0 is the uncorrected model. CG: iterative conjugate gradients with named relative residual tolerance. DST: direct sine-transform FOM. Existing development: original 30 sources; new_development: later 12 sources, already opened before this experiment.
