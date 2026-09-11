# Audited Poisson bank-capacity comparison

These are accepted measurements on already-opened development cases. No neural endpoint meets complete-cohort physical eligibility; bank and online targets remain separate.

| Cohort | Intervals | Method | GPU ms | Host ms | Worst field error (%) | Invalid | Physical target passes all |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| all | 64 | cg_1e-01 | 3.058094997 | 3.964717616 | 9.627059360 | 0 | False |
| all | 64 | cg_1e-06 | 6.020177971 | 6.963362452 | 0.273174612 | 0 | True |
| all | 64 | cg_3e-02 | 3.483387991 | 4.403996048 | 1.376768812 | 0 | True |
| all | 64 | dst | 0.141695025 | 1.751100994 | 0.273174595 | 0 | True |
| all | 64 | original_relative | 2.403330989 | 3.430020530 | 7.281746791 | 0 | False |
| all | 64 | r128_head | 2.396870521 | 3.411186510 | 7.387691100 | 0 | False |
| all | 64 | r128_joint | 2.355759963 | 3.407048527 | 7.555330339 | 0 | False |
| all | 64 | r64_head | 2.438977011 | 3.420327557 | 7.786737549 | 0 | False |
| all | 64 | r64_joint | 2.359436126 | 3.394084517 | 7.683120495 | 0 | False |
| all | 256 | cg_1e-01 | 10.800858610 | 12.039079913 | 3.036037861 | 0 | True |
| all | 256 | cg_1e-06 | 24.344405509 | 25.502974982 | 0.016505991 | 0 | True |
| all | 256 | cg_3e-02 | 12.619693065 | 13.823725400 | 0.721699967 | 0 | True |
| all | 256 | dst | 0.140484539 | 1.866018516 | 0.016506036 | 0 | True |
| all | 256 | original_relative | 2.368028508 | 3.535775002 | 7.280264410 | 0 | False |
| all | 256 | r128_head | 2.355972538 | 3.485481022 | 7.386227928 | 0 | False |
| all | 256 | r128_joint | 2.306459122 | 3.457924002 | 7.553790293 | 0 | False |
| all | 256 | r64_head | 2.364623942 | 3.491797834 | 7.785401273 | 0 | False |
| all | 256 | r64_joint | 2.311710850 | 3.525209962 | 7.681674572 | 0 | False |
| all | 1024 | cg_1e-01 | 83.922245540 | 87.625989458 | 1.182642857 | 0 | True |
| all | 1024 | cg_1e-06 | 186.990233487 | 190.897787921 | 0.000785193 | 0 | True |
| all | 1024 | cg_3e-02 | 97.868478973 | 101.794442977 | 0.269520101 | 0 | True |
| all | 1024 | dst | 0.328058959 | 4.505652585 | 0.000785131 | 0 | True |
| all | 1024 | original_relative | 2.610272495 | 6.245763972 | 7.280247642 | 0 | False |
| all | 1024 | r128_head | 2.870484372 | 6.464139558 | 7.386210144 | 0 | False |
| all | 1024 | r128_joint | 2.849252545 | 6.453854498 | 7.553764804 | 0 | False |
| all | 1024 | r64_head | 2.623934997 | 6.214081543 | 7.785384212 | 0 | False |
| all | 1024 | r64_joint | 2.584366943 | 6.217931514 | 7.681652988 | 0 | False |
| existing_development | 64 | original_relative | 2.418894554 | 3.448627074 | 6.802558107 | 0 | False |
| existing_development | 64 | r128_head | 2.390017500 | 3.396869986 | 6.952603768 | 0 | False |
| existing_development | 64 | r128_joint | 2.374464995 | 3.405548050 | 7.241974121 | 0 | False |
| existing_development | 64 | r64_head | 2.444370068 | 3.441689536 | 7.417989224 | 0 | False |
| existing_development | 64 | r64_joint | 2.378687495 | 3.394611063 | 6.839909709 | 0 | False |
| existing_development | 256 | original_relative | 2.363649081 | 3.481616965 | 6.801564318 | 0 | False |
| existing_development | 256 | r128_head | 2.345485380 | 3.474244964 | 6.951666471 | 0 | False |
| existing_development | 256 | r128_joint | 2.301919041 | 3.457924002 | 7.241040283 | 0 | False |
| existing_development | 256 | r64_head | 2.364623942 | 3.462780965 | 7.417101316 | 0 | False |
| existing_development | 256 | r64_joint | 2.304906026 | 3.514162498 | 6.838838637 | 0 | False |
| existing_development | 1024 | original_relative | 2.620680491 | 6.252412568 | 6.801556288 | 0 | False |
| existing_development | 1024 | r128_head | 2.895050566 | 6.496920483 | 6.951658172 | 0 | False |
| existing_development | 1024 | r128_joint | 2.854599035 | 6.476151408 | 7.241030517 | 0 | False |
| existing_development | 1024 | r64_head | 2.647182555 | 6.249436992 | 7.417093544 | 0 | False |
| existing_development | 1024 | r64_joint | 2.585209440 | 6.221145042 | 6.838829005 | 0 | False |
| new_development | 64 | original_relative | 2.298500505 | 3.361228621 | 7.281746791 | 0 | False |
| new_development | 64 | r128_head | 2.434501424 | 3.478593891 | 7.387691100 | 0 | False |
| new_development | 64 | r128_joint | 2.309801523 | 3.456350998 | 7.555330339 | 0 | False |
| new_development | 64 | r64_head | 2.344794571 | 3.397076041 | 7.786737549 | 0 | False |
| new_development | 64 | r64_joint | 2.261976944 | 3.331950633 | 7.683120495 | 0 | False |
| new_development | 256 | original_relative | 2.374369418 | 3.580020508 | 7.280264410 | 0 | False |
| new_development | 256 | r128_head | 2.380653401 | 3.487601993 | 7.386227928 | 0 | False |
| new_development | 256 | r128_joint | 2.326373360 | 3.473943914 | 7.553790293 | 0 | False |
| new_development | 256 | r64_head | 2.357885125 | 3.604071564 | 7.785401273 | 0 | False |
| new_development | 256 | r64_joint | 2.360188984 | 3.595152870 | 7.681674572 | 0 | False |
| new_development | 1024 | original_relative | 2.585379989 | 6.225148565 | 7.280247642 | 0 | False |
| new_development | 1024 | r128_head | 2.793289488 | 6.439503399 | 7.386210144 | 0 | False |
| new_development | 1024 | r128_joint | 2.820843598 | 6.421691505 | 7.553764804 | 0 | False |
| new_development | 1024 | r64_head | 2.604767098 | 6.155983428 | 7.785384212 | 0 | False |
| new_development | 1024 | r64_joint | 2.567450632 | 6.215336500 | 7.681652988 | 0 | False |

The `new_development` group label is retained for continuity: these later cases were already opened before capacity selection. Neither group is fresh independent confirmation for this study.

Full-bank projections and bounded stationary full-field head fits remain diagnostic; all values and per-case decompositions are in `panel.json`. Every timing row includes its corresponding full field from the same invocation.

| Saved head | Correction directions | Training residual energy captured (%) | Worst training reconstruction (%) |
| --- | ---: | ---: | ---: |
| r128_head | 0 | 0.000000000 | 5.616718248 |
| r128_head | 8 | 27.875203665 | 5.042698563 |
| r128_head | 16 | 46.654265569 | 4.012149403 |
| r128_head | 32 | 70.390078422 | 3.077072529 |
| r128_joint | 0 | 0.000000000 | 4.994142516 |
| r128_joint | 8 | 27.676685915 | 4.403181018 |
| r128_joint | 16 | 46.731305797 | 3.334735892 |
| r128_joint | 32 | 70.231433340 | 2.817541088 |

Correction directions are constructed only from normalized training residuals in the physical QR metric. The table uses saved training latent codes with analytically fitted linear corrections and does not measure an online PDE solve. The nonlinear training codes are not independently certified optimal fits. No measured online gain or speed gain for corrections is claimed.

Glossary: intervals counts mesh subdivisions per axis; cohort identifies original versus later opened development cases; GPU/host ms are pooled medians of equal-count repetitions; worst field error is current-relative full-field L2 error versus restricted fine truth; invalid means a failed numerical solver gate; physical eligibility includes the declared error and refinement gates; bank is the learned spatial feature span; head maps latent coordinates to feature coefficients; QR gives the exact physical field metric; residual energy is averaged over normalized training snapshots; correction directions are nested fixed linear coefficient vectors learned offline; CG/DST are iterative/direct Poisson controls.
