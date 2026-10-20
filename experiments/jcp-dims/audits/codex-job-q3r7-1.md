**SUBMIT — staged job passes; wait for queue capacity.**

| Check | Result |
|---|---|
| Membership | Exact ordered match: **7 members** from `front3d.round_next`, using 436 candidates, qualify’s labels, and rounds 4–6. |
| Evidence handling | All predecessor members labelled; missing/blocked evidence remains untested. |
| Successors | All `qp`/`qpp` match lane ladders; 64-case cohort, eight-case successor checks, and control retained. |
| Staging | **176 manifest hashes verified; 172 provenance files** match commit `dcbee6720`. No source drift; config matches certificate. |
| Resources | Reproduced **0.527649 h** projection; required margin **0.791473 h**, covered by **1 h**. |
| Ledger | Pending **1.03 h** satisfies A10.1. Validation passes; reserving this job yields **51.6479/55 GPU-h** committed. |
| Execution | H200, GPU partition/preflight, f64/highest, isolated paralab path; shell syntax passes. Remote directory absent. |

Live queue is **3/3 lane jobs and 8/8 account jobs**. Submit through the guarded wrapper once capacity opens. Resource projection remains an estimate.

The cited round-5 audit actually records FIX-FIRST for its historical ledger allowance; this audit independently validates the current accounting.

No files changed, jobs submitted, or LAB-LOG writes.