**Overall: FIX-FIRST — correct the pending ledger allowance, then wait for queue capacity.** The staged computation passes audit.

| Item | Verdict | Evidence |
|---|---|---|
| Round-5 selection | PASS | Exact ordered equality with `front3d.round_next`: 13 members from 436 candidates, using qualify’s 55 labels plus round 4’s 14 failures and two unverified labels. |
| Missing/blocked evidence | PASS | Generator leaves these labels untested. All 16 predecessor members have completed labels. |
| Successors | PASS | Every `q+`/`q++` matches the registered ladders, including gl64→gl80 and lat65536→lat131072→lat262144. |
| Settings and rules | PASS | Required settings, rules, control, 64-case cohort, and eight-case `q++` checks are present. |
| Staging/provenance | PASS | All 165 source files match commit `4268e40f8ee2fe04c1a8a02fe9bbbcced813687c`; all 169 manifest hashes verify. |
| Sbatch/namespace | PASS | H200, 2 hours, GPU partition/preflight, f64, highest precision, isolated paralab directory, output checksums. Remote attempt directory absent. |
| Resource certificate | PASS | Independently reproduced **0.8886305 h** projection; config hash matches. Two hours exceeds the required margin. Smoke peak **52.646 GB**, below 120 GB. Projection remains an estimate; `q++` and control work are not separately itemized. |
| Ledger | **FIX-FIRST** | Pending q3r5 allowance is **1 h**, below the **1.5 h** certificate; `ledger.validate` explicitly rejects it. Update to the staged **2 h** allowance. |
| Concurrency | **WAIT** | Live queue has **3/3 lane jobs and 8/8 account jobs**. Submit only through the cap-enforcing wrapper after capacity opens. |

With q3r5 charged at two hours, spent + reserved + remaining pending core totals **46.7356/55 GPU-h**, leaving **8.2644 h**. The reservation path correctly substitutes two hours, but the current pending inventory is inconsistent.

No files changed, jobs submitted, or large copies created.