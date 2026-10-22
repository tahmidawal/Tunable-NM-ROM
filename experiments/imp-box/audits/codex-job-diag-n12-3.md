**SUBMIT.** No safety, budget, or identifiable execution blocker remains. Round-2 requirements are partly addressed, with nonblocking diagnostic limitations.

| Check | Finding |
|---|---|
| Staged executable | Both job files and all four checked dependencies match local SHA256 hashes. Python syntax and shell syntax pass. |
| Safety / budget | One A100, `gpu` partition, 30-minute limit: **≤0.5 GPU-h**. Correct absolute Python, GPU preflight, f64, highest precision. Outputs/cache under paralab; **3.3 TB free**. Completion within the allocation remains unverified. |
| Cache | Separate processes, explicitly disabled first pass, job-specific cache, before/after file counts implemented. **Unchanged file counts do not prove cache hits**; freshness is not asserted against requeues. |
| Production trace | Selects the first observed failing case/configuration and retains nested loops with coefficients computed inside `q`. Records finiteness of three stages plus update per step. **Coefficient and initial-state finiteness remain unreported**, and instrumentation can change compiled behavior. |
| Independent reference | Dense NumPy sine/cosine calculation correctly uses the same padded grid and projection. However, its input is **eagerly reconstructed**, not captured from the failing compiled execution; it tests the step input, not necessarily the failing internal stage. If instrumentation passes, `first_failing_step=0` is a fallback, not an observed failure. |

The reported smk2-A4 reproduction on another H200 with its own cache weakens a single-node or single-cache explanation. A passing A100 run remains **inconclusive about H200**, as accepted.

No files changed, jobs submitted, or LAB-LOG writes performed.