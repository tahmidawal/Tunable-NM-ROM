**fom1k: NOT CLEAN.**  
**b3d65: NOT CLEAN.**

Reviewed HEAD `6b000d45d`. No files modified, lab-log entry appended, or jobs submitted. Tests used syntax parsing, in-memory serialization, extracted audit functions, and a synthetic execution of the complete 3D validator. No production GPU workload or full-size least-squares benchmark was run.

| Finding | Disposition | Round-4 assessment |
|---|---|---|
| **2** | **RESOLVED** | Calibration now saves F64 fields and per-step statistics, reconstructs verification, and applies the threshold to reconstructed discrepancies. Diagnostic completeness remains under finding 6. |
| **4** | **STILL-WRONG** | All **3,040** matches are now required, and designated paths, accepted audits, mesh and rule are checked. But the audit certificate is not bound to the result bytes: unchanged job ID permits an altered result or stale certificate. |
| **5** | **RESOLVED** | Order statistics are retained; expected step counts, nonlinear residuals and zero failures are reconstructed for every required trajectory. Remaining diagnostic validation deficiencies are under 6. |
| **6** | **STILL-WRONG** | Empty-step bypass is closed, but linear diagnostics and aggregate consistency remain unchecked. ROM verification still admits contradictory solver statistics. |
| **8** | **RESOLVED** | Evaluation and timing checkpoints address the original complaint. Calibration still saves fields only at phase end; there is no resume or transactional checkpoint. Six hours remains unvalidated. |
| **9** | **STILL-WRONG** | The exemption remains acceptable, but DESIGN A11 still says “not a coding error” and asserts ringing. Those causal claims were not established by the cited observations. This is a wording issue, not independently a submission blocker. |
| **11** | **RESOLVED** | Reference mesh, timestep and trajectory shape are now asserted. The existing initial-field check supports the output-layout contract. |
| **13** | **STILL-WRONG** | Vendor coefficients and principal distances are now reconstructed. Vendor solver-health evidence and per-time errors are still insufficiently validated; fixed-sweep eligibility needs an explicit definition. |
| **14** | **RESOLVED** | Timing hashes now bind to saved generic/vendor coefficients; drift and baseline identity are checked. |
| **15** | **STILL-WRONG** | Generic per-time errors, initial projection, model/rule hashes and manufactured certificate are checked. Solver acceptance remains incomplete, and the new projection check has substantial repeated computational cost. |
| **16** | **DEFERRED-TO-REPORT** | As instructed: `make_report` must reconstruct anchor eligibility/uncertainty, mask unresolved sensitivities, and apply **13-of-16** and **8-of-16** thresholds. This is not a pre-submission blocker. |
| **17** | **RESOLVED** | Coefficients now checkpoint after each case and timings every 50 triples. The six-hour reservation is still a planning estimate, not a demonstrated runtime. |
| **18** | **RESOLVED** | Reading the unrelated 2D reference manifest is now confined to the 2D staging branch. |

The remaining acceptance failures are concrete:

- **FOM diagnostics:** [`fom_verified`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/audit_fom.py:43) returned `(True, True)` for otherwise valid nonlinear residuals with `step_lres=[]`, negative iteration aggregates, an inconsistent `first_fail`, and contradictory worst-residual values. Validate all active diagnostic arrays, their padding, integer bounds and reconstructed aggregates. This does **not** mean linear tolerance must become a new acceptance criterion.
- **ROM verification:** both `rom_consistent` implementations accept impossible histograms and contradictory statistics. An all-reason-4 LSPG trajectory with `worst_ratio=1e100` passes, although reason 4 means stationarity tolerance was satisfied. Aggregate maxima also cannot establish the per-step stationarity-or-residual condition for mixed exits; retain sufficient per-step evidence.
- **Vendor evidence:** the complete [`audit_t3.validate`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/audit_t3.py:120), exercised with the configured inventories and small synthetic matrices, passed every check after replacing vendor reasons with **25 copies of 99**, `gn_per_step` and `e_ref_per_time` with NaNs, and `it_sum` with −999. Validate reason codes, diagnostic lengths/finiteness, histogram consistency and all reported errors. Define eligibility without assuming fixed-sweep outputs satisfy adaptive convergence.
- **Cross-job authentication:** the actual loader predicate still accepted an in-memory `a1kfast` result after changing both a row’s error and its commit. Bind the accepted audit to a digest of the exact result artifact, then verify that binding when loading.

For the specific crash/resource concerns:

- **FOM JSON:** a synthetic inventory of **384 calibration**, **40 order**, and **608 evaluation** diagnostic records serialized through the actual cleaner to **26,430,336 bytes**, taking about **1.3 seconds** locally. It parsed successfully. This excludes other result sections, but provides no evidence of a JSON-size crash.
- **Tuple/lambda timing:** the `block_until_ready(o)` → `Wf()` pattern works; the lambda closes over the call-local output. No tuple-related failure found.
- **`W_vendor`:** the vendor returns 26 internal coefficient states; `[::5]` produces the required six states. Saving after each case is consistent.
- **Large `lstsq`:** [`audit_t3.py:82`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/audit_t3.py:82) repeats factorization **288 times** across the initial validation and eight mutations. The 512-column matrix alone is **0.954 GiB**; each decoded case adds **1.475 GiB**, before copies and LAPACK workspace. This is a serious avoidable audit-time cost, not a demonstrated OOM. Solve all initial-field right-hand sides together once per arm and cache those immutable projections across mutations.
- **Unaudited `a1kacc`:** its archive and audit are currently absent. The loader skips it without crashing, and complete cross-job agreement consequently fails. That is correct incomplete-evidence behavior; absence alone need not prevent collecting `fom1k`.

The holds are for incomplete evidence validation and artifact authentication, not a demonstrated integrator, tuple-handling, or JSON serialization crash.