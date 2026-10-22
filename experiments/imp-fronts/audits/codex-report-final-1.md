- **WRONG — Stale status:** `report.md:188` says Phase B awaits release and submission; S3 is completed. Remove/update this unconditional generator paragraph.
- **WRONG — EXPORT accounting:** `gpu_hours_phase_a = 6.327` incorrectly assigns the entire GPU ledger to CPU-only Phase A. Phase A is 0; total is 6.327.
- **CORRECT — Opening status:** finality is scoped to Phase A and Phase B S3; arms (a)/(c) remain deferred.
- **CORRECT — Numerical traceability:** Phase B error/floor tables recompute from per-case JSON; qualification counts and terminal counts match; exported scores and E9 match their sources.
- **CORRECT — Endpoint:** arm (b) STOP is supported: FC1 qualification is 0/8, and diagnostic medians 5.81% versus 8.12% give only 1.40×, below 2×.
- **CORRECT — Claim restraint:** the report does not claim coverage “helps” or establishes a qualified improvement.
- **CORRECT — E9:** thresholds, counts, rates and bounds match `e9.json`; insufficient populations and unmeasured fallback cost are disclosed.
- **CORRECT — E10:** activation after E8 STOP and design-only scope match `E10-DESIGN.md`; no execution result is claimed.
- **CORRECT — Budget:** ledger charges sum to 6.327 GPU-h; 8.673 remaining correctly rounds to 8.67.
- **NEEDS-RESTATEMENT — Deferral rationale:** distinguish the original S3-only release and outstanding architecture amendment from exhausted funding; substantial budget remains.
- **NEEDS-RESTATEMENT — Diagnostic labels:** label the entire Phase B error table diagnostic, including worst errors; explicitly state that unqualified runs contribute.
- **NEEDS-RESTATEMENT — Conditional labels:** the opening qualification is correct, but repeat “conditional” beside the Phase B floor table and E9 error targets.
- **NEEDS-RESTATEMENT — Glossary:** K1 and S3 are defined; add STEP-UNCONVERGED, diagnostic median, numerical qualification/Q1/Q2, FI/FN0s/FC1/FC2, K0/K2/K3, and E9 targets/AUC/detection/false alarm. Explain target A’s frozen-threshold origin.
- **NEEDS-RESTATEMENT — Verdict table:** raw dictionaries leave units and “wins” unclear; identify these as diagnostic error comparisons, and distinguish unavailable K2/K3 ROM comparisons from failures.
- **CORRECT — Plots:** inspected PNGs agree with plotted quantities; Phase B explicitly labels diagnostic medians and conditional references.

Read-only throughout: no files changed, jobs launched, or scratch created; LAB-LOG and main untouched.