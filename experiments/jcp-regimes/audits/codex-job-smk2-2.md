**SUBMIT — audited HEAD `8f8c083fa`.** All three round-one findings are fixed.

1. **CORRECT — IC equality:** [smoke_accept.py:39](experiments/jcp-regimes/smoke_accept.py#L39), lines 131–150. Regeneration uses documented roundoff tolerance; saved-array comparisons remain exact. CPU replay passes R-upwind and all three calibration probes. Maximum regeneration discrepancy: **4.44e-16**; injected **1e-10** perturbations are rejected.

2. **CORRECT — stderr exception:** [smoke_accept.py:46](experiments/jcp-regimes/smoke_accept.py#L46), lines 82–93. Parsed dictionaries must differ only in `fingerprint`; remaining stderr is scanned. The original traceback passes. Added `RESOURCE_EXHAUSTED` before/after it, an additional mesh mismatch, and a second traceback all fail acceptance.

3. **CORRECT — time accounting:** [smoke_accept.py:298](experiments/jcp-regimes/smoke_accept.py#L298). Durations are calculated separately per job; timezone-aware SACCT timestamps use `astimezone(UTC)`. In-memory execution produced the expected **60 s / 120 s** intervals across separate jobs and rejected timezone-naive input.

Staging passes: **119/119 manifest entries** match staged bytes; **115/115 source entries** match HEAD blobs and provenance. Commit metadata matches HEAD. [run.sbatch](experiments/jcp-regimes/runs/smk2/run.sbatch) passes shell syntax validation and requests one H200 on `gpu` for two hours, with GPU preflight, f64/highest precision, isolated smk2 outputs, and UTC end marker.

This approves submission, not results acceptance; smk2 has not run. No files changed, scratch created, jobs submitted, or LAB-LOG writes performed.