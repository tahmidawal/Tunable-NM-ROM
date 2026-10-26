# Independent k32 training-only fitting implementation review

Reviewer: `/root/independent_review`, read-only Codex agent.
Verdict: **READY for exact job staging**, numerical source reviewed at `bd48ef820`.

The wrapper calls exactly one inherited NNLS path on the frozen training-only
construction. An interrupted refit or final-cleanup target record cannot be admitted:
normal support completion or an explicitly certified earlier global KKT optimum is
required. Weights must be positive, QR quality must pass, and observed memory must
be positive and below the registered cap. Original training descriptors and matrix/
candidate hashes are saved before fitting. Failed terminal weights are retained.

Tiny real NNLS target and early-KKT controls and NumPy-boolean serialization control
pass. No numerical job is authorized until the independent exact package/config/
batch gate. No validation-driven fitting or retry is permitted.
