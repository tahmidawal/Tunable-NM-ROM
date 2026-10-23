**Overall: READY** at `261d70059`. No compile-blocking or correctness defects found in the reviewed fixes.

- **TRANS:** Captured the callable from `diag.comps_reduced`. StableHLO exposes the basis as an argument; JAXPR has no captured constants. Output matched `make_shift_decoder` and an independent formula exactly.
- **G-HOIST:** Production comparisons and mutation fixtures call `controls.hoist_cmp`. Field NaNs, statistics NaNs, and finite statistics mismatches were rejected.
- **Nonblocking validation weakness:** `controls.py:182–184` combines mutation results using `max`, allowing one detected fault to mask another missed fault. Check each mutation independently.

Checks were small, CPU-only, f64, highest precision. Production GPU compilation was not tested. No files or lab log were modified.