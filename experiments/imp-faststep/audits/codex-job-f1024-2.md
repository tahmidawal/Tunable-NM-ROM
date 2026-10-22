**DO NOT SUBMIT.**

- Timeout repair passes: projected 7,108.77 s fits the 7,860 s driver timeout.
- Budget admission passes: 0.63 + 2.25 = 2.88 GPU-h < 4. Staged manifest checksums pass.
- One output-production blocker remains: classification refinements discard evidence needed for later validation.
- Staged `fsrun.py:453–475` retains only growth maxima, `refined_ok`, and the label; refinement states, outputs and certificates are not persisted.
- Staged `podarm.py:218–228` likewise discards refinement arrays, and its validity predicate omits stage finiteness.
- Consequently, the results audit cannot reconstruct the complete refinement-validity gate required by A4.1 from these outputs; A5.4 requires per-run evidence.
- Persist refinement evidence and apply complete validity checks before restaging. This is separate from the deferred offline-checker fixes.
- No files changed, SSH used, or jobs submitted.