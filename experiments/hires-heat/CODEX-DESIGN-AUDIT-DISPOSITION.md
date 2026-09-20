# Disposition of the Codex design audit (2026-09-20, before the first GPU job)

First attempt failed: the Codex sandbox could not start a shell (`bwrap ... Operation not permitted`, kept as `CODEX-DESIGN-AUDIT-attempt1-sandbox-failure.txt`). Second attempt pasted DESIGN.md, core.py, run.py and the 2D config on stdin; 15 findings in `CODEX-DESIGN-AUDIT.txt`.

| # | finding | disposition |
|---|---|---|
| 1 | row compression changes LM scaling/stopping | FIXED: LM takes the uncompressed projected-target norm explicitly; parity arm in job 3 |
| 2 | direct arm reported first-start stats | FIXED: selected-start statistics returned |
| 3 | failures only saved, not gated | FIXED in `summarize.py`: failing arms counted, failing FOMs excluded as comparators |
| 4 | CG recurrence drift could disqualify the FOM | FIXED: restart from the true residual (<= 3, charged to the FOM) |
| 5 | burn-in once per sweep; error reductions inside timing sweep | FIXED: untimed evaluation pass first; burn before every timed query |
| 6 | profile unsynchronised/no burn | FIXED |
| 7 | cohort shrinks with mesh | FIXED: all 12 (2D) / 16 (3D) cases at every mesh |
| 8-10 | ladder/limit gates not enforced; q=R-K is not an identity | ACCEPTED: DESIGN wording is 'empirical diagnostic'; ladder monotonicity and parity are reported per job by the summariser, not asserted |
| 11 | linear-bank baseline lacked a matching direct arm | FIXED: `linear_bank_*_direct_BASELINE` fits free coefficients to the same propagated moments |
| 12 | DST control vs truth tautological | ADDRESSED: `audit.py` recomputes every reference with `scipy.fft.dstn` independently |
| 13 | strided audit can miss inter-node error | ADDRESSED: exact sub-grid recomputation + 100k independently seeded random nodes on two cases per mesh + exact full-field audit where N <= 3e5; limitation stated in audit.json |
| 14 | conditional gates | PARTLY: repetitions >= 5 asserted; skipped gates visible as missing keys |
| 15 | dimension validation | FIXED: q and row-count asserts |
