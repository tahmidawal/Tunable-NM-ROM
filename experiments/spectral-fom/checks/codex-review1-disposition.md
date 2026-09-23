# Disposition: codex review 1 (2026-09-23)

The review is `checks/codex-review1.md`. It was an independent, read-only codex exec run. It found no
blockers. It recomputed two summary rows from the raw invocation records (Burgers 512² accurate, heat 1024²
accurate) and both matched.

1. **Major: the audit coverage for Burgers and heat is partial.** Accepted.
   - The fields have already been deleted on the cluster, so the audit cannot be widened after the fact.
   - The report now states the audit scope explicitly, in the section "Audit scope and review".
   - The Burgers truth is the paper's own FOM, as registered.
2. **Major: the Burgers DST-variant micro-benchmark has no burn-in and a fixed order.** Accepted and disclosed
   in the report.
   - The choice decides only which implementation of the same operator the spectral side uses. A wrong choice
     can only make the spectral solver slower. That biases the comparison in favour of NM-ROM, never against it.
   - The chosen variants were mm at 256² and fft at every other mesh. At 2048², fft was 72 ms per sweep
     trajectory against 93 ms for half.
3. **Minor: `check_timing.py` does not recompute T1.** Fixed. T1 is now recomputed independently for every run
   and compared with the recorded value or sidecar. Everything agrees.
4. **Minor: `check_timing.py` does not check the inventory.** Fixed. The check now requires each
   subject × phase × case × repetition exactly once, with no duplicates and no missing cases. It passes on
   every run.
5. **Minor: Burgers lane-parity coverage is not guaranteed.** Fixed offline in `check_timing.py`. It now
   requires both selected arms, the same case count as the lane, and every case within 1e-8 relative. It passes
   for every Burgers run.
