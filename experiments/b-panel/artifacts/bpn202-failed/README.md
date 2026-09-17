# bpn202 (job 3787247) — FAILED attempt, 1024² panel

This attempt produced **no timing and no error number**. It died 28 min 37 s in, on the H200 `pax011`,
inside the *untimed* best-found reconstruction diagnostic at q = 256: XLA's autotuner could not
allocate 17–32 GiB for the vmapped eight-start Levenberg–Marquardt batch with all 29 subjects already
built and resident. `FAILURE.json` holds the verbatim traceback, the scheduler record, what completed
(preflight, six 4096-interval references, 3328 snapshots, all six rule transfers, all 29 subject
builds, the q = 0 and q = 64 reconstructions) and the remediation; `logs/` the verbatim stdout/stderr;
`output/result.partial.json` the driver's partial record; `output/rule_xfer_*.npz` the six transferred
rules it wrote (small; the 280 MB reference fields are regenerated from the seed by any
resubmission). Every file's remote SHA256 is in `REMOTE-SHA256.txt`; the local copies hash
identically (`FAILURE.json` → `files`).

Nothing in this directory enters any table. The transferred-rule $\rho$ values it contains are the
values of record for the `clip(8192/M, 8, 64)` fit-state convention that DESIGN §A7 retires; a
resubmission recomputes them with 64 states at every rung.
