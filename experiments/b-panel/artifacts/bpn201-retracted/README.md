# bpn201 (job 3783817) — RETRACTED attempt, 1024² panel

This attempt produced **no timing and no error number**. It crashed in the driver after the
GPU preflight, the six 4096-interval references, the 1024² snapshot matrix and all six rule
transfers, before any subject was built. `FAILURE.json` holds the verbatim traceback, the
scheduler record, what completed, and the remediation; `logs/` the verbatim stdout/stderr;
`output/result.partial.json` the driver's partial record and `output/rule_xfer_*.npz` the six
transferred rules it wrote (small; the 280 MB reference fields and the 350 MB FNO cohort were
not collected because they are regenerated from the seed by the resubmission). Every file's
remote SHA256 is in `REMOTE-SHA256.txt`; the local copies hash identically (`FAILURE.json`
`files`).

Nothing in this directory enters any table. The transferred-rule $\rho$ values it contains are
recorded because they were seen before the resubmission and bear on DESIGN §A5.2's prediction;
the numbers of record are those of the resubmission (`bpn202`).
