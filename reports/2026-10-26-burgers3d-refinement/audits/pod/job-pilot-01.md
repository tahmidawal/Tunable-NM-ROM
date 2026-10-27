# Independent exact pilot job review

Reviewer: `/root/independent_review`, read-only Codex agent.
Verdict: **READY for submission of the diagnostic pilot only**.

Independently verified all 77 source-manifest hashes and the archived configuration.
The numerical source matches the approved replay SHA256. Frozen package:

- source commit: `7559135c3`
- source tar: `b6646e0c5c0930ac47314b45243eec9825297691ec6cd0b0a7d1002a1f3558b3`
- batch: `f13e31a2b62ebcde2036b9bd2d54693fd0e7824cff205a280e41a71711dad95b`
- provenance: `306c77d3833b67f3903e6b399796ad5a2ff0534756a630e96006937a8ff6c5be`

The reviewer independently confirmed these same remote hashes, an empty account queue
and available paralab space. Batch enforces one H200 in the gpu partition, f64/highest,
absolute environment, backend assertion, isolated output/cache/tmp, source/artifact/
reference pins, one-hour hard allocation and the internal measured budget.

Submit once with immediate queue checks; record the numeric id and inspect partial
outputs before any restart. Full validation remains gated on measured pilot cost
and a results audit.
