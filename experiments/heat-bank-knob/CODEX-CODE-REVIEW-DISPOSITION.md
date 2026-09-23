# Disposition of CODEX-CODE-REVIEW.md (codex, static read-only, 2026-09-23, after jobs h2d/h3d were submitted)

Codex found the model algebra, prefix-only bank reads, linear-rung maps, randomisation/phases/burn-in and the
validation-only selection consistent. Its findings concern acceptance enforcement and audit completeness:

1. Gates do not control publication — FIXED in reporting: `gates.usable` (complete ∧ determinism ∧ neighbour ∧ audit ∧ parity);
   every Table-1 row in `report.json` carries `usable` (also false if the arm had solver failures).
2. Audit coverage — FIXED (audit v2): every row method must have a saved field for every case (`coverage_missing`).
   Full-grid errors on large meshes are checked only by random-node samples on the first two cases; the sub-grid gap is
   reported as a diagnostic. This is a restricted audit and is stated as such.
3. Non-finite values — FIXED (audit v2).
4. Neighbour sub-ms rule used a one-sided difference — FIXED (`abs`); gate now also requires every model arm to be covered.
5. Knob timing condition missing — FIXED (`*_linear_time_drop`), ladder completeness and failures reported.
6. Neighbour outputs not fingerprinted — NOT fixable for the submitted jobs (in-job code); disclosed. Main-phase outputs are fingerprinted.
7. Controls global — FIXED: both controls required per mesh.

The in-job audit of h2d/h3d is v1; audit v2 is re-run locally on the pulled fields and is the one that counts.
