# Integration verification notes

The records in `checks/` describe consolidation checks, not new scientific results.
The accepted campaign reports and their interpretation are unchanged.

The first import attempt looked for the Poisson saved field at the run root. Its
restored location is `cluster/out/pilot/fields/`; the importer was corrected and
completed. No source artifact was changed.

The initial heat replay wrapper used a bytes-only hash to check an archived input.
Heat's native `field_hash` includes shape and dtype as well as bytes. The wrapper
now uses that native hash and the input check passes.

The initial Poisson replay required byte-identical regeneration of a full source
array that was never archived. The local source hash differs from the cluster's
recorded hash even though the original function and recorded parameters are used.
Different platform transcendental rounding is a possible explanation, not an
independently established attribution. The replay retains both hashes and checks
the computed field against the accepted archived output. It does not assert
bitwise identity of the supplied source.

All successful replays use the original numerical routines, unchanged acceptance
criteria, double precision and highest matrix multiplication precision. Their
field parity threshold was declared before any successful result. Exact report
regeneration uses only the copied evidence and leaves the snapshots untouched.

The complete staged whitespace check flags pre-existing whitespace in imported
`fresh_rom.py`, `accuracy_coverage_paths.py`, `test_audit_dynamics.py` and the archived
`accel12-close-preview.log`. Their bytes agree with the accepted source commits and
are preserved, including the frozen wave mathematics hash. The rest of the staged
files pass the whitespace check. Exact paths and source hashes are retained in
`checks/integration-review.json`.
