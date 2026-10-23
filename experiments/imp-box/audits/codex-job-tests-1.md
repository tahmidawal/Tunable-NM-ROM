No blocking findings at HEAD `b6497fc00b2b3716717504083334ba3e22a81f89`.

- **All nine cells pass:** committed freeze/index, source identity or committed equivalence, archive `frozen.json` equality, `controls.json` acceptance, artifact hashes, full archive checksums, and cluster-regenerated descriptor hashes.
- **Restore paths correct:** A3 uses `W40_A3t` / `W28_A3t`. No existing opening markers, test results, `.done_test` markers, or test job directories.
- **Test-once protection holds:** atomic `mkdir` admits one execution. A refusal preserves the marker and prevents reopening; downstream steps are gated on failure count.
- **Job specifications match:** all bodies reproduce the template and match RELEASE hashes; H200, 240G, 02:30:00. With `TIME=02:30:00`, auto-release uses the locked three-job lane cap and launches `wait_collect` under each `test_<cell>` name. Assembled scripts pass shell syntax checks.
- **Budget passes:** ledger 47.0636 + tests 22.5 + allowance 2.5 = **72.0636 ≤ 87 GPU-h**. Queue empty; archive filesystem has approximately 3,354 GiB free.

Read-only audit; no submissions or file changes.

**SUBMIT**