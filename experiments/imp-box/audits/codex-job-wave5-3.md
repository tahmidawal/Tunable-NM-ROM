Round 2’s collection blocker is closed.

- Both restart bodies verify the restored archive **before** removing inherited `_job` and `SHA256SUMS`. Cleanup is chained with `&&`, so failed verification prevents it.
- Both committed body hashes match `cluster/RELEASE.json`: W40 `b4ff432319d5…`, W28 `6de778970939d…`.
- Read-only SSH confirms both source archives have the required restore inputs and both destination archives, `W40_A3t` / `W28_A3t`, are absent.
- Under those fresh names, `collect.sh`’s cell and job-level inventory comparisons no longer encounter inherited logs. Collection should verify, conditional on successful execution and transfers; it was not run.

No files changed, lab log appended, or jobs submitted.

**SUBMIT**