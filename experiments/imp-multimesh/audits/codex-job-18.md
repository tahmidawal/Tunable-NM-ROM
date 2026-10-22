**SUBMIT** — the prior retry-certification blocker is closed.

- `retry_after_seconds=0` disables retries, consistent with DESIGN A11. Invalid panels remain UNRESOLVED.
- Per-configuration persistence and J5 array caching are correctly connected.
- Buffered timing estimate: **7,167.6 s < 9,000 s** timeout. Total estimate: **155.46 min < 215 min** allocation.
- All **157 manifest entries** verify; all **142 provenance files** match HEAD. Certificate values and config hash match.
- `COMMIT.txt` records `cba438ace`, the source commit preceding the restaging-only HEAD `cba0e96b`; staged source content matches HEAD.

No files changed, jobs submitted, large arrays loaded, or LAB-LOG/main writes.