READY

No blocking findings in `HEAD~1..HEAD` (`39ca0f7e4`).

- `analyze2d.py:146–167`: Project import closure covered. All 21 signature entries detect hash mutations and abort when missing.
- `analyze2d.py:171`: Job-field mutations rejected except the five explicit retry-packaging exemptions. Available accepted attempts pass.
- `analyze2d.py:36`: Current version unchanged and matches all 208 caches. Mutating either added dependency or any bound numerical global invalidates it. File pins match the setup commit.
- `resources.py:31`: All three smk2/smk3 slim `result.json` files match `runs/<job>/archive.sha256-listing`. Certificates reproduce: f2kA **11.2667**, b2f **1.9548**, f2kB **1.7404**, k3a **8.3966**, b3f **2.4419 hours**. Timing projection also works.
- `ledger.py:32`: Added mandatory jobs reject omissions and failed retries; live/completed retries satisfy coverage.
- **Nonblocking documentation discrepancy — `DESIGN.md:799`:** “committed sha256 listings” is inaccurate: both smoke listings are ignored and absent from HEAD. Local checksum verification passes.

Checks used the required interpreter with `JAX_PLATFORMS=cpu` and bytecode writes disabled. No file changes, submissions, array copies, or full analysis run.