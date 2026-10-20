**Verdict: FIX-FIRST.** The three staged jobs pass integrity checks, but A5.2’s incomplete-smoke rejection is not fully implemented.

| Round-1 item | Round-2 assessment |
|---|---|
| Claimed source revision | **RESOLVED.** All three attempts identify current HEAD `52951dc93ef31507ece4e19ac9839e08831d0fb2`; staged source bytes match its Git blobs. |
| Compilation excluded from accuracy projections | **RESOLVED.** Per-scheme elapsed measurements surround the first execution, certification, replays and archive writing. |
| Extra-block costs asserted rather than measured | **RESOLVED.** Controls, order, vendor and P1e blocks now have measured elapsed times. Production scaling remains an estimate. |
| Incomplete pilot could produce a certificate | **RESOLVED.** Pilot completion and positive, finite matched timings are required; the fallback factor is removed. |
| “Upper proxies” | **RESOLVED.** A5.3 explicitly calls projections estimates with headroom, not bounds. |
| Pilot peak guarantees | **RESOLVED as a specification.** A5.2 defines OOM/timeout as feasibility unestablished, followed by a prefix-bank attempt and an explicit incomplete outcome if that fails. |
| “Every DESIGN 6 control fully demonstrated” | **RESOLVED.** Preliminary dev2 reproduction and the descriptive two-mesh timing probe are distinguished from the full-population gate and synthetic three-mesh control. |
| Completion confidence / bounded-attempt disposition | **PARTIALLY RESOLVED.** Split jobs and bounded retries are documented; completion within their allocations remains unmeasured. The promised incomplete-smoke rejection has the gap below. |

**Remaining defect:** [resources.py:80](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/resources.py:80) only locates the accuracy result initially; subsequent branches validate completion solely for their selected driver. Consequently:

- An accuracy certificate can be generated with complete `smkA` and pilot while `smkB` is missing or incomplete.
- A timing/kernel certificate can be generated while the accuracy result exists with `complete=false`.

This contradicts A5.2’s blanket refusal of certificates from incomplete smoke runs. Require all four smoke results to exist and declare completion before dispatching any projection.

**Staging verification:** Each attempt has **127 verified manifest entries, 123 provenance records matching HEAD, and 58 successfully parsed Python files**; no unmanifested files were found. All three batch scripts pass shell syntax checks. GPU constraints, allocations, commands, output paths and reservations match the requested **smkA 2.5 h / smkB 2 h / pilot 2.5 h**. Absolute venv, f64/highest precision, GPU preflight and isolated paralab directories are present. Driver dependencies and required assets are staged; offline acceptance retains its documented local dependencies and now accepts both archive roots.

No additional staging defect found. Verification was static; no files changed or jobs submitted.