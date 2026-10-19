**NEEDS-FIX at `807090349`: one remaining smoke blocker.** Deferred production work does not drive this verdict.

Read-only audit completed. No files changed, jobs submitted, scratch files created, or lab-log writes made. CPU unit gates passed; 26 Python files parsed; Bash syntax passed. The actual wrong-mesh Python control passed with its output intercepted in memory.

Paths below are relative to [experiments/jcp-regimes](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes).

| Audit 4 blocker | Verdict | Evidence |
|---|---|---|
| 1. Write-boundary escapes | **NEEDS-FIX** | Symlinked `checks/` and `runs/` roots are refused (`guard.py:42–67`); wrong-mesh deletion now follows validation (`smoke_accept.py:218–220`); staging and ledger guards exist. The ledger temporary-file bypass remains below. |
| 2. Budget-zero control identities | **CORRECT** | Canonical viscosities, ordered case IDs, form/scheme, dimensions, rule and mesh are checked; operator/mode arrays must equal the production R384 arm (`smoke_accept.py:187–196`). Independent gamma recomputation covers all six cases (`202–209`). |
| 3. Fresh reference checkpoint before probe | **CORRECT** | Immediately after `write_case()`, the merged manifest is persisted before invoking any probe (`refjob.py:413–419`). Resumed cases also checkpoint first (`393–397`), and the initial resumed report preserves the previous manifest (`667–672`). |
| 4. Explicit refusal strands reservation | **CORRECT** | The ERR trap is removed before submission; statuses 75/76 release the pending reservation before the ambiguity branch (`cluster/submit.sh:29–42`). `set +e` no longer leaves the old trap active. |

The remaining smoke blocker is:

1. **The ledger writes an unchecked temporary destination.**

   [`ledger.py:29`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/ledger.py:29) validates `GPU-LEDGER.json` and `GPU-LEDGER.lock`, but lines **39–41** write `GPU-LEDGER.tmp` without validating that exact path.

   A pre-existing `runs/GPU-LEDGER.tmp` symlink to an external file therefore causes `tmp.write_text()` to truncate/write that external target. The subsequent rename does not undo that write.

   An intercepted execution of the actual `ledger.locked()` confirmed **two checked destinations and one unchecked temporary-file write**. The guard itself correctly rejects the simulated external temporary destination; the caller never invokes it.

   **Required fix:** call `guard.check_admin(tmp, HERE)` before writing the temporary file. This affects reservation handling on the smoke submission path.

Other requested checks found no additional smoke blocker:

- **Probes:** independent expected counts are 202/402/802; finite lattice shapes, canonical initial fields and selected production-attempt arrays are checked (`smoke_accept.py:89–124`). Calibration disagreement/invalidity remains non-blocking as specified.
- **Full-field verification:** valid non-finite comparisons fail; exact inventory, R-file hashes, field hashes and verification-config hash bind the pulled certificate (`fverify.py:40–67`; `smoke_accept.py:228–235`). The configured inventory contains exactly nine arms.
- **Qualification:** population/form/step expectations come through `arm_expect()` and are checked by `qualify()` (`rscore.py:220–235,482–488`; `smoke_accept.py:249–255`). GAL budget-zero remains diagnostic.
- **Wrong mesh:** the actual control accepted its own table and rejected both the mesh mismatch and perturbed eigenvalues. I inspected `jaxslot`/`jaxrun` without executing their lock/systemd writes.
- **Staging and ordering:** H200 restriction, committed-blob staging, GPU preflight, unit-gates-first ordering and matching output names remain intact. `fverify` exit **3** under `set -euo pipefail` stops subsequent steps and prevents `ALL-DONE`—correct fail-closed behavior.
- **Real-size memory:** six interior f64 fields at L=12288 occupy **7.247 GB**, before solver vectors, padded outputs, FFT workspace and cached executables (`refcore.py:164–189,230–233`; `refjob.py:113–120`). I found no demonstrated inevitable H200 OOM. Peak memory and two-hour completion remain smoke measurements.

A6’s production deferral is appropriately scoped. Its addendum covers evidence-bound verification, verification cost and analysis/export qualifications. The earlier calibration-writer concern is unreachable because `mode_calib()` immediately refuses execution (`refjob.py:446–447`); it is not a current blocker.