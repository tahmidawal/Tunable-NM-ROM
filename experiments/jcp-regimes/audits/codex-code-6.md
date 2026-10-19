**READY-FOR-SMOKE at `0597bed5d`. No remaining smoke blockers found.**

Read-only audit completed: no file changes, jobs submitted, scratch files created, or LAB-LOG writes. CPU unit gates passed; 26 Python files parsed; submission Bash syntax passed.

Paths below are relative to [experiments/jcp-regimes](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes).

| Check | Verdict | Evidence |
|---|---|---|
| Audit 5: ledger temporary-file escape | **CORRECT** | `ledger.py:39–44` checks the exact temporary destination and explicitly rejects symlinks before writing. An intercepted execution confirmed that refusal occurs before the write. |
| Smoke write boundaries | **CORRECT** | Resolved-path and administrative/local guards: `guard.py:20–67`. Reference files and temporaries: `refjob.py:256–268,622–625`; slimming: `refslim.py:43–62`; ROM files: `qacc.py:38–40,103–115`; verification: `fverify.py:30,65`. Wrong-mesh deletion follows validation at `smoke_accept.py:218–220`. Staging creates a fresh, guarded directory (`cluster/stage.py:65–67`) and guards copied destinations (`81–84`). |
| Reference resume | **CORRECT** | Configuration refusal precedes result replacement (`refjob.py:656–660`); the initial report preserves the previous manifest (`667–672`). Both resumed and newly completed cases checkpoint before probes (`393–399,413–421`). |
| Probe completeness and identity | **CORRECT** | Independent expected stage counts are **202/402/802** (`smoke_accept.py:24–27,89–94`). Canonical case/t₀ checks cover upwind members (`95–101`); calibration additionally binds production arrays to the selected attempt and checks both initial slices (`112–124`). Calibration disagreement/invalidity remains non-blocking as specified. |
| Full-field verification | **CORRECT** | Valid non-finite runs fail; shapes, hashes and arm inventory are checked (`fverify.py:40–67`). Acceptance binds the nine expected arms to pulled R-file hashes, field hashes and verification-config hash (`smoke_accept.py:228–235`). F-file download is unnecessary. |
| Qualification expectations | **CORRECT** | Expectations come from arm configuration (`rscore.py:482–488`; `smoke_accept.py:249–255`), with population, shape, form and step checks at `rscore.py:220–235`. |
| Controls | **CORRECT** | Registered populations and under-resolved dimensions are checked (`smoke_accept.py:175–183`). Budget-zero dimensions, viscosities, scheme and operator identity are checked, followed by independent LSPG gamma recomputation for all six cases (`187–213`). GAL remains diagnostic. |
| Wrong-mesh re-execution | **CORRECT** | Actual CPU execution accepted its own table and rejected both the 128→256 mismatch and perturbed eigenvalues. Output was intercepted in memory. The `jaxslot`/`jaxrun` command and CPU environment are consistent (`smoke_accept.py:221–227`); wrappers were inspected, not executed. |
| Stage, submit and ledger | **CORRECT** | H200-only staging, committed blobs, GPU preflight and unit-gates-first ordering remain intact (`cluster/stage.py:24,75–84,105–126`). Reservation precedes submission; known refusals release it before ambiguity handling (`cluster/submit.sh:24–42`). |

The specified step order and output filenames agree. **`fverify` exit 3 correctly stops the batch under `set -euo pipefail`**, preventing references and `ALL-DONE` after failed verification.

At **L=12288**, the six interior f64 output fields alone occupy approximately **7.247 GB** (`refcore.py:164–189`). Padding, solver vectors, FFT workspace and stiffness temporaries add to that (`refcore.py:230–233`; `refjob.py:113–120`). I found no demonstrated inevitable H200 OOM; peak memory and two-hour completion remain unverified smoke outcomes.

A6’s production deferral is appropriately scoped. Its addendum covers the previously missing evidence-bound verification, measured verification cost, and analysis/export qualifications. I found no additional production-gate omission that should block this smoke.