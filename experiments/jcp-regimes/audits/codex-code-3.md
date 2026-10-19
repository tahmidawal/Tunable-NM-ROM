**NEEDS-FIX for smoke at `f4e5cf27d`.** The deferred production work does not drive this verdict.

I made no file changes, submitted no jobs, and wrote no lab-log or scratch files. CPU unit gates passed. The actual wrong-mesh Python control passed with its output write intercepted. A mocked execution of `fverify.main()` correctly rejected a non-finite valid field with exit status **3**.

Paths below are relative to [experiments/jcp-regimes](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes).

**Audit 2’s nine required fixes**

| Fix | Verdict | Evidence |
|---|---|---|
| 1. Resolved read/write boundaries | **NEEDS-FIX** | Exact-file checks protect the four primary drivers, but directory creation precedes descendant validation (`refslim.py:51`, `refjob.py:267`). Slimming sources receive only lexical validation (`guard.py:31`, `refslim.py:41`). Additional smoke writers remain unguarded (`test_regimes.py:159`, `ctrl_wrongmesh.py:37`, `smoke_accept.py:227`). |
| 2. Resume validation before overwrite | **CORRECT** for the requested configuration-refusal fix | `refjob.py:640–644` now rejects incompatible smoke configurations before overwriting the result. A separate interruption-safety defect remains below. |
| 3. Complete, independently validated probes | **NEEDS-FIX** | Expected counts are correctly reconstructed as **202/402/802** (`smoke_accept.py:24–27`, `80–85`). Calibration binds production arrays to the selected attempt (`99–101`), but does not establish canonical case identity or shared tight-run initial data (`103–108`). |
| 4. Full-field verification | **NEEDS-FIX** | Non-finite rejection, shape checks, arm inventory and failure exit are corrected (`fverify.py:40–66`). The pulled verification result is still not bound to the pulled R files (`smoke_accept.py:196–197`). Production-analysis enforcement is legitimately deferred. |
| 5. Independent qualification expectations | **CORRECT** for smoke implementation | Form, populations and steps now come from configuration through `rscore.py:482–488`, called at `smoke_accept.py:213–216`; checks occur at `rscore.py:220–235`. The requested dedicated regression fixtures are still absent from `test_regimes.py`. |
| 6. Complete control evidence | **NEEDS-FIX** | Six identifiers are checked, but budget-zero evidence dimensions are not (`smoke_accept.py:160–180`). Wrong-mesh re-execution is implemented and its Python path passed. GAL’s diagnostic-only treatment matches A6. |
| 7. Missing sigma attempts and analysis completeness | **CORRECT for smoke; production portion deferred** | Missing finest attempts now return unavailable with a reason (`rscore.py:128–135`); missing middle attempts no longer crash. Empty baselines, arm inventory and reduced-C2 analysis remain within A6’s production gate. |
| 8. Resource plan and costing | **NEEDS-FIX — explicitly deferred, non-blocking here** | `fverify` now has a provisional fixed allowance (`resources.py:55–57`). Complete workload accounting, reductions and derived remaining reservations remain unfinished (`cluster/stage.py:44–48`). |
| 9. Calibration/separation propagation and reporting | **NEEDS-FIX — explicitly deferred, non-blocking here** | Production configuration and reporting still need the registered conditional behavior and qualifications (`make_configs.py:99–103`, `analyze.py:150–195`, `make_report.py:78–129`). |

**Smoke blockers**

1. **Budget-zero acceptance can pass incomplete evidence.**

   `smoke_accept.py:160–162` validates six case IDs but never requires six certificate/state trajectories. Lines `170–180` pool whatever certificates exist and independently recompute only case 0.

   A CPU counterexample using six registered IDs and a certificate array of shape **`(1, 50)`** passed both the population predicate and the budget-zero acceptance predicate. Five cases’ evidence can therefore disappear without rejection.

   Require the independently expected dimensions—six cases, 50 certificates and 51 states per case—and validate the associated normalization/operator/schedule metadata before computing the pooled result.

2. **Probe validation can report agreement for the wrong initial data.**

   Calibration compares outputs only at indices 1–5. It checks that `tight_lat` is finite and correctly shaped, but never checks `tight_lat[0]` against canonical `t0` (`smoke_accept.py:103–108`). An in-memory fixture with identical later fields and **twice the initial field** returned **`valid=True`, `verdict='agree'`**.

   The production-array equality checks are useful, but the containing case record and probe `t0` are not checked against the requested cohort/case generator. Upwind likewise checks internal shared `t0`, without establishing canonical identity (`76–88`). The stronger identity checks in `rscore.reference()` are not invoked for these regime pilots.

   Validate case metadata, canonical `t0`, and every probe member’s initial slice. Keep genuine calibration disagreement/invalidity non-blocking under A2.2; the blocker is falsely reporting agreement.

3. **The pulled full-field certificate is not tied to the evidence being accepted.**

   `fverify.py:57–63` records outcomes and counts, but no R-file digest or verified field digest. `smoke_accept.py:197` accepts `all_pass` plus **nine entries**, without checking exact arm keys or their binding to the current R files.

   Consequently, a successful `fverify.json` from another execution can certify the current output. Because F files are intentionally not pulled, this leaves no complete offline check connecting the full-field verification to the accepted coefficients.

   Persist and verify per-arm R-file hashes, the checked field hash, exact inventory and configuration identity. No F-file download is needed.

4. **The promised filesystem boundary still has smoke-path bypasses.**

   Primary file writes now resolve descendant symlinks correctly. However:

   - `refslim.py:51` and `refjob.py:267` can create directories through a descendant symlink **before** the eventual file check refuses the destination.
   - `refslim.py:41–48` validates source spelling and apparent cohort names, but not resolved source destinations. A symlink can redirect a permitted-looking source/cohort.
   - `test_regimes.py:159` and `ctrl_wrongmesh.py:37` accept arbitrary output filenames; local acceptance also writes an unchecked fixed destination at `smoke_accept.py:227`.

   Validate destinations before directory creation and protect the auxiliary writers. Local acceptance outputs need an explicit permitted checks directory; administrative staging/ledger outputs need their own appropriately scoped policy.

5. **A second interruption during resume loses the verified manifest.**

   The incompatible-configuration refusal is fixed, but same-configuration resume still overwrites `result.json` with a fresh report containing no manifest (`refjob.py:651–654`, `682`). Verified entries are reconstructed afterward, and resumed probes run before their progress is durably saved (`380–389`).

   If that resumed process is interrupted during a probe, the next restart finds no persisted old manifest. Previously verified files become “unverifiable leftovers” and are deleted/recomputed (`391–395`).

   Preserve the previous manifest in the initial resumed report and checkpoint verified entries before expensive probe work.

**Other smoke-path findings**

The requested step ordering is compatible with staging: unit gates run first, followed by the supplied steps (`cluster/stage.py:117–118`). With `set -euo pipefail`, `fverify` exit 3 correctly prevents subsequent references and `ALL-DONE`; this is fail-closed behavior, not a blocker. The named output directories and slimming/scoring layout agree.

H200 restriction, positive finite reservations, reservation-before-submit, the account lock and fail-closed queue query remain present. I found no additional ordinary-path submission blocker.

The wrong-mesh Python control accepted its own table and rejected both the 128→256 mismatch and perturbed eigenvalues. I inspected the `jaxslot`/`jaxrun` wrappers but did not execute them, because they create lock/systemd state.

At **L=12288**, six interior f64 fields occupy approximately **7.25 GB**, before padded fields, solver vectors, FFT workspaces and cached executables (`refcore.py:164–189`, `230–233`; `refjob.py:113–120`). I found no demonstrated inevitable H200 OOM. Peak memory and two-hour completion remain measurements for the smoke, not established by this CPU audit.

A6’s production deferral is appropriately scoped. Its checklist should explicitly include **evidence-bound** production `fverify` acceptance, measured/workload-scaled verification cost, and propagation of calibration/separation qualifications through **analysis and export**, not merely report prose. Those production tasks do not add blockers to this round.