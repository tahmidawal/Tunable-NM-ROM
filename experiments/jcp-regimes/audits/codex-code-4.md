**NEEDS-FIX for smoke at `066c77d6d`.** Deferred production work does not drive this verdict.

Read-only audit completed: no file changes, jobs, lab-log writes, or scratch files. CPU unit gates passed; 26 Python files parsed successfully. The actual wrong-mesh Python control passed with its output intercepted in memory.

Paths below are relative to [experiments/jcp-regimes](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes).

| Audit 3 smoke blocker | Verdict | Evidence |
|---|---|---|
| 1. Incomplete budget-zero evidence | **NEEDS-FIX** | Dimensions are now checked and gamma is recomputed for all six cases (`smoke_accept.py:187–201`). Registered case/operator/schedule identity remains unchecked; see blocker 2 below. |
| 2. Wrong initial data accepted by probes | **CORRECT** | Canonical descriptor and initial-field checks now cover upwind members (`smoke_accept.py:95–101`) and both calibration members, including the selected production arrays (`112–124`). Independently reconstructed stage counts are **202/402/802**. |
| 3. Unbound full-field certificate | **CORRECT** | `fverify.py:57–64` records R-file, field and configuration hashes. `smoke_accept.py:221–226` checks exact arm inventory and binds those hashes to pulled R files and the local verification config. F-file download is unnecessary. |
| 4. Filesystem boundary bypasses | **NEEDS-FIX** | Reference/slimming directory guards and resolved slimming-source checks are fixed (`refjob.py:267`, `refslim.py:43–59`). Local and administrative bypasses remain; see blocker 1. |
| 5. Second interruption loses previously verified manifest | **CORRECT for that exact defect** | The first resumed save preserves the old manifest (`refjob.py:664–669`), and verified cases checkpoint before probes (`393–399`). A fresh-case interruption gap remains; see blocker 3. |

This also resolves audit 3’s smoke-path fix-table items **3 and 4**, while **1 and 6 remain NEEDS-FIX**. Qualification expectations still come from configuration through `rscore.arm_expect()` (`482–488`), with population/form/step checks at `220–235`. Dedicated regression fixtures remain absent from `test_regimes.py`; that absence alone is not a blocker here.

The remaining smoke blockers are:

1. **The write boundary still permits writes outside the lane.**

   `guard.py:45` computes the permitted local root using `realpath(lane/checks)`. If `checks/` is a symlink outside the lane, its target becomes permitted. An in-memory filesystem-resolution fixture confirmed that `check_local()` accepts this escape.

   Additionally, `smoke_accept.py:209–211` deletes an existing wrong-mesh result **before any boundary check**. A redirected `checks/` directory can therefore delete an external file even where the later writer would refuse it.

   Administrative writers also lack the separately scoped policy requested in audit 3: `cluster/stage.py:62–79`, `ledger.py:29–38`, and `cluster/submit.sh:12` follow redirected `runs/` or lock paths without checking their resolved destinations.

   **Required:** anchor permitted directories beneath the resolved lane root, validate before deletion/creation, and guard staging/ledger/lock destinations under an explicit administrative policy.

2. **Budget-zero validation verifies self-consistency, but not the registered control identities.**

   `smoke_accept.py:187–188` checks dimensions, rule name and mesh. It does not verify canonical viscosities, expected scheme/form, mode/eigenvalue identity, or operator identity. Lines `194–196` then recompute gamma using those same unchecked inputs.

   A concrete invariant exposes the gap: permuting `states`, `nu` and `cert` together while leaving the six `case_ids` unchanged preserves every dimension, pooled fraction and gamma-comparison result. The wrong case assignments therefore pass. Likewise, a self-consistent BE control is not rejected for replacing the registered LSPG-CNR control: `scheme` is consumed directly at line `195`.

   This is the normalization/operator/schedule validation explicitly requested by audit 3’s blocker 1.

   **Required:** bind control metadata and case parameters to the frozen smoke specification and trusted operator evidence before recomputation.

3. **A newly completed reference is not checkpointed before its expensive probe.**

   `refjob.py:413` writes the case and appends its hashes to the local `manifest`. Probe execution starts at `416`, but `rep['manifest']` is not updated until `429`.

   An intercepted execution of the actual `mode_fd()` confirmed: the case writer completed, interruption occurred at probe entry, and the only persisted manifest was **empty**. On restart, those completed files fail `verified()` and are removed at `402–407`.

   The resumed-case branch is repaired; the fresh-case branch still violates the addendum’s “checkpoints before any probe” requirement.

   **Required:** persist the merged manifest immediately after `write_case()`, before invoking the probe.

4. **Explicit submission refusal incorrectly strands a reservation as AMBIGUOUS.**

   `cluster/submit.sh:28` installs the ambiguity `ERR` trap. `set +e` at line `29` does **not** disable that trap. When the account-cap or queue-query check returns **75 or 76**, the failed assignment at line `30` first invokes `ledger.py ambiguous`.

   The explicit refusal handlers then call `release` (`33–34`), but `ledger.py:81–85` releases only `PENDING_SUBMIT` entries. The reservation remains `AMBIGUOUS` despite a known non-submission, consuming campaign budget.

   A small Bash reproduction confirmed the trap executes before the refusal handler.

   **Required:** classify the return status before marking ambiguity; release known non-submissions and retain reservations only for genuinely uncertain outcomes.

Other requested checks found no additional blocker. `fverify` rejects non-finite valid comparisons and exits **3** on failure (`fverify.py:45–67`). With `set -euo pipefail`, that correctly stops subsequent references and prevents `ALL-DONE`. Unit gates precede the supplied steps; output names and slimming layout agree. H200 restriction, finite positive reservations, committed-blob staging, campaign locking and fail-closed queue queries remain present.

The wrong-mesh control accepted its own table and rejected both the mesh mismatch and perturbed eigenvalues. I inspected `jaxslot`/`jaxrun` but did not execute those wrappers because they create lock/systemd state.

At **L=12288**, six interior f64 fields occupy **7.247 GB**, before solver vectors, padded outputs, FFT workspace and cached executables (`refcore.py:164–189`, `230–233`; `refjob.py:113–120`). I found no demonstrated inevitable H200 OOM. Peak memory and two-hour completion remain smoke measurements.

A6’s production deferral is appropriately scoped, including its added evidence-bound verification, measured verification cost, and analysis/export qualifications. One additional deferred boundary item should be explicit if calibration mode will be used: `refjob.py:526` still writes calibration arrays without a per-file guard. That mode is absent from this smoke and does **not** block this round.