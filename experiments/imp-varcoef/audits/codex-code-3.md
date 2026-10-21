**NOT CLEAN: diagnostic validation still permits misleading reported costs.** The original production C=1 crash is fixed. I found no new real-size crash in this patch.

Read-only audit completed: no files changed, jobs submitted, GPU execution, or arrays copied. Checks used small, in-memory CPU fixtures. Paths below are relative to `experiments/imp-varcoef/`.

| Round-2 item | Verdict | Evidence and required fix |
|---|---|---|
| **2 — gate validation** | **CORRECT** | `gate.py:73–96,137–138,189–191,223–226` now requires registered slots/repetitions, nonempty exact scoring coverage, a qualified winner, and a valid, non-vetoed anchor. CPU checks confirmed empty scoring and repetition 99 are rejected. Remaining fabricated-record attacks are noted under item 5. |
| **4 — stale cross-checks** | **CORRECT** | `vref.py:236` records the checked reference hash; `elig.py:43–48` rejects a cross-check whose hash differs from the final reference. No further fix for this defect. |
| **5 — hash chain** | **NOTED (forgery-only residual)** | `vtime.py:135–147,205–207` now validates exact qualified group membership, ladder errors, and loaded reference hashes. `gate.py:127–135` requires the binding-file schema. It still does not authenticate the actual frozen-selection artifact against `T['selection']`, or bind the supplied ladder result directly to the candidates artifact. These are nonblocking under the stated hand-edited-artifact exclusion. Hardening: bind the ladder and frozen selection explicitly and check ROM specifications. |
| **6 — terminal-state guard** | **CORRECT** | `vspec.py:165–168,217–219` evaluates each interval’s terminal accepted state on the native nonlinear grid. Intermediate accepted states are checked by the following step. No fix needed. |
| **8 — cheap-edge registration** | **CORRECT** | `DESIGN.md:460–464` explicitly registers the superset and CCv level 4, matching `vladder.py:252–270`. Its additional workload still needs inclusion in the actual resource certificate. |
| **11 — CCv decomposition** | **CORRECT** | `vfd.py:320–326` prepares and consumes the denominator. For this CN-R schedule, both startup and ordinary stages have \(\theta\tau=\Delta t/2\), so one denominator is sufficient. Cosmetic fix: update the contradictory old docstring at `vfd.py:308–310`. |
| **12 — diagnostic validation** | **WRONG** | `vtime.py:299–301` includes kernels in G-FR, but that checks a **scalar reference error**, not equality with the query trajectory. `vtime.py:263,307–311` still compares each kernel only against its own scoring identity. Moreover, scoring failures do not populate `VETO`, and `gate.py:194–197` reports kernel cost using timing coverage and veto alone. **Fix:** compare each kernel against its corresponding query per case; mark failed reproduction/validity/identity diagnostics unusable; require that evidence before reporting their costs. Explicitly mark missing winner decompositions incomplete. |
| **13 — test-once enforcement** | **NOTED (deliberate replay)** | `make_configs.py:179` exclusively creates the opening record; `testgate.py:37–42` enforces exact winner keys and C=100. Reinvoking the evaluator remains possible, as acknowledged by A6.4. Hardening: persistent execution claim. Also move the selection copy at `make_configs.py:177` after the opening guard so a refused replay cannot first overwrite the frozen selection. |
| **15 — generated C=1 configuration** | **CORRECT** | `vtime.py:164–169` maps C=1 subjects to C=100 physical slots while retaining contrast 1 for coefficient preparation. `vtime.py:264–269` avoids requesting nonexistent C=1 references. The original `KeyError` is fixed. |
| **16 — budget enforcement** | **CORRECT under the registered sequential-submitter scope** | `cluster/submit.sh:12–13` rechecks immediately before submission; `ledger.py:62–69` uses the reservation’s rounded Slurm limit, generated at `cluster/stage.py:86,127`. A failed accounting refresh retains conservative reservations. Concurrent submitters remain excluded by A6.4. |
| **17 — bookkeeping races** | **NOTED (excluded concurrency/recovery limitation)** | `ledger.py:32` fixes temporary-file collisions. Read-modify-write remains unlocked, and `cluster/submit.sh:25–30` still has a submission-to-registration interruption window. This is not a normal uninterrupted sequential-run numerical defect. Hardening: locked transactions and durable submission reconciliation. |
| **24 — reporting** | **WRONG** | `gate.py:180–184` correctly reports fallback consistency as “not assessed.” However, `gate.py:198–201` can report C=1 cost despite failed C=1 validity; see new bug below. It also emits an anchor-normalized C=1 cost rather than the explicit C=100/C=1 comparison required by `DESIGN.md:146`. **Fix:** require valid C=1 scoring and report the explicitly defined paired contrast comparison. Include selected PSI-family diagnostics even when PSI is not the union winner. |

The five **NEW findings from round 2**, reassessed separately:

| Round-2 new finding | Verdict | Evidence and fix |
|---|---|---|
| C=1 input-preparation crash | **CORRECT** | Fixed by `vtime.py:166–169`; see item 15. |
| Empty bindings / wrong-target candidates | **CORRECT** | `gate.py:127–128` rejects the empty schema; `vtime.py:135–142` validates exact group membership. Broader forgery residuals are item 5. |
| Empty scoring accepted | **CORRECT** | `gate.py:87–96` requires exactly six named scoring rows. CPU fixture confirmed rejection. |
| Fallback falsely certified as fresh consistency | **CORRECT** | `gate.py:180–184` computes consistency only for successful fresh measurements. |
| Smoke acceptance lacks real-size/pilot checks | **CORRECT for the honest generated workflow** | `smoke_accept.py:36–37` now requires `small=False`, successful pilots, and nonempty pilot evidence. Source hashes remain unrevalidated by staging at `cluster/stage.py:48–49`; fabricated acceptance records remain **NOTED**, not blocking, under A6.4. |

**NEW bug introduced by this patch**

- **WRONG — newly reported diagnostic costs can certify invalid executions.**  
  `gate.py:194–201` introduces numerical kernel/setup/C=1 reporting based only on complete timing records and absence from `VETO`. But C=1 validity failures are stored only in scoring rows at `vtime.py:265–268`; `VETO` is populated by identity mismatches at `vtime.py:307–311`. A deterministic unstable C=1 solve can therefore have `valid=False` in scoring, reproduce its identity throughout timing, and receive an ordinary numerical `C1_same_config` cost. Failed kernel reproduction has the analogous reporting path.

  **Fix:** give diagnostics their own acceptance predicate: exact scoring coverage, all required validity checks, kernel/query trajectory agreement, and complete valid timing coverage. Report failed diagnostics as unavailable with a reason. This does not require C=1 accuracy qualification or making C=1 gate the main experiment.

The CPU checks also illustrated why equal reference-error norms cannot establish kernel/query identity: different fields can have exactly equal scalar errors. I did not establish that the current kernels produce different fields; the defect is that the requested validation still cannot detect that outcome.

**Overall verdict: NOT CLEAN — WRONG findings remain in items 12 and 24 and the new diagnostic-reporting path.**