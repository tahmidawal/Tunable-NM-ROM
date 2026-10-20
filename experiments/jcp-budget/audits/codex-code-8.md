**Overall verdict: NEEDS-FIX.** Audited HEAD `8c50e3e50`. One Round 7 requirement remains incomplete: binding 2D G-Q certificate evidence to its configured case identities.

Paths below are relative to `experiments/jcp-budget`. **CORRECT** means no smoke blocker found through source inspection and the bounded CPU checks described below; it does not certify real-size GPU execution.

**Round 7 blocking-fix verification**

| Required fix | Assessment | Evidence |
|---|---|---|
| Validate 3D coefficient/certificate inventories against configured roles, cases and dimensions | **CORRECT.** The independent inventory covers corners, checks, stars, B7 and controls. Inventory failures invalidate acceptance and fail smoke coverage. | `ana3d.py:85–151,530–541`; `smoke_accept.py:121`. A complete 91-role fixture passed; omitted cases, duplicate cases, wrong shapes, omitted arrays and duplicate JSON rows all failed. |
| Validate 2D G-Q configured population, certificate case identities and registered thresholds | **NEEDS-FIX — partially corrected.** Configured population size and fixed thresholds are enforced. Certificate case identities remain absent and unchecked. | `b2run.py:516–535`; `ana2d.py:561–573`. See blocker below. |
| Fail closed when the scheduler query inside the campaign lock fails | **CORRECT.** Failed `squeue` exits 76 before submission; cap refusal exits 75. Both release the pending reservation explicitly. | `cluster/submit.sh:35–44`. Mocked query failure stopped submission; empty/seven-job queues permitted it; eight jobs refused it. |

**Remaining blocker: G-Q certificates are not bound to cases**

The producer writes `cert_LSPG` and `cert_GAL` without case IDs at [b2run.py:534](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/b2run.py:534).

The acceptance branch checks `ids == conf`, but those IDs came from **another file**, `CTRL_b2.npz` (`ana2d.py:537–538`). For `CTRL_gq.npz`, it checks only array dimensions and finiteness before calculating the fractions at [ana2d.py:568](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/ana2d.py:568).

I executed that exact acceptance branch in memory:

| Fixture | Result |
|---|---|
| Configured two cases, only one produced control case | Correctly rejected |
| Complete certificates containing `1e-10`, stored thresholds changed to zero | Correctly reports both non-certified fractions as zero; control does not fire |
| Correct companion-file IDs, complete certificates labelled with unrelated cases | **`evaluable=True`, `fired=True`** |
| Correct companion-file IDs, complete certificates labelled with duplicate case IDs | **`evaluable=True`, `fired=True`** |

The last two fixtures supplied explicit certificate IDs, which the acceptance branch ignored. The current producer omits them altogether. Thus a same-shaped certificate file from the wrong population cannot be distinguished from the required evidence. `smoke_accept.py:88` directly accepts this `fired` result.

This is the still-unimplemented identity requirement from Round 7, rather than a recurrence of its repaired size or threshold defects.

**Re-audit of the requested areas**

| Area | Assessment and evidence |
|---|---|
| Möbius effects, factorial closure, floor identity, ideal end and \(U_\star\) | **CORRECT.** `ana2d.py:137–148,246–291`; `ana3d.py:266–309`. Unverified star quadrature removes star-dependent conclusions. |
| \(\chi\), \(\chi^\star\), numerical background | **CORRECT per-time formulas; non-blocking reporting issue below.** `ana2d.py:292–300`; `ana3d.py:310–322`. |
| Ties, dominance, incomplete populations and pair labels | **CORRECT.** `ana2d.py:358–428`; `ana3d.py:374–434`. M participates in the 2D candidate/pair inventory; 3D F×S is explicitly undefined. |
| Growth, stability, sampling adequacy with uncertainty, ranking | **CORRECT inspected rules.** `ana7.py:139–195,291–307,335–423`: adjacent-window growth, resolved populations, uncertainty-aware route comparisons and common-population rankings. |
| C-A amplitude and C-E closed form | **CORRECT.** `b2run.py:433–452`; `ana7.py:302–313`. The alias amplitude uses the actual rejection threshold; energy comparison uses the unnormalized balance. |
| C-F, C-I, C-S, C-T, C-Q and C-score | **CORRECT inspected constructions and predicates.** `ana2d.py:317–353,516–555,576–590`; `ana3d.py:338–371,578–614`. G-Q’s remaining identity defect is separate. |
| C-R1 and G-R1/G-R2 matching and bars | **CORRECT for configured smoke ladders.** `anaref.py:59–79,156–174,198–208`. Final-purpose matching and shared-IC checks are present. |
| G-B3, G-B5 and G-B6 | **CORRECT.** Same-mesh/configuration archive filtering: `ana2d.py:157–176`; LS acceptance: `budcommon.py:133–146`; logical switch assertions: `budcommon.py:171–179`. |
| `budcore` output count | **CORRECT, CPU-tested.** `budcore.py:118,168,228,342–349`. All eight form/scheme combinations passed bitwise equivalence, horizon-prefix and finite/distinct-tail checks. |
| Split initialization, scale and M=2048 | **CORRECT.** Scale survives the `w0` replacement at `budcommon.py:182–207`; M selects a separate setting. qinf/qcheck construction and replacement are explicit at `b2run.py:150–154,209–236`. |
| Streamed LS QR | **CORRECT by inspection.** `budcommon.py:95–157`: row-block ordering matches flattened interior ICs, columns use the intended prefix, local \(Q_b^\mathsf TU_b\) is applied directly, and global condition/orthogonality checks govern acceptance. |
| B7 certification and anchors | **CORRECT driver wiring.** `b2run.py:254–309`; `b3run.py:88–125`. LSPG/GAL certifiers differ appropriately; adaptive anchors require `nfail == 0`; q+/q++ convergence is assessed. |
| B8 decode, normalization, stencils, band and alias field | **CORRECT inspected paths.** `budcommon.py:215–253`; `b2run.py:323–452`. Reference files are manifest-checked; both routes use the same realization and their respective fixed-\(t_0\) norms. |
| Reference horizon, stage buffers, extraction and window judges | **CORRECT.** `ref/refcore.py:102–238,251–278`; `ref/refrun.py:102–239`. The 3D shared-node extraction includes the required interior-index offset; fine files are separated. |
| 3D horizon, caches and FS/QN acceptance | **CORRECT normal driver path.** `s3common.py:180–249`; `s3ctx.py:142–169`; `b3run.py:171–175`; `ana3d.py:170–195`. Query/certificate caches are cleared when the horizon changes. |
| Case alignment, reference identity and archive mesh | **CORRECT inspected normal paths**, except the G-Q blocker. Explicit case lookups and reference realization checks appear in `ana3d.py:250–262`, `ana7.py:68–94,203–234`, and `ana2d.py:157–176`. |
| Cluster staging and execution | **CORRECT.** `cluster/stage.py:58–119`: unique attempt directory, committed HEAD contents, paralab outputs, GPU partition/preflight, x64 and highest precision. |
| Caps and reservations | **CORRECT normal submission path.** `cluster/submit.sh:13–50`; `ledger.py:50–75`: lane cap 3, campaign cap 8 under the shared lock, sequential smokes and 45-hour reservations. |
| Collection, promotion and cleanup | **CORRECT normal workflow.** `cluster/collect.py:27–90`: checksums, no-overwrite promotion, partial-collection refusal and remote verification of retained fine files before cleanup. |
| Real-size memory/time | **No additional demonstrated smoke blocker.** `b2run.py:542–565` exercises the combined 4096² bank, both M settings, Gauss320/Gauss640, fine bank and solves. Reference buffers, LS QR and 3D table construction still require measured smoke evidence. |

**Non-blocking items**

- **NEEDS-FIX before interpreting the reported cross-term decomposition:** `chi_star` is selected at its own maximum-magnitude time, while `chi` and `chi_background` use the time maximizing `chi`. Their stored scalars therefore need not satisfy `chi = chi_star + chi_background`. Preserve the time series or also emit all three at a common time (`ana2d.py:297–300`; `ana3d.py:319–322`). This does not affect current smoke gates.
- **Production resource limitations:** the memory pilot omits the accumulated fine-floor QR cache (`b2run.py:385–389`), and `resources.py:113–115` asserts H200 timing provenance instead of verifying it.
- **Declared deferred coverage:** A4 defers B8 3D and its C-A/C-E controls. These are not implemented coverage.

All 27 available inherited source hashes and 26 applicable destination hashes matched provenance. I compared the consequential inherited edits in `s3common`, `s3ctx`, `refcore`, `verdict`, `pscore` and `cohorts`. All 23 generated configurations matched committed JSONs using an in-memory writer replacement. Python parsing and shell syntax checks passed.

No files changed, jobs submitted, GPU work performed or scratch files created. The CPU test’s writing statements were removed in memory. LAB-LOG.md was untouched; Git status still shows only the pre-existing `HEARTBEAT.json` modification.

**Overall verdict: NEEDS-FIX. Required fixes — blocking only:**

1. Persist the certificate case IDs in `CTRL_gq.npz` and independently require the exact configured first-two-case sequence before accepting either certificate array. Reject absent, duplicate or mismatched IDs; retain the corrected shape, finiteness and fixed-threshold checks.