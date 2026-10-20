**Overall verdict: NEEDS-FIX.** Audited commit `893943d2d329071a074459dce6ed92a20f8fdbdb`. Two blocking acceptance issues remain.

Paths below are relative to [experiments/jcp-budget](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget). Findings are based on source inspection and bounded CPU fixtures, not a real-size GPU run.

**Verification of audit-5’s blocking fixes**

| Audit-5 required fix | Verdict | Evidence |
|---|---|---|
| Require configured-member coverage for 2D B2 and 3D B7 | **CORRECT.** | `ana2d.py:459–463`, `ana3d.py:431–435`; consumed by `smoke_accept.py:69,121`. Executing the coverage statements with the actual smoke configurations rejected an omitted member in both dimensions. |
| Validate 3D certificate lengths against schedules, recompute stationarity, require complete cases | **NEEDS-FIX — partly resolved.** | `ana3d.py:92–101` now rejects empty, truncated, non-finite and above-threshold certificates; bounded fixtures confirmed this. G-Q independently requires ten certificates and configured cases at `ana3d.py:490–498`. Ordinary acceptance still derives its expected cases from the produced NPZ, leaving the omission path below. |

**Blocking findings**

**1. NEEDS-FIX — coverage checks still accept missing configured cases or required quadrature checks.**

Two concrete paths remain:

- **3D B7:** `tag_acceptance` takes `cases_expected` from `Wz['cases|…']` at `ana3d.py:95`. B7 then iterates that produced inventory at `ana3d.py:356–365`; G-L1 defines its expected count as the number of surviving rows at `ana3d.py:403–406`. The new member gate checks member names only.
- **2D G-B4:** `gb4_member` enumerates the produced check-file cases and corners at `ana2d.py:169–180`. Missing Q-containing corners produce no failed entry. `smoke_accept.py:76–78` consequently accepts an empty per-corner check inventory when the separate star-family check passes.

CPU reproductions using the actual functions/predicates:

| Fixture | Result |
|---|---|
| Actual smk2 configuration expects cases `[0,1]`; B7 coefficient inventories and certificate rows contain only case `0` | Ordinary acceptance passes for case `0`. |
| Same B7 fixture, with the surviving trajectory exactly reproducing its archive | **G-L1 passes, Q1 passes, Q2 reports `pass`**, despite the missing configured case. |
| 2D check file retains case IDs but has no Q-corner entries | `per_corner={}`; **smoke G-B4 passes** when star-family verification passes. |

These are acceptance-unit reproductions with synthetic small arrays, not claims that a complete smoke was executed. They demonstrate that missing evidence can remove its own verification obligation.

**Required:** derive expected case sets and required corner/role inventories from configuration and the registered design. Compare produced arrays and certificate rows against those inventories before aggregation. Apply this to B7 prod/tight/tighter/q+/q++, B2 check replays and star-family checks; missing evidence must explicitly fail the relevant smoke gate.

**2. WRONG — 2D G-Q can “fire” on NaNs or truncated certificate evidence.**

At `ana2d.py:523–527`, the negative-control fraction is computed using:

```python
~(certificate <= threshold)
```

NaNs therefore count as non-certified steps. Acceptance requires nonempty arrays but checks neither finiteness nor expected dimensions. `smoke_accept.py:88` directly consumes this `fired` result.

Executing those exact acceptance statements, with the wrong-table guard marked successful:

| Both certificate arrays contain | G-Q `fired` |
|---|---:|
| Shape `(2,50)`, entirely NaN | **True** |
| One NaN | **True** |
| One finite above-threshold value | **True** |

A broken certifier is thus indistinguishable from a successful premature-exit control.

**Required:** validate both forms’ finite certificate arrays against the configured control population and independently expected schedule. For smk1, the current driver emits two cases × fifty steps per form (`b2run.py:512–535`). Only complete, evaluable evidence should enter the “>1% non-certified” predicate; validate the registered thresholds independently too.

**Re-audit of the requested implementation**

“CORRECT” here means no additional smoke blocker found in the inspected implementation; it does not certify GPU behavior.

| Area | Assessment and evidence |
|---|---|
| Möbius effects, floor identity, closure, \(U_\star\), \(\chi\), \(\chi^\star\), numerical background | **CORRECT formulas.** `ana2d.py:207–280`; `ana3d.py:155–227`. M participates in 2D main/pair effects; 3D F uses the tight-end background. |
| Ties, dominance, incomplete candidates, pair labels | **CORRECT.** `ana2d.py:332–404`; `ana3d.py:273–344`. Missing ideal-end/Q evidence prevents complete attribution; population and completeness rules are implemented. |
| Growth, stability, sampling uncertainty and front order | **CORRECT principal calculations.** `ana7.py:139–195,283–302,330–418`. Adjacent-window growth, uncertainty bars, common populations and eligibility tiers are present. |
| C-A amplitude/classification, C-E closed form | **CORRECT inspected paths.** `b2run.py:433–452`; `ana7.py:297–308,420–563`. Fine recomputation independently checks injected alias errors and measured balance. |
| C-F, C-I, C-S, C-T, C-Q, C-score | **CORRECT inspected construction/predicates.** `ana2d.py:291–327,491–549`; `ana3d.py:468–512`. |
| C-R1; G-R1/G-R2 matching and bars | **CORRECT for configured smoke ladders.** `anaref.py:59–79,156–174,198–208`. C-R1 is rejudged from saved fields; G-R2 matches mesh, step and purpose and checks t0 separately. |
| G-B3/B4/B5/B6 | **CORRECT numerical wiring; NEEDS-FIX coverage.** Archive matching: `ana2d.py:145–156`; logical switches: `b2run.py:193–197`; LS acceptance: `budcommon.py:133–143`; check roles: `b2run.py:209–234`. Blocker 1 affects G-B4 acceptance. |
| Long-horizon Q1/Q2, q+ versus q++, anchors | **CORRECT formulas and driver wiring; NEEDS-FIX population validation.** `b2run.py:255–309`; `ana7.py:213–260`; `ana3d.py:382–406`. Adaptive anchors require `nfail == 0`; FS/QN certificate arithmetic is repaired, subject to blocker 1. |
| `budcore` nout; split-path initialization | **CORRECT.** Diff against `p1core` is output-count plumbing. `budcommon.py:182–207` preserves scale when replacing \(w_0\). All eight manufactured CPU cases passed equivalence, prefix and tail checks. |
| M=2048 and rule roles | **CORRECT.** Separate settings and role selection: `b2run.py:150–154,211–225`; `budcommon.py:185–190`. |
| Streamed LS QR | **CORRECT by inspection.** `budcommon.py:95–157`: row-block ordering, prefix columns, direct \(Q_b^\mathsf TU_b\), global QR, conditioning and orthogonality acceptance. No local triangular inverse. |
| B8 decode, reference loading, stencils, band and alias field | **CORRECT inspected paths.** `budcommon.py:215–253`; `b2run.py:323–452`. Identity, cohort, metric and hashes are checked; normalization is route-specific. |
| Reference horizons, stage buffers, extraction, window judges and fine split | **CORRECT.** `ref/refcore.py:102–238,251–278`; `ref/refrun.py:102–239`. The 3D shared-node extraction uses the required interior-index offset. |
| 3D horizons and caches | **CORRECT normal driver path.** `s3common.py:100–103,206–249`; `s3ctx.py:142`; `b3run.py:171–175`. Query/certification caches are cleared before the B7 horizon change. |
| Alignment, reference realization, archive mesh, manifest paths | **CORRECT inspected normal paths.** `ana2d.py:145–156`; `ana7.py:58–94,203–230`; `b3run.py:104–125`. Missing-population and 2D certificate handling remain the exceptions above. |

All inherited-source hashes and applicable destination hashes matched `PROVENANCE.json`. I compared the consequential edits in `s3common`, `s3ctx`, `refcore`, `verdict`, `pscore` and `cohorts` against their recorded sources. Python parsing passed, and all **23 generated configurations** matched committed JSONs using an in-memory replacement for the generator’s writer.

| Cluster/resource area | Assessment |
|---|---|
| Unique directory, namespace, HEAD staging, GPU partition/preflight, f64/highest, paralab outputs | **CORRECT:** `cluster/stage.py:58–119`. |
| Lane cap 3, campaign cap 8, sequential smokes, 45-hour reservation | **CORRECT for documented workflow:** `cluster/submit.sh:18–40`; `ledger.py:50–75`. |
| Checksummed collection, no-overwrite promotion, `--no-fine`, cleanup | **CORRECT normal workflow:** `cluster/collect.py:27–66,80–90`. Cleanup verifies retained fine files before deleting the job directory. |
| Real-size memory/time | **No additional demonstrated blocker.** `b2run.py:542–565` exercises the combined 4096² bank, both M settings, Gauss320/Gauss640, fine bank and solves. Reference output buffers and 3D table peaks still require measured smoke evidence. |

**Non-blocking for this smoke:** the pilot omits the accumulated fine-floor QR cache (`b2run.py:385–389`), and resource certificates assert H200 timing provenance rather than checking the measured GPU record (`resources.py:114–117`). These limit production extrapolation. The current DESIGN also contains A4, explicitly deferring B8 3D and its C-A/C-E controls; those are deferred scope, not implemented functionality.

No files were changed, no jobs submitted, and no GPU work performed. LAB-LOG.md was untouched; Git status still shows only the pre-existing `HEARTBEAT.json` modification.

**Overall verdict: NEEDS-FIX. Required blocking fixes:**

1. Validate configured case populations and required quadrature-check inventories independently of produced records.
2. Reject non-finite, truncated or otherwise incomplete 2D G-Q certificate evidence before evaluating whether the control fires.