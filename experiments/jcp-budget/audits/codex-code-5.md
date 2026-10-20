**Overall verdict: NEEDS-FIX.** Audited commit `88cf56e3d450d6b618db46625cec4607e1c01612`. Two acceptance blockers remain; neither requires a numerical-method redesign.

Paths below are relative to [experiments/jcp-budget](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget).

**Verification of audit-4’s blocking fixes**

| Required fix | Verdict | Evidence |
|---|---|---|
| Derive 2D B7/B8 and fine-check member coverage from configuration | **CORRECT.** | `ana7.py:589–593` derives members from `config['members']`; fine obligations use that inventory at `ana7.py:458`. `smoke_accept.py:95` consumes the coverage result. An in-memory omitted-member fixture now returns `passed=False`. The intentional missing reference for dev6 case 1 remains allowed. |
| Reject missing, empty or malformed 3D G-Q certificates | **CORRECT for the reported failure cases.** | `ana3d.py:477–485` checks the q0 specification, configured cases, finite certificates and consistency with `steps_n`. Reexecuting these statements with missing, empty and `[None]` certificates returned `evaluable=False, fired=False` in every case. Ordinary solver acceptance still has a separate defect below. |

**Remaining blockers**

1. **NEEDS-FIX — configured-member coverage is still missing from 3D B7 and 2D B2 acceptance.**

   The 3D analysis enumerates produced B7 records at [ana3d.py:498](experiments/jcp-budget/ana3d.py#L498). Its smoke gate requires only a nonempty result and passing G-L1 checks for those produced members: [smoke_accept.py:125](experiments/jcp-budget/smoke_accept.py#L125). It never compares that inventory with `cfg['members']`.

   Using the actual `smk2_b3.json` configuration and the actual gate expression:

   ```text
   configured members: 3
   produced B7 members: 2
   G-L1 smoke gate: True
   ```

   Thus omitting the W512 B7 member, for example, removes its long-horizon verification obligation without failing this gate.

   The equivalent omission remains possible for 2D B2: `ana2d.py:435–454` enumerates produced `res['b2']` members; `smoke_accept.py:69–77` accepts aggregate identities and the resulting G-B4 inventory. Removing the GAL B2 member leaves the independently produced B7/B8 coverage intact and need not remove the C-T fixture or other controls.

   These are acceptance-predicate reproductions and source traces, not end-to-end smoke executions.

   **Required:** validate B2 and B7 inventories against their configured members before accepting either dimension. Missing members must fail explicitly, independently of aggregate gate counts.

2. **WRONG — ordinary 3D FS/QN acceptance can pass without valid per-step certification evidence.**

   [ana3d.py:88–91](experiments/jcp-budget/ana3d.py#L88) computes pooled stationarity from stored `nonstationary` counts. Its certificate check merely requires every supplied entry to be finite; an empty array passes that universal check.

   Executing the actual `tag_acceptance` function produced:

   | Certificate evidence | Stored `nonstationary` / `steps_n` | Accepted |
   |---|---:|---|
   | `[]` | 0 / 10 | **True** |
   | Ten certificates, each `0.1` | 0 / 10 | **True** |
   | Ten certificates, each `1e-5` | 0 / 10 | True |

   The second row has every step above the `1e-3` threshold, yet passes. These acceptance values feed G-B4 comparisons (`ana3d.py:109–113`), B2 effects (`ana3d.py:158–159`) and B7 qualification.

   The repaired G-Q path does not protect these positive acceptance paths. Its own length check also compares against self-reported `steps_n`, rather than the schedule: one finite certificate plus `steps_n=1` remains evaluable for the ten-step q0 fixture.

   **Required:** validate certificate lengths against the independently expected schedule, require complete case coverage, and recompute nonstationary counts from the certificate arrays. Reject missing, empty, truncated or inconsistent evidence. Use this validation for ordinary FS/QN acceptance and G-Q.

**Re-audit of the requested implementation**

“CORRECT” below means no additional blocker found through source inspection and bounded CPU checks; it does not certify real-size GPU execution.

| Area | Assessment and file:line evidence |
|---|---|
| Möbius effects, floor identity, closure, \(U_\star\), \(\chi\), \(\chi^\star\) | **CORRECT formulas.** `ana2d.py:207–280`; `ana3d.py:155–220`. M joins the 2D main/pair design; the 3D F background is the tight-end trajectory. Coverage and certification remain subject to the blockers above. |
| Ties, dominance, incomplete populations, pair labels | **CORRECT.** `ana2d.py:332–404`; `ana3d.py:273–335`. Unavailable candidates prevent complete-case attribution; population and eligible-case denominator rules are implemented. |
| Growth, stability, sampling adequacy with uncertainty, front order | **CORRECT principal formulas.** `ana7.py:139–195,283–302,330–418`. Eligibility tiers distinguish verdicts from provisional/descriptive results. |
| C-A amplitude and classifier; C-E closed form | **CORRECT.** Construction: `b2run.py:433–452`; evaluation: `ana7.py:297–308`. Fine recomputation independently checks the injected fields and measured balance. |
| C-F, C-I, C-S, C-T, C-Q, C-R1, C-score | **CORRECT inspected construction/predicates.** `ana2d.py:291–327,486–544`; `ana3d.py:455–495`; `anaref.py:198–208`. C-R1 is rejudged from saved level fields. |
| G-R1/G-R2 matching and bars | **CORRECT for the configured smoke ladders.** `anaref.py:59–79,156–174`. G-R2 matches final levels by mesh, step and purpose, with separate t0 equality. |
| G-B3/B4/B5/B6 | **CORRECT driver wiring; NEEDS-FIX acceptance coverage/certification.** Archive matching: `ana2d.py:145–156`; logical switches: `b2run.py:193–197`; LS acceptance: `budcommon.py:133–143`; Q/check roles: `b2run.py:209–234`. |
| Long-horizon Q1/Q2, q+ versus q++ convergence, anchors | **CORRECT formulas and driver wiring.** `b2run.py:255–309`; `ana7.py:213–260`; `ana3d.py:374–394`. Adaptive anchors require `nfail == 0` at `budcommon.py:201–207` and `s3common.py:245–249`. FS/QN acceptance needs blocker 2. |
| `budcore` nout and split initialization | **CORRECT.** The core diff is output-count plumbing. `budcommon.py:182–207` preserves initialization scale when replacing \(w_0\). All eight manufactured CPU core cases passed bitwise equivalence, horizon-prefix equality and finite/distinct tails. |
| M=2048; qinf/qcheck roles | **CORRECT.** Separate settings and role-specific tables: `b2run.py:150–154,211–225`; selection: `budcommon.py:185–190`. |
| Streamed LS QR | **CORRECT by inspection.** Flattening, row ordering, prefix columns, direct \(Q_b^\mathsf{T}U_b\), global QR and acceptance: `budcommon.py:95–157`. No local triangular inverse is used. |
| B8 decode, reference loading, stencils, band mask, alias | **CORRECT inspected paths.** `budcommon.py:215–253`; `b2run.py:323–452`. Reference identity, cohort, metric and hashes are checked; normalization uses the appropriate route’s t0 field. |
| Reference horizons, stages, extraction, window judges, fine split | **CORRECT.** `ref/refcore.py:102–238,251–278`; `ref/refrun.py:102–239`. The 3D shared-node convention uses the required offset. Consequential edits were compared with their inherited sources. |
| 3D horizons and caches | **CORRECT normal driver path.** `s3common.py:100–103,206–249`; `s3ctx.py:142`; `b3run.py:171–175`. Query and certification caches are cleared before changing to B7’s horizon. |
| Case alignment, reference identity, NaNs, archive mesh, manifest paths | **CORRECT inspected normal paths, except blocker 2.** Case identifiers drive matching; invalid 3D rho has explicit status; reference hashes bind analysis eligibility to the loaded realization. `ana7.py:68–94,203–230`; `b3run.py:104–125`. |

All recorded inherited-source hashes and applicable destination hashes matched `PROVENANCE.json`. Python source parsing passed. All **23 lane-generated configurations** matched their committed JSONs in an in-memory check.

**Cluster and resource assessment**

| Area | Verdict |
|---|---|
| Unique job directory, namespace, committed HEAD staging, GPU partition/preflight, x64/highest precision, paralab outputs | **CORRECT:** `cluster/stage.py:58–119`. |
| Lane cap 3, campaign cap 8, sequential smokes, 45-hour reservations | **CORRECT for the documented workflow:** `cluster/submit.sh:18–40`; `ledger.py:50–75`. |
| Promotion without overwrite, checksums, `--no-fine`, cleanup | **CORRECT normal workflow:** `cluster/collect.py:27–66,80–90`. Retained fine files must be verified before cleanup. |
| Real-size memory/time | **No additional demonstrated blocker.** The 4096² bank is roughly 48 GiB before tables and workspaces. The pilot exercises both M settings, Gauss320/Gauss640, fine Bank(1024), LS initialization and corner solves (`b2run.py:542–565`). Reference output buffers and 3D table construction still need measured smoke evidence. |

**Non-blocking for this smoke:** the pilot does not reproduce the accumulated B8 fine-floor QR cache (`b2run.py:385–389`); resource certificates assert H200 timing provenance rather than verifying the measured GPU record (`resources.py:114–117`). These limit production extrapolation. B8 3D and its C-A/C-E controls are explicitly deferred by the appended A4; that is an additional specification amendment beyond A1–A3.

**Overall verdict: NEEDS-FIX. Required blocking fixes:**

1. Require configured-member coverage for 2D B2 and 3D B7 before smoke acceptance.
2. Validate complete 3D certificate evidence against expected schedules/cases and recompute stationarity counts for positive acceptance and G-Q.

Audit-handling disclosure: no jobs or GPU work ran, and LAB-LOG.md was untouched. One mistaken module import invoked the vendored configuration generator, rewriting existing JSONs with identical contents. Final Git status shows only the pre-existing `HEARTBEAT.json` modification.