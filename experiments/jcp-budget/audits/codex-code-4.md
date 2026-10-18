**Overall verdict: NEEDS-FIX.** Audited `24262c063d95daf7d7b61cdaa14285c2fbaaae19`. Two acceptance blockers remain.

Read-only throughout: no file changes, jobs, GPU use, or scratch files. The existing `HEARTBEAT.json` modification was present before the audit.

Paths below are relative to [experiments/jcp-budget](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget).

**Verification of audit-3’s four blocking fixes**

| # | Required fix | Verdict | Evidence |
|---|---|---|---|
| 1 | Bind reference JSON and verify window uncertainties | **CORRECT** | Both JSON/NPZ hashes and final-level identifiers are checked against anaref evidence (`ana7.py:89–92`). Window `eps_ref` is compared with recomputation, including missing/nonfinite handling (`anaref.py:95–102`). A manufactured ladder passed unchanged; changing W2 uncertainty to `999` made W2 agreement false. |
| 2 | Require local G-R3 and independent fine-check coverage | **NEEDS-FIX — partial** | Smoke acceptance now requires `G_R3_local_all` (`smoke_accept.py:56–57`). Missing fine files, records and LS coefficients fail for enumerated members (`ana7.py:473–552`). However, the expected member inventory still comes from produced B7 records, not the configuration. Blocker 1 below. |
| 3 | Recompute C-R1/G-Q and reject missing evidence | **NEEDS-FIX — partial** | X1 level fields and judge inputs are persisted and rejudged (`ref/refrun.py:311–315`; `anaref.py:198–208`). Both 2D certificate arrays are required (`ana2d.py:515–525`). The 3D path converts missing certificates into evidence that the control fired. Blocker 2 below. |
| 4 | Inspect stderr for warnings/errors | **CORRECT** | Both streams are scanned for captured-constant warnings and listed fatal errors; positive backend/completion checks remain (`smoke_accept.py:38–47`). |

**Remaining blocking findings**

1. **NEEDS-FIX — fine-route coverage can silently omit a configured member.**

   [ana7.py:588](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/ana7.py:588) constructs `members` from `res['b7']`. The “independent” expected fine-check set then uses that same inventory at `ana7.py:458`.

   [smoke_accept.py:95](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/smoke_accept.py:95) compares produced B8 members against produced B7 members. Removing a member from both therefore removes its verification obligation.

   An in-memory fixture using the actual member-extraction statement and coverage predicate produced:

   ```text
   configured members: 3
   derived expected members: 2
   smoke B8 coverage gate: True
   ```

   The remaining members can pass fine recomputation while the omitted member is never inspected. This is a coverage-check reproduction, not an end-to-end smoke execution.

   **Required:** derive expected members from the smoke configuration, validate the produced B7/B8 inventories against it, and construct fine-check obligations from configured members × required check cases with references. Preserve the intentional absence of dev6 case 1’s reference.

2. **WRONG — missing 3D G-Q certificates count as a successful negative control.**

   [ana3d.py:477](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/ana3d.py:477) uses:

   ```python
   r_.get('gam_cert') or [None]
   ```

   Lines 478–479 count that synthetic `None` as nonstationary evidence and report `fired=True`. Executing the actual statements in memory gave:

   | Certificate evidence | Reported fraction | `fired` |
   |---|---:|---|
   | Missing `gam_cert` | 1.0 | `True` |
   | Empty array | 1.0 | `True` |
   | `[None]` | 1.0 | `True` |

   The control’s failed ordinary acceptance does not rescue this: C-S requires acceptance only of its tight comparator (`ana3d.py:473–474`), while smoke acceptance consumes G-Q’s `fired` field directly.

   **Required:** validate the prescribed q0 specification, expected cases and per-step certificate coverage before computing the fraction. Missing, empty or malformed evidence must fail the control’s evaluability check, rather than establish that it fired.

**Re-audit against the specification**

“CORRECT” means no additional smoke blocker found through source inspection and the bounded checks below; it does not certify real-size GPU execution.

| Requested area | Verdict and evidence |
|---|---|
| Möbius effects, floor identity, closure, sum/RSS comparisons | **CORRECT.** Factorial inversion and identity checks are separate from quadrature-qualified attribution: `ana2d.py:207–265`; `ana3d.py:155–207`. |
| Candidate universe, M pairs, \(U_\star\), \(\chi\), \(\chi^\star\), background | **CORRECT.** M joins the six-ingredient contest/pairwise design; four ideal-family runs supply uncertainty. The 3D form effect uses its tight-end background: `b2run.py:45–52,165–177`; `ana2d.py:251–280`; `ana3d.py:179–220`. |
| Ties, dominance, incomplete cases, pair labels, minimum populations | **CORRECT.** Missing candidates prevent complete-case dominance; eligible-case pair denominators and population thresholds are explicit: `ana2d.py:332–400`; `ana3d.py:273–335`. |
| Growth, stability, sampling with uncertainty, front order | **CORRECT principal formulas.** Adjacent-window growth, eligibility tiers, energy checks and uncertainty-aware sampling comparisons are implemented: `ana7.py:139–178,182–195,283–302,349–418`. Coverage remains blocker 1. |
| C-A, C-E, C-F, C-I, C-S, C-T, C-Q, C-score | **CORRECT principal construction/predicates.** Alias amplitude and measured balance controls: `b2run.py:433–452`; floor construction: `ana2d.py:291–327`; other controls: `ana2d.py:486–546`, `ana3d.py:455–489`. C-R1’s revised offline judging is correct; G-Q retains blocker 2. |
| G-B3/B4/B5/B6 | **CORRECT numerical wiring.** Archive matching includes mesh/configuration; Q-containing corners and ideal-family replays are checked; rejected LS initializations withhold dependent runs; logical switches are asserted: `ana2d.py:145–195`; `b2run.py:105–115,158–163,193–225`; `budcommon.py:171–190`. |
| Long-horizon Q1/Q2, q+ versus q++ convergence, anchors | **CORRECT.** Tight/tighter distances, certificates, successor-rule distances and finer-rule convergence are implemented: `b2run.py:255–309`; `ana7.py:213–260`; `ana3d.py:374–394`. Adaptive anchors require `nfail == 0`: `budcommon.py:201–207`; `s3common.py:245–249`. |
| `budcore` nout and split initialization | **CORRECT.** The diff is confined to output-count plumbing. Replacement of \(w_0\) preserves the initialization scale: `budcommon.py:182–207`. All eight CPU core cases passed original equivalence, horizon-prefix equality and finite/distinct tails. |
| M=2048 and quadrature roles | **CORRECT.** M selects a separate setting with its own deployed/qinf/qcheck tables: `b2run.py:150–154,211–225`; `budcommon.py:162–190`. |
| Streamed LS QR | **CORRECT.** Row-block ordering, prefix columns, direct \(Q_b^\mathsf{T}U_b\), global QR and conditioning/orthogonality acceptance agree: `budcommon.py:95–157`. Small CPU checks, including a rank-deficient local block, matched dense least squares within \(4.2\times10^{-16}\). |
| B8 decode, reference identity, diagnostics | **CORRECT formulas and loading checks.** Fine Bank(1024), fixed-route normalizers, central-difference dissipation, interior band masks and alias field: `budcommon.py:215–253`; `b2run.py:323–452`. Independent member coverage remains blocker 1. |
| Reference horizon, stages, extraction, per-window judges, fine split | **CORRECT.** Horizon-sized buffers/stage counts preserve inherited arithmetic; 3D extraction uses \(sj-1\): `ref/refcore.py:102–238,251–278`; `ref/refrun.py:102–142,154–239`. |
| G-R1/G-R2 matching and bars | **CORRECT for configured smoke ladders.** Final-triple matching includes L/dt/purpose; t0 equality and evolved-time bars are explicit: `anaref.py:59–79,156–174`. |
| 3D horizon/cache and FS/QN acceptance | **CORRECT normal driver path.** Steps/output arrays follow the horizon, caches are cleared before B7, and deployed/check FS/QN runs request certification: `s3common.py:100–103,206–249`; `s3ctx.py:142`; `b3run.py:62–68,98–102,171–175`. |
| Alignment, reference realization/version, NaNs, archive lookup | **CORRECT inspected paths, subject to blockers above.** Case identifiers drive alignment; reference identity and hashes are checked; invalid 3D rho has explicit status; 2D archive matching checks mesh. `ana7.py:68–94,203–230`; `b3run.py:104–125`; `ana3d.py:380–392`; `ana2d.py:145–156`. |

All recorded inherited-source hashes and applicable destination hashes matched `PROVENANCE.json`; `resources.py` is explicitly designated rewritten. I compared the consequential reference and 3D edits against their source copies. All Python files parsed, and all **23 generated configurations** matched in memory.

**Cluster mechanics and resource assessment**

| Area | Verdict |
|---|---|
| HEAD staging, unique directory, namespace, GPU partition/preflight, f64/highest precision, paralab outputs | **CORRECT:** `cluster/stage.py:58–119`. |
| Lane cap 3, campaign cap 8, sequential smokes, 45-hour reservation | **CORRECT for the documented workflow:** `cluster/submit.sh:18–40`; `ledger.py:50–75`. |
| Promotion, checksums, `--no-fine`, cleanup | **CORRECT normal workflow:** destination existence checks precede promotion; retained fine files are checksum-verified before cleanup: `cluster/collect.py:27–66,80–90`. |
| Real-size memory/time | **No demonstrated additional blocker.** The 4096² bank alone is about 48 GiB. The pilot exercises both M sizes, Gauss320/Gauss640, fine decode and corner solves (`b2run.py:542–565`). Reference output buffers and 3D construction peaks still require measured smoke evidence. |

**Non-blocking:** the pilot does not reproduce the accumulated B8 fine-floor QR cache (`b2run.py:385–389`). Resource certificates assert H200 timing provenance rather than validating the measured GPU record (`resources.py:114–117`). These limit production extrapolation; they do not independently block a bounded smoke. The documented campaign submission race and nontransactional promotion remain operational limitations.

**Overall verdict: NEEDS-FIX. Required fixes before smoke approval:**

1. Derive B7/B8 and fine-check member coverage from the configuration; reject omitted configured members.
2. Make 3D G-Q reject missing, empty or malformed certificate evidence before evaluating whether the control fired.