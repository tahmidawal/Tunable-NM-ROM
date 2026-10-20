**Overall verdict: NEEDS-FIX.** Audited commit `8b440dd30`.

No files changed, jobs submitted, GPU used, or scratch created. CPU checks used the prescribed interpreter with bytecode writing disabled. The core test’s file-writing statements were removed **in memory**.

Paths below are relative to [experiments/jcp-budget](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget).

**Verification of audit-2’s eight blocking fixes**

| # | Prior required fix | Status | Evidence |
|---|---|---|---|
| 1 | Eligibility tiers and T=1 cross-check roster | **CORRECT** | Horizon-specific designated populations, suspended-case exclusion, and tier-aware growth/ranking/front labels are implemented: `anaref.py:34,235–265`; `ana7.py:113–124,163–165,369–373,414–418,587–593`. |
| 2 | Mandatory prefix checks and reference identity binding | **NEEDS-FIX — partial** | Missing/incomplete T=1 prefixes now prevent eligibility (`anaref.py:203–218,249–265`). Binding checks the NPZ hash and final identifiers, but omits the recorded JSON hash (`ana7.py:89–91`). See blocker 1. |
| 3 | Preserve invalid 3D rho through serialization | **CORRECT** | Driver persists `ok`/`trivial`/`invalid` statuses; analysis rejects invalid samples: `b3run.py:104–127`; `ana3d.py:380–392`. |
| 4 | Tight-end background for 3D F | **CORRECT** | F uses `F['S'] − PR`; the numerical background follows from subtracting chi-star: `ana3d.py:210–220`. |
| 5 | Complete independent fine verification and coverage | **NEEDS-FIX — partial** | Errors, normalizers, energy, dissipation, extrema, band errors, floors, alias fields and measured C-E are now recomputed (`ana7.py:475–533`). Mandatory local G-R3 evidence remains unenforced, and coverage still depends on the produced records. See blocker 2. |
| 6 | Certified 3D comparators and per-spec/per-case acceptance | **CORRECT** | C-Q comparator certification, distinct C-T tags, required FS/QN certificates and per-case acceptance are implemented: `b3run.py:135–144`; `ana3d.py:77–95,455–476`. A CPU fixture rejected a missing certificate and kept different specifications separate. |
| 7 | Missing 3D B2/B7 outputs | **CORRECT** | Sum/RSS comparisons, primary E0 W1 error/eligibility, and long-horizon floor are present: `ana3d.py:198–203,365–372`. |
| 8 | Tooling provenance | **CORRECT** | Every recorded source hash and every applicable final destination hash matched, including all three cluster tools. `resources.py` is explicitly recorded as rewritten. |

**Remaining smoke blockers**

1. **WRONG — reference uncertainty metadata is neither fully bound nor independently checked.**

   [ana7.py:89](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/ana7.py:89) compares the anaref NPZ hash and final-level names, but ignores `rec['sha256']['.json']`. A replacement JSON plus its matching manifest can therefore accompany stale eligibility evidence.

   Separately, [anaref.py:96](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/anaref.py:96) recomputes window judges but checks only their **outcome**, not their uncertainties. Downstream growth, ranking and sampling decisions read the stored window `eps_ref` (`ana7.py:187–194,280–290`).

   Small in-memory fixtures confirmed:

   ```text
   Changed JSON, unchanged NPZ/final IDs: loader accepted
   Stored window eps=999, recomputed eps=1e-6:
       whole_agree=True, window_agree=True
   ```

   **Required:** bind both reference file hashes to anaref evidence. Compare stored window uncertainty values against recomputation, including shape and finite/None handling, or use the recomputed values directly downstream.

2. **NEEDS-FIX — smoke acceptance still permits missing independent fine evidence.**

   [anaref.py:270](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/anaref.py:270) retains:

   ```python
   c['G_R3_flag'] is True and c['G_R3_local'] is not False
   ```

   Missing local fine files produce `None`, so this passes. The new `G_R3_local_all` field correctly rejects missing evidence, but [smoke_accept.py:52](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/smoke_accept.py:52) still consumes only `G_R3`. This directly affects **smk2**, which has no separate B8 fine-recomputation gate.

   Reproduced:

   ```text
   G_R3_flag=True, G_R3_local=None
   smoke summary=True, local_all=False
   ```

   Fine recomputation also skips missing member files, rows marked `has_ref=False`, and absent LS coefficients (`ana7.py:455–469,511–513`). Its final success test does not compare against an independently constructed expected coverage set (`539–540`).

   **Required:** require `G_R3_local_all` for smoke acceptance. Derive expected fine checks from the smoke configuration and produced reference inventory; require all expected member/case/array checks, including accepted LS coefficients. The intentionally absent dev6 case-1 reference need not be required.

3. **NEEDS-FIX — C-R1 and G-Q acceptance still trusts stored control results.**

   DESIGN §7 requires smoke predicates to be recomputed from raw outputs.

   - [ref/refrun.py:297](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/ref/refrun.py:297) computes the X1 ladder but retains only the judge’s outcome, orders and boolean checks. Its fields and stage evidence are discarded. `anaref.py:195–199` then constructs C-R1 from those stored booleans.
   - `ana3d.py:477` copies the driver’s G-Q result directly.
   - `ana2d.py:515–517` trusts stored `fired` flags. Its `all(...)` also succeeds if **no** `GQ_budget0` entries exist, provided the wrong-table flag is true.

   These paths cannot establish the advertised independent control acceptance.

   **Required:** retain sufficient X1 evidence to rerun its judge offline. Recompute G-Q’s non-certified fraction from certificate evidence, explicitly requiring both 2D forms and the prescribed 3D control. Missing control evidence must fail.

4. **WRONG — the warning gate ignores stderr.**

   [smoke_accept.py:38](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/smoke_accept.py:38) reads only `logs/*.out`, while `cluster/stage.py:94–95` directs stderr into separate `*.err` files. Consequently, the “no captured-constants warning” gate can pass without inspecting the stream containing the warning.

   **Required:** inspect both streams for disqualifying warnings/errors while retaining positive completion and backend checks.

**Re-audit of numerical implementation and specification coverage**

“CORRECT” below means source inspection and the stated CPU checks found no additional smoke blocker; it does not certify real-size GPU behavior.

| Area | Status and evidence |
|---|---|
| Möbius effects, closure and floor identity | **CORRECT:** `ana2d.py:207–280`; `ana3d.py:155–208`. Identity checks use run-valid corners separately from quadrature-qualified effects. |
| U-star, chi/chi-star, numerical background | **CORRECT:** four ideal-family runs, uncertainty sum and signed interactions are implemented; corrected 3D F background noted above. `ana2d.py:251–280`; `ana3d.py:187–220`. |
| Tie, dominance, incomplete and pair rules | **CORRECT:** candidate completeness, complete-case thresholds, eligible-case pair denominators and minimum populations are implemented. `ana2d.py:331–398`; `ana3d.py:273–335`. |
| G-B4 withholding and ideal-family failure | **CORRECT:** failed corner checks remove their effects; failed ideal-family checks remove U-star-dependent decisions. `ana2d.py:162–195,443–458`; `ana3d.py:98–132,436–447`. |
| `budcore` output-count change | **CORRECT:** diff is confined to output-count plumbing. All eight LSPG/GAL × BE/CNR/BDF2/CN CPU cases passed bitwise original equivalence, horizon-prefix equality and finite/distinct tails. |
| Split initialization, scale and M | **CORRECT:** scale is preserved when replacing coefficients; M=2048 selects its own setting and rule tables. `budcommon.py:181–207`; `b2run.py:150–158,211–241`. |
| Streamed LS QR | **CORRECT by inspection:** direct block Q-transpose application, prefix columns, interior flattening, global conditioning and orthogonality acceptance. `budcommon.py:96–157`. |
| G-B6 | **CORRECT for logical switches:** asserted before running corners; table guard runs before queries. `b2run.py:193–197`; `budcommon.py:167–190`. |
| B7 certificates, anchors and Q2 convergence | **CORRECT:** LSPG gamma/GAL eta, gcert, tight/tighter acceptance, qp/qpp anchors and convergence checks are wired. `b2run.py:255–309`; `ana7.py:212–260`. |
| 3D horizon/cache/certification | **CORRECT:** horizon-sized steps/outputs, `ad`/`adg`, query/certifier cache clearing and FS/QN acceptance are consistent. `s3common.py:100–103,206–249`; `s3ctx.py:142`; `b3run.py:171–177`. |
| B8 decode and diagnostics | **CORRECT formulas:** fine Bank(1024), interior energy, central-difference dissipation, band mask, alias amplitude and C-E closed form. `budcommon.py:215–253`; `b2run.py:363–452`. Independent acceptance limitations are blockers 1–2. |
| C-I, C-F, C-Q, C-S, C-T and C-score | **CORRECT principal predicates:** `ana2d.py:287–325,470–478,486–537`; `ana3d.py:235–269,455–487`. C-A/C-E formulas are implemented at `ana7.py:293–309`; metadata/coverage caveats remain above. |
| Reference stepping and extraction | **CORRECT:** inherited-source diffs preserve arithmetic while generalizing horizon/buffers; 3D extraction uses `s*j−1`. `ref/refcore.py:102–238,251–278`; `ref/refrun.py:103–142,154–239`. |
| G-R1/G-R2 and reference refinement | **CORRECT for configured smoke ladders:** final-level G-R2 matching includes L/dt/purpose, exact t0 and stated evolved-time bars. Whole-horizon judge drives refinement. `anaref.py:59–79,158–174`; `ref/refrun.py:185–207`. |
| Growth, stability, sampling and front order | **CORRECT principal decision formulas:** `ana7.py:138–195,280–418`, subject to blocker 1’s uncertainty integrity. |

**Cluster mechanics and resources**

**CORRECT for the normal staged smoke workflow:** committed-HEAD staging, unique attempt directories, paralab outputs/cache, GPU partition/preflight, x64/highest precision, lane cap 3, campaign cap 8, sequential smokes and atomic 45-hour reservations are implemented (`cluster/stage.py:60–125`; `cluster/submit.sh:18–39`; `ledger.py:50–75`). Promotion checks destinations and verifies copied hashes; `--no-fine` cleanup verifies retained fine files before deletion (`cluster/collect.py:34–66,81–90`).

All **23 configurations** regenerated identically in memory. Python syntax checks passed.

**Non-blocking resource limitations:** no OOM or excessive runtime was demonstrated. The 4096² rotated bank alone is approximately 48 GiB. The pilot exercises both test-space sizes and Gauss320/Gauss640 roles (`b2run.py:536–559`), but does not reproduce the accumulated B8 fine-floor QR cache (`385–389`). Reference output buffers, QR workspaces, 3D table construction and offline fine QR remain measured-smoke questions. Resource certificates remain estimates; their H200 designation is asserted rather than validated against the measured GPU record (`resources.py:114–117`).

**Non-blocking, outside the immediate smoke:** T=1 configurations specify `fine=False`, producing `G_R3=None`, but eligibility currently requires `G_R3_flag is True` (`make_configs.py:116–117`; `ref/refrun.py:237`; `anaref.py:261–265`). Thus T=1 remains non-rankable even with passing prefix/cross-check evidence. Address before that deferred production phase. The disclosed campaign submission race and nontransactional promotion also remain operational limitations.

**Overall verdict: NEEDS-FIX. Required fixes before smoke approval:**

1. Bind reference JSON hashes and verify window uncertainty metadata.
2. Require local G-R3 evidence and explicit expected fine-check coverage.
3. Independently recompute C-R1/G-Q acceptance and reject missing controls.
4. Inspect stderr in smoke warning/error acceptance.