**READY-FOR-SMOKE.** Audited HEAD `3f88a2d96`. The audit-8 blocker is fixed; I found no remaining blocker to the configured real-size cluster smoke.

Paths below are relative to `experiments/jcp-budget`. **CORRECT** means source inspection and bounded CPU checks found no smoke blocker; real-size memory, runtime, and numerical gates still need smoke evidence.

**Audit-8 verification**

| Item | Assessment | Evidence |
|---|---|---|
| G-Q certificate case identities | **CORRECT.** Producer saves the certificate population independently; acceptance requires the exact configured first-two-case sequence, uniqueness, expected shapes, and finite certificates. | `b2run.py:534`; `ana2d.py:562–575` |
| Registered G-Q thresholds | **CORRECT.** Acceptance uses fixed LSPG/GAL bars rather than trusting stored threshold fields. | `ana2d.py:571–575` |
| Common-time chi decomposition | **CORRECT.** `chi`, `chi_star`, and background use the same time; the separate `chi_star_peak` preserves the peak statistic. | `ana2d.py:292–301`; `ana3d.py:310–323` |

I executed the actual G-Q acceptance branch in memory. Correct evidence fired; absent, duplicate, unrelated, reversed, short, and NaN evidence did not. Spoofing stored thresholds did not change the registered comparisons.

**Re-audit**

| Area | Assessment and file:line evidence |
|---|---|
| Möbius effects, factorial closure, floor identity, ideal dynamics and \(U_\star\) | **CORRECT.** Full-factorial bookkeeping, per-time identities, uncertainty replays, and exclusion of quadrature-unverified ideal ends are implemented. `ana2d.py:137–148,246–310`; `ana3d.py:266–329`. |
| Candidate universe, ties, dominance, incomplete cases and pairs | **CORRECT.** M participates in the amended 2D contest and pair design; completeness and population thresholds apply. 3D F×S remains explicitly undefined. `b2run.py:44–53`; `ana2d.py:359–427`; `ana3d.py:375–435`. |
| Growth, stability, sampling adequacy and front order | **CORRECT inspected rules.** Adjacent-window growth, fixed-initial-energy stability, uncertainty-aware route comparisons, common populations and resolved reversals are present. `ana7.py:139–195,291–423`. |
| C-F, C-I, C-S, C-T and C-Q | **CORRECT inspected constructions and predicates.** Comparator validity, converged solver anchors, injected-floor checks and smoke-specific “at least one” predicates are wired. `b2run.py:461–515`; `ana2d.py:318–354,517–555`; `ana3d.py:339–372,579–597`. |
| C-A and C-E | **CORRECT.** Alias amplitude includes the actual uncertainty-adjusted rejection threshold; energy control compares the raw balance against its closed form. `b2run.py:433–452`; `ana7.py:302–313`. |
| C-R1, C-score and amended C-R2 | **CORRECT.** First-order references are rejudged with prerequisites; scoring controls exercise identity errors. `anaref.py:198–208`; `ana2d.py:576–590`; `ana3d.py:607–614`; `DESIGN.md:537`. |
| G-R1/G-R2 matching and bars | **CORRECT for configured smoke ladders.** G-R2 matches final levels by mesh, step and purpose, requires the finest match and shared IC, and applies the registered per-time bars. `anaref.py:59–79,156–174`. |
| `budcore` output-count change | **CORRECT, CPU-tested.** All eight form/scheme combinations passed bitwise equivalence, horizon-prefix equality, and finite/distinct extended outputs. `budcore.py:118,168,228,342–349`; `test_budcore.py:39–64`. |
| Split initialization and scale preservation | **CORRECT.** LS replaces coefficients while retaining the initialization scale passed into evolution. `budcommon.py:182–207`. |
| M=2048 and qinf/qcheck roles | **CORRECT.** M selects a separate setting; check tables replace qinf tables after the initial runs. `b2run.py:150–154,209–236`; `budcommon.py:162–188`. |
| Streamed LS QR and G-B5 | **CORRECT, small CPU fixture tested.** Row ordering and prefix columns match the interior flattening; direct block \(Q^\mathsf{T}U\) handles locally rank-deficient blocks. Condition and orthogonality checks govern acceptance. `budcommon.py:95–157`. |
| G-B3/G-B4/G-B6 | **CORRECT.** Archive lookup filters mesh/configuration; every required Q corner and star replay is checked; logical switch differences are asserted. `ana2d.py:157–176,182–220,460–489`; `budcommon.py:171–179`. |
| B7 certification, anchors and q+/q++ convergence | **CORRECT.** LSPG/GAL certifiers differ appropriately; adaptive anchors require zero failed steps; successor-rule convergence affects qualification. `b2run.py:254–309`; `budcommon.py:201–207`; `ana7.py:246–269`. |
| B8 decode and reference handling | **CORRECT.** Fine decode uses Bank(1024); manifest, cohort, horizon, metric and fine-file identity are checked. Normalizers, central-difference stencils, band mask and alias field match the specification. `b2run.py:323–452`; `budcommon.py:215–253`. |
| Reference horizon and extraction | **CORRECT.** Output/stage buffers and judges are horizon-aware; fine files are separated. Small synthetic checks confirmed bitwise shared nodes in both dimensions, including the 3D interior-index offset. `ref/refcore.py:102–238,251–278`; `ref/refrun.py:102–239`. |
| 3D horizon, caching and solver-specific acceptance | **CORRECT normal driver path.** Horizon changes clear query/certificate caches; adaptive and FS/QN acceptance follow different registered rules. `b3run.py:88–125,171–175`; `s3common.py:180–249`; `s3ctx.py:142–169`; `ana3d.py:170–195`. |
| Silent-failure defenses | **CORRECT inspected paths.** Configured inventories, explicit case lookup, reference-realization binding and certificate finiteness checks cover the principal alignment risks. `ana3d.py:85–195`; `ana7.py:68–94,203–234`; `ana2d.py:460–505,562–575`. |
| Cluster staging and precision | **CORRECT.** Unique attempt directories, committed HEAD blobs, paralab outputs, GPU partition/preflight, x64 and highest precision are enforced. `cluster/stage.py:58–119`. |
| Reservations and submission caps | **CORRECT inspected workflow.** Time-limit reservations enforce the 45 GPU-hour cap; lane cap 3, sequential smokes and account cap 8 are checked, with the latter inside the shared lock. Scheduler failure fails closed. `ledger.py:50–75`; `cluster/submit.sh:13–50`. |
| Collection, promotion and cleanup | **CORRECT inspected workflow.** Checksums, no-overwrite promotion, partial-collection refusal and remote verification of retained fine files precede cleanup. `cluster/collect.py:27–90`. |
| Real-size memory/time | **No demonstrated smoke blocker.** The pilot exercises the combined 4096² bank, both M settings, Gauss320/Gauss640 tables, fine bank and solves; acceptance requires a measured peak within 120 GiB. Reference buffers, QR and 3D construction peaks remain empirical smoke questions. `b2run.py:543–566`; `smoke_accept.py:108–114`. |

**Non-blocking limitations**

- **NEEDS-FIX before relying on production resource certificates:** `resources.py:113–115` asserts H200 timing provenance without verifying the recorded smoke hardware.
- **Production memory coverage limitation:** the pilot does not reproduce the accumulated fine-floor QR cache used by B8 (`b2run.py:385–389,543–566`).
- **Declared deferred coverage:** B8 3D and its C-A/C-E controls remain deferred under A4; this is not implemented coverage.

All 27 inherited source hashes and 26 applicable destination hashes matched provenance. I compared the consequential inherited edits, verified all 23 generated configurations in memory, and checked Python parsing and submission-shell syntax. The LS fixtures agreed with direct least squares within \(3.1\times10^{-16}\).

No files changed, jobs submitted, GPU work performed, or scratch created. LAB-LOG.md was untouched. Git status still shows only the pre-existing `HEARTBEAT.json` modification.

**Overall verdict: READY-FOR-SMOKE.**

Required fixes — blocking only:

1. None.