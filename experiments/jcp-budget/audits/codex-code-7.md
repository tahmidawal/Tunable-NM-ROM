**Overall verdict: NEEDS-FIX.** Audited HEAD `bd978fe5b3158d044fb1679e4269dc04a16d1b44`. Three blocking issues remain: two incomplete Round 6 fixes and a failure path in the new submission lock.

Paths below are relative to [experiments/jcp-budget](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget). Findings distinguish source inspection and bounded CPU reproductions from untested real-size GPU behavior.

**Verification of Round 6’s required fixes**

| Required fix | Assessment | Evidence |
|---|---|---|
| Validate configured cases and required check inventories independently of produced records | **NEEDS-FIX — partially corrected.** The cited 2D missing-Q-corner example now fails; 3D B7 now expects configured cases. However, 3D quadrature checks still enumerate produced coefficient inventories without validating them against configuration. | `ana2d.py:182–205,469–504`; `ana3d.py:76–118,122–153,379–429`. See blocker 1. |
| Reject incomplete/non-finite 2D G-Q evidence and validate registered thresholds independently | **NEEDS-FIX — partially corrected.** NaNs and truncated schedules now fail. Expected population still comes from another produced file, and thresholds remain supplied by the certificate file itself. | `ana2d.py:538,564–571`. See blocker 2. |

**Blocking findings**

**1. NEEDS-FIX — 3D G-B4 still accepts incomplete coefficient evidence.**

`tag_acceptance()` validates the **JSON rows’** case population, but no longer compares that population with the corresponding coefficient-file inventory (`ana3d.py:106–118`). Subsequently:

- `gb4()` iterates only the check NPZ’s surviving cases (`ana3d.py:132–138`).
- `star_check()` likewise iterates produced check cases (`ana3d.py:148–153`).
- Required corner names still come from produced `b['corners']` and `b['form']`, rather than an independent expected inventory (`ana3d.py:470–473`).

A bounded fixture executed the actual extracted functions with the actual smk2 configuration:

| Evidence | Result |
|---|---|
| Configuration and JSON certificates contain cases `[0,1]`; check NPZ contains only case `[0]`; surviving trajectories agree | `G_B4['Q'] = {'worst': 0.0, 'passed': True}` |
| Both JSON and NPZ omit case `1` | Certificate acceptance correctly rejects case `0` |

Thus the joint omission is repaired, but a partial coefficient dump can still remove a verification obligation. The smoke consumes this result directly at `smoke_accept.py:124–126`.

**Required:** independently validate every required 3D corner/check/star role, its JSON case inventory, its NPZ case inventory, and the coefficient array dimensions against configuration before calculating gates. Reject missing, duplicate or inconsistent entries.

**2. WRONG — 2D G-Q still permits false “fired” results.**

Two remaining dependencies defeat the requested independent validation:

- `n_ = min(2, len(ids))` uses `CTRL_b2.npz`’s produced case IDs, not configured control cases (`ana2d.py:538,566`).
- Comparisons use `zg['gamma_bar']` and `zg['eta_bar']` without checking the registered values (`ana2d.py:568–569`).

Executing the exact revised acceptance branch gave:

| Fixture | Observed result |
|---|---|
| Complete `(2,50)` arrays containing NaNs | Rejected — corrected |
| Truncated `(2,1)` arrays | Rejected — corrected |
| Produced control IDs reduced to one case; finite `(1,50)` certificates | `evaluable=True`, `fired=True` |
| Complete `(2,50)` arrays, every certificate `1e-10`, both stored thresholds changed to zero | `evaluable=True`, `fired=True` |

In the final fixture, **both non-certified fractions are actually zero** under the registered thresholds: LSPG `1e-3` and GAL `1e-8` (`b2run.py:37`; `qcert.py:20`). Nevertheless, `smoke_accept.py:88` accepts the control.

**Required:** derive the expected control population from configuration, preserve/check certificate case identities, and independently enforce the registered thresholds. Retain the new finite-array and schedule-length checks.

**3. WRONG — the locked campaign-cap check fails open when `squeue` fails.**

The remote shell at `cluster/submit.sh:36` executes:

```bash
n=$(squeue -u tawal01 -h | wc -l)
```

That inner `bash -c` has no `pipefail` or explicit scheduler-query status check. If `squeue` fails without stdout, `wc -l` returns zero successfully; the script proceeds to `sbatch`.

An isolated shell reproduction using only mock functions confirmed that a failing queue query reaches the mock submission and exits successfully. No scheduler commands or network calls were made.

The flock correctly serializes cooperating submissions, but it does not make an unsuccessful count trustworthy. This can violate the mandatory campaign cap despite holding the lock.

**Required:** fail closed on a failed queue query inside the shared lock, then count its successfully obtained output and submit while still holding that lock.

**Re-audit of the remaining requested areas**

Here, **CORRECT** means no additional smoke blocker found in the inspected implementation; it does not certify GPU execution.

| Area | Assessment and file:line evidence |
|---|---|
| Möbius effects, closure, floor identity, ideal end, \(U_\star\), \(\chi\), \(\chi^\star\), numerical background | **CORRECT.** `ana2d.py:246–300`; `ana3d.py:189–245`. M participates in the 2D candidate/pair universe; 3D F uses its tight-end background. |
| Ties, dominance, incomplete candidates, pair labels | **CORRECT.** `ana2d.py:358–428`; corresponding 3D decision code follows the narrowed F/S contract. Quadrature verification remains subject to blocker 1. |
| Growth, stability, sampling uncertainty, front order | **CORRECT inspected formulas.** `ana7.py:139–195,291–307,355–423`: adjacent-window growth, uncertainty-aware comparisons, common populations and eligibility tiers are present. |
| C-A amplitude and C-E closed form | **CORRECT.** `b2run.py:433–452`; `ana7.py:302–313`. Alias amplitude incorporates the actual rejection threshold; energy checks use the raw balance before normalization. |
| C-F, C-I, C-S, C-T, C-Q, C-score | **CORRECT inspected constructions/predicates.** `ana2d.py:317–353,516–555,574–589`; `ana3d.py:261–290,496–534`. G-Q remains blocker 2. |
| C-R1 and G-R1/G-R2 matching and bars | **CORRECT for configured smoke ladders.** `anaref.py:59–79,156–174,198–208`. G-R2 matches mesh, timestep and purpose and separately checks shared ICs. |
| G-B3/G-B5/G-B6 | **CORRECT.** Archive reproduction: `ana2d.py:307–312`; LS acceptance: `budcommon.py:133–146`; logical switch assertions: `budcommon.py:171–179`. G-B4 remains blocker 1. |
| `budcore` output count and split initialization | **CORRECT.** All eight CPU manufactured cases passed bitwise equivalence, horizon-prefix and finite/distinct-tail tests. Scale is preserved through initialization replacement at `budcommon.py:182–207`. |
| M=2048 and quadrature roles | **CORRECT.** Separate settings and role selection: `b2run.py:150–154,209–234`; `budcommon.py:185–190`. |
| Streamed LS QR | **CORRECT by inspection.** `budcommon.py:95–157`: row ordering matches flattened interior ICs, prefixes are preserved, local \(Q_b^\mathsf TU_b\) is applied directly, and global conditioning/orthogonality determine acceptance. |
| B7 certifiers, anchors, q+/q++ | **CORRECT driver wiring.** `b2run.py:254–309`; `b3run.py:88–125`. Adaptive anchors require `nfail == 0`; 3D FS/QN use their own certificates. Inventory validation remains incomplete as described above. |
| B8 decode, normalization, stencils, band and reference identity | **CORRECT inspected paths.** `budcommon.py:215–253`; `b2run.py:323–452`; `ana7.py:68–94`. Both observation routes use the same reference realization and route-specific normalization. |
| Reference horizon, stage buffers, extraction, window judges, fine-file split | **CORRECT.** `ref/refcore.py:102–238,251–278`; `ref/refrun.py:102–239`. The 3D shared-node extraction uses the required interior-index offset. |
| 3D horizon and caches | **CORRECT normal driver path.** `s3common.py:180–227`; `s3ctx.py:142`; `b3run.py:171–175`. Query and certificate caches are cleared when the horizon changes. |
| HEAD staging, unique directories, GPU partition/preflight, f64/highest, paralab outputs | **CORRECT.** `cluster/stage.py:58–119`. |
| Lane cap, sequential smokes, 45-hour reservation | **CORRECT normal path.** `cluster/submit.sh:13–26`; `ledger.py:50–75`. Campaign-cap failure handling is blocker 3. |
| Checksummed collection, no-overwrite promotion, `--no-fine`, cleanup | **CORRECT normal workflow.** `cluster/collect.py:27–66,69–90`. Retained fine files must verify before cleanup. |
| Real-size memory/time | **No additional demonstrated blocker.** `b2run.py:542–565` exercises the combined 4096² bank, both M settings, Gauss320/Gauss640, fine bank and solves. Reference buffers, LS QR and 3D construction peaks still require measured smoke evidence. |

All inherited source hashes and applicable destination hashes matched `PROVENANCE.json`. I compared the consequential inherited edits in `s3common`, `s3ctx`, `refcore`, `verdict`, `pscore` and `cohorts`. All **23 generated configurations** matched committed JSONs using an in-memory writer replacement. Python parsing and shell syntax checks passed.

**Non-blocking findings**

- **Cap-refusal reservation handling:** `set +e` does not disable the ERR trap. Exit 75 first marks the reservation `AMBIGUOUS`; the subsequent `release` only releases `PENDING_SUBMIT` entries (`submit.sh:33–39`; `ledger.py:80–83,105–108`). An inert shell reproduction confirmed this. It conservatively retains budget but makes the “reservation released” message false.
- The memory pilot omits the accumulated fine-floor QR cache (`b2run.py:385–389`), and resource certificates assert rather than verify H200 timing provenance (`resources.py:113–115`). These limit production extrapolation.
- Current A4 explicitly defers B8 3D and its C-A/C-E controls; they are deferred functionality, not implemented coverage.

No files changed, no jobs submitted, no GPU work performed, and no scratch files created. The CPU test’s filesystem-writing statements were removed in memory. LAB-LOG.md was untouched; Git status still shows only the pre-existing `HEARTBEAT.json` modification.

**Overall verdict: NEEDS-FIX. Required blocking fixes:**

1. Validate 3D coefficient and certificate inventories independently against configured roles and cases.
2. Validate 2D G-Q’s configured population, case identities and registered thresholds independently.
3. Make the campaign-cap query inside the flock fail closed on scheduler-query failure.