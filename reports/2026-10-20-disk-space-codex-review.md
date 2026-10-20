# Codex adversarial review of the 2026-10-20 disk-space plan

Codex (`codex exec -s read-only`) reviewed `reports/2026-10-20-disk-space-plan.md` five times as a read-only adversary. It looked for anything that running the plan could lose. Each round's findings were fixed in the plan before the next round. This file keeps every round verbatim, then lists how the last round's findings were handled.

**Outcome.**
- Round 1 was **UNSAFE**: two blockers (verification failures could still reach deletion; Codex audit scratch wrongly marked SAFE) and six majors.
- Round 2 was **UNSAFE** and rounds 3–4 were **FIX-FIRST**, as the archive, verify and delete procedures were hardened.
- Round 5 confirmed every round-4 item fixed and found one remaining defect: the uv prerequisite did not check exit statuses. That defect was fixed after round 5 (see the end of this file) and was not re-reviewed.
- Every round confirmed the core SAFE claim. The 62 non-cone re-sparse commands remove only checked-out copies of committed files from clean worktrees and never delete ignored or untracked data.

## Round 1


**UNSAFE — the plan contains deletion paths that can proceed after failed verification and SAFE-labelled scratch files with unpreserved content.**

Review was strictly read-only. No cleanup command, generator, fetch, commit, or filesystem modification was executed. Git inspections used `GIT_OPTIONAL_LOCKS=0`.

1. **[BLOCKER] Procedures B and C do not reliably stop after verification fails.**

   Both use `ssh … sha256sum -c … && echo VERIFIED`, followed by further commands. Bash exempts the left side of `&&` from `set -e`.

   **Evidence:** this harmless control-flow test:
   ```text
   bash -c 'set -e; false && echo VERIFIED; echo REACHED_NEXT_STATEMENT'
   REACHED_NEXT_STATEMENT
   ```
   Consequently, C can delete the raw tars after a failed checksum check. B can record a false “sha256-verified” commit and reach removal. Comments saying “only after VERIFIED” do not enforce anything. B also leaves pushing the archive record as a comment.

   **Fix:** explicitly abort on every failed copy, verification, commit, and required mirror operation. Make deletion conditional on successful verification, with a fresh local inventory/checksum comparison immediately beforehand.

2. **[BLOCKER] The SAFE audit-scratch deletion includes content absent from Git.**

   The claim that these directories contain only disposable copies is false.

   **Evidence:** inventoried the three directories with Python `Path.rglob`, hashed their files using `git hash-object --no-filters` without `-w`, and queried existence using `git cat-file --batch-check`. **56 files were absent from the local object store**, including:
   ```text
   /tmp/jcp-budget-audit-scratch/round4/audit.py                 3463 bytes
   /tmp/jcp-budget-audit-scratch/round4/floor-recomputed.json    7656 bytes
   ```
   Their blob IDs are respectively:
   ```text
   af1bcdf80e5d0fccfabe628b687e8f7cc26dfa62
   9ac500cdf2b73378290089a283e84ecc05d93aba
   ```
   Reading `audit.py` shows independent report regeneration and numerical checks. A committed audit verdict does not preserve that implementation or its recomputed output. No independent backup was established for these files.

   **Fix:** classify the audit scratch as SAFE-AFTER-STEP; preserve and verify unmatched scripts/results before deletion. Explicitly distinguish regenerable caches from audit evidence.

3. **[MAJOR] Procedure D excludes thousands of coordinator files from verification.**

   It checks only files over 1 MB, then permits deleting the entire scratchpad.

   **Evidence:** a read-only recursive inventory of the specified coordinator scratchpad returned:
   ```text
   files <= 1,000,000 bytes: 2597
   total bytes:             506248136
   ```
   Examples include `FROZEN-STAGEB-dry2d.json`, `front_test.json`, prompts, logs, and smoke-result JSONs. The assertion that unmatched content consists only of the named dry-run directories does not establish preservation of these smaller files.

   **Fix:** inventory every file regardless of size. Archive and verify all content without an independently verified retained copy, including scripts and logs. Confirm the session is finished before taking that inventory.

4. **[MAJOR] Procedure B creates an untracked inventory inside the tree it later tries to remove.**

   `ARCHIVE-FILES.list.z` is created during enumeration, can enumerate itself, and is never committed or otherwise dealt with before removal.

   **Evidence:** reading Procedure B shows only `ARCHIVE-FILES.sha256` and `ARCHIVE-LOCATION.txt` in its `git add`. The read-only check:
   ```text
   git -C worktrees/2026-09-23-burgers-compare-hires \
       check-ignore --no-index ARCHIVE-FILES.list.z
   exit status: 1
   ```
   establishes that the new list is not ignored there. Ordinary removal will therefore refuse even after successful archival. A rerun can also enumerate previously created archive metadata and then overwrite it.

   **Fix:** generate archive metadata outside the source worktree, explicitly exclude that metadata from the payload, and preserve the final inventory with the archive record. Do not resolve this failure by adding blanket `--force`.

5. **[MAJOR] Procedure B cannot checksum an existing untracked directory symlink.**

   NUL-delimited enumeration and `rsync --from0 --files-from` are appropriate, but `sha256sum` follows symlinks rather than recording their link text.

   **Evidence:** `os.readlink`, `Path.is_dir`, and `git ls-files --stage` returned:
   ```text
   worktree: 2026-09-21-burgers-heldout
   path: experiments/burgers-heldout/runs/smoke1/fakearc/output
   target: ../hout
   target is directory: True
   tracked index entry: none
   ```
   This path appears in the supplied archive inventory. Hashing it as a regular file fails. The filesystem scan found six symlinks overall; all resolved within their own worktrees, so no cross-worktree symlink dependency was found.

   **Fix:** distinguish regular files from symlinks in the manifest. Verify regular-file contents, symlink text, and required target coverage separately. Retain `rsync -a` link preservation.

6. **[MAJOR] “No process is using any worktree” cannot be certified from this review environment.**

   **Evidence:** the `/proc/*/cwd` and `/proc/*/fd` scan returned no worktree references, but:
   ```text
   ps -eo pid,comm
   PID COMMAND
     1 codex
     2 bash
     3 ps
     4 head
   ```
   This is an isolated process view, not evidence about all host processes. Furthermore, an idle session can reference a worktree without holding an open file or having its current directory there.

   **Fix:** run the process check in the host PID namespace immediately before cleanup, including open files and mapped files. Confirm relevant sessions have stopped and prevent writers from resuming during archival and deletion. Label the report’s existing observation as a dated snapshot.

7. **[MAJOR] Procedure B’s mirror instructions are incomplete for older lanes.**

   Procedure A supplies commands for seven branches only. B nevertheless instructs every lane to use “Procedure A for this branch (or the lane’s own `sync_github.sh`)”.

   **Evidence:** the supplied `mirrorcmp2.txt` reports `NO-SYNC-SCRIPT` for numerous removal candidates, including `2026-09-23-burgers-compare-hires` and `2026-09-21-heat3d-bank`. Thus the archive record has no concrete backup procedure for those lanes.

   The five explicit base pairs in A did pass the checks performed: all `LOCAL_BASE` commits are ancestors; the deltas contain **1, 1, 1, 1, and 3 commits**; all `REMOTE_BASE` values match `lsremote.txt`. Their largest changed files are below 5 MB. The two direct-push branches also have the stated parents.

   **Fix:** provide a concrete mirror configuration for every affected branch, then require remote tree verification before removal. A matching commit subject alone is insufficient. These remote checks used the supplied snapshot, not a fresh GitHub query.

8. **[MINOR] Space and restoration claims need qualification.**

   **Evidence:** recomputing from `final.json` reproduced:
   ```text
   re-sparse candidates:          62
   re-sparse logical bytes:       1125.380784513 GB
   additional worktree removal:  1205.794468513 GB
   ignored/untracked total:       897.957370525 GB
   ```
   However, the re-sparse estimate comes from blob sizes, not `du`; live allocated blocks for those removal paths total **1125.233025024 GB**. No hardlinked files occurred among those paths, so hardlinks did not invalidate that specific reclaim estimate.

   Procedure B archives every ignored/untracked file with no cross-worktree deduplication, so its “about 696 GB” estimate is not the size of the supplied procedure. Hardlinks also require separate consideration because `rsync -a` does not preserve their sharing.

   Finally, `git worktree add` restores tracked contents, not archived ignored/untracked data or deleted per-worktree configuration. Procedure E’s pattern undo is suitable for the inspected non-cone patterns, but does not constitute complete worktree restoration.

   **Fix:** distinguish logical, allocated, and deduplicated sizes; budget the actual transfer procedure; document restoration of archived files and per-worktree configuration.

The core re-sparse claim survived the checks: all **86 candidate worktrees** had no tracked changes; the only cone-mode worktrees were the two excluded by the plan; generated keep-directory lists covered the inventoried non-cache experiment data. Git 2.43’s implementation skips ignored-directory cleanup for non-cone patterns. [Git sparse-checkout source](https://raw.githubusercontent.com/git/git/v2.43.0/builtin/sparse-checkout.c)

`jcp-fom-pareto` also passed the local checks: **0 ignored files, 0 untracked files**, clean status, and its tip is an ancestor of consolidated. The supplied mirror snapshot identifies that tip. Ordinary worktree removal really does delete ignored files, making the archive defects above consequential. [Git worktree source](https://raw.githubusercontent.com/git/git/v2.43.0/builtin/worktree.c)
## Round 2

**UNSAFE — verification failures now abort, but source changes can still escape verification before deletion; process, mirror, and restoration safeguards remain incomplete.**

Review was strictly read-only. No plan command was executed and no files were created or modified. Git inspections used `GIT_OPTIONAL_LOCKS=0`. All six Bash blocks passed `bash -n` through stdin after substituting the worktree-name placeholder. No filesystem mutation tests were performed.

Line references below refer to [the revised plan](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-20-disk-space-plan.md).

**Round-1 finding status**

| Finding | Status | Evidence |
|---|---|---|
| **1. Verification failures can reach deletion** | **PARTLY FIXED** | B lines 300–303, C line 347 and D lines 365–366 now explicitly exit on verification failure. This fixes the original `&&` defect. The requested fresh source-content verification before deletion remains missing: B compares size/second-resolution mtime, C performs none, and D compares only counts. |
| **2. Audit scratch contains unpreserved evidence** | **FIXED** | The scratch directories are now SAFE-AFTER-STEP, explicitly acknowledge the unmatched audit evidence, and enter D’s archive procedure. D’s remaining implementation defects are identified below. |
| **3. Coordinator verification excludes small files** | **FIXED** | D line 361 inventories every regular file without a size threshold. Line 354 requires confirmation that the coordinator session has finished. |
| **4. Inventory created inside its own payload** | **FIXED** | B puts working lists under `~/disk-plan/$n`, excludes its three record filenames, and commits those records. No current record-name collisions were found in the candidate worktrees. |
| **5. Directory symlink cannot be checksummed** | **FIXED for the identified path** | B separates symlinks from regular files and compares link text remotely. The live ignored/untracked inventory contains `burgers-heldout/.../fakearc/output → ../hout`, resolving inside its worktree. It no longer enters `sha256sum`. General link verification limitations remain below. |
| **6. Process-use claim cannot be certified** | **PARTLY FIXED** | The report labels the earlier observation as a snapshot and B adds mapped files. However, B neither establishes host PID visibility nor rejects incomplete scans, and scans only before archival—not immediately before removal. This review still sees an isolated process namespace. |
| **7. Older lanes lack concrete mirror instructions** | **PARTLY FIXED** | B now supplies a generic mirror procedure and a direct-push fallback. However, it verifies only one record blob and does not establish that the selected mirror base corresponds to the local base. See finding 4. |
| **8. Space/restoration claims need qualification** | **PARTLY FIXED** | B now budgets approximately 898 GB logical, uses `rsync -aH`, and describes restoring archived data and sparse patterns. But lines 5 and 392 still describe sizes as disk usage while re-sparse totals derive from blob sizes. Round 1’s logical/allocated distinction remains unresolved. Complete per-worktree configuration and symlink restoration are still unspecified. |

**SAFE-row recheck**

- **All 62 re-sparse commands:** reconfirmed clean tracked state, non-cone configuration, readable existing patterns, and agreement with the generated candidate commands. The underlying checkout reduction remains acceptable; E’s backup/undo procedure needs the fixes below.
- **`jcp-fom-pareto`:** reconfirmed clean, **zero ignored files**, **zero untracked files**, and tip `73bab9c316f276c5f80678760f61f2230155faa0` ancestral to consolidated. Its data-preservation case passes, but unconditional removal conflicts with the lab-log retention instruction.
- **pip purge:** reconfirmed as cache-only cleanup; the inspected directory contains wheel/HTTP caches. [pip documents that purge clears those caches](https://pip.pypa.io/en/stable/topics/caching/).
- **uv prune:** **not reconfirmed as unconditionally SAFE**. Installed version is `0.11.19`; its behavior does not support the table’s “only entries no environment uses” explanation. See finding 7.
- **Node compile cache:** reconfirmed as regenerable compile-cache content; the path is a real directory containing the Node version-specific cache. [Node describes this as an on-disk compilation optimization](https://beta.docs.nodejs.org/module/module-compile-cache).

**New findings and residual defects in the revised procedures**

1. **[BLOCKER] B, C and D can delete content newer than the verified archive.**

   **Evidence:** B lines 305–306 compare filenames, size and `%Y`—mtime rounded to seconds—not current contents or current symlink targets. A same-size replacement with preserved mtime passes. Files added after this check, while the commit/push/fetch runs, also escape it.

   D line 367 compares only file counts. Replacing one file with another preserves the count; changing an existing file also passes. C deletes the entire directory after checking only the earlier `*.tar` manifest, without a final inventory. The live C directory currently contains exactly eight regular tars, but the procedure does not enforce that invariant.

   **Fix:** stop writers and prevent session resumption throughout archival and deletion. Immediately before deletion, regenerate and compare the complete typed inventory, regular-file hashes and symlink targets. Reject new, missing or changed entries. Delete only the verified scope.

2. **[MAJOR] B’s new process gate fails open and is placed before a potentially ten-hour transfer.**

   **Evidence:** line 278 suppresses every `/proc` read failure with `2>/dev/null || true`. An incomplete or namespace-limited scan can therefore produce the same outcome as “unused.” Line 279 treats a `grep` error like no match. The scan precedes `timeout 36000 rsync`; there is no second scan before removal.

   The cleanliness gate has a similar issue: `test -z "$(git ... status ...)"` can succeed when `git status` fails without stdout.

   **Fix:** require execution in the host process namespace, explicitly handle inspection failures, and separately check `git status`’s exit status. Establish session quiescence and repeat the process check immediately before deletion. A process snapshot alone does not prevent an idle session from resuming.

3. **[MAJOR] D does not verify or document restoration of ten existing symlinks.**

   **Evidence:** the live coordinator directory contains **3,172 regular files and ten symlinks**. Nine links point into `worktrees/2026-10-18-jcp-qualify`; one points back into the coordinator directory:

   ```text
   scratchpad/refjob/ref_refrun_x_ref
     → /tmp/claude-1002/.../2f07cec8-.../scratchpad/reftest/fd
   ```

   D’s `find -type f` excludes all ten links. `rsync -aH` preserves their text, but the remote absolute links do not establish accessible archived targets. Removing `jcp-qualify` also breaks the original external targets.

   **Fix:** archive and verify a typed symlink manifest. Record how every target is preserved and restored, including dependencies on other worktree archives and original absolute paths. B’s line-oriented link manifest should likewise reject unsupported newline/tab names or use an unambiguous encoding.

4. **[MAJOR] The mirror workflow can preserve the checksum file while failing to preserve the intended complete mirror.**

   **Evidence:** the template [sync script](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-jcp-consolidated/experiments/jcp-combined/sync_github.sh:20) fetches a current `PARENT`, but constructs the new tree from `REMOTE_BASE`. If the remote advances beyond the captured base, the new commit can have the newer parent while restoring an older tree plus the local delta. A normal fast-forward push does not prevent that tree-level rollback.

   B lines 321–323 compare only `ARCHIVE-FILES.sha256`. They do not verify `ARCHIVE-LINKS.txt`, `ARCHIVE-LOCATION.txt`, or the rest of the expected mirror tree. A’s line 263 requests a commit-subject check using `ls-remote`, which returns refs and object IDs, not subjects.

   **Reconfirmed:** A’s five local bases are ancestors, with **1, 1, 1, 1 and 3** additional commits; largest changed blobs remain below 5 MB. Both direct-push bases are the local tips’ parents. Remote-base agreement was checked against the supplied snapshot, not fresh GitHub state.

   **Fix:** pin and validate the local/remote base pair, abort if the remote advances, and verify the complete expected filtered tree after pushing. At minimum, independently verify all three archive records using checked object lookups and a pinned fetched commit. Do not infer mirror correctness from its subject.

5. **[MAJOR] C’s durable archive record remains a comment; D has no durable local archive record step.**

   **Evidence:** C line 348 says to record the location in `LAB-LOG.md`, but execution proceeds directly to `rm`. D removes each directory without committing or otherwise verifying a retained record of its archive location and manifests.

   **Fix:** make record creation and verification an executable prerequisite to deletion. Preserve the remote location, manifests, source identity and verification outcome outside both the deleted directory and transient scratch storage.

6. **[MAJOR] `jcp-fom-pareto` is incorrectly exempted from the lane-retention prerequisite.**

   **Evidence:** line 102 calls removal SAFE with no prerequisite. Yet B line 333 acknowledges the canonical lab-log instruction, “Do not delete the lane worktrees or that archive,” and requires explicit override for October lanes. The save-sweep entry does not exempt FOM-pareto.

   **Fix:** classify its removal as SAFE-AFTER-STEP pending the same explicit override. Its clean, fully preserved contents establish technical recoverability, not an exemption from the retention instruction.

7. **[MAJOR] The uv SAFE rationale is false for the installed implementation.**

   **Evidence:** installed uv is **0.11.19**. Its [`prune()` implementation](https://raw.githubusercontent.com/astral-sh/uv/0.11.19/crates/uv-cache/src/lib.rs) removes cached environments and determines unused archives from references inside cache buckets; it does not prove that no external environment uses an entry. The cache currently contains **two `environments-v2` entries**.

   That version also predates the [fix for following symlinks during pruning](https://github.com/astral-sh/uv/issues/19542). I found no top-level symlinks in the inspected `environments-v2` or `archive-v0` buckets, so I am **not** claiming that specific escape condition currently exists. Hardlinks themselves are not broken by unlinking the cache’s name; external symlink dependencies are the relevant distinction.

   **Fix:** remove the unconditional claim. Establish that cached environments are disposable/inactive and that retained installations do not depend on cache symlinks; use a version containing the symlink fix before treating pruning as routine SAFE cleanup.

8. **[MINOR] E can overwrite the original undo state, and a failed list command is mistaken for “previously not sparse.”**

   **Evidence:** line 380 unconditionally truncates `~/disk-plan/sparse-before/$n.txt`. A second execution saves the already-narrowed patterns over the original. E has no explicit abort on backup failure. Line 385 treats an empty output file as evidence that sparse checkout was disabled, although a failed command can also leave it empty.

   **Fix:** save the original patterns and sparse configuration once, refuse overwrite, and check every backup command’s exit status. Record enabled/disabled state explicitly. The existing non-cone pattern lists and proposed `xargs -d "\n"` undo are otherwise suitable for the inspected candidates.

B’s revised `inventory()` correctly handles `grep`’s no-match status without suppressing genuine pipeline failures under `pipefail`, and `rsync -H` addresses sharing among transferred hardlinks. Those improvements do not resolve the deletion and verification gaps above.
## Round 3

**FIX-FIRST — archival checks improved substantially, but process inspection still fails open and restoration remains incomplete.**

Strictly read-only: no plan commands or generators executed; no files or Git state changed. References below are to the [revised plan](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-20-disk-space-plan.md).

| Round-2 finding | Status | Evidence |
|---|---|---|
| **1. Source changes escape verification** | **PARTLY FIXED** | Lines 268–273 now compare complete file lists, freshly computed hashes and link text after recording the archive. This fixes the size/mtime/count checks. Writer exclusion still depends on the defective process guard below. |
| **2. Process gate fails open** | **PARTLY FIXED** | Host-PID check, repeated scans and session-quiescence instructions added. **Line 244 still suppresses every inspection error with `2>/dev/null || true`.** Same-user processes are not necessarily inspectable. At calls such as line 352, the function is inside an `&&` list, so implicit `set -e` protection also cannot be relied upon. Explicitly reject unreadable live relevant processes and check the scan’s exit status. |
| **3. Symlink verification/restoration** | **PARTLY FIXED** | Lines 252–266 now archive and compare link text; newline/tab **filenames** are rejected. Line 396 acknowledges external dependencies but supplies no target-preservation check or restoration procedure for the nine external links and the absolute internal link. Keeping an archive elsewhere does not make an original absolute link resolve. |
| **4. Mirror rollback/incomplete verification** | **FIXED for the five specified updates** | Lines 313–322 check the captured remote head, pin the generated commit’s parent, compare changed-path sets and compare their blob IDs. B now records manifests on main, avoiding the older-lane mirror-template problem. This is code inspection, not fresh GitHub verification. |
| **5. C/D lack durable records** | **FIXED** | Lines 275–287 commit and push all three records and compare their fetched GitHub object IDs; C/D call this before deletion. |
| **6. FOM-pareto retention exemption** | **FIXED** | Classified SAFE-AFTER-STEP, including explicit retention override (lines 12, 367, 525). |
| **7. uv safety claim** | **PARTLY FIXED** | Reclassified and requires checking cached environments plus upgrading. Still does not explicitly check retained installations for symlinks into other pruned cache entries; “nothing uses them” refers to the two cached environments. |
| **8. Sparse undo overwritten/misclassified** | **PARTLY FIXED** | Lines 427–430 preserve the first successful backup and record state explicitly. However, configuration failures remain swallowed by `|| true` (425, 428), potentially recording “unset”; backup publication is not atomic. |

**Open round-1 items:** #1 maps to round-2 #1 above; #6 to #2; **#7 FIXED** by moving archive records to main. **#8 PARTLY FIXED:** logical versus allocated sizes and transfer budgeting are now qualified (lines 5, 369), but B does not save sparse state for every removal candidate. E covers the re-sparse subset and explicitly rejects the two cone-mode worktrees. Their existing configurations and patterns therefore disappear on removal despite restoration step 2 referring to saved copies. Other previously fixed round-1 items remain fixed.

**NEW [MAJOR] — archive upload can overwrite an existing retained backup before verification.** Lines 260–262 accept an existing destination and rsync into it; only the *local* scratch-list existence is checked. If a destination already contains a prior archive, changed source files overwrite its verified bytes before the new checks run. A subsequent failure cannot recover the old backup. Require a fresh, exclusively created remote generation directory and retain existing generations unchanged.

The remaining process-gate defect is **[MAJOR]**; the missing sparse-state preservation is **[MINOR]** but causes actual loss of restoration metadata.
## Round 4

Read-only review of the plan and template; no plan commands, file changes, or Git writes.

Line references are to [the revised plan](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-20-disk-space-plan.md).

| Round-3 item | Status | Evidence |
|---|---|---|
| 1. Source changes escape verification | **FIXED** | Complete file lists, fresh hashes and link text are compared before deletion (287–292), with repeated process checks and explicit session quiescence. |
| 2. Process gate fails open | **FIXED** | Unreadable live own processes now cause refusal except the stated daemon allowlist (247–252). Call sites are plain statements. This remains conditional on the documented session-quiescence requirement. |
| 3. Symlink verification/restoration | **PARTLY FIXED** | D explains all ten links, the three pre-existing dangling links, and their recovery dependencies (416). However, its guard skips **every relative target** and checks absolute targets only by textual prefix (428–430); it does not actually reject all targets outside the allowed roots. |
| 4. Mirror rollback/incomplete verification | **FIXED** | Captured remote head, pinned parent, changed-path comparison and blob-ID checks remain (332–341). |
| 5. C/D lack durable records | **FIXED** | Records are committed, pushed and checked against fetched GitHub objects before deletion (294–306, 406, 435). |
| 6. FOM-pareto retention exemption | **FIXED** | Explicit override still applies to every October lane, including FOM-pareto (12, 387). |
| 7. uv safety claim | **PARTLY FIXED** | **Yes**, row 200 now explicitly checks installed-environment symlinks. But `find -lname "*/.cache/uv/*"` matches link text, not resolved destinations; relative links or aliases into the cache can escape it. |
| 8. Sparse undo overwritten/misclassified | **FIXED for sequential use** | Configuration exit 1 is distinguished from errors; an existing state record prevents replacement; state publication uses a temporary file and rename (258–264). The patterns/state pair is not a concurrency-safe transaction, but E explicitly requires sequential execution. |
| Open round-1 #8: sparse state missing before removal | **PARTLY FIXED** | B now saves state for every candidate (364), including cone-mode worktrees. However, restoration still unconditionally uses `--no-cone` (458), despite cone-mode `sparse-checkout list` containing directory selections rather than non-cone patterns. |
| New round-3: overwriting previous archives | **FIXED** | Timestamped generations and exclusive leaf `mkdir` reject an existing destination before rsync (278–280, 359, 395, 422). |

Open round-1 #1/#6 follow items 1/2 above; #7 remains fixed.

**Remaining defects:** resolve symlink targets relative to each link’s directory before checking containment in D and cache dependencies in the uv row. Otherwise an unrecognised dependency can be left dangling after cleanup. Restore sparse checkout using the saved cone mode; the current command cannot reproduce the two cone-mode checkouts correctly. No remaining overwrite defect was found in remote archive generation.

**Verdict: FIX-FIRST.**
## Round 5

Read-only review completed; no plan commands, modifications, or Git writes.

References below are to [the plan](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-20-disk-space-plan.md); the procedure template agrees.

| Round-4 item | Status | Evidence |
|---|---|---|
| D relative symlink targets | **FIXED** | Lines 428–432 resolve targets from each link’s directory using `realpath -m`, then check containment and reject unresolved/outside targets. |
| uv check matched link text only | **FIXED** | Line 200 now passes symlink paths through `realpath -m`, checking resolved destinations. |
| Sparse restore ignored cone mode | **FIXED** | Lines 459–461 select `--cone`, `--no-cone`, or `disable` from saved state. B restores sparse state before restoring archived data. |

**Remaining defect:** the uv prerequisite (line 200) accepts “prints nothing” without requiring successful traversal and resolution. A failed `find` or `realpath` can produce no matching stdout, incorrectly satisfying the prerequisite and allowing pruning despite an unchecked cache dependency. Require successful `find` and resolution statuses separately, then distinguish grep’s no-match status from errors.

The added explicit pipeline failure handlers in A and D are present (342, 431). No other data-loss defect found in the full re-scan, assuming the documented session-quiescence and sequential-execution requirements.

**Verdict: FIX-FIRST.**
## Disposition of round 5

- **uv prerequisite did not check exit statuses (FIXED after round 5).** The check now writes the symlink list and the resolved paths to files, with `&&` between the steps. It treats grep's no-match status (1) as success and any other failure as failure, and it must exit 0 and print `0` before `uv cache prune` may run. `uv cache prune` stays SAFE-AFTER-STEP in any case.
- The plan assumes two conditions, which Codex accepted: no Claude or Codex session is working in a target while its procedure runs, and procedures run one at a time. Both are stated in the plan next to the guard functions.
