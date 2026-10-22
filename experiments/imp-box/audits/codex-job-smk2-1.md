Audited HEAD `56fea6ea7` against SUBMIT audit 7 (`809e395a9`). No files changed or jobs submitted.

1. **Mechanics — CORRECT; cache containment — NEEDS-FIX.** Submission script, batch template and both body hashes are unchanged. The only collector change adds `smk2-*` to early-failure handling.

   Default `code/.jaxcache` is excluded from critical/rest pulls, the `out/cell` archive, and job-level preservation. However, [cell.py:429](experiments/imp-box/cell.py#L429) accepts unrestricted `IMPBOX_JAX_CACHE`, and sets no cache-size limit while enabling caching of every compilation. Archive exclusion does not prevent filling paralab during execution. Enforce a job-local path and explicit byte limit; establish available-space headroom before release.

   `RELEASE.json` still approves only `smk-A2`/`smk-A4`. The proposed new approvals must be committed before submission.

2. **Budget — CORRECT.** Live `sacct` reports both jobs CANCELLED, each with one GPU and **2,002 seconds** elapsed. Count allocation records, not their batch/extern steps again.

   | Reservation | GPU-h |
   |---|---:|
   | Cancelled pair | 1.1122 |
   | New wall limits | 36 |
   | Total exposure | **37.1122** |
   | Remaining under 87 | **49.8878** |
   | Spent + 1.15 × 36 + additional 24-hour failure | **66.5122** |

   Even that contingency leaves **20.4878 GPU-h**. This does not certify the remaining campaign.

3. **Directory isolation — CORRECT; collection — NEEDS-FIX.** Old collectors target exact `smk-A2`/`smk-A4` directories; distinct new job, local-result and archive names avoid collisions. Live `ls` found neither new directory.

   **Actual blocker:** [collect.sh:26](experiments/imp-box/cluster/collect.sh#L26) unconditionally checks empty checksum manifests. Both waiters repeatedly report:
   ```
   sha256sum: .pull-critbig.sums: no properly formatted checksum lines found
   ```
   This prevents `CRITICAL-COMPLETE`, archival and deletion. Handle an empty selected-file set successfully while retaining checksum verification for nonempty sets. Audit 7 missed this bug.

4. **Re-smoke informativeness — NEEDS-FIX; A9.3 justification — WRONG.** Reusing the single-query executable for cohort evaluation and screening is sensible. Persistent-cache reuse remains a hypothesis requiring measured confirmation.

   The new [relative stopping rule](experiments/imp-box/ladder.py#L162) does **not** establish that finer rows cannot meet another target. A row just above a target can cross it after an improvement smaller than 5% of its current error. Two successive outputs also do not bound all remaining refinement error. Remove this prune or establish a target-aware justification before using the smoke to certify coverage and cost.

   **Historical correction:** A4 was cancelled during bank training: its log ends at step **295,000/300,000**, and `.done_bank` is absent. A9’s claim that both banks completed is unsupported; A4 bank time/memory is partial evidence.

**DO-NOT-SUBMIT.**