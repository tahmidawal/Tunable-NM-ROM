Audited HEAD `bd87f90c3` against audit 7’s SUBMIT baseline. No files changed, jobs submitted, or lab-log writes.

1. **Job mechanics/cache — CORRECT for these pilots.** Submission script, batch template and both supplied body hashes are unchanged. Committed `RELEASE.json` approves the exact `smk2-A2`/`smk2-A4` specifications.

   `code/.jaxcache` is excluded from critical/rest pulls, the `out/cell` archive and job-level preservation. Each cache has a **30-GiB limit**, with a **200-GiB free-space check before each stage**. This closes the unbounded-growth blocker; it does not reserve space against other users. Live free space was not checked under the permitted query restrictions.

   Containment has an explicit `IMPBOX_LOCAL_DEBUG` escape hatch. Neither approved body sets it. Changes beyond A9 include the disclosed A8.1 POD tolerance relaxation, empty-checksum fix and ledger metadata preservation.

2. **Budget — CORRECT.** Live `sacct` confirms both cancelled allocations used one GPU for **2,002 seconds each**. Batch/extern records are not counted again.

   | Exposure | GPU-h |
   |---|---:|
   | Cancelled pair | 1.1122 |
   | New wall limits | 36 |
   | Total | **37.1122** |
   | Remaining under 87 | **49.8878** |
   | Remaining after 15% uplift on new limits plus another 24-hour failure | **20.4878** |

   This approves bounded pilot exposure, not the remaining campaign’s runtime forecast.

3. **Cancelled-job collection/isolation — CORRECT.** Both waiters recorded verified collection, archival, job-level preservation and `collect rc=0`. Live `ls` confirms the old directories are absent, archive checksum manifests exist, and neither new directory exists. Exact job/result/archive names isolate cancelled-smoke collection from the re-smokes. Ledger updates merge metadata under the lock.

4. **Re-smoke informativeness — CORRECT for pilots.** The unsupported relative and target-aware stops are removed; termination logic matches the previously audited baseline, and resource certification no longer accepts `converged-target-aware`. Cohort evaluation and screening reuse the same single-query executable. Full-size stages and controls remain enabled.

   Cache effectiveness and completion within the wall limits remain measurements these pilots must establish. Production coverage/memory certification remains separately gated.

   **Historical claim — WRONG, corrected by the addendum:** A4 did not finish bank training; it stopped at 295,000/300,000 steps without `.done_bank`. Its bank evidence is partial.

**SUBMIT — only the two exact approved re-smokes.**