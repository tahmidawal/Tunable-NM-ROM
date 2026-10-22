Audited HEAD `10e0a3987` against SUBMIT audit 7. No files changed, jobs submitted, or lab-log writes.

1. **Job mechanics/cache — CORRECT.** Submission script, batch template, and both body hashes are unchanged. Committed `RELEASE.json` approves the exact `smk2-A2`/`smk2-A4` specifications.

   The cache is restricted to `code/.jaxcache`, excluded from critical/rest pulls, the cell archive, and job-level preservation. It now has a 30-GiB limit per job and a 200-GiB free-space gate at stage startup. This closes the unbounded-cache blocker; concurrent activity elsewhere can still exhaust the shared filesystem. Live free space was not independently checked under the permitted query restrictions.

   Changes beyond A9 include the disclosed A8.1 POD tolerance relaxation, empty-checksum collection fix, and GPU-type ledger enrichment.

2. **Budget — CORRECT.** Live `sacct` confirms both cancelled allocations used one GPU for **2,002 seconds each**; batch/extern records are not counted again.

   | Exposure | GPU-hours |
   |---|---:|
   | Cancelled pair | 1.1122 |
   | New wall limits | 36 |
   | Total | **37.1122** |
   | Remaining under 87 | **49.8878** |
   | Remaining after 15% uplift on new limits plus another 24-hour failure | **20.4878** |

   This bounds these pilots, not the remaining campaign.

3. **Cancelled-job collection/isolation — CORRECT.** Both waiters now report critical verification, archival, job-level preservation, deletion, and `collect rc=0`. Live `ls` confirms both old job directories are absent, archive checksum manifests exist, and neither new directory exists. Exact job names keep old collectors isolated from the re-smokes.

4. **Re-smoke informativeness — NEEDS-FIX.** Single-query executable reuse is appropriate, and cache effectiveness can be measured by these pilots. However, the revised stopping rule still treats an unsupported extrapolation as convergence.

   [ladder.py:180](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/ladder.py:180) assumes twice the latest step difference bounds future improvement without checking that assumption. An isolated probe of the actual predicate gives:

   | Quantity | Value |
   |---|---:|
   | Previous error | 0.0112 |
   | Current error | 0.011 |
   | Step difference | 0.0002 |
   | Next error | 0.009183203125 |
   | Reference allowance | 0.00001 |

   These values follow the smooth fourth-order error expansion $e(h)=0.009+0.0038625h^4-0.0018625h^5$ at $h=2,1,1/2$. The absolute stop does **not** fire; the new stop **does**, dropping the next rung that meets the 1% target.

   This is a counterexample to the stopping justification, not an observed PDE result. Restricting the rule to explicit steppers does not resolve it. `resources.py:67` nevertheless accepts `converged-target-aware` for certification.

   **Required:** remove this stop, establish a validated remainder bound, or treat it as heuristic truncation with incomplete-search labels and exclude affected searches from coverage/cost certification. Calling it “conditional” in DESIGN alone does not implement that distinction.

**DO-NOT-SUBMIT.**