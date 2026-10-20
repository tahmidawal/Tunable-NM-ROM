**The staged payload is intact, but this is not a clean pre-submission pass.** The target-retry policy and report’s completeness handling remain WRONG. Memory looks feasible; completion within 10 hours is unverified.

Read-only audit of commit `c63a955b3f1f472af76d6f104e496e86fa75ebfd`: no files modified, no submission, no SSH, no GPU execution. Checks included hashes, committed-content comparisons, import-path inspection, allocation arithmetic, and isolated NumPy fault tests.

1. **CORRECT — batch and submission rules.**

   [run.sbatch](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-mechanism/experiments/jcp-mechanism/runs/a1d3/run.sbatch) requests `gpu`, one H200, and 10 hours; no `preempt`. It uses the absolute paralab interpreter, highest precision, x64, and GPU preflight exiting 42. Logs, results, temporary files, and cache are under `/cluster/tufts/paralab/tawal01/jcpmech/a1d3/`.

   `submit.sh a1d3` refuses an existing remote directory, checks the queue before and after submission, and repeats the lane-cap check under an atomic namespace lock. The two mesh invocations are sequential within one job. Actual queue and remote-directory state were deliberately not checked.

2. **CORRECT — staged dependencies and provenance. WRONG — new retry policy.**

   All **15 manifest entries** match. All **10 provenance entries** match both their recorded hashes and the committed bytes. Staged `a1_3d.py` is byte-identical to the recorded commit.

   The repository import chain resolves entirely inside the payload:

   `a1_3d.py → quad3d/offmesh.py → burgers3d-span/common.py + burgers3d-retry/tables.py → paper-b3d/vendor/b3d_common.py`.

   External packages remain dependencies of the cluster venv; their installation was not remotely verified.

   The new [retry loop](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-mechanism/experiments/jcp-mechanism/runs/a1d3/code/experiments/jcp-mechanism/a1_3d.py:334) is **not an adequate recovery policy for the cited XLA red-zone warning**:

   - Injecting a non-finite first target followed by matching finite targets produces `continuum_target_valid=True`.
   - The failure history is recorded, but neither validity nor report labels consume it.
   - Two failed attempts correctly produce an invalid certification target; this is **not** a claim that persistent NaNs pass.

   A finite second evaluation does not establish that a process reporting memory corruption is trustworthy. Require a clean rerun for that failure class, or explicitly invalidate the affected result. The first failed arrays and warning context are also not preserved by this loop.

3. **CORRECT — production configs and references.**

   Both configs resolve relative reference paths through `TASK_ROOT`. Their expected reference hashes match `REFS.json` and `REFS.sha256`; I also hashed all three local source files successfully.

   Both use `Rps=[512,256]`. Adaptive sensitivity covers eight cases at n65 and is disabled at n129. Each width’s historical `expected_rho` keys match the computed historical arms: `tensor`, `dense_upwind`, `gl24`, `lat4096`, `lat32768`. Omitting the new `nodes` arm from historical expectations is correct.

4. **NEEDS-RESTATEMENT — memory appears feasible; 10-hour completion is not established.**

   Static f64 sizes at n129, in decimal GB:

   | Allocation | GB |
   |---|---:|
   | Bank rows, \(512\times127^3\) | 8.390 |
   | Tensor, \(2049\times512\times512\) | 4.297 |
   | Nodes B and D, R′=512 combined | 16.780 |
   | Full nodes Jacobian input columns | 8.390 |
   | One G2c test block, \(131072\times2049\) | 2.149 |
   | One 16-field G3 batch, before DST temporaries | 0.262 |
   | All 64 initial fields | 1.049 |

   G2c avoids the full roughly 33.6 GB test matrix. G3 covers all columns through 16-direction batches. **The nodes contraction still forms the full 8.39 GB column array before chunking the DST**; its memory is not bounded by 16 fields alone.

   These allocations leave substantial room within the requested H200 capacity, including the 0.92 allocator fraction. They are not a measured peak: concatenation copies, FFT workspaces, JIT temporaries, and executable lifetimes add memory.

   The job contains **640 primary rollouts per mesh, 1,280 total**, plus **96 certification rollouts** and **48 adaptive-sensitivity rollouts**. Compilation and adaptive iteration counts remain unknown for this exact payload. Ten hours is a plausible budget, not a verified completion guarantee.

5. **CORRECT — complete-run data sufficiency. WRONG — current report acceptance path.**

   A completed run supplies the necessary evidence:

   - Per-case \(s_j\) and \(d_j(\mathrm{nodes},\mathrm{conv})\) in distance arrays.
   - Per-case finite flags, reason counts, and iteration totals.
   - n65 sensitivity distances and diagnostics.
   - Certification target-valid flags and nodes-reached target-check values.
   - Output coefficients and all internal coefficient states, supporting offline reconstruction and historical comparisons.

   However, [make_report.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-mechanism/experiments/jcp-mechanism/make_report.py:222) still does not enforce the full completion contract. Isolated tests reproduced **R with `complete=False`**, and **R after removing incumbent case 63**. Case coverage is derived only from `nodes`.

   There is also a reachable checkpoint between saving nodes-reached diagnostics and completing n65 sensitivity. Missing sensitivity is treated as a provisional qualification rather than pending required evidence. Labels must wait for required diagnostics, check every required arm’s case identities, and report missing cases explicitly.

Remaining WRONG: non-finite-target retry acceptance policy in `a1_3d.py`; completeness and required-evidence enforcement in `make_report.py`.