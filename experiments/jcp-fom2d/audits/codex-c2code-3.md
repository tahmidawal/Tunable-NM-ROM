**READY to submit `runs/smkc2`.** Audited HEAD `a7be0c8f77a0e79368b639a5dec4f39cc2fef7c8`. No remaining smoke-blocking regression found.

Paths below are relative to `experiments/jcp-fom2d/`.

| Round-2 finding | Status | Evidence |
|---|---|---|
| Surviving stage-0 closure/output | **FIXED** | `c2time.py:233` deletes both; warm-up outputs are deleted at `:277`. |
| Release preserves stage 2 and controls | **FIXED** | `c2time.py:313–328` retains selected FOMs plus both `fault_subject` and `identity_fault_subject`, including their inputs. |
| Complete unused-program release | **PARTIAL, nonblocking for this smoke** | `c2time.py:319–321` retains every FD solver at a needed mesh, including potentially unused advection/Newton-limit variants. The staged smoke has one variant per candidate FD mesh. Pilot programs also remain cached through completion (`:391–401`). |
| Recompilation events lost | **FIXED** | `fomrun2.py:110–115,123–133,164–179` records compilation events, including rebuilt maps; `c2time.py:403–404` persists them. |
| Too-small targets disappear from H-comp | **FIXED** | `make_c2configs.py:89–93` creates explicit missing cells; `c2analyze.py:126–148` includes their reasons in H-comp. |
| Crossover summaries discard reasons | **FIXED** | `c2analyze.py:189–198` retains `other_statuses` and explains missing timed endpoints. |
| Empty versus all-vetoed groups | **FIXED** | `c2time.py:285,310–312` distinguishes original candidate absence from identity vetoes. |

CPU checks confirmed:

- The actual release block retained a selected PS program and two distinct FD control programs. All three executed afterward without recompilation.
- Released PS and FD programs rebuilt successfully for pilot calls, passed convergence checks, and recorded recompilation events. Warm calls added no events.
- A too-small-target fixture passed through the config generator and analyzer: **four H-comp entries and eight crossover summaries** preserved the reason.
- Selection fixtures distinguished **“no candidate”** from **“all candidates vetoed (identity)”**.

**Staging matches HEAD:** all **111 provenance-listed files** were compared byte-for-byte with Git HEAD. Manifest verification and `bash -n run.sbatch` passed; reservation and commit metadata agree. The staged controls and three pilots remain configured correctly.

Two nonblocking limitations remain:

- `c2analyze.py:154–156` can label a missing K2 member cell `status: "ok"` when `union_joint_group` is absent. Its classification remains unset, so this does **not** produce a positive K2 conclusion; the fixture remained inconclusive.
- **NOTED — forgery-only:** fingerprints are not full-byte hashes, and analysis trusts the recorded frozen-selection hash.

This verdict authorizes the smoke check, not acceptance of its eventual GPU results. Actual control firing and combined-workload memory remain smoke outcomes.

No repository or lab-log writes, cluster operations, or large-bank builds were performed. Reused scratch totals approximately **6.3 MB**.