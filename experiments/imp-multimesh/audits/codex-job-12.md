| Job | Verdict | Blocking fixes |
|---|---|---|
| J1 — 256 | **FIX-FIRST** | Repair exhausted-budget handling and the shared NNLS deadline. |
| J3 — 1024 | **FIX-FIRST** | Same two fixes. |
| J4 — 4096 | **FIX-FIRST** | Same fixes, plus release cached rank prefixes before construction and resolve the resource-certificate mismatch. |

1. **Exhausted NNLS budget crashes the target job.** `contrib_and_nnls()` returns `paths.selected=None, attempts={}` when fewer than 60 seconds remain. Its caller then indexes `attempts[None]`. A small CPU probe reproduced `KeyError(None)`. Preserve the returned reason, save both configurations as `UNRESOLVED`, and continue safely. See [mmbuild.py:210](experiments/imp-multimesh/mmbuild.py#L210) and [mmbuild.py:319](experiments/imp-multimesh/mmbuild.py#L319).

2. **The initial NNLS path and retry do not share one allowance.** The allowance is correctly calculated *after* contributions, but `ecsw_weights()` gives each attempt the same `max_seconds`. A 1,500-second allowance can therefore consume approximately 3,000 seconds. Use one deadline and pass its remaining time to the retry, as A9.3 specifies. See [build.py:350](experiments/jcp-hr-baselines/build.py#L350).

3. **J4 retains an unnecessary 34.3 GB prefix during rank-384 construction.** `Cell._slice` still holds rank 256 when the next contribution build begins; requesting rank 384 also leaves it cached. A CPU probe confirmed that lifetime. The smoke peak was **107.25 GB without this additional prefix**; adding it gives approximately **141.6 GB**, beyond the cited usable memory. Clear the cached prefix **before each new rank’s contribution build**, and clear it on the full-rank return. Host states around 48 GB fit the requested 240 GB. See [mmcore.py:332](experiments/imp-multimesh/mmcore.py#L332).

4. **J4’s resource certificate cannot pass submission.** Its projection is **5.044 hours**, exceeding both its **3.583-hour allocation** and **3.5-hour command timeout**. The ledger explicitly rejects `projected > limit`. Reconcile the projection, registered deferrals, command budget, and allocation within the lane cap. See [resources-j4.json](experiments/imp-multimesh/checks/resources-j4.json).

The remaining requested checks pass inspection: exact arms and six configurations per arm; resampling, combined QR factor, rank-specific codes/trust and nested-node mapping; target states generated once only when refits need them; blocked contributions only above 1024; expected `W` archives, lattice bases, statuses, scoring rows, Q1 and rollout-pair outputs; and the seeded eight-trajectory PM estimate. All three staged manifests verify and their Python sources match the live tree. J0’s source certificates and all supports are present; cluster source files exist.

The local ledger needs reconciliation: J0 is completed (**1,667 seconds**), but remains recorded as pending. With J5’s full reservation retained, approximately **3.921 hours** remain; the three target limits total **6.917 hours**, so submissions must be sequenced against settled usage. J1/J3 command limits leave five minutes before Slurm termination. Writes are confined to each job’s namespace; shared source/reference paths are read-only.

No files, jobs, or lab-log entries were changed.