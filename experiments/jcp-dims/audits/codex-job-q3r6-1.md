**FIX-FIRST.** Member selection passes, but the time allowance fails and no q3r6 staging directory exists.

| Check | Finding |
|---|---|
| Members | PASS: exact ordered equality with `front3d.round_next`: **17 members**, from 436 candidates using qualify’s 55 labels plus rounds 4–5. |
| Evidence handling | PASS: all predecessor members have completed labels. Missing/blocked labels remain untested; unverified labels remain non-passing. |
| Successors | PASS: every `q+`/`q++` matches the lane’s extended ladders. |
| Resource certificate | Config hash matches; independently reproduced **1.0219396 h** projection. Required 1.5× margin is **1.53285 h**, yielding a **1.75 h** limit. **1.5 h fails staging validation.** |
| Ledger | **FAIL:** pending q3r6 is **1.5 h**; `ledger.validate` rejects it against the **1.75 h** certificate. With 1.75 h, total commitments are **53.7137/55 GPU-h**. |
| Staging/provenance | **UNVERIFIABLE:** `experiments/jcp-dims/runs/q3r6` is absent. There is no staged sbatch, manifest, reservation, or commit proof to audit. The source template uses H200 and the isolated `jcp-dims/q3r6` paralab namespace. |

Raise the pending allowance to **1.75 h**, stage with that limit, and audit the resulting artifacts before submission.

No files changed or jobs submitted; LAB-LOG.md was not written.