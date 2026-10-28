# Pending allocation resource update — 2026-10-28

Job 6612209 remains pending with zero runtime. Its H200-specific request has a scheduler
start estimate the next morning. The coordinator inspected current node allocations:
the H100 node has a free GPU plus sufficient CPU and memory. This is CPU finite-element
control work; no result or timing comparison depends on the GPU model.

Authorize one pending-job resource update from `gpu:h200:1` to `gpu:h100:1`, preserving
the same job ID, directory, exact source/configuration, one GPU, sixteen CPUs, 64 GiB and
twenty-minute limit. The existing mandatory GPU/f64/highest preflight still applies. This
is a prospective scheduling change, not a physics or acceptance change. Preserve the
original submission receipt and the before/after scheduler state.

The lane owner must recheck exact ID/name/owner and pending state immediately before
`scontrol update JobId=6612209 Gres=gpu:h100:1`. If it has started, abort the update. If
the scheduler rejects it, retain the error and original job; do not cancel or duplicate it.
Verify the final resource request and actual allocated GPU in collected evidence.

The supported job-update field is documented in the primary
[Slurm scontrol reference](https://slurm.schedmd.com/scontrol.html#OPT_Gres). The coordinator
checked the job-specific field, not the separate privileged node-GRES update operation.
All scientific gates and the original A8 source review remain unchanged.
