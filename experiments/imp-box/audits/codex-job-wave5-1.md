One submission blocker: **collection under the original cell names does not merge correctly.**

[collect.sh:51](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/cluster/collect.sh:51) overlays restart evidence into the existing `_job`, then compares its entire inventory against the restart job. Both archives already contain old `slurm-6203321.out` / `slurm-6203563.out` and `out/ref.log`, `data.log`, `pod.log`, etc. Those remain after the overlay but are absent from the restart job. The checksum-list `diff` therefore fails; collection cannot finish. Use separate job-specific evidence directories and keep their manifests separate from cell-data manifests.

Other checks passed:

- **Archives:** both checksum manifests verify. Required POD arrays, 30 ECSW NPZs per cell, dev/validation references, ladder records and prerequisite markers exist. W40 additionally has `bank.pkl`, `.done_bank` and translated-POD arrays.
- **Bodies:** restore and verify archives, clear time/control completion markers and selection outputs, then invoke only `time` and `controls`.
- **Release:** both body hashes match; H200, 240G, six hours each. GPU preflight, f64 and highest matmul precision are configured.
- **Budget:** `41.3547 + 12 + 6 + 17.76 = 77.1147 ≤ 87`, leaving **9.8853 GPU-h**. W08_A2 production remains deferred; its cost is not covered by this release.

Read-only audit completed; no files changed or jobs submitted.

**DO-NOT-SUBMIT**