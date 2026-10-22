Read-only audit of HEAD `34fd6a104`; no files changed or jobs submitted.

1. **CORRECT — W08 configuration.** Both body hashes match `RELEASE.json`: H200, 240G, 10 hours. They run all seven stages, including controls, without size overrides. `BANK_CELLS_W` enables both bank arms for W08; `TRANS_W` excludes its TRANS arm. Translation checker fixtures still run intentionally. GPU preflight, f64 and highest precision are enforced. W40_A1 records `all_fire=true`.

2. **NEEDS-FIX — certificate inputs; cap still passes.** The stored arithmetic reproduces **63.204275 GPU-h**, but `committed_hours=26.4` rounds two **6.2275-hour** bounds downward to 6.2. Correct committed cost is **26.455**, giving **63.259275 < 87**. Also, polling cancellation cannot guarantee exactly 14 hours. Reserving smk2-A4’s full 24-hour scheduler limit instead gives **73.259275 < 87**, avoiding that dependency.

3. **NEEDS-FIX — cancellation guard.** The live invocation is `cancel_at.sh 6187596 50400`; scheduler records confirm that ID belongs to `tawal01`, named `ibox-smk2-A4`. **That invocation targets only the intended job.** However, [cancel_at.sh](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/cluster/cancel_at.sh:3) validates neither arguments nor job identity. More concretely:
   - A failed/empty accounting query causes it to exit permanently.
   - A failed `scancel` still logs “cancelled” and exits successfully.
   - Five-minute polling permits cancellation after the stated bound.

   Validate one numeric ID and its expected owner/name; retry query failures; verify cancellation before reporting success. At inspection, smk2-A4 was **still running at 7:54:23**, not already cancelled.

4. **NEEDS-FIX — release readiness.** Fix the cancellation guard and regenerate the certificate with unrounded commitments and adequate cancellation allowance. The submission wrapper enforces lane/account caps and committed specifications; the release loop launches detached collectors. Currently all three lane slots are occupied, and the active releaser contains **only W28_A1**—W08 automation is not yet armed. Preserve the requested ordering when configuring it. No additional W08 execution blocker found; cluster free space was 3.2T.

**DO-NOT-SUBMIT**