# JCP lane: speed3d

Part of the JCP paper campaign (`reports/2026-10-06-jcp-offmesh-paper-plan.md` on main). Role: 3D speed: turn the second-order accuracy gain into a speed-up (one-pass big steps, narrower bank, overhead cuts, large-mesh FOM comparison).

- Branch `exp/2026-10-14-jcp-speed3d`, forked from `exp/2026-10-01-ns3d-coordnet-bank` @ `21c175a1b`. Never pushed directly; mirrored to `origin/codeonly/exp/2026-10-14-jcp-speed3d` by `sync_github.sh` after every commit.
- `vendor/quad2d` and `vendor/quad3d` are byte-identical copies of the off-mesh code from the two 2026-10-01 quadrature lanes (provenance and sha256 in `vendor/PROVENANCE.json`). Do not edit them in place; copy into the lane directory first.
- Next step: `DESIGN.md` (pre-registration), Codex design audit, then code. No cluster job before the audited design.

## Operational notes (for the report)
- 2026-10-14: one local GB10 GPU process reported "Red zone modified" and a Gram condition number of 70264 for the M2 bank
  (1.0000 on CPU). The coordinator re-checked the device in a fresh process (f64, highest: cond(QᵀQ) = 1.0000, matmul error
  about 1e-16), so it is judged process-specific (memory pressure / 0.25 memory fraction with large tables). Lane policy:
  all local checks on CPU; every GPU number comes from the cluster (H200).
