# JCP lane: combined

Part of the JCP paper campaign (`reports/2026-10-06-jcp-offmesh-paper-plan.md` on main). Role: Step 2: best version of the method — second-order stepping (C2) combined with the wide bank (C1), plus a fast fixed-sweep second-order 3D solver.

- Branch `exp/2026-10-13-jcp-combined`, forked from `exp/2026-10-01-ns3d-coordnet-bank` @ `21c175a1b`. Never pushed directly; mirrored to `origin/codeonly/exp/2026-10-13-jcp-combined` by `sync_github.sh` after every commit.
- `vendor/quad2d` and `vendor/quad3d` are byte-identical copies of the off-mesh code from the two 2026-10-01 quadrature lanes (provenance and sha256 in `vendor/PROVENANCE.json`). Do not edit them in place; copy into the lane directory first.
- Next step: `DESIGN.md` (pre-registration), Codex design audit, then code. No cluster job before the audited design.
