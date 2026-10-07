# JCP lane: jcp-references

Part of the JCP paper campaign (`reports/2026-10-06-jcp-offmesh-paper-plan.md` on main). Role: E0: order-verified second-order references, pseudo-spectral cross-check, sealed and out-of-distribution cohorts (shared read-only store).

- Branch `exp/2026-10-06-jcp-references`, forked from `exp/2026-10-01-ns3d-coordnet-bank` @ `21c175a1b`. Never pushed directly; mirrored to `origin/codeonly/exp/2026-10-06-jcp-references` by `sync_github.sh` after every commit.
- `vendor/quad2d` and `vendor/quad3d` are byte-identical copies of the off-mesh code from the two 2026-10-01 quadrature lanes (provenance and sha256 in `vendor/PROVENANCE.json`). Do not edit them in place; copy into the lane directory first.
- Next step: `DESIGN.md` (pre-registration), Codex design audit, then code. No cluster job before the audited design.
