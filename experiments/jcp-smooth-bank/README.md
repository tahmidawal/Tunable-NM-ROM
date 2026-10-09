# JCP lane: smooth-bank

Part of the JCP paper campaign (`reports/2026-10-06-jcp-offmesh-paper-plan.md` on main). Role: C3: derivative-aware bank - Sobolev (value+gradient) training, smoothness / feature-scale sweep, coarse-data control; 2D first.

- Branch `exp/2026-10-08-jcp-smooth-bank`, forked from `exp/2026-10-01-ns3d-coordnet-bank` @ `21c175a1b`. Never pushed directly; mirrored to `origin/codeonly/exp/2026-10-08-jcp-smooth-bank` by `sync_github.sh` after every commit.
- `vendor/quad2d` and `vendor/quad3d` are byte-identical copies of the off-mesh code from the two 2026-10-01 quadrature lanes (provenance and sha256 in `vendor/PROVENANCE.json`). Do not edit them in place; copy into the lane directory first.
- Next step: `DESIGN.md` (pre-registration), Codex design audit, then code. No cluster job before the audited design.

## Status (2026-10-09 05:00)

Rounds 1 and 2 are complete (training tr1b and tr2; evaluation ev1 and ev2r). The generated report is `report.md`
(`python make_report.py --train tr1b,tr2 --eval ev1,ev2r`), the plots are in `plots/`, and every Codex audit is in
`audits/`. Pre-registered outcome: no useful winner; nothing is promoted to 3D, and `comb` is not run. Known
remaining report nits (audit `codex-report-final-2.md`): frontier labels overlap in `fast`, there is no ev2r cost
panel in `e2e.png`, some glossary prose is not in LaTeX, and configuration constants are typed in the generator.
