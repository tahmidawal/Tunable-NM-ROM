# JCP lane: wide-bank

Part of the JCP paper campaign (`reports/2026-10-06-jcp-offmesh-paper-plan.md` on main). Role: C1: wider bank - 2D accurate setting at R'=512, trimming the test count M, then a wider 3D bank (R=768/1024) and the error/cost scaling law.

- Branch `exp/2026-10-08-jcp-wide-bank`, forked from `exp/2026-10-01-ns3d-coordnet-bank` @ `21c175a1b`. Never pushed directly; mirrored to `origin/codeonly/exp/2026-10-08-jcp-wide-bank` by `sync_github.sh` after every commit.
- `vendor/quad2d` and `vendor/quad3d` are byte-identical copies of the off-mesh code from the two 2026-10-01 quadrature lanes (provenance and sha256 in `vendor/PROVENANCE.json`). Do not edit them in place; copy into the lane directory first.
- Status (2026-10-09): complete. Pre-registration `DESIGN.md` (A0–A8), code (`w2d.py`, `w3d.py`, `train3d/train2w.py`, `reselect3d.py`, independent audits `audit_w.py`, `audit_w3.py`), every Codex audit in `audits/`, generated `report.md` + `plots/` (`make_report.py`). Large files (> 50 MB: the R=1024 bank, archived outputs) are listed by SHA256 in `runs/MANIFEST-large.sha256` and are not on the GitHub mirror. Cluster namespace `jcpwide` removed after collection.
