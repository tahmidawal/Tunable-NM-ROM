# Per-case error vs quadrature points (Burgers 2D/3D, NS3D)

Interactive page of mean ROM error (over cases) against number of off-mesh quadrature points, for every rule family and mesh in the 2026-11-05 lanes `exp/2026-11-05-burgers-quadrature-plateau` and `exp/2026-11-05-ns3d-offmesh-quadrature`. Numbers are final for those lanes (NS3D: development cases only).

Published at https://claude.ai/artifact/KY7fxEYzYQpsA52msFXFE7 (private).

Rebuild: `python -I extract.py data.json` (reads the pulled run JSONs in both worktrees; Burgers statistics and M* come from the lane's own `plateau.py`), then drop each family's `cases` from `data.json` and substitute it for `__DATA__` in `page.html` to get `quadrature-points-per-case.html`.
