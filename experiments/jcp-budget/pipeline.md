# jcp-budget production pipeline (operator notes; the order matters)

1. References (2D): `ref2a` collected **in full** (dev6 fine fields for the offline recomputation), `ref2b`, `ref2c` with
   `--no-fine`; then **serially** `cluster/collect.py <a> --promote` for ref2a, ref2b, ref2c (FD), then ps2a, ps2b (PS
   files land in `refs/2d/val32/T0.5/PS/`). Cleanup (`--cleanup`) only after promotion.
2. `anaref.py --run runs/ref2a/archive/output --run ... (ref2b, ref2c, ps2a, ps2b, ref3, ps3, [ref2t1, ps2t1]) --out
   checks/refs-all.json`.
3. Pull the cluster `refs/REFS-MANIFEST.sha256` to `runs/REFS-MANIFEST.sha256` (the B7/B8 analyses verify every
   reference file against it); the lattice reference files are the collected job outputs (`runs/<ref job>/archive/output/
   ref_refrun_*/2d/...`), assembled locally into `runs/refs-local/2d/...` by `assemble_refs.py`.
4. b-jobs (after promotion): b1024a, b1024b, b256, b4096 (2D), b3d (3D: does not need references in-job).
5. `ana2d.py --run <b-job output>` per job; `ana2d.py --compare` over all four; `ana7.py --run <one or two b-job outputs>
   --refs runs/refs-local --manifest runs/REFS-MANIFEST.sha256 --anaref checks/refs-all.json --fine runs/refs-local`;
   `ana3d.py --run <b3d output> --refs runs/refs-local --manifest runs/REFS-MANIFEST.sha256 --anaref checks/refs-all.json`.
5b. Before `ana2d` / `ana7` of a mesh: `fetch_qualify.py --mesh <m> --cache <scratch>/qcache<m>` and run both with
   `QUALIFY_CACHE=<scratch>/qcache<m>` (A12); delete the cache afterwards. B7 at T = 1.0: `ana7.py --run <b1024t1 output>
   --refs runs/refs-local --manifest runs/REFS-MANIFEST.sha256 --anaref checks/refs-all.json` (no `--fine`: B7-only run,
   A11.1); `anaref` must have read both the T = 0.5 and the T = 1.0 reference outputs (prefix check, A1.5).
6. Codex results audit; `make_report.py`; Codex report audit; hand-back.
