# Backup manifest — 2026-09-27

Goal: every piece of *code* (scripts, configs, notes, paper sources, small result JSON/logs) from the local repo
`Tunable-NM-ROM-Claude` (128 branches, 127 worktrees) is on GitHub. Bulk data (.npz/.npy/.pt/.pkl/.tar/.gz/.bin, chunked `part*` files, anything >1 MB non-figure / >5 MB figure) is NOT pushed.
The local repo, its branches, commits and worktrees were NOT modified. Rewritten copies were built in a side repo (`../nmrom-codeonly-side.git`, uses the original as a git alternate) and pushed from there.

## How to read this

- **as-is**: branch was already identical on GitHub.
- **codeonly/<branch>**: the local branch had commits not on GitHub that include data. We pushed a copy with the data files removed from every new commit (same authors/dates/messages; parents chain onto the existing GitHub history). The *original* commits (with data) still exist only locally.
- **backup/2026-09-27/main-local**: local `main` has diverged from GitHub `main` (12 local commits, GitHub has others) — pushed under a new name instead of force-pushing. No data had to be stripped.
- **backup/2026-09-27/<worktree>-wip**: uncommitted/untracked code in a worktree, plus gitignored small code/text files (e.g. per-run .py/.json/.sbatch snapshots), committed on top of that worktree's branch (or its codeonly copy).

Code-only rule (per file): drop data extensions (npz npy pt pth ckpt bin pkl h5 nc gz tar tgz zip xz safetensors mat vtk dat raw joblib parquet onnx 7z, `*.partNNNN`, `large-artifacts/`, `archive-parts/`); keep figures/notebooks/tex (png pdf svg jpg ipynb tex eps) up to 5 MB; keep anything else up to 1 MB. Files already on GitHub are never removed.

## Branches

| Local branch | On GitHub as | New commits | Files stripped | Bytes stripped |
|---|---|---|---|---|
| `exp/2026-08-12-coord-decoder` | as-is | 0 | 0 | 0 |
| `exp/2026-08-13-cost-scaling-coordnet` | as-is | 0 | 0 | 0 |
| `exp/2026-08-13-cost-scaling-cp` | as-is | 0 | 0 | 0 |
| `exp/2026-08-13-gn-tolerance-sweep` | as-is | 0 | 0 | 0 |
| `exp/2026-08-13-heat2d-coord-decoder` | as-is | 0 | 0 | 0 |
| `exp/2026-08-14-burgers2d-coord-rom` | as-is | 0 | 0 | 0 |
| `exp/2026-08-14-multistage-precision` | as-is | 0 | 0 | 0 |
| `exp/2026-08-14-wave2d-coord-rom` | as-is | 0 | 0 | 0 |
| `exp/2026-08-16-burgers2d-rom-latent-stepping` | as-is | 0 | 0 | 0 |
| `exp/2026-08-16-cascade-nmrom` | as-is | 0 | 0 | 0 |
| `exp/2026-08-16-heat2d-rom-latent-stepping` | as-is | 0 | 0 | 0 |
| `exp/2026-08-16-poisson2d-rom-objective` | as-is | 0 | 0 | 0 |
| `exp/2026-08-16-wave2d-rom-latent-stepping` | as-is | 0 | 0 | 0 |
| `exp/2026-08-17-cost-to-tolerance` | as-is | 0 | 0 | 0 |
| `exp/2026-08-17-inr-rom-consolidated` | as-is | 0 | 0 | 0 |
| `exp/2026-08-17-rom-warmstart-fom` | as-is | 0 | 0 | 0 |
| `exp/2026-08-18-codex-handoff` | as-is | 0 | 0 | 0 |
| `exp/2026-08-19-burgers-1e3-10x` | `codeonly/exp/2026-08-19-burgers-1e3-10x` | 107 | 44 | 2.28 GB |
| `exp/2026-08-19-burgers-hybrid-1024` | as-is | 0 | 0 | 0 |
| `exp/2026-08-19-heat-amgx-nmrom` | as-is | 0 | 0 | 0 |
| `exp/2026-08-19-nonlinear-decoder-architecture` | as-is | 0 | 0 | 0 |
| `exp/2026-08-19-poisson-hybrid-1024` | as-is | 0 | 0 | 0 |
| `exp/2026-08-20-burgers-hybrid-2048` | `codeonly/exp/2026-08-20-burgers-hybrid-2048` | 13 | 2 | 102.0 MB |
| `exp/2026-08-20-poisson-hybrid-2048` | as-is | 0 | 0 | 0 |
| `exp/2026-08-20-two-pde-k64-eq4m` | as-is | 0 | 0 | 0 |
| `exp/2026-08-22-separable-decoder` | as-is | 0 | 0 | 0 |
| `exp/2026-08-23-n256-push` | as-is | 0 | 0 | 0 |
| `exp/2026-08-23-sepdec-n1024` | as-is | 0 | 0 | 0 |
| `exp/2026-08-23-sepdec-n128` | as-is | 0 | 0 | 0 |
| `exp/2026-08-23-sepdec-n256` | as-is | 0 | 0 | 0 |
| `exp/2026-08-23-sepdec-n512` | as-is | 0 | 0 | 0 |
| `exp/2026-08-25-burgers-accuracy` | as-is | 0 | 0 | 0 |
| `exp/2026-08-25-eq-fidelity-ladder` | as-is | 0 | 0 | 0 |
| `exp/2026-08-25-sepdec-consolidated` | as-is | 0 | 0 | 0 |
| `exp/2026-08-26-codesign` | as-is | 0 | 0 | 0 |
| `exp/2026-08-26-eq-learned` | as-is | 0 | 0 | 0 |
| `exp/2026-08-27-b1d-poissonqf` | as-is | 0 | 0 | 0 |
| `exp/2026-08-27-nodes-mm` | as-is | 0 | 0 | 0 |
| `exp/2026-08-29-b1d-tensor` | as-is | 0 | 0 | 0 |
| `exp/2026-08-29-b2d-tensor` | as-is | 0 | 0 | 0 |
| `exp/2026-08-30-stokes-vector` | as-is | 0 | 0 | 0 |
| `exp/2026-08-30-waves-vector` | as-is | 0 | 0 | 0 |
| `exp/2026-09-03-burgers3d-tensor` | as-is | 0 | 0 | 0 |
| `exp/2026-09-03-wave2d-mechanism` | as-is | 0 | 0 | 0 |
| `exp/2026-09-04-separable-tensor-consolidated` | as-is | 0 | 0 | 0 |
| `exp/2026-09-06-b3d-anchor` | as-is | 0 | 0 | 0 |
| `exp/2026-09-06-b3d-encoder` | as-is | 0 | 0 | 0 |
| `exp/2026-09-06-b3d-mixture` | as-is | 0 | 0 | 0 |
| `exp/2026-09-06-b3d-quadratic` | as-is | 0 | 0 | 0 |
| `exp/2026-09-06-burgers3d-repair` | as-is | 0 | 0 | 0 |
| `exp/2026-09-06-wave-head-transfer` | as-is | 0 | 0 | 0 |
| `exp/2026-09-07-mr-burgers2d` | `codeonly/exp/2026-09-07-mr-burgers2d` | 35 | 3461 | 43.12 GB |
| `exp/2026-09-07-mr-heat2d` | `codeonly/exp/2026-09-07-mr-heat2d` | 44 | 869 | 49.21 GB |
| `exp/2026-09-07-mr-poisson2d` | `codeonly/exp/2026-09-07-mr-poisson2d` | 49 | 715 | 30.82 GB |
| `exp/2026-09-07-mr-wave2d` | `codeonly/exp/2026-09-07-mr-wave2d` | 34 | 1201 | 83.89 GB |
| `exp/2026-09-10-modcp-burgers2d` | `codeonly/exp/2026-09-10-modcp-burgers2d` | 49 | 3371 | 26.51 GB |
| `exp/2026-09-10-modcp-wave2d` | as-is | 0 | 0 | 0 |
| `exp/2026-09-14-head-ablation` | `codeonly/exp/2026-09-14-head-ablation` | 65 | 966 | 51.26 GB |
| `exp/2026-09-14-mesh-ladder` | `codeonly/exp/2026-09-14-mesh-ladder` | 56 | 942 | 50.37 GB |
| `exp/2026-09-14-no-audit` | `codeonly/exp/2026-09-14-no-audit` | 65 | 998 | 53.98 GB |
| `exp/2026-09-14-no-burgers` | `codeonly/exp/2026-09-14-no-burgers` | 82 | 1081 | 59.41 GB |
| `exp/2026-09-14-no-poisson` | `codeonly/exp/2026-09-14-no-poisson` | 51 | 3271 | 50.98 GB |
| `exp/2026-09-14-paper-draft` | `codeonly/exp/2026-09-14-paper-draft` | 48 | 924 | 49.38 GB |
| `exp/2026-09-15-cheap-corrections` | `codeonly/exp/2026-09-15-cheap-corrections` | 75 | 1027 | 54.20 GB |
| `exp/2026-09-15-head-refine` | `codeonly/exp/2026-09-15-head-refine` | 72 | 1023 | 53.94 GB |
| `exp/2026-09-15-prior-dial` | `codeonly/exp/2026-09-15-prior-dial` | 78 | 999 | 52.74 GB |
| `exp/2026-09-16-b-head-train` | `codeonly/exp/2026-09-16-b-head-train` | 85 | 996 | 52.62 GB |
| `exp/2026-09-16-b-ladder-top` | `codeonly/exp/2026-09-16-b-ladder-top` | 87 | 1058 | 55.53 GB |
| `exp/2026-09-16-b-speed` | `codeonly/exp/2026-09-16-b-speed` | 95 | 1102 | 57.49 GB |
| `exp/2026-09-16-p-bank-head` | `codeonly/exp/2026-09-16-p-bank-head` | 87 | 1064 | 53.75 GB |
| `exp/2026-09-16-paper-refresh` | `codeonly/exp/2026-09-16-paper-refresh` | 113 | 925 | 49.39 GB |
| `exp/2026-09-16-q-diag` | `codeonly/exp/2026-09-16-q-diag` | 89 | 1060 | 55.53 GB |
| `exp/2026-09-16-q-ridge` | `codeonly/exp/2026-09-16-q-ridge` | 115 | 1100 | 57.38 GB |
| `exp/2026-09-16-q-trajdirs` | `codeonly/exp/2026-09-16-q-trajdirs` | 105 | 1069 | 55.90 GB |
| `exp/2026-09-17-b-eqtop` | `codeonly/exp/2026-09-17-b-eqtop` | 127 | 1138 | 57.84 GB |
| `exp/2026-09-17-b-lowvisc` | `codeonly/exp/2026-09-17-b-lowvisc` | 140 | 1131 | 58.74 GB |
| `exp/2026-09-17-b-panel` | `codeonly/exp/2026-09-17-b-panel` | 152 | 1428 | 72.00 GB |
| `exp/2026-09-17-b-qxm` | `codeonly/exp/2026-09-17-b-qxm` | 124 | 1142 | 59.14 GB |
| `exp/2026-09-17-b-seeds` | `codeonly/exp/2026-09-17-b-seeds` | 131 | 1196 | 61.34 GB |
| `exp/2026-09-17-lshape` | `codeonly/exp/2026-09-17-lshape` | 115 | 5188 | 60.38 GB |
| `exp/2026-09-17-no-second` | `codeonly/exp/2026-09-17-no-second` | 96 | 2425 | 64.97 GB |
| `exp/2026-09-17-ns2d` | `codeonly/exp/2026-09-17-ns2d` | 78 | 1025 | 52.19 GB |
| `exp/2026-09-17-p-linear` | `codeonly/exp/2026-09-17-p-linear` | 108 | 2196 | 59.36 GB |
| `exp/2026-09-17-w-ladder` | `codeonly/exp/2026-09-17-w-ladder` | 68 | 989 | 50.67 GB |
| `exp/2026-09-20-bank-floor` | `codeonly/exp/2026-09-20-bank-floor` | 101 | 999 | 52.64 GB |
| `exp/2026-09-20-hires-burgers` | `codeonly/exp/2026-09-20-hires-burgers` | 174 | 1450 | 72.01 GB |
| `exp/2026-09-20-hires-heat` | `codeonly/exp/2026-09-20-hires-heat` | 118 | 1842 | 90.51 GB |
| `exp/2026-09-20-hires-poisson` | `codeonly/exp/2026-09-20-hires-poisson` | 170 | 15195 | 79.72 GB |
| `exp/2026-09-20-nmrom-baselines` | `codeonly/exp/2026-09-20-nmrom-baselines` | 97 | 1025 | 55.85 GB |
| `exp/2026-09-20-paper-b3d` | `codeonly/exp/2026-09-20-paper-b3d` | 190 | 2559 | 131.70 GB |
| `exp/2026-09-20-paper-h3d` | `codeonly/exp/2026-09-20-paper-h3d` | 111 | 5329 | 117.27 GB |
| `exp/2026-09-20-paper-ns3d` | `codeonly/exp/2026-09-20-paper-ns3d` | 130 | 2322 | 128.26 GB |
| `exp/2026-09-20-paper-p3d` | `codeonly/exp/2026-09-20-paper-p3d` | 148 | 15182 | 79.67 GB |
| `exp/2026-09-21-burgers-eqcert` | `codeonly/exp/2026-09-21-burgers-eqcert` | 187 | 1450 | 72.01 GB |
| `exp/2026-09-21-burgers-heldout` | `codeonly/exp/2026-09-21-burgers-heldout` | 194 | 1452 | 72.01 GB |
| `exp/2026-09-21-heat3d-bank` | `codeonly/exp/2026-09-21-heat3d-bank` | 150 | 1871 | 90.92 GB |
| `exp/2026-09-21-ns3d-grok` | `codeonly/exp/2026-09-21-ns3d-grok` | 148 | 2322 | 128.26 GB |
| `exp/2026-09-22-burgers-repanel` | `codeonly/exp/2026-09-22-burgers-repanel` | 182 | 1450 | 72.01 GB |
| `exp/2026-09-22-ns3d-shift-decoder` | `codeonly/exp/2026-09-22-ns3d-shift-decoder` | 173 | 2322 | 128.26 GB |
| `exp/2026-09-22-ops-deeponet-b2d` | `codeonly/exp/2026-09-22-ops-deeponet-b2d` | 107 | 2444 | 65.81 GB |
| `exp/2026-09-22-ops-timing-panel` | `codeonly/exp/2026-09-22-ops-timing-panel` | 106 | 2444 | 64.97 GB |
| `exp/2026-09-22-ops-tune-deeponet` | `codeonly/exp/2026-09-22-ops-tune-deeponet` | 120 | 2444 | 65.81 GB |
| `exp/2026-09-22-ops-tune-grid` | `codeonly/exp/2026-09-22-ops-tune-grid` | 124 | 2444 | 65.81 GB |
| `exp/2026-09-22-quadratic-manifold` | `codeonly/exp/2026-09-22-quadratic-manifold` | 160 | 1432 | 72.00 GB |
| `exp/2026-09-23-burgers-bank-knob` | `codeonly/exp/2026-09-23-burgers-bank-knob` | 213 | 1451 | 72.01 GB |
| `exp/2026-09-23-burgers-compare-hires` | `codeonly/exp/2026-09-23-burgers-compare-hires` | 187 | 1448 | 72.01 GB |
| `exp/2026-09-23-burgers2d-speed` | `codeonly/exp/2026-09-23-burgers2d-speed` | 223 | 1451 | 72.01 GB |
| `exp/2026-09-23-burgers3d-retry` | `codeonly/exp/2026-09-23-burgers3d-retry` | 242 | 2597 | 131.88 GB |
| `exp/2026-09-23-burgers3d-span` | `codeonly/exp/2026-09-23-burgers3d-span` | 214 | 2585 | 131.82 GB |
| `exp/2026-09-23-heat-bank-knob` | `codeonly/exp/2026-09-23-heat-bank-knob` | 129 | 1854 | 90.56 GB |
| `exp/2026-09-23-heat-compare-hires` | `codeonly/exp/2026-09-23-heat-compare-hires` | 146 | 1847 | 90.52 GB |
| `exp/2026-09-23-ns3d-operators` | `codeonly/exp/2026-09-23-ns3d-operators` | 205 | 2356 | 128.29 GB |
| `exp/2026-09-23-ns3d-shift-head` | `codeonly/exp/2026-09-23-ns3d-shift-head` | 186 | 2350 | 128.29 GB |
| `exp/2026-09-23-poisson-bank-knob` | `codeonly/exp/2026-09-23-poisson-bank-knob` | 192 | 15202 | 79.74 GB |
| `exp/2026-09-23-poisson-bank-knob-3d` | `codeonly/exp/2026-09-23-poisson-bank-knob-3d` | 195 | 15214 | 79.80 GB |
| `exp/2026-09-23-spectral-fom` | `codeonly/exp/2026-09-23-spectral-fom` | 235 | 15220 | 79.78 GB |
| `exp/2026-09-24-burgers-eq-tol-knobs` | `codeonly/exp/2026-09-24-burgers-eq-tol-knobs` | 230 | 1451 | 72.01 GB |
| `exp/2026-09-24-operators-all-pdes` | `codeonly/exp/2026-09-24-operators-all-pdes` | 205 | 2356 | 128.29 GB |
| `exp/2026-09-25-burgers2d-test` | `codeonly/exp/2026-09-25-burgers2d-test` | 231 | 1451 | 72.01 GB |
| `exp/2026-09-25-ns3d-test` | `codeonly/exp/2026-09-25-ns3d-test` | 208 | 2358 | 128.29 GB |
| `exp/2026-09-25-poisson-lshape3d-test` | `codeonly/exp/2026-09-25-poisson-lshape3d-test` | 206 | 15220 | 79.83 GB |
| `exp/2026-09-25-poisson2d-test` | `codeonly/exp/2026-09-25-poisson2d-test` | 199 | 15206 | 79.75 GB |
| `exp/2026-09-25-t2-burgers-test` | `codeonly/exp/2026-09-25-t2-burgers-test` | 194 | 1448 | 72.01 GB |
| `exp/2026-09-25-t2-ns3d-test` | `codeonly/exp/2026-09-25-t2-ns3d-test` | 208 | 2356 | 128.29 GB |
| `exp/2026-09-25-t3-burgers-test` | `codeonly/exp/2026-09-25-t3-burgers-test` | 193 | 1451 | 72.02 GB |
| `fix/heat-rollout-warm-start` | as-is | 0 | 0 | 0 |
| `main` | `backup/2026-09-27/main-local` | 12 | 0 | 0.0 MB |

Not pushed: **none** — every local branch is on GitHub as itself or as a codeonly/backup copy (verified with `git ls-remote origin`).

Full list of stripped files per branch: `stripped-files.tsv` (in this branch). Biggest single stripped files: 1.22 GB `.npz` (historical-burgers2d-replay), 1.19 GB `collection.tar.gz` (lshape lsh07), 429 MB `fno-large.pt` (no-second res02).

## Worktree WIP backups

| Worktree | Backup branch | Uncommitted/untracked files | Ignored small code files | Skipped (data or >5 MB) |
|---|---|---|---|---|
| `2026-08-12-coord-decoder` | `backup/2026-09-27/2026-08-12-coord-decoder-wip` | 1 | 0 | 0 (0.0 MB) |
| `2026-08-13-cost-scaling-coordnet` | `backup/2026-09-27/2026-08-13-cost-scaling-coordnet-wip` | 0 | 1 | 0 (0.0 MB) |
| `2026-08-13-cost-scaling-cp` | `backup/2026-09-27/2026-08-13-cost-scaling-cp-wip` | 0 | 2 | 0 (0.0 MB) |
| `2026-08-14-burgers2d-coord-rom` | `backup/2026-09-27/2026-08-14-burgers2d-coord-rom-wip` | 0 | 14 | 0 (0.0 MB) |
| `2026-08-14-wave2d-coord-rom` | `backup/2026-09-27/2026-08-14-wave2d-coord-rom-wip` | 0 | 12 | 0 (0.0 MB) |
| `2026-08-16-burgers2d-rom-latent-stepping` | `backup/2026-09-27/2026-08-16-burgers2d-rom-latent-stepping-wip` | 0 | 206 | 0 (0.0 MB) |
| `2026-08-16-heat2d-rom-latent-stepping` | `backup/2026-09-27/2026-08-16-heat2d-rom-latent-stepping-wip` | 0 | 61 | 0 (0.0 MB) |
| `2026-08-16-poisson2d-rom-objective` | `backup/2026-09-27/2026-08-16-poisson2d-rom-objective-wip` | 0 | 417 | 0 (0.0 MB) |
| `2026-08-16-wave2d-rom-latent-stepping` | `backup/2026-09-27/2026-08-16-wave2d-rom-latent-stepping-wip` | 0 | 114 | 0 (0.0 MB) |
| `2026-08-17-cost-to-tolerance` | `backup/2026-09-27/2026-08-17-cost-to-tolerance-wip` | 0 | 234 | 0 (0.0 MB) |
| `2026-08-17-rom-warmstart-fom` | `backup/2026-09-27/2026-08-17-rom-warmstart-fom-wip` | 0 | 209 | 0 (0.0 MB) |
| `2026-08-19-burgers-1e3-10x` | `backup/2026-09-27/2026-08-19-burgers-1e3-10x-wip` | 11 | 836 | 0 (0.0 MB) |
| `2026-08-19-burgers-hybrid-1024` | `backup/2026-09-27/2026-08-19-burgers-hybrid-1024-wip` | 0 | 427 | 0 (0.0 MB) |
| `2026-08-19-nonlinear-decoder-architecture` | `backup/2026-09-27/2026-08-19-nonlinear-decoder-architecture-wip` | 0 | 872 | 0 (0.0 MB) |
| `2026-08-19-poisson-hybrid-1024` | `backup/2026-09-27/2026-08-19-poisson-hybrid-1024-wip` | 0 | 121 | 0 (0.0 MB) |
| `2026-08-20-burgers-hybrid-2048` | `backup/2026-09-27/2026-08-20-burgers-hybrid-2048-wip` | 0 | 13 | 0 (0.0 MB) |
| `2026-08-20-poisson-hybrid-2048` | `backup/2026-09-27/2026-08-20-poisson-hybrid-2048-wip` | 0 | 22 | 0 (0.0 MB) |
| `2026-08-20-two-pde-k64-eq4m` | `backup/2026-09-27/2026-08-20-two-pde-k64-eq4m-wip` | 2 | 26 | 0 (0.0 MB) |
| `2026-08-22-separable-decoder` | `backup/2026-09-27/2026-08-22-separable-decoder-wip` | 0 | 23 | 0 (0.0 MB) |
| `2026-08-23-n256-push` | `backup/2026-09-27/2026-08-23-n256-push-wip` | 0 | 575 | 0 (0.0 MB) |
| `2026-08-23-sepdec-n1024` | `backup/2026-09-27/2026-08-23-sepdec-n1024-wip` | 0 | 13 | 0 (0.0 MB) |
| `2026-08-23-sepdec-n128` | `backup/2026-09-27/2026-08-23-sepdec-n128-wip` | 0 | 42 | 0 (0.0 MB) |
| `2026-08-23-sepdec-n256` | `backup/2026-09-27/2026-08-23-sepdec-n256-wip` | 0 | 84 | 0 (0.0 MB) |
| `2026-08-23-sepdec-n512` | `backup/2026-09-27/2026-08-23-sepdec-n512-wip` | 0 | 63 | 0 (0.0 MB) |
| `2026-08-25-burgers-accuracy` | `backup/2026-09-27/2026-08-25-burgers-accuracy-wip` | 0 | 835 | 0 (0.0 MB) |
| `2026-08-25-eq-fidelity-ladder` | `backup/2026-09-27/2026-08-25-eq-fidelity-ladder-wip` | 0 | 140 | 0 (0.0 MB) |
| `2026-08-26-codesign` | `backup/2026-09-27/2026-08-26-codesign-wip` | 2 | 0 | 0 (0.0 MB) |
| `2026-08-26-eq-learned` | `backup/2026-09-27/2026-08-26-eq-learned-wip` | 0 | 534 | 0 (0.0 MB) |
| `2026-08-27-b1d-poissonqf` | `backup/2026-09-27/2026-08-27-b1d-poissonqf-wip` | 3 | 73 | 0 (0.0 MB) |
| `2026-08-27-nodes-mm` | `backup/2026-09-27/2026-08-27-nodes-mm-wip` | 1 | 160 | 1 (1.6 MB) |
| `2026-08-29-b1d-tensor` | `backup/2026-09-27/2026-08-29-b1d-tensor-wip` | 0 | 54 | 0 (0.0 MB) |
| `2026-08-29-b2d-tensor` | `backup/2026-09-27/2026-08-29-b2d-tensor-wip` | 0 | 294 | 0 (0.0 MB) |
| `2026-08-30-stokes-vector` | `backup/2026-09-27/2026-08-30-stokes-vector-wip` | 0 | 2 | 0 (0.0 MB) |
| `2026-09-03-burgers3d-tensor` | `backup/2026-09-27/2026-09-03-burgers3d-tensor-wip` | 5 | 48 | 7 (2.2 MB) |
| `2026-09-03-wave2d-mechanism` | `backup/2026-09-27/2026-09-03-wave2d-mechanism-wip` | 0 | 72 | 0 (0.0 MB) |
| `2026-09-06-b3d-anchor` | `backup/2026-09-27/2026-09-06-b3d-anchor-wip` | 0 | 24 | 0 (0.0 MB) |
| `2026-09-06-b3d-encoder` | `backup/2026-09-27/2026-09-06-b3d-encoder-wip` | 0 | 24 | 0 (0.0 MB) |
| `2026-09-06-b3d-mixture` | `backup/2026-09-27/2026-09-06-b3d-mixture-wip` | 0 | 12 | 0 (0.0 MB) |
| `2026-09-06-b3d-quadratic` | `backup/2026-09-27/2026-09-06-b3d-quadratic-wip` | 0 | 12 | 0 (0.0 MB) |
| `2026-09-06-burgers3d-repair` | `backup/2026-09-27/2026-09-06-burgers3d-repair-wip` | 0 | 516 | 0 (0.0 MB) |
| `2026-09-07-mr-burgers2d` | `backup/2026-09-27/2026-09-07-mr-burgers2d-wip` | 0 | 201 | 0 (0.0 MB) |
| `2026-09-07-mr-heat2d` | `backup/2026-09-27/2026-09-07-mr-heat2d-wip` | 0 | 69 | 0 (0.0 MB) |
| `2026-09-07-mr-poisson2d` | `backup/2026-09-27/2026-09-07-mr-poisson2d-wip` | 0 | 443 | 0 (0.0 MB) |
| `2026-09-07-mr-wave2d` | `backup/2026-09-27/2026-09-07-mr-wave2d-wip` | 0 | 735 | 0 (0.0 MB) |
| `2026-09-10-modcp-burgers2d` | `backup/2026-09-27/2026-09-10-modcp-burgers2d-wip` | 0 | 151 | 0 (0.0 MB) |
| `2026-09-10-modcp-wave2d` | `backup/2026-09-27/2026-09-10-modcp-wave2d-wip` | 0 | 276 | 0 (0.0 MB) |
| `2026-09-14-head-ablation` | `backup/2026-09-27/2026-09-14-head-ablation-wip` | 0 | 103 | 0 (0.0 MB) |
| `2026-09-14-mesh-ladder` | `backup/2026-09-27/2026-09-14-mesh-ladder-wip` | 0 | 370 | 0 (0.0 MB) |
| `2026-09-14-no-audit` | `backup/2026-09-27/2026-09-14-no-audit-wip` | 0 | 258 | 0 (0.0 MB) |
| `2026-09-14-no-burgers` | `backup/2026-09-27/2026-09-14-no-burgers-wip` | 0 | 197 | 0 (0.0 MB) |
| `2026-09-14-no-poisson` | `backup/2026-09-27/2026-09-14-no-poisson-wip` | 0 | 33 | 0 (0.0 MB) |
| `2026-09-14-paper-draft` | `backup/2026-09-27/2026-09-14-paper-draft-wip` | 0 | 8 | 0 (0.0 MB) |
| `2026-09-15-cheap-corrections` | `backup/2026-09-27/2026-09-15-cheap-corrections-wip` | 0 | 36 | 0 (0.0 MB) |
| `2026-09-15-head-refine` | `backup/2026-09-27/2026-09-15-head-refine-wip` | 0 | 74 | 0 (0.0 MB) |
| `2026-09-15-prior-dial` | `backup/2026-09-27/2026-09-15-prior-dial-wip` | 0 | 64 | 0 (0.0 MB) |
| `2026-09-16-b-head-train` | `backup/2026-09-27/2026-09-16-b-head-train-wip` | 0 | 61 | 0 (0.0 MB) |
| `2026-09-16-b-ladder-top` | `backup/2026-09-27/2026-09-16-b-ladder-top-wip` | 0 | 131 | 0 (0.0 MB) |
| `2026-09-16-p-bank-head` | `backup/2026-09-27/2026-09-16-p-bank-head-wip` | 0 | 58 | 0 (0.0 MB) |
| `2026-09-16-paper-refresh` | `backup/2026-09-27/2026-09-16-paper-refresh-wip` | 0 | 7 | 0 (0.0 MB) |
| `2026-09-16-q-ridge` | `backup/2026-09-27/2026-09-16-q-ridge-wip` | 0 | 223 | 0 (0.0 MB) |
| `2026-09-16-q-trajdirs` | `backup/2026-09-27/2026-09-16-q-trajdirs-wip` | 0 | 83 | 0 (0.0 MB) |
| `2026-09-17-b-eqtop` | `backup/2026-09-27/2026-09-17-b-eqtop-wip` | 0 | 176 | 0 (0.0 MB) |
| `2026-09-17-b-lowvisc` | `backup/2026-09-27/2026-09-17-b-lowvisc-wip` | 0 | 173 | 0 (0.0 MB) |
| `2026-09-17-b-panel` | `backup/2026-09-27/2026-09-17-b-panel-wip` | 0 | 253 | 0 (0.0 MB) |
| `2026-09-17-b-qxm` | `backup/2026-09-27/2026-09-17-b-qxm-wip` | 0 | 205 | 0 (0.0 MB) |
| `2026-09-17-b-seeds` | `backup/2026-09-27/2026-09-17-b-seeds-wip` | 0 | 356 | 0 (0.0 MB) |
| `2026-09-17-p-linear` | `backup/2026-09-27/2026-09-17-p-linear-wip` | 0 | 120 | 0 (0.0 MB) |
| `2026-09-17-w-ladder` | `backup/2026-09-27/2026-09-17-w-ladder-wip` | 0 | 93 | 0 (0.0 MB) |
| `2026-09-20-bank-floor` | `backup/2026-09-27/2026-09-20-bank-floor-wip` | 0 | 77 | 0 (0.0 MB) |
| `2026-09-20-hires-burgers` | `backup/2026-09-27/2026-09-20-hires-burgers-wip` | 0 | 214 | 0 (0.0 MB) |
| `2026-09-20-hires-heat` | `backup/2026-09-27/2026-09-20-hires-heat-wip` | 0 | 74 | 0 (0.0 MB) |
| `2026-09-20-hires-poisson` | `backup/2026-09-27/2026-09-20-hires-poisson-wip` | 0 | 90 | 0 (0.0 MB) |
| `2026-09-20-nmrom-baselines` | `backup/2026-09-27/2026-09-20-nmrom-baselines-wip` | 0 | 133 | 0 (0.0 MB) |
| `2026-09-20-paper-b3d` | `backup/2026-09-27/2026-09-20-paper-b3d-wip` | 5 | 616 | 9 (0.1 MB) |
| `2026-09-20-paper-h3d` | `backup/2026-09-27/2026-09-20-paper-h3d-wip` | 0 | 41 | 0 (0.0 MB) |
| `2026-09-20-paper-ns3d` | `backup/2026-09-27/2026-09-20-paper-ns3d-wip` | 233 | 463 | 0 (0.0 MB) |
| `2026-09-21-burgers-eqcert` | `backup/2026-09-27/2026-09-21-burgers-eqcert-wip` | 0 | 172 | 0 (0.0 MB) |
| `2026-09-21-burgers-heldout` | `backup/2026-09-27/2026-09-21-burgers-heldout-wip` | 0 | 338 | 0 (0.0 MB) |
| `2026-09-21-heat3d-bank` | `backup/2026-09-27/2026-09-21-heat3d-bank-wip` | 0 | 159 | 0 (0.0 MB) |
| `2026-09-21-ns3d-grok` | `backup/2026-09-27/2026-09-21-ns3d-grok-wip` | 23 | 0 | 0 (0.0 MB) |
| `2026-09-22-burgers-repanel` | `backup/2026-09-27/2026-09-22-burgers-repanel-wip` | 1 | 81 | 0 (0.0 MB) |
| `2026-09-22-ops-deeponet-b2d` | `backup/2026-09-27/2026-09-22-ops-deeponet-b2d-wip` | 0 | 66 | 0 (0.0 MB) |
| `2026-09-22-ops-timing-panel` | `backup/2026-09-27/2026-09-22-ops-timing-panel-wip` | 0 | 170 | 0 (0.0 MB) |
| `2026-09-22-quadratic-manifold` | `backup/2026-09-27/2026-09-22-quadratic-manifold-wip` | 0 | 83 | 0 (0.0 MB) |
| `2026-09-23-burgers-bank-knob` | `backup/2026-09-27/2026-09-23-burgers-bank-knob-wip` | 0 | 379 | 0 (0.0 MB) |
| `2026-09-23-burgers-compare-hires` | `backup/2026-09-27/2026-09-23-burgers-compare-hires-wip` | 0 | 391 | 0 (0.0 MB) |
| `2026-09-23-burgers2d-speed` | `backup/2026-09-27/2026-09-23-burgers2d-speed-wip` | 0 | 192 | 0 (0.0 MB) |
| `2026-09-23-burgers3d-retry` | `backup/2026-09-27/2026-09-23-burgers3d-retry-wip` | 0 | 170 | 0 (0.0 MB) |
| `2026-09-23-burgers3d-span` | `backup/2026-09-27/2026-09-23-burgers3d-span-wip` | 0 | 130 | 0 (0.0 MB) |
| `2026-09-23-heat-bank-knob` | `backup/2026-09-27/2026-09-23-heat-bank-knob-wip` | 0 | 48 | 0 (0.0 MB) |
| `2026-09-23-poisson-bank-knob` | `backup/2026-09-27/2026-09-23-poisson-bank-knob-wip` | 0 | 169 | 0 (0.0 MB) |
| `2026-09-23-poisson-bank-knob-3d` | `backup/2026-09-27/2026-09-23-poisson-bank-knob-3d-wip` | 0 | 246 | 0 (0.0 MB) |
| `2026-09-23-spectral-fom` | `backup/2026-09-27/2026-09-23-spectral-fom-wip` | 0 | 572 | 0 (0.0 MB) |
| `2026-09-24-burgers-eq-tol-knobs` | `backup/2026-09-27/2026-09-24-burgers-eq-tol-knobs-wip` | 0 | 141 | 0 (0.0 MB) |
| `2026-09-24-operators-all-pdes` | `backup/2026-09-27/2026-09-24-operators-all-pdes-wip` | 11761 | 0 | 71 (387.0 MB) |
| `2026-09-25-burgers2d-test` | `backup/2026-09-27/2026-09-25-burgers2d-test-wip` | 0 | 188 | 0 (0.0 MB) |
| `2026-09-25-poisson2d-test` | `backup/2026-09-27/2026-09-25-poisson2d-test-wip` | 0 | 76 | 0 (0.0 MB) |
| `2026-09-25-t2-burgers-test` | `backup/2026-09-27/2026-09-25-t2-burgers-test-wip` | 0 | 260 | 0 (0.0 MB) |
| `2026-09-25-t3-burgers-test` | `backup/2026-09-27/2026-09-25-t3-burgers-test-wip` | 0 | 110 | 0 (0.0 MB) |
| `Tunable-NM-ROM-Claude` | `backup/2026-09-27/Tunable-NM-ROM-Claude-wip` | 18 | 21 | 0 (0.0 MB) |

Skipped WIP files are all data (.npz/.pkl smoke outputs, checkpoints) — see `wip-skipped.tsv`.

