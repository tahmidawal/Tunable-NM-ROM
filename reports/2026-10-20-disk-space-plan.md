# Disk-space plan for the GB10 box, 2026-10-20

This report lists what can be deleted to free space on `/` (3.7 TB, about 31 GB free, 100 % full on 2026-10-20), sorted into SAFE, SAFE-AFTER-STEP and KEEP, with the exact command for each item. **Nothing has been deleted, moved or changed;** every command below waits for the user's approval. The numbers are measurements taken read-only on 2026-10-20 and are final for that date. The one soft number is the "found on the cluster" column, which matches files by name and size only; it is marked provisional wherever it appears. In short, **about 1137.9 GB can be freed with no risk** (mostly by shrinking the sparse checkout of 62 finished worktrees, which removes only copies of committed files that are still in `.git`). **About 1406.5 GB more** can be freed after copying local-only run data to the cluster and checking the copy with sha256. Almost every finished worktree still holds run outputs that exist nowhere else.

GB means $10^9$ bytes throughout. Worktree, cache and directory sizes are disk usage (`du`). Re-sparse figures are the logical sizes of the committed blobs that would leave the checkout; the Codex review measured the allocated blocks of those same paths and found them within 0.2 GB of the logical total. Ignored-data totals count a hardlinked file once per path; the de-duplicated figure counts each distinct content once.

## Totals per class

| Class | GB reclaimable | What it is |
|---|---:|---|
| SAFE | 1137.9 | Narrow the sparse checkout of 62 worktrees (1125.4 GB); pip cache 12.2 GB; `/tmp/node-compile-cache` 0.3 GB |
| SAFE-AFTER-STEP | 1406.5 (+ up to 22.1 from the uv cache) | Remove 86 worktrees after archiving their local-only data (1205.9 GB beyond the re-sparse; every 2026-10 lane, even the data-free `jcp-fom-pareto`, also needs the user's override of the lab-log retention note); the modcp raw tars in `artifacts/` (191.7 GB); the Burgers-campaign coordinator scratchpad in `/tmp` (8.9 GB); Codex audit scratch in `/tmp` (0.0 GB); `~/.cache/uv` after checking its cached environments |
| KEEP | — | `worktrees/2026-10-20-jcp-consolidated` (47.6 GB, the fork point); the shared `.git` (529.7 GB); the `main` checkout; older clones under `pod-ae-nmrom/` and `Dev/Tunable-NM-ROM`; this session's scratchpad |
| Other projects (report only) | — | Docker volumes 153GB (148.1GB (96%) reclaimable), Docker images 51.63GB, Hugging Face models 100.7 GB, `/tmp` scratch of other projects, `360Viserly` and other repos. The user decides; this plan does not recommend touching them. |

SAFE alone takes free space from about 31 GB to about 1168.9 GB.

## What was found

- **The worktrees use 2378.9 GB.** Of that, 1467.0 GB is checked-out copies of committed files, and 898.0 GB is ignored or untracked files (run arrays, checkpoints, logs, staged job code) that Git does not hold.
- **Every worktree is clean.** No worktree has a modified tracked file. Only two have untracked files that are not ignored: `jcp-wide-bank` and `operators-all-pdes`, and those files are listed in `LARGE_FILES.sha256` manifests.
- **No process was using any worktree at the time of the check** (2026-10-20, about 16:10 EDT, host PID namespace). No process had its working directory inside `worktrees/`, and no open file descriptor pointed into one (checked through `/proc/*/cwd` and `/proc/*/fd`). This is a snapshot: Procedure B re-checks immediately before each removal, including memory-mapped files. The only recent writer under the project is the Burgers-campaign coordinator session, whose scratchpad is in `/tmp`.
- **Most finished worktrees hold local-only data.** 778.8 GB of ignored files (counted per worktree; about 696.2 GB after removing byte-identical duplicates across worktrees) are neither in any Git blob (checked by `git hash-object`) nor matched on the cluster by name and size. The cluster namespaces of the September lanes were emptied after their results were pulled, as CLAUDE.md requires, so these local arrays are the only copies. They are reproducible only by rerunning GPU jobs. Only `jcp-fom-pareto` has no ignored data at all.
- **The 49 GB heat archive is no longer duplicated, but two other archives are.** No worktree has `experiments/mr-heat2d/runs/` checked out (0 copies). The big committed duplicates now are `experiments/ns3d/` (76.1 GB, checked out in 7 worktrees) and `experiments/paper-h3d/` (41.2–68.0 GB, checked out in 5 worktrees); `experiments/paper-b3d/` (59.7 GB, checked out in 1 worktree) is not duplicated. Re-sparsing reclaims every copy except in the worktree that owns each archive (`paper-ns3d`, `paper-h3d`, `paper-b3d`).
- **Backups.** Every JCP lane's `origin/codeonly/` mirror contains every lane-changed file of 50 MB or less. This was checked by applying the `sync_github.sh` rule to the mirror tree in a blob-less fetch. The September lanes' mirrors match their local tips (same commit subject and timestamp). **Seven branches have 2026-10-20 "preserve uncommitted files" commits that never reached GitHub:** `2026-09-20-paper-b3d`, `2026-09-20-paper-ns3d`, `2026-09-21-ns3d-grok`, `2026-09-22-burgers-repanel`, `2026-09-24-operators-all-pdes`, `2026-08-12-coord-decoder`, `2026-08-26-codesign`. The whole `operators-all-pdes` experiment (`experiments/ops-all`, 11,786 files) exists only in the local `.git`.
- **Two git behaviours make careless cleanup lose data:**
  1. `git worktree remove` (without `--force`) **silently deletes ignored files**. It refuses only when there are modified or untracked-but-not-ignored files. This was verified on a scratch repository with git 2.43.
  2. `git sparse-checkout set` in **cone mode** deletes directories that contain only ignored files; **non-cone mode keeps them** (also verified). Two worktrees (`2026-09-24-burgers-eq-tol-knobs`, `2026-09-25-burgers2d-test`) are in cone mode and are excluded from re-sparsing. Every command below passes `--no-cone`.
- **`.git` cannot be shrunk usefully.** Its packs hold no material duplication (about 1.0 GB of objects appear in more than one pack), and only 2.3 GB of objects are unreachable. `git gc` would need hundreds of GB of temporary space for a gain of a few GB, so it is not recommended.
- **The cluster has room.** `/cluster/tufts/paralab` has 3.4 TB free (28 % used), so archiving the local-only data there is feasible. The lab log's 2026-10-17 policy already moved raw arrays over 100 MB to `burgers-archive/` for this reason.

## Candidates

Run every command from `/home/tahmid/Dev/Tunable-NM-ROM-Claude` unless it says otherwise. Branches are never deleted. Removing a worktree keeps its branch, and `git worktree add worktrees/<name> exp/<name>` restores the checkout.

### SAFE: no data can be lost

| Path | Size GB | Class | Reason | Command |
|---|---:|---|---|---|
| `worktrees/2026-10-01-quadrature-study` (tracked checkout outside `quadrature-study`) | 80.4 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-10-01-quadrature-study sparse-checkout set --no-cone '/*' '!/*/' '/experiments/quadrature-study/'` |
| `worktrees/2026-10-01-quadrature-burgers3d` (tracked checkout outside `quadrature-burgers3d`) | 80.4 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-10-01-quadrature-burgers3d sparse-checkout set --no-cone '/*' '!/*/' '/experiments/quadrature-burgers3d/'` |
| `worktrees/2026-10-06-jcp-mechanism` (tracked checkout outside `jcp-mechanism`) | 80.4 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-10-06-jcp-mechanism sparse-checkout set --no-cone '/*' '!/*/' '/experiments/jcp-mechanism/'` |
| `worktrees/2026-10-06-jcp-references` (tracked checkout outside `jcp-references`) | 80.4 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-10-06-jcp-references sparse-checkout set --no-cone '/*' '!/*/' '/experiments/jcp-references/'` |
| `worktrees/2026-10-01-ns3d-coordnet-bank` (tracked checkout outside `ns3d-coordnet`) | 80.3 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-10-01-ns3d-coordnet-bank sparse-checkout set --no-cone '/*' '!/*/' '/experiments/ns3d-coordnet/'` |
| `worktrees/2026-09-21-ns3d-grok` (tracked checkout outside `ns3d-grok`) | 80.2 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-21-ns3d-grok sparse-checkout set --no-cone '/*' '!/*/' '/experiments/ns3d-grok/'` |
| `worktrees/2026-09-21-heat3d-bank` (tracked checkout outside `heat3d-bank`) | 42.6 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-21-heat3d-bank sparse-checkout set --no-cone '/*' '!/*/' '/experiments/heat3d-bank/'` |
| `worktrees/2026-09-23-heat-bank-knob` (tracked checkout outside `heat-bank-knob`) | 42.6 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-23-heat-bank-knob sparse-checkout set --no-cone '/*' '!/*/' '/experiments/heat-bank-knob/'` |
| `worktrees/2026-09-23-heat-compare-hires` (tracked checkout outside `heat-compare-hires`) | 42.6 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-23-heat-compare-hires sparse-checkout set --no-cone '/*' '!/*/' '/experiments/heat-compare-hires/'` |
| `worktrees/2026-09-20-hires-heat` (tracked checkout outside `hires-heat`) | 42.5 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-20-hires-heat sparse-checkout set --no-cone '/*' '!/*/' '/experiments/hires-heat/'` |
| `worktrees/2026-09-23-spectral-fom` (tracked checkout outside `spectral-fom`) | 31.7 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-23-spectral-fom sparse-checkout set --no-cone '/*' '!/*/' '/experiments/spectral-fom/'` |
| `worktrees/2026-09-23-poisson-bank-knob` (tracked checkout outside `poisson-bank-knob`) | 31.7 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-23-poisson-bank-knob sparse-checkout set --no-cone '/*' '!/*/' '/experiments/poisson-bank-knob/'` |
| `worktrees/2026-09-20-hires-poisson` (tracked checkout outside `hires-poisson`) | 31.6 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-20-hires-poisson sparse-checkout set --no-cone '/*' '!/*/' '/experiments/hires-poisson/'` |
| `worktrees/2026-09-23-burgers-bank-knob` (tracked checkout outside `burgers-bank-knob`) | 24.0 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-23-burgers-bank-knob sparse-checkout set --no-cone '/*' '!/*/' '/experiments/burgers-bank-knob/'` |
| `worktrees/2026-09-21-burgers-eqcert` (tracked checkout outside `burgers-eqcert`) | 24.0 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-21-burgers-eqcert sparse-checkout set --no-cone '/*' '!/*/' '/experiments/burgers-eqcert/'` |
| `worktrees/2026-09-21-burgers-heldout` (tracked checkout outside `burgers-heldout`) | 24.0 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-21-burgers-heldout sparse-checkout set --no-cone '/*' '!/*/' '/experiments/burgers-heldout/'` |
| `worktrees/2026-09-22-burgers-repanel` (tracked checkout outside `burgers-repanel`) | 24.0 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-22-burgers-repanel sparse-checkout set --no-cone '/*' '!/*/' '/experiments/burgers-repanel/'` |
| `worktrees/2026-09-23-burgers-compare-hires` (tracked checkout outside `burgers-compare-hires`) | 24.0 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-23-burgers-compare-hires sparse-checkout set --no-cone '/*' '!/*/' '/experiments/burgers-compare-hires/'` |
| `worktrees/2026-09-20-hires-burgers` (tracked checkout outside `hires-burgers`) | 24.0 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-20-hires-burgers sparse-checkout set --no-cone '/*' '!/*/' '/experiments/hires-burgers/'` |
| `worktrees/2026-09-22-quadratic-manifold` (tracked checkout outside `quadratic-manifold`) | 24.0 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-22-quadratic-manifold sparse-checkout set --no-cone '/*' '!/*/' '/experiments/quadratic-manifold/'` |
| `worktrees/2026-09-22-ops-deeponet-b2d` (tracked checkout outside `ops-deeponet-b2d`) | 17.0 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-22-ops-deeponet-b2d sparse-checkout set --no-cone '/*' '!/*/' '/experiments/ops-deeponet-b2d/'` |
| `worktrees/2026-09-22-ops-timing-panel` (tracked checkout outside `ops-timing-panel`) | 17.0 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-22-ops-timing-panel sparse-checkout set --no-cone '/*' '!/*/' '/experiments/ops-timing-panel/'` |
| `worktrees/2026-09-17-b-lowvisc` (tracked checkout outside `b-lowvisc`) | 9.4 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-17-b-lowvisc sparse-checkout set --no-cone '/*' '!/*/' '/experiments/b-lowvisc/'` |
| `worktrees/2026-09-17-b-seeds` (tracked checkout outside `b-seeds`) | 9.4 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-17-b-seeds sparse-checkout set --no-cone '/*' '!/*/' '/experiments/b-seeds/'` |
| `worktrees/2026-09-17-b-panel` (tracked checkout outside `b-panel`) | 9.4 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-17-b-panel sparse-checkout set --no-cone '/*' '!/*/' '/experiments/b-panel/'` |
| `worktrees/2026-09-17-b-qxm` (tracked checkout outside `b-qxm`) | 9.4 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-17-b-qxm sparse-checkout set --no-cone '/*' '!/*/' '/experiments/b-qxm/'` |
| `worktrees/2026-09-16-q-ridge` (tracked checkout outside `q-ridge`) | 7.5 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-16-q-ridge sparse-checkout set --no-cone '/*' '!/*/' '/experiments/q-ridge/'` |
| `worktrees/2026-09-16-q-trajdirs` (tracked checkout outside `q-trajdirs`) | 7.5 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-16-q-trajdirs sparse-checkout set --no-cone '/*' '!/*/' '/experiments/q-trajdirs/'` |
| `worktrees/2026-09-17-b-eqtop` (tracked checkout outside `b-eqtop, q-ridge`) | 7.5 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-17-b-eqtop sparse-checkout set --no-cone '/*' '!/*/' '/experiments/b-eqtop/' '/experiments/q-ridge/'` |
| `worktrees/2026-09-16-b-ladder-top` (tracked checkout outside `b-ladder-top`) | 6.2 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-16-b-ladder-top sparse-checkout set --no-cone '/*' '!/*/' '/experiments/b-ladder-top/'` |
| `worktrees/2026-09-20-nmrom-baselines` (tracked checkout outside `nmrom-baselines`) | 6.0 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-20-nmrom-baselines sparse-checkout set --no-cone '/*' '!/*/' '/experiments/nmrom-baselines/'` |
| `worktrees/2026-09-17-p-linear` (tracked checkout outside `p-linear`) | 5.6 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-17-p-linear sparse-checkout set --no-cone '/*' '!/*/' '/experiments/p-linear/'` |
| `worktrees/2026-09-20-bank-floor` (tracked checkout outside `bank-floor`) | 4.6 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-20-bank-floor sparse-checkout set --no-cone '/*' '!/*/' '/experiments/bank-floor/'` |
| `worktrees/2026-10-06-jcp-fom-pareto` (tracked checkout outside `jcp-fom-pareto`) | 4.3 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-10-06-jcp-fom-pareto sparse-checkout set --no-cone '/*' '!/*/' '/experiments/jcp-fom-pareto/'` |
| `worktrees/2026-10-06-jcp-hr-baselines` (tracked checkout outside `jcp-hr-baselines`) | 4.3 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-10-06-jcp-hr-baselines sparse-checkout set --no-cone '/*' '!/*/' '/experiments/jcp-hr-baselines/'` |
| `worktrees/2026-10-08-jcp-smooth-bank` (tracked checkout outside `jcp-smooth-bank`) | 4.3 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-10-08-jcp-smooth-bank sparse-checkout set --no-cone '/*' '!/*/' '/experiments/jcp-smooth-bank/'` |
| `worktrees/2026-10-08-jcp-time2` (tracked checkout outside `jcp-time2`) | 4.3 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-10-08-jcp-time2 sparse-checkout set --no-cone '/*' '!/*/' '/experiments/jcp-time2/'` |
| `worktrees/2026-10-08-jcp-wide-bank` (tracked checkout outside `jcp-wide-bank`) | 4.3 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-10-08-jcp-wide-bank sparse-checkout set --no-cone '/*' '!/*/' '/experiments/jcp-wide-bank/'` |
| `worktrees/2026-10-13-jcp-combined` (tracked checkout outside `jcp-combined`) | 4.3 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-10-13-jcp-combined sparse-checkout set --no-cone '/*' '!/*/' '/experiments/jcp-combined/'` |
| `worktrees/2026-10-14-jcp-speed3d` (tracked checkout outside `jcp-speed3d`) | 4.3 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-10-14-jcp-speed3d sparse-checkout set --no-cone '/*' '!/*/' '/experiments/jcp-speed3d/'` |
| `worktrees/2026-10-17-jcp-p1-scaling` (tracked checkout outside `jcp-p1-scaling`) | 4.3 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-10-17-jcp-p1-scaling sparse-checkout set --no-cone '/*' '!/*/' '/experiments/jcp-p1-scaling/'` |
| `worktrees/2026-10-18-jcp-budget` (tracked checkout outside `jcp-budget`) | 4.3 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-10-18-jcp-budget sparse-checkout set --no-cone '/*' '!/*/' '/experiments/jcp-budget/'` |
| `worktrees/2026-10-18-jcp-dims` (tracked checkout outside `jcp-dims`) | 4.3 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-10-18-jcp-dims sparse-checkout set --no-cone '/*' '!/*/' '/experiments/jcp-dims/'` |
| `worktrees/2026-10-18-jcp-fom2d` (tracked checkout outside `jcp-fom2d`) | 4.3 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-10-18-jcp-fom2d sparse-checkout set --no-cone '/*' '!/*/' '/experiments/jcp-fom2d/'` |
| `worktrees/2026-10-18-jcp-heads` (tracked checkout outside `jcp-heads`) | 4.3 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-10-18-jcp-heads sparse-checkout set --no-cone '/*' '!/*/' '/experiments/jcp-heads/'` |
| `worktrees/2026-10-18-jcp-qualify` (tracked checkout outside `jcp-qualify`) | 4.3 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-10-18-jcp-qualify sparse-checkout set --no-cone '/*' '!/*/' '/experiments/jcp-qualify/'` |
| `worktrees/2026-10-19-jcp-regimes` (tracked checkout outside `jcp-regimes`) | 4.3 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-10-19-jcp-regimes sparse-checkout set --no-cone '/*' '!/*/' '/experiments/jcp-regimes/'` |
| `worktrees/2026-10-19-jcp-robust` (tracked checkout outside `jcp-robust`) | 4.3 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-10-19-jcp-robust sparse-checkout set --no-cone '/*' '!/*/' '/experiments/jcp-robust/'` |
| `worktrees/2026-09-16-b-head-train` (tracked checkout outside `b-head-train`) | 3.2 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-16-b-head-train sparse-checkout set --no-cone '/*' '!/*/' '/experiments/b-head-train/'` |
| `worktrees/2026-09-16-p-bank-head` (tracked checkout outside `p-bank-head`) | 3.2 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-16-p-bank-head sparse-checkout set --no-cone '/*' '!/*/' '/experiments/p-bank-head/'` |
| `worktrees/2026-09-15-cheap-corrections` (tracked checkout outside `cheap-corrections`) | 3.2 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-15-cheap-corrections sparse-checkout set --no-cone '/*' '!/*/' '/experiments/cheap-corrections/'` |
| `worktrees/2026-09-15-head-refine` (tracked checkout outside `head-refine`) | 3.2 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-15-head-refine sparse-checkout set --no-cone '/*' '!/*/' '/experiments/head-refine/'` |
| `worktrees/2026-09-15-prior-dial` (tracked checkout outside `prior-dial`) | 3.2 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-15-prior-dial sparse-checkout set --no-cone '/*' '!/*/' '/experiments/prior-dial/'` |
| `worktrees/2026-09-23-burgers3d-retry` (tracked checkout outside `burgers3d-retry`) | 1.5 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-23-burgers3d-retry sparse-checkout set --no-cone '/*' '!/*/' '/experiments/burgers3d-retry/'` |
| `worktrees/2026-09-23-burgers2d-speed` (tracked checkout outside `burgers2d-speed`) | 1.4 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-23-burgers2d-speed sparse-checkout set --no-cone '/*' '!/*/' '/experiments/burgers2d-speed/'` |
| `worktrees/2026-09-23-burgers3d-span` (tracked checkout outside `burgers3d-span`) | 1.4 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-23-burgers3d-span sparse-checkout set --no-cone '/*' '!/*/' '/experiments/burgers3d-span/'` |
| `worktrees/2026-09-14-no-burgers` (tracked checkout outside `neural-operator-burgers`) | 1.4 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-14-no-burgers sparse-checkout set --no-cone '/*' '!/*/' '/experiments/neural-operator-burgers/'` |
| `worktrees/2026-09-14-head-ablation` (tracked checkout outside `head-ablation`) | 1.4 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-14-head-ablation sparse-checkout set --no-cone '/*' '!/*/' '/experiments/head-ablation/'` |
| `worktrees/2026-09-14-mesh-ladder` (tracked checkout outside `mesh-ladder`) | 1.4 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-14-mesh-ladder sparse-checkout set --no-cone '/*' '!/*/' '/experiments/mesh-ladder/'` |
| `worktrees/2026-09-14-no-audit` (tracked checkout outside `neural-operator-audit`) | 1.4 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-14-no-audit sparse-checkout set --no-cone '/*' '!/*/' '/experiments/neural-operator-audit/'` |
| `worktrees/2026-09-17-ns2d` (tracked checkout outside `ns2d`) | 1.4 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-17-ns2d sparse-checkout set --no-cone '/*' '!/*/' '/experiments/ns2d/'` |
| `worktrees/2026-09-17-w-ladder` (tracked checkout outside `w-ladder`) | 1.4 | SAFE | Clean worktree. The removed files are committed and stay in `.git`; ignored data stays on disk (non-cone mode keeps it). Save the current patterns first and undo by re-applying them (Procedure E). | `git -C worktrees/2026-09-17-w-ladder sparse-checkout set --no-cone '/*' '!/*/' '/experiments/w-ladder/'` |
| `~/.cache/pip` | 12.2 | SAFE | Download cache; the venvs do not depend on it. | `/home/tahmid/Dev/.venv/bin/pip cache purge` |
| `/tmp/node-compile-cache` | 0.3 | SAFE | Node compile cache, rebuilt automatically. | `rm -rf /tmp/node-compile-cache` |

### SAFE-AFTER-STEP: free once a prerequisite is done

"Local-only GB" counts ignored files found neither in any Git blob nor on the cluster (name+size). Every one of these worktrees needs Procedure B (archive and verify) before removal. The "Size GB" column is what removal frees **after** the SAFE re-sparse above has already run.

| Path | Size GB | Class | Reason / prerequisite | Command (after the prerequisite) |
|---|---:|---|---|---|
| `worktrees/2026-09-23-burgers-compare-hires` | 113.7 | SAFE-AFTER-STEP | Local-only 112.4 GB. archive 112.4 GB of local-only ignored data to the cluster and verify sha256; sha256-verify 1.0 GB matched on the cluster by name+size only. | `git worktree remove worktrees/2026-09-23-burgers-compare-hires` |
| `worktrees/2026-09-20-paper-b3d` | 99.2 | SAFE-AFTER-STEP | Local-only 35.1 GB. push 2026-10-20 commit(s) to GitHub first; archive 35.1 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-20-paper-b3d` |
| `worktrees/2026-09-20-paper-ns3d` | 94.4 | SAFE-AFTER-STEP | Local-only 35.4 GB. push 2026-10-20 commit(s) to GitHub first; archive 35.4 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-20-paper-ns3d` |
| `worktrees/2026-09-21-heat3d-bank` | 73.7 | SAFE-AFTER-STEP | Local-only 72.9 GB. archive 72.9 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-21-heat3d-bank` |
| `worktrees/2026-09-20-paper-h3d` | 70.5 | SAFE-AFTER-STEP | Local-only 1.7 GB. archive 1.7 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-20-paper-h3d` |
| `worktrees/2026-09-23-heat-bank-knob` | 50.0 | SAFE-AFTER-STEP | Local-only 49.8 GB. archive 49.8 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-23-heat-bank-knob` |
| `worktrees/2026-09-17-b-panel` | 44.0 | SAFE-AFTER-STEP | Local-only 29.0 GB. archive 29.0 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-17-b-panel` |
| `worktrees/2026-09-20-nmrom-baselines` | 39.0 | SAFE-AFTER-STEP | Local-only 36.8 GB. archive 36.8 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-20-nmrom-baselines` |
| `worktrees/2026-09-21-burgers-heldout` | 29.3 | SAFE-AFTER-STEP | Local-only 29.1 GB. archive 29.1 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-21-burgers-heldout` |
| `worktrees/2026-09-14-no-burgers` | 29.2 | SAFE-AFTER-STEP | Local-only 13.9 GB. archive 13.9 GB of local-only ignored data to the cluster and verify sha256; sha256-verify 7.1 GB matched on the cluster by name+size only. | `git worktree remove worktrees/2026-09-14-no-burgers` |
| `worktrees/2026-10-01-quadrature-burgers3d` | 28.8 | SAFE-AFTER-STEP | Local-only 25.0 GB. archive 25.0 GB of local-only ignored data to the cluster and verify sha256; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override. | `git worktree remove worktrees/2026-10-01-quadrature-burgers3d` |
| `worktrees/2026-09-25-burgers2d-test` | 23.8 | SAFE-AFTER-STEP | Local-only 22.6 GB. archive 22.6 GB of local-only ignored data to the cluster and verify sha256; sha256-verify 0.0 GB matched on the cluster by name+size only. | `git worktree remove worktrees/2026-09-25-burgers2d-test` |
| `worktrees/2026-10-19-jcp-robust` | 23.2 | SAFE-AFTER-STEP | Local-only 0.5 GB. archive 0.5 GB of local-only ignored data to the cluster and verify sha256; sha256-verify 0.3 GB matched on the cluster by name+size only; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override. | `git worktree remove worktrees/2026-10-19-jcp-robust` |
| `worktrees/2026-10-13-jcp-combined` | 22.7 | SAFE-AFTER-STEP | Local-only 2.9 GB. archive 2.9 GB of local-only ignored data to the cluster and verify sha256; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override. | `git worktree remove worktrees/2026-10-13-jcp-combined` |
| `worktrees/2026-09-23-burgers-bank-knob` | 22.6 | SAFE-AFTER-STEP | Local-only 21.9 GB. archive 21.9 GB of local-only ignored data to the cluster and verify sha256; sha256-verify 0.0 GB matched on the cluster by name+size only. | `git worktree remove worktrees/2026-09-23-burgers-bank-knob` |
| `worktrees/2026-09-20-paper-p3d` | 21.8 | SAFE-AFTER-STEP | Local-only 1.2 GB. archive 1.2 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-20-paper-p3d` |
| `worktrees/2026-09-23-spectral-fom` | 20.3 | SAFE-AFTER-STEP | Local-only 19.6 GB. archive 19.6 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-23-spectral-fom` |
| `worktrees/2026-09-20-hires-heat` | 19.1 | SAFE-AFTER-STEP | Local-only 19.0 GB. archive 19.0 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-20-hires-heat` |
| `worktrees/2026-09-25-t3-burgers-test` | 19.0 | SAFE-AFTER-STEP | Local-only 18.8 GB. archive 18.8 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-25-t3-burgers-test` |
| `worktrees/2026-10-18-jcp-dims` | 16.5 | SAFE-AFTER-STEP | Local-only 2.8 GB. archive 2.8 GB of local-only ignored data to the cluster and verify sha256; sha256-verify 7.2 GB matched on the cluster by name+size only; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override. | `git worktree remove worktrees/2026-10-18-jcp-dims` |
| `worktrees/2026-09-17-w-ladder` | 16.2 | SAFE-AFTER-STEP | Local-only 14.7 GB. archive 14.7 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-17-w-ladder` |
| `worktrees/2026-09-23-burgers2d-speed` | 15.9 | SAFE-AFTER-STEP | Local-only 15.6 GB. archive 15.6 GB of local-only ignored data to the cluster and verify sha256; sha256-verify 0.0 GB matched on the cluster by name+size only. | `git worktree remove worktrees/2026-09-23-burgers2d-speed` |
| `worktrees/2026-09-20-hires-burgers` | 15.7 | SAFE-AFTER-STEP | Local-only 15.3 GB. archive 15.3 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-20-hires-burgers` |
| `worktrees/2026-09-17-p-linear` | 15.2 | SAFE-AFTER-STEP | Local-only 9.5 GB. archive 9.5 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-17-p-linear` |
| `worktrees/2026-09-17-b-seeds` | 14.4 | SAFE-AFTER-STEP | Local-only 9.7 GB. archive 9.7 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-17-b-seeds` |
| `worktrees/2026-10-08-jcp-wide-bank` | 13.8 | SAFE-AFTER-STEP | Local-only 7.9 GB. archive 7.9 GB of local-only ignored data to the cluster and verify sha256; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override. | `git worktree remove --force worktrees/2026-10-08-jcp-wide-bank` |
| `worktrees/2026-09-21-burgers-eqcert` | 13.7 | SAFE-AFTER-STEP | Local-only 13.4 GB. archive 13.4 GB of local-only ignored data to the cluster and verify sha256; sha256-verify 0.0 GB matched on the cluster by name+size only. | `git worktree remove worktrees/2026-09-21-burgers-eqcert` |
| `worktrees/2026-10-14-jcp-speed3d` | 11.7 | SAFE-AFTER-STEP | Local-only 1.9 GB. archive 1.9 GB of local-only ignored data to the cluster and verify sha256; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override. | `git worktree remove worktrees/2026-10-14-jcp-speed3d` |
| `worktrees/2026-10-06-jcp-references` | 11.2 | SAFE-AFTER-STEP | Local-only 0.0 GB. sha256-verify 22.3 GB matched on the cluster by name+size only; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override. | `git worktree remove worktrees/2026-10-06-jcp-references` |
| `worktrees/2026-09-23-heat-compare-hires` | 10.4 | SAFE-AFTER-STEP | Local-only 10.4 GB. archive 10.4 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-23-heat-compare-hires` |
| `worktrees/2026-09-23-burgers3d-retry` | 9.6 | SAFE-AFTER-STEP | Local-only 8.9 GB. archive 8.9 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-23-burgers3d-retry` |
| `worktrees/2026-09-14-no-audit` | 9.4 | SAFE-AFTER-STEP | Local-only 1.8 GB. archive 1.8 GB of local-only ignored data to the cluster and verify sha256; sha256-verify 2.2 GB matched on the cluster by name+size only. | `git worktree remove worktrees/2026-09-14-no-audit` |
| `worktrees/2026-10-06-jcp-hr-baselines` | 9.4 | SAFE-AFTER-STEP | Local-only 0.5 GB. archive 0.5 GB of local-only ignored data to the cluster and verify sha256; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override. | `git worktree remove worktrees/2026-10-06-jcp-hr-baselines` |
| `worktrees/2026-09-23-ns3d-operators` | 9.0 | SAFE-AFTER-STEP | Local-only 8.7 GB. archive 8.7 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-23-ns3d-operators` |
| `worktrees/2026-10-17-jcp-p1-scaling` | 8.7 | SAFE-AFTER-STEP | Local-only 5.8 GB. archive 5.8 GB of local-only ignored data to the cluster and verify sha256; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override. | `git worktree remove worktrees/2026-10-17-jcp-p1-scaling` |
| `worktrees/2026-09-15-head-refine` | 8.1 | SAFE-AFTER-STEP | Local-only 5.3 GB. archive 5.3 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-15-head-refine` |
| `worktrees/2026-09-21-ns3d-grok` | 7.1 | SAFE-AFTER-STEP | Local-only 7.2 GB. push 2026-10-20 commit(s) to GitHub first; archive 7.2 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-21-ns3d-grok` |
| `worktrees/2026-09-16-p-bank-head` | 6.8 | SAFE-AFTER-STEP | Local-only 4.2 GB. archive 4.2 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-16-p-bank-head` |
| `worktrees/2026-09-25-t2-burgers-test` | 6.5 | SAFE-AFTER-STEP | Local-only 6.2 GB. archive 6.2 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-25-t2-burgers-test` |
| `worktrees/2026-09-16-q-ridge` | 6.1 | SAFE-AFTER-STEP | Local-only 3.8 GB. archive 3.8 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-16-q-ridge` |
| `worktrees/2026-10-08-jcp-time2` | 5.8 | SAFE-AFTER-STEP | Local-only 4.2 GB. archive 4.2 GB of local-only ignored data to the cluster and verify sha256; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override. | `git worktree remove worktrees/2026-10-08-jcp-time2` |
| `worktrees/2026-09-14-head-ablation` | 5.7 | SAFE-AFTER-STEP | Local-only 3.6 GB. archive 3.6 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-14-head-ablation` |
| `worktrees/2026-09-20-bank-floor` | 5.6 | SAFE-AFTER-STEP | Local-only 5.4 GB. archive 5.4 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-20-bank-floor` |
| `worktrees/2026-10-08-jcp-smooth-bank` | 5.5 | SAFE-AFTER-STEP | Local-only 4.3 GB. archive 4.3 GB of local-only ignored data to the cluster and verify sha256; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override. | `git worktree remove worktrees/2026-10-08-jcp-smooth-bank` |
| `worktrees/2026-09-17-b-qxm` | 5.5 | SAFE-AFTER-STEP | Local-only 3.3 GB. archive 3.3 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-17-b-qxm` |
| `worktrees/2026-09-23-burgers3d-span` | 5.4 | SAFE-AFTER-STEP | Local-only 4.4 GB. archive 4.4 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-23-burgers3d-span` |
| `worktrees/2026-09-15-cheap-corrections` | 5.1 | SAFE-AFTER-STEP | Local-only 2.0 GB. archive 2.0 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-15-cheap-corrections` |
| `worktrees/2026-10-18-jcp-qualify` | 4.9 | SAFE-AFTER-STEP | Local-only 0.8 GB. archive 0.8 GB of local-only ignored data to the cluster and verify sha256; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override. | `git worktree remove worktrees/2026-10-18-jcp-qualify` |
| `worktrees/2026-10-01-quadrature-study` | 4.8 | SAFE-AFTER-STEP | Local-only 4.2 GB. archive 4.2 GB of local-only ignored data to the cluster and verify sha256; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override. | `git worktree remove worktrees/2026-10-01-quadrature-study` |
| `worktrees/2026-09-17-b-lowvisc` | 4.6 | SAFE-AFTER-STEP | Local-only 3.1 GB. archive 3.1 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-17-b-lowvisc` |
| `worktrees/2026-09-25-t2-ns3d-test` | 4.6 | SAFE-AFTER-STEP | Local-only 4.4 GB. archive 4.4 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-25-t2-ns3d-test` |
| `worktrees/2026-10-18-jcp-budget` | 4.6 | SAFE-AFTER-STEP | Local-only 0.0 GB. sha256-verify 2.3 GB matched on the cluster by name+size only; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override. | `git worktree remove worktrees/2026-10-18-jcp-budget` |
| `worktrees/2026-09-16-b-head-train` | 4.5 | SAFE-AFTER-STEP | Local-only 3.0 GB. archive 3.0 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-16-b-head-train` |
| `worktrees/2026-09-15-prior-dial` | 4.5 | SAFE-AFTER-STEP | Local-only 2.9 GB. archive 2.9 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-15-prior-dial` |
| `worktrees/2026-09-16-b-ladder-top` | 4.1 | SAFE-AFTER-STEP | Local-only 2.5 GB. archive 2.5 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-16-b-ladder-top` |
| `worktrees/2026-09-17-ns2d` | 3.8 | SAFE-AFTER-STEP | Local-only 0.7 GB. archive 0.7 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-17-ns2d` |
| `worktrees/2026-09-17-b-eqtop` | 3.8 | SAFE-AFTER-STEP | Local-only 1.1 GB. archive 1.1 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-17-b-eqtop` |
| `worktrees/2026-09-22-ops-timing-panel` | 3.6 | SAFE-AFTER-STEP | Local-only 2.6 GB. archive 2.6 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-22-ops-timing-panel` |
| `worktrees/2026-08-25-burgers-accuracy` | 3.6 | SAFE-AFTER-STEP | Local-only 2.5 GB. archive 2.5 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-08-25-burgers-accuracy` |
| `worktrees/2026-09-24-burgers-eq-tol-knobs` | 3.2 | SAFE-AFTER-STEP | Local-only 2.1 GB. archive 2.1 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-24-burgers-eq-tol-knobs` |
| `worktrees/2026-09-25-ns3d-test` | 2.9 | SAFE-AFTER-STEP | Local-only 2.9 GB. archive 2.9 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-25-ns3d-test` |
| `worktrees/2026-10-19-jcp-regimes` | 2.9 | SAFE-AFTER-STEP | Local-only 1.4 GB. archive 1.4 GB of local-only ignored data to the cluster and verify sha256; sha256-verify 0.1 GB matched on the cluster by name+size only; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override. | `git worktree remove worktrees/2026-10-19-jcp-regimes` |
| `worktrees/2026-09-23-poisson-bank-knob` | 2.4 | SAFE-AFTER-STEP | Local-only 2.2 GB. archive 2.2 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-23-poisson-bank-knob` |
| `worktrees/2026-08-25-sepdec-consolidated` | 2.1 | SAFE-AFTER-STEP | Local-only 1.2 GB. archive 1.2 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-08-25-sepdec-consolidated` |
| `worktrees/2026-09-14-mesh-ladder` | 2.0 | SAFE-AFTER-STEP | Local-only 0.9 GB. archive 0.9 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-14-mesh-ladder` |
| `worktrees/2026-08-26-codesign` | 1.9 | SAFE-AFTER-STEP | Local-only 1.0 GB. push 2026-10-20 commit(s) to GitHub first; archive 1.0 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-08-26-codesign` |
| `worktrees/2026-10-06-jcp-mechanism` | 1.9 | SAFE-AFTER-STEP | Local-only 1.5 GB. archive 1.5 GB of local-only ignored data to the cluster and verify sha256; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override. | `git worktree remove worktrees/2026-10-06-jcp-mechanism` |
| `worktrees/2026-09-22-ops-deeponet-b2d` | 1.8 | SAFE-AFTER-STEP | Local-only 0.3 GB. archive 0.3 GB of local-only ignored data to the cluster and verify sha256; sha256-verify 0.5 GB matched on the cluster by name+size only. | `git worktree remove worktrees/2026-09-22-ops-deeponet-b2d` |
| `worktrees/2026-10-18-jcp-heads` | 1.7 | SAFE-AFTER-STEP | Local-only 0.1 GB. archive 0.1 GB of local-only ignored data to the cluster and verify sha256; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override. | `git worktree remove worktrees/2026-10-18-jcp-heads` |
| `worktrees/2026-09-20-hires-poisson` | 1.7 | SAFE-AFTER-STEP | Local-only 1.5 GB. archive 1.5 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-20-hires-poisson` |
| `worktrees/2026-08-30-stokes-vector` | 1.6 | SAFE-AFTER-STEP | Local-only 0.5 GB. archive 0.5 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-08-30-stokes-vector` |
| `worktrees/2026-09-22-burgers-repanel` | 1.3 | SAFE-AFTER-STEP | Local-only 0.7 GB. push 2026-10-20 commit(s) to GitHub first; archive 0.7 GB of local-only ignored data to the cluster and verify sha256; sha256-verify 0.5 GB matched on the cluster by name+size only. | `git worktree remove worktrees/2026-09-22-burgers-repanel` |
| `worktrees/2026-09-22-quadratic-manifold` | 1.3 | SAFE-AFTER-STEP | Local-only 1.2 GB. archive 1.2 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-22-quadratic-manifold` |
| `worktrees/2026-09-16-q-trajdirs` | 1.2 | SAFE-AFTER-STEP | Local-only 0.6 GB. archive 0.6 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-16-q-trajdirs` |
| `worktrees/2026-10-18-jcp-fom2d` | 1.1 | SAFE-AFTER-STEP | Local-only 0.1 GB. archive 0.1 GB of local-only ignored data to the cluster and verify sha256; sha256-verify 0.1 GB matched on the cluster by name+size only; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override. | `git worktree remove worktrees/2026-10-18-jcp-fom2d` |
| `worktrees/2026-08-14-burgers2d-coord-rom` | 0.9 | SAFE-AFTER-STEP | Local-only 0.9 GB. archive 0.9 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-08-14-burgers2d-coord-rom` |
| `worktrees/2026-08-14-wave2d-coord-rom` | 0.8 | SAFE-AFTER-STEP | Local-only 0.8 GB. archive 0.8 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-08-14-wave2d-coord-rom` |
| `worktrees/2026-08-13-heat2d-coord-decoder` | 0.8 | SAFE-AFTER-STEP | Local-only 0.8 GB. archive 0.8 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-08-13-heat2d-coord-decoder` |
| `worktrees/2026-09-24-operators-all-pdes` | 0.6 | SAFE-AFTER-STEP | Local-only 0.1 GB. push 2026-10-20 commit(s) to GitHub first; archive 0.1 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove --force worktrees/2026-09-24-operators-all-pdes` |
| `worktrees/2026-08-12-coord-decoder` | 0.5 | SAFE-AFTER-STEP | Local-only 0.5 GB. push 2026-10-20 commit(s) to GitHub first; archive 0.5 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-08-12-coord-decoder` |
| `worktrees/2026-09-25-poisson2d-test` | 0.5 | SAFE-AFTER-STEP | Local-only 0.4 GB. archive 0.4 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-09-25-poisson2d-test` |
| `worktrees/2026-10-01-ns3d-coordnet-bank` | 0.4 | SAFE-AFTER-STEP | Local-only 0.4 GB. archive 0.4 GB of local-only ignored data to the cluster and verify sha256; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override. | `git worktree remove worktrees/2026-10-01-ns3d-coordnet-bank` |
| `worktrees/2026-08-19-poisson-hybrid-1024` | 0.2 | SAFE-AFTER-STEP | Local-only 0.1 GB. archive 0.1 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-08-19-poisson-hybrid-1024` |
| `worktrees/2026-08-16-poisson2d-rom-objective` | 0.1 | SAFE-AFTER-STEP | Local-only 0.1 GB. archive 0.1 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-08-16-poisson2d-rom-objective` |
| `worktrees/2026-10-06-jcp-fom-pareto` | 0.1 | SAFE-AFTER-STEP | Local-only 0.0 GB. lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override. | `git worktree remove worktrees/2026-10-06-jcp-fom-pareto` |
| `worktrees/2026-08-13-gn-tolerance-sweep` | 0.1 | SAFE-AFTER-STEP | Local-only 0.1 GB. archive 0.1 GB of local-only ignored data to the cluster and verify sha256. | `git worktree remove worktrees/2026-08-13-gn-tolerance-sweep` |
| `artifacts/modcp-eq/2026-09-10/raw/` (8 tars, on `main`) | 191.7 | SAFE-AFTER-STEP | Ignored pilot field archives of the 2026-09-10 modified-CP lanes. Their remote run directories were deleted, so this is the only copy; SHA256s are recorded in `LAB-LOG.md`. Procedure C. | `rm -r artifacts/modcp-eq/2026-09-10/raw` |
| `/tmp/claude-1002/-home-tahmid-Dev-pod-ae-nmrom-Tunable-NM-ROM-Claude/2f07cec8-5e56-4c01-8246-62353b26fb5e` | 8.9 | SAFE-AFTER-STEP | Scratchpad of the Burgers-campaign coordinator (last written 2026-10-19). About 90 % of its bytes match `burgers-archive/` by name+size (provisional). Confirm the session is closed, then archive every file and verify (Procedure D). | `rm -rf /tmp/claude-1002/-home-tahmid-Dev-pod-ae-nmrom-Tunable-NM-ROM-Claude/2f07cec8-5e56-4c01-8246-62353b26fb5e` |
| `/tmp/jcp-{budget,dims,robust}-audit-scratch` | 0.0 | SAFE-AFTER-STEP | Codex audit scratch. 56 of its files (audit scripts, recomputed outputs) are in no Git object (found by the Codex review). Procedure D. | `rm -rf /tmp/jcp-budget-audit-scratch /tmp/jcp-dims-audit-scratch /tmp/jcp-robust-audit-scratch` |
| `~/.cache/uv` | up to 22.1 | SAFE-AFTER-STEP | The installed uv (0.11.19) `cache prune` also removes cached environments, and that version predates uv's fix for following symlinks while pruning (Codex review). The cache holds two `environments-v2` entries. First confirm nothing uses them, check that no installed environment symlinks into the cache, resolving each link: `find /home/tahmid -xdev -type l -print0 > ~/disk-plan/links.z && xargs -0 -r realpath -m -- < ~/disk-plan/links.z > ~/disk-plan/links.resolved && { grep -c /.cache/uv/ ~/disk-plan/links.resolved \|\| test $? = 1; }` must exit 0 and print `0`, and upgrade uv; never use `uv cache clean`. | `uv cache prune` (after the check) |

### KEEP

| Path | Size GB | Class | Reason | Command |
|---|---:|---|---|---|
| `worktrees/2026-10-20-jcp-consolidated` | 47.6 | KEEP | The current fork point (CLAUDE.md); new lanes branch from it. | none |
| `.git` (shared) | 529.7 | KEEP | The only complete copy of every branch, including the archive blobs GitHub rejects. | none |
| `/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM` | 26.3 | KEEP | An older separate clone (not the symlink) with 3 modified files and 16 GB of `burgers-transfer/` data; not audited for uniqueness. | none (review separately) |
| `/home/tahmid/Dev/pod-ae-nmrom/{Tunable-NM-ROM-podmlp,cascade-nmrom}` | 3.4 | KEEP | Older working copies; small. | none |
| `/home/tahmid/Dev/Tunable-NM-ROM` | 5.2 | KEEP | Older clone with 23 changed paths and an open shell in it. | none |

### Other projects (report only, never touched by this plan)

| Path | Size GB | Note |
|---|---:|---|
| Docker local volumes (`/var/lib/docker`, same disk) | 153GB | 148.1GB (96%) reclaimable: 447 dangling anonymous volumes, most likely test databases of 360Viserly/PlanSync. The named `plansync-devdb-data` volume is in use. The owner of those projects decides on `docker volume prune`. |
| Docker images / build cache | 51.63GB / 1.979GB | 16.21GB (31%) of the images reclaimable; `vllm/vllm-openai:cu130-nightly` is 32.7 GB. |
| `~/.cache/huggingface` | 100.7 | Nemotron-3-Super-120B (75 GB) and Qwen3-Coder-30B (17 GB) model weights for the local-LLM setup. |
| `/tmp/claude-1002/-home-tahmid-Dev-ai-researcher-numerical-physics/4563badc-…` | 41.1 | Session scratch of `ai-researcher-numerical-physics` (`scratchpad/pull`, last written 2026-10-06). |
| `/tmp/claude-1002/-home-tahmid-360Viserly*` | 14.3 | 360Viserly session scratch (deps venvs). |
| `/home/tahmid/360Viserly` | 44.5 | Active project. |
| `/home/tahmid/Dev/Local-High-Frequency-Information-Aware-Operator-` | 29.8 |  |
| `/home/tahmid/Dev/Final_Project` | 29.0 |  |
| `/home/tahmid/Dev/Octree-Refine-Net` | 16.0 |  |
| `/home/tahmid/Dev/ae-research` | 12.5 | Holds the Tufts helper scripts; keep. |
| `/home/tahmid/Dev/.venv` | 10.7 | The required local JAX environment; keep. |
| `/home/tahmid/Dev/vllm-spark` | 9.2 |  |

## Procedures

### 0. Shared guard functions (used by B, C and D)

Save once as `~/disk-plan/guards.sh`. Every function fails closed: a failed check exits before anything is deleted.

```bash
# ~/disk-plan/guards.sh -- sourced by procedures B, C and D
set -euo pipefail; export LC_ALL=C
REPO=/home/tahmid/Dev/Tunable-NM-ROM-Claude
CL=/cluster/tufts/paralab/tawal01/worktree-archive

not_in_use() {   # $1 = absolute path. Fails if any of this user's processes has its cwd, an open file or a mapping under it.
  [ "$(cat /proc/1/comm)" = systemd ] || { echo "not in the host PID namespace: cannot check"; exit 1; }
  local me p out hits=""
  me=$(id -u)
  for p in /proc/[0-9]*; do
    [ "$(stat -c %u "$p" 2>/dev/null)" = "$me" ] || continue   # other users' processes are not inspectable (see note)
    if ! out=$( { readlink "$p/cwd" && ls -l "$p/fd" && cat "$p/maps"; } 2>/dev/null ); then
      [ -d "$p" ] || continue                                    # the process exited during the scan
      case "$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null)" in   # non-dumpable session daemons; they never open project files
        "/usr/lib/systemd/systemd --user "*|"(sd-pam)"*|"/usr/bin/ssh-agent "*|"/usr/bin/gpg-agent "*) continue ;;
      esac
      echo "cannot inspect own process ${p#/proc/}: refusing"; exit 1
    fi
    case "$out" in *"$1"*) hits="$hits ${p#/proc/}";; esac
  done
  [ -z "$hits" ] || { echo "IN USE: $1 (PIDs$hits)"; exit 1; }
}
save_sparse_state() {   # $1 = worktree name. Saves the original sparse state once (never overwrites), atomically.
  local B=~/disk-plan/sparse-before w=$REPO/worktrees/$1 sp cone rc
  mkdir -p "$B"; [ ! -e "$B/$1.state" ] || return 0
  sp=$(git -C "$w" config --worktree --get core.sparseCheckout) || { rc=$?; [ $rc = 1 ] || { echo "git config failed"; exit 1; }; sp=unset; }
  cone=$(git -C "$w" config --worktree --get core.sparseCheckoutCone) || { rc=$?; [ $rc = 1 ] || { echo "git config failed"; exit 1; }; cone=unset; }
  if [ "$sp" = true ]; then git -C "$w" sparse-checkout list > "$B/$1.txt.tmp" || { echo "sparse-checkout list failed"; exit 1; }; mv "$B/$1.txt.tmp" "$B/$1.txt"; fi
  printf 'sparseCheckout=%s\nsparseCheckoutCone=%s\n' "$sp" "$cone" > "$B/$1.state.tmp" && mv -n "$B/$1.state.tmp" "$B/$1.state"
}
make_manifests() {   # $1 = NUL-separated path list (relative to cwd), $2 = output dir
  local f; : > "$2/reg.z"; : > "$2/lnk.z"
  while IFS= read -r -d '' f; do
    case "$f" in *$'\n'*|*$'\t'*) echo "unsupported file name: $f"; exit 1;; esac
    if [ -L "$f" ]; then printf '%s\0' "$f" >> "$2/lnk.z"
    elif [ -f "$f" ]; then printf '%s\0' "$f" >> "$2/reg.z"
    else echo "unexpected file type: $f"; exit 1; fi
  done < "$1"
  xargs -0 -r sha256sum < "$2/reg.z" > "$2/ARCHIVE-FILES.sha256" || { echo "hashing failed"; exit 1; }
  (while IFS= read -r -d '' f; do printf '%s\t%s\n' "$f" "$(readlink "$f")"; done < "$2/lnk.z") | sort > "$2/ARCHIVE-LINKS.txt"
}
upload_and_verify() {   # $1 = NUL list, $2 = manifest dir, $3 = NEW remote dir (must not exist); run from the source root
  timeout 60 ssh -o ConnectTimeout=20 tufts-login "mkdir -p '$(dirname "$3")' && mkdir '$3'" \
    || { echo "remote $3 exists or cannot be created: earlier archives are never overwritten"; exit 1; }
  timeout 36000 rsync -aH --partial --from0 --files-from="$1" ./ "tufts-login:$3/"
  rsync -a "$2/ARCHIVE-FILES.sha256" "$2/ARCHIVE-LINKS.txt" "tufts-login:$3/"
  timeout 36000 ssh -o ConnectTimeout=20 tufts-login "cd '$3' && LC_ALL=C sha256sum -c --quiet ARCHIVE-FILES.sha256" \
    || { echo "REMOTE VERIFY FAILED: nothing deleted"; exit 1; }
  timeout 600 ssh -o ConnectTimeout=20 tufts-login "cd '$3' && find . -type l -printf '%P\t%l\n' | LC_ALL=C sort" \
    | diff - "$2/ARCHIVE-LINKS.txt" || { echo "REMOTE SYMLINK MISMATCH: nothing deleted"; exit 1; }
}
recheck_unchanged() {   # $1 = command printing the NUL list, $2 = original list, $3 = original manifest dir, $4 = scratch dir
  mkdir -p "$4"; eval "$1" > "$4/list.z" || { echo "listing failed"; exit 1; }
  cmp "$4/list.z" "$2" || { echo "FILE SET CHANGED: nothing deleted"; exit 1; }
  make_manifests "$4/list.z" "$4"   # full re-read of every byte
  cmp "$4/ARCHIVE-FILES.sha256" "$3/ARCHIVE-FILES.sha256" || { echo "CONTENT CHANGED: nothing deleted"; exit 1; }
  cmp "$4/ARCHIVE-LINKS.txt" "$3/ARCHIVE-LINKS.txt" || { echo "SYMLINKS CHANGED: nothing deleted"; exit 1; }
}
record_on_main() {   # $1 = record name, $2 = manifest dir, $3 = remote dir. Commits the record on main and pushes it.
  local d=reports/archive-records/$1
  mkdir -p "$REPO/$d"; cp "$2/ARCHIVE-FILES.sha256" "$2/ARCHIVE-LINKS.txt" "$REPO/$d/"
  printf 'Archived %s\nsource: %s\nremote: tufts-login:%s\nverified: remote sha256sum -c and symlink list; local full re-read before deletion\n' \
    "$(date -u +%FT%TZ)" "$1" "$3" > "$REPO/$d/ARCHIVE-LOCATION.txt"
  git -C "$REPO" add -f -- "$d"
  git -C "$REPO" commit -q -m "Archive record: $1 -> $3 (sha256-verified)" -- "$d"
  git -C "$REPO" push -q origin main            # a rejected push (remote ahead) exits here
  git -C "$REPO" fetch -q origin main
  for f in ARCHIVE-FILES.sha256 ARCHIVE-LINKS.txt ARCHIVE-LOCATION.txt; do
    test "$(git -C "$REPO" rev-parse "FETCH_HEAD:$d/$f")" = "$(git -C "$REPO" rev-parse "HEAD:$d/$f")" \
      || { echo "RECORD NOT ON GITHUB: nothing deleted"; exit 1; }
  done
}
```

The `/proc` scan inspects this user's processes and refuses if one of them cannot be read. Other users' processes cannot be inspected without root; every Claude, Codex and job process on this box runs as `tahmid`. Each archive goes to a new, time-stamped remote directory, so an earlier archive is never overwritten. A process snapshot cannot stop an idle session from resuming later, so first close (or confirm finished) any Claude/Codex session whose working directory is the target, and do not start one there until the procedure ends.

### A. Back up the seven unmirrored branches (do this first, whatever else is approved)

These commits live only in the local `.git`. Removing a worktree does not touch them, but a disk failure would lose them. Two branches were always pushed directly and only gained tiny commits (largest file 25 kB), so they fast-forward:

```bash
set -euo pipefail; cd /home/tahmid/Dev/Tunable-NM-ROM-Claude
git push origin exp/2026-08-12-coord-decoder        # non-forced; origin is at 60afad2d7, the parent of the local tip
test "$(git ls-remote origin refs/heads/exp/2026-08-12-coord-decoder | cut -f1)" = "$(git rev-parse exp/2026-08-12-coord-decoder)" || { echo "push of 2026-08-12-coord-decoder not confirmed"; exit 1; }
git push origin exp/2026-08-26-codesign        # non-forced; origin is at da96a5853, the parent of the local tip
test "$(git ls-remote origin refs/heads/exp/2026-08-26-codesign | cut -f1)" = "$(git rev-parse exp/2026-08-26-codesign)" || { echo "push of 2026-08-26-codesign not confirmed"; exit 1; }
```

The other five are mirrored through `origin/codeonly/`. Each mirror is rebuilt with the lane `sync_github.sh` from the consolidated worktree, with the base pair **pinned**. `LOCAL_BASE` is the last local commit already mirrored. `REMOTE_BASE`, and the mirror commit's parent, are the mirror head checked on 2026-10-20. If the remote has moved since, the non-forced push is rejected and the script stops. Afterwards the pushed tree is checked path by path. The changed files are 5 MB or smaller, so nothing is skipped.

```bash
set -euo pipefail; cd /home/tahmid/Dev/Tunable-NM-ROM-Claude
TPL=worktrees/2026-10-20-jcp-consolidated/experiments/jcp-combined/sync_github.sh
mirror_pinned() {   # $1 = worktree name, $2 = LOCAL_BASE, $3 = REMOTE_BASE (= current mirror head)
  local n=$1 lb=$2 rb=$3 s=~/disk-plan/sync-$1.sh p
  mkdir -p ~/disk-plan
  test "$(git ls-remote origin "refs/heads/codeonly/exp/$n" | cut -f1)" = "$rb" || { echo "mirror of $n moved: re-plan"; exit 1; }
  sed -e "s/^LOCAL_BASE=.*/LOCAL_BASE=$lb/" -e "s/^REMOTE_BASE=.*/REMOTE_BASE=$rb/" -e "s/^PARENT=.*/PARENT=$rb/" "$TPL" > "$s"
  grep -q "^PARENT=$rb\$" "$s" || { echo "template changed: PARENT line not pinned"; exit 1; }
  (cd "worktrees/$n" && bash "$s")
  git fetch -q origin "refs/heads/codeonly/exp/$n"
  test "$(git rev-parse FETCH_HEAD^)" = "$rb" || { echo "unexpected mirror parent for $n"; exit 1; }
  diff <(git diff --no-renames --name-only "$rb" FETCH_HEAD | sort) <(git diff --no-renames --name-only "$lb" "exp/$n" | sort) \
    || { echo "mirror of $n does not contain exactly the new paths"; exit 1; }
  git diff --no-renames --name-only "$lb" "exp/$n" | while IFS= read -r p; do
    test "$(git rev-parse "exp/$n:$p")" = "$(git rev-parse "FETCH_HEAD:$p")" || { echo "content differs: $p"; exit 1; }
  done || { echo "mirror of $n failed the content check"; exit 1; }
  echo "mirror of $n verified"
}
mirror_pinned 2026-09-20-paper-b3d 37f92a5dbeabea86da824952a051e37ca096571c ccebc0c3b4979d72373195908080b53a417a7315   # 1 new commit(s)
mirror_pinned 2026-09-20-paper-ns3d a104a637c0a2c59693a5efe95e187066709ed22f 790c6e15b7d7b65640efa0cb19f74772bb273a3f   # 1 new commit(s)
mirror_pinned 2026-09-21-ns3d-grok 8852b7cd6365de7a8fdfd747bedee587f1cf9f29 9ccc132e39184f409ef842d80a8c69bd47439ab3   # 1 new commit(s)
mirror_pinned 2026-09-22-burgers-repanel 219feb6ecc3710b4c25514dac238737df93bfc5c 2d0f8c8b2b42f19a526cedf3c807ffd5373dc4dc   # 1 new commit(s)
mirror_pinned 2026-09-24-operators-all-pdes 766c3247ecbc55d8850b22e312b535a223916878 b1357c4f0f00e3971e4e3ac3018b30b9f17625b0   # 3 new commit(s)
```

### B. Archive a worktree's local-only data, verify, then remove the worktree

This follows `reports/2026-10-17-full-experiment-plan/campaign/offload.sh` (the campaign's verified offload) but archives every ignored and untracked file of one worktree. The file lists and manifests are written outside the worktree, so they neither enter the payload nor block `git worktree remove`. The record is committed on `main` and pushed before anything is removed. Immediately before removal the procedure re-reads every local byte and re-checks process use.

```bash
source ~/disk-plan/guards.sh
n=WORKTREE_NAME                                     # e.g. 2026-09-23-burgers-compare-hires
W=$REPO/worktrees/$n; R=$CL/$n/$(date -u +%Y%m%dT%H%M%SZ); T=~/disk-plan/$n
[ -d "$W" ] || { echo "no worktree $n"; exit 1; }
[ ! -e "$T/files.z" ] || { echo "$T exists from an earlier run: inspect it first"; exit 1; }
mkdir -p "$T"
cd /; not_in_use "$W"                # checks run from / so this shell's own cwd is not a hit
save_sparse_state "$n"
st=$(git -C "$W" status --porcelain --untracked-files=no); [ -z "$st" ] || { echo "tracked changes in $n"; exit 1; }
cd "$W"
LIST='{ git ls-files -o -z --exclude-standard && git ls-files -o -i -z --exclude-standard; } | sort -zu'
eval "$LIST" > "$T/files.z"
make_manifests "$T/files.z" "$T"
upload_and_verify "$T/files.z" "$T" "$R"
record_on_main "worktrees-$n" "$T" "$R"
cd /; not_in_use "$W"; cd "$W"
recheck_unchanged "$LIST" "$T/files.z" "$T" "$T/after"
cd "$REPO" && git worktree remove "worktrees/$n"   # --force only for jcp-wide-bank and operators-all-pdes (their untracked files were archived above)
```

To restore a removed worktree:
1. Run `git worktree add worktrees/<n> exp/<n>`.
2. Re-apply the sparse state saved under `~/disk-plan/sparse-before/` by `save_sparse_state`, using the undo rules in Procedure E (cone or non-cone, as recorded). Per-worktree configuration is lost on removal; the only setting these worktrees carry is sparse checkout.
3. Run `rsync -aH tufts-login:<R>/ worktrees/<n>/`, then `sha256sum -c ARCHIVE-FILES.sha256` inside the worktree.
4. Delete the two copied manifests from the worktree.

Symlinks are restored as the same link text. All six symlinks in the worktrees are relative and point inside their own worktree.

Notes:
- **Order.** The step frees space locally only at the last line, and the cluster copy must be verified before that line runs. Do the largest worktrees first: `burgers-compare-hires`, `heat3d-bank`, `heat-bank-knob`, `paper-b3d`, `nmrom-baselines` and `paper-ns3d` hold about half the local-only data.
- **JCP lanes (`2026-10-*`).** The 2026-10-20 lab-log entry says "Do not delete the lane worktrees or that archive." Removing a JCP lane worktree therefore needs the user's explicit override even after archiving. Re-sparsing them (SAFE) is compatible with that note, because it keeps every raw array.
- **Lanes whose data is "found on the cluster" by name and size** (`jcp-references`, `jcp-budget`, the archive part of `jcp-dims`, `jcp-qualify`, `jcp-robust` and others) still run the full procedure. The name+size match is provisional. rsync of an identical file is cheap, and the sha256 check is the real test.
- **Size.** Procedure B as written uploads every worktree's files separately: at most 898.0 GB logical (an upper bound; `rsync -H` keeps hardlinks within one worktree shared). Content identical across worktrees accounts for the gap to about 696 GB of distinct content. Paralab has 3.4 TB free. The September lanes are research history already reported. If the user decides some outputs are disposable (for example regenerable `data/` arrays), that is a separate decision for each lane.

### C. modcp raw tars on `main`

```bash
source ~/disk-plan/guards.sh
D=$REPO/artifacts/modcp-eq/2026-09-10/raw; R=$CL/main-modcp-eq-2026-09-10-raw/$(date -u +%Y%m%dT%H%M%SZ); T=~/disk-plan/modcp-raw
[ ! -e "$T/files.z" ] || { echo "$T exists: inspect it first"; exit 1; }; mkdir -p "$T"
cd /; not_in_use "$D"; cd "$D"
LIST='find . -mindepth 1 \( -type f -o -type l \) -printf "%P\0" | sort -z'
test "$(find . -mindepth 1 | wc -l)" = 8 && test "$(find . -maxdepth 1 -type f -name '*.tar' | wc -l)" = 8 \
  || { echo "raw/ is not exactly the 8 tars"; exit 1; }
eval "$LIST" > "$T/files.z"
make_manifests "$T/files.z" "$T"
grep -q '^f39290ebcf47a962d5ef8664882101cb2abe5c52e7d1ef5f6f7f40855ccade9f  burgers2d-evaluation01.tar$' "$T/ARCHIVE-FILES.sha256" \
  || { echo "burgers2d-evaluation01.tar does not match its LAB-LOG SHA256"; exit 1; }
upload_and_verify "$T/files.z" "$T" "$R"
record_on_main "main-modcp-eq-2026-09-10-raw" "$T" "$R"
cd /; not_in_use "$D"; cd "$D"
recheck_unchanged "$LIST" "$T/files.z" "$T" "$T/after"
cd "$REPO" && rm -r -- "$D"
```

`LAB-LOG.md` records three of the eight tar SHA256s (burgers2d-evaluation01, wave-reflective-evaluation01, wave-absorbing-evaluation01). Before running, compare all three by eye with `ARCHIVE-FILES.sha256`.

### D. Scratch directories in `/tmp` (coordinator scratchpad, Codex audit scratch)

The coordinator scratchpad holds about 2,600 small files besides the large arrays: prompts, frozen-stage JSONs, logs and smoke results. It also holds ten symlinks, archived as link text. One points inside the scratchpad, so it resolves when the scratchpad is restored to its original path. Nine point into `worktrees/2026-10-18-jcp-qualify`: six to committed code or run directories there, which stay recoverable from Git and from that worktree's own archive (Procedure B), and three to `runs/smkA/archive/output/qacc_smkA`, which is already dangling because smkA was offloaded to `burgers-archive/jcp-qualify/smkA` on 2026-10-18. The links themselves carry no data. The three Codex audit scratch directories hold audit scripts and recomputed outputs that are not in Git (for example `round4/audit.py` and `floor-recomputed.json`). Every file is archived, with no size threshold. First confirm in the Claude app that session `2f07cec8` (the Burgers-campaign coordinator) is finished.

```bash
source ~/disk-plan/guards.sh
for D in /tmp/claude-1002/-home-tahmid-Dev-pod-ae-nmrom-Tunable-NM-ROM-Claude/2f07cec8-5e56-4c01-8246-62353b26fb5e \
         /tmp/jcp-budget-audit-scratch /tmp/jcp-dims-audit-scratch /tmp/jcp-robust-audit-scratch; do
  k=tmp-$(basename "$D"); R=$CL/$k/$(date -u +%Y%m%dT%H%M%SZ); T=~/disk-plan/$k
  [ ! -e "$T/files.z" ] || { echo "$T exists: inspect it first"; exit 1; }; mkdir -p "$T"
  cd /; not_in_use "$D"; cd "$D"
  LIST='find . -mindepth 1 \( -type f -o -type l \) -printf "%P\0" | sort -z'
  test -z "$(find . -mindepth 1 ! -type f ! -type l ! -type d)" || { echo "special files in $D"; exit 1; }
  # symlinks are archived as link text; refuse any target (resolved from the link's own directory) outside D and jcp-qualify
  bad=$(find . -type l -print0 | while IFS= read -r -d '' f; do
          l=$(readlink -- "$f"); r=$(cd "$(dirname -- "$f")" && realpath -m -- "$l") || r="UNRESOLVABLE:$l"
          case "$r" in "$D"|"$D"/*|"$REPO/worktrees/2026-10-18-jcp-qualify/"*) ;; *) printf '%s -> %s\n' "$f" "$r" ;; esac
        done) || { echo "symlink scan failed in $D"; exit 1; }
  [ -z "$bad" ] || { echo "link targets outside the known set:"; printf '%s\n' "$bad"; exit 1; }
  eval "$LIST" > "$T/files.z"
  make_manifests "$T/files.z" "$T"
  upload_and_verify "$T/files.z" "$T" "$R"
  record_on_main "$k" "$T" "$R"
  cd /; not_in_use "$D"; cd "$D"
  recheck_unchanged "$LIST" "$T/files.z" "$T" "$T/after"
  cd / && rm -rf -- "$D"
done
```

### E. Re-sparse (SAFE)

Run one worktree at a time. The original patterns and sparse state are saved once (by `save_sparse_state` from the guard file) and never overwritten, so the change can be undone exactly. Procedure B saves the same state before any removal, so restoration step 2 works for every removed worktree:

```bash
set -euo pipefail; cd /home/tahmid/Dev/Tunable-NM-ROM-Claude
n=WORKTREE_NAME; B=~/disk-plan/sparse-before; mkdir -p "$B"
st=$(git -C "worktrees/$n" status --porcelain --untracked-files=no); [ -z "$st" ] || { echo "not clean: $n"; exit 1; }
cone=$(git -C "worktrees/$n" config --worktree --get core.sparseCheckoutCone) || { rc=$?; [ $rc = 1 ] || exit 1; cone=unset; }
[ "$cone" != true ] || { echo "cone mode: skip (cone mode deletes ignored-only directories)"; exit 1; }
source ~/disk-plan/guards.sh; save_sparse_state "$n"; set -euo pipefail; cd "$REPO"
# then the command from the SAFE table, e.g.
# git -C worktrees/$n sparse-checkout set --no-cone '/*' '!/*/' '/experiments/<lane>/'
```

To undo:
- If `$B/$n.state` says `sparseCheckout=true` and `sparseCheckoutCone=true`, run `xargs -d "\n" -a "$B/$n.txt" git -C "worktrees/$n" sparse-checkout set --cone`. A cone-mode list holds directories, not patterns.
- If it says `sparseCheckout=true` with any other cone value, run `xargs -d "\n" -a "$B/$n.txt" git -C "worktrees/$n" sparse-checkout set --no-cone`. The patterns are passed without shell globbing.
- Otherwise the worktree was not sparse before; run `git -C "worktrees/$n" sparse-checkout disable`.

Each command rewrites only that worktree's index and deletes checked-out copies of committed files outside the kept paths. The non-cone patterns `'/*' '!/*/' '/experiments/<lane>/'` keep the top-level files, the lane's own directory, and every `experiments/` directory that holds ignored data. In non-cone mode Git never removes ignored files.

## Method

- Sizes: `du -x` (disk usage). Tracked versus ignored split: `git ls-files -t` (checked-out tracked paths, sized from `git ls-tree -l`), `git ls-files -o --exclude-standard` (untracked) and `git ls-files -o -i --exclude-standard` (ignored), sized by `lstat`.
- "In Git": every ignored file over 1 MB whose size equals some blob size was hashed with `git hash-object --no-filters` and looked up in the object store (70.6 GB matched).
- "On the cluster": a `find -size +1M` listing of `/cluster/tufts/paralab/tawal01` (18,615 files), matched by basename and byte size. This is **provisional**; it is never used to call anything SAFE.
- Cross-worktree duplicates: BLAKE2 (`b2sum`) of every ignored file over 1 MB whose size occurs more than once.
- Mirrors: a blob-less, depth-1 fetch of `refs/heads/codeonly/*` and `refs/heads/exp/*` into a scratch repository; trees compared path by path with the local branch tip, applying the `sync_github.sh` rule (paths changed since `LOCAL_BASE`, files of 50 MB or less).
- Processes: `/proc/*/cwd` and `/proc/*/fd` for every process.
- The measurement scripts are in the session scratchpad (`make_tables.py`, `make_report.py`). By instruction, only this report and its Codex review are committed.

## Per-worktree status (generated)

| Worktree | Branch status | Size GB | Tracked checkout GB | Ignored GB | Local-only GB | Re-sparse (SAFE) frees GB | Remove worktree: class | Prerequisites before removal |
|---|---|---:|---:|---:|---:|---:|---|---|
| `2026-08-12-coord-decoder` | clean; not merged (branch kept); mirror stale | 0.5 | 0.0 | 0.5 | 0.5 | — | SAFE-AFTER-STEP | push 2026-10-20 commit(s) to GitHub first; archive 0.5 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-08-13-gn-tolerance-sweep` | clean; not merged (branch kept); mirror current | 0.1 | 0.0 | 0.1 | 0.1 | — | SAFE-AFTER-STEP | archive 0.1 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-08-13-heat2d-coord-decoder` | clean; not merged (branch kept); mirror current | 0.8 | 0.0 | 0.8 | 0.8 | — | SAFE-AFTER-STEP | archive 0.8 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-08-14-burgers2d-coord-rom` | clean; not merged (branch kept); mirror current | 0.9 | 0.0 | 0.9 | 0.9 | — | SAFE-AFTER-STEP | archive 0.9 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-08-14-wave2d-coord-rom` | clean; not merged (branch kept); mirror current | 0.8 | 0.0 | 0.8 | 0.8 | — | SAFE-AFTER-STEP | archive 0.8 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-08-16-poisson2d-rom-objective` | clean; merged into consolidated; mirror current | 0.1 | 0.0 | 0.1 | 0.1 | — | SAFE-AFTER-STEP | archive 0.1 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-08-19-poisson-hybrid-1024` | clean; contained in `exp/2026-08-20-poisson-hybrid-2048`; mirror current | 0.2 | 0.2 | 0.1 | 0.1 | — | SAFE-AFTER-STEP | archive 0.1 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-08-25-burgers-accuracy` | clean; merged into consolidated; mirror current | 3.6 | 0.6 | 3.0 | 2.5 | — | SAFE-AFTER-STEP | archive 2.5 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-08-25-sepdec-consolidated` | clean; merged into consolidated; mirror current | 2.1 | 0.9 | 1.2 | 1.2 | — | SAFE-AFTER-STEP | archive 1.2 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-08-26-codesign` | clean; not merged (branch kept); mirror stale | 1.9 | 1.0 | 1.0 | 1.0 | — | SAFE-AFTER-STEP | push 2026-10-20 commit(s) to GitHub first; archive 1.0 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-08-30-stokes-vector` | clean; merged into consolidated; mirror current | 1.6 | 1.1 | 0.5 | 0.5 | — | SAFE-AFTER-STEP | archive 0.5 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-14-head-ablation` | clean; contained in `exp/2026-09-16-b-head-train`; mirror current | 7.1 | 3.2 | 3.8 | 3.6 | 1.4 | SAFE-AFTER-STEP | archive 3.6 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-14-mesh-ladder` | clean; not merged (branch kept); mirror current | 3.3 | 2.3 | 1.0 | 0.9 | 1.4 | SAFE-AFTER-STEP | archive 0.9 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-14-no-audit` | clean; contained in `exp/2026-09-17-no-second`; mirror current | 10.8 | 6.0 | 4.8 | 1.8 | 1.4 | SAFE-AFTER-STEP | archive 1.8 GB of local-only ignored data to the cluster and verify sha256; sha256-verify 2.2 GB matched on the cluster by name+size only |
| `2026-09-14-no-burgers` | clean; not merged (branch kept); mirror current | 30.5 | 11.4 | 21.5 | 13.9 | 1.4 | SAFE-AFTER-STEP | archive 13.9 GB of local-only ignored data to the cluster and verify sha256; sha256-verify 7.1 GB matched on the cluster by name+size only |
| `2026-09-15-cheap-corrections` | clean; contained in `exp/2026-09-16-b-ladder-top`; mirror current | 8.3 | 6.2 | 2.1 | 2.0 | 3.2 | SAFE-AFTER-STEP | archive 2.0 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-15-head-refine` | clean; not merged (branch kept); mirror current | 11.3 | 5.9 | 5.4 | 5.3 | 3.2 | SAFE-AFTER-STEP | archive 5.3 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-15-prior-dial` | clean; not merged (branch kept); mirror current | 7.7 | 4.7 | 3.0 | 2.9 | 3.2 | SAFE-AFTER-STEP | archive 2.9 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-16-b-head-train` | clean; contained in `exp/2026-09-20-bank-floor`; mirror current | 7.7 | 4.6 | 3.1 | 3.0 | 3.2 | SAFE-AFTER-STEP | archive 3.0 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-16-b-ladder-top` | clean; contained in `exp/2026-09-16-q-diag`; mirror current | 10.3 | 7.5 | 2.8 | 2.5 | 6.2 | SAFE-AFTER-STEP | archive 2.5 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-16-p-bank-head` | clean; contained in `exp/2026-09-17-lshape`; mirror current | 10.0 | 5.6 | 4.4 | 4.2 | 3.2 | SAFE-AFTER-STEP | archive 4.2 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-16-q-ridge` | clean; contained in `exp/2026-09-17-b-eqtop`; mirror current | 13.6 | 9.4 | 4.2 | 3.8 | 7.5 | SAFE-AFTER-STEP | archive 3.8 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-16-q-trajdirs` | clean; not merged (branch kept); mirror current | 8.7 | 7.9 | 0.8 | 0.6 | 7.5 | SAFE-AFTER-STEP | archive 0.6 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-17-b-eqtop` | clean; not merged (branch kept); mirror current | 11.3 | 9.8 | 1.5 | 1.1 | 7.5 | SAFE-AFTER-STEP | archive 1.1 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-17-b-lowvisc` | clean; not merged (branch kept); mirror current | 14.0 | 10.7 | 3.2 | 3.1 | 9.4 | SAFE-AFTER-STEP | archive 3.1 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-17-b-panel` | clean; contained in `exp/2026-09-20-hires-burgers`; mirror current | 53.4 | 24.0 | 29.4 | 29.0 | 9.4 | SAFE-AFTER-STEP | archive 29.0 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-17-b-qxm` | clean; not merged (branch kept); mirror current | 14.8 | 11.1 | 3.7 | 3.3 | 9.4 | SAFE-AFTER-STEP | archive 3.3 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-17-b-seeds` | clean; not merged (branch kept); mirror current | 23.8 | 13.3 | 10.4 | 9.7 | 9.4 | SAFE-AFTER-STEP | archive 9.7 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-17-ns2d` | clean; merged into consolidated; mirror current | 5.2 | 4.2 | 1.0 | 0.7 | 1.4 | SAFE-AFTER-STEP | archive 0.7 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-17-p-linear` | clean; not merged (branch kept); mirror current | 20.8 | 11.2 | 9.6 | 9.5 | 5.6 | SAFE-AFTER-STEP | archive 9.5 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-17-w-ladder` | clean; not merged (branch kept); mirror current | 17.5 | 2.7 | 14.9 | 14.7 | 1.4 | SAFE-AFTER-STEP | archive 14.7 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-20-bank-floor` | clean; not merged (branch kept); mirror current | 10.2 | 4.6 | 5.6 | 5.4 | 4.6 | SAFE-AFTER-STEP | archive 5.4 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-20-hires-burgers` | clean; contained in `exp/2026-09-21-burgers-eqcert`; mirror current | 39.7 | 24.0 | 15.6 | 15.3 | 24.0 | SAFE-AFTER-STEP | archive 15.3 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-20-hires-heat` | clean; contained in `exp/2026-09-21-heat3d-bank`; mirror current | 61.6 | 42.6 | 19.1 | 19.0 | 42.5 | SAFE-AFTER-STEP | archive 19.0 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-20-hires-poisson` | clean; contained in `exp/2026-09-23-poisson-bank-knob`; mirror current | 33.3 | 31.7 | 1.5 | 1.5 | 31.6 | SAFE-AFTER-STEP | archive 1.5 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-20-nmrom-baselines` | clean; not merged (branch kept); mirror current | 44.9 | 6.0 | 39.0 | 36.8 | 6.0 | SAFE-AFTER-STEP | archive 36.8 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-20-paper-b3d` | clean; not merged (branch kept); mirror stale | 99.2 | 59.7 | 39.4 | 35.1 | — | SAFE-AFTER-STEP | push 2026-10-20 commit(s) to GitHub first; archive 35.1 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-20-paper-h3d` | clean; not merged (branch kept); mirror current | 70.5 | 68.0 | 2.5 | 1.7 | — | SAFE-AFTER-STEP | archive 1.7 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-20-paper-ns3d` | clean; not merged (branch kept); mirror stale | 94.4 | 76.1 | 35.4 | 35.4 | — | SAFE-AFTER-STEP | push 2026-10-20 commit(s) to GitHub first; archive 35.4 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-20-paper-p3d` | clean; contained in `exp/2026-09-20-hires-poisson`; mirror current | 21.8 | 20.5 | 1.2 | 1.2 | — | SAFE-AFTER-STEP | archive 1.2 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-21-burgers-eqcert` | clean; not merged (branch kept); mirror current | 37.7 | 24.0 | 13.7 | 13.4 | 24.0 | SAFE-AFTER-STEP | archive 13.4 GB of local-only ignored data to the cluster and verify sha256; sha256-verify 0.0 GB matched on the cluster by name+size only |
| `2026-09-21-burgers-heldout` | clean; not merged (branch kept); mirror current | 53.3 | 24.0 | 29.3 | 29.1 | 24.0 | SAFE-AFTER-STEP | archive 29.1 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-21-heat3d-bank` | clean; not merged (branch kept); mirror current | 116.2 | 43.0 | 73.3 | 72.9 | 42.6 | SAFE-AFTER-STEP | archive 72.9 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-21-ns3d-grok` | clean; not merged (branch kept); mirror stale | 87.4 | 80.3 | 7.2 | 7.2 | 80.2 | SAFE-AFTER-STEP | push 2026-10-20 commit(s) to GitHub first; archive 7.2 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-22-burgers-repanel` | clean; not merged (branch kept); mirror stale | 25.3 | 24.0 | 1.3 | 0.7 | 24.0 | SAFE-AFTER-STEP | push 2026-10-20 commit(s) to GitHub first; archive 0.7 GB of local-only ignored data to the cluster and verify sha256; sha256-verify 0.5 GB matched on the cluster by name+size only |
| `2026-09-22-ops-deeponet-b2d` | clean; contained in `exp/2026-09-22-ops-tune-deeponet`; mirror current | 18.7 | 17.8 | 0.9 | 0.3 | 17.0 | SAFE-AFTER-STEP | archive 0.3 GB of local-only ignored data to the cluster and verify sha256; sha256-verify 0.5 GB matched on the cluster by name+size only |
| `2026-09-22-ops-timing-panel` | clean; not merged (branch kept); mirror current | 20.6 | 17.0 | 3.6 | 2.6 | 17.0 | SAFE-AFTER-STEP | archive 2.6 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-22-quadratic-manifold` | clean; contained in `exp/2026-09-23-burgers-compare-hires`; mirror current | 25.3 | 24.0 | 1.3 | 1.2 | 24.0 | SAFE-AFTER-STEP | archive 1.2 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-23-burgers-bank-knob` | clean; not merged (branch kept); mirror current | 46.6 | 24.0 | 22.6 | 21.9 | 24.0 | SAFE-AFTER-STEP | archive 21.9 GB of local-only ignored data to the cluster and verify sha256; sha256-verify 0.0 GB matched on the cluster by name+size only |
| `2026-09-23-burgers-compare-hires` | clean; contained in `exp/2026-09-25-t2-burgers-test`; mirror current | 137.7 | 24.0 | 113.7 | 112.4 | 24.0 | SAFE-AFTER-STEP | archive 112.4 GB of local-only ignored data to the cluster and verify sha256; sha256-verify 1.0 GB matched on the cluster by name+size only |
| `2026-09-23-burgers2d-speed` | clean; contained in `exp/2026-09-24-burgers-eq-tol-knobs`; mirror current | 17.3 | 1.4 | 15.9 | 15.6 | 1.4 | SAFE-AFTER-STEP | archive 15.6 GB of local-only ignored data to the cluster and verify sha256; sha256-verify 0.0 GB matched on the cluster by name+size only |
| `2026-09-23-burgers3d-retry` | clean; not merged (branch kept); mirror current | 11.0 | 1.5 | 9.5 | 8.9 | 1.5 | SAFE-AFTER-STEP | archive 8.9 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-23-burgers3d-span` | clean; contained in `exp/2026-09-23-burgers3d-retry`; mirror current | 6.7 | 1.5 | 5.2 | 4.4 | 1.4 | SAFE-AFTER-STEP | archive 4.4 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-23-heat-bank-knob` | clean; not merged (branch kept); mirror current | 92.5 | 42.6 | 49.9 | 49.8 | 42.6 | SAFE-AFTER-STEP | archive 49.8 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-23-heat-compare-hires` | clean; not merged (branch kept); mirror current | 53.0 | 42.6 | 10.4 | 10.4 | 42.6 | SAFE-AFTER-STEP | archive 10.4 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-23-ns3d-operators` | clean; merged into consolidated; mirror current | 9.0 | 0.3 | 8.7 | 8.7 | — | SAFE-AFTER-STEP | archive 8.7 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-23-poisson-bank-knob` | clean; contained in `exp/2026-09-23-spectral-fom`; mirror current | 34.1 | 31.7 | 2.3 | 2.2 | 31.7 | SAFE-AFTER-STEP | archive 2.2 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-23-spectral-fom` | clean; not merged (branch kept); mirror current | 52.0 | 31.8 | 20.2 | 19.6 | 31.7 | SAFE-AFTER-STEP | archive 19.6 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-24-burgers-eq-tol-knobs` | clean; not merged (branch kept); mirror current | 3.2 | 0.9 | 2.4 | 2.1 | — | SAFE-AFTER-STEP | archive 2.1 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-24-operators-all-pdes` | clean; not merged (branch kept); mirror stale | 0.6 | 0.2 | 0.3 | 0.1 | — | SAFE-AFTER-STEP | push 2026-10-20 commit(s) to GitHub first; archive 0.1 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-25-burgers2d-test` | clean; not merged (branch kept); mirror current | 23.8 | 0.9 | 22.9 | 22.6 | — | SAFE-AFTER-STEP | archive 22.6 GB of local-only ignored data to the cluster and verify sha256; sha256-verify 0.0 GB matched on the cluster by name+size only |
| `2026-09-25-ns3d-test` | clean; not merged (branch kept); mirror current | 2.9 | 0.1 | 2.9 | 2.9 | — | SAFE-AFTER-STEP | archive 2.9 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-25-poisson2d-test` | clean; not merged (branch kept); mirror current | 0.5 | 0.0 | 0.4 | 0.4 | — | SAFE-AFTER-STEP | archive 0.4 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-25-t2-burgers-test` | clean; not merged (branch kept); mirror current | 6.5 | 0.1 | 6.4 | 6.2 | — | SAFE-AFTER-STEP | archive 6.2 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-25-t2-ns3d-test` | clean; merged into consolidated; mirror current | 4.6 | 0.3 | 4.4 | 4.4 | — | SAFE-AFTER-STEP | archive 4.4 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-09-25-t3-burgers-test` | clean; not merged (branch kept); mirror current | 19.0 | 0.1 | 18.9 | 18.8 | — | SAFE-AFTER-STEP | archive 18.8 GB of local-only ignored data to the cluster and verify sha256 |
| `2026-10-01-ns3d-coordnet-bank` | clean; merged into consolidated; mirror current | 80.7 | 80.4 | 0.4 | 0.4 | 80.3 | SAFE-AFTER-STEP | archive 0.4 GB of local-only ignored data to the cluster and verify sha256; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override |
| `2026-10-01-quadrature-burgers3d` | clean; merged into consolidated; mirror current | 109.2 | 83.0 | 26.3 | 25.0 | 80.4 | SAFE-AFTER-STEP | archive 25.0 GB of local-only ignored data to the cluster and verify sha256; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override |
| `2026-10-01-quadrature-study` | clean; merged into consolidated; mirror current | 85.3 | 80.7 | 4.6 | 4.2 | 80.4 | SAFE-AFTER-STEP | archive 4.2 GB of local-only ignored data to the cluster and verify sha256; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override |
| `2026-10-06-jcp-fom-pareto` | clean; merged into consolidated; mirror current | 4.4 | 4.4 | 0.0 | 0.0 | 4.3 | SAFE-AFTER-STEP | lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override |
| `2026-10-06-jcp-hr-baselines` | clean; merged into consolidated; mirror current | 13.8 | 9.8 | 3.9 | 0.5 | 4.3 | SAFE-AFTER-STEP | archive 0.5 GB of local-only ignored data to the cluster and verify sha256; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override |
| `2026-10-06-jcp-mechanism` | clean; merged into consolidated; mirror current | 82.3 | 80.9 | 1.5 | 1.5 | 80.4 | SAFE-AFTER-STEP | archive 1.5 GB of local-only ignored data to the cluster and verify sha256; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override |
| `2026-10-06-jcp-references` | clean; merged into consolidated; mirror current | 91.6 | 80.5 | 22.3 | 0.0 | 80.4 | SAFE-AFTER-STEP | sha256-verify 22.3 GB matched on the cluster by name+size only; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override |
| `2026-10-08-jcp-smooth-bank` | clean; merged into consolidated; mirror current | 9.8 | 4.8 | 5.1 | 4.3 | 4.3 | SAFE-AFTER-STEP | archive 4.3 GB of local-only ignored data to the cluster and verify sha256; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override |
| `2026-10-08-jcp-time2` | clean; merged into consolidated; mirror current | 10.2 | 5.8 | 4.4 | 4.2 | 4.3 | SAFE-AFTER-STEP | archive 4.2 GB of local-only ignored data to the cluster and verify sha256; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override |
| `2026-10-08-jcp-wide-bank` | clean; merged into consolidated; mirror current | 18.2 | 6.4 | 11.8 | 7.9 | 4.3 | SAFE-AFTER-STEP | archive 7.9 GB of local-only ignored data to the cluster and verify sha256; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override |
| `2026-10-13-jcp-combined` | clean; merged into consolidated; mirror current | 27.1 | 14.1 | 12.9 | 2.9 | 4.3 | SAFE-AFTER-STEP | archive 2.9 GB of local-only ignored data to the cluster and verify sha256; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override |
| `2026-10-14-jcp-speed3d` | clean; merged into consolidated; mirror current | 16.1 | 14.2 | 1.9 | 1.9 | 4.3 | SAFE-AFTER-STEP | archive 1.9 GB of local-only ignored data to the cluster and verify sha256; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override |
| `2026-10-17-jcp-p1-scaling` | clean; merged into consolidated; mirror current | 13.1 | 4.6 | 8.5 | 5.8 | 4.3 | SAFE-AFTER-STEP | archive 5.8 GB of local-only ignored data to the cluster and verify sha256; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override |
| `2026-10-18-jcp-budget` | clean; merged into consolidated; mirror current | 8.9 | 7.7 | 5.2 | 0.0 | 4.3 | SAFE-AFTER-STEP | sha256-verify 2.3 GB matched on the cluster by name+size only; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override |
| `2026-10-18-jcp-dims` | clean; merged into consolidated; mirror current | 20.9 | 4.6 | 16.2 | 2.8 | 4.3 | SAFE-AFTER-STEP | archive 2.8 GB of local-only ignored data to the cluster and verify sha256; sha256-verify 7.2 GB matched on the cluster by name+size only; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override |
| `2026-10-18-jcp-fom2d` | clean; merged into consolidated; mirror current | 5.5 | 4.5 | 0.9 | 0.1 | 4.3 | SAFE-AFTER-STEP | archive 0.1 GB of local-only ignored data to the cluster and verify sha256; sha256-verify 0.1 GB matched on the cluster by name+size only; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override |
| `2026-10-18-jcp-heads` | clean; merged into consolidated; mirror current | 6.0 | 4.6 | 1.4 | 0.1 | 4.3 | SAFE-AFTER-STEP | archive 0.1 GB of local-only ignored data to the cluster and verify sha256; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override |
| `2026-10-18-jcp-qualify` | clean; merged into consolidated; mirror current | 9.2 | 4.6 | 4.7 | 0.8 | 4.3 | SAFE-AFTER-STEP | archive 0.8 GB of local-only ignored data to the cluster and verify sha256; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override |
| `2026-10-19-jcp-regimes` | clean; merged into consolidated; mirror current | 7.2 | 5.7 | 1.5 | 1.4 | 4.3 | SAFE-AFTER-STEP | archive 1.4 GB of local-only ignored data to the cluster and verify sha256; sha256-verify 0.1 GB matched on the cluster by name+size only; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override |
| `2026-10-19-jcp-robust` | clean; merged into consolidated; mirror current | 27.5 | 9.5 | 18.0 | 0.5 | 4.3 | SAFE-AFTER-STEP | archive 0.5 GB of local-only ignored data to the cluster and verify sha256; sha256-verify 0.3 GB matched on the cluster by name+size only; lab log 2026-10-20 says do not delete JCP lane worktrees: needs explicit user override |


## Glossary

- **Worktree**: a second checkout of the same repository in its own directory (`worktrees/<date-slug>`) on its own branch, sharing one `.git`. Removing it deletes the directory, not the branch or its commits.
- **Tracked checkout**: files in the worktree that are committed on its branch. They are copies of data stored in `.git`, so deleting them loses nothing.
- **Ignored / untracked files**: files in the worktree that Git does not store (matched by `.gitignore`, or simply never added): run arrays (`.npz`, `.npy`), checkpoints (`.pkl`, `.pt`), compressed outputs (`.gz`), logs, and staged copies of job code.
- **Local-only**: ignored or untracked data whose bytes are not in any Git object and that was not found on the cluster. Removing it without archiving destroys it.
- **Sparse checkout / re-sparse**: a Git setting that checks out only some paths. "Re-sparse" means narrowing it so committed files outside the lane's own directory are removed from the working directory (they remain in `.git`). **Cone mode** is a restricted pattern style that also deletes ignored-only directories; **non-cone** (`--no-cone`) does not.
- **Mirror / `codeonly`**: the GitHub backup branch `origin/codeonly/exp/<slug>`, a filtered copy of a lane's code and small files; files over 50 MB are skipped and recorded by SHA256 instead.
- **Merged into consolidated**: the worktree's branch tip is an ancestor of `exp/2026-10-20-jcp-consolidated`. **Contained in `<branch>`**: the tip is an ancestor of another kept branch.
- **SAFE / SAFE-AFTER-STEP / KEEP**: SAFE frees space with no possible data loss; SAFE-AFTER-STEP becomes safe only after the stated prerequisite (archive and verify, or push to GitHub); KEEP must not be removed.
- **Archive (cluster)**: `/cluster/tufts/paralab/tawal01/...`, the group share on the Tufts cluster; `burgers-archive/` holds the JCP campaign raw arrays.
- **sha256-verify**: compute a SHA-256 checksum for every file locally and check the remote copy against that list; any mismatch stops the procedure.
- **Name+size match**: a cheap test that a file of the same name and byte size exists on the cluster. It suggests a copy exists but proves nothing about content.
