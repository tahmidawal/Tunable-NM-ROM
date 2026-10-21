# operators-all-pdes — neural operators for every Table 1 cell (shared protocol)

Requested by the user on 2026-09-24: bring FNO, U-Net, Transolver and DeepONet data in for every
problem and mesh of paper Table 1 that has none, "no need to tune, just good enough results, as fast
as possible". Output feeds the time/error-vs-mesh figure and possibly Table 2.

Worktree `worktrees/2026-09-24-operators-all-pdes`, branch `exp/2026-09-24-operators-all-pdes`
(local only, not pushed — same as every lane since 09-17), forked from `exp/2026-09-23-ns3d-operators`
@ 766c3247 as a **sparse checkout** (the full branch tracks 129 GB; never run `git sparse-checkout
disable` or a full checkout — it fills the local disk).

## Groups (one agent each, own subdirectory, own cluster namespace, own GPU cap)

| group | subdirectory | cells (mesh) | GPU cap |
|---|---|---|---|
| A `p2d` | `experiments/ops-all/p2d/` | Poisson 2D 256², 1024², 2048², 4096²; Poisson L-shape 2D 256², 512², 1024², 2048² | 5 |
| B `l3d` | `experiments/ops-all/l3d/` | Poisson 3D 32³, 64³, 128³, 256³; Heat 3D 32³, 64³, 128³, 256³ | 5 |
| C `b3d` | `experiments/ops-all/b3d/` | Burgers 3D 64³, 128³, 256³ | 3 |
| D `g2d` | `experiments/ops-all/g2d/` | Burgers 2D 512², 4096²; Heat 2D 4096² | 3 |

Total ≤ 16 concurrent GPUs (user cap). Cluster namespace:
`/cluster/tufts/paralab/tawal01/opsall_20260924/<group>/<job>/`, one directory per job, never reused.

## Contract per cell (same as paper Table 2)

* Four families: FNO, U-Net, Transolver, DeepONet. **No tuning**: one size per family, taken from
  the nearest existing lane's validation-selected configuration (record which). Same optimiser
  (AdamW), seed per network, **3000 s wall budget** per network, checkpoint chosen on a validation
  split of the training data — never on the evaluation cohort.
* Training data: the same generator and training seed the NM-ROM of that Table 1 row used,
  regenerated on the cluster at the target mesh.
* Evaluation: the **same cases as the Table 1 row** (same seed/cohort; development or held-out as
  Table 1 says). No choice of any kind is made on them.
* **One panel job per cell** times, on one GPU in one allocation: the four operators, the Table 1
  NM-ROM accurate and fast settings (frozen, exactly as in the source lane), and the Table 1
  full-order setting (plus the source lane's full-order grid so the "fastest FOM at least as accurate
  as NM-ROM accurate" rule can be applied). Error = the Table 1 row's metric (worst same-grid
  relative L2, over evolved or all times as that row uses). Every ratio is same-job.
* Largest meshes (4096², 256³): attempt once on an H200 (`--mem` ≥ 240G) with the smallest trained
  sizes; if a family cannot train or run, record the error (OOM etc.) as the cell's result. Never
  fill a cell with a number from another mesh.

## Compute rules (from both CLAUDE.md files — read them)

* `gpu` partition only, never `preempt`. GPU preflight (`jax_backend=gpu`, exit 42 otherwise) in
  every job; grep it in the log. Venv `/cluster/tufts/paralab/tawal01/ae-research/venv`.
  `JAX_DEFAULT_MATMUL_PRECISION=highest`; x64 where the source lane used it.
* All job output under the paralab namespace, never `~` on the cluster. Check `squeue` before and
  after every submit (duplicate-submit race). Data regenerated on the cluster from seeds.
* **Local disk has ~27 GB free.** Pull back only `summary.json`, audit JSON, logs and small
  checksums — never fields, checkpoints or datasets. Delete finished job dirs' large files on the
  cluster after the summary is collected (group share ~92 % full).
* Do not `git commit` (the coordinator commits per group), do not append to `LAB-LOG.md`, do not
  touch other groups' subdirectories or namespaces, do not edit `paper_latex/`.
* Code may be read from any other worktree and copied into your subdirectory; record sources and
  commits in `experiments/ops-all/<group>/COPIED-FROM.json`.

## Deliverable per group

`experiments/ops-all/<group>/results.json`: for every cell — job id, GPU, cohort, FOM setting and
ms/err, NM-ROM accurate and fast ms/err, and per operator family: arm, parameters, epochs, stop
reason, worst %, median %, median GPU ms, or the failure reason. Plus a short `REPORT.md` with the
same table and every caveat. Report partial results as soon as a cell is done.
