# Running notes — nodes-only at m=M on the real meshes (exp/2026-08-27-nodes-mm)

Cluster follow-up to the co-design mini-pilot's F3 result (report:
`reports/2026-08-26-codesign-minipilot.md`): at N=64, frozen-decoder node
learning at the m=M budget bought −17.3% rollout error for free, and tied at
m=4M. This experiment asks whether that transfers to N=256 and N=1024 — the
meshes the published numbers live on — with the same driver, hyperparameters,
and seed as the pilot (arm n, `sep_codesign.py` unchanged, commit `da96a58`
lineage). If it holds, the sampled solve shrinks 4× (m=64 vs 256) on top of
the exlin line's speedup.

## 2026-08-27 — setup and submission

- Worktree `worktrees/2026-08-27-nodes-mm`, branch `exp/2026-08-27-nodes-mm`,
  from `exp/2026-08-26-codesign` (user-approved). Cluster namespace
  `/cluster/tufts/paralab/tawal01/codesign-mm/`, one job dir per arm.
- **Checkpoints (both git-tracked):** N=256 →
  `runs/n256_j2/out/sep_burgers_N256_K16_R64.pkl` (canonical default data
  draw, max_snaps 8192, warm recon mean 2.41e-2); N=1024 →
  `sep_burgers_N1024_K16_R64.pkl` from the sepdec_n1024 j2 branch, copied
  into `runs/inherited/n1024_j2/` and committed. Its data draw was 96+8
  trajectories / max_snaps 2048, so the N=1024 jobs set `N_TRAIN=96 N_VAL=8`
  (gate R enforces the reproduction; both checkpoints' recorded recon means
  are well under the 5e-2 gate).
- **Four jobs submitted** (arm n: TRAIN_H=0 TRAIN_NODES=1 SAMP_REL=1
  JAC_REL=1, STEPS=2000, LR_NODES=3e-3, REFIT_EVERY=500,
  REFIT_JAC_STATES=16, SEED0=0 — bit-identical driver to the pilot):
  - 2956403 `n256m64` — N=256, m=M=64, A100, **primary**
  - 2956404 `n256m256` — N=256, m=4M=256, A100, expect-a-tie control
  - 2956408 `n1024m64` — N=1024, m=M=64, H200 240G, **primary**
  - 2956409 `n1024m256` — N=1024, m=4M=256, H200 240G, control
- Staged with the proven `stage_exlin.sh` (deps byte-identical to the
  verified N=64 stage; Helmholtz-preconditioner patch on the truth
  generator — required at N=1024). Staged-layout import of `sep_codesign.py`
  smoke-checked locally before submission (the pilot only ever ran the
  worktree layout).
- What "transfer" means here: in each job the same instrument certifies the
  NNLS-grid-node baseline AND the learned-node variant (rollout on 4 fresh
  test trajectories, held-out rungs, gate L), so each JSON carries its own
  control; the m=4M jobs check that the win stays confined to the binding
  budget.
