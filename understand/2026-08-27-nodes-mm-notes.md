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

## 2026-08-27 — N=256 results (both COMPLETED); N=1024 OOM + fix + resubmit

- **N=256, m=M=64 (2956403): the pilot result TRANSFERS.** Base (NNLS grid
  nodes) rollout 7.184e-2 → learned nodes 6.136e-2 = **−14.6%** (pilot at
  N=64 was −17.3%). Quadrature binding as expected: held (b) 1.021 → 0.412.
  Gate L 3.1e-16; recon identical both arms (2.964e-2, decoder frozen).
- **N=256, m=4M=256 (2956404): tie, as at N=64.** Base 4.792e-2 → 4.833e-2
  (+0.9%); held (b) 4.1e-2 → 1.3e-2 (mismatch improves, rollout doesn't —
  quadrature not binding). Job wall-times: 4.5 and 6 min on an A100.
- **N=1024 jobs 2956408/09 FAILED (retracted, nothing measured): GPU OOM**,
  15.94 GiB allocation in `dens_of` → `t_jac_of`: the vmapped
  `jacfwd` of the full-grid teacher materializes an (S=128, K=16,
  n_i2≈1.044M) f64 tangent block. Fine at N≤256, fatal at N=1024 even on an
  H200. Failed logs kept at
  `runs/cdmm/failed-oom-logs/` on the branch.
- **Fix:** `TEACHER_CHUNK` env in `sep_codesign.py` (commit ce73d52) —
  `adv_full_of`/`t_jac_of` switch from `vmap` to `lax.map(batch_size=chunk)`;
  default 0 keeps the pilot's unchunked path. Arm n differentiates only
  w.r.t. node positions, which the teacher terms don't depend on, so
  chunking changes memory only. Validated at N=64 (TEACHER_CHUNK=8,
  20 steps): gate R and the full base certification reproduce the pilot
  exactly (rollout 7.113e-2, held b 0.805, recon 2.733e-2).
- **Resubmitted with TEACHER_CHUNK=8: jobs 2956820 (n1024m64), 2956822
  (n1024m256).** N=256 results + logs pulled, checksums verified, finished
  cluster dirs deleted.

## 2026-08-27 — N=1024 done; experiment closed

- **2956820 (m=M): transfers, smaller — base 1.411e-1 → 1.309e-1 (−7.2%).**
- **2956822 (m=4M): tie — 1.172e-1 → 1.170e-1 (−0.2%).**
- **Matched-accuracy verdict (the decisive one): learned m=M nodes never
  reach the NNLS m=4M baseline** — +13.4% / +28.1% / +11.7% worse at
  N=64/256/1024. The 4×-cheaper-solve hypothesis is refuted; no speed story.
- Written up: `reports/2026-08-27-nodes-at-mM-transfer.md` (generated
  tables T-N1/T-N2, glossary). All results committed on the branch;
  namespace `codesign-mm/` deleted; queue empty.
