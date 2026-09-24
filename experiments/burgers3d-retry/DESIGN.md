# DESIGN — burgers3d-retry: a second attempt at a Burgers 3D row

Branch `exp/2026-09-23-burgers3d-retry`, forked from `exp/2026-09-23-burgers3d-span` @ `2c9e7b39`, local commits only,
never pushed. Cluster namespace `/cluster/tufts/paralab/tawal01/b3dretry_20260923/`, one directory per job. Brief:
`reports/2026-09-23-burgers3d-retry-handoff.md` (main). This file is written in two parts: §§1–3 (diagnosis, probes,
cohorts) before any ROM result of this lane; §§4–9 (the pre-registered protocol) are committed before any validation
panel is run. Amendments are appended, dated, and never rewrite earlier text.

## 1. Starting point (lane 1, `burgers3d-span`, negative on its sealed cohort)

Held-out (seed 923401, 32 cases, opened; **never used here for any choice**): accurate span $R'=512$
1.32 / 2.39 / 3.48 % at 0.02 / 0.03 / 0.21× Newton–BiCGStab (32³/64³/128³); fast 8.29 / 8.31 / 7.89 % (> 5 % bar).
Validation (16 cases) under-estimated the held-out worst error by 2.1–2.7×.

What limits it (from lane 1's records only):

1. **Accuracy is the linear floor of the bank, not the solver.** Evolved error ≈ the projection floor of the ordered
   bank at the same $R'$ (validation 128³: span 512 1.50 % vs floor 2.2 %; span 192 3.7 % vs floor 5.0 %). The R = 512
   bank equals POD of its own training data on bank-validation fields (POD 512: 0.86 / 1.61 / 2.36 % worst; bank:
   0.93 / 1.63 / 2.20 %), i.e. the coordinate network is not the bottleneck — the *linear n-width* of this family and
   the amount of training data are.
2. **The nonlinear head generalises badly** (training error ≈ 0.05 %, bank-validation best-found 21–25 %): 384
   trajectories of a ~17-parameter family (1–3 blobs × centre/width/amplitude, ν, time) is sparse for an auto-decoder.
3. **Cost.** The tensor rule costs $O(M R'^2)$ with $M = 4R'$; at $R'=512$ the table is 4.3 GB and the fixed-sweep path
   read it twice per step (predictor + sweep) → ≈ 300 ms on an A100. The full-order solver is very cheap at 32³/64³
   (7.7 / 11.6 ms at its paper-rule setting) and grows ≈ 5× per mesh doubling (61 ms at 128³).
4. **Validation too small.** 16 cases cannot estimate a worst case over 32.

## 2. Probes (diagnostic only; probe seeds 923601 / 923651, disjoint from every cohort of both lanes)

Run before any design choice below; results are recorded in §2.1 when collected.

- `pod65`, `pod129`, `pod257h` (jobs 4246732, 4246733, 4246816): POD validation floors for nested training sizes
  (96 … 1536 trajectories; 96 … 384 at 257 nodes) on each mesh's point group, and the frozen lane-1 bank's floor on the
  same fields. Question: does more data lower the floor, and how does the floor grow with the mesh?
- `fineh` (4246813): full-grid floor of the frozen bank at 129 and 257 nodes on 8 probe cases, and Newton–BiCGStab GPU
  times of six settings at 129 and 257 nodes (H200). Question: is 256³ a regime where a reduced model can win?
- `fscchkh` (4246812): equivalence and speed of the cached-predictor solver (`common2.make_query_fsc`) and its float32
  table variant against lane 1's `make_query_fs`, at 33/65/129 nodes on 4 probe cases.
- Training candidates `trw512h` / `trw1024h` (4246814 / 4246815): the lane-1 recipe on 4× the data (1536 training
  trajectories, seed 923701; bank-validation 923751 × 96), bank MLP width 512 / 1024, 12 000 bank steps; head variants
  K ∈ {16, 32, 64} and K = 32 with an L2 code penalty 1e-3 / 1e-2; the lane-1 bank's floors on the same bank-validation
  fields are recorded in the same job (the model-choice rule of §4 compares them).

Local runs on the GB10 are not used for any number: the box returned NaN from f64 GEMM/QR non-deterministically on
2026-09-23 (the same code gives finite tables on the cluster); every check runs on the cluster.

### 2.1 Probe results

(filled in from the probe JSONs before §4 is written)

## 3. Cohorts of this lane (fresh seeds, never used by lane 1 or by the probes)

| cohort | seed | size | use |
|---|---|---|---|
| training | 923701 | 1536 | bank, ordering, heads (candidates M1/M2) |
| bank-validation | 923751 | 96 | model choice (§4), head choice, floors |
| ROM validation | 923801 | 64 | the settings rule |
| certification draws 1–5 | 923811–923815 | 8 each | quadrature/tensor certificates |
| confirmation draw | 923816 | 16 | certificates |
| **sealed held-out** | **923901** | **32** | opened once, frozen settings only |

The validation cohort is 4× lane 1's (64 vs 16) so that its worst case is a less optimistic estimate of a 32-case
worst case.
