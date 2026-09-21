# SPEED-LOG — nmrom-baselines

The protocol's speed mandate (profile → hypothesis → fix → re-measure) applies to the project ROM's timing arm. This lane
does not modify the project ROM: it times the unmodified b-panel production query (`vendor/topfix.py` @ 25434a27) beside the
baselines, in the same allocation. Optimisations of our ROM belong to the speed lanes; entries here are limited to choices
that affect how fast the *baselines* run, so that they are not strawmen.

| date | hypothesis | change | measured effect | kept |
|---|---|---|---|---|
| 2026-09-20 | A dense `n x M2` masked matmul would make the Kim decoder cost O(n^2) on GPU | store the masked layer as per-row (value, index) tables, `n x (3b+2db)`; gather + einsum | structural (dense is infeasible beyond 128²: 16129 x 161380 weights); table form verified equal to the paper's dense mask (`test_mask.py`) | yes |
| 2026-09-20 | Forming the decoder Jacobian with `vmap` over K directions needs `n*P*K` floats (21 GB at 512², K=32) | `lspg.make_rollout(jac_batch=...)`: `lax.map` over latent directions when the gather exceeds 6 GB | pending first cluster job | pending |
| 2026-09-20 | gate01 (job 4051709) finished no training epoch in 22 min at 98 % GPU: the gather's transpose (XLA scatter-add, 258 M updates per batch) is pathological | `kimae.masked_out`: every stored window is a run of length-`db` blocks of the hidden vector, so the masked product is a sum of 32 sliced products; slices transpose to pads. Value and gradient equality with the gather form tested to 1e-12 (`test_mask.py`) | > 20 min/epoch (never finished) → 0.05 s per 240-batch on the loaded GB10 (≈ 1.1 s/epoch at the gate's size) | yes |
| 2026-09-21 | (coordinator check) our "ours" arm is ~6x (q=0) / ~25x (q=256) slower than the campaign's optimised query at 256² — bug or path? | none (diagnosis). `family.py` calls `topfix.make_query(..., 'dense', 'base')`: dense residual, M=4(K+q) sine tests (64 / 1088), gtol=ic_gtol=1e-6, gj (q=0) / lu (q=256), no b-speed kernels, compile excluded by a warm call | it is the dense reference path, reproduced: b-panel job 3780638's same-path rows at 256² (`q0_M64_dense_g1em06` 284.8 ms, `q256_M1088_dense_g1em06` 4164.6 ms) vs this lane's 262.2 / 3746.3 ms (fam256, 4095408), within ~10 %. The ~40 ms figure is b-panel's certified-EQ path. Cross-job numbers used here only to diagnose, never in the report table | report rows relabelled "dense reference path"; speed comparisons tagged dense-vs-dense or HR-vs-dense |
