# SPEED-LOG — nmrom-baselines

The protocol's speed mandate (profile → hypothesis → fix → re-measure) applies to the project ROM's timing arm. This lane
does not modify the project ROM: it times the unmodified b-panel production query (`vendor/topfix.py` @ 25434a27) beside the
baselines, in the same allocation. Optimisations of our ROM belong to the speed lanes; entries here are limited to choices
that affect how fast the *baselines* run, so that they are not strawmen.

| date | hypothesis | change | measured effect | kept |
|---|---|---|---|---|
| 2026-09-20 | A dense `n x M2` masked matmul would make the Kim decoder cost O(n^2) on GPU | store the masked layer as per-row (value, index) tables, `n x (3b+2db)`; gather + einsum | structural (dense is infeasible beyond 128²: 16129 x 161380 weights); table form verified equal to the paper's dense mask (`test_mask.py`) | yes |
| 2026-09-20 | Forming the decoder Jacobian with `vmap` over K directions needs `n*P*K` floats (21 GB at 512², K=32) | `lspg.make_rollout(jac_batch=...)`: `lax.map` over latent directions when the gather exceeds 6 GB | pending first cluster job | pending |
