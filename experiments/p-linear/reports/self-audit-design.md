# Self-audit of DESIGN.md and the drivers before job 1 (substitute for the Codex audit)

Codex (gpt-6-astra) refused every call with a usage-limit error until 2026-09-19 11:33; a
fresh-context Claude substitute agent was killed by an API session limit before it read a
file. Per the coordinator's notice appended to LANE-PROTOCOL.md, this is a written self-audit:
each claim the design or the drivers rest on, what it rests on, and the check run. The
independence guarantee is weaker than a second model family and the report says so.

| # | claim / risk | rests on | check run | outcome |
|---|---|---|---|---|
| 1 | D1–D3 cannot be gamed by arm choice | DESIGN §4 fixes the ladder (`m4` rule, `new_K32`, six rungs), the comparator set and both metrics before the job | re-read §3–§4 after the smoke; the only post-smoke change (A4) *removes* an artefact that would have made D2 fail spuriously and was declared before submission with the smoke numbers | D1–D3 unchanged; A4 tightens, does not loosen |
| 2 | q ≤ 32 rungs are the parent's arms bit for bit | `extend_basis` copies `basis['coefficient_directions']` verbatim as columns 1..32; `retained_prefix_exact` asserted | smoke `q32_m256@new_K32` vs pbh02 `a_neural_q32@new_K32`: 3.3e-13 physical, 8.9e-14 same-grid; `q32_m4` vs `q32_m256`: 0.0 | pass |
| 3 | the extension is nested and orthonormal in the field metric | SVD of the residual projected off V32 = R_G C32; `extension_orthonormality_error` | 4.6e-10 with the full 2611-source fit split (first smoke); assert at 1e-7 | pass |
| 4 | the training-mesh metric R_G matches the retained basis's | `retained_metric_R_max_abs_difference` | 5.6e-17 on the GB10 | pass; cluster value recorded in the job |
| 5 | rebuilding the 32 from scratch would NOT reproduce them | `rebuilt_prefix_subspace_defect` | 6.0e-6 (full split); 0.9995 (640-source smoke subset) | recorded as A3; verbatim prefix is the right choice |
| 6 | q = R rung reaches the bank floor | eliminated path with 512 columns, M = 2176 | smoke worst same-grid 0.0080923 = floor 0.0080923 | pass |
| 7 | q = R eliminated LM is inert and its cost is an artefact | `jacobians`, `reason`, `total_seconds` of `q512_m4` | 194 Jacobians, reason 0 (budget), 106 ms vs 7 ms at q = 256 | A4: top rung = direct QR solve |
| 8 | identity-head free bank equals the LS optimum | `d_freebank_m4` vs floor | 0.845 % vs 0.809 %: gradient-tolerance exit on cond(B)² ≈ 7e14 normal equations | reported as a finding; not the top rung (A4) |
| 9 | M > unknowns at every run rung | skip rule `M <= K+q` (q < R) or `M <= R` (q = R) | smoke skips `q256_m256`, `q512_m256`, `e_pod512_m256`; every built arm asserts `linear_rank_valid` | pass (A1) |
| 10 | POD-LSPG with identity head is well posed at M = 4k' | `e_pod512_m4` M = 2048 > 512; stationary in 2 iterations | smoke 12/12 reason 4 | pass |
| 11 | gates compare the same computation | pairs in `config-1024.json`: same kernel (`PA.make_query` for `a_neural`/`d_freebank`/`e_pod`, `CC` for `q32`), same M = 257, same directions, same nearest-code initializer; ccpoi01 pairs use the ccpoi01 direction rule via `directions.audited` with its seed/counts | all 17 non-degenerate smoke gates ≤ 7.8e-13 | pass |
| 12 | `dst_direct` same-grid gate is 0/0 | both sides ~1e-16 | ratio 0.256; A5 absolute floor 1e-12 | fixed |
| 13 | flat staging has no name collisions | `stage.py` asserts `not dest.exists()` per flat name | dry inspection of SOLVE/HEAD lists: all basenames distinct (`core.py` only from multiresolution-poisson; `pilot.py` likewise) | pass |
| 14 | config paths resolve flat | `resolve()` falls back to basename | smoke resolves relative paths; flat fallback exercised by staging the same config | pass by construction; verified on first cluster log |
| 15 | memory at 1024 | banks 4.3 + 1.1 GB, POD modes 4.3 GB, `whiten(bank_inc)` 1.1 GB, reconstruction Jacobians ≤ 2 GB, ~37 small kernels | parent pbh02 held 2 R=512 banks + 20 kernels on 80 GB; `XLA_PYTHON_CLIENT_PREALLOCATE=false` | expected < 25 GB; a 40 GB A100 suffices |
| 16 | `jax.clear_caches()` never invalidates a live kernel | called only after `extend_basis` (no kernels built yet for that model... but `built` holds kernels of the previous model) | the previous model's kernels are Python-held jitted functions; `clear_caches` drops compilation caches, not live executables of held `jax.jit` objects — they recompile on next call | first invocation after warm-up is not timed (warm-up pass precedes the timed grid) |
| 17 | closures in loops capture correctly | `head = lambda z, p=L['params']`, `pred` lambdas bind `Bk=Bk` | reviewed each `lambda` in the loops | pass |
| 18 | development sources never select anything | no selection in jobs 1–2; job 3 selects by the validation split only | grep for `dev` in `plin_head.py` selection code: `best = min(arms, key=validation)` in the generator | pass |
| 19 | job-3 arch knobs are honoured | `pbh_fit.train_head_phase` calls `sc.init_separable(..., **cfg['arch'])` with `h_hidden`, `h_layers`; `sc.head` applies whatever `params['h']` holds | read `pbh_fit.py` l.145–150 and `sep_common.py` l.84–135 | pass |
| 20 | job-3 shared bank | `g_track_sha` asserted equal between `bankarm_head.pkl` and `primary_K32.pkl`, and after every arm's training | code review | pass |
| 21 | H1 is well posed | same seed, same data, same recipe; XLA reduction order can differ across GPUs so bitwise identity is not expected — 5 % relative on dev best-found | stated | pass |
| 22 | timing contract differs between CC and PA kernels | CC rows carry stationarity/rank diagnostics inside the timed region | stated in DESIGN §3 and in the report; the direct QR arm and the PA arms are the fair pair for the top rung vs POD | recorded |

Items 7, 8 and 12 changed the design (A4, A5) before submission; nothing changed after a
cluster number existed.
