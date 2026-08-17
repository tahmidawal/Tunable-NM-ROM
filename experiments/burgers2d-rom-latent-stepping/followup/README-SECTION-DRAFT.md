## Follow-up (2026-08-17) — k ladder, multi-seed, EQ knobs, honest cost

`followup/` extends this study without touching anything above: new code in `followup/`,
new results in `runs/followup/<cell>/`, generated tables in `followup/FOLLOWUP_TABLES.md`,
figures in `followup/figs/` (also copied to `Plots/`).  Every table above is unchanged;
the K=4/8/16 rows of the frozen round are reused verbatim as three points of the new k
ladder.

Everything below: N=64 unless stated, 16 fresh test trajectories from `TEST_SEED=1`, f64,
`jax_backend=gpu`, `JAX_DEFAULT_MATMUL_PRECISION=highest`, one isolated cluster job
directory per cell, data regenerated on the cluster from the seed, checksummed pull,
cluster directories deleted.  Adversarial review before the fan-out:
`CODEX-REVIEW-followup.md` (all MUST items applied; disposition table in that file).

### New code

| path | what |
|---|---|
| `blat_train_ad.py` | + `TRAIN_SEED`: the FiLM network initialisation and the minibatch / collocation-point order.  The per-snapshot latents are initialised **deterministically** from the top-K POD coefficients, and the data draw, the split and the `TEST_SEED` test set never see it — so a multi-seed sweep varies training randomness only.  Default = the data seed, reproducing the frozen weights and latents; non-default seeds get their own file name (`_S<seed>`), as do their ROM reports. |
| `followup/fu_common.py` | an **exact** `lax.while_loop` port of `ms_autodecoder.lm_solve` (the Python-loop LM that `blat_common.fit_ic` uses for the online cold start, which dominated end-to-end time at 0.6–1.3 s): same damping, acceptance, both stopping tests applied only after an accepted step, same accounting.  Plus the nearest-training-IC rule by **field** distance at the tested mesh, matching `blat_rom.py`. |
| `followup/fu_timing.py` | one-GPU, one-process cost measurement.  `MODE=n`: N ladder at fixed (k, M, m) — the coordinate decoder is meshfree so the same N=64 checkpoint runs at every N, with the EQ weights refit per N and the FOM's own Newton residual asserted < 1e-8 before anything is timed.  `MODE=k`: K ladder + the POD control, and (with `POD_KS=` empty) the m/M cost ladder.  2 warm-ups, median of 7, `block_until_ready`, FOM = the testbed's own jitted implicit rollout at batch 1. |
| `followup/fu_summarize.py`, `fu_style.py` | tables + figures straight from the JSONs (verified to reproduce the frozen K=4/8/16 rows exactly). |
| `followup/cluster/{make_cell.sh,launch.sh,fu_cells.sh,pull.sh}` | `blat2/` namespace: one job per cell dir, sha256-verified staging and pull, `squeue` checked before and after every submit, `git_commit` + `git_dirty` recorded in every batch script, cluster dirs deleted after a verified pull. |

`blat_common.fit_eq_weights` additionally reports per-row NNLS fit diagnostics (median /
p95 / max of |Gw−b|/|b| over the fitted rows) so a single badly-fitted test mode cannot
hide behind a small global norm; the max is dominated by rows whose target projection is
near zero and should be read next to the median.
