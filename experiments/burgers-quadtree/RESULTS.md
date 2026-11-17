# Burgers quadtree pilot: scoped negative result

This completed development pilot does not establish an accurate, faster quadtree NM-ROM. It retains a verified reference pipeline, reproduced incumbents, trained classical/neural local representations, matched-data controls and actual dense-output PDE measurements. These are development results, not untouched-test confirmation.

## Decision

Stop adaptive-selector development for these checkpoints: after both registered repairs, the best worst evolved projection error is 1.63335%, and 2/8 cases meet the 0.5% screen. This configuration uses coarse rank 64, depth 2, local prefix 16 and 320 active coefficients. Projection uses the pilot grid and known snapshots; it is not a PDE result.

The fixed-space rollout also fails the physical target and is slower than the same-allocation classical control. This does not rule out larger training sets, different neural representations, specialized local networks, nonlinear local coordinates or the separate localized family. None of those unrun extensions is claimed complete.

## Reference and output qualification

All 8 registered development cases have regular second-order finite-difference space/time ladders. The largest conditional reference uncertainty is 0.010294%; the largest independent spectral disagreement is 0.009002%. Manufactured fixtures show the expected order and every numerical residual passed. These are numerical convergence assessments, not rigorous error bounds.

The reference extension was checked on successively denser grids and every final reference field was saved on exactly 4096×4096 boundary-inclusive nodes. Independent NumPy audits reproduced the stored projection errors and the saved dense rollout errors, including deliberate wrong-normalization detection.

## Actual PDE diagnostic, common input/output

All methods start from the same resident 4096×4096 input array and return a resident final field of that shape. Restriction, initialization, stepping and decoding are charged. Fixed ROMs use backward Euler and exact sign-upwind on 128 intervals, with four orthonormal sine tests per coefficient. Every saved step met the stationarity criterion. The following warm ROM measurements are one invocation per case; spectral rows use medians of five repetitions. They are exploratory costs, not the registered final confidence-interval speed test.

| Method | Worst evolved error, internal scoring (%) | Worst final error, actual4096 (%) | Median warm query (ms) |
|---|---:|---:|---:|
| pod128 | 6.38169 | 5.96128 | 100.922 |
| pod192 | 5.72658 | 5.72595 | 161.957 |
| localpoly | 5.67645 | 5.67582 | 221.641 |
| localneural | 5.67790 | 5.67726 | 882.431 |
| Spectral N64 (all evolved errors on actual4096) | 0.00902 | 0.00813 | 57.885 |
| Spectral N128 (all evolved errors on actual4096) | 0.00901 | 0.00811 | 120.199 |

The ROMs are inaccurate at these settings; the table is not a matched-accuracy speedup comparison. Global reusable basis/test tables are prepared outside the query. Their preparation time was not measured, so recorded cold-query times are compilation-inclusive subtotals, not full cold setup costs. Tree selection, coarsening, hyper-reduced local equations and final-panel timing remain unqualified.

## Tuned classical output policy

This separate paired allocation compares exact sine synthesis with boundary-aligned bilinear extension of a smaller spectral field. Both return actual4096 output. It cannot supply cross-allocation ROM timing ratios.

| Output policy | Median query (ms) | Worst final physical error (%) |
|---|---:|---:|
| sine | 56.893 | 0.00813 |
| bilinear256 | 52.808 | 0.02016 |

Largest interpolation discrepancy over all required evolved times: 0.02173%. Output-policy tuning does not change the reference or accuracy requirement.

## Nonlinear comparator and training limits

The historical frozen nonlinear head and direct bank were reproduced on exposed development cases. New global POD and local models used the same train32/dev8 snapshots. Local POD uses an approximate interpolated residual basis, not an asserted optimal smooth basis. The local neural bank completed the initial throughput pilot and a full mixed-configuration training repair; all checkpoints and failures are retained.

POD-autoencoder encoder reconstruction is not best manifold projection. Its linear POD truncation was reported separately; the following floors are measured on the pilot grid. No autoencoder rollout superiority or failure beyond these tested representations is inferred.

| POD lift | Worst evolved POD truncation (%) | Best tested encoder reconstruction worst error (%) |
|---|---:|---:|
| 128 | 5.19806 | 15.66990 |
| 256 | 1.44562 | 19.89323 |

## Discretization diagnosis

| Full-order upwind mesh | Step | Worst evolved error (%) |
|---|---:|---:|
| 128 | 0.00125 | 5.22060 |
| 128 | 0.005 | 5.65114 |
| 256 | 0.00125 | 2.88733 |
| 512 | 0.00125 | 1.60557 |
| 1024 | 0.00125 | 0.93729 |

This isolates the bias of the fixed upwind/backward-Euler diagnostic. Shared rollout error must not be attributed entirely to the learned features. These full-order times are not used in cross-job speed ratios.

## Resources and reproducibility

10 owned allocations, 1281 allocated A100 GPU-seconds (0.355833 GPU-hours), 2.846667 allocated CPU-hours. Compilation and allocation overhead are included. Local GB10 tests were limited to three sub-minute smoke calls; their combined configured wall-time ceiling was 165 seconds, but total local GPU time was not instrumented. No local production training or experiment ran.

Branch: `exp/2026-11-17-burgers-quadtree`, forked from the prescribed consolidated baseline. Code-only GitHub mirrors preserve every implementation commit. `vendor/PROVENANCE.json` ties every inherited source/checkpoint to Git and SHA256. `cluster/ledger.json` records job IDs, limits and source commits. `evidence/*-retention.json` binds the local and owned remote raw arrays, checkpoints, logs and scripts. `AMENDMENTS.md` records prospective changes; `DECISION-A0.md` scopes the stop. Untouched final-test solutions were never generated, no unrelated jobs or archived results were changed, and no branches were merged.

## Glossary

- **Projection:** fit a known field; this is a representation diagnostic, not a forecast or physics solve.
- **POD / POD-autoencoder:** a linear snapshot basis / a nonlinear latent map inside that basis.
- **Rank or prefix:** number of active basis coefficients; distinct from network width and latent dimension.
- **Stationarity:** the weak-residual optimization reached its declared derivative criterion; this does not imply physical accuracy.
- **Warm query:** solve and requested output using already prepared reusable tables.
- **Reference uncertainty:** a conditional numerical convergence estimate for the comparison solution.
- **Actual4096:** exactly 4096 nodes per axis, including boundaries, rather than 4096 intervals.
