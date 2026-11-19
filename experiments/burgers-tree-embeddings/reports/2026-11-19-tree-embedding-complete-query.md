# Burgers tree embeddings: complete fixed-query cost diagnostic

Complete-query diagnostic remains in progress. Outstanding: b5compact, b5block0, b5block1.

The direct24/52-node tree and stable global960 control use the admitted fine weak PDE, numerical tolerances and training data. Reusable tables cover all340 cells through depth4. Source verification, cache construction/storage and training-only prewarming are recorded separately. Every query charges initial sampling/selection, gathering, visibility, initial fitting, solving and actual requested reconstruction. Repeated timed queries disable internal stopwatches and synchronize all returned work. These fixed trees have no refinement trials or fallback; their zero counts do not predict adaptive cost.

|Run|Invocations|All gates pass|Distinct actual fields checked|
|---|---:|---|---:|
|b5cache|6|True|6|
|b5normal|6|True|6|

Paired cost results are not yet admitted.



Instrumented diagnostic components fromb5normal, in seconds. These single invocations locate work; the repeated uninstrumented measurements above determine cost comparisons.

|Case|Method|Selection/assembly|Initial fit|PDE stepping|Requested output|Whole query|
|---|---|---:|---:|---:|---:|---:|
|dev000|global_stable960|0.000010|0.023612|4.504307|0.000508|4.529387|
|dev000|tree_direct24|0.166529|0.010789|17.194734|0.001843|17.375303|
|dev003|global_stable960|0.000001|0.009102|3.720572|0.000522|3.731592|
|dev003|tree_direct24|0.197258|0.010561|13.903286|0.001667|14.114029|
|align002|tree_direct24|0.175318|0.011264|16.757714|0.001817|16.948334|
|align002|global_stable960|0.000003|0.010093|4.511556|0.000534|4.523154|

Ratios always compare invocations on the same GPU/allocation. The two allocations use opposite randomized method orders, with five repetitions per case and output contract. Accuracy and cost use the same returned states. One final4096-square device field is the primary contract; five4096-square fields use separate complete invocations. Host transfer is recorded separately. Each distinct float64 output is retained by raw-byte hash; exact repeats map to that retained array. No coefficient-only output substitutes for a requested field.

This panel contains only three development trajectories. The exact small-panel bootstrap resamples trajectories, not repetitions; its interval is a diagnostic with limited coverage, not final confirmation. The global control is POD plus prescribed Fourier completion, not a native Tunable/CROM/SNF/masked-autoencoder reproduction. Native competitors, deployable adaptation, method freeze and unopened final-test solutions remain separate requirements. An unqualified timing result is retained but cannot support a speed claim.

Latest scheduler snapshot: 35749 allocated GPU-seconds (9.930278h), with full outstanding reservations charged to35749 seconds against the cumulative cap42296. Active owned jobs: none. Generated 2026-11-19T17:01:43.787556+00:00. The old campaign remains paused; no merge or unrelated3D work.

## Glossary

- **Complete query:** the known initial callable enters, and actual requested device fields leave; all query-specific work is timed.
- **Reusable setup:** fixed-decoder tables and training-only compilation that can serve previously unseen inputs; construction and storage remain reported.
- **Compact/padded:** local patch arithmetic versus full-axis contractions containing support zeros; both implement the same decoder.
- **Paired ratio:** tree query duration divided by the global query duration from the same case, repetition and allocation. Below1 is faster.
- **Trajectory bootstrap:** resampling whole cases to summarize uncertainty, without treating repeated measurements as independent cases.
- **Equivalence gate:** field change at most1e-7 of the current reference norm, alongside physical and full solver-convergence checks.
- **Allocated GPU time:** complete scheduler duration, including compilation, failed jobs, scoring and output storage.
