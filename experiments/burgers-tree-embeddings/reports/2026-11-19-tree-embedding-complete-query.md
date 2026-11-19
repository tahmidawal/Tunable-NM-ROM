# Burgers tree embeddings: complete fixed-query cost diagnostic

Complete-query diagnostic remains in progress. Outstanding: b5normal, b5compact, b5block0, b5block1.

The direct24/52-node tree and stable global960 control use the admitted fine weak PDE, numerical tolerances and training data. Reusable tables cover all340 cells through depth4. Source verification, cache construction/storage and training-only prewarming are recorded separately. Every query charges initial sampling/selection, gathering, visibility, initial fitting, solving and actual requested reconstruction. Repeated timed queries disable internal stopwatches and synchronize all returned work. These fixed trees have no refinement trials or fallback; their zero counts do not predict adaptive cost.

|Run|Invocations|All gates pass|Distinct actual fields checked|
|---|---:|---|---:|
|b5cache|6|True|6|

Paired cost results are not yet admitted.



Instrumented diagnostic components fromb5cache, in seconds. These single invocations locate work; the repeated uninstrumented measurements above determine cost comparisons.

|Case|Method|Selection/assembly|Initial fit|PDE stepping|Requested output|Whole query|
|---|---|---:|---:|---:|---:|---:|
|dev000|global_stable960|0.000004|0.021048|4.895241|0.000489|4.917818|
|dev000|tree_direct24|0.164411|0.010549|18.024940|0.001765|18.202961|
|dev003|global_stable960|0.000002|0.011631|4.361309|0.000505|4.375208|
|dev003|tree_direct24|0.192118|0.010397|14.711364|0.001775|14.917050|
|align002|tree_direct24|0.168064|0.011658|17.606749|0.001639|17.789929|
|align002|global_stable960|0.000006|0.010146|4.845809|0.000542|4.857576|

Ratios always compare invocations on the same GPU/allocation. The two allocations use opposite randomized method orders, with five repetitions per case and output contract. Accuracy and cost use the same returned states. One final4096-square device field is the primary contract; five4096-square fields use separate complete invocations. Host transfer is recorded separately. Each distinct float64 output is retained by raw-byte hash; exact repeats map to that retained array. No coefficient-only output substitutes for a requested field.

This panel contains only three development trajectories. The exact small-panel bootstrap resamples trajectories, not repetitions; its interval is a diagnostic with limited coverage, not final confirmation. The global control is POD plus prescribed Fourier completion, not a native Tunable/CROM/SNF/masked-autoencoder reproduction. Native competitors, deployable adaptation, method freeze and unopened final-test solutions remain separate requirements. An unqualified timing result is retained but cannot support a speed claim.

Latest scheduler snapshot: 35303 allocated GPU-seconds (9.806389h), with full outstanding reservations charged to36503 seconds against the cumulative cap42296. Active owned jobs: b5normal. Generated 2026-11-19T05:15:53.314063+00:00. The old campaign remains paused; no merge or unrelated3D work.

## Glossary

- **Complete query:** the known initial callable enters, and actual requested device fields leave; all query-specific work is timed.
- **Reusable setup:** fixed-decoder tables and training-only compilation that can serve previously unseen inputs; construction and storage remain reported.
- **Compact/padded:** local patch arithmetic versus full-axis contractions containing support zeros; both implement the same decoder.
- **Paired ratio:** tree query duration divided by the global query duration from the same case, repetition and allocation. Below1 is faster.
- **Trajectory bootstrap:** resampling whole cases to summarize uncertainty, without treating repeated measurements as independent cases.
- **Equivalence gate:** field change at most1e-7 of the current reference norm, alongside physical and full solver-convergence checks.
- **Allocated GPU time:** complete scheduler duration, including compilation, failed jobs, scoring and output storage.
