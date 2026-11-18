# B1 recipes 3 and 4: stronger matched-data global controls

Train normalized global heads with 64 and 128 adjustable variables on the expanded 128-trajectory training cohort. Both use the same training-only spectral POD768 coordinate span, centered output, unit initial latent scales, unrestricted global offset, three width128 tanh decoder layers, encoder256/128 and linear skips as recipe1. Keep the same Adam learning-rate schedule, epsilon, batch size, update cap, refits and checkpoint selection from B1-NORMALIZED-PARENT.md. Register both dimensions before execution; they are two trained recipes, not a search over dimensions.

The development panel is the original sixteen random cases, sixteen prefix-stable new cases and the same twelve alignment cases. Learn no weights or coordinate features from development. Report each panel separately. All optimised reconstruction errors use the current-time physical norm including the part outside POD768. Select checkpoints by all-case worst error, with median as tie-breaker. Record encoder error and optimized error separately.

These are architecture-inspired spectral-feature NM-ROM controls, not faithful reproductions of the native coordinate-MLP Tunable bank. The original matched-data coordinate-MLP controls remain in the report with their limitations. These stronger spectral controls prevent comparing a richer tree only against the weak original parent. They do not complete the planned NM-ROM confirmation stage.

Reserve at most 60 allocated GPU-minutes per model within B1, at most two owned jobs including the local-cache job. No final-test access or PDE-speed claim. The next decision compares local coefficient compression with adding global variables on this common expanded dataset.
