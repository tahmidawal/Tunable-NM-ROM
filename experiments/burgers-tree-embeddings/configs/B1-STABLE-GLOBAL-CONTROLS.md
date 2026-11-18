# Resolve the numerical-null global-control ambiguity

The saved high-rank global POD space is more accurate than the qualified direct tree at fewer variables, but its tail is at numerical training rank deficiency. Keep that result and test its stability rather than treating it as a robust learned-space win or discarding it.

Finite diagnostic, maximum20 allocated GPU-minutes: recompute the identical weighted training SVD under exactly two fixed row permutations (seeds2026111887 and2026111888), then score prefixes1024/1280/1376. No new trajectories, optimized weights or development-dependent ordering. Separately preserve the trained POD512 prefix and append data-independent tensor-sine directions in ascending frequency, twice orthogonalizing and rejecting candidate residual norm at most0.001. Freeze the complete basis before reading development fields. Score total counts768/960/1024/1280/1376. Record orthogonality, accepted indices and candidate residual norms. The same physical normalization and all snapshots apply.

This is an explicitly named POD-plus-Fourier global-space control, not a pure POD reproduction or new neural training recipe. Its purpose is a stable strong global alternative at comparable variable count. No PDE-speed conclusion follows. All original pure-POD controls remain in the report. The job must fit B1 and the two-job limit before submission.
