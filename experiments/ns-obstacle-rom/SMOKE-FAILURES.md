# Infrastructure smoke failures

The first CPU synthetic check completed its numerical assertions but failed while serializing a NumPy boolean in the directional-derivative diagnostic. The boolean is now converted explicitly to a Python boolean. No physical experiment or reference admission was attempted; the rerun is preserved in `CPU-SMOKE.json`.

A subsequent generic weighted-Gram check found an invalid Einstein-summation ellipsis when vector components were present. The implementation now flattens component axes and repeats physical weights explicitly; the vector and scalar cases are included in the saved smoke checks.

Constant pressure snapshots initially produced a spurious centered pressure mode because the rank test normalized by the tiny roundoff-sized centered matrix itself. Centered singular values now use the recorded original weighted pressure-matrix scale with a unit floor. The exact constant remains a pressure unknown; no physical pressure gauge shift is introduced. The constant-only fixture is retained in the smoke checks.
