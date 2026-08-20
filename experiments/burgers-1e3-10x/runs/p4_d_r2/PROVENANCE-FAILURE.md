# Excluded P4-D provenance failure

Job 2668613 completed with exit code zero on an H200 and its remote checksums
verified, but the independent audit stopped before reading or accepting any
scientific metric or decision.  The root staging manifest excluded every file named
`MANIFEST.sha256`, unintentionally omitting the nested immutable P3 staging manifest.
Therefore this run is excluded from all scientific selection, accuracy, cost, and
cell-count claims.  Its raw result files are retained only as immutable excluded
evidence and were not inspected before this classification.
