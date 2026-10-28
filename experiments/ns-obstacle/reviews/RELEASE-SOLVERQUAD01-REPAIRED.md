# Release corrected P4 solver/quadrature controls

Release exactly one fresh g2-solverquad01 allocation from a886389af6b2182f39ed84cb7fc79d931183e55d, inventory ef7970411c2c163497d21642d8e25b497ede1ce8b0d5d0b68919612defa46ad9. This supersedes the unsubmitted package with missing dependency metadata. All46 inputs, including20 runtime files, passed independent exact-byte and dependency-closure review; the exact bundle is retained in repaired-a886389af-runtime.tar.gz. Stage this committed source, not later worktree edits or the optional batched producer.

Apply the unchanged solverquad scope and limits: identical adopted P4 mesh; separate100-fold tolerance tightening and assembly+4/diagnostic16 controls; full higher-rule comparison with adopted denominators. One GPU/16CPU/64GiB,48GiB hard working guard,600seconds per process,1100seconds runner,1200seconds allocation. Role point caps1.3M solver/2M quadrature remain enforced. Mandatory GPU/f64/highest, fresh directory, queue-before/after, two-job maximum and disk reserves apply.

Both previous crossp02 and capacity04 jobs are terminal. This is one bounded allocation, with no automatic retry, reference admission or training. Preserve failures and complete checksummed retention. Domain packages remain separately execution-held.
