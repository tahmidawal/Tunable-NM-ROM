# Native preflight failure and source ownership audit

The second allocation passed the private-prefix integrity and native imports, then the
API/inverse preflight process terminated with a segmentation fault. No flow case started.
Its exact failing native call is unresolved because the original process had no per-call
markers or Python fault reporting, and no core file was retained in its job directory.
The job ledger and full failure archive retain the measured outcome and charged cost.

A concrete source hazard was identified in two 2D preflight expressions: they constructed
a mapped point using a temporary element transformation, then accessed the mapped point
after the transformation could be destroyed. The pinned binding gives the transformation
ownership to Python but does not keep it alive through the mapped point's call binding.
A named transformation must outlive mapped point access. Integration-rule indexing instead
returns its point by value and does not establish the same hazard.

This is a source-identified defect and a hypothesis for the observed crash, not a measured
native backtrace. One bounded ordinary repair will retain transformations, copy retained
arrays, and add fault reporting plus flushed stage markers. Formulas and thresholds stay
fixed. Independent exact-diff review and fresh staging remain required before a retry.

Pinned upstream source bytes and hashes are in `pinned-api/SOURCE-INVENTORY.json`. The web
reader returned older cached line numbers for the tagged URL; direct downloads determine
saved evidence. The 3D author independently verified its transformation lifetimes are
already retained; the ROM adapter is applying the same ownership rule.
