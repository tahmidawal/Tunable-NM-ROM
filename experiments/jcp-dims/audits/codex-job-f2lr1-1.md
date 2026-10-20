SUBMIT

- **Partial archive verified:** `PARTIAL` present; all 15 `OUTPUTS.sha256` entries pass. Exactly **14 complete settings**, each with 38 cases and six tighter dev cases: `R384M1536lmf` + Gauss `{64,96,128,160,192,256,320,384}` and Smolyak `{8,9,10,11,12,13}`. All arms are **LSPG/CNR, factor 1**.
- **Retry exact:** `f2lr1.json` matches `make_retry.py f2l 1` logic: only missing `R384M1536lmfsmolyak14`, unchanged; no blocks.
- **A9.3 merge compatible:** numerical code/input signatures match; non-exempt job fields match. Discovery supports both output stems and merges newest-first per arm. Original GPU is H200; retry requests H200. Actual retry GPU identity remains checked after collection.
- **Staging verified:** all 182 manifest entries pass; all 178 provenance files match staged content and HEAD `bcda08d79`. Batch script has GPU partition, H200, 0.5 hours, GPU preflight exit 42, f64/highest, and paralab output/cache/temp paths.
- **Certificate valid:** config hash matches; projection **0.0171 h**. Certificate’s rounded minimum is **0.25 h**; staged **0.5 h** exceeds it.
- **Budget passes:** **29.609 + 8 + 3.75 + 0.5 = 41.859 GPU-h ≤ 55**, leaving **13.141 GPU-h**.
- **Queue passes:** SSH check found two dims jobs (`6056624`, `6072828`), four account-wide. Adding this job gives **3 lane / 5 account-wide**.

Read-only audit completed; no files changed or jobs submitted.