**GO for retry**, with one qualification: literal payload identity is false.

- Batch script differs only by `ev2 → ev2r` names/paths and `--exclude=pax007`.
- All **117 staged checksums verify**; **112 hashes are unchanged**. Five changes: batch script, commit/provenance metadata, `stage.py` adding exclusion support, and job JSON updating attempt/purpose/exclusion. Computational sources and data are identical.
- Logs confirm CUDA initialization failed on **pax007, device 2** during preflight; the shell exits **42** before launching any evaluation or timing.

No files modified.