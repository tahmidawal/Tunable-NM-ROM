**S1: FIX-FIRST. J2: FIX-FIRST.** Early-exit evidence remains broken.

Paths below are relative to `experiments/imp-multimesh/`.

| Requested fix | Status | Evidence |
|---|---|---|
| Preserve uncertain submission reservations | **FIXED** | `cluster/submit.sh:59–67` checks both lookup statuses and requires exactly one candidate before binding. Inconclusive results retain the reservation. **Caveat:** an empty candidate list causes the pipeline at `:62` to exit under `set -e`; the reservation remains `RESERVED`, rather than becoming `AMBIGUOUS`. This preserves the budget. |
| Early-exit structured evidence and checksums | **NOT FIXED — both jobs** | Python consumes the escaping in `cluster/stage.py:114` and `cluster/stage_probe.py:108`. Generated `runs/s1/run.sbatch:24` and `runs/j2/run.sbatch:17` emit **invalid JSON**, reproduced read-only: `{early_exit: 3, reason: manifest}`. Checksum generation is now reached, and exits remain nonzero, but S1 also does not produce failed `ACCEPT.json` on these paths. |
| Restage, manifests, certificate hashes | **FIXED for current artifacts** | All **146 S1** and **63 J2** manifest entries verify. S1’s config matches `checks/resources-s1.json:7` and `runs/s1/MANIFEST.sha256:18`; J2’s script matches `checks/resources-j2.json:7` and `runs/j2/MANIFEST.sha256:63`. |

No additional GPU-hour-waste or cross-lane namespace defect found in these changes. The empty-lookup issue conservatively holds budget rather than releasing it.

Fix JSON generation in both staging templates, provide S1’s failed acceptance evidence, then restage and refresh affected hashes. Shell syntax checks alone miss this quoting defect.

Read-only audit: no files changed, jobs submitted, large arrays loaded, or writes to main or `LAB-LOG.md`.