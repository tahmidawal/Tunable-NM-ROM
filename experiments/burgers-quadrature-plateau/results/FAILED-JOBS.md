# Failed jobs of this lane (logs not retained — transcribed from the session before the job directories were deleted)

| job | slurm id | GPU / node | state, elapsed | error read from the log | disposition |
|---|---|---|---|---|---|
| p2d256 | 6897098 | A100 80GB PCIe, pax050 | FAILED 00:02:06 | `plateau2d.py` line 169 `assert not bad`: `['gauss24', …, 'gauss256']` (every Gauss rule's sha256 differs from the local manifest) | amendment A3 |
| p3d65 | 6897571 | A100 80GB PCIe | FAILED 00:00:11 | `plateau3d.py` line 116 G-coh assert: table sha256 `85cef1e7…` (expected October `f20a1aac…`) | amendment A4 |
| p3d257 | 6901029 | H200, pax011 | FAILED 00:00:26 | `plateau3d.py` line 116 G-coh assert: table sha256 `f20a1aac…` (expected E0 `85cef1e7…`) | amendment A5 |
| p3d257b | 6902975 | H200, pax010 | FAILED 00:00:48 | `tables.py` line 45 `build_tables` → `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 32.00GiB [jit_concatenate]` | bank rows built in two 256-column blocks (`table_ladder_extra=[256]`), rerun as p3d257c |

All four stopped before any rollout was computed. Their remote directories under `/cluster/tufts/paralab/tawal01/bqplateau-20261105/` were deleted (p2d256 was empty except `logs/` at that point for the first, aborted staging; the others after failure) without pulling the logs — retained here as transcription only.
