**k3a: FIX-FIRST. k3b: FIX-FIRST.** Square pricing is repaired, but the split has not been propagated to the analysis consumer.

- **Blocker:** [`analyze3d.py:108`](experiments/jcp-dims/analyze3d.py:108) discovers only `m3a`, `m3b`, **`k3`**, `b3f`, and `b3m`. It never loads `k3a` or `k3b`; both jobs’ evidence would be silently omitted. Add both attempts and verify their combined inventory reaches the existing A3/B6 analysis.

The requested resource and staging checks otherwise pass:

| Check | k3a | k3b |
|---|---:|---:|
| Recomputed projection | 8.396626 h | 3.435396 h |
| Staged limit | 12.75 h | 5.25 h |
| In-memory reservation total, individually | 51.3568 h | 48.8168 h |
| Manifest verification | 175/175 | 175/175 |

Both sequential reservations total **53.1668/55 GPU-h**, including remaining pending core projections. Ledger validation returns no errors; completed charges agree with `sacct`.

- **Exact split verified:** concatenated settings and blocks equal `configs/k3.json`, without omissions or duplication. Shared scientific configuration matches; attempt/mode identifiers differ as expected.
- **Square pricing verified:** **46.2935 s/unit**, calibrated on smoke W512k1, with point-count scaling floored at one. Smoke tighter coverage is 1/2 versus production 8/64; retaining the smoke average does not discount that extra smoke work. This remains an estimate.
- Both bundles’ provenance matches committed blobs at `13acb0c6086a081910ed04e50b2da58fd2cf5490`. Config hashes, reservations, H200 requests, GPU preflight, f64/highest, and isolated output paths agree.

**A10: ACCEPTABLE for cap accounting, not a guarantee of core completion.** The brief permits smoke-certified projections; A10 still reserves each submitted job at its full scheduler limit. Counting unsubmitted work at projection therefore preserves the admission safeguard for the **55 GPU-h measured cap**, while weakening protection against future core shortfalls. Any such shortfall must remain explicitly reported under A9.2.

**Live capacity:** account **8/8**, lane **2/3** at inspection. Neither job may submit now; after capacity opens, only one additional lane job fits until another finishes. The wrapper checks these caps.

No files changed, ledger written, or jobs submitted.