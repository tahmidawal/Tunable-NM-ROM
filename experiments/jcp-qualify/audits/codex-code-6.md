**FIX-FIRST** at `7de61f277`.

Paths relative to `experiments/jcp-qualify/`.

| Previous must-fix | Verdict | Evidence |
|---|---|---|
| Fallback reservation validation | **PARTIALLY** | Zero-hour fallback rejected, but timing repricing remains bypassable (`ledger.py:78–83`). |
| Pilot decoded-output footprint | **RESOLVED** | Holds six decoded outputs simultaneously, covering timing’s five retained outputs plus replacement overlap (`qpilot.py:71–92`). Static assessment; no GPU execution. |

Ranked must-fix:

1. **New: fallback validation never finishes its lifecycle.** After submitting an enlarged first timing panel—or its repeat—validation demands another *pending* repeat. Subsequent reservations fail indefinitely (`ledger.py:79–83,106`). Recognize already-live/completed panels that cover the fallback subjects. Reproduced in memory.

2. **Repricing remains a marker-only check.** Replacing `fballow256: 2` with `q2known256: 0`, `fb256: 0.001`, `time256enlarged: 0` passes validation while `time256` stays unchanged at 1.5 hours (`ledger.py:78–81`). Require a validated projection covering the enlarged panel.

Checks passed:

- **Restaging:** all **136 source files/job** equal HEAD; all **140 manifest hashes/job** pass. Commit, provenance, reservation and batch time limits agree for smkA/smkB/smk3.
- **Reservations:** installed pending inventory = **22.5 GPU-h**. Sequential **2.5 + 2 + 1.5 h** reservations pass: **25 → 27 → 28.5 GPU-h**, leaving **6.5 h** under 35. Projections remain a-priori.
- Pending-inventory installation remains **RESOLVED** (`runs/GPU-LEDGER.json:6`).

No files changed, scratch created, cluster jobs submitted, or GPU runs performed.