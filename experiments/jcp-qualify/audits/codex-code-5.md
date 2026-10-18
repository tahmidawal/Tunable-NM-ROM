**FIX-FIRST** at `bd207e3a2`.

Paths below are relative to `experiments/jcp-qualify/`.

| Code-4 must-fix | Verdict | Evidence |
|---|---|---|
| 1. Reservation enforcement | **PARTIALLY** | Minimum allowances, reservation-time validation, and failed-job restoration added (`ledger.py:68,96,162`). Fallback replacement remains bypassable; see #1 below. |
| 2. Full pilot footprint | **PARTIALLY** | All initial fields and split-query/certifier added (`qpilot.py:48,77`). Still underrepresents simultaneous decoded outputs; see #2 below. |
| 3. Install pending inventory | **RESOLVED** | `runs/GPU-LEDGER.json:6` contains all eight mandatory jobs plus three 2-hour allowances. Validation passes. |

Ranked must-fix:

1. **Fallback coverage still accepts zero-cost placeholders.** Replacing `fballow256: 2` with `q2known256: 0, fb256: 0` returns **no validation errors**, leaving timing projection unchanged. `ledger.py:65–73` checks presence/prefix, not positive fallback reservations or A5.1’s enlarged timing projection. Require validated fallback projections and the corresponding timing update in the replacement transaction.

2. **Pilot still does not establish timing’s peak footprint.** `qpilot.py:71–89` holds two decoded outputs (`v`, `F`) and deletes them each iteration. Actual timing retains decoded outputs in **`F`, `fq`, `w`, `q_`, and `qq`** during successor replays (`qtime.py:138,164,170,177,185`), including across iterations. Reproduce those lifetimes or supply a conservative bound before accepting `feasible` at `qpilot.py:97`.

Checks passed:

- **Restaging:** all **136 source files per job** equal committed HEAD; all **140 manifest checksums per job** pass. COMMIT/provenance/reservation metadata agree for smkA, smkB, smk3.
- **Reservations:** in-memory evaluation of the ledger’s pure functions passes sequential reservations of **2.5 + 2 + 1.5 hours**. Pending inventory **22.5 hours**; successive commitments **25.0 → 27.0 → 28.5 GPU-h**, leaving **6.5 hours** under the cap. These remain a-priori projections.
- Prior closures—fallback timing pricing, attempt/source discovery, P1 manifest comparison—remain unchanged.
- No additional independent blocker found in the diff.

No files changed, scratch created, cluster jobs submitted, or GPU runs performed.