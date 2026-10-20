**NEEDS-FIX.** Paths relative to `experiments/jcp-qualify/`.

| codex-code-6 must-fix | Verdict | Evidence |
|---|---|---|
| Fallback lifecycle | **PARTIALLY** | Live/completed repeats now count. Enlarged first panels still require an extra marker or repeat after submission/completion (`ledger.py:79–92`). |
| Validated timing repricing | **PARTIALLY** | Numeric enlargement check added, but baseline manipulation and placeholder markers bypass it (`ledger.py:84–90,133–136`). |

Remaining blockers, reproduced using pure functions on in-memory copies:

- **Baseline reset bypass:** install `time256: 0` before Q2 is known; validation accepts it and the pending handler overwrites the baseline with zero. The fallback transaction then accepts `time256: 0` (`ledger.py:65–66,84–86,133–136`). A5.1 remains unenforced.
- **Unverified coverage:** after an ordinary timing panel is submitted/completed, `time256coversfb: 0` suppresses the repeat requirement. Alternatively, `time256rplaceholder: 0.000001` passes (`ledger.py:88–90`). Neither establishes coverage or a credible projection.
- **Enlarged-panel lifecycle still stalls:** a valid enlarged reservation passes, but once submitted/completed, subsequent validation demands a repeat unless the caller adds the unchecked coverage marker (`ledger.py:79–92,123–125`).
- **Subject counts charged as GPU-hours:** `fbsubjects256` is summed into pending cost. A valid 36-subject fixture produces **58.001 pending GPU-h**, including **36 fictitious hours**, blocking submissions under A3.1 (`ledger.py:80,99–101,119`).
- **Failed fallback can disappear:** `nofb256: 0` bypasses fallback/timing checks even alongside a pending fallback or after a failed fallback is removed. Failed repeats correctly cease counting, but this marker permits dropping the obligation (`ledger.py:72–78`). Boolean subject counts also pass (`ledger.py:65,81`).

No files changed; no ledger commands, cluster calls, or GPU execution.