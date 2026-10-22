**DO NOT SUBMIT.**

- One evidence gap remains: staged `fsrun.py` saves G-CN refinement states, outputs and certificates, but discards per-step `reason`.
- `g_valid` uses those reasons to reject reason 3 and detect unconverged budget exits. Saved residuals cannot establish the recorded exit reasons.
- Persist `v_['reason']` for both G-CN classification refinements so the results audit can reconstruct their validity.
- The LSPG diagnostic persistence and finiteness repairs pass; POD refinement evidence remains present.
- Staged manifest checksums pass. Projected 7,108.77 seconds fits the 7,860-second driver timeout.
- Budget admission passes: 0.63 + 2.25 = 2.88 GPU-hours < 4.
- Deferred offline-checker repairs and pricing conservatism are excluded from this verdict.
- No files changed, SSH used, or jobs submitted.