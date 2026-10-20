**CORRECT** — round-8’s blocker is closed at `970fa8c43`.

- `smoke_accept.py:121` requires `tighter_valid` on every applicable dev6 row.
- The positive fixture includes it; the deletion regression rejects missing evidence.
- CPU `test_accept.py` passed: 22 timing, seven rule, and five accuracy rejection cases, plus positives.
- Additional in-memory checks rejected each of 20 individual deletions across 84 rows. `False` remains accepted as a reportable outcome.
- `hacc.py:184` emits `tighter_valid=bool(R['valid_tighter'][k])` for dev6 rows, populated from `HA.run_valid(vtt)`.

No further blocking findings. Round-8’s fabricated doubled `attempt=200` remains **forgery-only**, noted.

**smk1: ACCEPTANCE-READY** — checker readiness, not acceptance of actual run results.  
**smk2 (H200): GO for submission.**

No files changed, scratch created, jobs submitted, or lab-log writes. Bytecode writing was disabled during tests.