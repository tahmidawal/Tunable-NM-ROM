- **CORRECT — Batch:** H200, 200G, 1.25 h ≥ certificate ×1.5 = 1.13918 h. `JCPREF_STORE`, GPU preflight, f64/highest checks and manifest verification are present.
- **CORRECT — Config provenance:** `candidates.json` is byte-identical to lad2’s output. In-memory regeneration with `make_configs.py time` reproduces `time.json` byte-for-byte.
- **CORRECT — A/ROMs:** A matches frozen `rom-box.json`; both target denominators follow A5.12. No qualified 1% denominator.
- **CORRECT — Slots/groups:** C100 slots exactly match the frozen mask (`dev|0`–`dev|5`). All 14 family/union groups match the ladder; every included candidate is qualified.
- **CORRECT — Diagnostics:** 25 query subjects, 25 KERNEL, 25 SETUP and six paired PSI C1 subjects.
- **CORRECT — Bindings:** Frozen hashes, config bindings, staged manifest and provenance all match. No staged/current source discrepancies.
- **CORRECT — Controls/reference:** T-dbl, T-fault and isolated T-id are configured; `refdir` is the ref100 cluster output directory.
- **CORRECT — Gate evidence:** Outputs retain selection, fresh timing repetitions, scoring, identity vetoes and controls needed by `gate.py`, alongside ladder/eligibility/ROM-box inputs. T-gate fixtures pass.
- **NEEDS-RESTATEMENT — “Pre-registered gate”:** Registered eligibility already implies **UNRESOLVED**. Timing supports the **post-hoc A10 gate**; report both outcomes explicitly. `gate.py` does not automatically emit both.
- **NOTED — Forgery-only:** Hash consistency is not independent authentication against coordinated artifact replacement.

**Overall verdict: CLEAN.**