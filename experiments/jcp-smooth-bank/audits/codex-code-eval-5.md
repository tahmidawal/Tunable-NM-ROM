**Item 9: CORRECT. GO for staging ev1 after training**, for this coverage gate.

In-memory fixtures executing the actual validator confirmed:

- Correct full layout: **PASS** (648 invocations).
- Duplicates replacing missing coverage: **REJECT**.
- Missing cases, repetitions, or reverse directions: **REJECT**, including with doctored `expected_invocations`.

Caveat: `expected_invocations` is ignored. Doctoring it alone on otherwise complete coverage **passes**; it cannot conceal missing coverage.

No repository files modified.