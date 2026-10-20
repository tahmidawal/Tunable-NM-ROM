SUBMIT

1. **Budget passes:** ledger t2 = **5.0489 GPU-h**, consistent with DESIGN A13.2. Recomputed spent + t3s + other pending = **28.6212 + 12 + 2 + 5.0489 = 47.6701 ≤ 55**, leaving **7.3299 GPU-h**.
2. **Version binding passes:** current raw binding `bca3bc9e…` aliases to `c6e315cf…`. Independent in-memory mutations of `DS.BAR`, `DS.RHO_BAR`, and `DS.q2_label` each changed the returned version; restoration recovered the original.
3. **Staged payload unaffected:** `f7e1977a1 → HEAD` changes no d3run runtime dependency. The staged commit remains `f7e1977a1`; all **179 manifest hashes pass**.

No files changed, submissions made, or scratch arrays copied.