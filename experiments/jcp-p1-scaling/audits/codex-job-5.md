**FIX-FIRST** — HEAD `8ea870b42597`.

- **Remaining blocker:** `analyze.p1e_judge` crashes on serialized nonfinite ρ. `QS.clean` converts NaN/∞ to `null`; `np.isnan(None)` raises `TypeError` (reproduced read-only). Preserve the distinction between undefined and failed ρ, count exclusions, and handle these cases without crashing.
- **Saturation fix passes:** fs3 BE, K=1, a=6 checks saturation across steps and incorporates it into validity; vendor bridge is recorded. The upstream diff contains only the import redirection and provenance docstring.
- **Staging passes:** manifest verifies; all **126 provenance files** match HEAD, including fs3 and dependencies. Sbatch uses H200, isolated paralab paths, f64/highest precision, and GPU preflight.
- **Certificate matches** config and invocation; 1 hour exceeds 1.5 × 0.515 hours. Its projection remains an estimate based on the older smoke, without explicit costing of the added bridge.

No files changed or jobs submitted.