**fom1k: CLEAN. b3d65: CLEAN** for this narrow pre-submission re-check at HEAD `1fb83beae`.

| Round-5 item | Disposition | Re-check |
|---|---|---|
| **6** | **RESOLVED** | All previously demonstrated FOM/ROM diagnostic bypasses are rejected. Valid controls pass. |
| **9** | **RESOLVED** | A11.1 now explicitly says CN ringing is not established. |
| **13** | **RESOLVED** | Complete 3D validator rejects corrupted vendor reasons, histogram, iteration arrays and iteration sum. |
| **15** | **RESOLVED** | Cached projection fix remains; the cited solver-evidence counterexamples now fail validation. |
| **FOM diagnostics** | **RESOLVED** | Contradictory `worst_lres` fails consistency; fractional Newton counts and zero-Newton NaN/∞ residuals are rejected. Producer-correct zero residuals pass. |
| **ROM verification** | **RESOLVED** | Impossible reason-4 evidence, fractional iterations and corrupted tolerance-ratio padding all return `(False, False)`. |
| **Vendor evidence** | **RESOLVED** | Combined and individual round-5 mutations fail `vendor_rescored`. Adaptive over-budget and extra fixed-sweep iterations also fail; the complete valid fixture passes. |
| **A13 wording** | **RESOLVED** | Cancellation is qualified as consistent with the observation, not isolated; anchor-relative uncertainty is explicit. |

Cross-job digest authentication still passes its unchanged control. **No new blocking defect established.**

Checks used in-memory execution of current audit functions and the complete 3D validator with the configured inventory and synthetic fields. No files modified, lab-log entry appended, or jobs submitted.