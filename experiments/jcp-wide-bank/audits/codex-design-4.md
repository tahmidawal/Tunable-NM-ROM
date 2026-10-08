| Item | Verdict | Reason / blocking fix |
|---|---|---|
| Timing cohorts | CLOSED | A3 explicitly restores 2D dev6 and the first 16 3D validation cases. |
| 3D memory policy/accounting | NOT-CLOSED | Chunking and cache release are specified, but lifetime-peak increments cannot measure individual setting peaks; require isolated processes or interval peak monitoring. |
| J3 host-memory feasibility | NOT-CLOSED | The incomplete extra-array estimate omits wider-network/POD workspaces, establishes no headroom within 320 GB, and explicitly abandons phase-resolved peaks; supply a complete bound and phase measurements. |
| Whole-job budgets | NOT-CLOSED | Several components remain unmeasured lump sums, J4 lacks an explicit whole-job recalibration requirement, and the table’s arithmetic needs correction. |
| `lstsq` pin | CLOSED | A3 fixes `rcond=1e-12` and specifies an absolute-residual fallback near zero. |
| Gates above R′=512 | CLOSED | A3 explicitly limits what the 32-column gates certify and requires disclosure. |
| Smoke warmup | CLOSED | `warmup=5` resolves the 20-step schedule conflict. |

New contradictions: J1’s ladder count and stated per-query cost imply **5.4×** its converged/check subtotal, not **0.4/0.9×**; each optional J4 trim adds four settings to six (**67% more settings**, not a justified 50% workload allowance).

**Fit to proceed to code: NO — fix per-setting memory measurement, demonstrate J3 memory headroom with phase accounting, and provide consistent, fully enumerated, smoke-calibrated whole-job budgets including J4 and trims.**