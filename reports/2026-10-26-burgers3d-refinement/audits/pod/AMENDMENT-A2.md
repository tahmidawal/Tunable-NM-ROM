# A2: separate common consistency from extra POD-only gradient checks

Status: independently approved before any k32 validation. This amendment
is made after the frozen k8/k16 pilot and before the fitted k32 support is available.

All original thresholds and the original joint admission verdict remain frozen.
Report three independent layers, with explicit masks rather than one opaque label:

1. Complete case/run validity and own-objective Q1, including its pooled stationarity,
   native anchors and field distances.
2. Shared field-refinement consistency against the full-residual no-HR trajectory,
   with the frozen field-distance bar on every case.
3. Additional POD-only full-gradient discrepancy and pooled full-stationarity checks.

The bank's tested-advection rho and the POD full-gradient discrepancy have different
definitions and thresholds. They cannot be treated as symmetric admission criteria.
The original strict POD admission passes only when every original predicate passes;
an extra-gradient failure is still a failure of that preregistered contract.

Approved validation-release clarification: if the fixed k32 pilot passes layers
one and two while failing only layer three, the same frozen support may receive the
already-reserved complete validation as a descriptive common-profile comparator.
It must carry the original extra-gate failure and cannot be called preregistered
HR-qualified. Its accuracy/cost evidence must remain visible in a common-check
comparison so an asymmetric extra requirement cannot manufacture a bank advantage.
If either common layer fails, stop k32 validation and retain its failure. If all
layers pass, complete validation follows the original strict profile.

No new fitting, alternate support, threshold relaxation, case selection or resource
extension is authorized. Every full validation/Stage F inclusion still requires its
exact frozen artifact/configuration, independent job gate and results review. The
existing reservation and overall four-GPU-hour cap remain unchanged.
