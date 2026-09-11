# One bounded Burgers initial-family coverage test

This protocol follows the audited `accuracy08` comparison, where the initial-only refinement improved training reconstruction but worsened development accuracy. It tests one additional training-coverage arm; no further search is authorized in this round.

Continue the existing approved Burgers tree and namespace in a private `accuracy09` allocation. Compare the original head, the frozen audited 576-field head, and one newly trained 4608-field head. Every head uses the same spatial bank, $k=16$, $r=512$, hidden widths, strict solver budgets and unchanged normalized stationarity threshold. All six opened development cases, all three meshes and the reused independently refined reference fields are identical to `accuracy08`. The final paper cohort stays sealed. Freeze the sole terminal new checkpoint before evaluating any query; there is no development-driven endpoint selection.

The new initial fields reproduce the checkpoint's original expanded training-family draw: 576 fields from seed 0 and 4032 from seed 1000. Only initial fields are regenerated; no large set of PDE trajectories is generated. QR compression uses the exact N1024 interior-field metric and retains the bank-perpendicular error constant. Every new initial code receives the same nearest-old-code initialization and 200-iteration fitting budget. Keep 30000 joint Adam updates, learning-rate schedule, clipping, 256 initial examples per update, 256 replay examples per update, the identical 4096 replay code IDs, and replay weight one. Replay targets remain original decoded states, not new trajectory truth.

Coverage increases the number of optimized initial codes from 576 to 4608. The head forward/backward batch sizes are unchanged, but the full code array and its Adam state grow eightfold. Each initial example is sampled less often at the same total update count. The driver records QR, field projection, initial-code fitting and optimizer-loop seconds separately, including compilation and periodic diagnostics in the optimizer-loop interval. The archived 576-field training time comes from another allocation and is not a matched training-speed comparison. No claim of identical offline work follows from identical update counts.

The cold-start library changes with the training procedure: the original model uses 8192 sampled old codes; the 576-field model adds all 576 optimized initial codes; the 4608-field model adds all 4608. Any observed query gain therefore belongs to the complete trained procedure, including its larger starting-code library. Its actual lookup cost is included in the complete-query timer. Trust radii remain based on the original full code pool for every model. EQ still uses the same 64 original code IDs for each head, with 64 weak modes and 256 samples; only decoder outputs and refitted weights differ. The added initial codes are not substituted into EQ training selection. The FOM-exact sign-upwind operator remains unchanged.

All three strict ROM controls plus the loose and tight iterative FFT-preconditioned FOMs are retimed in one GPU allocation, with three shuffled repetitions for six cases at 64, 256 and 1024 intervals. These are 270 complete invocations. Burn-in, f64/highest, mandatory GPU preflight, full-field retention and conservative charged/posthoc/NumPy stationarity classification remain required. Reused checkpoint/reference artifacts are copied directly between private paralab directories and verified against the closed archived parent; actual supplied inputs are regenerated from seed. Both inherited control fields/counters must reproduce the accepted parent. The new fine-grid training metric receives the same direct streamed NumPy residual/Gram/projection and rank/conditioning audit as the 576-field model.

The one-hour job limit is a ceiling, not a promise of matched runtime. Expected training and benchmark work is bounded by the explicit configuration. Once collected, retain restorable checkpoints, timing arrays and source/hash provenance, remove the exact remote directory, and stop this round after reporting whether broader initial-family coverage helped.

## Glossary

- **Coverage:** the number of different training initial fields represented by optimized latent codes.
- **Head / bank / latent:** coefficient network / fixed spatial feature functions / small reduced coordinate vector.
- **QR metric / perpendicular error:** coefficient coordinates preserving the full-grid least-squares norm / representation error outside the fixed bank.
- **Replay:** a regularizer retaining original decoder outputs at recorded training codes.
- **Cold-start library:** cached training coordinates used to select the starting guess from the supplied field.
- **EQ:** empirical quadrature approximating nonlinear weak sums with fitted nonnegative sample weights.
- **Strict stationarity:** the unchanged normalized objective-gradient criterion, separately checked by three implementations.
- **FOM / ROM:** full-grid PDE solver / nonlinear reduced model.
- **Complete query:** supplied initial field through initialization, all time steps and all requested full output fields.
- **Development cohort:** opened cases available for method development; the final cohort remains separate and sealed.
