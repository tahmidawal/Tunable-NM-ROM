# Frozen heat runtime pilot

This predeclared follow-up measures implementation and stopping costs on the exact
saved heat pilot checkpoint. It makes no new representation, broad-family or
paper-level accuracy claim. `config-runtime.json` pins every setting and the
checkpoint content hash; no neural training occurs.

Both original meshes and all development cases are reused. The timestep is fixed
at the primary pilot setting. Initialization and each weak Crank–Nicolson solve
use the same predeclared stationarity tolerance within an arm. A small tolerance
ladder includes the original strict control in the same GPU job.

The modular path is the original initialize, rollout and readout sequence, including
its timed CN-factor construction. The compiled path wraps those exact functions
and factor construction in one `jax.jit`; all model weights, banks, operators,
training-code libraries and full input fields remain runtime arguments. The two
paths use identical multistart selection, LM budgets, residuals and termination
rules at each tolerance. Every warm and timed output must pass relative field and
latent parity and exact iteration/acceptance/reason-counter parity before fusion
is described as an equivalent acceleration. Failed parity remains in the results.

Each complete query begins with a host full interior initial field and ends with
all requested host full interior output fields. Input/output transfer and all
initial preparation, fitting, evolution and field reconstruction are charged.
Device work is synchronized; counter transfer occurs after the physical-output
timer for both paths. Compilation, mesh setup and reference regeneration remain
separate. The efficient direct sine-transform FOM is interleaved in the same GPU
job. Forward/reversed order alternates after clock burn-in. Compression and
error calculation occur after a complete paired timing block, never between its
subjects. Every output is content-addressed and every invocation records the
corresponding array hash, errors, phases, latents and solver counters.

The fresh analytic/refinement checks run again. Every method output and physical
reference is restricted to one shared observation grid. Current-reference,
initial-reference and absolute spatial errors remain distinct. With measured
reference discrepancy $e$ and empirical relative reference refinement estimate
$\delta$, the explicitly empirical adjusted ratio is

$$e_{\mathrm{adjusted}}=\frac{e+\delta}{1-\delta}.$$

No rigorous continuum error bound has been established, so that result field is
null. Empirical target eligibility additionally requires the reference estimate
to be comfortably below target, the configured solver criterion to pass, and
fusion parity where applicable. Fields and counters from failed or budget-limited
attempts remain visible. A separate predeclared error-drift ceiling compares each
relaxed arm with the same-job strict control; loosening termination cannot be
presented as accuracy preserving solely because it runs faster.

The nonlinear-head initial-state approximation gap is unchanged by this study.
After the measured runtime result, the next accuracy experiment will be proposed
separately; no new training or mesh growth is authorized within this pilot.

## Plain-language glossary

- **Checkpoint / frozen:** saved model weights / weights unchanged in this study.
- **Modular / compiled query:** individually called initialization/evolution/output
  functions / the same functions composed into one compiled device program.
- **LM / CN / FOM:** damped least-squares solve / Crank–Nicolson timestep /
  full-grid sine-transform solver.
- **Stationarity tolerance / parity:** required smallness of the normalized
  objective gradient / agreement of fields, latent states and solve counters.
- **Current / initial / absolute error:** discrepancy divided by current reference
  norm / initial norm / no normalization beyond spatial integration weights.
- **Empirical reference estimate / rigorous bound:** observed change under
  refinement / mathematically established upper error limit, unavailable here.
- **Paired repetition / content hash:** one timed run of every method on the same
  case and GPU / identifier of the exact returned field values.
- **Error drift / eligible target:** change from the strict same-job control /
  target attained under the stated empirical accuracy and solver checks.
