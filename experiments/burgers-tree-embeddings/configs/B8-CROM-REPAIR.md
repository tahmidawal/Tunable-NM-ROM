# B8 CROM training collapse diagnosis and prospective repair

November 19, 2026. This records a failed training recipe and its planned repair; it is not evidence that CROM is an inferior method. Keep the pinned author networks, native objective and GN solver, matched expanded trajectories, training-only normalization, float64 port and all prior limitations.

## Evidence before changing or cancelling the running attempt

Owned job b8cromexpanded, 7386569, uses base learning rate 0.0001 with author multipliers 10 and 5, hence an effective first-stage rate 0.001. Its requested schedule is 200 epochs. The root reports native losses about 15.248 at epochs 20, 40 and 60, with projected worst relative errors remaining about 0.93. The epoch-40 checkpoint was streamed read-only and verified against the running result's recorded SHA256 for independent diagnosis; no weights were changed.

That checkpoint stores training output standard deviation 0.12346837. The native loss of the constant training-mean predictor, calculated directly as 1000 times the squared stored standard deviation, is 15.244438087953238. Observed epoch-40 native loss is 15.24784153682838. This is strong evidence of collapse toward a constant predictor. Decoder hidden SIREN weights have grown far beyond their authored initialization scale: one hidden layer has maximum absolute weight 0.62614969230062, versus initialization bound approximately 0.0072169. Excessive effective learning rate is a plausible explanation, not a proven unique cause.

Source inspection found no evident normalization or manual-objective defect: the native training-step gradient parity check passes, encoder standardization and decoder inverse standardization use the same training-only statistics, coordinates use the author's SIREN normalization, and recorded optimizer learning rate matches its declared scheduler. The native architecture supports choices; author command-line defaults are not a guarantee that this learning rate trains this dataset successfully. The earlier lower-rate B6 recipe did learn substantially below the constant baseline, although it remained an unqualified weak reproduction and used a different data cohort.

Finite native projection execution must not be called convergence. The old driver records `passed_execution` after checking finite values and absence of exceptions; the author's routine does not expose its termination reason. Epoch-40 scaled stationarity examples are approximately 0.000187 and 0.000446, not small enough to pass a 1e-8 physical stationarity diagnostic. The repaired driver must report execution, observed native iteration usage, unexposed stop reason and physical stationarity separately.

## Prospective disposition of the current job

The sustained epochs 20/40/60 constant-baseline plateau justifies stopping the current high-rate attempt now rather than spending the rest of its allocation on an unchanging failed recipe. The root owner may cancel exactly job 7386569 through the owned-job cancellation safeguards after recording this decision. Retain all available checkpoints, projections, logs and source/data hashes, and charge its full actual allocation. Label it an interrupted high-rate recipe at its last retained epoch; do not claim the planned 200 epochs completed. This subagent neither cancels nor submits jobs.

## Prospective fresh repair

One fresh 200-epoch attempt, initially with a 60-minute allocation, uses the identical native architecture, objective, batch size, dataset, train/development split and 100/100 epoch schedule, with learning-rate multipliers 1 and 0.5 instead of 10 and 5. Effective rates are therefore 0.0001 and 0.00005. Start fresh under the authored initialization procedure, not from the collapsed high-phase weights. Initialization is retained explicitly; this does not claim identical initialization to the earlier run, whose constructor biases were not independently seeded before the authored internal seed resets.

Log a fixed representative training batch at initialization and epochs 1, 2, 5, 10 and subsequent 20-epoch checkpoints: native loss, constant-mean baseline, output and latent variation, weight norms and gradient norms. These diagnostics do not update weights or change selection. Preserve full encoded-development and native projected-subset evaluation every 20 epochs. Selection remains the recorded projected-error criterion; no diagnostic or unfinished fit becomes method admission. The default CLI multipliers remain 10 and 5 so an existing command does not silently change recipe; the repair requires explicit `--lr-multipliers 1 .5`.

No new early-stop rule is silently introduced in code. If the repair still fails, diagnose retained evidence and register the next change before execution. Native full-case projection, actual4096 output, time/integration checks and the 2D viscous online adapter remain necessary before a fair PDE/runtime comparison.
