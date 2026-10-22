**NEEDS-FIX.** Several defects can change the selected competitors, qualification, or reported speed ratio. Read-only audit completed: no files changed, no jobs submitted, no scratch files created. Checks used CPU-only f64.

References below are relative to `experiments/imp-geometry/`. **CORRECT** denotes the inspected component, not production certification.

- **run_cell.py — WRONG, search coverage:** `248–263` runs SEM step ladders only on a spatial Pareto subset; over-integration and eig solves are further restricted. A3.7 requires every registered size’s step ladder. `286–296` omits ECSW’s Galerkin CN and linearised CN steppers. These omissions can weaken the best competitor and inflate S.
- **run_cell.py — WRONG, selection:** `429–438` screens only six candidates chosen by **compile-inclusive, batched cohort time**. A3.4 requires warm single-query screening of the eligible candidates. A fast competitor with expensive compilation can disappear before official timing.
- **run_cell.py — WRONG, bank search:** `302–311` searches all steps only at q=8, searches other rules only at the last two valid steps, and never tests the q=8 tensor implementation. This can miss the fastest qualified bank configuration.
- **run_cell.py — WRONG, validity:** `459` checks failed solves only for FD/SEM. POD/ECSW can enter the benchmark despite nonlinear solver caps, damping saturation, or ETDRK4 validity failures. Competitor qualification is report-only; benchmark **validity** is still mandatory.
- **run_cell.py — NEEDS-FIX, decision implementation:** `469–562,591–628` produces ratios and individual test results but does not implement A4.2’s complete decision table, reference-eligibility gate, zero-field target exclusion, cell/geometry outcomes, or attribution. Missing finalists are reduced to booleans, losing censored/unresolved-match/invalid distinctions.
- **run_cell.py — WRONG, leakage guard:** `593–595` accepts any manifest matching a caller-supplied hash. It never verifies commitment or binds the manifest to frozen code, bank, POD/ECSW artefacts and descriptor hashes. `84–100` also computes and exposes test reference eligibility before finalists freeze.
- **run_cell.py — NEEDS-FIX, confirmation:** `604–625` reruns identical configurations and re-times identical pairs across targets/statistics, contrary to A4.2’s reuse rule. It does not compute the test max–min statistic or enforce test timing controls.
- **run_cell.py — NEEDS-FIX, required diagnostics:** `173–174` hard-codes quadrature subdivision without the registered snapshot-norm convergence selection. Training-data accuracy, SEM reference cross-check, SEM IC-interpolation error, cross-span residual, G-FLOOR enforcement and matched feature-window errors are absent. `44` also doubles the registered training step counts without a corresponding amendment.

- **finalize.py — WRONG, Q1:** `73–109` ignores reduced solver `n_unconverged` at stats index 6. An in-memory fixture with **100% unconverged steps passed Q1**. The shared ≤1% nonstationarity allowance also permits failed linear-solve certificates although A3.5 requires those every step.
- **finalize.py — WRONG, Q2:** `130–159` omits the IC-projection-difference bar. M/K checks cover only r×r blocks, although LSPG uses Mt×r operators. Incorrect extra test rows can escape these checks. **CORRECT:** rho’s population-wide denominator, skip fraction and dev+val population follow A4.3.
- **finalize.py / run_cell.py — WRONG, timing certification:** `finalize.py:164–179` discards timed outputs; `run_cell.py:512–515` compares a fresh replay of only the first case with the official rerun, not stored timed outputs against ladder outputs. T-id, compilation counts and T-dbl failures do not invalidate S. The registered drift sentinel/retry is absent.
- **finalize.py — NEEDS-FIX, T-dbl:** `182–191` places two independent calls in one JIT graph; it does not enforce back-to-back execution. XLA may overlap them, so the ratio is not a reliable doubled-serial-work control.
- **run_cell.py — CORRECT, validation statistic:** `520–533` implements max-bank/min-competitor selection and pair reuse. Exact competitor ties sort by family before configuration key (`525`), rather than the registered configuration-key rule.

- **ladder.py — WRONG:** `81–86` declares validity solely from finite field errors, ignoring solver statistics. A failed-solve fixture was marked valid. This also lets unconverged rungs trigger step-distance stopping. `101–124` otherwise implements the consecutive-rung/double-invalid stopping structure correctly.

- **rom.py — CORRECT, core algebra:** `50–72,161–175,269–274` has consistent tensor contraction, nonlinear Jacobian and CN/BE residuals. Tiny tensor-versus-point and Jacobian discrepancies were below \(9\times10^{-16}\).
- **rom.py — WRONG, ETDRK4 admissibility:** `186–196,263` records but never enforces cond(V)≤10⁸. A synthetic matrix with **cond(V)=1.50×10⁸ was accepted**.
- **rom.py — WRONG, charging certificates:** `201–203,225–260` computes independent coefficients and a second ETDRK4 update inside every timed query, regardless of `cert`. This violates the untimed-certificate contract and distorts ETDRK4 comparisons.
- **rom.py — NEEDS-FIX, certificates:** `245` checks the reconstructed updated state rather than explicitly checking every stage. `99–118` uses exp−1, threshold 1 and 24 series terms rather than A3.5’s specified expm1/0.2/eight-term implementation; reconcile this explicitly. **CORRECT:** full-circle complex coefficients agreed with the independent formula to \(1.7\times10^{-15}\) in the sampled check.

- **pod.py — WRONG, ECSW candidates:** `120–124` draws 16,384 uniform plus **32,768** importance samples, sorts/deduplicates with `np.unique`, then keeps the lowest 32,768 indices. This introduces spatial index bias instead of the registered mixed sample. A uniform-importance fixture retained no index above 74,869 of 99,999.
- **pod.py — CORRECT, inspected mathematics:** `32–60,80–83,127–145` correctly applies h² POD weighting, projected FD operators and per-state ECSW normalization; fitted weights absorb quadrature weights.
- **pod.py / run_cell.py — NEEDS-FIX, IC contract:** `pod.py:84–99`, `run_cell.py:206` substitute a fixed 128-grid IC quadrature for the registered data-grid projection, without measuring its error. This can handicap POD/ECSW. `run_cell.py:414–422` reports ECSW successor rollout distance only, omitting the required rho diagnostic.

- **fd.py — CORRECT, inspected kernels:** `63–72` has the Shortley–Weller second/first-derivative coefficients with correct signs; `112–119,152–160` has full weighting, bilinear prolongation and transfer masks.
- **fd.py / run_cell.py — WRONG, RK4 admission:** `run_cell.py:228` passes **theta_min²** into a bound already dividing by theta_min (`fd.py:254–255`). For H,n=64,S1 this starts at **81,920 steps**, beyond the 20,480 cap, suppressing the entire ladder; a conservative bound computed from the actual stencil gives 5,120. Use the actual stencil bound, then test stability.

- **sem.py — CORRECT, inspected operators:** `87–115,144–195,172–180,301–325` consistently implements map metrics, stiffness, diagonal, over-integration, generalized eig solves and SBDF startup. Tiny L/H/B checks gave assembly/diagonal errors ≤1.5×10⁻¹⁴ and eig residuals ≤3.3×10⁻¹⁵. Lattice partition-of-unity checks passed. Production-size conditioning/convergence remains unverified.

- **spec.py — CORRECT:** `24–57` consistently scales viscosity/time and physical IC coordinates; with frozen SIDE=1 it reduces to the intended unit-box formulation. Sine normalization and synthesis are consistent.

- **geom.py — CORRECT:** `26–47,72–91,136–160,204–219` implements v4a, shared rejection, envelopes and zero-extended ICs. Sampled composite areas matched analytic areas to roundoff; inspected graded interfaces and curved maps are consistent.
- **bank.py — CORRECT, envelope/ordering:** `38–51,144–180` applies physical-coordinate derivatives, weighted QR, normalized-code SVD, frozen rotation and rule-specific mass/stiffness consistently.
- **bank.py — NEEDS-FIX, diagnostics/smoke:** `151` calls a QR-diagonal ratio `condR`; it is not a matrix condition number. `112–116` makes every 2,000-step smoke iteration full-batch because `full_last=10000`, so that smoke does not exercise production subsampling.

- **score.py — CORRECT:** `14–63` implements positive-time normalized error, Richardson bands/allowance and three-way matching. All supplied fixtures passed. Reference convergence/eligibility enforcement must occur in the driver.
- **descriptors.py — CORRECT:** `12–28` matches A4.1’s deterministic hashes and separately estimated acceptance rates. All **16 committed descriptor hashes** matched regeneration.

- **interaction.py — NEEDS-FIX:** `36–38,50–52` ignores solver failures and retains `passes` based on superseded D_geo. Keep it explicitly historical/diagnostic. The primary D_bnd implementation in `run_cell.py:565–583` uses the correct shared IC, but also discards convergence status.

- **controls.py — WRONG, weakened controls:** `58–80` accepts MG contraction ≤0.35 instead of ≤0.2, tests n=256 rather than the finest reference mesh, and checks quadrature area without the required envelope-squared integral. At its actual L rule, k=4,q=8, the latter’s relative discrepancy was **3.60×10⁻⁹**, above 10⁻¹².
- **controls.py — NEEDS-FIX, fault independence:** `168–179` compares only aggregate MF-Q1 violation counts, not exact step identities. `217` replaces the FD field with reference+noise rather than perturbing the FD field by 2τ.
- **controls.py — CORRECT:** `139–146,197–202,219–228` independently implements the tensor point-form fault and establishes the A4.4 Q2/matcher violations before declaring detection.

- **probe.py — WRONG, immediate failure:** `149–150` invokes `main()` before `bankstep_sec` and `ecsw_sec` are defined (`153,177`). Both proposed smoke6 sections raise NameError.
- **probe.py — WRONG, cost measurement:** `74–79` overwrites input data-grid n/nt with the finest reference n/nt, then uses the original data mask. S2 therefore probes a 1024-grid solve with 768-grid indices instead of the production data solve.
- **resources.py — NEEDS-FIX:** `49–55` uses hand-coded assumed row counts and Pareto pruning, not enumeration of the registered ladders; `20–33` consumes old probes and has no smoke6 recertification path. `85` excludes conditional box banks from “total_registered”. Its provisional label is appropriate; it cannot certify release.

- **cluster/submit.sh — WRONG:** `11–20` checks directory reuse outside the lock; concurrent identical submissions can overwrite code and submit twice. It enforces campaign eight but not lane three, stages dirty working-tree bytes while recording HEAD, and lacks `pipefail`. `14` also violates the absolute-Python rule.
- **cluster/job.sbatch.tmpl — CORRECT, preflight:** `3,12–17` selects gpu, sets highest precision and exits on CPU fallback.
- **cluster/job.sbatch.tmpl — NEEDS-FIX:** `9,19–20` lacks fail-fast error handling; a failed body can finish with successful “JOB DONE”. No execution-directory lock protects outputs.

- **vendor/ — CORRECT:** SHA256 matches all three specified sources: sep_common `1f1c2796…`, nnls `c0c5792e…`, pscore `83d8a004…`.

**OVERALL: NEEDS-FIX**

1. Restore registered search coverage and screen candidates using warm single-query timings.
2. Enforce solver validity throughout ladders, selection, Q1 and confirmation; enforce ETDRK4 conditioning.
3. Complete Q2’s IC and tested-operator checks; move ETDRK4 certificates outside timed calls.
4. Implement A4.2 decisions, reference/zero-field gates, attribution and deterministic test max–min aggregation.
5. Bind test access to a committed, immutable manifest and artefact provenance.
6. Store/check actual timed outputs; gate timing failures; implement sentinel, serial T-dbl and test pair reuse.
7. Fix RK4 starting bounds and ECSW sampling; validate or restore POD IC projection.
8. Restore control thresholds, exact stepwise fault checks and missing diagnostics.
9. Fix probe execution/grid bugs and regenerate the cost certificate from actual registered coverage.
10. Make staging/submission atomic, enforce both job caps, preserve content provenance and propagate batch failures.