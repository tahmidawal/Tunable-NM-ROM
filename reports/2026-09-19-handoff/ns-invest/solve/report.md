# NS2D solve-loss investigation (Fable agent, 2026-09-18) — condensed by coordinator

Data/scripts beside this file: ns_solve_probe.py, probe_N256_run1.json, probe_N256_run2.json, probe_run*.log, cpu_spectra.py/.json, cpu_bank_visibility.py/.json. Lane ns2d @ 9830d202, job ns304 (3808502).

## Headline
1. Half of the reported "solve 2.9–10.8x above manifold" is a STATISTIC MISMATCH: decomposition.manifold_median is the median over all 48 states incl. t=0; aggregates.median_evolved is the median over cases of worst evolved time. Matched (median over 8 cases of worst evolved time): solved/manifold = 2.51 / 2.31 / 2.20 / 1.91 / 2.04 / 1.40 for q = 0/32/64/128/256/512. Linear controls with the same solver: POD-32 1.49x, POD-512 1.26x, bank-512 1.37x.
2. Burgers' "1.0–1.1x" is the same artefact reversed: b-panel's solved_over_best_found is worst over ALL times, pinned by the t=0 compression (2.56 %) which exceeds every evolved error; on evolved times Burgers solved is BELOW best-found worst-evolved (manifold error decays over 50 diffusive steps).
3. The real NS loss (~2–2.5x at q<=256) is slowly accumulating in-manifold drift: per-time ratio 1.28→1.64→2.18→2.53→3.36 at t=0.2…1.0 (q=0); excess grows ~linearly (~0.06 per 100 steps) while the manifold layer falls after t=0.4; error concentrates at |k|<=2; ROM under-dissipative (enstrophy ratio 1.04–1.16); Re-dependent (1.5x at Re~115, 2.3x at Re 500–900).
4. GPU probe (18 arms, harness reproduces archived trajectories to >=6 digits): all-Fourier-modes test space 0.687655 vs 0.687654; gtol 1e-10 0.687666; dt/2 0.659; tangent Galerkin 0.707; oracle restart every 100 steps 0.418 (t=1: 0.319) with per-window excess 0.023/0.100/0.150/0.128/0.094 (Re 890) and 0.050/0.030/0.026/0.011/0.006 (Re 118). One step from the oracle costs 0.03–0.55 % of ||w0||; ROM weak residual below the oracle's at all 20 sampled steps (no wrong basin). q=512 reproduces neural_q512 with the head removed.
5. The matched loss is a property of the manifold's off-training geometry: same solver gives POD-32 1.49x vs neural q=0 2.51x on a smaller manifold error; the neural trajectory leaves the region where the head was trained.

## Residual as implemented
Scalar vorticity on the periodic torus, vorticity–streamfunction, Arakawa Jacobian, psi = -Lap^-1 w (FFT). NO pressure, NO velocity unknowns, NO divergence constraint (incompressibility identically satisfied). Implicit midpoint, dt=2e-3, 500 steps to T=1. Tests: M=2176 lowest real Fourier modes (|k|<=26.2), row-scaled by the Helmholtz diagonal (factor >=0.79). Damped LM, rel. stationarity 1e-6, budget 200; warm start by linear extrapolation guard. ns304: every step exits on stationarity, median 2 LM it/step, zero budget exits.

## Differences from Burgers
Non-local bilinear Arakawa vs local upwind; Re 100–1000 (one eddy turnover) vs cell Re <~ 40 decaying; implicit midpoint vs backward Euler; 500 steps/T=1 vs 50/T=0.25; M=2176 fixed vs M=4(K+q); plain LM vs block-damped LM with z trust radius; statistic mismatch (see above). Nothing in the NS residual machinery is weaker than Burgers'.

## Dead hypotheses (measured)
Test starvation / spectral blindness (truth, bank, error all inside M=2176; all-modes arm identical); pressure/divergence (no such equation); row scaling; LM basin/stationarity; dt; budget exits.

## Ranked hypotheses
H1 (cause): closure drift of a manifold that misrepresents 10–30 % of the state; per-step bias 1–3 % of the manifold error, transported not damped, accumulates linearly over 500 advective steps. Test: k-step re-projection ladder (oracle-replace previous state every k in {1,10,100,500}); predicted excess ∝ k.
H2 (accounting): statistic mismatch; fix the generator (manifold_medcase_worstT), zero cost.
H3 (where to act): ratio set by the manifold's off-training geometry, not the solver; any head that passes H-ORACLE should land ~1.4–1.5x like the linear arms. Test: solved-z distance to nearest training code vs oracle-z per time.

## Recommendation
No change to residual/tests/scaling/projection/dt/LM moves q<=256. Fix the statistic first (both lanes). For q<=256 the lever is the head (a Phase-2 problem). One solve-side mitigation worth one job: a closure term (per-step enstrophy budget or eddy viscosity nu_eff = nu + nu_t(q)); predicted to buy 10–30 % of the excess, not 2x. Online re-projection is inadmissible (needs the truth).
