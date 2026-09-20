# NS improvement plan — synthesis of four Fable investigations (2026-09-18)
Reports: head/report.md, solve/report.md, cost/report.md, lit/report.md.

## Diagnosis (all measured)
- Solve: no solver/test/dt/projection change moves the error (all arms identical); matched solve/manifold 2.5→1.4x, same order as linear controls. The 2.9–10.8x was a statistic mismatch (fixed, ns2d 2d70f36a, paper d094ad00).
- Head: the binding cause is coverage of a 13-dim family (1.6 samples/axis; nearest training state 60 % away vs 9–14 % on Burgers) combined with weak nonlinearity of the decaying family (POD-K already captures it). Every interpolant tried (parametric, quadratic manifold, free-code MLP, encoder predicted) lands at ~POD-K. Auto-decoder collapse excluded. Phase-aligned (shifted-manifold) decoder is the best single idea: predicted ratio 1.45–1.7 (pass at 1.5, FAIL at 2.0). A predicted PASS needs a transport-dominated NS family, pre-screened at 64² by: held-out POD-16 floor >= 0.35 and same-time coverage <= 0.25.
- Cost: 4.56 GB tensor streamed per residual; safe combo (certified EQ + M=4(K+q) + analytic Jacobian + 1–2 LM it, warm start) → 0.26–0.48 s at q=0 vs FOM 0.42 s (parity); ceiling ~2.4x (latency floor). Larger ROM dt is the literature's lever.
- Literature: no NM-ROM beats a tuned same-hardware spectral FOM; parity at 256² is the ceiling; few x plausible at 1024² with hyper-reduction + larger dt.

## Pre-registered experiments (ranked by cost / payoff)
E1 (no training, ~1 h, 1 job or local): 64² pre-screen of candidate transport-dominated NS families (advected vortex pair, drifting Kolmogorov frame, shear layer) by the two statistics; pick the family that passes both.
E2 (2 jobs, ~8 h): shifted-manifold decoder (spectral phase alignment, D4 augmentation, encoder/param-conditioned time-smooth latent) on the E1 family, K=16; PASS = H-ORACLE >= 2.0 and held-out/train <= 1.5. Negative control: free-code auto-decoder on the same family must fail.
E3 (1 job): if E2 passes, phase-3 ladder with certified EQ + M=4(K+q) + analytic Jacobian + 1–2 LM it + 5x ROM dt (arm) vs tuned FOM tolerance ladder and POD-LSPG; PASS = >= 1 neural rung non-dominated on (ms, worst evolved).
E4 (1 job, Burgers 256²): bank+head trained on fine-reference (4096-interval) solutions restricted to coarse nodes; dense_m4 ladder; PASS = a rung with vs-ref < 4.03 % AND ms < tuned Newton; predicted PASS on accuracy at q<=32, FAIL on cost.
E5 (Burgers, 1 job): cost levers on the head-only rung (analytic Jacobian, 1–2 LM it, warm start) to close 59 ms (EQ q=0) vs 32 ms tuned Newton.

Total: ~6 jobs, ~2 days wall-clock. Deadline 2026-09-25.
