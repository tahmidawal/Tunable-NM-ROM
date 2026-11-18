# R1 — refit latent codes inside assembled training

Registered after diag01 and before R1 submission. No threshold or data change.

The initial direct-local training recipe used Adam to jointly update weights and snapshot codes. Diagnostics show a large training-code optimization gap: re-solving those same training fields on the frozen checkpoint substantially improves the fit. The initial recipe's early stop is therefore not evidence of a feature-learning plateau. Its development finite-start fits and frozen-span failures remain valid evidence for that checkpoint, not for the architecture.

Repair R1 changes one factor: before each assembled weight update, perform up to25 damped GN/LM iterations on that minibatch's parent and child variables. Stop gradients through those fitted variables for the weight update, then retain them as the next warm start. Use the same reference trajectories, seed, global parent, local width64/features16, prefix/configuration mixture, physical weights, update ceiling and development checkpoint rule. The stochastic fitting integration is the registered sampled training quadrature; all reported projections still use full independent integration and multiple starts. Log stationarity and fit quality again before declaring a plateau.

Reserve at most120 allocated GPU-minutes for localR1 and at most120 for its frozen-feature nonlinear head, inside B0's existing eight-hour ceiling. Stop this recipe if the common development projection gate fails after its bounded schedule. A second distinct repair will be chosen only from the resulting span/code diagnostics. No PDE advancement, final-test opening, criterion change, or automatic merge.
