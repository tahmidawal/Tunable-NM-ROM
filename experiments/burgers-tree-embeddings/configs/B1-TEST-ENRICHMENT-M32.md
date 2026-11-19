# Finite review of the remaining weak-test sensitivity

The completed 11008-to-22016 test check keeps all three direct24/global768 pilot trajectories physically accurate and stationary. Its largest returned-field change is 0.106837831% on align002, above the unchanged 0.1% admission limit. The other pulse changes by 0.095934393%; global changes are below 0.001%. Thus 11008 tests are not admitted. This is evidence of converging but still marginal tree test truncation, not a representation failure.

Reserve at most 45 GPU-minutes, subject to the current finite batch accounting guard, for one further independent test-factor32 check: 44032 starting tests, L2048, CN dt0.0025, dense integration, identical direct24/global768 checkpoints, initial allocation, visibility safeguard and solver tolerances. Compare every returned state against b1cnM16. No simultaneous model, integration or timestep change.

Require stationary finite trajectories, physical error plus reference allowance at most1%, and all returned-field changes at most0.1%. If this passes, 22016 tests can enter the next independent integration/time/precision checks. If it fails, write a scoped review before any additional test ladder; do not silently keep increasing the count. No nonlinear candidate inherits this direct-arm admission. Retain all runs and their costs.
