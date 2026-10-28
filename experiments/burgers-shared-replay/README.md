# Shared-decoder standalone trajectory and fine replay

This is Test B of the approved shared-decoder protocol in `DESIGN.md`. It is an implementation with no numerical result yet. Root coordinates training, the combined budget and all submissions.

`replay.py` first computes a complete standalone K16 trajectory. It then runs the unchanged fine initial fit and every K64 step in order. A padded K16 state is only a guarded initial guess; accepted fine states supply every fine residual history. Both initial fits and both sets of six output fields are charged inside the fused query. There is no K32 online stage.

Coarse initial-fit or step invalidity is recorded. Coarse advancement stops at its first failure, and later states and output fields are explicitly unavailable. Failed states never enter an accepted history. Later coarse guesses are not offered to the fine solve. The candidate arm also requires a fully valid coarse trajectory for numerical admission; fallback does not conceal a failed standalone solve.

Only root may stage the final immutable model bundle, submit work or change the combined resource ledger. The stager writes a source/model package without contacting the cluster or altering that ledger; root supplies the reviewed allocation script. Every large operator/cold-fit table remains a dynamic query argument, with only the retained small neural-weight closures.

Glossary: K is the number of latent coordinates. Replay means solving the fine time steps again in order. A guard compares a proposed starting point with the original fine temporal predictor. A preview is one of the six coarse output fields. An invalid history contains an initial fit or timestep that failed the prescribed production criteria.
